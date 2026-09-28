"""Nano and TabPFN model adapters; imported only by model collection."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, is_dataclass, replace
import importlib.metadata
import math
from pathlib import Path

import torch

from .torch_adapter import TorchAdapter
from .gates import RowLayout


def load_adapter(kind, checkpoint, *, d, device, dtype, tabpfn_version="v2.5"):
    if kind == "nano":
        return NanoModelAdapter.load(checkpoint, d=d, device=device, dtype=dtype)
    if kind == "tabpfn":
        return TabPFNModelAdapter.load(checkpoint, d=d, device=device, dtype=dtype,
                                     version=tabpfn_version)
    raise ValueError(f"unsupported model backend: {kind}")


@contextmanager
def patch_method(instance, name, method):
    owned = name in instance.__dict__
    previous = instance.__dict__.get(name)
    setattr(instance, name, method)
    try:
        yield
    finally:
        if owned:
            setattr(instance, name, previous)
        else:
            delattr(instance, name)


class NanoModelAdapter(TorchAdapter):
    @property
    def supports_feature_branch(self):
        from models.model_2d import NanoTabPFNModel
        return type(self.model) is NanoTabPFNModel

    @property
    def supports_batch_prediction(self):
        # These frozen architectures are deterministic and independent across
        # batch entries. Do not assume this for arbitrary/stochastic backends.
        from models.model_1d import NanoTabDPTModel
        from models.model_2d import NanoTabPFNModel
        return type(self.model) in (NanoTabDPTModel, NanoTabPFNModel)

    def forward_batch(self, context, targets, queries):
        output = self.model((torch.cat((context, queries), dim=1), targets),
                            train_test_split_index=context.shape[1])
        return output[:, :, 0]

    @contextmanager
    def _source_gate(self, gates, *, layout: RowLayout):
        """Apply a separate intervention to each Nano 2D batch entry."""
        from models.attention_value import multihead_attention_value_paths

        layers = {}
        for slot, gate in enumerate(gates):
            if not gate.is_baseline:
                self._check_gate(gate, len(self.model.transformer_blocks), self.d)
                if not hasattr(self.model.transformer_blocks[gate.layer], "self_attention_between_features"):
                    raise ValueError("requested model/layer has no feature-message gate")
                layers.setdefault(gate.layer, {}).setdefault((gate.source_token, gate.row_scope), []).append(
                    (slot, gate.epsilon))

        def request_weights(module, args, kwargs):
            return args, dict(kwargs, need_weights=True, average_attn_weights=False)

        def subtract_messages(sources):
            def hook(module, args, kwargs, output):
                value = args[2] if len(args) > 2 else kwargs["value"]
                # Nano 2D folds (task/state, table row) into the attention batch.
                layout.check_batch(value.shape[0], len(gates))
                result = output[0].clone()
                for (source, scope), entries in sources.items():
                    slots, epsilons = zip(*entries)
                    ids = layout.indices(slots, scope, device=value.device)
                    selected_rows = len(ids) // len(slots)
                    message = multihead_attention_value_paths(
                        module, value.index_select(0, ids), output[1].index_select(0, ids),
                        key_slice=slice(source, source + 1)).squeeze(2)
                    epsilon = value.new_tensor(epsilons).repeat_interleave(selected_rows)[:, None, None]
                    result.index_add_(0, ids, -epsilon * message)
                return result, output[1]
            return hook

        with ExitStack() as stack:
            for layer, sources in layers.items():
                attention = self.model.transformer_blocks[layer].self_attention_between_features
                before = attention.register_forward_pre_hook(request_weights, with_kwargs=True)
                stack.callback(before.remove)
                after = attention.register_forward_hook(subtract_messages(sources), with_kwargs=True)
                stack.callback(after.remove)
            yield

    @contextmanager
    def _branch_gate(self, entries, *, layout, batch_count):
        scales = self._branch_scales(entries, n_layers=len(self.model.transformer_blocks),
                                     layout=layout, batch_count=batch_count)
        with ExitStack() as stack:
            for layer, scale in scales.items():
                attention = self.model.transformer_blocks[layer].self_attention_between_features
                handle = attention.register_forward_hook(
                    lambda module, args, output, scale=scale: (scale.apply(output[0]), output[1]))
                stack.callback(handle.remove)
            yield

    @classmethod
    def load(cls, checkpoint, *, d, device="cpu", dtype="float32"):
        from models.architecture_registry import get_architecture

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("task_type", "regression") != "regression":
            raise ValueError("a regression checkpoint is required")
        architecture = get_architecture(payload["arch"])
        built = architecture.build_from_checkpoint(payload, num_outputs=1)
        if architecture.input_policy == "fixed_width" and built.config["num_features"] != d:
            raise ValueError("checkpoint input width differs from d")
        built.model.load_state_dict(payload.get("model_state_dict", payload.get("state_dict")))
        scaling = payload.get("training_config", {}).get("target_scaling", "none")
        adapter = cls(built.model, checkpoint=checkpoint, d=d, target_scaling=scaling,
                      device=device, dtype=dtype)
        adapter._metadata.update(adapter="nano", architecture=payload["arch"],
                                 model_config=dict(built.config),
                                 training_seed=payload.get("seed"))
        return adapter

    def token_groups(self, layers):
        blocks = getattr(self.model, "transformer_blocks", [])
        result = {}
        for layer in layers:
            if (not 0 <= layer < len(blocks)
                    or not hasattr(blocks[layer], "self_attention_between_features")):
                raise ValueError("requested model/layer has no feature-message gate")
            result[str(layer)] = [[{"feature": j, "slot": 0, "column": j}]
                                  for j in range(self.d)]
        return result


class TabPFNModelAdapter(TorchAdapter):
    """Direct frozen-checkpoint inference with no sklearn ensemble preprocessing.

    Native column encoders/token grouping remain part of the model, including
    their configured seeds and batch sharing. External target scaling uses the
    current task's own context.
    """

    @property
    def supports_feature_branch(self):
        return not self.v3 and self.version in ("v2", "v2.5")

    @property
    def supports_batch_prediction(self):
        # A randomly shortened layer stack would need a separate execution
        # group per sampled depth; keep that nonstandard configuration serial.
        for module in self.model.modules():
            minimum = getattr(module, "min_num_layers_layer_dropout", None)
            if minimum is not None and minimum != len(module.layers):
                return False
        return True

    def forward_batch(self, context, targets, queries):
        from tabpfn.architectures.interface import PerformanceOptions

        x = torch.cat((context, queries), dim=1)
        if len(context) > 1 and hasattr(self.model, "_add_column_embeddings") and (
                x[:, 1:] == x[:, :1]).all(dim=1).any():
            # v2.5 removes constant columns for B=1 but pads them for B>1.
            # That changes token count and attention; preserve scalar semantics.
            raise ValueError("TabPFN v2.5 constant columns require --task-batch-size 1")
        logits = self.model(
            x.transpose(0, 1), targets.transpose(0, 1),
            only_return_standard_out=True,
            performance_options=PerformanceOptions(force_recompute_layer=False,
                                                    save_peak_memory_factor=None,
                                                    use_chunkwise_inference=False,
                                                    enable_torch_compile=False),
            task_type="regression")
        return self.criterion.mean(logits).transpose(0, 1)

    @staticmethod
    def _scale_batch_values(values, scopes, batch_count, layout):
        # Feature attention folds (task/state, table row) into the leading axis.
        # Internal memory chunking is disabled so that row ownership is stable.
        layout.check_batch(values.shape[0], batch_count)
        factors = values.new_ones(values.shape[:2])
        for scope, entries in scopes.items():
            slots, sources, epsilons = zip(*entries)
            ids = layout.indices(slots, scope, device=values.device)
            selected_rows = len(ids) // len(slots)
            sources = torch.as_tensor(sources, device=values.device).repeat_interleave(selected_rows)
            factors[ids, sources] = 1 - values.new_tensor(epsilons).repeat_interleave(selected_rows)
        return values * factors.reshape(*factors.shape, *([1] * (values.ndim - 2)))

    @contextmanager
    def _source_gate(self, gates, *, layout: RowLayout):
        layout = replace(layout, n_prefix=self.thinking_rows)
        layers = {}
        for slot, gate in enumerate(gates):
            if not gate.is_baseline:
                sources = self.d if self.v3 else math.ceil(self.d / self.group_size)
                self._check_gate(gate, len(self.attentions), sources)
                # v3 places CLS tokens before the feature tokens.
                source = gate.source_token + (
                    self.model.column_aggregator.num_cls_tokens if self.v3 else 0)
                layers.setdefault(gate.layer, {}).setdefault(gate.row_scope, []).append(
                    (slot, source, gate.epsilon))

        def value_hook(entries):
            def scale(module, args, values):
                return self._scale_batch_values(values, entries, len(gates), layout)
            return scale

        def qkv_method(original, entries):
            def scale(*args, **kwargs):
                q, k, v, kv, qkv = original(*args, **kwargs)
                if qkv is not None:
                    qkv = qkv.clone()
                    qkv[:, :, 2] = self._scale_batch_values(qkv[:, :, 2], entries, len(gates), layout)
                elif kv is not None:
                    kv = kv.clone()
                    kv[:, :, 1] = self._scale_batch_values(kv[:, :, 1], entries, len(gates), layout)
                else:
                    v = self._scale_batch_values(v, entries, len(gates), layout)
                return q, k, v, kv, qkv
            return scale

        with ExitStack() as stack:
            for layer, entries in layers.items():
                attention = self.attentions[layer]
                if hasattr(attention, "v_projection"):
                    handle = attention.v_projection.register_forward_hook(value_hook(entries))
                    stack.callback(handle.remove)
                else:
                    stack.enter_context(patch_method(attention, "compute_qkv",
                                                     qkv_method(attention.compute_qkv, entries)))
            yield

    @contextmanager
    def _branch_gate(self, entries, *, layout, batch_count):
        layout = replace(layout, n_prefix=self.thinking_rows)
        scales = self._branch_scales(entries, n_layers=len(self.attentions),
                                     layout=layout, batch_count=batch_count)

        def with_residual(original, scale):
            def forward(x, *args, **kwargs):
                # v2 may reuse x's storage; preserve the residual for interpolation.
                kwargs["allow_inplace"] = False
                output = original(x, *args, **kwargs)
                return scale.apply(output, residual=x if kwargs.get("add_input", False) else None)
            return forward

        with ExitStack() as stack:
            for layer, scale in scales.items():
                attention = self.attentions[layer]
                if self.version == "v2":
                    stack.enter_context(patch_method(
                        attention, "forward", with_residual(attention.forward, scale)))
                else:
                    # v2.5 returns a pure update; its caller owns the residual.
                    handle = attention.register_forward_hook(
                        lambda module, args, output, scale=scale: scale.apply(output))
                    stack.callback(handle.remove)
            yield

    def __init__(self, model, criterion, *, version, **kwargs):
        super().__init__(model, **kwargs)
        self.criterion = criterion.to(device=self.device, dtype=self.dtype)
        self.version = version
        self.v3 = hasattr(model, "column_aggregator")
        if self.v3:
            self.attentions = [block.attention for block in model.column_aggregator.blocks]
        else:
            self.attentions = [
                attention for module in model.modules()
                for attention in [getattr(module, "self_attn_between_features", None)
                                  or getattr(module, "per_sample_attention_between_features", None)]
                if attention is not None]
        if not self.attentions:
            raise ValueError("unsupported TabPFN feature-attention architecture")
        self.group_size = int(model.feature_group_size if self.v3 else model.features_per_group)
        thinking = (getattr(model, "add_thinking_rows", None)
                    or getattr(model, "add_thinking_tokens", None))
        self.thinking_rows = int(thinking.num_thinking_rows) if thinking is not None else 0
        if getattr(model, "cache_trainset_representation", False):
            raise ValueError("trainset caching must be disabled")
        self._metadata.update(adapter="tabpfn", version=version,
                              inference_api="frozen_model_forward",
                              column_embeddings="native",
                              regression_output="bar_distribution_mean",
                              native_feature_grouping="overlapping" if self.v3 else "contiguous",
                              features_per_group=self.group_size)
        self._metadata["software"]["tabpfn"] = importlib.metadata.version("tabpfn")

    @classmethod
    def load(cls, checkpoint, *, d, version="v2.5", device="cpu", dtype="float32"):
        from tabpfn.model_loading import load_model_criterion_config

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        models, criterion, configs, _ = load_model_criterion_config(
            str(checkpoint), check_bar_distribution_criterion=True,
            cache_trainset_representation=False, which="regressor", version=version,
            download_if_not_exists=False)
        if len(models) != 1:
            raise ValueError("exact-k requires exactly one frozen model")
        config = configs[0]
        for name in ("remove_duplicate_features",):
            if getattr(config, name, False):
                raise ValueError(f"checkpoint violates input protocol: {name}")
        adapter = cls(models[0], criterion, version=version, checkpoint=checkpoint,
                      d=d, target_scaling="context_standardize", device=device, dtype=dtype)
        adapter._metadata["model_config"] = asdict(config) if is_dataclass(config) else vars(config)
        return adapter

    def token_groups(self, layers):
        if self.v3:
            groups = [[{"feature": (j + 2**slot) % self.d, "slot": slot,
                        "column": (j + 2**slot) % self.d}
                       for slot in range(self.group_size)] for j in range(self.d)]
        else:
            groups = [[{"feature": j, "slot": j % self.group_size, "column": j}
                       for j in range(start, min(start + self.group_size, self.d))]
                      for start in range(0, self.d, self.group_size)]
        if any(not 0 <= layer < len(self.attentions) for layer in layers):
            raise ValueError("TabPFN feature-attention layer is out of range")
        return {str(layer): groups for layer in layers}
