"""Shared frozen-model prediction, normalization, and input-gradient execution."""

from abc import abstractmethod
from contextlib import ExitStack, contextmanager, nullcontext
import math
from pathlib import Path

import numpy as np
import torch

from ..artifacts.storage import sha256
from ..artifacts.protocol import Gate
from .base import ModelAdapter, Prediction
from .gates import FeatureBranchGate, RowLayout


class TorchAdapter(ModelAdapter):
    """Subclasses supply batched forward and gate operations; B=1 uses the same path."""

    def __init__(self, model, *, checkpoint, d, target_scaling, device="cpu",
                 dtype="float32"):
        if dtype not in ("float32", "float64"):
            raise ValueError("collection dtype must be float32 or float64")
        if target_scaling not in ("none", "context_standardize"):
            raise ValueError("unsupported target scaling")
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.model = model.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        self.d, self.target_scaling = d, target_scaling
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self._metadata = {
            "checkpoint": {"path": str(self.checkpoint), "sha256": sha256(self.checkpoint)}
            if checkpoint else None,
            "device": str(self.device), "dtype": dtype,
            "software": {"torch": torch.__version__, "numpy": np.__version__},
            "preprocessing": {"input": "raw_columns", "target": target_scaling,
                              "fit_on": "task_context", "target_std_floor": 1e-8},
            "n_estimators": 1, "augmentation": False, "feature_order": "original",
        }

    @property
    def metadata(self):
        return self._metadata

    def _normalize_targets(self, y):
        """Fit each batch entry's target scale using only its own context."""
        mean, scale = y.new_zeros((len(y), 1)), y.new_ones((len(y), 1))
        if self.target_scaling == "context_standardize":
            mean = y.mean(dim=1, keepdim=True)
            std = y.std(dim=1, keepdim=True, unbiased=False)
            scale = torch.where(std < 1e-8, torch.ones_like(std), std)
        return (y - mean) / scale, mean, scale

    def predict_batch(self, requests, *, gradients=False, batch_size=256):
        if not requests or batch_size < 1:
            raise ValueError("a nonempty request batch and positive query batch size are required")
        if len(requests) > 1 and not self.supports_batch_prediction:
            # Stochastic/nonstandard backends retain each request's seed. B=1
            # still executes the same packing, forward, gate, and gradient path.
            return [self.predict_batch([request], gradients=gradients,
                                       batch_size=batch_size)[0] for request in requests]
        n = len(requests[0].task.context_x)
        q = len(requests[0].queries)
        for request in requests:
            if (n < 1 or q < 1 or request.task.context_x.shape != (n, self.d)
                    or request.task.context_y.shape != (n,)
                    or request.queries.shape != (q, self.d)):
                raise ValueError("batched requests must share context length and query shape")
            if not all(np.isfinite(a).all() for a in (
                    request.task.context_x, request.task.context_y, request.queries)):
                raise ValueError("batched requests contain nonfinite inputs")
        context = torch.as_tensor(np.stack([r.task.context_x for r in requests]),
                                  device=self.device, dtype=self.dtype)
        y = torch.as_tensor(np.stack([r.task.context_y for r in requests]),
                            device=self.device, dtype=self.dtype)
        targets, mean, scale = self._normalize_targets(y)
        queries = np.stack([r.queries for r in requests])
        return self._predict_queries(
            context, targets, queries, mean, scale, gates=[r.gate for r in requests],
            seed=requests[0].seed, gradients=gradients, batch_size=batch_size)

    def _predict_queries(self, context, targets, queries, mean, scale, *, gates, seed,
                         gradients, batch_size):
        """Execute all requests through one query-chunk and gradient loop."""
        if not np.isfinite(queries).all() or batch_size < 1:
            raise ValueError("invalid queries or batch size")
        if self.model.training:
            raise ValueError("model must remain in eval mode")
        b, q, _ = queries.shape
        values, gradient_sum = [], np.zeros((b, self.d))
        counts = np.zeros(b, dtype=np.int64)
        errors = [None] * b
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        # MATH SDPA supports input derivatives; forward-only work can use the
        # default backend. Gates may independently request attention weights.
        attention = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                     if gradients else nullcontext())
        autograd = torch.enable_grad() if gradients else torch.inference_mode()
        with attention, autograd:
            for start in range(0, q, batch_size):
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    x = torch.as_tensor(queries[:, start:start + batch_size],
                                        device=self.device, dtype=self.dtype)
                    if gradients:
                        x = x.detach().requires_grad_(True)
                    layout = RowLayout(context.shape[1], x.shape[1])
                    with self.batch_gate(gates, layout=layout):
                        prediction = self.forward_batch(context, targets, x) * scale + mean
                        if prediction.shape != x.shape[:2] or not torch.isfinite(prediction).all():
                            raise ValueError("model returned invalid query predictions")
                        values.append(prediction.detach().cpu().numpy())
                        if gradients and any(error is None for error in errors):
                            try:
                                derivative = torch.autograd.grad(prediction.sum(), x)[0]
                                finite = torch.isfinite(derivative).all(dim=(1, 2)).cpu().numpy()
                                sums = derivative.detach().to("cpu", torch.float64).sum(1).numpy()
                                for i in range(b):
                                    if errors[i] is None:
                                        if finite[i]:
                                            gradient_sum[i] += sums[i]
                                            counts[i] += x.shape[1]
                                        else:
                                            errors[i] = "nonfinite input derivatives"
                            except (RuntimeError, ValueError) as error:
                                errors = [previous or str(error) for previous in errors]
        values = np.concatenate(values, axis=1)
        return [Prediction(values[i], gradient_sum[i] / q if counts[i] == q else None,
                           int(counts[i]), errors[i]) for i in range(b)]

    @abstractmethod
    def forward_batch(self, context, targets, queries):
        """Map (B,N,D), (B,N), (B,Q,D) tensors to normalized predictions (B,Q)."""

    @contextmanager
    def batch_gate(self, gates, *, layout: RowLayout):
        """Install backend gates while preserving one batch/forward/gradient path."""
        sources = [gate if gate.kind == "source" else Gate() for gate in gates]
        branches = [(slot, gate) for slot, gate in enumerate(gates)
                    if gate.kind == "feature_branch"]
        with ExitStack() as stack:
            if branches:
                if not self.supports_feature_branch:
                    raise ValueError("this model does not support feature_branch interventions")
                stack.enter_context(self._branch_gate(branches, layout=layout, batch_count=len(gates)))
            stack.enter_context(self._source_gate(sources, layout=layout))
            yield

    @abstractmethod
    def _source_gate(self, gates, *, layout: RowLayout):
        """Backend context manager for independent source gates."""

    def _branch_gate(self, entries, *, layout, batch_count):
        raise ValueError("this model does not support feature_branch interventions")

    def _branch_scales(self, entries, *, n_layers, layout, batch_count):
        layers = {}
        for slot, gate in entries:
            self._check_gate(gate, n_layers, 0)
            if gate.epsilon != 0:
                layers.setdefault(gate.layer, {}).setdefault(gate.row_scope, []).append(
                    (slot, gate.epsilon))
        return {layer: FeatureBranchGate(layout, batch_count, scopes)
                for layer, scopes in layers.items()}

    def _check_gate(self, gate, n_layers, sources):
        valid_source = (0 <= gate.source_token < sources if gate.kind == "source"
                        else gate.source_token == -1)
        if (not isinstance(gate.layer, int) or isinstance(gate.layer, bool)
                or not isinstance(gate.source_token, int) or isinstance(gate.source_token, bool)
                or not 0 <= gate.layer < n_layers or not valid_source
                or not math.isfinite(gate.epsilon) or not 0 <= gate.epsilon <= 1):
            raise ValueError("invalid layer, source token or epsilon")
