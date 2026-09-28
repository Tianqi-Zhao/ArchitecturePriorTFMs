"""Pure optimization engine for registered NanoTab architectures."""

from __future__ import annotations

import copy
import random
import time
from typing import Literal

import h5py
import numpy as np
import torch
from data.task_types import normalize_task_type as _normalize_task_type
from models.architecture_registry import classifier_class_for_model
from torch import nn
from torch.utils.data import DataLoader

from data.synthetic_data.datasets import (
    PriorDumpDataLoader,
    PriorDumpDataLoader1D,
    get_default_device,
    table_padding_masks,
)
from training.optimization import OptimizerController, OptimizerName
from training.metrics import ScalarMetricLogger


TargetScaling = Literal["none", "context_standardize"]


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def standardize_regression_targets_torch(
    y: torch.Tensor,
    train_test_split_index: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize regression targets using train-context statistics only."""
    if train_test_split_index <= 0:
        raise ValueError("train_test_split_index must be > 0 for regression scaling")

    y = y.to(torch.float32)
    squeeze_back = False
    if y.ndim == 1:
        y = y.unsqueeze(0)
        squeeze_back = True
    elif y.ndim != 2:
        raise ValueError(f"Expected y to be 1D or 2D, got shape={tuple(y.shape)}")

    y_train = y[:, :train_test_split_index]
    y_mean = y_train.mean(dim=1, keepdim=True)
    y_std = y_train.std(dim=1, keepdim=True, unbiased=False)
    y_std = torch.where(y_std < eps, torch.ones_like(y_std), y_std)
    y_scaled = (y - y_mean) / y_std

    if squeeze_back:
        return y_scaled.squeeze(0), y_mean.squeeze(0), y_std.squeeze(0)
    return y_scaled, y_mean, y_std


def prepare_regression_targets_torch(
    y: torch.Tensor,
    train_test_split_index: int,
    target_scaling: TargetScaling,
) -> torch.Tensor:
    """Apply the configured regression-target transform."""

    if target_scaling == "none":
        return y.to(torch.float32)
    if target_scaling == "context_standardize":
        scaled, _, _ = standardize_regression_targets_torch(
            y,
            train_test_split_index,
        )
        return scaled
    raise ValueError(
        f"unknown target_scaling {target_scaling!r}; "
        "expected 'none' or 'context_standardize'"
    )


def standardize_regression_train_targets_numpy(
    y_train: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float, float]:
    """Standardize train targets and return transformed values with stats."""
    y_train = np.asarray(y_train, dtype=np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < eps:
        y_std = 1.0
    y_train_scaled = (y_train - y_mean) / y_std
    return y_train_scaled, y_mean, y_std


def inverse_regression_target_scaling_numpy(
    y_scaled: np.ndarray,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    """Map standardized regression outputs back to raw target space."""
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    return y_scaled * float(y_std) + float(y_mean)


def _forward_mask_kwargs(
    padding_mask: torch.Tensor | None,
    feature_padding_mask: torch.Tensor | None,
) -> dict:
    kwargs = {}
    if padding_mask is not None:
        kwargs["padding_mask"] = padding_mask
    if feature_padding_mask is not None:
        kwargs["feature_padding_mask"] = feature_padding_mask
    return kwargs


def evaluate_validation_loss(
    model: nn.Module,
    prior: DataLoader,
    device: torch.device = None,
    max_batches: int = 100,
    task_type: str = "classification",
    target_scaling: TargetScaling = "context_standardize",
) -> float:
    if not device:
        device = get_default_device()
    task_type = _normalize_task_type(task_type)
    criterion: nn.Module
    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    was_training = model.training
    model.eval()

    # Keep validation comparable across eval points by always starting from
    # the same slice of the prior dump. Without this, pointer-based loaders
    # create cyclic val_loss patterns from changing validation windows.
    # For slice-based loaders (e.g. the early-stop split), pointer_start marks
    # the intended start of the slice; fall back to 0 for regular val loaders.
    if hasattr(prior, "pointer"):
        prior.pointer = int(getattr(prior, "pointer_start", 0))

    losses = []
    try:
        with torch.no_grad():
            for step, full_data in enumerate(prior):
                if step >= max_batches:
                    break

                train_test_split_index = full_data["train_test_split_index"]
                y_all = full_data["y"].to(device)
                if task_type == "regression":
                    y_all = prepare_regression_targets_torch(
                        y_all,
                        train_test_split_index,
                        target_scaling,
                    )

                data = (
                    full_data["x"].to(device).float(),
                    y_all[:, :train_test_split_index].to(torch.float32),
                )
                targets = y_all

                padding_mask, feature_padding_mask = table_padding_masks(
                    full_data, device=device
                )
                output = model(
                    data,
                    train_test_split_index=train_test_split_index,
                    **_forward_mask_kwargs(padding_mask, feature_padding_mask),
                )

                targets = targets[:, train_test_split_index:]
                if task_type == "classification":
                    targets = targets.reshape((-1,)).to(torch.long)
                    output = output.view(-1, output.shape[-1])
                else:
                    targets = targets.reshape((-1,)).to(torch.float32)
                    output = output.reshape((-1,))

                if padding_mask is not None:
                    valid_test = ~padding_mask[:, train_test_split_index:].reshape(-1)
                    if not valid_test.any():
                        continue
                    output = output[valid_test]
                    targets = targets[valid_test]

                loss = criterion(output, targets).mean()
                losses.append(float(loss.detach().cpu().item()))
    finally:
        model.train(was_training)
    if len(losses) == 0:
        return float("nan")
    return float(np.mean(losses))


_UNSET = object()

# Shared defaults for all synthetic-prior training entry points. Keeping these
# next to train() prevents experiment runners from silently using a different
# early-stopping policy.
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4
DEFAULT_EARLY_STOPPING_VAL_FRACTION = 0.05
DEFAULT_EARLY_STOPPING_SMOOTHING = 0.5
DEFAULT_EARLY_STOPPING_ACTIVATION_DROP = 0.05
DEFAULT_EARLY_STOPPING_ACTIVATION_RELATIVE = True
DEFAULT_EARLY_STOPPING_BASELINE_EVALS = 3
DEFAULT_LEARNING_RATE = 2e-3


def train(model: nn.Module, prior: DataLoader,
          lr: float = DEFAULT_LEARNING_RATE, device: torch.device = None, steps_per_eval=10,
          eval_func=None, classifier_class=None, val_prior: DataLoader = None,
          val_steps: int = 100, task_type: str = "classification",
          target_scaling: TargetScaling = "context_standardize",
          optimizer_name: OptimizerName = "schedulefree-adamw",
          weight_decay: float = 0.0,
          gradient_clip_norm: float = 1.0,
          adam_beta2: float = 0.95,
          metric_logger: ScalarMetricLogger | None = None,
          early_stopping=_UNSET,
          early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
          early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
          early_stopping_monitor: str = "es_val_loss",
          early_stopping_mode: str = "min",
          restore_best_weights: bool = True,
          early_stopping_val_fraction: float = DEFAULT_EARLY_STOPPING_VAL_FRACTION,
          early_stopping_start_after_steps: int = 0,
          early_stopping_smoothing: float = DEFAULT_EARLY_STOPPING_SMOOTHING,
          early_stopping_activation_drop: float = DEFAULT_EARLY_STOPPING_ACTIVATION_DROP,
          early_stopping_activation_relative: bool = DEFAULT_EARLY_STOPPING_ACTIVATION_RELATIVE,
          early_stopping_baseline_evals: int = DEFAULT_EARLY_STOPPING_BASELINE_EVALS,
          early_stopping_source: Literal["internal", "validation"] = "internal"):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: one of the repository's 2D, 1D, or 1DICL models
        prior: (DataLoader) torch-compatible dataloader
        lr: (float) learning rate
        device: (torch.device) the device we are using
        steps_per_eval: (int) how many steps we wait before running evaluation again
        eval_func: a function that takes in a classifier and returns a dict containing the average scores
                   for some metrics and datasets
        classifier_class: the classifier class to use for evaluation (auto-detected from model if None)
        val_prior: optional validation prior dataloader
        val_steps: number of validation batches to average for val_loss
        target_scaling: regression-target transform; ``none`` preserves the
            controlled Bayes scale and ``context_standardize`` uses context statistics.
        optimizer_name: ``schedulefree-adamw`` or AdamW with cosine decay (``adamw``).
        weight_decay: optimizer weight decay.
        gradient_clip_norm: maximum gradient norm.
        adam_beta2: beta2 used by regular AdamW.
        metric_logger: optional scalar sink for per-step train loss and validation loss.
        early_stopping: True/False to enable/disable early stopping. When omitted, it is enabled
                        for synthetic H5 prior loaders and disabled for unsupported real-data loaders.
        early_stopping_patience: number of eval intervals with no improvement before stopping (default 5)
        early_stopping_min_delta: minimum change in monitored metric to count as an improvement (default 1e-4)
        early_stopping_monitor: metric name to monitor, must appear in the eval record
            (default "es_val_loss" — the internal early-stop validation loss)
        early_stopping_mode: "min" to stop when metric stops decreasing, "max" for increasing (default "min")
        restore_best_weights: if True (default), load the best-seen state_dict before returning
        early_stopping_source: ``internal`` reserves rows from the training H5;
            ``validation`` monitors the explicit ``val_prior`` without modifying train rows.
        early_stopping_val_fraction: with the ``internal`` source, reserve this fraction of H5 rows
            (i.e. of X.shape[0]) as the internal early-stop validation slice. The split boundary
            is the largest multiple of batch_size not exceeding (1-fraction)*n_rows, so batches
            are always row-aligned and the two regions never overlap. Training reads only from
            [0, boundary) and es-val reads only from [boundary, n_rows), regardless of num_steps
            (the loader wraps within its own region). Requires `prior` to be a
            PriorDumpDataLoader or PriorDumpDataLoader1D. Default: 0.05.
        early_stopping_start_after_steps: do not count patience toward early stopping until the
            training iteration index reaches this value (1-based: first step is 1). While below
            this threshold, es_val_loss is still logged and best weights can still update on
            improvement, but evals_since_improve is not incremented and training never stops early.
            Default: 0 (patience active from the first eval).
        early_stopping_smoothing: EMA weight beta in [0, 1) applied to the monitored metric BEFORE
            best/patience comparisons. Update rule: smoothed_t = beta * smoothed_{t-1} + (1-beta) * raw_t.
            beta=0.0 disables smoothing (identical to historical behaviour). Higher beta -> more
            smoothing and more lag (effective averaging window ~ 1/(1-beta)). Smoothing prevents a
            single noisy low/high eval point on a plateau from latching as "best" and silently
            burning patience for the rest of training. Recommended for 2D runs with long plateaus:
            0.5. Default: 0.5.
        early_stopping_activation_drop: minimum drop (in min-mode) or rise (in max-mode) of the
            SMOOTHED metric relative to a `baseline_value` before patience is allowed to count
            toward stopping. 0.0 disables this gate (identical to historical behaviour). When >0,
            patience stays inactive on the initial plateau and only activates once the model has
            shown a clear improvement of at least this magnitude; once activated, it stays active
            for the rest of training (sticky). The baseline_value is the median of the first
            `early_stopping_baseline_evals` raw monitored values. Use this together with
            `early_stopping_smoothing` to avoid early stopping during the long pre-descent plateau
            seen in 2D models. Default: 0.05.
        early_stopping_activation_relative: if True, interpret `early_stopping_activation_drop` as a
            fraction of |baseline_value| (relative), otherwise as an absolute change.
            Use relative for regression where the baseline scale varies. Default: True.
        early_stopping_baseline_evals: number of initial eval points whose median forms
            `baseline_value` for the activation-drop gate. Must be >= 1. Default: 3.

    Returns:
        (model) our trained numpy model
        (list) a list of dicts containing iteration/time/losses/metrics

    Raises:
        KeyboardInterrupt: propagated if training is interrupted. Callers must
            not persist the returned-from-train checkpoint path in that case;
            an unfinished run is not a completed artifact and must be retried.
    """
    if early_stopping is _UNSET:
        early_stopping = isinstance(
            prior,
            (PriorDumpDataLoader, PriorDumpDataLoader1D),
        )

    es_val_prior = None
    es_val_step_count = 0

    if early_stopping:
        if early_stopping_mode not in {"min", "max"}:
            raise ValueError(f"early_stopping_mode must be 'min' or 'max', got {early_stopping_mode!r}")
        if early_stopping_source not in {"internal", "validation"}:
            raise ValueError(
                "early_stopping_source must be 'internal' or 'validation', "
                f"got {early_stopping_source!r}"
            )

        if early_stopping_source == "validation":
            if val_prior is None:
                raise ValueError(
                    "early_stopping_source='validation' requires val_prior"
                )
            es_val_prior = val_prior
            es_val_step_count = val_steps
            print(
                "[early-stop] using the explicit validation loader "
                f"({es_val_step_count} batches)"
            )
        else:
            # Carve an internal validation region while keeping loader batches aligned.
            if not (0.0 < early_stopping_val_fraction < 1.0):
                raise ValueError(
                    "early_stopping_val_fraction must be in (0, 1) when "
                    "early_stopping_source='internal', got "
                    f"{early_stopping_val_fraction}"
                )
            if not all(
                hasattr(prior, attribute)
                for attribute in ("filename", "num_steps", "batch_size")
            ):
                raise TypeError(
                    "internal early stopping requires a PriorDumpDataLoader or "
                    "PriorDumpDataLoader1D"
                )
            batch_size = prior.batch_size
            with h5py.File(prior.filename, "r") as source:
                n_rows = int(source["X"].shape[0])
            boundary = (
                int((1.0 - early_stopping_val_fraction) * n_rows) // batch_size
            ) * batch_size
            es_val_rows = n_rows - boundary
            if boundary < batch_size:
                raise ValueError(
                    f"early_stopping_val_fraction={early_stopping_val_fraction} "
                    f"leaves fewer than batch_size={batch_size} training rows"
                )
            if es_val_rows < batch_size:
                raise ValueError(
                    f"early_stopping_val_fraction={early_stopping_val_fraction} "
                    f"leaves fewer than batch_size={batch_size} validation rows"
                )

            prior.pointer_start = 0
            prior.pointer_end = boundary
            prior.pointer = 0
            es_val_step_count = es_val_rows // batch_size
            loader_arguments = dict(
                filename=prior.filename,
                num_steps=es_val_step_count,
                batch_size=batch_size,
                shuffle_features=getattr(prior, "shuffle_features", False),
                device=prior.device,
                pointer_start=boundary,
                pointer_end=n_rows,
            )
            if isinstance(prior, PriorDumpDataLoader1D):
                es_val_prior = PriorDumpDataLoader1D(
                    num_features=prior.num_features,
                    **loader_arguments,
                )
            else:
                es_val_prior = PriorDumpDataLoader(**loader_arguments)
            print(
                f"[early-stop] split (rows): train [0, {boundary}) / "
                f"es-val [{boundary}, {n_rows}), batch_size={batch_size}"
            )
            print(f"[early-stop] es_val_step_count={es_val_step_count}")

        if early_stopping_monitor == "es_val_loss" and es_val_prior is None:
            raise ValueError(
                "early_stopping_monitor='es_val_loss' is set but es_val_prior could not be created. "
                "Ensure early_stopping_val_fraction > 0."
            )
        if early_stopping_monitor not in {"es_val_loss", "val_loss", "train_loss"} and eval_func is None:
            raise ValueError(
                f"early_stopping_monitor={early_stopping_monitor!r} is not a built-in metric key "
                "and no eval_func was provided to produce it."
            )
        if early_stopping_start_after_steps < 0:
            raise ValueError(
                f"early_stopping_start_after_steps must be >= 0, got {early_stopping_start_after_steps}"
            )
        if early_stopping_start_after_steps > 0:
            print(
                f"[early-stop] patience inactive until training step >= {early_stopping_start_after_steps} "
                f"(1-based iteration counter)"
            )
        if not (0.0 <= early_stopping_smoothing < 1.0):
            raise ValueError(
                f"early_stopping_smoothing must be in [0, 1), got {early_stopping_smoothing}"
            )
        if early_stopping_activation_drop < 0.0:
            raise ValueError(
                f"early_stopping_activation_drop must be >= 0, got {early_stopping_activation_drop}"
            )
        if early_stopping_baseline_evals < 1:
            raise ValueError(
                f"early_stopping_baseline_evals must be >= 1, got {early_stopping_baseline_evals}"
            )
        if early_stopping_smoothing > 0.0:
            print(
                f"[early-stop] EMA smoothing beta={early_stopping_smoothing} "
                f"(effective window ~ {1.0/(1.0-early_stopping_smoothing):.1f} evals)"
            )
        if early_stopping_activation_drop > 0.0:
            kind = "relative" if early_stopping_activation_relative else "absolute"
            print(
                f"[early-stop] patience gated by {kind} drop of >= {early_stopping_activation_drop} "
                f"from baseline (median of first {early_stopping_baseline_evals} raw evals); "
                "patience stays inactive until that drop is reached."
            )
        best_value = float("inf") if early_stopping_mode == "min" else float("-inf")
        best_iter: int | None = None
        best_state: dict | None = None
        evals_since_improve = 0
        _early_stopped = False
        smoothed_value: float | None = None
        baseline_buffer: list[float] = []
        baseline_value: float | None = None
        activation_drop_satisfied = (early_stopping_activation_drop <= 0.0)

    if classifier_class is None:
        classifier_class = classifier_class_for_model(model)

    if not device:
        device = get_default_device()
    if steps_per_eval < 1:
        raise ValueError("steps_per_eval must be >= 1")
    if val_steps < 1:
        raise ValueError("val_steps must be >= 1")
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be > 0")
    if target_scaling not in {"none", "context_standardize"}:
        raise ValueError(
            "target_scaling must be 'none' or 'context_standardize', "
            f"got {target_scaling!r}"
        )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    if optimizer_name == "adamw":
        try:
            optimizer_steps = len(prior)
        except TypeError as error:
            raise TypeError("AdamW cosine scheduling requires a sized prior") from error
    else:
        optimizer_steps = int(getattr(prior, "num_steps", 1))
    optimizer = OptimizerController.build(
        model.parameters(),
        name=optimizer_name,
        learning_rate=lr,
        weight_decay=weight_decay,
        adam_beta2=adam_beta2,
        total_steps=optimizer_steps,
    )
    task_type = _normalize_task_type(task_type)
    criterion: nn.Module
    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    model.train()
    optimizer.train()

    train_time = 0
    eval_history = []
    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]
            y_all = full_data["y"].to(device)
            if task_type == "regression":
                y_all = prepare_regression_targets_torch(
                    y_all,
                    train_test_split_index,
                    target_scaling,
                )

            data = (
                full_data["x"].to(device).float(),
                y_all[:, :train_test_split_index].to(torch.float32),
            )
            targets = y_all

            padding_mask, feature_padding_mask = table_padding_masks(
                full_data, device=device
            )
            output = model(
                data,
                train_test_split_index=train_test_split_index,
                **_forward_mask_kwargs(padding_mask, feature_padding_mask),
            )

            targets = targets[:, train_test_split_index:]
            if task_type == "classification":
                targets = targets.reshape((-1,)).to(torch.long)
                output = output.view(-1, output.shape[-1])
            else:
                targets = targets.reshape((-1,)).to(torch.float32)
                output = output.reshape((-1,))

            if padding_mask is not None:
                valid_test = ~padding_mask[:, train_test_split_index:].reshape(-1)
                if not valid_test.any():
                    continue
                output = output[valid_test]
                targets = targets[valid_test]

            loss = criterion(output, targets).mean()
            total_loss = float(loss.cpu().detach().item())

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            if metric_logger is not None:
                metric_logger.log_scalar("loss/train", total_loss, step + 1)
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            if step % steps_per_eval == steps_per_eval - 1:
                model.eval()
                optimizer.eval()

                val_loss = None
                if val_prior is not None:
                    val_loss = evaluate_validation_loss(
                        model,
                        val_prior,
                        device=device,
                        max_batches=val_steps,
                        task_type=task_type,
                        target_scaling=target_scaling,
                    )

                es_val_loss = None
                if es_val_prior is val_prior:
                    es_val_loss = val_loss
                elif es_val_prior is not None:
                    es_val_loss = evaluate_validation_loss(
                        model,
                        es_val_prior,
                        device=device,
                        max_batches=es_val_step_count,
                        task_type=task_type,
                        target_scaling=target_scaling,
                    )

                if metric_logger is not None:
                    if val_loss is not None:
                        metric_logger.log_scalar(
                            "loss/validation",
                            float(val_loss),
                            step + 1,
                        )
                    if es_val_loss is not None and es_val_prior is not val_prior:
                        metric_logger.log_scalar(
                            "loss/early_stop_validation",
                            float(es_val_loss),
                            step + 1,
                        )
                    metric_logger.flush()

                scores = {}
                if eval_func is not None:
                    classifier = classifier_class(model, device)
                    scores = eval_func(classifier)

                record = {
                    "iteration": int(step + 1),
                    "training_time": float(train_time),
                    "train_loss": float(total_loss),
                }
                if val_loss is not None:
                    record["val_loss"] = float(val_loss)
                if es_val_loss is not None:
                    record["es_val_loss"] = float(es_val_loss)
                record.update({k: float(v) for k, v in scores.items()})
                eval_history.append(record)

                parts = [f"iter {step + 1:6d}", f"time {train_time:7.1f}s", f"train_loss {total_loss:7.4f}"]
                if es_val_loss is not None:
                    parts.append(f"es_val_loss {es_val_loss:7.4f}")
                if val_loss is not None:
                    parts.append(f"val_loss {val_loss:7.4f}")
                if scores:
                    parts.append(" | ".join(f"{k} {v:7.4f}" for k, v in scores.items()))
                print(" | ".join(parts))

                if early_stopping:
                    raw_value = record.get(early_stopping_monitor)
                    if raw_value is None:
                        raise ValueError(
                            f"early_stopping_monitor={early_stopping_monitor!r} not found in eval record. "
                            f"Available keys: {list(record.keys())}. "
                            "Make sure the monitored metric is produced by es_val_prior or eval_func."
                        )
                    raw_value = float(raw_value)

                    # EMA smoothing on the monitored metric. Decisions (best/patience) use
                    # the smoothed value so a single noisy eval cannot latch as best on a
                    # plateau and silently burn patience. The raw value is also recorded.
                    if early_stopping_smoothing > 0.0 and smoothed_value is not None:
                        beta = early_stopping_smoothing
                        smoothed_value = beta * smoothed_value + (1.0 - beta) * raw_value
                    else:
                        smoothed_value = raw_value
                    metric_value = smoothed_value
                    eval_history[-1][f"smoothed_{early_stopping_monitor}"] = float(smoothed_value)

                    # Baseline accumulation: take the median of the first
                    # `early_stopping_baseline_evals` RAW values. baseline_value stays None
                    # until enough samples are collected, which disables the drop gate.
                    if baseline_value is None:
                        baseline_buffer.append(raw_value)
                        if len(baseline_buffer) >= early_stopping_baseline_evals:
                            baseline_value = float(np.median(baseline_buffer))
                            print(
                                f"[early-stop] baseline_{early_stopping_monitor}={baseline_value:.4f} "
                                f"(median of first {len(baseline_buffer)} raw evals)"
                            )

                    # Activation by drop: sticky flag, once true stays true. Uses smoothed
                    # value so a transient noisy spike doesn't toggle it. Only checked once
                    # baseline_value is known.
                    if (
                        not activation_drop_satisfied
                        and baseline_value is not None
                        and early_stopping_activation_drop > 0.0
                    ):
                        if early_stopping_mode == "min":
                            improvement = baseline_value - smoothed_value
                        else:
                            improvement = smoothed_value - baseline_value
                        threshold = (
                            early_stopping_activation_drop * abs(baseline_value)
                            if early_stopping_activation_relative
                            else early_stopping_activation_drop
                        )
                        if improvement >= threshold:
                            activation_drop_satisfied = True
                            print(
                                f"[early-stop] activation drop reached at iter {step + 1}: "
                                f"baseline={baseline_value:.4f} smoothed={smoothed_value:.4f} "
                                f"improvement={improvement:.4f} >= threshold={threshold:.4f}. "
                                "Patience is now active."
                            )

                    patience_active = (
                        (step + 1) >= early_stopping_start_after_steps
                        and activation_drop_satisfied
                    )
                    eval_history[-1]["early_stop_patience_active"] = bool(patience_active)
                    eval_history[-1]["early_stop_activation_drop_satisfied"] = bool(activation_drop_satisfied)

                    if early_stopping_mode == "min":
                        is_better = metric_value < best_value - early_stopping_min_delta
                    else:
                        is_better = metric_value > best_value + early_stopping_min_delta

                    if is_better:
                        best_value = metric_value
                        best_iter = step + 1
                        evals_since_improve = 0
                        if restore_best_weights:
                            best_state = copy.deepcopy(model.state_dict())
                        eval_history[-1]["best_iter_so_far"] = best_iter
                        print(f"[early-stop] new best {early_stopping_monitor}={best_value:.4f} at iter {best_iter}")
                    elif patience_active:
                        evals_since_improve += 1
                        if evals_since_improve >= early_stopping_patience:
                            print(
                                f"[early-stop] stopping at iter {step + 1}: "
                                f"no improvement for {evals_since_improve} evals. "
                                f"Best {early_stopping_monitor}={best_value:.4f} at iter {best_iter}."
                            )
                            eval_history[-1]["early_stopped"] = True
                            eval_history[-1]["stop_iter"] = int(step + 1)
                            eval_history[-1]["best_iter"] = best_iter
                            eval_history[-1][f"best_{early_stopping_monitor}"] = best_value
                            _early_stopped = True
                            model.train()
                            optimizer.train()
                            break

                model.train()
                optimizer.train()
    except KeyboardInterrupt:
        # Do not fall through to the normal return: checkpoint training treats
        # a returned model as finished and writes a skippable artifact.
        print(
            "Training interrupted by KeyboardInterrupt; "
            "not saving a completed checkpoint."
        )
        raise

    if early_stopping and eval_history and not _early_stopped:
        eval_history[-1]["early_stopped"] = False
        if best_iter is not None:
            eval_history[-1]["best_iter"] = best_iter
            eval_history[-1][f"best_{early_stopping_monitor}"] = best_value

    # ScheduleFree interpolates weights in train mode. Switch to eval weights
    # before returning or restoring so the saved checkpoint is not the
    # train-mode interpolation.
    model.eval()
    optimizer.eval()

    if early_stopping and restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[early-stop] restored best weights from iter {best_iter} ({early_stopping_monitor}={best_value:.4f})")

    return model, eval_history
