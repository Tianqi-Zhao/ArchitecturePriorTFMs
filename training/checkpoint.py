"""Train one registered architecture and atomically persist its checkpoint."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from data.architecture_loaders import build_synthetic_loader
from models.architecture_registry import get_architecture
from training.engine import (
    DEFAULT_EARLY_STOPPING_ACTIVATION_DROP,
    DEFAULT_EARLY_STOPPING_ACTIVATION_RELATIVE,
    DEFAULT_EARLY_STOPPING_BASELINE_EVALS,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EARLY_STOPPING_SMOOTHING,
    DEFAULT_EARLY_STOPPING_VAL_FRACTION,
    DEFAULT_LEARNING_RATE,
    set_randomness_seed,
    train,
)
from training.metrics import ScalarMetricLogger


@dataclass(frozen=True)
class CheckpointTrainingConfig:
    num_features: int = 10
    task_type: str = "regression"
    num_steps: int = 10_000
    batch_size: int = 32
    learning_rate: float = DEFAULT_LEARNING_RATE
    optimizer_name: str = "schedulefree-adamw"
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    adam_beta2: float = 0.95
    target_scaling: str = "context_standardize"
    steps_per_eval: int = 200
    val_num_batches: int = 100
    val_batch_size: int = 32
    shuffle_features: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    early_stopping_monitor: str = "es_val_loss"
    early_stopping_mode: str = "min"
    restore_best_weights: bool = True
    early_stopping_val_fraction: float = DEFAULT_EARLY_STOPPING_VAL_FRACTION
    early_stopping_start_after_steps: int = 0
    early_stopping_smoothing: float = DEFAULT_EARLY_STOPPING_SMOOTHING
    early_stopping_activation_drop: float = DEFAULT_EARLY_STOPPING_ACTIVATION_DROP
    early_stopping_activation_relative: bool = DEFAULT_EARLY_STOPPING_ACTIVATION_RELATIVE
    early_stopping_baseline_evals: int = DEFAULT_EARLY_STOPPING_BASELINE_EVALS
    early_stopping_source: str = "internal"
    model_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointTrainingResult:
    checkpoint_path: Path
    history: list[dict[str, Any]]
    metrics: dict[str, Any]


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def train_checkpoint(
    *,
    run: Mapping[str, Any],
    architecture_name: str,
    config: CheckpointTrainingConfig,
    device: torch.device,
    seed: int,
    destination: str | Path,
    metric_logger: ScalarMetricLogger | None = None,
) -> CheckpointTrainingResult:
    """Train only; performance benchmarking is owned by Experiment 0."""

    train_path = str(run["train_path"])
    val_path = run.get("val_path")
    destination = Path(destination)
    architecture = get_architecture(architecture_name)
    set_randomness_seed(seed)

    overrides = dict(config.model_config)
    overrides.setdefault("num_outputs", 2 if config.task_type == "classification" else 1)
    if "num_features" in architecture.defaults:
        overrides.setdefault("num_features", config.num_features)
    built = architecture.build(overrides)

    prior = build_synthetic_loader(
        architecture,
        train_path,
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        num_features=config.num_features,
        request_feature_shuffle=config.shuffle_features,
        device=device,
    )
    val_prior = None
    if val_path:
        val_prior = build_synthetic_loader(
            architecture,
            str(val_path),
            num_steps=config.val_num_batches,
            batch_size=config.val_batch_size,
            num_features=config.num_features,
            request_feature_shuffle=False,
            device=device,
        )

    model, history = train(
        built.model,
        prior,
        lr=config.learning_rate,
        device=device,
        optimizer_name=config.optimizer_name,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        adam_beta2=config.adam_beta2,
        target_scaling=config.target_scaling,
        metric_logger=metric_logger,
        steps_per_eval=config.steps_per_eval,
        classifier_class=built.classifier_class,
        val_prior=val_prior,
        val_steps=config.val_num_batches,
        task_type=config.task_type,
        early_stopping=config.early_stopping,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_monitor=config.early_stopping_monitor,
        early_stopping_mode=config.early_stopping_mode,
        restore_best_weights=config.restore_best_weights,
        early_stopping_val_fraction=config.early_stopping_val_fraction,
        early_stopping_start_after_steps=config.early_stopping_start_after_steps,
        early_stopping_smoothing=config.early_stopping_smoothing,
        early_stopping_activation_drop=config.early_stopping_activation_drop,
        early_stopping_activation_relative=config.early_stopping_activation_relative,
        early_stopping_baseline_evals=config.early_stopping_baseline_evals,
        early_stopping_source=config.early_stopping_source,
    )

    serialized_training_config = asdict(config)
    serialized_training_config["model_config"] = dict(config.model_config)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "task_type": config.task_type,
        "run_name": str(run["run_name"]),
        "seed": int(seed),
        "train_path": train_path,
        "training_config": serialized_training_config,
        "run_metadata": dict(run.get("metadata") or {}),
        **built.checkpoint_metadata(),
    }
    checkpoint.setdefault("num_features", config.num_features)
    _atomic_torch_save(checkpoint, destination)

    metrics: dict[str, Any] = {
        "checkpoint_path": str(destination),
        "history/num_evals": len(history),
        "history/early_stopping_enabled": config.early_stopping,
    }
    if history:
        last = history[-1]
        for key in (
            "iteration",
            "training_time",
            "early_stopped",
            "stop_iter",
            "best_iter",
            "best_es_val_loss",
            "best_val_loss",
            "best_train_loss",
        ):
            if key in last and last[key] is not None:
                metrics[f"history/{key}"] = last[key]
    return CheckpointTrainingResult(destination, history, metrics)
