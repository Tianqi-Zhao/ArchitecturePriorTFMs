"""Train one controlled-Bayes model for one hyperparameter combination."""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch

from training.checkpoint import (
    CheckpointTrainingConfig,
    CheckpointTrainingResult,
    train_checkpoint,
)
from training.metrics import TensorBoardMetricLogger


ARCHITECTURE_LAYERS = {"1d": 5, "2d": 3}
OPTIMIZERS = ("schedulefree-adamw",)


@dataclass(frozen=True)
class DatasetManifest:
    """Validated training view of one generated data condition."""

    path: Path
    sha256: str
    d: int
    train_path: Path
    validation_path: Path
    train_rows: int
    validation_rows: int

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        manifest_path = Path(path).expanduser().resolve()
        try:
            content = manifest_path.read_bytes()
            payload = json.loads(content)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read dataset manifest {manifest_path}: {error}"
            ) from error
        if not isinstance(payload, Mapping) or payload.get("complete") is not True:
            raise ValueError("dataset manifest is not marked complete")

        arguments = payload.get("command_arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("dataset manifest has no command_arguments object")
        d = arguments.get("d")
        if not isinstance(d, int) or isinstance(d, bool) or d < 1:
            raise ValueError(f"dataset manifest has invalid d={d!r}")

        splits = payload.get("splits")
        if not isinstance(splits, Mapping):
            raise ValueError("dataset manifest has no splits object")
        train_path, train_rows = cls._split(manifest_path, splits, "train")
        validation_path, validation_rows = cls._split(
            manifest_path,
            splits,
            "validation",
        )
        return cls(
            path=manifest_path,
            sha256=hashlib.sha256(content).hexdigest(),
            d=d,
            train_path=train_path,
            validation_path=validation_path,
            train_rows=train_rows,
            validation_rows=validation_rows,
        )

    @staticmethod
    def _split(
        manifest_path: Path,
        splits: Mapping[str, Any],
        name: str,
    ) -> tuple[Path, int]:
        split = splits.get(name)
        if not isinstance(split, Mapping):
            raise ValueError(f"dataset manifest has no {name!r} split")
        raw_path = split.get("path")
        rows = split.get("rows")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"dataset manifest {name!r} split has no path")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            raise ValueError(
                f"dataset manifest {name!r} split has invalid rows={rows!r}"
            )
        split_path = Path(raw_path).expanduser()
        if not split_path.is_absolute():
            split_path = manifest_path.parent / split_path
        split_path = split_path.resolve()
        if not split_path.is_file():
            raise ValueError(f"dataset {name!r} split does not exist: {split_path}")
        return split_path, rows


@dataclass(frozen=True)
class TrainConfig:
    """All inputs that determine one training run."""

    dataset_manifest: Path
    output_dir: Path
    architecture: str
    optimizer: str
    learning_rate: float
    max_steps: int
    num_layers: int
    num_heads: int = 4
    embedding_size: int = 96
    mlp_hidden_size: int = 192
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    batch_size: int = 32
    validation_interval_steps: int = 500
    validation_batches: int = 100
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    early_stopping_ema: float = 0.0
    early_stopping_start_step: int = 0
    early_stopping_activation_drop: float = 0.0
    restore_best_weights: bool = True
    adam_beta2: float = 0.95
    target_scaling: str = "none"
    shuffle_features: bool = False
    tensorboard: bool = True
    seed: int = 0
    device: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_manifest",
            Path(self.dataset_manifest).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "output_dir",
            Path(self.output_dir).expanduser().resolve(),
        )
        self.validate()

    def validate(self) -> None:
        if self.architecture not in ARCHITECTURE_LAYERS:
            raise ValueError("architecture must be '1d' or '2d'")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        positive_integers = {
            "max_steps": self.max_steps,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "embedding_size": self.embedding_size,
            "mlp_hidden_size": self.mlp_hidden_size,
            "batch_size": self.batch_size,
            "validation_interval_steps": self.validation_interval_steps,
            "validation_batches": self.validation_batches,
        }
        for name, value in positive_integers.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.embedding_size % self.num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads")
        if self.max_steps < self.validation_interval_steps:
            raise ValueError("max_steps must include at least one validation interval")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and >= 0")
        if (
            not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be finite and > 0")
        if not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must be in [0, 1)")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        if not 0.0 <= self.early_stopping_ema < 1.0:
            raise ValueError("early_stopping_ema must be in [0, 1)")
        if self.early_stopping_start_step < 0:
            raise ValueError("early_stopping_start_step must be >= 0")
        if self.early_stopping_activation_drop < 0.0:
            raise ValueError("early_stopping_activation_drop must be >= 0")
        if self.target_scaling not in {"none", "context_standardize"}:
            raise ValueError(
                "target_scaling must be 'none' or 'context_standardize'"
            )

    @property
    def scheduler(self) -> str:
        return "none"

    @property
    def model_config(self) -> dict[str, int]:
        return {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "embedding_size": self.embedding_size,
            "mlp_hidden_size": self.mlp_hidden_size,
        }

    def validate_dataset(self, dataset: DatasetManifest) -> None:
        expected_train_rows = self.max_steps * self.batch_size
        expected_validation_rows = self.validation_batches * self.batch_size
        if dataset.train_rows != expected_train_rows:
            raise ValueError(
                "train split size must equal max_steps * batch_size: "
                f"expected {expected_train_rows}, got {dataset.train_rows}"
            )
        if dataset.validation_rows != expected_validation_rows:
            raise ValueError(
                "validation split size must equal validation_batches * batch_size: "
                f"expected {expected_validation_rows}, got {dataset.validation_rows}"
            )

    def checkpoint_config(self, dataset: DatasetManifest) -> CheckpointTrainingConfig:
        return CheckpointTrainingConfig(
            num_features=dataset.d,
            task_type="regression",
            num_steps=self.max_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            optimizer_name=self.optimizer,
            weight_decay=self.weight_decay,
            gradient_clip_norm=self.gradient_clip_norm,
            adam_beta2=self.adam_beta2,
            target_scaling=self.target_scaling,
            steps_per_eval=self.validation_interval_steps,
            val_num_batches=self.validation_batches,
            val_batch_size=self.batch_size,
            shuffle_features=self.shuffle_features,
            early_stopping=self.early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            early_stopping_monitor="val_loss",
            early_stopping_mode="min",
            restore_best_weights=self.restore_best_weights,
            early_stopping_start_after_steps=self.early_stopping_start_step,
            early_stopping_smoothing=self.early_stopping_ema,
            early_stopping_activation_drop=self.early_stopping_activation_drop,
            early_stopping_source="validation",
            model_config=self.model_config,
        )

    def resolved_arguments(self, dataset: DatasetManifest) -> dict[str, Any]:
        arguments = asdict(self)
        arguments["dataset_manifest"] = str(self.dataset_manifest)
        arguments["output_dir"] = str(self.output_dir)
        arguments["num_features"] = dataset.d if self.architecture == "1d" else None
        arguments["activation"] = "gelu"
        arguments["scheduler"] = self.scheduler
        arguments["dataset_manifest_sha256"] = dataset.sha256
        arguments["tensorboard_log_dir"] = (
            str(self.output_dir / "tensorboard") if self.tensorboard else None
        )
        return arguments


class _TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class ControlledBayesTrainer:
    """Own one run from manifest validation through completed artifacts."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.dataset = DatasetManifest.load(config.dataset_manifest)
        config.validate_dataset(self.dataset)

    def train(self) -> Path:
        output_dir = self.config.output_dir
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            raise FileExistsError(f"completed run already exists: {metrics_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / "checkpoint.pt"
        history_path = output_dir / "history.csv"
        arguments_path = output_dir / "args.json"
        log_path = output_dir / "train.log"
        resolved = self.config.resolved_arguments(self.dataset)
        self._write_json(resolved, arguments_path)

        run = {
            "run_name": output_dir.name,
            "train_path": str(self.dataset.train_path),
            "val_path": str(self.dataset.validation_path),
            "metadata": {
                "dataset_manifest": str(self.dataset.path),
                "dataset_manifest_sha256": self.dataset.sha256,
                "resolved_arguments": resolved,
            },
        }
        device = self._device()
        metric_logger = (
            TensorBoardMetricLogger(output_dir / "tensorboard")
            if self.config.tensorboard
            else None
        )
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
                stream = _TeeStream(sys.stdout, log_file)
                try:
                    with (
                        contextlib.redirect_stdout(stream),
                        contextlib.redirect_stderr(stream),
                    ):
                        result = train_checkpoint(
                            run=run,
                            architecture_name=self.config.architecture,
                            config=self.config.checkpoint_config(self.dataset),
                            device=device,
                            seed=self.config.seed,
                            destination=checkpoint_path,
                            metric_logger=metric_logger,
                        )
                finally:
                    stream.close()
        finally:
            if metric_logger is not None:
                metric_logger.close()

        self._write_history(result, history_path)
        metrics = self._metrics(result, checkpoint_path)
        self._write_json(metrics, metrics_path)
        print(f"completed training run: {metrics_path}")
        return metrics_path

    def _device(self) -> torch.device:
        if self.config.device:
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _metrics(
        self,
        result: CheckpointTrainingResult,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        validation_records = [
            record for record in result.history if "val_loss" in record
        ]
        if not validation_records:
            raise RuntimeError("training produced no validation measurement")
        last_record = validation_records[-1]
        best_iteration = last_record.get("best_iter")
        best_record = next(
            (
                record
                for record in validation_records
                if record["iteration"] == best_iteration
            ),
            min(validation_records, key=lambda record: record["val_loss"]),
        )
        return {
            "schema_version": 1,
            "complete": True,
            "dataset_manifest": str(self.dataset.path),
            "dataset_manifest_sha256": self.dataset.sha256,
            "architecture": self.config.architecture,
            "optimizer": self.config.optimizer,
            "scheduler": self.config.scheduler,
            "learning_rate": self.config.learning_rate,
            "training_seed": self.config.seed,
            "best_validation_loss": float(best_record["val_loss"]),
            "best_validation_step": int(best_record["iteration"]),
            "last_validation_step": int(last_record["iteration"]),
            "early_stopped": bool(last_record.get("early_stopped", False)),
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(self.config.output_dir / "history.csv"),
            "tensorboard_log_dir": (
                str(self.config.output_dir / "tensorboard")
                if self.config.tensorboard
                else None
            ),
            "resolved_arguments": self.config.resolved_arguments(self.dataset),
        }

    @staticmethod
    def _write_history(result: CheckpointTrainingResult, path: Path) -> None:
        fieldnames = list(
            dict.fromkeys(key for record in result.history for key in record)
        )
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        try:
            with temporary.open("x", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(result.history)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_json(value: Mapping[str, Any], path: Path) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(value, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=tuple(ARCHITECTURE_LAYERS),
        required=True,
    )
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="schedulefree-adamw")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--embedding-size", type=int, default=96)
    parser.add_argument("--mlp-hidden-size", type=int, default=192)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stopping-ema", type=float, default=0.0)
    parser.add_argument("--early-stopping-start-step", type=int, default=0)
    parser.add_argument("--early-stopping-activation-drop", type=float, default=0.0)
    parser.add_argument(
        "--restore-best-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument(
        "--target-scaling",
        choices=("none", "context_standardize"),
        default="none",
    )
    parser.add_argument(
        "--shuffle-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    num_layers = (
        args.num_layers
        if args.num_layers is not None
        else ARCHITECTURE_LAYERS[args.architecture]
    )
    return TrainConfig(
        dataset_manifest=args.dataset_manifest,
        output_dir=args.output_dir,
        architecture=args.architecture,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_layers=num_layers,
        num_heads=args.num_heads,
        embedding_size=args.embedding_size,
        mlp_hidden_size=args.mlp_hidden_size,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        batch_size=args.batch_size,
        validation_interval_steps=args.validation_interval,
        validation_batches=args.validation_batches,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_ema=args.early_stopping_ema,
        early_stopping_start_step=args.early_stopping_start_step,
        early_stopping_activation_drop=args.early_stopping_activation_drop,
        restore_best_weights=args.restore_best_weights,
        adam_beta2=args.adam_beta2,
        target_scaling=args.target_scaling,
        shuffle_features=args.shuffle_features,
        tensorboard=args.tensorboard,
        seed=args.seed,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics_path = ControlledBayesTrainer(config_from_args(args)).train()
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
