"""Evaluate one controlled-Bayes checkpoint on one manifest's test split."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from data.architecture_loaders import build_synthetic_loader
from data.synthetic_data.datasets import table_padding_masks
from models.architecture_registry import ArchitectureSpec, get_architecture
from training.engine import standardize_regression_targets_torch


EVALUATION_SCHEMA_VERSION = 1
RAW_RESULTS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationDataset:
    """Validated test view of one generated controlled-Bayes dataset."""

    manifest_path: Path
    manifest_sha256: str
    d: int
    k: int
    generator_batch_size: int
    test_path: Path
    test_rows: int
    test_sha256: str | None
    max_queries: int

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationDataset":
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
        d = cls._positive_integer(arguments.get("d"), "d")
        k = cls._positive_integer(arguments.get("k"), "k")
        generator_batch_size = cls._positive_integer(
            arguments.get("generator_batch_size"),
            "generator_batch_size",
        )
        if k > d:
            raise ValueError(f"dataset manifest has k={k} greater than d={d}")

        splits = payload.get("splits")
        if not isinstance(splits, Mapping):
            raise ValueError("dataset manifest has no splits object")
        split = splits.get("test")
        if not isinstance(split, Mapping):
            raise ValueError("dataset manifest has no 'test' split")
        raw_path = split.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("dataset manifest 'test' split has no path")
        test_path = Path(raw_path).expanduser()
        if not test_path.is_absolute():
            test_path = manifest_path.parent / test_path
        test_path = test_path.resolve()
        if not test_path.is_file():
            raise ValueError(f"dataset 'test' split does not exist: {test_path}")

        test_rows = cls._positive_integer(split.get("rows"), "test rows")
        test_sha256 = split.get("sha256")
        if test_sha256 is not None and (
            not isinstance(test_sha256, str) or not test_sha256
        ):
            raise ValueError("dataset manifest 'test' split has invalid sha256")
        max_queries = cls._inspect_test_h5(test_path, expected_rows=test_rows)
        return cls(
            manifest_path=manifest_path,
            manifest_sha256=hashlib.sha256(content).hexdigest(),
            d=d,
            k=k,
            generator_batch_size=generator_batch_size,
            test_path=test_path,
            test_rows=test_rows,
            test_sha256=test_sha256,
            max_queries=max_queries,
        )

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"dataset manifest has invalid {name}={value!r}")
        return value

    @staticmethod
    def _inspect_test_h5(path: Path, *, expected_rows: int) -> int:
        required = ("X", "y", "task_id", "single_eval_pos", "num_datapoints")
        try:
            with h5py.File(path, "r") as source:
                missing = [name for name in required if name not in source]
                if missing:
                    raise ValueError(
                        f"dataset 'test' split is missing datasets: {missing}"
                    )
                if source["X"].shape[0] != expected_rows:
                    raise ValueError(
                        "dataset manifest test rows do not match test H5: "
                        f"{expected_rows} != {source['X'].shape[0]}"
                    )
                for name in required[1:]:
                    if source[name].shape[0] != expected_rows:
                        raise ValueError(
                            f"test H5 dataset {name!r} has "
                            f"{source[name].shape[0]} rows; expected {expected_rows}"
                        )
                n_context = np.asarray(source["single_eval_pos"][:], dtype=np.int64)
                n_datapoints = np.asarray(
                    source["num_datapoints"][:], dtype=np.int64
                )
                if np.any(n_context < 1) or np.any(
                    n_datapoints > source["y"].shape[1]
                ):
                    raise ValueError("test H5 contains invalid context or sequence lengths")
                n_queries = n_datapoints - n_context
                if np.any(n_queries < 1):
                    raise ValueError("every test task must contain at least one query")
                return int(n_queries.max())
        except OSError as error:
            raise ValueError(f"cannot inspect test H5 {path}: {error}") from error


@dataclass(frozen=True)
class EvaluationConfig:
    """All explicit inputs for one checkpoint evaluation."""

    dataset_manifest: Path
    checkpoint: Path
    output_dir: Path
    batch_size: int = 32
    device: str = ""

    def __post_init__(self) -> None:
        for name in ("dataset_manifest", "checkpoint", "output_dir"):
            value = Path(getattr(self, name)).expanduser().resolve()
            object.__setattr__(self, name, value)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not self.checkpoint.is_file():
            raise ValueError(f"checkpoint does not exist: {self.checkpoint}")


@dataclass
class RegressionMetricAccumulator:
    """Stream exact regression metrics without retaining all predictions."""

    count: int = 0
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    target_sum: float = 0.0
    target_squared_sum: float = 0.0

    def update(self, targets: torch.Tensor, predictions: torch.Tensor) -> None:
        targets = targets.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        predictions = (
            predictions.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        )
        if targets.shape != predictions.shape:
            raise ValueError(
                "target/prediction shapes differ: "
                f"{tuple(targets.shape)} != {tuple(predictions.shape)}"
            )
        if targets.numel() == 0:
            return
        if not bool(torch.isfinite(targets).all()):
            raise ValueError("test targets contain non-finite values")
        if not bool(torch.isfinite(predictions).all()):
            raise ValueError("model predictions contain non-finite values")

        residual = targets - predictions
        self.count += int(targets.numel())
        self.squared_error_sum += float(torch.sum(residual.square()).item())
        self.absolute_error_sum += float(torch.sum(torch.abs(residual)).item())
        self.target_sum += float(torch.sum(targets).item())
        self.target_squared_sum += float(torch.sum(targets.square()).item())

    def metrics(self) -> dict[str, float | None]:
        if self.count < 1:
            raise RuntimeError("evaluation produced no valid query predictions")
        mse = self.squared_error_sum / self.count
        target_ss = self.target_squared_sum - self.target_sum**2 / self.count
        r2 = None if target_ss <= 0.0 else 1.0 - self.squared_error_sum / target_ss
        return {
            "mse": float(mse),
            "rmse": float(math.sqrt(mse)),
            "mae": float(self.absolute_error_sum / self.count),
            "r2": None if r2 is None else float(r2),
        }


class RawEvaluationWriter:
    """Stream one fixed-array row per evaluated task to an HDF5 artifact."""

    def __init__(
        self,
        path: Path,
        *,
        num_tasks: int,
        max_queries: int,
        batch_size: int,
        attributes: Mapping[str, Any],
    ) -> None:
        self.path = path
        self.num_tasks = num_tasks
        self.max_queries = max_queries
        self._next_row = 0
        chunk_rows = min(batch_size, num_tasks)
        self._file = h5py.File(path, "x")
        for name, value in attributes.items():
            if value is not None:
                self._file.attrs[name] = value
        self._file.attrs["schema_version"] = RAW_RESULTS_SCHEMA_VERSION
        self._datasets = {
            "task_id": self._file.create_dataset(
                "task_id", shape=(num_tasks,), dtype=np.int64, chunks=(chunk_rows,)
            ),
            "n_context": self._file.create_dataset(
                "n_context", shape=(num_tasks,), dtype=np.int32, chunks=(chunk_rows,)
            ),
            "n_queries": self._file.create_dataset(
                "n_queries", shape=(num_tasks,), dtype=np.int32, chunks=(chunk_rows,)
            ),
            "y_true": self._file.create_dataset(
                "y_true",
                shape=(num_tasks, max_queries),
                dtype=np.float32,
                chunks=(chunk_rows, max_queries),
                fillvalue=np.nan,
            ),
            "y_pred": self._file.create_dataset(
                "y_pred",
                shape=(num_tasks, max_queries),
                dtype=np.float32,
                chunks=(chunk_rows, max_queries),
                fillvalue=np.nan,
            ),
        }
        for name in ("mse", "rmse", "mae", "r2"):
            self._datasets[name] = self._file.create_dataset(
                name,
                shape=(num_tasks,),
                dtype=np.float64,
                chunks=(chunk_rows,),
                fillvalue=np.nan,
            )

    def __enter__(self) -> "RawEvaluationWriter":
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.close()
        elif self._file is not None:
            self._file.close()
            self._file = None

    def write_batch(
        self,
        *,
        task_ids: np.ndarray,
        n_context: np.ndarray,
        n_queries: np.ndarray,
        targets: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        if self._file is None:
            raise RuntimeError("raw evaluation writer is already closed")
        task_ids = np.asarray(task_ids, dtype=np.int64)
        n_context = np.asarray(n_context, dtype=np.int32)
        n_queries = np.asarray(n_queries, dtype=np.int32)
        batch_size = len(task_ids)
        stop = self._next_row + batch_size
        if stop > self.num_tasks:
            raise RuntimeError("raw evaluation contains more tasks than expected")
        if n_context.shape != (batch_size,) or n_queries.shape != (batch_size,):
            raise ValueError("task metadata does not match the evaluation batch size")
        if np.any(n_queries < 1) or np.any(n_queries > self.max_queries):
            raise ValueError("evaluation batch contains an invalid query count")

        y_true = targets.detach().to(device="cpu", dtype=torch.float32).numpy()
        y_pred = predictions.detach().to(device="cpu", dtype=torch.float32).numpy()
        if y_true.shape != y_pred.shape or y_true.ndim != 2:
            raise ValueError("raw targets and predictions must be matching 2D arrays")
        if y_true.shape[0] != batch_size or np.any(n_queries > y_true.shape[1]):
            raise ValueError("query counts do not match raw prediction shapes")

        padded_true = np.full(
            (batch_size, self.max_queries), np.nan, dtype=np.float32
        )
        padded_pred = np.full_like(padded_true, np.nan)
        mse = np.empty(batch_size, dtype=np.float64)
        rmse = np.empty(batch_size, dtype=np.float64)
        mae = np.empty(batch_size, dtype=np.float64)
        r2 = np.full(batch_size, np.nan, dtype=np.float64)
        for row, query_count in enumerate(n_queries):
            count = int(query_count)
            row_true = y_true[row, :count]
            row_pred = y_pred[row, :count]
            if not np.isfinite(row_true).all() or not np.isfinite(row_pred).all():
                raise ValueError(
                    "test targets or model predictions contain non-finite values"
                )
            padded_true[row, :count] = row_true
            padded_pred[row, :count] = row_pred
            true64 = row_true.astype(np.float64)
            residual = true64 - row_pred.astype(np.float64)
            mse[row] = float(np.mean(np.square(residual)))
            rmse[row] = math.sqrt(mse[row])
            mae[row] = float(np.mean(np.abs(residual)))
            target_ss = float(np.sum(np.square(true64 - np.mean(true64))))
            if target_ss > 0.0:
                r2[row] = 1.0 - float(np.sum(np.square(residual))) / target_ss

        rows = slice(self._next_row, stop)
        self._datasets["task_id"][rows] = task_ids
        self._datasets["n_context"][rows] = n_context
        self._datasets["n_queries"][rows] = n_queries
        self._datasets["y_true"][rows] = padded_true
        self._datasets["y_pred"][rows] = padded_pred
        self._datasets["mse"][rows] = mse
        self._datasets["rmse"][rows] = rmse
        self._datasets["mae"][rows] = mae
        self._datasets["r2"][rows] = r2
        self._next_row = stop

    def close(self) -> None:
        if self._file is None:
            return
        if self._next_row != self.num_tasks:
            self._file.close()
            self._file = None
            raise RuntimeError(
                f"wrote {self._next_row} raw tasks, expected {self.num_tasks}"
            )
        self._file.flush()
        self._file.close()
        self._file = None


def predict_regression_batch(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    target_scaling: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict a regression batch and restore context-standardized y units."""
    split_index = int(batch["train_test_split_index"])
    x = batch["x"].to(device=device, dtype=torch.float32)
    raw_y = batch["y"].to(device=device, dtype=torch.float32)
    if target_scaling == "context_standardize":
        model_y, target_mean, target_std = standardize_regression_targets_torch(
            raw_y,
            split_index,
        )
    else:
        model_y = raw_y
        target_mean = target_std = None

    padding_mask, feature_padding_mask = table_padding_masks(batch, device=device)
    forward_kwargs: dict[str, torch.Tensor] = {}
    if padding_mask is not None:
        forward_kwargs["padding_mask"] = padding_mask
    if feature_padding_mask is not None:
        forward_kwargs["feature_padding_mask"] = feature_padding_mask
    output = model(
        (x, model_y[:, :split_index]),
        train_test_split_index=split_index,
        **forward_kwargs,
    )
    if output.ndim != 3 or output.shape[-1] != 1:
        raise RuntimeError(
            "regression model output must have shape (batch, query, 1), got "
            f"{tuple(output.shape)}"
        )
    predictions = output.squeeze(-1)
    if target_mean is not None and target_std is not None:
        predictions = predictions * target_std + target_mean
    targets = raw_y[:, split_index:]
    if predictions.shape != targets.shape:
        raise RuntimeError(
            "model output does not match query targets: "
            f"{tuple(predictions.shape)} != {tuple(targets.shape)}"
        )
    return targets, predictions


class CheckpointEvaluator:
    """Reconstruct a checkpoint and evaluate its complete test split."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.dataset = EvaluationDataset.load(config.dataset_manifest)
        if config.batch_size != self.dataset.generator_batch_size:
            raise ValueError(
                "batch_size must equal the dataset generator_batch_size so every "
                "batch has one context boundary: "
                f"expected {self.dataset.generator_batch_size}, "
                f"got {config.batch_size}"
            )
        if self.dataset.test_rows % config.batch_size != 0:
            raise ValueError(
                "test rows must be divisible by batch_size so the complete split is "
                f"evaluated: rows={self.dataset.test_rows}, "
                f"batch_size={config.batch_size}"
            )

    def evaluate(self) -> Path:
        if self.config.output_dir.exists():
            raise FileExistsError(
                f"evaluation output already exists: {self.config.output_dir}"
            )
        checkpoint = self._load_checkpoint()
        architecture, model = self._build_model(checkpoint)
        device = self._device()
        model.to(device)
        model.eval()

        target_scaling = self._target_scaling(checkpoint)
        checkpoint_sha256 = self._sha256(self.config.checkpoint)
        loader = build_synthetic_loader(
            architecture,
            self.dataset.test_path,
            num_steps=self.dataset.test_rows // self.config.batch_size,
            batch_size=self.config.batch_size,
            num_features=self.dataset.d,
            request_feature_shuffle=False,
            device=device,
        )
        accumulator = RegressionMetricAccumulator()
        task_count = 0
        self.config.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.config.output_dir.name}.",
                suffix=".partial",
                dir=self.config.output_dir.parent,
            )
        )
        raw_path = temporary_dir / "raw_results.h5"
        try:
            with (
                h5py.File(self.dataset.test_path, "r") as source,
                RawEvaluationWriter(
                    raw_path,
                    num_tasks=self.dataset.test_rows,
                    max_queries=self.dataset.max_queries,
                    batch_size=self.config.batch_size,
                    attributes=self._raw_attributes(
                        checkpoint,
                        checkpoint_sha256=checkpoint_sha256,
                    ),
                ) as raw_writer,
                torch.inference_mode(),
            ):
                for batch in loader:
                    targets, predictions = self._predict_batch(
                        model,
                        batch,
                        device=device,
                        target_scaling=target_scaling,
                    )
                    batch_size = int(targets.shape[0])
                    stop = task_count + batch_size
                    task_ids = np.asarray(
                        source["task_id"][task_count:stop], dtype=np.int64
                    )
                    n_context = np.asarray(
                        source["single_eval_pos"][task_count:stop], dtype=np.int32
                    )
                    n_datapoints = np.asarray(
                        source["num_datapoints"][task_count:stop], dtype=np.int32
                    )
                    self._validate_batch_metadata(
                        batch,
                        n_context=n_context,
                        n_datapoints=n_datapoints,
                    )
                    n_queries = n_datapoints - n_context
                    valid = torch.arange(
                        targets.shape[1], device=targets.device
                    ).unsqueeze(0) < torch.as_tensor(
                        n_queries, device=targets.device
                    ).unsqueeze(1)
                    accumulator.update(targets[valid], predictions[valid])
                    raw_writer.write_batch(
                        task_ids=task_ids,
                        n_context=n_context,
                        n_queries=n_queries,
                        targets=targets,
                        predictions=predictions,
                    )
                    task_count = stop

            if task_count != self.dataset.test_rows:
                raise RuntimeError(
                    f"evaluated {task_count} tasks, expected {self.dataset.test_rows}"
                )
            raw_sha256 = self._sha256(raw_path)
            payload = self._result(
                checkpoint,
                accumulator,
                task_count=task_count,
                target_scaling=target_scaling,
                checkpoint_sha256=checkpoint_sha256,
                raw_results_sha256=raw_sha256,
            )
            self._write_json(payload, temporary_dir / "metrics.json")
            os.rename(temporary_dir, self.config.output_dir)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

        metrics_path = self.config.output_dir / "metrics.json"
        print(f"completed evaluation: {metrics_path}")
        return metrics_path

    @staticmethod
    def _validate_batch_metadata(
        batch: Mapping[str, Any],
        *,
        n_context: np.ndarray,
        n_datapoints: np.ndarray,
    ) -> None:
        batch_size = int(batch["y"].shape[0])
        if n_context.shape != (batch_size,) or n_datapoints.shape != (batch_size,):
            raise RuntimeError("test H5 ended before the evaluation loader")
        split_index = int(batch["train_test_split_index"])
        if not np.all(n_context == split_index):
            raise RuntimeError("evaluation loader and test H5 context boundaries differ")
        loaded_lengths = np.asarray(
            batch["num_datapoints"].cpu(), dtype=np.int32
        )
        if not np.array_equal(loaded_lengths, n_datapoints):
            raise RuntimeError("evaluation loader and test H5 sequence lengths differ")

    def _load_checkpoint(self) -> Mapping[str, Any]:
        try:
            checkpoint = torch.load(
                self.config.checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            EOFError,
            pickle.UnpicklingError,
        ) as error:
            raise ValueError(
                f"cannot load checkpoint {self.config.checkpoint}: {error}"
            ) from error
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must contain a mapping")
        if checkpoint.get("task_type", "regression") != "regression":
            raise ValueError("only regression checkpoints can be evaluated")
        return checkpoint

    def _build_model(
        self,
        checkpoint: Mapping[str, Any],
    ) -> tuple[ArchitectureSpec, torch.nn.Module]:
        architecture_name = checkpoint.get("arch")
        if not isinstance(architecture_name, str) or not architecture_name:
            raise ValueError("checkpoint has no architecture name")
        architecture = get_architecture(architecture_name)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint has no model_state_dict/state_dict mapping")
        built = architecture.build_from_checkpoint(checkpoint, num_outputs=1)
        if architecture.input_policy == "fixed_width":
            model_width = int(built.config["num_features"])
            if model_width != self.dataset.d:
                raise ValueError(
                    f"checkpoint expects {model_width} features, "
                    f"but evaluation data has d={self.dataset.d}"
                )
        built.model.load_state_dict(state)
        return architecture, built.model

    @staticmethod
    def _target_scaling(checkpoint: Mapping[str, Any]) -> str:
        training_config = checkpoint.get("training_config")
        value = (
            training_config.get("target_scaling", "none")
            if isinstance(training_config, Mapping)
            else "none"
        )
        if value not in {"none", "context_standardize"}:
            raise ValueError(f"checkpoint has invalid target_scaling={value!r}")
        return str(value)

    # Backward-compatible name for callers of the original evaluator.
    _predict_batch = staticmethod(predict_regression_batch)

    def _raw_attributes(
        self,
        checkpoint: Mapping[str, Any],
        *,
        checkpoint_sha256: str,
    ) -> dict[str, Any]:
        training_config = checkpoint.get("training_config")
        training_config = (
            dict(training_config) if isinstance(training_config, Mapping) else {}
        )
        run_metadata = checkpoint.get("run_metadata")
        run_metadata = dict(run_metadata) if isinstance(run_metadata, Mapping) else {}
        resolved = run_metadata.get("resolved_arguments")
        resolved = dict(resolved) if isinstance(resolved, Mapping) else {}
        return {
            "checkpoint_path": str(self.config.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_manifest": str(self.dataset.manifest_path),
            "dataset_manifest_sha256": self.dataset.manifest_sha256,
            "test_path": str(self.dataset.test_path),
            "test_split_sha256": self.dataset.test_sha256,
            "d": self.dataset.d,
            "k": self.dataset.k,
            "architecture": checkpoint.get("arch"),
            "optimizer": training_config.get(
                "optimizer_name", resolved.get("optimizer")
            ),
            "learning_rate": training_config.get(
                "learning_rate", resolved.get("learning_rate")
            ),
            "training_seed": checkpoint.get("seed", resolved.get("seed")),
        }

    def _result(
        self,
        checkpoint: Mapping[str, Any],
        accumulator: RegressionMetricAccumulator,
        *,
        task_count: int,
        target_scaling: str,
        checkpoint_sha256: str,
        raw_results_sha256: str,
    ) -> dict[str, Any]:
        training_config = checkpoint.get("training_config")
        training_config = (
            dict(training_config) if isinstance(training_config, Mapping) else {}
        )
        run_metadata = checkpoint.get("run_metadata")
        run_metadata = dict(run_metadata) if isinstance(run_metadata, Mapping) else {}
        resolved = run_metadata.get("resolved_arguments")
        resolved = dict(resolved) if isinstance(resolved, Mapping) else {}
        model_config = checkpoint.get("model_config")
        model_config = dict(model_config) if isinstance(model_config, Mapping) else {}
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "complete": True,
            "checkpoint_path": str(self.config.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_manifest": str(self.dataset.manifest_path),
            "dataset_manifest_sha256": self.dataset.manifest_sha256,
            "split": "test",
            "test_path": str(self.dataset.test_path),
            "test_split_sha256": self.dataset.test_sha256,
            "d": self.dataset.d,
            "k": self.dataset.k,
            "architecture": checkpoint.get("arch"),
            "model_config": model_config,
            "optimizer": training_config.get(
                "optimizer_name",
                resolved.get("optimizer"),
            ),
            "scheduler": resolved.get("scheduler"),
            "learning_rate": training_config.get(
                "learning_rate",
                resolved.get("learning_rate"),
            ),
            "training_seed": checkpoint.get("seed", resolved.get("seed")),
            "target_scaling": target_scaling,
            "training_config": training_config,
            "training_dataset_manifest": run_metadata.get("dataset_manifest"),
            "training_dataset_manifest_sha256": run_metadata.get(
                "dataset_manifest_sha256"
            ),
            "batch_size": self.config.batch_size,
            "generator_batch_size": self.dataset.generator_batch_size,
            "num_tasks": task_count,
            "num_predictions": accumulator.count,
            "raw_results_path": str(self.config.output_dir / "raw_results.h5"),
            "raw_results_sha256": raw_results_sha256,
            "raw_results_schema_version": RAW_RESULTS_SCHEMA_VERSION,
            "metrics": accumulator.metrics(),
        }

    def _device(self) -> torch.device:
        if self.config.device:
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_json(value: Mapping[str, Any], path: Path) -> None:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, allow_nan=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="")
    return parser


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        dataset_manifest=args.dataset_manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    output = CheckpointEvaluator(config).evaluate()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
