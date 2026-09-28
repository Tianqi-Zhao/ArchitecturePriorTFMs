"""Concrete prediction runner and cache schemas for Part I.1."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping
from uuid import uuid4

import numpy as np

from experiment_core.adapters import PredictiveModel, PretrainedModelFactory
from experiment_core.pretrained import PRETRAINED_MODEL_SPECS
from experiment_core.runtime import guarded_csv_outputs
from experiments_arch.experiment_0_pretrained.config import ExperimentConfig
from experiments_arch.experiment_0_pretrained.data.base import PreparedDataset
from experiments_arch.experiment_0_pretrained.data.store import (
    ConditionRef,
    HDF5DatasetStore,
    SKIP_COLUMNS,
)


PREDICTION_ALGORITHM_VERSION = "part-i-1-prediction-v2"

# Cache-leaf rows deliberately contain no run ids. A leaf is content-addressed
# and can therefore be referenced after datasets are added to a later run.
PREDICTION_COLUMNS = (
    "schema_version", "dataset_id", "dataset_name", "source", "condition_id",
    "pair_id", "model_id", "model_version", "requested_irrelevant_ratio",
    "actual_irrelevant_ratio", "ratio_denominator", "rounding", "d_base",
    "d_irrelevant", "d_total", "transform", "transform_repeat",
    "transform_seed", "column_layout", "n_context", "n_query",
    "query_row_id", "y_true", "y_pred", "prediction_seconds",
)
PREDICTION_INDEX_COLUMNS = (
    "schema_version", "data_run_id", "prediction_run_id", "model_cache_key",
    "data_cache_key", "dataset_id", "dataset_name", "source", "condition_id",
    "pair_id", "model_id", "model_version", "requested_irrelevant_ratio",
    "actual_irrelevant_ratio", "ratio_denominator", "rounding", "d_base",
    "d_irrelevant", "d_total", "transform", "transform_repeat",
    "transform_seed", "column_layout", "n_context", "n_query",
    "predictions_path", "cache_status",
)
PREDICTION_FAILURE_COLUMNS = (
    "dataset_id", "dataset_name", "source", "condition_id", "pair_id",
    "model_id", "requested_irrelevant_ratio", "actual_irrelevant_ratio",
    "transform_repeat", "error_type", "error",
)


@dataclass(frozen=True)
class PredictionRow:
    schema_version: int
    dataset_id: str
    dataset_name: str
    source: str
    condition_id: str
    pair_id: str
    model_id: str
    model_version: str
    requested_irrelevant_ratio: float
    actual_irrelevant_ratio: float
    ratio_denominator: int
    rounding: str
    d_base: int
    d_irrelevant: int
    d_total: int
    transform: str
    transform_repeat: int
    transform_seed: int
    column_layout: Literal["shuffle"]
    n_context: int
    n_query: int
    query_row_id: str | int
    y_true: float
    y_pred: float
    prediction_seconds: float

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionResult:
    prediction_run_id: str
    output_dir: Path
    num_files: int
    num_rows: int
    num_failures: int


@dataclass(frozen=True)
class _ModelFingerprint:
    model_cache_key: str
    model_version: str
    checkpoint_name: str
    checkpoint_sha256: str
    payload: Mapping[str, object]


class PredictionRunner(ABC):
    """Stage contract for prepared index -> per-dataset/model raw CSVs."""

    @abstractmethod
    def run(
        self,
        *,
        config: ExperimentConfig,
        data_index: Path,
        output_dir: Path,
        device: str,
        model_ids: tuple[str, ...] | None = None,
    ) -> PredictionResult:
        """Run configured models, optionally restricted to ``model_ids``."""

        raise NotImplementedError


def _canonical_hash(value: object, *, prefix: str) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(
    rows: list[Mapping[str, object]],
    path: Path,
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _model_fingerprint(model: PredictiveModel, device: str) -> _ModelFingerprint:
    descriptor = model.descriptor
    options = dict(descriptor.options)
    path_value = options.pop("model_path", "")
    checkpoint = Path(str(path_value)).resolve() if path_value else None
    checkpoint_name = checkpoint.name if checkpoint is not None else "none"
    checkpoint_sha256 = (
        _file_sha256(checkpoint)
        if checkpoint is not None and checkpoint.is_file()
        else "missing"
    )
    spec = PRETRAINED_MODEL_SPECS.get(descriptor.model_id)
    package = spec.package if spec is not None else "custom"
    package_version = _package_version(package) if spec is not None else "custom"
    model_version = descriptor.model_version
    if model_version == "unprepared":
        model_version = f"{package}=={package_version};checkpoint={checkpoint_name}"
    payload = {
        "algorithm": PREDICTION_ALGORITHM_VERSION,
        "model_id": descriptor.model_id,
        "model_group": descriptor.model_group,
        "model_origin": descriptor.model_origin,
        "model_version": model_version,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "inference_options": options,
        "device": str(device),
    }
    return _ModelFingerprint(
        model_cache_key=_canonical_hash(payload, prefix="model"),
        model_version=model_version,
        checkpoint_name=checkpoint_name,
        checkpoint_sha256=checkpoint_sha256,
        payload=payload,
    )


def _selected_configs(config: ExperimentConfig, model_ids: tuple[str, ...] | None):
    by_id = {item.model_id: item for item in config.models}
    if model_ids is None:
        selected = tuple(by_id)
    else:
        selected = tuple(str(model_id) for model_id in model_ids)
        if not selected:
            raise ValueError("model_ids must not be explicitly empty")
        if len(selected) != len(set(selected)):
            raise ValueError(f"duplicate model ids: {selected}")
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise ValueError(f"models are not configured: {unknown}")
    return tuple(by_id[model_id] for model_id in selected)


def _row_id(value: object) -> str | int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _prediction_rows(
    reference: ConditionRef,
    condition: PreparedDataset,
    *,
    model_id: str,
    model_version: str,
    predictions: np.ndarray,
    prediction_seconds: float,
) -> list[dict[str, object]]:
    return [
        PredictionRow(
            schema_version=reference.schema_version,
            dataset_id=reference.dataset_id,
            dataset_name=reference.dataset_name,
            source=reference.source,
            condition_id=reference.condition_id,
            pair_id=reference.pair_id,
            model_id=model_id,
            model_version=model_version,
            requested_irrelevant_ratio=reference.requested_irrelevant_ratio,
            actual_irrelevant_ratio=reference.actual_irrelevant_ratio,
            ratio_denominator=reference.ratio_denominator,
            rounding=reference.rounding,
            d_base=reference.d_base,
            d_irrelevant=reference.d_irrelevant,
            d_total=reference.d_total,
            transform=reference.transform,
            transform_repeat=reference.transform_repeat,
            transform_seed=reference.transform_seed,
            column_layout=reference.column_layout,
            n_context=reference.n_context,
            n_query=reference.n_query,
            query_row_id=_row_id(query_id),
            y_true=float(y_true),
            y_pred=float(y_pred),
            prediction_seconds=float(prediction_seconds),
        ).to_row()
        for query_id, y_true, y_pred in zip(
            condition.query_row_id, condition.y_query, predictions
        )
    ]


def _valid_prediction_file(
    path: Path,
    reference: ConditionRef,
    condition: PreparedDataset,
    *,
    model_id: str,
    model_version: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                return False
            rows = list(reader)
        if len(rows) != reference.n_query:
            return False
        expected_ids = [str(_row_id(value)) for value in condition.query_row_id]
        if [row["query_row_id"] for row in rows] != expected_ids:
            return False
        for row, y_true in zip(rows, condition.y_query):
            if (
                row["dataset_id"] != reference.dataset_id
                or row["condition_id"] != reference.condition_id
                or row["model_id"] != model_id
                or row["model_version"] != model_version
            ):
                return False
            if not math.isclose(float(row["y_true"]), float(y_true)):
                return False
            if not math.isfinite(float(row["y_pred"])):
                return False
            if not math.isfinite(float(row["prediction_seconds"])):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _index_row(
    reference: ConditionRef,
    *,
    prediction_run_id: str,
    fingerprint: _ModelFingerprint,
    model_id: str,
    predictions_path: Path,
    stage_root: Path,
    cache_status: str,
) -> dict[str, object]:
    return {
        "schema_version": reference.schema_version,
        "data_run_id": reference.data_run_id,
        "prediction_run_id": prediction_run_id,
        "model_cache_key": fingerprint.model_cache_key,
        "data_cache_key": reference.data_cache_key,
        "dataset_id": reference.dataset_id,
        "dataset_name": reference.dataset_name,
        "source": reference.source,
        "condition_id": reference.condition_id,
        "pair_id": reference.pair_id,
        "model_id": model_id,
        "model_version": fingerprint.model_version,
        "requested_irrelevant_ratio": reference.requested_irrelevant_ratio,
        "actual_irrelevant_ratio": reference.actual_irrelevant_ratio,
        "ratio_denominator": reference.ratio_denominator,
        "rounding": reference.rounding,
        "d_base": reference.d_base,
        "d_irrelevant": reference.d_irrelevant,
        "d_total": reference.d_total,
        "transform": reference.transform,
        "transform_repeat": reference.transform_repeat,
        "transform_seed": reference.transform_seed,
        "column_layout": reference.column_layout,
        "n_context": reference.n_context,
        "n_query": reference.n_query,
        "predictions_path": predictions_path.relative_to(stage_root).as_posix(),
        "cache_status": cache_status,
    }


def _failure_row(
    reference: ConditionRef,
    model_id: str,
    error_type: str,
    error: str,
) -> dict[str, object]:
    return {
        "dataset_id": reference.dataset_id,
        "dataset_name": reference.dataset_name,
        "source": reference.source,
        "condition_id": reference.condition_id,
        "pair_id": reference.pair_id,
        "model_id": model_id,
        "requested_irrelevant_ratio": reference.requested_irrelevant_ratio,
        "actual_irrelevant_ratio": reference.actual_irrelevant_ratio,
        "transform_repeat": reference.transform_repeat,
        "error_type": error_type,
        "error": error,
    }


def _preparation_failures(
    data_index: Path,
    config: ExperimentConfig,
    model_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    path = Path(data_index).with_name("skips.csv")
    if not path.is_file():
        return []
    source_ratios = {
        source.source.value: source.ratios for source in config.sources
    }
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SKIP_COLUMNS:
            raise ValueError("prepared-data skips columns do not match the schema")
        for skipped in reader:
            source = skipped["source"]
            for model_id in model_ids:
                for ratio in source_ratios.get(source, (0.0,)):
                    rows.append({
                        "dataset_id": skipped["dataset_id"],
                        "dataset_name": skipped["dataset_name"],
                        "source": source,
                        "condition_id": "prepare-failed",
                        "pair_id": "",
                        "model_id": model_id,
                        "requested_irrelevant_ratio": ratio,
                        "actual_irrelevant_ratio": "",
                        "transform_repeat": "",
                        "error_type": "DataPreparationError",
                        "error": skipped["reason"],
                    })
    return rows


class DefaultPredictionRunner(PredictionRunner):
    """Run selected pretrained models against immutable prepared leaves."""

    def __init__(
        self,
        model_factory: PretrainedModelFactory,
        store: HDF5DatasetStore | None = None,
    ) -> None:
        self.model_factory = model_factory
        self.store = store or HDF5DatasetStore()

    def run(
        self,
        *,
        config: ExperimentConfig,
        data_index: Path,
        output_dir: Path,
        device: str,
        model_ids: tuple[str, ...] | None = None,
    ) -> PredictionResult:
        references = tuple(self.store.read_index(Path(data_index)))
        if not references:
            raise ValueError("prepared-data index contains no conditions")
        data_run_ids = {reference.data_run_id for reference in references}
        if len(data_run_ids) != 1:
            raise ValueError("prepared-data index must contain exactly one data_run_id")
        data_run_id = next(iter(data_run_ids))
        selected = _selected_configs(config, model_ids)
        plans = []
        for model_config in selected:
            model = self.model_factory.build(model_config)
            plans.append((model_config, model, _model_fingerprint(model, str(device))))

        run_payload = {
            "algorithm": PREDICTION_ALGORITHM_VERSION,
            "experiment_id": config.experiment_id,
            "schema_version": config.schema_version,
            "data_run_id": data_run_id,
            "model_cache_keys": [fingerprint.model_cache_key for _, _, fingerprint in plans],
        }
        prediction_run_id = _canonical_hash(run_payload, prefix="prediction")
        root = Path(output_dir or config.artifact_root).resolve()
        stage_root = root / "predictions"
        cache_root = stage_root / "cache"
        run_dir = stage_root / "runs" / prediction_run_id
        index_rows: list[dict[str, object]] = []
        failures: list[dict[str, object]] = _preparation_failures(
            Path(data_index), config, tuple(item.model_id for item in selected)
        )
        num_rows = 0
        cache_hits = 0
        cache_misses = 0

        for model_config, model, fingerprint in plans:
            model_id = model_config.model_id
            model_is_prepared = False
            model_prepare_error: Exception | None = None
            for reference in references:
                if reference.d_total > model.descriptor.max_input_features:
                    failures.append(_failure_row(
                        reference,
                        model_id,
                        "InputFeatureLimitError",
                        (
                            f"d_total={reference.d_total} exceeds max_input_features="
                            f"{model.descriptor.max_input_features}"
                        ),
                    ))
                    continue
                try:
                    condition = self.store.read_condition(reference)
                except Exception as error:
                    failures.append(_failure_row(
                        reference, model_id, type(error).__name__,
                        f"read_condition_failed:{error}",
                    ))
                    continue
                prediction_path = (
                    cache_root / fingerprint.model_cache_key
                    / reference.data_cache_key / reference.condition_id
                    / "predictions.csv"
                )
                if _valid_prediction_file(
                    prediction_path, reference, condition,
                    model_id=model_id, model_version=fingerprint.model_version,
                ):
                    index_rows.append(_index_row(
                        reference,
                        prediction_run_id=prediction_run_id,
                        fingerprint=fingerprint,
                        model_id=model_id,
                        predictions_path=prediction_path,
                        stage_root=stage_root,
                        cache_status="hit",
                    ))
                    num_rows += reference.n_query
                    cache_hits += 1
                    continue

                if model_prepare_error is not None:
                    failures.append(_failure_row(
                        reference, model_id, type(model_prepare_error).__name__,
                        f"model_prepare_failed:{model_prepare_error}",
                    ))
                    continue
                if not model_is_prepared:
                    try:
                        model.prepare(str(device))
                        if model.descriptor.model_version != fingerprint.model_version:
                            raise RuntimeError(
                                "prepared model identity changed after cache key "
                                f"construction: {fingerprint.model_version!r} -> "
                                f"{model.descriptor.model_version!r}"
                            )
                        model_is_prepared = True
                    except Exception as error:
                        model_prepare_error = error
                        failures.append(_failure_row(
                            reference, model_id, type(error).__name__,
                            f"model_prepare_failed:{error}",
                        ))
                        continue
                try:
                    started = perf_counter()
                    prediction = np.asarray(model.fit_predict(
                        condition.X_context, condition.y_context, condition.X_query,
                    ), dtype=np.float64)
                    elapsed = perf_counter() - started
                    if prediction.shape != (reference.n_query,):
                        raise RuntimeError(
                            f"prediction shape {prediction.shape} does not match "
                            f"n_query={reference.n_query}"
                        )
                    if not np.isfinite(prediction).all():
                        raise RuntimeError("prediction contains non-finite values")
                    rows = _prediction_rows(
                        reference, condition, model_id=model_id,
                        model_version=fingerprint.model_version,
                        predictions=prediction, prediction_seconds=elapsed,
                    )
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    with guarded_csv_outputs(
                        prediction_path,
                        label=f"prediction leaf {model_id}/{reference.condition_id}",
                    ) as staging:
                        staged = staging[prediction_path]
                        with staged.open("w", encoding="utf-8", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
                            writer.writeheader()
                            writer.writerows(rows)
                        if not _valid_prediction_file(
                            staged, reference, condition,
                            model_id=model_id,
                            model_version=fingerprint.model_version,
                        ):
                            raise RuntimeError("staged prediction leaf failed validation")
                    index_rows.append(_index_row(
                        reference,
                        prediction_run_id=prediction_run_id,
                        fingerprint=fingerprint,
                        model_id=model_id,
                        predictions_path=prediction_path,
                        stage_root=stage_root,
                        cache_status="miss",
                    ))
                    num_rows += len(rows)
                    cache_misses += 1
                except Exception as error:
                    failures.append(_failure_row(
                        reference, model_id, type(error).__name__, str(error)
                    ))

        index_rows.sort(key=lambda row: (
            str(row["source"]), str(row["dataset_id"]),
            int(row["transform_repeat"]), float(row["requested_irrelevant_ratio"]),
            str(row["model_id"]),
        ))
        failures.sort(key=lambda row: (
            str(row["dataset_id"]), str(row["condition_id"]), str(row["model_id"])
        ))
        _atomic_csv(index_rows, run_dir / "index.csv", PREDICTION_INDEX_COLUMNS)
        _atomic_csv(failures, run_dir / "failures.csv", PREDICTION_FAILURE_COLUMNS)
        _atomic_json({
            "experiment": asdict(config),
            "data_index": str(data_index),
            "device": str(device),
            "selected_model_ids": [item.model_id for item in selected],
            "model_fingerprints": [fingerprint.payload for _, _, fingerprint in plans],
        }, run_dir / "config.json")
        _atomic_json({
            "complete": True,
            "prediction_run_id": prediction_run_id,
            "data_run_id": data_run_id,
            "num_files": len(index_rows),
            "num_rows": num_rows,
            "num_failures": len(failures),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        }, run_dir / "manifest.json")
        return PredictionResult(
            prediction_run_id=prediction_run_id,
            output_dir=run_dir,
            num_files=len(index_rows),
            num_rows=num_rows,
            num_failures=len(failures),
        )


__all__ = [
    "DefaultPredictionRunner",
    "PREDICTION_ALGORITHM_VERSION",
    "PREDICTION_COLUMNS",
    "PREDICTION_FAILURE_COLUMNS",
    "PREDICTION_INDEX_COLUMNS",
    "PredictionResult",
    "PredictionRunner",
]
