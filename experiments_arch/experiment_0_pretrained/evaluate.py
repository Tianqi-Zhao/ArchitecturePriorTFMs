"""Evaluate cached Part I.1 predictions and Part I.2 functional artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from experiment_core.metrics import mae, r2_score, rmse
from experiments_arch.experiment_0_pretrained.config import ExperimentConfig
from experiments_arch.experiment_0_pretrained.predict import (
    PREDICTION_COLUMNS,
    PREDICTION_FAILURE_COLUMNS,
    PREDICTION_INDEX_COLUMNS,
)
from experiments_arch.experiment_0_pretrained.functional import (
    FEATURE_METRIC_COLUMNS,
    PART_I2_FAILURE_COLUMNS,
    PART_I2_INDEX_COLUMNS,
)


EVALUATION_ALGORITHM_VERSION = "part-i-1-evaluation-v1"
FRIEDMAN_SWEEP_EVALUATION_VERSION = "part-i-1-friedman-sweeps-v1"
EXPANDED_PART_I1_EVALUATION_VERSION = "part-i-1-expanded-friedman-v1"
PART_I2_EVALUATION_VERSION = "part-i-2-evaluation-v1"
PART_I2_FRIEDMAN_SWEEP_EVALUATION_VERSION = "part-i-2-friedman-sweeps-v1"
FLAT_EVALUATION_LAYOUT_VERSION = "part-i-flat-evaluations-v1"
PRIMARY_MODEL_A = "tabdpt_retrieval_off"
PRIMARY_MODEL_B = "tabpfn_v2"

METRIC_COLUMNS = (
    "dataset_id", "dataset_name", "source", "condition_id", "pair_id",
    "model_id", "requested_irrelevant_ratio", "actual_irrelevant_ratio",
    "d_base", "d_irrelevant", "d_total", "transform_repeat", "n_query",
    "mse", "normalized_mse", "r2", "rmse", "mae",
)
PAIRED_EFFECT_COLUMNS = (
    "dataset_id", "dataset_name", "source", "pair_id", "condition_id",
    "baseline_condition_id", "model_id", "requested_irrelevant_ratio",
    "actual_irrelevant_ratio", "transform_repeat", "risk_at_dose",
    "risk_at_baseline", "risk_increase", "mse_at_dose", "mse_at_baseline",
    "mse_increase", "r2_at_dose", "r2_at_baseline", "r2_effect",
    "r2_drop", "rmse_at_dose", "rmse_at_baseline", "rmse_increase",
)
MODEL_CONTRAST_COLUMNS = (
    "dataset_id", "source", "pair_id", "condition_id",
    "requested_irrelevant_ratio", "actual_irrelevant_ratio",
    "transform_repeat", "model_a", "model_b", "risk_increase_a",
    "risk_increase_b", "delta_risk_increase", "r2_drop_a", "r2_drop_b",
    "delta_r2_drop",
)
SUMMARY_COLUMNS = (
    "source", "model_id", "requested_irrelevant_ratio", "actual_ratio_min",
    "actual_ratio_median", "actual_ratio_max", "n_datasets",
    "n_complete_pairs", "n_transform_repeats", "risk_increase_mean",
    "risk_increase_std", "risk_increase_median", "risk_increase_q25",
    "risk_increase_q75", "r2_drop_mean", "r2_drop_std", "r2_drop_median",
    "r2_drop_q25", "r2_drop_q75", "rmse_increase_mean",
    "rmse_increase_std", "rmse_increase_median", "rmse_increase_q25",
    "rmse_increase_q75",
)
MODEL_CONTRAST_SUMMARY_COLUMNS = (
    "source", "requested_irrelevant_ratio", "actual_ratio_min",
    "actual_ratio_median", "actual_ratio_max", "model_a", "model_b",
    "n_datasets", "n_complete_pairs", "n_transform_repeats",
    "delta_risk_mean", "delta_risk_std", "delta_risk_median",
    "delta_risk_q25", "delta_risk_q75", "delta_r2_drop_mean",
    "delta_r2_drop_std", "delta_r2_drop_median", "delta_r2_drop_q25",
    "delta_r2_drop_q75",
)
COVERAGE_COLUMNS = (
    "source", "model_id", "requested_irrelevant_ratio",
    "requested_datasets", "successful_datasets", "failed_datasets",
)


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_run_id: str
    output_dir: Path
    metrics_path: Path
    paired_effects_path: Path
    model_contrasts_path: Path
    contrast_summary_path: Path
    summary_path: Path
    coverage_path: Path


FRIEDMAN_SWEEP_MODEL_SUMMARY_COLUMNS = (
    "sweep", "n_total", "n_context", "n_query", "noise_std", "model_id",
    "requested_irrelevant_ratio", "n_tasks", "n_complete_pairs",
    "n_transform_repeats", "clean_r2_mean", "clean_r2_std",
    "clean_r2_median", "clean_r2_q25", "clean_r2_q75", "r2_drop_mean",
    "r2_drop_std", "r2_drop_median", "r2_drop_q25", "r2_drop_q75",
)
FRIEDMAN_SWEEP_CONTRAST_SUMMARY_COLUMNS = (
    "sweep", "n_total", "n_context", "n_query", "noise_std",
    "requested_irrelevant_ratio", "model_a", "model_b", "n_tasks",
    "n_complete_pairs", "n_transform_repeats", "delta_r2_drop_mean",
    "delta_r2_drop_std", "delta_r2_drop_median", "delta_r2_drop_q25",
    "delta_r2_drop_q75",
)


@dataclass(frozen=True)
class FriedmanSweepEvaluationResult:
    evaluation_run_id: str
    output_dir: Path
    model_summary_path: Path
    contrast_summary_path: Path


@dataclass(frozen=True)
class ExpandedPartI1EvaluationResult:
    evaluation_run_id: str
    output_dir: Path
    paired_effects_path: Path
    model_contrasts_path: Path
    contrast_summary_path: Path
    summary_path: Path
    manifest_path: Path


class RawPredictionEvaluator(ABC):
    """Stage contract for raw CSVs -> metrics, paired effects, and summaries."""

    @abstractmethod
    def evaluate(self, predictions_dir: Path, output_dir: Path) -> EvaluationResult:
        raise NotImplementedError


def _canonical_hash(value: object, *, prefix: str) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _atomic_csv(
    rows: Iterable[Mapping[str, object]], path: Path, columns: tuple[str, ...]
) -> Path:
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
    return path


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


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"CSV columns do not match schema: {path}")
        return list(reader)


def _resolve_prediction_run(path: Path) -> tuple[Path, Path]:
    path = Path(path).resolve()
    if path.is_file():
        if path.name != "index.csv":
            raise ValueError("prediction input file must be a run index.csv")
        run_dir = path.parent
    elif (path / "index.csv").is_file():
        run_dir = path
    else:
        runs_dir = path / "runs"
        candidates = sorted(
            directory for directory in runs_dir.glob("prediction-*")
            if (directory / "index.csv").is_file()
            and (directory / "manifest.json").is_file()
        ) if runs_dir.is_dir() else []
        if len(candidates) != 1:
            raise ValueError(
                "predictions_dir must identify one prediction run directory; "
                f"found {len(candidates)} complete candidates under {path}"
            )
        run_dir = candidates[0]
    if run_dir.parent.name != "runs":
        raise ValueError("prediction run must live below predictions/runs")
    stage_root = run_dir.parent.parent.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"prediction run has no manifest: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"prediction run is incomplete: {run_dir}")
    return run_dir, stage_root


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _prediction_path(stage_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("prediction index path escapes the prediction root")

    stage_root = stage_root.resolve()
    path = (stage_root / relative).resolve()
    allowed_roots = [stage_root]
    if relative.parts[:1] == ("cache",):
        # The cache directory may intentionally be a symlink to a larger disk.
        # Permit only the resolved target of that designated cache root.
        allowed_roots.append((stage_root / "cache").resolve())
    if not any(_is_below(path, root) for root in allowed_roots):
        raise ValueError("prediction index path escapes the prediction root")
    return path


def _metric_rows(
    index_rows: Sequence[Mapping[str, str]], stage_root: Path
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for index in index_rows:
        identity = (
            index["model_cache_key"], index["data_cache_key"], index["condition_id"]
        )
        if identity in seen:
            raise ValueError(f"duplicate prediction leaf in run index: {identity}")
        seen.add(identity)
        leaf_path = _prediction_path(stage_root, index["predictions_path"])
        leaf = _read_csv(leaf_path, PREDICTION_COLUMNS)
        expected_n = int(index["n_query"])
        if len(leaf) != expected_n:
            raise ValueError(
                f"prediction leaf has {len(leaf)} rows, expected {expected_n}: {leaf_path}"
            )
        if any(
            row["dataset_id"] != index["dataset_id"]
            or row["condition_id"] != index["condition_id"]
            or row["model_id"] != index["model_id"]
            or row["model_version"] != index["model_version"]
            for row in leaf
        ):
            raise ValueError(f"prediction leaf identity disagrees with index: {leaf_path}")
        query_ids = [row["query_row_id"] for row in leaf]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError(f"prediction leaf repeats query_row_id: {leaf_path}")
        y_true = np.asarray([float(row["y_true"]) for row in leaf], dtype=np.float64)
        y_pred = np.asarray([float(row["y_pred"]) for row in leaf], dtype=np.float64)
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise ValueError(f"prediction leaf contains non-finite values: {leaf_path}")
        mse_value = float(np.mean((y_true - y_pred) ** 2))
        variance = float(np.mean((y_true - np.mean(y_true)) ** 2))
        normalized = float("nan") if variance <= 0.0 else mse_value / variance
        rows.append({
            "dataset_id": index["dataset_id"],
            "dataset_name": index["dataset_name"],
            "source": index["source"],
            "condition_id": index["condition_id"],
            "pair_id": index["pair_id"],
            "model_id": index["model_id"],
            "requested_irrelevant_ratio": float(index["requested_irrelevant_ratio"]),
            "actual_irrelevant_ratio": float(index["actual_irrelevant_ratio"]),
            "d_base": int(index["d_base"]),
            "d_irrelevant": int(index["d_irrelevant"]),
            "d_total": int(index["d_total"]),
            "transform_repeat": int(index["transform_repeat"]),
            "n_query": expected_n,
            "mse": mse_value,
            "normalized_mse": normalized,
            "r2": r2_score(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
        })
    return rows


def _paired_effects(metrics: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = {}
    for row in metrics:
        key = (
            str(row["dataset_id"]), str(row["source"]),
            str(row["pair_id"]), str(row["model_id"]),
        )
        groups.setdefault(key, []).append(row)
    effects: list[dict[str, object]] = []
    for group in groups.values():
        baselines = [
            row for row in group
            if float(row["requested_irrelevant_ratio"]) == 0.0
            and int(row["d_irrelevant"]) == 0
        ]
        if len(baselines) != 1:
            continue
        baseline = baselines[0]
        for dose in group:
            risk_dose = float(dose["normalized_mse"])
            risk_base = float(baseline["normalized_mse"])
            r2_dose = float(dose["r2"])
            r2_base = float(baseline["r2"])
            effects.append({
                "dataset_id": dose["dataset_id"],
                "dataset_name": dose["dataset_name"],
                "source": dose["source"],
                "pair_id": dose["pair_id"],
                "condition_id": dose["condition_id"],
                "baseline_condition_id": baseline["condition_id"],
                "model_id": dose["model_id"],
                "requested_irrelevant_ratio": dose["requested_irrelevant_ratio"],
                "actual_irrelevant_ratio": dose["actual_irrelevant_ratio"],
                "transform_repeat": dose["transform_repeat"],
                "risk_at_dose": risk_dose,
                "risk_at_baseline": risk_base,
                "risk_increase": risk_dose - risk_base,
                "mse_at_dose": dose["mse"],
                "mse_at_baseline": baseline["mse"],
                "mse_increase": float(dose["mse"]) - float(baseline["mse"]),
                "r2_at_dose": r2_dose,
                "r2_at_baseline": r2_base,
                "r2_effect": r2_dose - r2_base,
                "r2_drop": r2_base - r2_dose,
                "rmse_at_dose": dose["rmse"],
                "rmse_at_baseline": baseline["rmse"],
                "rmse_increase": float(dose["rmse"]) - float(baseline["rmse"]),
            })
    return effects


def _model_contrasts(effects: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], dict[str, Mapping[str, object]]] = {}
    for row in effects:
        key = (
            str(row["dataset_id"]), str(row["source"]),
            str(row["pair_id"]), str(row["condition_id"]),
        )
        groups.setdefault(key, {})[str(row["model_id"])] = row
    contrasts = []
    for models in groups.values():
        if PRIMARY_MODEL_A not in models or PRIMARY_MODEL_B not in models:
            continue
        a, b = models[PRIMARY_MODEL_A], models[PRIMARY_MODEL_B]
        contrasts.append({
            "dataset_id": a["dataset_id"],
            "source": a["source"],
            "pair_id": a["pair_id"],
            "condition_id": a["condition_id"],
            "requested_irrelevant_ratio": a["requested_irrelevant_ratio"],
            "actual_irrelevant_ratio": a["actual_irrelevant_ratio"],
            "transform_repeat": a["transform_repeat"],
            "model_a": PRIMARY_MODEL_A,
            "model_b": PRIMARY_MODEL_B,
            "risk_increase_a": a["risk_increase"],
            "risk_increase_b": b["risk_increase"],
            "delta_risk_increase": float(a["risk_increase"]) - float(b["risk_increase"]),
            "r2_drop_a": a["r2_drop"],
            "r2_drop_b": b["r2_drop"],
            "delta_r2_drop": float(a["r2_drop"]) - float(b["r2_drop"]),
        })
    return contrasts


def _finite(values: Iterable[object]) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def _stats(values: Iterable[object]) -> dict[str, float]:
    array = _finite(values)
    if len(array) == 0:
        return {key: float("nan") for key in ("mean", "std", "median", "q25", "q75")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _dataset_averages(
    rows: Sequence[Mapping[str, object]], value_columns: Sequence[str]
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float, str], list[Mapping[str, object]]] = {}
    for row in rows:
        model = str(row.get("model_id", "contrast"))
        key = (
            str(row["source"]), model,
            float(row["requested_irrelevant_ratio"]), str(row["dataset_id"]),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for (source, model, ratio, dataset_id), group in groups.items():
        item: dict[str, object] = {
            "source": source,
            "model_id": model,
            "requested_irrelevant_ratio": ratio,
            "dataset_id": dataset_id,
            "actual_irrelevant_ratio": float(np.mean([
                float(row["actual_irrelevant_ratio"]) for row in group
            ])),
            "n_pairs": len(group),
            "repeats": tuple(sorted({int(row["transform_repeat"]) for row in group})),
        }
        for column in value_columns:
            values = _finite(row[column] for row in group)
            item[column] = float(np.mean(values)) if len(values) else float("nan")
        result.append(item)
    return result


def _summary(effects: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    dataset_rows = _dataset_averages(
        effects, ("risk_increase", "r2_drop", "rmse_increase")
    )
    groups: dict[tuple[str, str, float], list[Mapping[str, object]]] = {}
    for row in dataset_rows:
        key = (
            str(row["source"]), str(row["model_id"]),
            float(row["requested_irrelevant_ratio"]),
        )
        groups.setdefault(key, []).append(row)
    rows = []
    for (source, model_id, ratio), group in groups.items():
        actual = _finite(row["actual_irrelevant_ratio"] for row in group)
        risk = _stats(row["risk_increase"] for row in group)
        r2 = _stats(row["r2_drop"] for row in group)
        rmse_values = _stats(row["rmse_increase"] for row in group)
        rows.append({
            "source": source,
            "model_id": model_id,
            "requested_irrelevant_ratio": ratio,
            "actual_ratio_min": float(np.min(actual)),
            "actual_ratio_median": float(np.median(actual)),
            "actual_ratio_max": float(np.max(actual)),
            "n_datasets": len(group),
            "n_complete_pairs": sum(int(row["n_pairs"]) for row in group),
            "n_transform_repeats": len(set().union(*(set(row["repeats"]) for row in group))),
            **{f"risk_increase_{key}": value for key, value in risk.items()},
            **{f"r2_drop_{key}": value for key, value in r2.items()},
            **{f"rmse_increase_{key}": value for key, value in rmse_values.items()},
        })
    return rows


def _contrast_summary(
    contrasts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    dataset_rows = _dataset_averages(
        contrasts, ("delta_risk_increase", "delta_r2_drop")
    )
    groups: dict[tuple[str, float], list[Mapping[str, object]]] = {}
    for row in dataset_rows:
        key = (str(row["source"]), float(row["requested_irrelevant_ratio"]))
        groups.setdefault(key, []).append(row)
    rows = []
    for (source, ratio), group in groups.items():
        actual = _finite(row["actual_irrelevant_ratio"] for row in group)
        risk = _stats(row["delta_risk_increase"] for row in group)
        r2 = _stats(row["delta_r2_drop"] for row in group)
        rows.append({
            "source": source,
            "requested_irrelevant_ratio": ratio,
            "actual_ratio_min": float(np.min(actual)),
            "actual_ratio_median": float(np.median(actual)),
            "actual_ratio_max": float(np.max(actual)),
            "model_a": PRIMARY_MODEL_A,
            "model_b": PRIMARY_MODEL_B,
            "n_datasets": len(group),
            "n_complete_pairs": sum(int(row["n_pairs"]) for row in group),
            "n_transform_repeats": len(set().union(*(set(row["repeats"]) for row in group))),
            **{f"delta_risk_{key}": value for key, value in risk.items()},
            **{f"delta_r2_drop_{key}": value for key, value in r2.items()},
        })
    return rows


def _coverage(
    index_rows: Sequence[Mapping[str, str]],
    failure_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    requested: dict[tuple[str, str, float], set[str]] = {}
    successful: dict[tuple[str, str, float], set[str]] = {}
    failed: dict[tuple[str, str, float], set[str]] = {}
    for row in index_rows:
        key = (
            row["source"], row["model_id"], float(row["requested_irrelevant_ratio"])
        )
        dataset = row["dataset_id"]
        requested.setdefault(key, set()).add(dataset)
        successful.setdefault(key, set()).add(dataset)
    for row in failure_rows:
        ratio_value = row["requested_irrelevant_ratio"]
        if ratio_value == "":
            continue
        key = (row["source"], row["model_id"], float(ratio_value))
        dataset = row["dataset_id"]
        requested.setdefault(key, set()).add(dataset)
        failed.setdefault(key, set()).add(dataset)
    rows = []
    for source, model, ratio in sorted(requested):
        failed_datasets = failed.get((source, model, ratio), set())
        successful_datasets = successful.get((source, model, ratio), set()) - failed_datasets
        rows.append({
            "source": source,
            "model_id": model,
            "requested_irrelevant_ratio": ratio,
            "requested_datasets": len(requested[(source, model, ratio)]),
            "successful_datasets": len(successful_datasets),
            "failed_datasets": len(failed_datasets),
        })
    return rows


class DefaultRawPredictionEvaluator(RawPredictionEvaluator):
    """Produce task-paired Part I.1 metrics and cross-dataset summaries."""

    def evaluate(self, predictions_dir: Path, output_dir: Path) -> EvaluationResult:
        prediction_run_dir, stage_root = _resolve_prediction_run(Path(predictions_dir))
        index_rows = _read_csv(
            prediction_run_dir / "index.csv", PREDICTION_INDEX_COLUMNS
        )
        if not index_rows:
            raise ValueError("prediction run contains no successful leaves")
        failure_path = prediction_run_dir / "failures.csv"
        failures = (
            _read_csv(failure_path, PREDICTION_FAILURE_COLUMNS)
            if failure_path.is_file() else []
        )
        metrics = _metric_rows(index_rows, stage_root)
        effects = _paired_effects(metrics)
        contrasts = _model_contrasts(effects)
        summary = _summary(effects)
        contrast_summary = _contrast_summary(contrasts)
        coverage = _coverage(index_rows, failures)

        prediction_run_ids = {row["prediction_run_id"] for row in index_rows}
        if len(prediction_run_ids) != 1:
            raise ValueError("prediction index contains multiple prediction_run_id values")
        prediction_run_id = next(iter(prediction_run_ids))
        evaluation_payload = {
            "algorithm": EVALUATION_ALGORITHM_VERSION,
            "prediction_run_id": prediction_run_id,
            "metric_columns": METRIC_COLUMNS,
        }
        evaluation_run_id = _canonical_hash(evaluation_payload, prefix="evaluation")
        run_dir = (
            Path(output_dir).resolve()
            / "evaluations_part_i1"
            / evaluation_run_id
        )
        metrics_path = _atomic_csv(metrics, run_dir / "metrics.csv", METRIC_COLUMNS)
        effects_path = _atomic_csv(
            effects, run_dir / "paired_effects.csv", PAIRED_EFFECT_COLUMNS
        )
        contrasts_path = _atomic_csv(
            contrasts, run_dir / "model_contrasts.csv", MODEL_CONTRAST_COLUMNS
        )
        contrast_summary_path = _atomic_csv(
            contrast_summary,
            run_dir / "model_contrast_summary.csv",
            MODEL_CONTRAST_SUMMARY_COLUMNS,
        )
        summary_path = _atomic_csv(summary, run_dir / "summary.csv", SUMMARY_COLUMNS)
        coverage_path = _atomic_csv(
            coverage, run_dir / "coverage.csv", COVERAGE_COLUMNS
        )
        _atomic_json({
            "complete": True,
            "evaluation_run_id": evaluation_run_id,
            "prediction_run_id": prediction_run_id,
            "algorithm": EVALUATION_ALGORITHM_VERSION,
            "num_metric_rows": len(metrics),
            "num_paired_effect_rows": len(effects),
            "num_model_contrast_rows": len(contrasts),
            "num_summary_rows": len(summary),
            "num_coverage_rows": len(coverage),
        }, run_dir / "manifest.json")
        return EvaluationResult(
            evaluation_run_id=evaluation_run_id,
            output_dir=run_dir,
            metrics_path=metrics_path,
            paired_effects_path=effects_path,
            model_contrasts_path=contrasts_path,
            contrast_summary_path=contrast_summary_path,
            summary_path=summary_path,
            coverage_path=coverage_path,
        )


def _friedman_sweep_memberships(
    config: ExperimentConfig,
) -> dict[str, tuple[dict[str, object], ...]]:
    memberships: dict[str, tuple[dict[str, object], ...]] = {}
    sample_levels: set[int] = set()
    noise_levels: set[float] = set()
    for source in config.sources:
        if source.source.value != "synthetic":
            continue
        if source.generator.lower() not in {"friedman", "friedman1", "friedman_1"}:
            raise ValueError("Friedman sweep sources must use the Friedman generator")
        parameters = dict(source.parameters)
        n_context = int(parameters.get("n_context", 800))
        n_query = int(parameters.get("n_query", 200))
        n_total = n_context + n_query
        noise_std = float(parameters.get("noise_std", 1.0))
        relevant = int(parameters.get("relevant_features", 5))
        if relevant != 5:
            raise ValueError("Friedman sweeps keep exactly five active features")
        if noise_std <= 0.0:
            raise ValueError("Friedman sweep noise_std values must be positive")
        if not math.isclose(n_context / n_total, 0.8, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Friedman sweeps require an exact 80/20 context/query split")
        rows: list[dict[str, object]] = []
        metadata = {
            "n_total": n_total,
            "n_context": n_context,
            "n_query": n_query,
            "noise_std": noise_std,
        }
        if math.isclose(noise_std, 1.0, rel_tol=0.0, abs_tol=1e-12):
            rows.append({"sweep": "sample_size", **metadata})
            sample_levels.add(n_total)
        if n_total == 1000:
            rows.append({"sweep": "noise_std", **metadata})
            noise_levels.add(noise_std)
        if rows:
            memberships[source.source_id] = tuple(rows)
    if len(sample_levels) < 2:
        raise ValueError("sample-size sweep must contain at least two total sample sizes")
    if len(noise_levels) < 2:
        raise ValueError("noise sweep must contain at least two positive noise_std values")
    return memberships


def _friedman_source_id(row: Mapping[str, object]) -> str:
    source_id, separator, _ = str(row["pair_id"]).partition(":")
    if not separator or not source_id:
        raise ValueError(f"cannot recover source_id from pair_id={row['pair_id']!r}")
    return source_id


def _friedman_sweep_model_summary(
    effects: Sequence[Mapping[str, object]],
    memberships: Mapping[str, tuple[Mapping[str, object], ...]],
) -> list[dict[str, object]]:
    task_groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in effects:
        for member in memberships.get(_friedman_source_id(row), ()):
            key = (
                member["sweep"], member["n_total"], member["n_context"],
                member["n_query"], member["noise_std"], row["model_id"],
                float(row["requested_irrelevant_ratio"]), row["dataset_id"],
            )
            task_groups.setdefault(key, []).append(row)

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for key, rows in task_groups.items():
        task = {
            "r2_drop": float(np.mean(_finite(row["r2_drop"] for row in rows))),
            "clean_r2": float(np.mean(_finite(row["r2_at_baseline"] for row in rows))),
            "n_pairs": len(rows),
            "repeats": {int(row["transform_repeat"]) for row in rows},
        }
        grouped.setdefault(key[:-1], []).append(task)

    result: list[dict[str, object]] = []
    for key, tasks in grouped.items():
        sweep, n_total, n_context, n_query, noise_std, model_id, ratio = key
        clean = _stats(task["clean_r2"] for task in tasks)
        drop = _stats(task["r2_drop"] for task in tasks)
        result.append({
            "sweep": sweep,
            "n_total": n_total,
            "n_context": n_context,
            "n_query": n_query,
            "noise_std": noise_std,
            "model_id": model_id,
            "requested_irrelevant_ratio": ratio,
            "n_tasks": len(tasks),
            "n_complete_pairs": sum(int(task["n_pairs"]) for task in tasks),
            "n_transform_repeats": len(set().union(*(
                set(task["repeats"]) for task in tasks
            ))),
            **{f"clean_r2_{name}": value for name, value in clean.items()},
            **{f"r2_drop_{name}": value for name, value in drop.items()},
        })
    return sorted(result, key=lambda row: (
        str(row["sweep"]), float(row["noise_std"]), int(row["n_total"]),
        str(row["model_id"]), float(row["requested_irrelevant_ratio"]),
    ))


def _friedman_sweep_contrast_summary(
    contrasts: Sequence[Mapping[str, object]],
    memberships: Mapping[str, tuple[Mapping[str, object], ...]],
) -> list[dict[str, object]]:
    task_groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in contrasts:
        for member in memberships.get(_friedman_source_id(row), ()):
            key = (
                member["sweep"], member["n_total"], member["n_context"],
                member["n_query"], member["noise_std"],
                float(row["requested_irrelevant_ratio"]), row["model_a"],
                row["model_b"], row["dataset_id"],
            )
            task_groups.setdefault(key, []).append(row)

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for key, rows in task_groups.items():
        task = {
            "delta": float(np.mean(_finite(
                row["delta_r2_drop"] for row in rows
            ))),
            "n_pairs": len(rows),
            "repeats": {int(row["transform_repeat"]) for row in rows},
        }
        grouped.setdefault(key[:-1], []).append(task)

    result: list[dict[str, object]] = []
    for key, tasks in grouped.items():
        (
            sweep, n_total, n_context, n_query, noise_std, ratio,
            model_a, model_b,
        ) = key
        delta = _stats(task["delta"] for task in tasks)
        result.append({
            "sweep": sweep,
            "n_total": n_total,
            "n_context": n_context,
            "n_query": n_query,
            "noise_std": noise_std,
            "requested_irrelevant_ratio": ratio,
            "model_a": model_a,
            "model_b": model_b,
            "n_tasks": len(tasks),
            "n_complete_pairs": sum(int(task["n_pairs"]) for task in tasks),
            "n_transform_repeats": len(set().union(*(
                set(task["repeats"]) for task in tasks
            ))),
            **{f"delta_r2_drop_{name}": value for name, value in delta.items()},
        })
    return sorted(result, key=lambda row: (
        str(row["sweep"]), float(row["noise_std"]), int(row["n_total"]),
        float(row["requested_irrelevant_ratio"]),
    ))


class DefaultFriedmanSweepEvaluator:
    """Aggregate separate sample-size and noise sweeps without crossing them."""

    def evaluate(
        self,
        evaluation: EvaluationResult,
        config: ExperimentConfig,
    ) -> FriedmanSweepEvaluationResult:
        memberships = _friedman_sweep_memberships(config)
        effects = _read_csv(evaluation.paired_effects_path, PAIRED_EFFECT_COLUMNS)
        contrasts = _read_csv(
            evaluation.model_contrasts_path, MODEL_CONTRAST_COLUMNS
        )
        model_summary = _friedman_sweep_model_summary(effects, memberships)
        contrast_summary = _friedman_sweep_contrast_summary(
            contrasts, memberships
        )
        # A single-model run has no model contrasts; only the per-model
        # summary is required.
        if not model_summary:
            raise ValueError("Friedman sweep evaluation contains no complete pairs")
        sweep_id = _canonical_hash({
            "algorithm": FRIEDMAN_SWEEP_EVALUATION_VERSION,
            "evaluation_run_id": evaluation.evaluation_run_id,
            "memberships": memberships,
        }, prefix="friedman-sweeps")
        output = evaluation.output_dir / "friedman_sweeps"
        model_path = _atomic_csv(
            model_summary,
            output / "model_summary.csv",
            FRIEDMAN_SWEEP_MODEL_SUMMARY_COLUMNS,
        )
        contrast_path = _atomic_csv(
            contrast_summary,
            output / "model_contrast_summary.csv",
            FRIEDMAN_SWEEP_CONTRAST_SUMMARY_COLUMNS,
        )
        _atomic_json({
            "complete": True,
            "algorithm": FRIEDMAN_SWEEP_EVALUATION_VERSION,
            "friedman_sweep_evaluation_id": sweep_id,
            "evaluation_run_id": evaluation.evaluation_run_id,
            "num_model_summary_rows": len(model_summary),
            "num_contrast_summary_rows": len(contrast_summary),
            "source_memberships": memberships,
        }, output / "manifest.json")
        return FriedmanSweepEvaluationResult(
            evaluation_run_id=sweep_id,
            output_dir=output,
            model_summary_path=model_path,
            contrast_summary_path=contrast_path,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expanded_friedman_sources(
    config: ExperimentConfig,
) -> dict[str, tuple[float, ...]]:
    sources: dict[str, tuple[float, ...]] = {}
    for source in config.sources:
        if source.source.value != "synthetic":
            continue
        if source.generator.lower() not in {"friedman", "friedman1", "friedman_1"}:
            raise ValueError("expanded Part I.1 only accepts Friedman synthetic sources")
        if source.source_id in sources:
            raise ValueError(f"duplicate Friedman source_id: {source.source_id}")
        sources[source.source_id] = tuple(sorted(float(value) for value in source.ratios))
    if not sources:
        raise ValueError("expanded Part I.1 requires at least one Friedman source")
    return sources


def _complete_dataset_ids(
    rows: Sequence[Mapping[str, object]],
    *,
    ratios: Sequence[float],
    models: Sequence[str],
) -> set[str]:
    required = {(model, float(ratio)) for model in models for ratio in ratios}
    observed: dict[str, set[tuple[str, float]]] = {}
    for row in rows:
        observed.setdefault(str(row["dataset_id"]), set()).add((
            str(row["model_id"]), float(row["requested_irrelevant_ratio"]),
        ))
    return {
        dataset_id for dataset_id, conditions in observed.items()
        if required.issubset(conditions)
    }


class DefaultExpandedPartI1Evaluator:
    """Pool equally weighted Friedman settings and retain paired OpenML tasks."""

    def evaluate(
        self,
        part_i1_evaluation_dir: Path,
        friedman_sweep_evaluation_dir: Path,
        config: ExperimentConfig,
        output_dir: Path,
        *,
        tasks_per_setting: int = 50,
    ) -> ExpandedPartI1EvaluationResult:
        if tasks_per_setting <= 0:
            raise ValueError("tasks_per_setting must be positive")
        part_i1_effects_path = Path(part_i1_evaluation_dir).resolve() / "paired_effects.csv"
        sweep_effects_path = (
            Path(friedman_sweep_evaluation_dir).resolve() / "paired_effects.csv"
        )
        part_i1_effects = _read_csv(part_i1_effects_path, PAIRED_EFFECT_COLUMNS)
        sweep_effects = _read_csv(sweep_effects_path, PAIRED_EFFECT_COLUMNS)
        sources = _expanded_friedman_sources(config)
        models = tuple(model.model_id for model in config.models)
        if len(set(models)) != len(models) or not models:
            raise ValueError("expanded Part I.1 requires unique configured models")

        synthetic_by_source: dict[str, list[dict[str, str]]] = {
            source_id: [] for source_id in sources
        }
        for row in sweep_effects:
            if row["source"] != "synthetic" or row["model_id"] not in models:
                continue
            source_id = _friedman_source_id(row)
            if source_id in synthetic_by_source:
                synthetic_by_source[source_id].append(row)

        combined: list[dict[str, object]] = []
        selected_tasks: dict[str, tuple[str, ...]] = {}
        for source_id, ratios in sources.items():
            source_rows = synthetic_by_source[source_id]
            complete = sorted(_complete_dataset_ids(
                source_rows, ratios=ratios, models=models
            ))
            if len(complete) < tasks_per_setting:
                raise ValueError(
                    f"{source_id} has {len(complete)} complete tasks; "
                    f"{tasks_per_setting} required"
                )
            chosen = tuple(complete[:tasks_per_setting])
            selected_tasks[source_id] = chosen
            chosen_set = set(chosen)
            for row in source_rows:
                if row["dataset_id"] not in chosen_set:
                    continue
                item: dict[str, object] = dict(row)
                item["dataset_id"] = f"{source_id}::{row['dataset_id']}"
                item["dataset_name"] = f"{source_id}/{row['dataset_name']}"
                combined.append(item)

        real_rows = [
            row for row in part_i1_effects
            if row["source"] == "real" and row["model_id"] in models
        ]
        real_ratios = tuple(sorted({
            ratio for ratios in sources.values() for ratio in ratios
        }))
        complete_real = _complete_dataset_ids(
            real_rows, ratios=real_ratios, models=models
        )
        if not complete_real:
            raise ValueError("Part I.1 evaluation contains no complete OpenML datasets")
        combined.extend(
            dict(row) for row in real_rows if row["dataset_id"] in complete_real
        )
        combined.sort(key=lambda row: (
            str(row["source"]), str(row["dataset_id"]), str(row["model_id"]),
            float(row["requested_irrelevant_ratio"]), int(row["transform_repeat"]),
        ))

        contrasts = _model_contrasts(combined)
        summary = _summary(combined)
        contrast_summary = _contrast_summary(contrasts)
        payload = {
            "algorithm": EXPANDED_PART_I1_EVALUATION_VERSION,
            "part_i1_effects_sha256": _file_sha256(part_i1_effects_path),
            "friedman_sweep_effects_sha256": _file_sha256(sweep_effects_path),
            "friedman_sources": sources,
            "tasks_per_setting": tasks_per_setting,
            "models": models,
        }
        evaluation_id = _canonical_hash(payload, prefix="expanded-part-i1")
        output = Path(output_dir).resolve() / evaluation_id
        effects_path = _atomic_csv(
            combined, output / "paired_effects.csv", PAIRED_EFFECT_COLUMNS
        )
        contrasts_path = _atomic_csv(
            contrasts, output / "model_contrasts.csv", MODEL_CONTRAST_COLUMNS
        )
        contrast_summary_path = _atomic_csv(
            contrast_summary,
            output / "model_contrast_summary.csv",
            MODEL_CONTRAST_SUMMARY_COLUMNS,
        )
        summary_path = _atomic_csv(summary, output / "summary.csv", SUMMARY_COLUMNS)
        manifest_path = output / "manifest.json"
        _atomic_json({
            "complete": True,
            "evaluation_run_id": evaluation_id,
            **payload,
            "selected_friedman_tasks": selected_tasks,
            "num_friedman_settings": len(sources),
            "num_friedman_task_setting_units": len(sources) * tasks_per_setting,
            "num_openml_datasets": len(complete_real),
            "num_paired_effect_rows": len(combined),
            "num_model_contrast_rows": len(contrasts),
        }, manifest_path)
        return ExpandedPartI1EvaluationResult(
            evaluation_run_id=evaluation_id,
            output_dir=output,
            paired_effects_path=effects_path,
            model_contrasts_path=contrasts_path,
            contrast_summary_path=contrast_summary_path,
            summary_path=summary_path,
            manifest_path=manifest_path,
        )


# Part I.2 functional-suppression evaluation lives beside Part I.1 so the
# experiment has one evaluation module. Its private helpers are explicitly
# prefixed because the two stages intentionally use different aggregation
# schemas.
PART_I2_TASK_EFFECT_COLUMNS = (
    "source", "dataset_id", "dataset_name", "model_id",
    "requested_irrelevant_ratio", "metric", "feature_role",
    "n_transform_repeats", "n_feature_rows", "value",
)


# Every Part I figure and table covers exactly these benchmarks for every
# model and dose: the seven Friedman settings with 50 tasks each, and the 42
# OpenML datasets. The original 200-task Friedman suite is never used.
BENCHMARK_SIZES = {"synthetic": 350, "real": 42}


def require_benchmark_sizes(rows, what: str) -> None:
    """Refuse inputs that mix in, or miss, any task-setting or dataset.

    Synthetic ``dataset_id`` values must already be qualified by their
    Friedman setting (``<setting>::<task>``), because the same task seeds are
    reused across settings.
    """

    datasets: dict[tuple[str, str, float], set[str]] = {}
    for row in rows:
        key = (
            row["source"], row["model_id"],
            float(row["requested_irrelevant_ratio"]),
        )
        datasets.setdefault(key, set()).add(row["dataset_id"])
    wrong = sorted(
        (source, model, ratio, len(ids))
        for (source, model, ratio), ids in datasets.items()
        if len(ids) != BENCHMARK_SIZES[source]
    )
    if wrong:
        source, model, ratio, count = wrong[0]
        raise ValueError(
            f"{what}: {model} at rho={ratio:g} has {count} {source} tasks, "
            f"expected {BENCHMARK_SIZES[source]} ({len(wrong)} such groups)"
        )
PART_I2_SUMMARY_COLUMNS = (
    "source", "model_id", "requested_irrelevant_ratio", "metric",
    "feature_role", "n_datasets", "mean", "std", "median", "q25", "q75",
)
PART_I2_CONTRAST_COLUMNS = (
    "source", "dataset_id", "requested_irrelevant_ratio", "metric",
    "feature_role", "model_a", "model_b", "value_a", "value_b",
    "delta_a_minus_b",
)
PART_I2_CONTRAST_SUMMARY_COLUMNS = (
    "source", "requested_irrelevant_ratio", "metric", "feature_role",
    "model_a", "model_b", "n_datasets", "mean", "std", "median", "q25",
    "q75",
)
PART_I2_COVERAGE_COLUMNS = (
    "source", "model_id", "requested_irrelevant_ratio", "successful_datasets",
    "successful_leaves", "failed_datasets", "failed_leaves",
)
PART_I2_DATASET_DIMENSION_COLUMNS = (
    "source", "dataset_id", "dataset_name", "d_base",
)
PART_I2_PDP_CURVE_COLUMNS = (
    "source", "dataset_id", "dataset_name", "model_id", "condition_id",
    "requested_irrelevant_ratio", "transform_repeat", "feature_role",
    "source_feature_index", "copy_block_id", "feature_bank_id", "grid_index",
    "grid_value", "pdp_value", "n_query",
)
PART_I2_FRIEDMAN_SWEEP_SUMMARY_COLUMNS = (
    "sweep", "n_total", "n_context", "n_query", "noise_std",
    "model_id", "requested_irrelevant_ratio", "metric", "feature_role",
    "n_tasks", "n_transform_repeats", "n_feature_rows", "mean", "std",
    "median", "q25", "q75",
)
PART_I2_FRIEDMAN_SWEEP_CONTRAST_SUMMARY_COLUMNS = (
    "sweep", "n_total", "n_context", "n_query", "noise_std",
    "requested_irrelevant_ratio", "metric", "feature_role", "model_a",
    "model_b", "n_tasks", "mean", "std", "median", "q25", "q75",
)
PART_I2_FRIEDMAN_SWEEP_COVERAGE_COLUMNS = (
    "sweep", "n_total", "n_context", "n_query", "noise_std", "model_id",
    "requested_irrelevant_ratio", "successful_tasks", "successful_leaves",
    "failed_tasks", "failed_leaves",
)


@dataclass(frozen=True)
class PartI2EvaluationResult:
    evaluation_run_id: str
    output_dir: Path
    task_effects_path: Path
    summary_path: Path
    contrasts_path: Path
    contrast_summary_path: Path
    coverage_path: Path
    pdp_curves_path: Path
    dataset_dimensions_path: Path | None = None


@dataclass(frozen=True)
class FriedmanSweepPartI2EvaluationResult:
    evaluation_run_id: str
    output_dir: Path
    task_effects_path: Path
    summary_path: Path
    contrasts_path: Path
    contrast_summary_path: Path
    coverage_path: Path
    manifest_path: Path


def _read_part_i2_csv(
    path: Path, columns: tuple[str, ...]
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"unexpected columns in {path}")
        return list(reader)


def _resolve_part_i2_run(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    run_dir = path.parent if path.is_file() else path
    if not (run_dir / "index.csv").is_file():
        raise FileNotFoundError(f"Part I.2 index not found: {run_dir}")
    if run_dir.parent.name != "runs":
        raise ValueError("Part I.2 run directory must be below part_i2/runs")
    return run_dir, run_dir.parents[1]


def _part_i2_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            key: float("nan")
            for key in ("mean", "std", "median", "q25", "q75")
        }
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _part_i2_feature_rows(
    index_rows: Sequence[Mapping[str, str]], stage_root: Path
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in index_rows:
        path = stage_root / index["metrics_path"]
        leaf = _read_part_i2_csv(path, FEATURE_METRIC_COLUMNS)
        for row in leaf:
            if (
                row["condition_id"] != index["condition_id"]
                or row["model_id"] != index["model_id"]
            ):
                raise ValueError(f"metric leaf identity does not match index: {path}")
        rows.extend(leaf)
    return rows


def _part_i2_task_effects(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    # Track the earliest positive-dose irrelevant cohort through later nested
    # doses without extra inference. The cohort key excludes model so paired
    # model contrasts always refer to the same logical features.
    anchor_groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        if row["metric"] == "pdp_variation" and row["feature_role"] == "irrelevant":
            key = (row["source"], row["dataset_id"], row["transform_repeat"])
            anchor_groups.setdefault(key, []).append(row)
    anchor_ids: dict[tuple[str, str, str], set[str]] = {}
    for key, group in anchor_groups.items():
        minimum = min(float(row["requested_irrelevant_ratio"]) for row in group)
        anchor_ids[key] = {
            row["feature_bank_id"]
            for row in group
            if float(row["requested_irrelevant_ratio"]) == minimum
        }
    expanded: list[Mapping[str, str]] = list(rows)
    for key, group in anchor_groups.items():
        for row in group:
            if row["feature_bank_id"] in anchor_ids[key]:
                duplicate = dict(row)
                duplicate["metric"] = "pdp_anchor_variation"
                expanded.append(duplicate)

    repeat_groups: dict[tuple[str, ...], list[float]] = {}
    names: dict[tuple[str, ...], str] = {}
    for row in expanded:
        key = (
            row["source"], row["dataset_id"], row["model_id"],
            row["requested_irrelevant_ratio"], row["metric"],
            row["feature_role"], row["transform_repeat"],
        )
        repeat_groups.setdefault(key, []).append(float(row["value"]))
        names[key] = row["dataset_name"]
    repeated: list[dict[str, object]] = []
    for key, values in repeat_groups.items():
        source, dataset_id, model_id, ratio, metric, role, repeat = key
        repeated.append({
            "source": source,
            "dataset_id": dataset_id,
            "dataset_name": names[key],
            "model_id": model_id,
            "requested_irrelevant_ratio": float(ratio),
            "metric": metric,
            "feature_role": role,
            "transform_repeat": int(repeat),
            "n_features": len(values),
            "value": float(np.mean(values)),
        })

    task_groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in repeated:
        key = (
            row["source"], row["dataset_id"], row["dataset_name"],
            row["model_id"], row["requested_irrelevant_ratio"], row["metric"],
            row["feature_role"],
        )
        task_groups.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for key, group in task_groups.items():
        source, dataset_id, dataset_name, model_id, ratio, metric, role = key
        result.append({
            "source": source,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "model_id": model_id,
            "requested_irrelevant_ratio": ratio,
            "metric": metric,
            "feature_role": role,
            "n_transform_repeats": len(group),
            "n_feature_rows": sum(int(row["n_features"]) for row in group),
            "value": float(np.mean([float(row["value"]) for row in group])),
        })
    return sorted(result, key=lambda row: (
        str(row["source"]), str(row["metric"]), str(row["feature_role"]),
        float(row["requested_irrelevant_ratio"]), str(row["model_id"]),
        str(row["dataset_id"]),
    ))


def _part_i2_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = (
            row["source"], row["model_id"], row["requested_irrelevant_ratio"],
            row["metric"], row["feature_role"],
        )
        groups.setdefault(key, []).append(float(row["value"]))
    result = [{
        "source": key[0],
        "model_id": key[1],
        "requested_irrelevant_ratio": key[2],
        "metric": key[3],
        "feature_role": key[4],
        "n_datasets": len(values),
        **_part_i2_stats(values),
    } for key, values in groups.items()]
    return sorted(result, key=lambda row: (
        str(row["source"]), str(row["metric"]), str(row["feature_role"]),
        float(row["requested_irrelevant_ratio"]), str(row["model_id"]),
    ))


def _part_i2_contrasts(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    lookup: dict[tuple[object, ...], dict[str, float]] = {}
    for row in rows:
        key = (
            row["source"], row["dataset_id"], row["requested_irrelevant_ratio"],
            row["metric"], row["feature_role"],
        )
        lookup.setdefault(key, {})[str(row["model_id"])] = float(row["value"])
    result = []
    for key, values in lookup.items():
        if PRIMARY_MODEL_A not in values or PRIMARY_MODEL_B not in values:
            continue
        result.append({
            "source": key[0],
            "dataset_id": key[1],
            "requested_irrelevant_ratio": key[2],
            "metric": key[3],
            "feature_role": key[4],
            "model_a": PRIMARY_MODEL_A,
            "model_b": PRIMARY_MODEL_B,
            "value_a": values[PRIMARY_MODEL_A],
            "value_b": values[PRIMARY_MODEL_B],
            "delta_a_minus_b": values[PRIMARY_MODEL_A] - values[PRIMARY_MODEL_B],
        })
    return sorted(result, key=lambda row: (
        str(row["source"]), str(row["metric"]), str(row["feature_role"]),
        float(row["requested_irrelevant_ratio"]), str(row["dataset_id"]),
    ))


def _part_i2_contrast_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = (
            row["source"], row["requested_irrelevant_ratio"], row["metric"],
            row["feature_role"], row["model_a"], row["model_b"],
        )
        groups.setdefault(key, []).append(float(row["delta_a_minus_b"]))
    result = [{
        "source": key[0],
        "requested_irrelevant_ratio": key[1],
        "metric": key[2],
        "feature_role": key[3],
        "model_a": key[4],
        "model_b": key[5],
        "n_datasets": len(values),
        **_part_i2_stats(values),
    } for key, values in groups.items()]
    return sorted(result, key=lambda row: (
        str(row["source"]), str(row["metric"]), str(row["feature_role"]),
        float(row["requested_irrelevant_ratio"]),
    ))


def _part_i2_coverage(
    index_rows: Sequence[Mapping[str, str]],
    failures: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    successful: dict[tuple[str, str, float], list[Mapping[str, str]]] = {}
    failed: dict[tuple[str, str, float], list[Mapping[str, str]]] = {}
    for row in index_rows:
        key = (row["source"], row["model_id"], float(row["requested_irrelevant_ratio"]))
        successful.setdefault(key, []).append(row)
    for row in failures:
        key = (row["source"], row["model_id"], float(row["requested_irrelevant_ratio"]))
        failed.setdefault(key, []).append(row)
    result = []
    for key in sorted(set(successful) | set(failed)):
        good, bad = successful.get(key, []), failed.get(key, [])
        result.append({
            "source": key[0],
            "model_id": key[1],
            "requested_irrelevant_ratio": key[2],
            "successful_datasets": len({row["dataset_id"] for row in good}),
            "successful_leaves": len(good),
            "failed_datasets": len({row["dataset_id"] for row in bad}),
            "failed_leaves": len(bad),
        })
    return result


def _part_i2_pdp_curves(
    index_rows: Sequence[Mapping[str, str]], stage_root: Path
) -> list[dict[str, object]]:
    import h5py

    result: list[dict[str, object]] = []
    for index in index_rows:
        metric_rows = [
            row for row in _read_part_i2_csv(
                stage_root / index["metrics_path"], FEATURE_METRIC_COLUMNS
            )
            if row["metric"] == "pdp_variation"
        ]
        with h5py.File(stage_root / index["raw_path"], "r") as handle:
            columns = handle["pdp/physical_column"][:]
            grids = handle["pdp/grid"][:]
            predictions = handle["pdp/predictions"][:]
        if len(metric_rows) != len(columns):
            raise ValueError("PDP metrics and raw feature arrays disagree")
        for row, column, grid, prediction in zip(
            metric_rows, columns, grids, predictions
        ):
            if int(row["physical_column"]) != int(column):
                raise ValueError("PDP metric order does not match raw arrays")
            for grid_index, (grid_value, values) in enumerate(zip(grid, prediction)):
                result.append({
                    "source": row["source"],
                    "dataset_id": row["dataset_id"],
                    "dataset_name": row["dataset_name"],
                    "model_id": row["model_id"],
                    "condition_id": row["condition_id"],
                    "requested_irrelevant_ratio": float(
                        row["requested_irrelevant_ratio"]
                    ),
                    "transform_repeat": int(row["transform_repeat"]),
                    "feature_role": row["feature_role"],
                    "source_feature_index": int(row["source_feature_index"]),
                    "copy_block_id": int(row["copy_block_id"]),
                    "feature_bank_id": int(row["feature_bank_id"]),
                    "grid_index": grid_index,
                    "grid_value": float(grid_value),
                    "pdp_value": float(np.mean(values)),
                    "n_query": int(row["n_query"]),
                })
    return result


def _part_i2_dataset_dimensions(
    index_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Record one pre-augmentation feature width per evaluated dataset."""
    grouped: dict[tuple[str, str, str], set[int]] = {}
    for row in index_rows:
        key = (row["source"], row["dataset_id"], row["dataset_name"])
        grouped.setdefault(key, set()).add(int(row["d_base"]))
    result: list[dict[str, object]] = []
    for (source, dataset_id, dataset_name), widths in sorted(grouped.items()):
        if len(widths) != 1:
            raise ValueError(
                f"dataset {dataset_id} has inconsistent base widths: {sorted(widths)}"
            )
        result.append({
            "source": source,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "d_base": next(iter(widths)),
        })
    return result


class DefaultPartI2Evaluator:
    """Evaluate cached Part I.2 feature-level functional artifacts."""

    def evaluate(self, run_path: Path, output_dir: Path) -> PartI2EvaluationResult:
        run_dir, stage_root = _resolve_part_i2_run(Path(run_path))
        index_rows = _read_part_i2_csv(
            run_dir / "index.csv", PART_I2_INDEX_COLUMNS
        )
        if not index_rows:
            raise ValueError("Part I.2 run contains no successful leaves")
        failure_path = run_dir / "failures.csv"
        failures = (
            _read_part_i2_csv(failure_path, PART_I2_FAILURE_COLUMNS)
            if failure_path.is_file() else []
        )
        feature_rows = _part_i2_feature_rows(index_rows, stage_root)
        task_effects = _part_i2_task_effects(feature_rows)
        summary = _part_i2_summary(task_effects)
        contrasts = _part_i2_contrasts(task_effects)
        contrast_summary = _part_i2_contrast_summary(contrasts)
        coverage = _part_i2_coverage(index_rows, failures)
        curves = _part_i2_pdp_curves(index_rows, stage_root)
        dataset_dimensions = _part_i2_dataset_dimensions(index_rows)

        run_ids = {row["part_i2_run_id"] for row in index_rows}
        if len(run_ids) != 1:
            raise ValueError("Part I.2 index contains multiple run ids")
        part_i2_run_id = next(iter(run_ids))
        evaluation_run_id = _canonical_hash({
            "algorithm": PART_I2_EVALUATION_VERSION,
            "part_i2_run_id": part_i2_run_id,
        }, prefix="part-i2-evaluation")
        output = (
            Path(output_dir).resolve()
            / "evaluations_part_i2"
            / evaluation_run_id
        )
        task_path = _atomic_csv(
            task_effects, output / "task_effects.csv", PART_I2_TASK_EFFECT_COLUMNS
        )
        summary_path = _atomic_csv(
            summary, output / "summary.csv", PART_I2_SUMMARY_COLUMNS
        )
        contrasts_path = _atomic_csv(
            contrasts, output / "model_contrasts.csv", PART_I2_CONTRAST_COLUMNS
        )
        contrast_summary_path = _atomic_csv(
            contrast_summary,
            output / "model_contrast_summary.csv",
            PART_I2_CONTRAST_SUMMARY_COLUMNS,
        )
        coverage_path = _atomic_csv(
            coverage, output / "coverage.csv", PART_I2_COVERAGE_COLUMNS
        )
        curves_path = _atomic_csv(
            curves, output / "pdp_curves.csv", PART_I2_PDP_CURVE_COLUMNS
        )
        dataset_dimensions_path = _atomic_csv(
            dataset_dimensions,
            output / "dataset_dimensions.csv",
            PART_I2_DATASET_DIMENSION_COLUMNS,
        )
        _atomic_json({
            "complete": True,
            "algorithm": PART_I2_EVALUATION_VERSION,
            "evaluation_run_id": evaluation_run_id,
            "part_i2_run_id": part_i2_run_id,
            "num_task_effect_rows": len(task_effects),
            "num_summary_rows": len(summary),
            "num_contrast_rows": len(contrasts),
            "num_pdp_curve_rows": len(curves),
            "num_datasets": len(dataset_dimensions),
        }, output / "manifest.json")
        return PartI2EvaluationResult(
            evaluation_run_id=evaluation_run_id,
            output_dir=output,
            task_effects_path=task_path,
            summary_path=summary_path,
            contrasts_path=contrasts_path,
            contrast_summary_path=contrast_summary_path,
            coverage_path=coverage_path,
            pdp_curves_path=curves_path,
            dataset_dimensions_path=dataset_dimensions_path,
        )


def _part_i2_friedman_task_effects(
    feature_rows: Sequence[Mapping[str, str]],
    memberships: Mapping[str, tuple[Mapping[str, object], ...]],
) -> list[dict[str, object]]:
    tagged: list[dict[str, str]] = []
    for row in feature_rows:
        source_id = _friedman_source_id(row)
        if source_id not in memberships:
            raise ValueError(f"Part I.2 row has unknown Friedman source: {source_id}")
        item = dict(row)
        item["dataset_id"] = f"{source_id}::{row['dataset_id']}"
        item["dataset_name"] = f"{source_id}/{row['dataset_name']}"
        tagged.append(item)
    return _part_i2_task_effects(tagged)


def _part_i2_friedman_summary(
    task_effects: Sequence[Mapping[str, object]],
    memberships: Mapping[str, tuple[Mapping[str, object], ...]],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in task_effects:
        source_id, separator, _ = str(row["dataset_id"]).partition("::")
        if not separator or source_id not in memberships:
            raise ValueError("tagged Friedman task identity is invalid")
        for member in memberships[source_id]:
            key = (
                member["sweep"], member["n_total"], member["n_context"],
                member["n_query"], member["noise_std"], row["model_id"],
                row["requested_irrelevant_ratio"], row["metric"],
                row["feature_role"],
            )
            groups.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for key, rows in groups.items():
        repeats = {int(row["n_transform_repeats"]) for row in rows}
        if len(repeats) != 1:
            raise ValueError("Friedman sweep tasks have inconsistent repeat counts")
        result.append({
            "sweep": key[0],
            "n_total": key[1],
            "n_context": key[2],
            "n_query": key[3],
            "noise_std": key[4],
            "model_id": key[5],
            "requested_irrelevant_ratio": key[6],
            "metric": key[7],
            "feature_role": key[8],
            "n_tasks": len(rows),
            "n_transform_repeats": next(iter(repeats)),
            "n_feature_rows": sum(int(row["n_feature_rows"]) for row in rows),
            **_part_i2_stats(float(row["value"]) for row in rows),
        })
    return sorted(result, key=lambda row: (
        str(row["sweep"]), str(row["metric"]), str(row["feature_role"]),
        float(row["noise_std"]), int(row["n_total"]),
        float(row["requested_irrelevant_ratio"]), str(row["model_id"]),
    ))


def _part_i2_friedman_contrast_summary(
    contrasts: Sequence[Mapping[str, object]],
    memberships: Mapping[str, tuple[Mapping[str, object], ...]],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in contrasts:
        source_id, separator, _ = str(row["dataset_id"]).partition("::")
        if not separator or source_id not in memberships:
            raise ValueError("tagged Friedman contrast identity is invalid")
        for member in memberships[source_id]:
            key = (
                member["sweep"], member["n_total"], member["n_context"],
                member["n_query"], member["noise_std"],
                row["requested_irrelevant_ratio"], row["metric"],
                row["feature_role"], row["model_a"], row["model_b"],
            )
            groups.setdefault(key, []).append(float(row["delta_a_minus_b"]))
    result = [{
        "sweep": key[0],
        "n_total": key[1],
        "n_context": key[2],
        "n_query": key[3],
        "noise_std": key[4],
        "requested_irrelevant_ratio": key[5],
        "metric": key[6],
        "feature_role": key[7],
        "model_a": key[8],
        "model_b": key[9],
        "n_tasks": len(values),
        **_part_i2_stats(values),
    } for key, values in groups.items()]
    return sorted(result, key=lambda row: (
        str(row["sweep"]), str(row["metric"]), str(row["feature_role"]),
        float(row["noise_std"]), int(row["n_total"]),
        float(row["requested_irrelevant_ratio"]),
    ))


def _part_i2_friedman_coverage(
    index_rows: Sequence[Mapping[str, str]],
    failures: Sequence[Mapping[str, str]],
    memberships: Mapping[str, tuple[Mapping[str, object], ...]],
) -> list[dict[str, object]]:
    successful: dict[tuple[object, ...], list[Mapping[str, str]]] = {}
    failed: dict[tuple[object, ...], list[Mapping[str, str]]] = {}
    for rows, target in ((index_rows, successful), (failures, failed)):
        for row in rows:
            try:
                source_id = _friedman_source_id(row)
            except ValueError:
                continue
            for member in memberships.get(source_id, ()):
                key = (
                    member["sweep"], member["n_total"], member["n_context"],
                    member["n_query"], member["noise_std"], row["model_id"],
                    float(row["requested_irrelevant_ratio"]),
                )
                target.setdefault(key, []).append(row)
    result = []
    for key in sorted(set(successful) | set(failed), key=str):
        good, bad = successful.get(key, []), failed.get(key, [])
        result.append({
            "sweep": key[0],
            "n_total": key[1],
            "n_context": key[2],
            "n_query": key[3],
            "noise_std": key[4],
            "model_id": key[5],
            "requested_irrelevant_ratio": key[6],
            "successful_tasks": len({row["dataset_id"] for row in good}),
            "successful_leaves": len(good),
            "failed_tasks": len({row["dataset_id"] for row in bad}),
            "failed_leaves": len(bad),
        })
    return result


class DefaultFriedmanSweepPartI2Evaluator:
    """Evaluate Part I.2 while preserving each Friedman setting identity."""

    def evaluate(
        self,
        run_path: Path,
        config: ExperimentConfig,
        output_dir: Path,
    ) -> FriedmanSweepPartI2EvaluationResult:
        run_dir, stage_root = _resolve_part_i2_run(Path(run_path))
        index_rows = _read_part_i2_csv(run_dir / "index.csv", PART_I2_INDEX_COLUMNS)
        if not index_rows:
            raise ValueError("Friedman sweep Part I.2 run contains no successful leaves")
        failure_path = run_dir / "failures.csv"
        failures = (
            _read_part_i2_csv(failure_path, PART_I2_FAILURE_COLUMNS)
            if failure_path.is_file() else []
        )
        memberships = _friedman_sweep_memberships(config)
        feature_rows = _part_i2_feature_rows(index_rows, stage_root)
        task_effects = _part_i2_friedman_task_effects(feature_rows, memberships)
        summary = _part_i2_friedman_summary(task_effects, memberships)
        contrasts = _part_i2_contrasts(task_effects)
        contrast_summary = _part_i2_friedman_contrast_summary(
            contrasts, memberships
        )
        coverage = _part_i2_friedman_coverage(index_rows, failures, memberships)
        run_ids = {row["part_i2_run_id"] for row in index_rows}
        if len(run_ids) != 1:
            raise ValueError("Part I.2 index contains multiple run ids")
        part_i2_run_id = next(iter(run_ids))
        evaluation_run_id = _canonical_hash({
            "algorithm": PART_I2_FRIEDMAN_SWEEP_EVALUATION_VERSION,
            "part_i2_run_id": part_i2_run_id,
            "memberships": memberships,
        }, prefix="part-i2-friedman-sweeps")
        output = (
            Path(output_dir).resolve()
            / "evaluations_part_i2"
            / "friedman_sweeps"
            / evaluation_run_id
        )
        task_path = _atomic_csv(
            task_effects, output / "task_effects.csv", PART_I2_TASK_EFFECT_COLUMNS
        )
        summary_path = _atomic_csv(
            summary, output / "summary.csv", PART_I2_FRIEDMAN_SWEEP_SUMMARY_COLUMNS
        )
        contrasts_path = _atomic_csv(
            contrasts, output / "model_contrasts.csv", PART_I2_CONTRAST_COLUMNS
        )
        contrast_summary_path = _atomic_csv(
            contrast_summary,
            output / "model_contrast_summary.csv",
            PART_I2_FRIEDMAN_SWEEP_CONTRAST_SUMMARY_COLUMNS,
        )
        coverage_path = _atomic_csv(
            coverage,
            output / "coverage.csv",
            PART_I2_FRIEDMAN_SWEEP_COVERAGE_COLUMNS,
        )
        manifest_path = output / "manifest.json"
        _atomic_json({
            "complete": True,
            "algorithm": PART_I2_FRIEDMAN_SWEEP_EVALUATION_VERSION,
            "evaluation_run_id": evaluation_run_id,
            "part_i2_run_id": part_i2_run_id,
            "models": sorted({row["model_id"] for row in index_rows}),
            "num_failures": len(failures),
            "num_task_effect_rows": len(task_effects),
            "num_summary_rows": len(summary),
            "num_contrast_rows": len(contrasts),
            "source_memberships": memberships,
        }, manifest_path)
        return FriedmanSweepPartI2EvaluationResult(
            evaluation_run_id=evaluation_run_id,
            output_dir=output,
            task_effects_path=task_path,
            summary_path=summary_path,
            contrasts_path=contrasts_path,
            contrast_summary_path=contrast_summary_path,
            coverage_path=coverage_path,
            manifest_path=manifest_path,
        )


@dataclass(frozen=True)
class FlatEvaluationBundleResult:
    part_i1_dir: Path
    part_i2_dir: Path


class DefaultFlatEvaluationAssembler:
    """Write the final Part I tables without source-specific subdirectories."""

    def assemble(
        self,
        *,
        part_i1_results_dir: Path,
        part_i1_condition_dir: Path,
        part_i2_openml_dir: Path,
        part_i2_friedman_dir: Path,
        output_root: Path,
        additional_pdp_curves_path: Path | None = None,
    ) -> FlatEvaluationBundleResult:
        output_root = Path(output_root).resolve()
        part_i1_dir = output_root / "evaluations_part_i1"
        part_i2_dir = output_root / "evaluations_part_i2"
        if part_i1_dir.exists() or part_i2_dir.exists():
            raise FileExistsError(
                "flat evaluation destinations must be absent before assembly"
            )

        part_i1_results_dir = Path(part_i1_results_dir).resolve()
        part_i1_condition_dir = Path(part_i1_condition_dir).resolve()
        part_i2_openml_dir = Path(part_i2_openml_dir).resolve()
        part_i2_friedman_dir = Path(part_i2_friedman_dir).resolve()

        part_i1_files = (
            ("paired_effects.csv", PAIRED_EFFECT_COLUMNS),
            ("summary.csv", SUMMARY_COLUMNS),
            ("model_contrasts.csv", MODEL_CONTRAST_COLUMNS),
            ("model_contrast_summary.csv", MODEL_CONTRAST_SUMMARY_COLUMNS),
        )
        part_i1_counts: dict[str, int] = {}
        for filename, columns in part_i1_files:
            rows = _read_csv(part_i1_results_dir / filename, columns)
            _atomic_csv(rows, part_i1_dir / filename, columns)
            part_i1_counts[filename] = len(rows)

        condition_files = (
            (
                "model_summary.csv",
                "friedman_condition_summary.csv",
                FRIEDMAN_SWEEP_MODEL_SUMMARY_COLUMNS,
            ),
            (
                "model_contrast_summary.csv",
                "friedman_condition_contrasts.csv",
                FRIEDMAN_SWEEP_CONTRAST_SUMMARY_COLUMNS,
            ),
        )
        condition_counts: dict[str, int] = {}
        for source_name, output_name, columns in condition_files:
            rows = _read_csv(part_i1_condition_dir / source_name, columns)
            _atomic_csv(rows, part_i1_dir / output_name, columns)
            condition_counts[output_name] = len(rows)

        openml_task_effects = _read_part_i2_csv(
            part_i2_openml_dir / "task_effects.csv", PART_I2_TASK_EFFECT_COLUMNS
        )
        friedman_task_effects = _read_part_i2_csv(
            part_i2_friedman_dir / "task_effects.csv", PART_I2_TASK_EFFECT_COLUMNS
        )
        task_effects = sorted(
            openml_task_effects + friedman_task_effects,
            key=lambda row: (
                row["source"], row["dataset_id"], row["model_id"],
                float(row["requested_irrelevant_ratio"]), row["metric"],
                row["feature_role"],
            ),
        )
        _atomic_csv(
            task_effects, part_i2_dir / "task_effects.csv",
            PART_I2_TASK_EFFECT_COLUMNS,
        )

        openml_contrasts = _read_part_i2_csv(
            part_i2_openml_dir / "model_contrasts.csv",
            PART_I2_CONTRAST_COLUMNS,
        )
        friedman_contrasts = _read_part_i2_csv(
            part_i2_friedman_dir / "model_contrasts.csv",
            PART_I2_CONTRAST_COLUMNS,
        )
        contrasts = sorted(
            openml_contrasts + friedman_contrasts,
            key=lambda row: (
                row["source"], row["dataset_id"],
                float(row["requested_irrelevant_ratio"]), row["metric"],
                row["feature_role"], row["model_a"], row["model_b"],
            ),
        )
        _atomic_csv(
            contrasts, part_i2_dir / "model_contrasts.csv",
            PART_I2_CONTRAST_COLUMNS,
        )

        part_i2_files = (
            (
                part_i2_openml_dir / "summary.csv", "summary_openml.csv",
                PART_I2_SUMMARY_COLUMNS,
            ),
            (
                part_i2_friedman_dir / "summary.csv",
                "summary_friedman_conditions.csv",
                PART_I2_FRIEDMAN_SWEEP_SUMMARY_COLUMNS,
            ),
            (
                part_i2_openml_dir / "model_contrast_summary.csv",
                "model_contrast_summary_openml.csv",
                PART_I2_CONTRAST_SUMMARY_COLUMNS,
            ),
            (
                part_i2_friedman_dir / "model_contrast_summary.csv",
                "model_contrast_summary_friedman_conditions.csv",
                PART_I2_FRIEDMAN_SWEEP_CONTRAST_SUMMARY_COLUMNS,
            ),
            (
                part_i2_openml_dir / "coverage.csv", "coverage_openml.csv",
                PART_I2_COVERAGE_COLUMNS,
            ),
            (
                part_i2_friedman_dir / "coverage.csv",
                "coverage_friedman_conditions.csv",
                PART_I2_FRIEDMAN_SWEEP_COVERAGE_COLUMNS,
            ),
            (
                part_i2_openml_dir / "dataset_dimensions.csv",
                "dataset_dimensions.csv", PART_I2_DATASET_DIMENSION_COLUMNS,
            ),
        )
        for source_path, output_name, columns in part_i2_files:
            rows = _read_part_i2_csv(source_path, columns)
            _atomic_csv(rows, part_i2_dir / output_name, columns)

        pdp_curves = _read_part_i2_csv(
            part_i2_openml_dir / "pdp_curves.csv", PART_I2_PDP_CURVE_COLUMNS
        )
        if additional_pdp_curves_path is not None:
            pdp_curves.extend(_read_part_i2_csv(
                Path(additional_pdp_curves_path).resolve(),
                PART_I2_PDP_CURVE_COLUMNS,
            ))
        _atomic_csv(
            pdp_curves, part_i2_dir / "pdp_curves.csv",
            PART_I2_PDP_CURVE_COLUMNS,
        )

        _atomic_json({
            "complete": True,
            "algorithm": FLAT_EVALUATION_LAYOUT_VERSION,
            "num_paired_effect_rows": part_i1_counts["paired_effects.csv"],
            "num_summary_rows": part_i1_counts["summary.csv"],
            "num_model_contrast_rows": part_i1_counts["model_contrasts.csv"],
            "num_friedman_condition_summary_rows": condition_counts[
                "friedman_condition_summary.csv"
            ],
            "sources": ["synthetic", "real"],
        }, part_i1_dir / "manifest.json")
        _atomic_json({
            "complete": True,
            "algorithm": FLAT_EVALUATION_LAYOUT_VERSION,
            "num_task_effect_rows": len(task_effects),
            "num_model_contrast_rows": len(contrasts),
            "num_pdp_curve_rows": len(pdp_curves),
            "sources": ["synthetic", "real"],
        }, part_i2_dir / "manifest.json")
        return FlatEvaluationBundleResult(
            part_i1_dir=part_i1_dir,
            part_i2_dir=part_i2_dir,
        )


__all__ = [
    "BENCHMARK_SIZES",
    "require_benchmark_sizes",
    "COVERAGE_COLUMNS",
    "DefaultExpandedPartI1Evaluator",
    "DefaultFlatEvaluationAssembler",
    "DefaultFriedmanSweepEvaluator",
    "DefaultFriedmanSweepPartI2Evaluator",
    "DefaultRawPredictionEvaluator",
    "EVALUATION_ALGORITHM_VERSION",
    "EvaluationResult",
    "EXPANDED_PART_I1_EVALUATION_VERSION",
    "ExpandedPartI1EvaluationResult",
    "FlatEvaluationBundleResult",
    "FLAT_EVALUATION_LAYOUT_VERSION",
    "FRIEDMAN_SWEEP_CONTRAST_SUMMARY_COLUMNS",
    "FRIEDMAN_SWEEP_EVALUATION_VERSION",
    "FRIEDMAN_SWEEP_MODEL_SUMMARY_COLUMNS",
    "FriedmanSweepEvaluationResult",
    "FriedmanSweepPartI2EvaluationResult",
    "METRIC_COLUMNS",
    "MODEL_CONTRAST_COLUMNS",
    "MODEL_CONTRAST_SUMMARY_COLUMNS",
    "PAIRED_EFFECT_COLUMNS",
    "PART_I2_CONTRAST_COLUMNS",
    "PART_I2_CONTRAST_SUMMARY_COLUMNS",
    "PART_I2_COVERAGE_COLUMNS",
    "PART_I2_DATASET_DIMENSION_COLUMNS",
    "PART_I2_EVALUATION_VERSION",
    "PART_I2_FRIEDMAN_SWEEP_CONTRAST_SUMMARY_COLUMNS",
    "PART_I2_FRIEDMAN_SWEEP_COVERAGE_COLUMNS",
    "PART_I2_FRIEDMAN_SWEEP_EVALUATION_VERSION",
    "PART_I2_FRIEDMAN_SWEEP_SUMMARY_COLUMNS",
    "PART_I2_PDP_CURVE_COLUMNS",
    "PART_I2_SUMMARY_COLUMNS",
    "PART_I2_TASK_EFFECT_COLUMNS",
    "DefaultPartI2Evaluator",
    "PartI2EvaluationResult",
    "PRIMARY_MODEL_A",
    "PRIMARY_MODEL_B",
    "RawPredictionEvaluator",
    "SUMMARY_COLUMNS",
]
