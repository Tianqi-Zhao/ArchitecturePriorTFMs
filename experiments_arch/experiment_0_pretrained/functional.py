"""Cached Part I.2 functional-suppression runner.

The runner traverses each model-condition once, keeps its context fixed, and
shares the fitted backend between exhaustive single-column permutation and
bounded one-way PDP interventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np

from experiment_core.adapters import (
    ModelConfig,
    PredictiveModel,
    PretrainedModelFactory,
)
from experiments_arch.experiment_0_pretrained.config import (
    ExperimentConfig,
    PartI2Config,
)
from experiments_arch.experiment_0_pretrained.data.base import (
    DatasetSource,
    FeatureRole,
    PreparedDataset,
)
from experiments_arch.experiment_0_pretrained.data.store import (
    ConditionRef,
    HDF5DatasetStore,
    SKIP_COLUMNS,
)
from experiments_arch.experiment_0_pretrained.predict import (
    _atomic_csv,
    _atomic_json,
    _canonical_hash,
    _model_fingerprint,
    _selected_configs,
)


PART_I2_ALGORITHM_VERSION = "part-i-2-functional-v1"

FEATURE_METRIC_COLUMNS = (
    "schema_version", "dataset_id", "dataset_name", "source", "condition_id",
    "pair_id", "model_id", "model_version", "requested_irrelevant_ratio",
    "actual_irrelevant_ratio", "d_base", "d_irrelevant", "d_total",
    "transform_repeat", "metric", "feature_role", "physical_column",
    "source_feature_index", "copy_block_id", "feature_bank_id", "n_query",
    "n_grid_points", "value",
)

PART_I2_INDEX_COLUMNS = (
    "schema_version", "data_run_id", "part_i2_run_id", "analysis_cache_key",
    "model_cache_key", "data_cache_key", "dataset_id", "dataset_name",
    "source", "condition_id", "pair_id", "model_id", "model_version",
    "requested_irrelevant_ratio", "actual_irrelevant_ratio", "d_base",
    "d_irrelevant", "d_total", "transform_repeat", "n_query",
    "raw_path", "metrics_path", "cache_status",
)

PART_I2_FAILURE_COLUMNS = (
    "dataset_id", "dataset_name", "source", "condition_id", "pair_id",
    "model_id", "requested_irrelevant_ratio", "transform_repeat",
    "error_type", "error",
)


@dataclass(frozen=True)
class PartI2RunResult:
    data_run_id: str
    part_i2_run_id: str
    output_dir: Path
    num_files: int
    num_metric_rows: int
    num_failures: int
    cache_hits: int
    cache_misses: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _FeaturePlan:
    base_ids: tuple[int, ...]
    irrelevant_ids: tuple[int, ...]
    grids: Mapping[tuple[int, int], np.ndarray]


def _stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _permutation_seed(
    condition: PreparedDataset,
    reference: ConditionRef,
    column: int,
) -> int:
    """Pair the same logical feature permutation across doses and models."""

    return _stable_seed(
        condition.transform_seed,
        reference.dataset_id,
        reference.transform_repeat,
        int(condition.feature_role[column]),
        int(condition.feature_bank_id[column]),
        "query_permutation",
    )


def _quiet_model_config(config: ModelConfig) -> ModelConfig:
    """Disable backend progress rendering without changing inference numerics."""

    options = dict(config.options)
    if "verbose" in options:
        options["verbose"] = False
    return ModelConfig(model_id=config.model_id, options=options)


def _feature_role_name(source: str, role: int) -> str:
    if int(role) == int(FeatureRole.IRRELEVANT):
        return "irrelevant"
    return "active" if source == DatasetSource.SYNTHETIC.value else "original"


def _preparation_failures(
    data_index: Path,
    settings: PartI2Config,
    model_ids: Sequence[str],
    ratios: Sequence[float],
) -> list[dict[str, object]]:
    path = Path(data_index).with_name("skips.csv")
    if not path.is_file():
        return []
    selected_sources = {source.value for source in settings.sources}
    result: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SKIP_COLUMNS:
            raise ValueError("prepared-data skips columns do not match schema")
        for row in reader:
            if row["source"] not in selected_sources:
                continue
            for model_id in model_ids:
                for ratio in ratios:
                    result.append({
                        "dataset_id": row["dataset_id"],
                        "dataset_name": row["dataset_name"],
                        "source": row["source"],
                        "condition_id": "prepare-failed",
                        "pair_id": "",
                        "model_id": model_id,
                        "requested_irrelevant_ratio": ratio,
                        "transform_repeat": "",
                        "error_type": "DataPreparationError",
                        "error": row["reason"],
                    })
    return result


def _grid(values: np.ndarray, config: PartI2Config) -> np.ndarray:
    quantiles = np.linspace(
        config.pdp_grid.lower_quantile,
        config.pdp_grid.upper_quantile,
        config.pdp_grid.points,
    )
    result = np.asarray(np.quantile(np.asarray(values, dtype=np.float64), quantiles))
    if result.shape != (config.pdp_grid.points,) or not np.isfinite(result).all():
        raise ValueError("PDP grid contains invalid values")
    return result


def _model_predict(
    model: PredictiveModel,
    condition: PreparedDataset,
    X_query: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(
        model.fit_predict(condition.X_context, condition.y_context, X_query),
        dtype=np.float64,
    )
    if prediction.shape != (len(X_query),):
        raise RuntimeError(
            f"prediction shape {prediction.shape} does not match query rows {len(X_query)}"
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("counterfactual prediction contains non-finite values")
    return prediction


class _BatchPredictor:
    """Batch independent query variants after an empirical invariance check."""

    def __init__(
        self,
        model: PredictiveModel,
        condition: PreparedDataset,
        batch_size: int,
        validate: bool,
    ) -> None:
        self.model = model
        self.condition = condition
        self.batch_size = int(batch_size)
        self.validate = bool(validate)
        self.batch_safe: bool | None = None if self.batch_size > 1 else False

    def predict(self, variants: Sequence[np.ndarray]) -> np.ndarray:
        variants = tuple(np.asarray(item) for item in variants)
        if not variants:
            return np.empty((0, len(self.condition.X_query)), dtype=np.float64)
        n_query = len(self.condition.X_query)
        if any(item.shape != self.condition.X_query.shape for item in variants):
            raise ValueError("counterfactual query shape changed")
        if self.batch_safe is False:
            return np.stack([
                _model_predict(self.model, self.condition, item) for item in variants
            ])

        outputs: list[np.ndarray] = []
        for start in range(0, len(variants), self.batch_size):
            chunk = variants[start:start + self.batch_size]
            stacked = np.concatenate(chunk, axis=0)
            try:
                flat = _model_predict(self.model, self.condition, stacked)
            except RuntimeError:
                if len(chunk) == 1:
                    raise
                # Wide real-data leaves may exceed device memory only when
                # counterfactual queries are stacked. Fall back to the exact
                # one-variant semantics without failing the condition.
                self.batch_safe = False
                outputs.extend(
                    _model_predict(self.model, self.condition, item) for item in chunk
                )
                continue
            batched = flat.reshape(len(chunk), n_query)
            if self.batch_safe is None and len(chunk) > 1:
                if self.validate:
                    single = _model_predict(self.model, self.condition, chunk[0])
                    if not np.allclose(single, batched[0], rtol=1e-5, atol=1e-6):
                        self.batch_safe = False
                        outputs.append(single)
                        outputs.extend(
                            _model_predict(self.model, self.condition, item)
                            for item in chunk[1:]
                        )
                        continue
                self.batch_safe = True
            outputs.extend(batched)
        return np.stack(outputs)


def _feature_plan(
    references: Sequence[ConditionRef],
    store: HDF5DatasetStore,
    config: PartI2Config,
) -> _FeaturePlan:
    clean_ref = min(references, key=lambda item: item.d_irrelevant)
    max_ref = max(references, key=lambda item: item.d_irrelevant)
    if clean_ref.d_irrelevant != 0:
        raise ValueError("Part I.2 feature plan requires a clean ratio-zero condition")
    clean = store.read_condition(clean_ref)
    maximum = store.read_condition(max_ref)
    limit = config.pdp_max_features_per_role

    irrelevant_ids = tuple(sorted(
        int(value)
        for value in maximum.feature_bank_id[
            maximum.feature_role == FeatureRole.IRRELEVANT
        ]
    )[:limit])
    if maximum.source == DatasetSource.SYNTHETIC:
        base_ids = tuple(range(maximum.dimensions.d_base))
    else:
        matched_sources: list[int] = []
        for bank_id in irrelevant_ids:
            column = maximum.column_for_irrelevant_feature(bank_id)
            source_id = int(maximum.source_feature_index[column])
            if source_id >= 0 and source_id not in matched_sources:
                matched_sources.append(source_id)
        for source_id in range(maximum.dimensions.d_base):
            if len(matched_sources) >= limit:
                break
            if source_id not in matched_sources:
                matched_sources.append(source_id)
        base_ids = tuple(matched_sources[:limit])

    grids: dict[tuple[int, int], np.ndarray] = {}
    for feature_id in base_ids:
        column = clean.column_for_base_feature(feature_id)
        grids[(int(FeatureRole.BASE), feature_id)] = _grid(
            clean.X_context[:, column], config
        )
    for bank_id in irrelevant_ids:
        column = maximum.column_for_irrelevant_feature(bank_id)
        if maximum.source == DatasetSource.REAL:
            source_id = int(maximum.source_feature_index[column])
            grid_values = clean.X_context[:, clean.column_for_base_feature(source_id)]
        else:
            grid_values = maximum.X_context[:, column]
        grids[(int(FeatureRole.IRRELEVANT), bank_id)] = _grid(
            grid_values, config
        )
    return _FeaturePlan(base_ids=base_ids, irrelevant_ids=irrelevant_ids, grids=grids)


def _metric_row(
    reference: ConditionRef,
    *,
    model_id: str,
    model_version: str,
    metric: str,
    condition: PreparedDataset,
    column: int,
    n_grid_points: int,
    value: float,
) -> dict[str, object]:
    role = int(condition.feature_role[column])
    return {
        "schema_version": reference.schema_version,
        "dataset_id": reference.dataset_id,
        "dataset_name": reference.dataset_name,
        "source": reference.source,
        "condition_id": reference.condition_id,
        "pair_id": reference.pair_id,
        "model_id": model_id,
        "model_version": model_version,
        "requested_irrelevant_ratio": reference.requested_irrelevant_ratio,
        "actual_irrelevant_ratio": reference.actual_irrelevant_ratio,
        "d_base": reference.d_base,
        "d_irrelevant": reference.d_irrelevant,
        "d_total": reference.d_total,
        "transform_repeat": reference.transform_repeat,
        "metric": metric,
        "feature_role": _feature_role_name(reference.source, role),
        "physical_column": column,
        "source_feature_index": int(condition.source_feature_index[column]),
        "copy_block_id": int(condition.copy_block_id[column]),
        "feature_bank_id": int(condition.feature_bank_id[column]),
        "n_query": reference.n_query,
        "n_grid_points": n_grid_points,
        "value": float(value),
    }


def _write_leaf(
    path: Path,
    metrics_path: Path,
    *,
    reference: ConditionRef,
    condition: PreparedDataset,
    model_id: str,
    model_version: str,
    analysis_cache_key: str,
    baseline: np.ndarray,
    permutation_predictions: np.ndarray,
    pdp_columns: np.ndarray,
    pdp_grids: np.ndarray,
    pdp_predictions: np.ndarray,
    metric_rows: list[Mapping[str, object]],
    elapsed: float,
) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    # metadata.json is the commit marker. Removing a stale marker first keeps
    # an interrupted repair from making mixed old/new files appear complete.
    path.with_name("metadata.json").unlink(missing_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = reference.schema_version
            handle.attrs["algorithm"] = PART_I2_ALGORITHM_VERSION
            handle.attrs["analysis_cache_key"] = analysis_cache_key
            handle.attrs["condition_id"] = reference.condition_id
            handle.attrs["model_id"] = model_id
            handle.attrs["model_version"] = model_version
            handle.attrs["complete"] = True
            handle.attrs["elapsed_seconds"] = float(elapsed)
            handle.create_dataset("query_row_id", data=np.asarray(
                [str(value) for value in condition.query_row_id], dtype=object
            ), dtype=h5py.string_dtype("utf-8"))
            handle.create_dataset("y_true", data=condition.y_query)
            handle.create_dataset("baseline", data=baseline.astype(np.float32))
            perm = handle.create_group("permutation")
            perm.create_dataset("physical_column", data=np.arange(reference.d_total))
            perm.create_dataset(
                "predictions", data=permutation_predictions.astype(np.float32),
                compression="lzf", shuffle=True,
            )
            pdp = handle.create_group("pdp")
            pdp.create_dataset("physical_column", data=pdp_columns)
            pdp.create_dataset("grid", data=pdp_grids)
            pdp.create_dataset(
                "predictions", data=pdp_predictions.astype(np.float32),
                compression="lzf", shuffle=True,
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_csv(list(metric_rows), metrics_path, FEATURE_METRIC_COLUMNS)
    _atomic_json({
        "complete": True,
        "algorithm": PART_I2_ALGORITHM_VERSION,
        "analysis_cache_key": analysis_cache_key,
        "condition_id": reference.condition_id,
        "model_id": model_id,
        "model_version": model_version,
        "d_total": reference.d_total,
        "n_query": reference.n_query,
        "num_metric_rows": len(metric_rows),
        "num_pdp_features": len(pdp_columns),
        "elapsed_seconds": float(elapsed),
    }, path.with_name("metadata.json"))


def _valid_leaf(
    path: Path,
    metrics_path: Path,
    reference: ConditionRef,
    *,
    model_id: str,
    model_version: str,
    analysis_cache_key: str,
) -> tuple[bool, int]:
    metadata_path = path.with_name("metadata.json")
    if not path.is_file() or not metrics_path.is_file() or not metadata_path.is_file():
        return False, 0
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not all((
            metadata.get("complete") is True,
            metadata.get("algorithm") == PART_I2_ALGORITHM_VERSION,
            metadata.get("analysis_cache_key") == analysis_cache_key,
            metadata.get("condition_id") == reference.condition_id,
            metadata.get("model_id") == model_id,
            metadata.get("model_version") == model_version,
            int(metadata.get("d_total", -1)) == reference.d_total,
            int(metadata.get("n_query", -1)) == reference.n_query,
        )):
            return False, 0
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FEATURE_METRIC_COLUMNS:
                return False, 0
            rows = list(reader)
        if len(rows) != int(metadata["num_metric_rows"]):
            return False, 0
        import h5py
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                return False, 0
            if handle["baseline"].shape != (reference.n_query,):
                return False, 0
            if handle["permutation/predictions"].shape != (
                reference.d_total, reference.n_query
            ):
                return False, 0
            pdp_features = int(metadata["num_pdp_features"])
            pdp_shape = handle["pdp/predictions"].shape
            if pdp_shape != (
                pdp_features,
                handle["pdp/grid"].shape[1],
                reference.n_query,
            ):
                return False, 0
            if not np.isfinite(handle["baseline"][:]).all():
                return False, 0
        return True, len(rows)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False, 0


class DefaultPartI2Runner:
    """Run paired functional suppression over immutable prepared conditions."""

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
        dataset_ids: tuple[str, ...] | None = None,
        ratios: tuple[float, ...] | None = None,
    ) -> PartI2RunResult:
        run_started = perf_counter()
        settings = config.part_i2
        if settings is None:
            raise ValueError("config.part_i2 is required")
        all_references = tuple(self.store.read_index(Path(data_index)))
        sources = {item.value for item in settings.sources}
        selected_ratios = set(settings.ratios if ratios is None else ratios)
        unknown_ratios = sorted(selected_ratios - set(settings.ratios))
        if unknown_ratios:
            raise ValueError(f"Part I.2 ratios are not configured: {unknown_ratios}")
        selected_datasets = None if dataset_ids is None else set(dataset_ids)
        references = tuple(
            item for item in all_references
            if item.source in sources
            and item.requested_irrelevant_ratio in selected_ratios
            and (selected_datasets is None or item.dataset_id in selected_datasets)
        )
        if not references:
            raise ValueError("Part I.2 selection contains no prepared conditions")
        data_run_ids = {item.data_run_id for item in references}
        if len(data_run_ids) != 1:
            raise ValueError("prepared-data index must contain exactly one data_run_id")
        data_run_id = next(iter(data_run_ids))

        requested_models = (
            settings.default_models if model_ids is None else model_ids
        )
        unknown_models = sorted(set(requested_models) - set(settings.models))
        if unknown_models:
            raise ValueError(
                f"Part I.2 models are not configured: {unknown_models}"
            )
        selected = tuple(
            _quiet_model_config(item)
            for item in _selected_configs(config, requested_models)
        )
        plans = []
        for model_config in selected:
            model = self.model_factory.build(model_config)
            plans.append((
                model_config,
                model,
                _model_fingerprint(model, str(device)),
            ))
        analysis_payload = {
            "algorithm": PART_I2_ALGORITHM_VERSION,
            "permutation_method": settings.permutation_method,
            "permutation_repeats": settings.permutation_repeats,
            "pdp_max_features_per_role": settings.pdp_max_features_per_role,
            "pdp_grid": asdict(settings.pdp_grid),
            "pdp_reference_ratio": max(settings.ratios),
        }
        analysis_cache_key = _canonical_hash(analysis_payload, prefix="functional")
        run_payload = {
            "algorithm": PART_I2_ALGORITHM_VERSION,
            "data_run_id": data_run_id,
            "analysis_cache_key": analysis_cache_key,
            "model_cache_keys": [item[2].model_cache_key for item in plans],
            "sources": sorted(sources),
            "ratios": sorted(selected_ratios),
            "dataset_ids": sorted(selected_datasets) if selected_datasets is not None else None,
        }
        part_i2_run_id = _canonical_hash(run_payload, prefix="part-i2")
        root = Path(output_dir or config.artifact_root).resolve()
        stage_root = root / "part_i2"
        cache_root = stage_root / "cache" / analysis_cache_key
        run_dir = stage_root / "runs" / part_i2_run_id

        groups: dict[str, list[ConditionRef]] = {}
        selected_cache_keys = {item.data_cache_key for item in references}
        for reference in all_references:
            if reference.data_cache_key not in selected_cache_keys:
                continue
            groups.setdefault(reference.data_cache_key, []).append(reference)
        feature_plans: dict[str, _FeaturePlan] = {}
        index_rows: list[dict[str, object]] = []
        failures: list[dict[str, object]] = _preparation_failures(
            Path(data_index),
            settings,
            [item.model_id for item in selected],
            sorted(selected_ratios),
        )
        num_metric_rows = 0
        cache_hits = 0
        cache_misses = 0

        for model_config, model, fingerprint in plans:
            prepared = False
            prepare_error: Exception | None = None
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(
                    references,
                    desc=f"Part I.2 {model_config.model_id}",
                    unit="condition",
                    disable=len(references) < 10,
                )
            except ImportError:
                iterator = references
            for reference in iterator:
                leaf_dir = (
                    cache_root / fingerprint.model_cache_key
                    / reference.data_cache_key / reference.condition_id
                )
                raw_path = leaf_dir / "predictions.h5"
                metrics_path = leaf_dir / "feature_metrics.csv"
                valid, cached_rows = _valid_leaf(
                    raw_path,
                    metrics_path,
                    reference,
                    model_id=model_config.model_id,
                    model_version=fingerprint.model_version,
                    analysis_cache_key=analysis_cache_key,
                )
                if valid:
                    cache_status = "hit"
                    num_metric_rows += cached_rows
                    cache_hits += 1
                else:
                    if reference.d_total > model.descriptor.max_input_features:
                        failures.append(self._failure(
                            reference, model_config.model_id,
                            "InputFeatureLimitError",
                            f"d_total={reference.d_total} exceeds "
                            f"max_input_features={model.descriptor.max_input_features}",
                        ))
                        continue
                    if prepare_error is not None:
                        failures.append(self._failure(
                            reference, model_config.model_id,
                            type(prepare_error).__name__,
                            f"model_prepare_failed:{prepare_error}",
                        ))
                        continue
                    if not prepared:
                        try:
                            model.prepare(str(device))
                            if model.descriptor.model_version != fingerprint.model_version:
                                raise RuntimeError(
                                    "prepared model identity changed after cache key "
                                    f"construction: {fingerprint.model_version!r} -> "
                                    f"{model.descriptor.model_version!r}"
                                )
                            prepared = True
                        except Exception as error:
                            prepare_error = error
                            failures.append(self._failure(
                                reference, model_config.model_id,
                                type(error).__name__,
                                f"model_prepare_failed:{error}",
                            ))
                            continue
                    try:
                        condition = self.store.read_condition(reference)
                        plan = feature_plans.get(reference.data_cache_key)
                        if plan is None:
                            plan = _feature_plan(
                                groups[reference.data_cache_key], self.store, settings
                            )
                            feature_plans[reference.data_cache_key] = plan
                        started = perf_counter()
                        rows = self._compute_leaf(
                            model=model,
                            condition=condition,
                            reference=reference,
                            settings=settings,
                            plan=plan,
                            model_id=model_config.model_id,
                            model_version=fingerprint.model_version,
                            raw_path=raw_path,
                            metrics_path=metrics_path,
                            analysis_cache_key=analysis_cache_key,
                        )
                        elapsed = perf_counter() - started
                        # _compute_leaf records inference time in the HDF5 leaf;
                        # this assignment only keeps the progress accounting local.
                        _ = elapsed
                        num_metric_rows += rows
                        cache_misses += 1
                        cache_status = "miss"
                    except Exception as error:
                        failures.append(self._failure(
                            reference, model_config.model_id,
                            type(error).__name__, str(error),
                        ))
                        continue

                index_rows.append({
                    "schema_version": reference.schema_version,
                    "data_run_id": reference.data_run_id,
                    "part_i2_run_id": part_i2_run_id,
                    "analysis_cache_key": analysis_cache_key,
                    "model_cache_key": fingerprint.model_cache_key,
                    "data_cache_key": reference.data_cache_key,
                    "dataset_id": reference.dataset_id,
                    "dataset_name": reference.dataset_name,
                    "source": reference.source,
                    "condition_id": reference.condition_id,
                    "pair_id": reference.pair_id,
                    "model_id": model_config.model_id,
                    "model_version": fingerprint.model_version,
                    "requested_irrelevant_ratio": reference.requested_irrelevant_ratio,
                    "actual_irrelevant_ratio": reference.actual_irrelevant_ratio,
                    "d_base": reference.d_base,
                    "d_irrelevant": reference.d_irrelevant,
                    "d_total": reference.d_total,
                    "transform_repeat": reference.transform_repeat,
                    "n_query": reference.n_query,
                    "raw_path": raw_path.relative_to(stage_root).as_posix(),
                    "metrics_path": metrics_path.relative_to(stage_root).as_posix(),
                    "cache_status": cache_status,
                })

        index_rows.sort(key=lambda row: (
            str(row["source"]), str(row["dataset_id"]),
            int(row["transform_repeat"]), float(row["requested_irrelevant_ratio"]),
            str(row["model_id"]),
        ))
        failures.sort(key=lambda row: (
            str(row["source"]), str(row["dataset_id"]),
            float(row["requested_irrelevant_ratio"]), str(row["model_id"]),
        ))
        _atomic_csv(index_rows, run_dir / "index.csv", PART_I2_INDEX_COLUMNS)
        _atomic_csv(failures, run_dir / "failures.csv", PART_I2_FAILURE_COLUMNS)
        elapsed_seconds = perf_counter() - run_started
        _atomic_json({
            "experiment_id": config.experiment_id,
            "schema_version": config.schema_version,
            "data_index_name": Path(data_index).name,
            "device": str(device),
            "part_i2": asdict(settings),
            "selected_model_ids": [item.model_id for item in selected],
            "model_fingerprints": [item[2].payload for item in plans],
        }, run_dir / "config.json")
        _atomic_json({
            "complete": True,
            "algorithm": PART_I2_ALGORITHM_VERSION,
            "part_i2_run_id": part_i2_run_id,
            "data_run_id": data_run_id,
            "num_files": len(index_rows),
            "num_metric_rows": num_metric_rows,
            "num_failures": len(failures),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "elapsed_seconds": elapsed_seconds,
        }, run_dir / "manifest.json")
        return PartI2RunResult(
            data_run_id=data_run_id,
            part_i2_run_id=part_i2_run_id,
            output_dir=run_dir,
            num_files=len(index_rows),
            num_metric_rows=num_metric_rows,
            num_failures=len(failures),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            elapsed_seconds=elapsed_seconds,
        )

    def _compute_leaf(
        self,
        *,
        model: PredictiveModel,
        condition: PreparedDataset,
        reference: ConditionRef,
        settings: PartI2Config,
        plan: _FeaturePlan,
        model_id: str,
        model_version: str,
        raw_path: Path,
        metrics_path: Path,
        analysis_cache_key: str,
    ) -> int:
        started = perf_counter()
        baseline = _model_predict(model, condition, condition.X_query)
        y_scale = float(np.std(condition.y_query, ddof=0))
        if not math.isfinite(y_scale) or y_scale <= 0.0:
            raise ValueError("y_query must have positive finite standard deviation")
        predictor = _BatchPredictor(
            model,
            condition,
            settings.counterfactual_batch_size,
            settings.validate_batch_invariance,
        )
        metric_rows: list[Mapping[str, object]] = []
        permutation_predictions = np.empty(
            (reference.d_total, reference.n_query), dtype=np.float32
        )
        for start in range(0, reference.d_total, settings.counterfactual_batch_size):
            columns = range(
                start,
                min(start + settings.counterfactual_batch_size, reference.d_total),
            )
            variants = []
            for column in columns:
                variant = condition.X_query.copy()
                rng = np.random.default_rng(
                    _permutation_seed(condition, reference, column)
                )
                variant[:, column] = variant[rng.permutation(reference.n_query), column]
                variants.append(variant)
            predictions = predictor.predict(variants)
            for column, prediction in zip(columns, predictions):
                permutation_predictions[column] = prediction
                shift = float(np.sqrt(np.mean(np.square(prediction - baseline))) / y_scale)
                metric_rows.append(_metric_row(
                    reference,
                    model_id=model_id,
                    model_version=model_version,
                    metric="permutation_shift",
                    condition=condition,
                    column=column,
                    n_grid_points=0,
                    value=shift,
                ))

        pdp_columns: list[int] = []
        pdp_grids: list[np.ndarray] = []
        pdp_predictions: list[np.ndarray] = []
        feature_keys = [
            (int(FeatureRole.BASE), feature_id) for feature_id in plan.base_ids
        ] + [
            (int(FeatureRole.IRRELEVANT), feature_id)
            for feature_id in plan.irrelevant_ids
        ]
        for role, feature_id in feature_keys:
            try:
                column = (
                    condition.column_for_base_feature(feature_id)
                    if role == int(FeatureRole.BASE)
                    else condition.column_for_irrelevant_feature(feature_id)
                )
            except ValueError:
                continue
            grid_values = np.asarray(plan.grids[(role, feature_id)], dtype=np.float64)
            variants = []
            for value in grid_values:
                variant = condition.X_query.copy()
                variant[:, column] = value
                variants.append(variant)
            predictions = predictor.predict(variants)
            curve = np.mean(predictions, axis=1)
            variation = float(np.std(curve, ddof=0) / y_scale)
            pdp_columns.append(column)
            pdp_grids.append(grid_values)
            pdp_predictions.append(predictions)
            metric_rows.append(_metric_row(
                reference,
                model_id=model_id,
                model_version=model_version,
                metric="pdp_variation",
                condition=condition,
                column=column,
                n_grid_points=len(grid_values),
                value=variation,
            ))

        grid_points = settings.pdp_grid.points
        pdp_columns_array = np.asarray(pdp_columns, dtype=np.int32)
        pdp_grids_array = (
            np.stack(pdp_grids).astype(np.float64)
            if pdp_grids else np.empty((0, grid_points), dtype=np.float64)
        )
        pdp_predictions_array = (
            np.stack(pdp_predictions).astype(np.float32)
            if pdp_predictions
            else np.empty((0, grid_points, reference.n_query), dtype=np.float32)
        )
        _write_leaf(
            raw_path,
            metrics_path,
            reference=reference,
            condition=condition,
            model_id=model_id,
            model_version=model_version,
            analysis_cache_key=analysis_cache_key,
            baseline=baseline,
            permutation_predictions=permutation_predictions,
            pdp_columns=pdp_columns_array,
            pdp_grids=pdp_grids_array,
            pdp_predictions=pdp_predictions_array,
            metric_rows=metric_rows,
            elapsed=perf_counter() - started,
        )
        return len(metric_rows)

    @staticmethod
    def _failure(
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
            "transform_repeat": reference.transform_repeat,
            "error_type": error_type,
            "error": error,
        }
__all__ = [
    "DefaultPartI2Runner",
    "FEATURE_METRIC_COLUMNS",
    "PART_I2_ALGORITHM_VERSION",
    "PART_I2_FAILURE_COLUMNS",
    "PART_I2_INDEX_COLUMNS",
    "PartI2RunResult",
]
