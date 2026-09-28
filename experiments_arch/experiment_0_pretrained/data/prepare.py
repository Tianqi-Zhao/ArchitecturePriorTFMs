"""Concrete Part I.1 prepared-data pipeline."""

from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

from experiment_core.openml import (
    DEFAULT_PRETRAINED_REAL_REGRESSION_DATASET_IDS,
    OpenMLConfig,
)
from experiments_arch.experiment_0_pretrained.config import ExperimentConfig, SourceConfig
from experiments_arch.experiment_0_pretrained.data.base import (
    BaseDataset,
    DatasetPreparer,
    DatasetSource,
    PreparationResult,
)
from experiments_arch.experiment_0_pretrained.data.irrelevant import (
    DefaultIrrelevantFeatureAdder,
    GrowingWidthStrategy,
    RoundingMode,
    _derived_seed,
)
from experiments_arch.experiment_0_pretrained.data.real import OpenMLGenerator
from experiments_arch.experiment_0_pretrained.data.store import (
    HDF5DatasetStore,
    SKIP_COLUMNS,
)
from experiments_arch.experiment_0_pretrained.data.synthetic import (
    FriedmanConfig,
    FriedmanGenerator,
)


PREPARATION_ALGORITHM_VERSION = "part-i-1-ratio-bank-v1"


def _canonical_hash(value: object, *, prefix: str) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


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


def _atomic_skips(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SKIP_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ratio_token(value: float) -> str:
    return format(float(value), ".12g").replace(".", "p")


class DefaultDatasetPreparer(DatasetPreparer):
    """Generate, expand, and cache Part I.1 datasets independently by dataset."""

    def __init__(self, store: HDF5DatasetStore | None = None) -> None:
        self.store = store or HDF5DatasetStore()
        self.adder = DefaultIrrelevantFeatureAdder()

    def prepare(self, config: ExperimentConfig, output_dir: Path) -> PreparationResult:
        root = Path(output_dir or config.artifact_root).resolve()
        run_payload = {
            "algorithm": PREPARATION_ALGORITHM_VERSION,
            "experiment_id": config.experiment_id,
            "schema_version": config.schema_version,
            "run_seed": config.run_seed,
            "transform_repeats": config.transform_repeats,
            "sources": [self._source_payload(source) for source in config.sources],
        }
        data_run_id = _canonical_hash(run_payload, prefix="data")
        run_dir = root / "prepared_runs" / data_run_id
        cache_root = root / "prepared_cache"
        references = []
        skips: list[dict[str, object]] = []
        successful_datasets: set[tuple[str, str]] = set()
        cache_hits = 0
        cache_misses = 0

        for source_config in config.sources:
            strategy = GrowingWidthStrategy(RoundingMode(source_config.rounding))
            for dataset_id, loader, dataset_payload in self._descriptors(
                source_config, config.run_seed
            ):
                repeat_entries = []
                dataset_references = []
                missing = []
                for repeat in range(config.transform_repeats):
                    cache_payload = {
                        "algorithm": PREPARATION_ALGORITHM_VERSION,
                        "source": self._source_payload(source_config, dataset_local=True),
                        "dataset": dataset_payload,
                        "dataset_id": dataset_id,
                        "repeat": repeat,
                        "run_seed": config.run_seed,
                    }
                    cache_key = _canonical_hash(cache_payload, prefix="cache")
                    cache_dir = (
                        cache_root / source_config.source.value / dataset_id
                        / f"repeat-{repeat:03d}" / cache_key
                    )
                    if (cache_dir / "metadata.json").exists() and (cache_dir / "data.h5").exists():
                        try:
                            stored = self.store.load_cached_dataset(cache_dir, data_run_id)
                        except (OSError, ValueError, KeyError, TypeError):
                            missing.append((repeat, cache_key, cache_dir))
                        else:
                            dataset_references.extend(stored.condition_refs)
                            repeat_entries.append({"repeat": repeat, "cache": "hit",
                                                   "data_cache_key": cache_key})
                            cache_hits += 1
                    else:
                        missing.append((repeat, cache_key, cache_dir))

                if missing:
                    try:
                        base = loader()
                        for repeat, cache_key, cache_dir in missing:
                            transform_seed = _derived_seed(
                                config.run_seed, source_config.source_id, dataset_id, repeat
                            )
                            dimensions = [strategy.resolve(base, ratio)
                                          for ratio in source_config.ratios]
                            bank = self.adder.build_bank(
                                base,
                                max(item.d_irrelevant for item in dimensions),
                                repeat=repeat,
                                seed=transform_seed,
                            )
                            pair_id = (
                                f"{source_config.source_id}:{dataset_id}:repeat-{repeat:03d}"
                            )
                            conditions = []
                            for item in dimensions:
                                condition = self.adder.materialize(
                                    base,
                                    item,
                                    bank,
                                    condition_id=f"ratio-{_ratio_token(item.requested_irrelevant_ratio)}",
                                    pair_id=pair_id,
                                    repeat=repeat,
                                    seed=transform_seed,
                                )
                                conditions.append(replace(condition, metadata={
                                    **dict(condition.metadata),
                                    "base": dict(base.metadata),
                                    "source_id": source_config.source_id,
                                    "data_run_id": data_run_id,
                                    "data_cache_key": cache_key,
                                }))
                            stored = self.store.write_dataset(conditions, cache_dir)
                            dataset_references.extend(stored.condition_refs)
                            repeat_entries.append({"repeat": repeat, "cache": "miss",
                                                   "data_cache_key": cache_key})
                            cache_misses += 1
                    except Exception as error:
                        skips.append({
                            "dataset_id": dataset_id,
                            "dataset_name": dataset_id,
                            "source": source_config.source.value,
                            "requested_irrelevant_ratio": "",
                            "d_base": "",
                            "d_irrelevant": "",
                            "d_total": "",
                            "max_input_features": config.max_input_features,
                            "reason": f"prepare_failed:{type(error).__name__}:{error}",
                        })
                        continue

                if len(repeat_entries) == config.transform_repeats:
                    references.extend(dataset_references)
                    successful_datasets.add((source_config.source.value, dataset_id))

        references.sort(key=lambda ref: (
            ref.source, ref.dataset_id, ref.transform_repeat, ref.requested_irrelevant_ratio
        ))
        index_path = self.store.write_index(references, run_dir / "index.csv")
        skips_path = run_dir / "skips.csv"
        _atomic_skips(skips, skips_path)
        _atomic_json(asdict(config), run_dir / "config.json")
        _atomic_json({
            "complete": True,
            "data_run_id": data_run_id,
            "run_config": run_payload,
            "num_datasets": len(successful_datasets),
            "num_conditions": len(references),
            "num_skips": len(skips),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "index_path": str(index_path),
        }, run_dir / "manifest.json")
        return PreparationResult(
            data_run_id=data_run_id,
            output_dir=run_dir,
            index_path=index_path,
            skips_path=skips_path,
            num_datasets=len(successful_datasets),
            num_conditions=len(references),
            num_skips=len(skips),
        )

    @staticmethod
    def _source_payload(source: SourceConfig, dataset_local: bool = False) -> dict[str, object]:
        parameters = dict(source.parameters)
        if dataset_local:
            parameters.pop("dataset_ids", None)
            parameters.pop("num_tasks", None)
            parameters.pop("cache_dir", None)
        return {
            "source": source.source.value,
            "source_id": source.source_id,
            "generator": source.generator,
            "transform": source.transform,
            "ratios": list(source.ratios),
            "rounding": source.rounding,
            "parameters": parameters,
        }

    @staticmethod
    def _descriptors(
        source: SourceConfig, run_seed: int
    ) -> Iterable[tuple[str, Callable[[], BaseDataset], dict[str, object]]]:
        parameters = dict(source.parameters)
        if source.source is DatasetSource.SYNTHETIC:
            if source.generator.lower() not in {"friedman", "friedman1", "friedman_1"}:
                raise ValueError(f"unsupported synthetic generator: {source.generator}")
            config = FriedmanConfig(
                num_tasks=int(parameters.get("num_tasks", 200)),
                n_context=int(parameters.get("n_context", 800)),
                n_query=int(parameters.get("n_query", 200)),
                relevant_features=int(parameters.get("relevant_features", 5)),
                noise_std=float(parameters.get("noise_std", 1.0)),
                seed=int(parameters.get("seed", run_seed)),
            )
            generator = FriedmanGenerator(config)
            for task_index in range(config.num_tasks):
                dataset_id = f"friedman1-task-{task_index:03d}"
                payload = {**asdict(config), "task_index": task_index}
                payload.pop("num_tasks")
                yield dataset_id, (lambda index=task_index: generator.generate_task(index)), payload
            return

        if source.generator.lower() not in {"openml", "openml_regression"}:
            raise ValueError(f"unsupported real generator: {source.generator}")
        dataset_ids = tuple(int(value) for value in parameters.get(
            "dataset_ids", DEFAULT_PRETRAINED_REAL_REGRESSION_DATASET_IDS
        ))
        for openml_id in dataset_ids:
            openml_config = OpenMLConfig(
                dataset_ids=(openml_id,),
                max_rows=int(parameters.get("max_rows", 1000)),
                context_fraction=float(parameters.get("context_fraction", 0.8)),
                cache_dir=str(parameters.get("cache_dir", "")),
                seed=int(parameters.get("seed", run_seed)),
            )
            dataset_id = f"openml-{openml_id}"
            payload = {**asdict(openml_config), "cache_dir": ""}
            yield dataset_id, (
                lambda cfg=openml_config: next(iter(OpenMLGenerator(cfg).generate_base()))
            ), payload
