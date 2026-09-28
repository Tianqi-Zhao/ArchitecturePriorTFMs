"""Prepared-data storage with one maximum expansion and nested column views."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Literal, Sequence
from uuid import uuid4

import numpy as np

from experiments_arch.experiment_0_pretrained.data.base import (
    DatasetSource,
    DimensionInfo,
    PreparedDataset,
)


DATA_SCHEMA_VERSION = 1
HDF5_ARRAYS = (
    "X_context", "y_context", "X_query", "y_query",
    "context_row_id", "query_row_id", "feature_role",
    "source_feature_index", "copy_block_id", "feature_bank_id", "column_index",
)
DATASET_METADATA_FIELDS = (
    "schema_version", "complete", "data_cache_key", "dataset_id", "dataset_name",
    "source", "source_n", "source_d", "saved_n", "max_d_irrelevant", "conditions",
)
CONDITION_METADATA_FIELDS = (
    "condition_id", "pair_id", "requested_irrelevant_ratio",
    "actual_irrelevant_ratio", "ratio_denominator", "rounding", "d_base",
    "d_irrelevant", "d_total", "relevant_feature_count", "n_context", "n_query",
    "transform", "transform_repeat", "transform_seed", "column_layout",
)
INDEX_COLUMNS = (
    "schema_version", "data_run_id", "data_cache_key", "dataset_id", "dataset_name",
    "source", "data_path", "condition_id", "pair_id",
    "requested_irrelevant_ratio", "actual_irrelevant_ratio", "ratio_denominator",
    "rounding", "d_base", "d_irrelevant", "d_total", "relevant_feature_count",
    "n_context", "n_query", "transform", "transform_repeat", "transform_seed",
    "column_layout",
)
SKIP_COLUMNS = (
    "dataset_id", "dataset_name", "source", "requested_irrelevant_ratio", "d_base",
    "d_irrelevant", "d_total", "max_input_features", "reason",
)


@dataclass(frozen=True)
class ConditionRef:
    schema_version: int
    data_run_id: str
    data_cache_key: str
    dataset_id: str
    dataset_name: str
    source: str
    data_path: Path
    condition_id: str
    pair_id: str
    requested_irrelevant_ratio: float
    actual_irrelevant_ratio: float
    ratio_denominator: int
    rounding: str
    d_base: int
    d_irrelevant: int
    d_total: int
    relevant_feature_count: int | None
    n_context: int
    n_query: int
    transform: str
    transform_repeat: int
    transform_seed: int
    column_layout: Literal["shuffle"]


@dataclass(frozen=True)
class StoredDataset:
    dataset_id: str
    data_path: Path
    metadata_path: Path
    condition_refs: tuple[ConditionRef, ...]


class DatasetStore(ABC):
    @abstractmethod
    def write_dataset(self, conditions: Sequence[PreparedDataset], output_dir: Path) -> StoredDataset:
        """Atomically write all conditions for exactly one logical dataset."""

    @abstractmethod
    def read_condition(self, reference: ConditionRef) -> PreparedDataset:
        """Load one condition without loading unrelated datasets."""

    @abstractmethod
    def write_index(self, references: Iterable[ConditionRef], path: Path) -> Path:
        """Write ``INDEX_COLUMNS`` in their declared order."""

    @abstractmethod
    def read_index(self, path: Path) -> Iterable[ConditionRef]:
        """Read and validate a prepared-data index."""


def _condition_metadata(condition: PreparedDataset) -> dict[str, object]:
    d = condition.dimensions
    return {
        "condition_id": condition.condition_id,
        "pair_id": condition.pair_id,
        "requested_irrelevant_ratio": d.requested_irrelevant_ratio,
        "actual_irrelevant_ratio": d.actual_irrelevant_ratio,
        "ratio_denominator": d.ratio_denominator,
        "rounding": d.rounding,
        "d_base": d.d_base,
        "d_irrelevant": d.d_irrelevant,
        "d_total": d.d_total,
        "relevant_feature_count": d.relevant_feature_count,
        "n_context": len(condition.X_context),
        "n_query": len(condition.X_query),
        "transform": condition.transform_name,
        "transform_repeat": condition.transform_repeat,
        "transform_seed": condition.transform_seed,
        "column_layout": condition.column_layout,
    }


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return value


def _feature_keys(condition: PreparedDataset) -> list[tuple[int, int]]:
    return [(int(role), int(bank_id)) for role, bank_id in zip(
        condition.feature_role, condition.feature_bank_id
    )]


def _hdf5_row_ids(values: np.ndarray):
    array = np.asarray(values)
    if all(isinstance(value, (int, np.integer)) for value in array.tolist()):
        return array.astype(np.int64), None
    import h5py
    return np.asarray([str(value) for value in array], dtype=object), h5py.string_dtype()


def _decoded_row_ids(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in {"S", "O"}:
        return np.asarray([
            value.decode("utf-8") if isinstance(value, bytes) else value for value in array
        ], dtype=object)
    return array


class HDF5DatasetStore(DatasetStore):
    """Store the maximum expanded matrix once and each dose as a column view."""

    def write_dataset(self, conditions: Sequence[PreparedDataset], output_dir: Path) -> StoredDataset:
        if not conditions:
            raise ValueError("write_dataset requires at least one condition")
        conditions = tuple(conditions)
        first = conditions[0]
        identity = (first.dataset_id, first.source, first.transform_repeat)
        if any((c.dataset_id, c.source, c.transform_repeat) != identity for c in conditions):
            raise ValueError("all conditions must describe one dataset and repeat")
        if len({c.condition_id for c in conditions}) != len(conditions):
            raise ValueError("condition ids must be unique")
        data_run_id = str(first.metadata.get("data_run_id", ""))
        data_cache_key = str(first.metadata.get("data_cache_key", ""))
        if not data_run_id or not data_cache_key:
            raise ValueError("condition metadata must contain data_run_id and data_cache_key")
        if any(str(c.metadata.get("data_cache_key", "")) != data_cache_key for c in conditions):
            raise ValueError("conditions disagree on data_cache_key")

        maximum = max(conditions, key=lambda c: c.dimensions.d_irrelevant)
        maximum_keys = _feature_keys(maximum)
        if len(maximum_keys) != len(set(maximum_keys)):
            raise ValueError("maximum condition feature provenance is not unique")
        lookup = {key: index for index, key in enumerate(maximum_keys)}
        views: dict[str, np.ndarray] = {}
        for condition in conditions:
            if not all((
                np.array_equal(condition.y_context, maximum.y_context),
                np.array_equal(condition.y_query, maximum.y_query),
                np.array_equal(condition.context_row_id, maximum.context_row_id),
                np.array_equal(condition.query_row_id, maximum.query_row_id),
            )):
                raise ValueError("paired conditions must share targets and row ids")
            try:
                view = np.asarray([lookup[key] for key in _feature_keys(condition)], dtype=np.int32)
            except KeyError as error:
                raise ValueError("a condition is not nested in the maximum bank") from error
            if not all((
                np.array_equal(maximum.X_context[:, view], condition.X_context),
                np.array_equal(maximum.X_query[:, view], condition.X_query),
            )):
                raise ValueError("condition values are not an exact maximum-bank view")
            views[condition.condition_id] = view

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = (output_dir / "data.h5").resolve()
        metadata_path = (output_dir / "metadata.json").resolve()
        if metadata_path.exists() and data_path.exists():
            existing = _read_json(metadata_path)
            if existing.get("data_cache_key") != data_cache_key:
                raise ValueError(f"incompatible prepared-data cache already exists: {output_dir}")
            try:
                cached = self.load_cached_dataset(output_dir, data_run_id)
                cached_ids = {ref.condition_id for ref in cached.condition_refs}
                requested_ids = {condition.condition_id for condition in conditions}
                if cached_ids != requested_ids:
                    raise ValueError("cached conditions do not match the requested conditions")
            except (OSError, ValueError, KeyError, TypeError):
                # An incomplete/corrupt leaf is rebuilt below using atomic file
                # replacement; valid immutable leaves return immediately.
                pass
            else:
                return StoredDataset(first.dataset_id, data_path, metadata_path,
                    self._references(conditions, data_path, data_run_id, data_cache_key))

        import h5py
        temporary = data_path.with_name(f".{data_path.name}.{uuid4().hex}.partial")
        try:
            with h5py.File(temporary, "w") as handle:
                shared = handle.create_group("shared")
                shared.create_dataset("y_context", data=maximum.y_context)
                shared.create_dataset("y_query", data=maximum.y_query)
                for name, values in (("context_row_id", maximum.context_row_id),
                                     ("query_row_id", maximum.query_row_id)):
                    data, dtype = _hdf5_row_ids(values)
                    shared.create_dataset(name, data=data, dtype=dtype)
                max_group = handle.create_group("max_expansion")
                for name in ("X_context", "X_query", "feature_role", "source_feature_index",
                             "copy_block_id", "feature_bank_id"):
                    max_group.create_dataset(name, data=getattr(maximum, name))
                condition_group = handle.create_group("conditions")
                for condition in conditions:
                    group = condition_group.create_group(condition.condition_id)
                    group.create_dataset("column_index", data=views[condition.condition_id])
                handle.attrs["schema_version"] = DATA_SCHEMA_VERSION
                handle.attrs["data_cache_key"] = data_cache_key
            os.replace(temporary, data_path)
        finally:
            temporary.unlink(missing_ok=True)

        condition_metadata = []
        for condition in conditions:
            item = _condition_metadata(condition)
            item["column_index_path"] = f"conditions/{condition.condition_id}/column_index"
            condition_metadata.append(item)
        _atomic_json({
            "schema_version": DATA_SCHEMA_VERSION,
            "complete": True,
            "data_cache_key": data_cache_key,
            "dataset_id": first.dataset_id,
            "dataset_name": first.dataset_name,
            "source": first.source.value,
            "source_n": len(first.X_context) + len(first.X_query),
            "source_d": first.dimensions.d_base,
            "saved_n": len(maximum.X_context) + len(maximum.X_query),
            "max_d_irrelevant": maximum.dimensions.d_irrelevant,
            "conditions": condition_metadata,
        }, metadata_path)
        return StoredDataset(first.dataset_id, data_path, metadata_path,
            self._references(conditions, data_path, data_run_id, data_cache_key))

    @staticmethod
    def _references(conditions: Sequence[PreparedDataset], data_path: Path,
                    data_run_id: str, data_cache_key: str) -> tuple[ConditionRef, ...]:
        return tuple(ConditionRef(
            schema_version=DATA_SCHEMA_VERSION,
            data_run_id=data_run_id,
            data_cache_key=data_cache_key,
            dataset_id=condition.dataset_id,
            dataset_name=condition.dataset_name,
            source=condition.source.value,
            data_path=data_path,
            **_condition_metadata(condition),
        ) for condition in conditions)

    def load_cached_dataset(self, output_dir: Path, data_run_id: str) -> StoredDataset:
        """Return references for a complete cache without regenerating its dataset."""

        output_dir = Path(output_dir)
        data_path = (output_dir / "data.h5").resolve()
        metadata_path = (output_dir / "metadata.json").resolve()
        if not data_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(output_dir)
        metadata = _read_json(metadata_path)
        if metadata.get("complete") is not True:
            raise ValueError(f"prepared-data cache is incomplete: {output_dir}")
        if int(metadata.get("schema_version", -1)) != DATA_SCHEMA_VERSION:
            raise ValueError(f"unsupported prepared-data cache schema: {output_dir}")
        cache_key = str(metadata["data_cache_key"])
        condition_items = list(metadata["conditions"])
        import h5py
        with h5py.File(data_path, "r") as handle:
            if int(handle.attrs.get("schema_version", -1)) != DATA_SCHEMA_VERSION:
                raise ValueError(f"HDF5 schema does not match metadata: {output_dir}")
            if str(handle.attrs.get("data_cache_key", "")) != cache_key:
                raise ValueError(f"HDF5 cache key does not match metadata: {output_dir}")
            for required in ("shared", "max_expansion", "conditions"):
                if required not in handle:
                    raise ValueError(f"HDF5 cache is missing {required}: {output_dir}")
            for item in condition_items:
                condition_id = item["condition_id"]
                if f"conditions/{condition_id}/column_index" not in handle:
                    raise ValueError(
                        f"HDF5 cache is missing condition {condition_id}: {output_dir}"
                    )
        references = []
        for item in condition_items:
            values = {field: item[field] for field in CONDITION_METADATA_FIELDS}
            references.append(ConditionRef(
                schema_version=DATA_SCHEMA_VERSION,
                data_run_id=data_run_id,
                data_cache_key=cache_key,
                dataset_id=str(metadata["dataset_id"]),
                dataset_name=str(metadata["dataset_name"]),
                source=str(metadata["source"]),
                data_path=data_path,
                **values,
            ))
        return StoredDataset(
            dataset_id=str(metadata["dataset_id"]),
            data_path=data_path,
            metadata_path=metadata_path,
            condition_refs=tuple(references),
        )

    def read_condition(self, reference: ConditionRef) -> PreparedDataset:
        if reference.schema_version != DATA_SCHEMA_VERSION:
            raise ValueError("unsupported prepared-data schema version")
        data_path = Path(reference.data_path)
        metadata = _read_json(data_path.with_name("metadata.json"))
        if metadata.get("complete") is not True:
            raise ValueError("prepared-data cache is incomplete")
        if metadata.get("data_cache_key") != reference.data_cache_key:
            raise ValueError("index data_cache_key does not match dataset metadata")
        import h5py
        with h5py.File(data_path, "r") as handle:
            group_path = f"conditions/{reference.condition_id}"
            if group_path not in handle:
                raise ValueError(f"unknown condition: {reference.condition_id}")
            view = handle[f"{group_path}/column_index"][:]
            maximum, shared = handle["max_expansion"], handle["shared"]
            d = DimensionInfo(
                requested_irrelevant_ratio=reference.requested_irrelevant_ratio,
                actual_irrelevant_ratio=reference.actual_irrelevant_ratio,
                ratio_denominator=reference.ratio_denominator,
                d_base=reference.d_base,
                d_irrelevant=reference.d_irrelevant,
                d_total=reference.d_total,
                relevant_feature_count=reference.relevant_feature_count,
                rounding=reference.rounding,
            )
            return PreparedDataset(
                dataset_id=reference.dataset_id, dataset_name=reference.dataset_name,
                source=DatasetSource(reference.source), condition_id=reference.condition_id,
                pair_id=reference.pair_id, dimensions=d, transform_name=reference.transform,
                transform_repeat=reference.transform_repeat, transform_seed=reference.transform_seed,
                column_layout=reference.column_layout,
                X_context=maximum["X_context"][:, view], y_context=shared["y_context"][:],
                X_query=maximum["X_query"][:, view], y_query=shared["y_query"][:],
                context_row_id=_decoded_row_ids(shared["context_row_id"][:]),
                query_row_id=_decoded_row_ids(shared["query_row_id"][:]),
                feature_role=maximum["feature_role"][:][view],
                source_feature_index=maximum["source_feature_index"][:][view],
                copy_block_id=maximum["copy_block_id"][:][view],
                feature_bank_id=maximum["feature_bank_id"][:][view],
                metadata={"data_run_id": reference.data_run_id,
                          "data_cache_key": reference.data_cache_key,
                          "loaded_from": str(data_path)},
            )

    def write_index(self, references: Iterable[ConditionRef], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
                writer.writeheader()
                for reference in references:
                    row = asdict(reference)
                    row["data_path"] = str(reference.data_path)
                    row["relevant_feature_count"] = ("" if reference.relevant_feature_count is None
                                                     else reference.relevant_feature_count)
                    writer.writerow(row)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def read_index(self, path: Path) -> Iterable[ConditionRef]:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != INDEX_COLUMNS:
                raise ValueError("prepared-data index columns do not match the schema")
            for row in reader:
                relevant = row["relevant_feature_count"]
                yield ConditionRef(
                    schema_version=int(row["schema_version"]), data_run_id=row["data_run_id"],
                    data_cache_key=row["data_cache_key"], dataset_id=row["dataset_id"],
                    dataset_name=row["dataset_name"], source=row["source"],
                    data_path=Path(row["data_path"]), condition_id=row["condition_id"],
                    pair_id=row["pair_id"],
                    requested_irrelevant_ratio=float(row["requested_irrelevant_ratio"]),
                    actual_irrelevant_ratio=float(row["actual_irrelevant_ratio"]),
                    ratio_denominator=int(row["ratio_denominator"]), rounding=row["rounding"],
                    d_base=int(row["d_base"]), d_irrelevant=int(row["d_irrelevant"]),
                    d_total=int(row["d_total"]),
                    relevant_feature_count=None if relevant == "" else int(relevant),
                    n_context=int(row["n_context"]), n_query=int(row["n_query"]),
                    transform=row["transform"], transform_repeat=int(row["transform_repeat"]),
                    transform_seed=int(row["transform_seed"]), column_layout=row["column_layout"],
                )
