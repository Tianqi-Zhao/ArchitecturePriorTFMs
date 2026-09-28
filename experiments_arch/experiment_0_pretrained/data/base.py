"""Domain objects and abstract data-generation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

import numpy as np

if TYPE_CHECKING:
    from experiments_arch.experiment_0_pretrained.config import ExperimentConfig


class DatasetSource(str, Enum):
    SYNTHETIC = "synthetic"
    REAL = "real"


class FeatureRole(IntEnum):
    BASE = 0
    IRRELEVANT = 1


def _matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _vector(value, name: str, *, dtype=np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if dtype != object and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class BaseDataset:
    """One clean task before a collaborator adds irrelevant columns."""

    dataset_id: str
    dataset_name: str
    source: DatasetSource
    X_context: np.ndarray
    y_context: np.ndarray
    X_query: np.ndarray
    y_query: np.ndarray
    context_row_id: np.ndarray | None = None
    query_row_id: np.ndarray | None = None
    relevant_feature_count: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_name:
            raise ValueError("dataset_id and dataset_name must not be empty")
        if not isinstance(self.source, DatasetSource):
            object.__setattr__(self, "source", DatasetSource(str(self.source)))
        X_context = _matrix(self.X_context, "X_context")
        X_query = _matrix(self.X_query, "X_query")
        y_context = _vector(self.y_context, "y_context")
        y_query = _vector(self.y_query, "y_query")
        if X_context.shape[1] != X_query.shape[1] or X_context.shape[1] < 1:
            raise ValueError("context and query must share a positive feature width")
        if len(y_context) != len(X_context) or len(y_query) != len(X_query):
            raise ValueError("feature and target row counts do not match")
        context_ids = (
            np.arange(len(X_context), dtype=np.int64)
            if self.context_row_id is None
            else _vector(self.context_row_id, "context_row_id", dtype=object)
        )
        query_ids = (
            np.arange(len(X_query), dtype=np.int64)
            if self.query_row_id is None
            else _vector(self.query_row_id, "query_row_id", dtype=object)
        )
        if len(context_ids) != len(X_context) or len(query_ids) != len(X_query):
            raise ValueError("row-id lengths do not match their feature arrays")
        if self.relevant_feature_count is not None and not (
            1 <= int(self.relevant_feature_count) <= X_context.shape[1]
        ):
            raise ValueError("relevant_feature_count must lie within the base width")
        object.__setattr__(self, "X_context", X_context)
        object.__setattr__(self, "X_query", X_query)
        object.__setattr__(self, "y_context", y_context)
        object.__setattr__(self, "y_query", y_query)
        object.__setattr__(self, "context_row_id", context_ids)
        object.__setattr__(self, "query_row_id", query_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def d_base(self) -> int:
        return int(self.X_context.shape[1])

    @property
    def ratio_denominator(self) -> int:
        return int(self.relevant_feature_count or self.d_base)


@dataclass(frozen=True)
class DimensionInfo:
    """Resolved integer geometry for one requested dose."""

    requested_irrelevant_ratio: float
    actual_irrelevant_ratio: float
    ratio_denominator: int
    d_base: int
    d_irrelevant: int
    d_total: int
    relevant_feature_count: int | None
    rounding: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.requested_irrelevant_ratio) or not math.isfinite(
            self.actual_irrelevant_ratio
        ):
            raise ValueError("irrelevant ratios must be finite")
        if self.requested_irrelevant_ratio < 0.0:
            raise ValueError("requested_irrelevant_ratio must be non-negative")
        if self.ratio_denominator < 1 or self.d_base < 1:
            raise ValueError("ratio_denominator and d_base must be positive")
        if self.d_irrelevant < 0 or self.d_total != self.d_base + self.d_irrelevant:
            raise ValueError("dimension identity d_total=d_base+d_irrelevant failed")
        if self.relevant_feature_count is not None and not (
            1 <= self.relevant_feature_count <= self.d_base
        ):
            raise ValueError("relevant_feature_count must lie within d_base")
        if self.rounding not in {"ceil", "half_up"}:
            raise ValueError("rounding must be 'ceil' or 'half_up'")
        expected = self.d_irrelevant / self.ratio_denominator
        if not np.isclose(self.actual_irrelevant_ratio, expected):
            raise ValueError(
                "actual_irrelevant_ratio must equal d_irrelevant/ratio_denominator"
            )


@dataclass(frozen=True)
class IrrelevantFeatureBank:
    """Maximum-dose irrelevant columns shared by all lower doses."""

    X_context: np.ndarray
    X_query: np.ndarray
    source_feature_index: np.ndarray
    copy_block_id: np.ndarray
    feature_bank_id: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        context = _matrix(self.X_context, "irrelevant X_context")
        query = _matrix(self.X_query, "irrelevant X_query")
        sources = np.asarray(self.source_feature_index, dtype=np.int32).reshape(-1)
        blocks = np.asarray(self.copy_block_id, dtype=np.int32).reshape(-1)
        bank_ids = np.asarray(self.feature_bank_id, dtype=np.int32).reshape(-1)
        width = context.shape[1]
        if (
            width != query.shape[1]
            or len(sources) != width
            or len(blocks) != width
            or len(bank_ids) != width
        ):
            raise ValueError("irrelevant bank widths and provenance arrays must match")
        if np.any(blocks < 0):
            raise ValueError("copy_block_id must be non-negative")
        if not np.array_equal(bank_ids, np.arange(width, dtype=np.int32)):
            raise ValueError("feature_bank_id must be the contiguous maximum-bank order")
        object.__setattr__(self, "X_context", context)
        object.__setattr__(self, "X_query", query)
        object.__setattr__(self, "source_feature_index", sources)
        object.__setattr__(self, "copy_block_id", blocks)
        object.__setattr__(self, "feature_bank_id", bank_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def width(self) -> int:
        return int(self.X_context.shape[1])


@dataclass(frozen=True)
class PreparedDataset:
    """One fully materialized condition shared unchanged by predict and PDP.

    ``source_feature_index`` stores stable base-feature ids independently of
    their physical columns. Consumers must resolve a base feature through that
    provenance instead of assuming any fixed column position.
    """

    dataset_id: str
    dataset_name: str
    source: DatasetSource
    condition_id: str
    pair_id: str
    dimensions: DimensionInfo
    transform_name: str
    transform_repeat: int
    transform_seed: int
    column_layout: Literal["shuffle"]
    X_context: np.ndarray
    y_context: np.ndarray
    X_query: np.ndarray
    y_query: np.ndarray
    context_row_id: np.ndarray
    query_row_id: np.ndarray
    feature_role: np.ndarray
    source_feature_index: np.ndarray
    copy_block_id: np.ndarray | None = None
    feature_bank_id: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_name:
            raise ValueError("dataset_id and dataset_name must not be empty")
        if not self.condition_id or not self.pair_id or not self.transform_name:
            raise ValueError("condition, pair, and transform ids must not be empty")
        if not isinstance(self.source, DatasetSource):
            object.__setattr__(self, "source", DatasetSource(str(self.source)))
        if self.transform_repeat < 0:
            raise ValueError("transform_repeat must be non-negative")
        if self.column_layout != "shuffle":
            raise ValueError("column_layout must be 'shuffle'")
        X_context = _matrix(self.X_context, "X_context")
        X_query = _matrix(self.X_query, "X_query")
        y_context = _vector(self.y_context, "y_context")
        y_query = _vector(self.y_query, "y_query")
        context_ids = _vector(self.context_row_id, "context_row_id", dtype=object)
        query_ids = _vector(self.query_row_id, "query_row_id", dtype=object)
        roles = np.asarray(self.feature_role, dtype=np.uint8).reshape(-1)
        sources = np.asarray(self.source_feature_index, dtype=np.int32).reshape(-1)
        blocks = (
            np.zeros(len(roles), dtype=np.int32)
            if self.copy_block_id is None
            else np.asarray(self.copy_block_id, dtype=np.int32).reshape(-1)
        )
        bank_ids = (
            np.where(
                roles == FeatureRole.BASE,
                sources,
                np.arange(len(roles), dtype=np.int32),
            ).astype(np.int32)
            if self.feature_bank_id is None
            else np.asarray(self.feature_bank_id, dtype=np.int32).reshape(-1)
        )
        if X_context.shape[1] != self.dimensions.d_total:
            raise ValueError("prepared context width does not match dimensions")
        if X_query.shape[1] != self.dimensions.d_total:
            raise ValueError("prepared query width does not match dimensions")
        if len(y_context) != len(X_context) or len(y_query) != len(X_query):
            raise ValueError("prepared feature and target row counts do not match")
        if len(context_ids) != len(X_context) or len(query_ids) != len(X_query):
            raise ValueError("prepared row-id lengths do not match feature arrays")
        if any(
            len(values) != self.dimensions.d_total
            for values in (roles, sources, blocks, bank_ids)
        ):
            raise ValueError("feature provenance width does not match dimensions")
        allowed_roles = {int(FeatureRole.BASE), int(FeatureRole.IRRELEVANT)}
        if not set(int(value) for value in roles).issubset(allowed_roles):
            raise ValueError("feature_role contains an unknown role")
        if int(np.count_nonzero(roles == FeatureRole.BASE)) != self.dimensions.d_base:
            raise ValueError("feature_role base count does not match d_base")
        if (
            int(np.count_nonzero(roles == FeatureRole.IRRELEVANT))
            != self.dimensions.d_irrelevant
        ):
            raise ValueError("feature_role irrelevant count does not match d_irrelevant")
        base_sources = sources[roles == FeatureRole.BASE]
        expected_base_sources = np.arange(self.dimensions.d_base, dtype=np.int32)
        if not np.array_equal(np.sort(base_sources), expected_base_sources):
            raise ValueError(
                "base source_feature_index must contain every stable base-feature "
                "id exactly once"
            )
        irrelevant_sources = sources[roles == FeatureRole.IRRELEVANT]
        if np.any(irrelevant_sources < -1) or np.any(
            irrelevant_sources >= self.dimensions.d_base
        ):
            raise ValueError(
                "irrelevant source_feature_index must be -1 or a valid base-feature id"
            )
        if np.any(blocks < 0):
            raise ValueError("copy_block_id must be non-negative")
        if np.any(blocks[roles == FeatureRole.BASE] != 0):
            raise ValueError("base features must use copy_block_id=0")
        irrelevant_bank_ids = bank_ids[roles == FeatureRole.IRRELEVANT]
        if len(np.unique(irrelevant_bank_ids)) != len(irrelevant_bank_ids):
            raise ValueError("irrelevant feature_bank_id values must be unique")
        object.__setattr__(self, "X_context", X_context)
        object.__setattr__(self, "X_query", X_query)
        object.__setattr__(self, "y_context", y_context)
        object.__setattr__(self, "y_query", y_query)
        object.__setattr__(self, "context_row_id", context_ids)
        object.__setattr__(self, "query_row_id", query_ids)
        object.__setattr__(self, "feature_role", roles)
        object.__setattr__(self, "source_feature_index", sources)
        object.__setattr__(self, "copy_block_id", blocks)
        object.__setattr__(self, "feature_bank_id", bank_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def column_for_base_feature(self, feature_id: int) -> int:
        """Return the physical column containing one stable base-feature id."""

        feature_id = int(feature_id)
        if not 0 <= feature_id < self.dimensions.d_base:
            raise ValueError(
                f"base feature id must lie in [0, {self.dimensions.d_base}), "
                f"got {feature_id}"
            )
        matches = np.flatnonzero(
            (self.feature_role == FeatureRole.BASE)
            & (self.source_feature_index == feature_id)
        )
        if len(matches) != 1:
            raise ValueError(
                f"base feature id {feature_id} must map to exactly one physical column"
            )
        return int(matches[0])

    def column_for_irrelevant_feature(self, feature_bank_id: int) -> int:
        """Return the physical column for one stable maximum-bank feature id."""

        matches = np.flatnonzero(
            (self.feature_role == FeatureRole.IRRELEVANT)
            & (self.feature_bank_id == int(feature_bank_id))
        )
        if len(matches) != 1:
            raise ValueError(
                f"irrelevant feature_bank_id {feature_bank_id} must map to exactly "
                "one physical column"
            )
        return int(matches[0])


class DatasetGenerator(ABC):
    """Extension point for one synthetic family or real-data source."""

    @abstractmethod
    def generate_base(self) -> Iterable[BaseDataset]:
        """Yield deterministic clean tasks without injected irrelevant columns."""


@dataclass(frozen=True)
class PreparationResult:
    data_run_id: str
    output_dir: Path
    index_path: Path
    skips_path: Path
    num_datasets: int
    num_conditions: int
    num_skips: int


class DatasetPreparer(ABC):
    """Stage contract for config -> immutable prepared-data artifacts."""

    @abstractmethod
    def prepare(
        self,
        config: "ExperimentConfig",
        output_dir: Path,
    ) -> PreparationResult:
        raise NotImplementedError
