"""Configuration contract for Experiment 0.

The concrete generators and model factories consume these immutable values.
This module intentionally has no dependency on OpenML, torch, or model packages.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from experiment_core.adapters import (
    DEFAULT_MAX_INPUT_FEATURES_BY_ORIGIN,
    MODEL_ORIGIN_PRETRAINED,
    ModelConfig,
)
from experiments_arch.experiment_0_pretrained.data.base import DatasetSource


DEFAULT_SYNTHETIC_RATIOS = (0.0, 0.2, 0.4, 1.0, 2.0, 4.0, 7.0)
DEFAULT_REAL_RATIOS = (0.0, 0.2, 0.4, 1.0, 2.0, 4.0, 7.0)
DEFAULT_MAX_INPUT_FEATURES = DEFAULT_MAX_INPUT_FEATURES_BY_ORIGIN[
    MODEL_ORIGIN_PRETRAINED
]
PDP_GRID_METHOD = "empirical_quantiles"
PART_I2_PERMUTATION_METHOD = "independent_query_permutation"
PART_I2_PDP_GRID_SCOPE = "paired_feature_reference"


def _validated_ratios(values) -> tuple[float, ...]:
    ratios = tuple(float(value) for value in values)
    if not ratios:
        raise ValueError("a source must declare at least one irrelevant ratio")
    if any(not math.isfinite(value) for value in ratios):
        raise ValueError(f"irrelevant ratios must be finite, got {ratios}")
    if any(value < 0.0 for value in ratios):
        raise ValueError(f"irrelevant ratios must be non-negative, got {ratios}")
    if 0.0 not in ratios:
        raise ValueError("irrelevant ratios must contain the clean baseline 0")
    if len(set(ratios)) != len(ratios):
        raise ValueError(f"irrelevant ratios must not contain duplicates: {ratios}")
    if tuple(sorted(ratios)) != ratios:
        raise ValueError(f"irrelevant ratios must be sorted: {ratios}")
    return ratios


@dataclass(frozen=True)
class SourceConfig:
    """One prepared-data source and its source-specific dose grid."""

    source: DatasetSource
    source_id: str
    generator: str
    transform: str
    ratios: tuple[float, ...]
    rounding: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, DatasetSource):
            object.__setattr__(self, "source", DatasetSource(str(self.source)))
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.generator:
            raise ValueError("generator must not be empty")
        if not self.transform:
            raise ValueError("transform must not be empty")
        if self.rounding not in {"ceil", "half_up"}:
            raise ValueError("rounding must be 'ceil' or 'half_up'")
        object.__setattr__(self, "ratios", _validated_ratios(self.ratios))
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceConfig":
        return cls(
            source=DatasetSource(str(value["source"])),
            source_id=str(value["source_id"]),
            generator=str(value["generator"]),
            transform=str(value["transform"]),
            ratios=tuple(value["ratios"]),
            rounding=str(value["rounding"]),
            parameters=value.get("parameters", {}),
        )


@dataclass(frozen=True)
class PDPGridConfig:
    """Part I.2 PDP grid shared across models and paired dose conditions."""

    points: int = 5
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    method: str = PDP_GRID_METHOD
    scope: str = PART_I2_PDP_GRID_SCOPE

    def __post_init__(self) -> None:
        if self.method != PDP_GRID_METHOD:
            raise ValueError(f"unsupported PDP grid method: {self.method!r}")
        if self.scope != PART_I2_PDP_GRID_SCOPE:
            raise ValueError(f"unsupported PDP grid scope: {self.scope!r}")
        if self.points < 2:
            raise ValueError("PDP grid must request at least two points")
        if not (
            math.isfinite(self.lower_quantile)
            and math.isfinite(self.upper_quantile)
            and 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0
        ):
            raise ValueError(
                "PDP grid quantiles must satisfy 0 <= lower < upper <= 1"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PDPGridConfig":
        return cls(
            points=int(value.get("points", 5)),
            lower_quantile=float(value.get("lower_quantile", 0.05)),
            upper_quantile=float(value.get("upper_quantile", 0.95)),
            method=str(value.get("method", PDP_GRID_METHOD)),
            scope=str(value.get("scope", PART_I2_PDP_GRID_SCOPE)),
        )


@dataclass(frozen=True)
class PartI2Config:
    """Functional-suppression settings for Experiment I.2."""

    models: tuple[str, ...]
    default_models: tuple[str, ...] | None = None
    sources: tuple[DatasetSource, ...] = (
        DatasetSource.SYNTHETIC,
        DatasetSource.REAL,
    )
    ratios: tuple[float, ...] = DEFAULT_SYNTHETIC_RATIOS
    permutation_method: str = PART_I2_PERMUTATION_METHOD
    permutation_repeats: int = 1
    pdp_max_features_per_role: int = 5
    pdp_grid: PDPGridConfig = field(default_factory=lambda: PDPGridConfig(
        points=5,
        scope=PART_I2_PDP_GRID_SCOPE,
    ))
    counterfactual_batch_size: int = 4
    validate_batch_invariance: bool = True

    def __post_init__(self) -> None:
        models = tuple(str(model_id) for model_id in self.models)
        default_models = (
            models
            if self.default_models is None
            else tuple(str(model_id) for model_id in self.default_models)
        )
        sources = tuple(
            source
            if isinstance(source, DatasetSource)
            else DatasetSource(str(source))
            for source in self.sources
        )
        ratios = tuple(float(value) for value in self.ratios)
        if not models or len(models) != len(set(models)):
            raise ValueError("Part I.2 models must be non-empty and unique")
        if not default_models or len(default_models) != len(set(default_models)):
            raise ValueError(
                "Part I.2 default_models must be non-empty and unique"
            )
        unknown_defaults = sorted(set(default_models) - set(models))
        if unknown_defaults:
            raise ValueError(
                f"Part I.2 default models are not supported: {unknown_defaults}"
            )
        if not sources or len(sources) != len(set(sources)):
            raise ValueError("Part I.2 sources must be non-empty and unique")
        if not ratios or len(ratios) != len(set(ratios)):
            raise ValueError("Part I.2 ratios must be non-empty and unique")
        if any(not math.isfinite(value) or value < 0.0 for value in ratios):
            raise ValueError("Part I.2 ratios must be finite and non-negative")
        if tuple(sorted(ratios)) != ratios:
            raise ValueError("Part I.2 ratios must be sorted")
        if 0.0 not in ratios:
            raise ValueError("Part I.2 ratios must include the clean ratio 0")
        if self.permutation_method != PART_I2_PERMUTATION_METHOD:
            raise ValueError(
                f"unsupported Part I.2 permutation method: {self.permutation_method!r}"
            )
        if self.permutation_repeats != 1:
            raise ValueError(
                "the production Part I.2 design uses exactly one permutation "
                "inside each construction repeat"
            )
        if self.pdp_max_features_per_role < 1:
            raise ValueError("pdp_max_features_per_role must be positive")
        if self.pdp_grid.scope != PART_I2_PDP_GRID_SCOPE:
            raise ValueError(
                f"Part I.2 PDP grid scope must be {PART_I2_PDP_GRID_SCOPE!r}"
            )
        if self.counterfactual_batch_size < 1:
            raise ValueError("counterfactual_batch_size must be positive")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "default_models", default_models)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "ratios", ratios)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PartI2Config":
        grid_defaults = {
            "points": 5,
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
            "method": PDP_GRID_METHOD,
            "scope": PART_I2_PDP_GRID_SCOPE,
        }
        return cls(
            models=tuple(value["models"]),
            default_models=tuple(value.get("default_models", value["models"])),
            sources=tuple(value.get("sources", ("synthetic", "real"))),
            ratios=tuple(value.get("ratios", DEFAULT_SYNTHETIC_RATIOS)),
            permutation_method=str(
                value.get("permutation_method", PART_I2_PERMUTATION_METHOD)
            ),
            permutation_repeats=int(value.get("permutation_repeats", 1)),
            pdp_max_features_per_role=int(
                value.get("pdp_max_features_per_role", 5)
            ),
            pdp_grid=PDPGridConfig.from_mapping(
                value.get("pdp_grid", grid_defaults)
            ),
            counterfactual_batch_size=int(
                value.get("counterfactual_batch_size", 4)
            ),
            validate_batch_invariance=bool(
                value.get("validate_batch_invariance", True)
            ),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved settings shared by the experiment's independent stages."""

    artifact_root: Path
    sources: tuple[SourceConfig, ...]
    models: tuple[ModelConfig, ...]
    experiment_id: str = "experiment_0_pretrained"
    schema_version: int = 1
    run_seed: int = 0
    transform_repeats: int = 5
    max_input_features: int = DEFAULT_MAX_INPUT_FEATURES
    part_i2: PartI2Config | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_root", Path(self.artifact_root))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "models", tuple(self.models))
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if self.transform_repeats < 1:
            raise ValueError("transform_repeats must be positive")
        if self.max_input_features < 1:
            raise ValueError("max_input_features must be positive")
        if not self.sources:
            raise ValueError("at least one source is required")
        if not self.models:
            raise ValueError("at least one model is required")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"duplicate source ids: {source_ids}")
        model_ids = [model.model_id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError(f"duplicate model ids: {model_ids}")
        configured_model_ids = set(model_ids)
        configured_sources = {source.source for source in self.sources}
        if self.part_i2 is not None:
            unknown_models = sorted(set(self.part_i2.models) - configured_model_ids)
            if unknown_models:
                raise ValueError(f"Part I.2 models are not configured: {unknown_models}")
            unknown_sources = sorted(
                source.value
                for source in set(self.part_i2.sources) - configured_sources
            )
            if unknown_sources:
                raise ValueError(
                    f"Part I.2 sources are not configured: {unknown_sources}"
                )
            configured_ratios = {
                source.source: set(source.ratios) for source in self.sources
            }
            for source in self.part_i2.sources:
                missing = sorted(
                    set(self.part_i2.ratios) - configured_ratios.get(source, set())
                )
                if missing:
                    raise ValueError(
                        f"Part I.2 ratios are not configured for {source.value}: {missing}"
                    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        if "pdp_models" in value:
            raise ValueError(
                "pdp_models is obsolete; configure Part I.2 under part_i2"
            )
        if "pdp" in value:
            raise ValueError(
                "the standalone pdp stage is obsolete; configure Part I.2 under part_i2"
            )
        part_i2_value = value.get("part_i2")
        return cls(
            experiment_id=str(value.get("experiment_id", "experiment_0_pretrained")),
            schema_version=int(value.get("schema_version", 1)),
            artifact_root=Path(value["artifact_root"]),
            run_seed=int(value.get("run_seed", 0)),
            transform_repeats=int(value.get("transform_repeats", 5)),
            max_input_features=int(
                value.get("max_input_features", DEFAULT_MAX_INPUT_FEATURES)
            ),
            part_i2=(
                None
                if part_i2_value is None
                else PartI2Config.from_mapping(part_i2_value)
            ),
            sources=tuple(
                SourceConfig.from_mapping(source) for source in value["sources"]
            ),
            models=tuple(ModelConfig.from_mapping(model) for model in value["models"]),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a JSON file and return the validated experiment configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("experiment config must be a JSON object")
    return ExperimentConfig.from_mapping(value)
