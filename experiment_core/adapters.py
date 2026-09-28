"""Experiment-neutral model contracts and model-origin policies.

Concrete pretrained backends may be large and dependency-heavy.  They implement
these contracts, while experiment packages only depend on this neutral module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


MODEL_ORIGIN_OWN = "own"
MODEL_ORIGIN_PRETRAINED = "pretrained"
MODEL_ORIGIN_TFM_PRIOR = "tfm_prior"
MODEL_ORIGINS = (
    MODEL_ORIGIN_OWN,
    MODEL_ORIGIN_PRETRAINED,
    MODEL_ORIGIN_TFM_PRIOR,
)

# One source of truth for experiment-level input guards. Concrete adapters may
# impose a stricter limit, but experiments must never silently widen this map.
DEFAULT_MAX_INPUT_FEATURES_BY_ORIGIN = {
    MODEL_ORIGIN_OWN: 10,
    MODEL_ORIGIN_PRETRAINED: 1000,
    MODEL_ORIGIN_TFM_PRIOR: 10,
}


@dataclass(frozen=True)
class ModelConfig:
    """Unique model identity plus adapter-specific, JSON-safe options.

    Behaviorally distinct models must use distinct ``model_id`` values. Model
    variants are deliberately unsupported so artifact identity stays uniform.
    """

    model_id: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        object.__setattr__(self, "options", dict(self.options))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelConfig":
        if "variant" in value:
            raise ValueError(
                "model variants are unsupported; encode the distinction in "
                "model_id instead"
            )
        return cls(
            model_id=str(value["model_id"]),
            options=value.get("options", {}),
        )


@dataclass(frozen=True)
class ModelDescriptor:
    """Resolved model identity written alongside raw predictions.

    ``model_version`` identifies the loaded package or checkpoint revision.
    """

    model_id: str
    model_group: str
    model_origin: str
    model_version: str
    max_input_features: int
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_version:
            raise ValueError("model_id and model_version must not be empty")
        if self.model_origin not in MODEL_ORIGINS:
            raise ValueError(f"unknown model origin {self.model_origin!r}")
        if self.max_input_features < 1:
            raise ValueError("max_input_features must be positive")
        object.__setattr__(self, "options", dict(self.options))


class PredictiveModel(ABC):
    """Minimal model boundary shared by prediction-based experiments."""

    descriptor: ModelDescriptor

    @abstractmethod
    def prepare(self, device: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_predict(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        """Return one finite prediction per query row in original target units.

        A backend that scales regression targets must fit that transform on
        ``y_context`` only and inverse-transform its predictions before return.
        Ordinary prediction and intervention runners depend on this shared
        behavior.
        """

        raise NotImplementedError


class PretrainedModelFactory(ABC):
    """Translate a JSON-safe request into one model adapter.

    The implementation belongs at the project integration boundary. Concrete
    backends live in the neutral core; legacy experiment adapters may extend
    those backends with attention capabilities, but the core never imports an
    experiment package.
    """

    @abstractmethod
    def build(self, config: ModelConfig) -> PredictiveModel:
        raise NotImplementedError


__all__ = [
    "DEFAULT_MAX_INPUT_FEATURES_BY_ORIGIN",
    "MODEL_ORIGINS",
    "MODEL_ORIGIN_OWN",
    "MODEL_ORIGIN_PRETRAINED",
    "MODEL_ORIGIN_TFM_PRIOR",
    "ModelConfig",
    "ModelDescriptor",
    "PredictiveModel",
    "PretrainedModelFactory",
]
