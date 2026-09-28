"""Compatibility exports for the project-level model boundary."""

from experiment_core.adapters import (
    ModelConfig,
    ModelDescriptor,
    PredictiveModel,
    PretrainedModelFactory,
)
from experiment_core.pretrained import (
    DefaultPretrainedModelFactory,
    TabDPTPredictiveModel,
    TabICLPredictiveModel,
    TabPFNPredictiveModel,
    TargetScaledPredictiveModel,
)


__all__ = [
    "DefaultPretrainedModelFactory",
    "ModelConfig",
    "ModelDescriptor",
    "PredictiveModel",
    "PretrainedModelFactory",
    "TabDPTPredictiveModel",
    "TabICLPredictiveModel",
    "TabPFNPredictiveModel",
    "TargetScaledPredictiveModel",
]
