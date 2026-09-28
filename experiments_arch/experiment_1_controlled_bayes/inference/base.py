"""The array-only boundary between experimental orchestration and models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from ..artifacts.protocol import Gate
from ..artifacts.data import Task


@dataclass(frozen=True)
class Prediction:
    values: np.ndarray
    gradient_mean: np.ndarray | None = None
    gradient_count: int = 0
    gradient_error: str | None = None


@dataclass(frozen=True)
class PredictionRequest:
    task: Task
    queries: np.ndarray
    gate: Gate = Gate()
    seed: int = 0


class ModelAdapter(ABC):
    """One frozen model; every call carries a complete batch of independent tasks."""

    @property
    @abstractmethod
    def metadata(self) -> dict:
        pass

    @abstractmethod
    def token_groups(self, layers) -> dict:
        """Map each requested layer to source tokens and their raw feature members."""

    @abstractmethod
    def predict_batch(self, requests, *, gradients=False, batch_size=256):
        """Return one Prediction per request, in order; one task is a batch of one.

        Each request supplies its own context, raw queries, gate, and seed.
        batch_size limits queries per forward, not the number of requests.
        """

    @property
    def supports_batch_prediction(self) -> bool:
        """Whether multiple requests may share a forward; False limits it to B=1."""
        return True

    @property
    def supports_feature_branch(self) -> bool:
        return False

    def validate_protocol(self, protocol):
        """Reject unsupported interventions before collection creates any output."""
        if protocol.intervention_kind == "feature_branch" and not self.supports_feature_branch:
            raise ValueError("this model does not support feature_branch interventions")
        groups = self.token_groups(protocol.layers)
        for layer in protocol.layers:
            if any(j >= len(groups[str(layer)]) for j in protocol.source_tokens):
                raise ValueError("source token lies outside the actual feature mapping")
