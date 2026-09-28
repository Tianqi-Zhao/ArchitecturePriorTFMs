"""Exact-k protocol shared by collection and offline evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import numpy as np

SCHEMA_VERSION = "exact_k_pipeline/4"
SUPPORTED_SCHEMAS = ("exact_k_pipeline/3", SCHEMA_VERSION)
CODE_VERSION = "5"
# Evaluation 6 uses equal-source specificity; collection/estimation are unchanged.
EVALUATION_VERSION = "6"
METHOD_NAMES = ("moment", "gradient", "affine_blp")
TERMINAL = {"ok", "failed", "unavailable"}
ROW_SCOPES = ("both", "context", "query")
INTERVENTION_KINDS = ("source", "feature_branch")


def validate_row_scope(scope):
    if scope not in ROW_SCOPES:
        raise ValueError(f"row_scope must be one of {ROW_SCOPES}")


def validate_intervention_kind(kind):
    if kind not in INTERVENTION_KINDS:
        raise ValueError(f"intervention kind must be one of {INTERVENTION_KINDS}")


class UnavailableError(RuntimeError):
    """A valid request cannot be computed from this backend or saved material."""


@dataclass(frozen=True)
class Prior:
    d: int
    k: int
    signal_energy: float
    noise_std: float

    def __post_init__(self):
        if (any(not isinstance(x, int) or isinstance(x, bool) for x in (self.d, self.k))
                or not 1 <= self.k <= self.d):
            raise ValueError("prior requires 1 <= k <= d")
        if any(not math.isfinite(x) or x <= 0
               for x in (self.signal_energy, self.noise_std)):
            raise ValueError("signal_energy and noise_std must be finite and positive")


@dataclass(frozen=True)
class Protocol:
    layers: tuple[int, ...] = ()
    epsilons: tuple[float, ...] = (0.05, 0.1, 0.2)
    source_tokens: tuple[int, ...] = ()
    query_batch_size: int = 256
    task_batch_size: int = 32
    inference_seed: int = 0
    atol: float = 1e-5
    rtol: float = 1e-4
    row_scope: str = "both"
    intervention_kind: str = "source"

    def __post_init__(self):
        validate_row_scope(self.row_scope)
        validate_intervention_kind(self.intervention_kind)
        if self.intervention_kind == "feature_branch" and self.source_tokens:
            raise ValueError("feature_branch does not accept source_tokens")
        for name in ("layers", "epsilons", "source_tokens"):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        for name in ("layers", "source_tokens"):
            if any(not isinstance(x, int) or isinstance(x, bool) or x < 0
                   for x in getattr(self, name)):
                raise ValueError(f"{name} must contain nonnegative integers")
        if not self.epsilons or any(not math.isfinite(x) or not 0 < x <= 1
                                   for x in self.epsilons):
            raise ValueError("epsilons must satisfy 0 < epsilon <= 1")
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1
               for x in (self.query_batch_size, self.task_batch_size)):
            raise ValueError("batch sizes must be positive")
        if (not isinstance(self.inference_seed, int) or isinstance(self.inference_seed, bool)
                or self.inference_seed < 0):
            raise ValueError("inference_seed must be nonnegative")
        if any(not math.isfinite(x) or x < 0 for x in (self.atol, self.rtol)):
            raise ValueError("prediction tolerances must be finite and nonnegative")

    def to_dict(self):
        values = asdict(self)
        for name in ("layers", "epsilons", "source_tokens"):
            values[name] = list(values[name])
        return values

    @classmethod
    def from_args(cls, args):
        """Share protocol extraction across prediction, preflight and reuse checks."""
        return cls(**{field.name: getattr(args, field.name) for field in fields(cls)
                      if hasattr(args, field.name)})

    @classmethod
    def from_run(cls, config):
        """Read legacy source runs without relaxing the new schema's contract."""
        schema = config.get("schema_version")
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError("unsupported run schema")
        values = config.get("protocol")
        if not isinstance(values, dict):
            raise ValueError("run is missing a protocol object")
        if schema == SCHEMA_VERSION and "intervention_kind" not in values:
            raise ValueError("run protocol is missing intervention_kind")
        protocol = cls(**values)
        if schema != SCHEMA_VERSION and protocol.intervention_kind != "source":
            raise ValueError("legacy schema only supports source interventions")
        return protocol

    def task_seed(self, task_id):
        seed = self.inference_seed
        return int(np.random.SeedSequence(
            [seed & 0xFFFFFFFF, seed >> 32, task_id & 0xFFFFFFFF, task_id >> 32]
        ).generate_state(1)[0])


@dataclass(frozen=True)
class Gate:
    layer: int = -1
    source_token: int = -1
    epsilon: float = 0.0
    row_scope: str = "both"
    kind: str = "source"

    def __post_init__(self):
        validate_row_scope(self.row_scope)
        validate_intervention_kind(self.kind)

    @property
    def is_baseline(self):
        return (self.kind == "source" and self.layer == -1
                and self.source_token == -1 and self.epsilon == 0)

    @property
    def state_kind(self):
        return "baseline" if self.is_baseline else self.kind


@dataclass(frozen=True)
class EvaluationConfig:
    denominator_threshold: float = 1e-12
    projection_diagnostics: bool = False

    def __post_init__(self):
        if (not math.isfinite(self.denominator_threshold)
                or self.denominator_threshold < 0):
            raise ValueError("denominator_threshold must be finite and nonnegative")
