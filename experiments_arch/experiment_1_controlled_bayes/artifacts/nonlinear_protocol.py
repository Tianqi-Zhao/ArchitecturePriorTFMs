"""Linear-energy sidecar and affine residual summary contracts."""

from dataclasses import dataclass
import math

ENERGY_SCHEMA = "exact_k_linear_energy/1"
SUMMARY_SCHEMA = "exact_k_nonlinear_summary/1"
ALGORITHM_VERSION = "1"
SAVED_METRICS = ("C", "Z", "N", "E", "delta_IV2")
METRICS = ("L",) + SAVED_METRICS
RATIOS = {
    "nonlinear_fraction": ("N", ("L", "N")),
    "nonlinear_to_error": ("N", ("E",)),
    "gap_to_error": ("delta_IV2", ("E",)),
    "abs_gap_to_error": ("delta_IV2", ("E",)),
}


@dataclass(frozen=True)
class NonlinearEvaluationConfig:
    denominator_threshold: float = 1e-12

    def __post_init__(self):
        if not math.isfinite(self.denominator_threshold) or self.denominator_threshold < 0:
            raise ValueError("denominator_threshold must be finite and nonnegative")


@dataclass(frozen=True)
class SummaryConfig:
    bootstrap_repeats: int = 2000
    bootstrap_seed: int = 0
    confidence_level: float = 0.95

    def __post_init__(self):
        for name, minimum in (("bootstrap_repeats", 1), ("bootstrap_seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie between 0 and 1")
