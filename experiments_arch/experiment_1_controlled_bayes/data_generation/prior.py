"""Coefficient priors for controlled Bayes dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


FLOAT_DTYPE = np.dtype("float32")
SUPPORT_DTYPE = np.dtype("uint8")


@dataclass(frozen=True)
class PriorBatch:
    """Latent coefficients and their exact support for a batch of tasks."""

    beta: np.ndarray
    support_mask: np.ndarray


@dataclass(frozen=True)
class FixedKSparseGaussianPrior:
    """Uniform fixed-k support with energy-normalized Gaussian coefficients."""

    d: int
    k: int
    signal_energy: float

    def __post_init__(self) -> None:
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")
        if not 1 <= self.k <= self.d:
            raise ValueError(
                f"k must satisfy 1 <= k <= d, got k={self.k}, d={self.d}"
            )
        if not math.isfinite(self.signal_energy) or self.signal_energy <= 0.0:
            raise ValueError(
                f"signal_energy must be finite and > 0, got {self.signal_energy}"
            )

    @property
    def active_variance(self) -> float:
        return float(self.signal_energy / self.k)

    @property
    def expected_covariance_diagonal(self) -> float:
        return float(self.signal_energy / self.d)

    def sample(
        self,
        batch_size: int,
        *,
        support_rng: np.random.Generator,
        beta_rng: np.random.Generator,
    ) -> PriorBatch:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if self.k == self.d:
            support = np.ones((batch_size, self.d), dtype=SUPPORT_DTYPE)
        else:
            # The k smallest iid continuous scores form a uniform size-k subset.
            scores = support_rng.random((batch_size, self.d))
            active = np.argpartition(scores, kth=self.k - 1, axis=1)[:, : self.k]
            support = np.zeros((batch_size, self.d), dtype=SUPPORT_DTYPE)
            support[np.arange(batch_size)[:, None], active] = 1

        beta = beta_rng.normal(
            loc=0.0,
            scale=math.sqrt(self.active_variance),
            size=(batch_size, self.d),
        )
        beta *= support
        sample = PriorBatch(
            beta=np.asarray(beta, dtype=FLOAT_DTYPE),
            support_mask=support,
        )
        self.validate_sample(sample)
        return sample

    def validate_sample(self, sample: PriorBatch) -> None:
        if sample.beta.ndim != 2:
            raise RuntimeError(f"beta must be 2D, got shape={sample.beta.shape}")
        expected_shape = sample.beta.shape
        if expected_shape[1] != self.d:
            raise RuntimeError(
                f"beta has width {expected_shape[1]}, expected d={self.d}"
            )
        if sample.support_mask.shape != expected_shape:
            raise RuntimeError(
                "support_mask and beta must have the same shape, got "
                f"{sample.support_mask.shape} and {expected_shape}"
            )
        if sample.beta.dtype != FLOAT_DTYPE:
            raise RuntimeError(f"beta must be float32, got {sample.beta.dtype}")
        if sample.support_mask.dtype != SUPPORT_DTYPE:
            raise RuntimeError(
                f"support_mask must be uint8, got {sample.support_mask.dtype}"
            )
        if not np.isfinite(sample.beta).all():
            raise RuntimeError("beta contains NaN or infinite values")
        if not np.all(sample.support_mask.sum(axis=1) == self.k):
            raise RuntimeError("support does not contain exactly k features")
        if np.any(sample.beta[sample.support_mask == 0] != 0.0):
            raise RuntimeError("inactive coefficients must be exactly zero")

    def metadata(self) -> dict[str, Any]:
        return {
            "support_prior": "uniform_fixed_k",
            "coefficient_prior": "centered_gaussian_on_support",
            "d": self.d,
            "k": self.k,
            "coefficient_variance": self.active_variance,
            "expected_signal_energy": self.signal_energy,
            "expected_coefficient_covariance_diagonal": (
                self.expected_covariance_diagonal
            ),
        }
