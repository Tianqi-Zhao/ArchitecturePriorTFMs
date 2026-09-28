"""In-context regression episode generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from experiments_arch.experiment_1_controlled_bayes.data_generation.prior import (
    FLOAT_DTYPE,
    FixedKSparseGaussianPrior,
    PriorBatch,
)


MAX_SEED = (1 << 64) - 1
ORTHOGONAL_ATOL = 5e-5
_RANDOM_COMPONENTS = (
    "single_eval_pos",
    "support",
    "beta",
    "context",
    "query",
    "noise",
)


@dataclass(frozen=True)
class EpisodeBatch:
    """A fixed-length task batch sharing one context/query boundary."""

    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    support_mask: np.ndarray
    single_eval_pos: int


@dataclass(frozen=True)
class _RandomStreams:
    single_eval_pos: np.random.Generator
    support: np.random.Generator
    beta: np.random.Generator
    context: np.random.Generator
    query: np.random.Generator
    noise: np.random.Generator

    @classmethod
    def create(cls, seed: int, stream_index: int) -> _RandomStreams:
        streams = {
            name: np.random.default_rng(
                np.random.SeedSequence(
                    seed,
                    spawn_key=(stream_index, component_index),
                )
            )
            for component_index, name in enumerate(_RANDOM_COMPONENTS)
        }
        return cls(**streams)

    @staticmethod
    def spawn_keys(stream_index: int) -> dict[str, list[int]]:
        return {
            name: [stream_index, component_index]
            for component_index, name in enumerate(_RANDOM_COMPONENTS)
        }


class ControlledBayesEpisodeGenerator:
    """Sample a task batch with one random ``single_eval_pos``."""

    def __init__(
        self,
        *,
        prior: FixedKSparseGaussianPrior,
        n_context_min: int,
        n_context_max: int,
        n_total: int,
        noise_std: float,
        seed: int,
        stream_index: int,
    ) -> None:
        if n_context_min < prior.d:
            raise ValueError(
                "n_context_min must be >= d for an orthogonal context, "
                f"got n_context_min={n_context_min}, d={prior.d}"
            )
        if n_context_max < n_context_min:
            raise ValueError(
                "n_context_max must be >= n_context_min, got "
                f"{n_context_max} < {n_context_min}"
            )
        if n_total <= n_context_max:
            raise ValueError(
                "n_total must be > n_context_max so every task has a query, "
                f"got n_total={n_total}, n_context_max={n_context_max}"
            )
        if not math.isfinite(noise_std) or noise_std <= 0.0:
            raise ValueError(
                f"noise_std must be finite and > 0, got {noise_std}"
            )
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
        if stream_index < 0:
            raise ValueError(f"stream_index must be >= 0, got {stream_index}")

        self.prior = prior
        self.n_context_min = int(n_context_min)
        self.n_context_max = int(n_context_max)
        self.n_total = int(n_total)
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.stream_index = int(stream_index)
        self._streams = _RandomStreams.create(self.seed, self.stream_index)

    @property
    def noise_variance(self) -> float:
        return float(self.noise_std**2)

    def sample(self, batch_size: int) -> EpisodeBatch:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        single_eval_pos = int(
            self._streams.single_eval_pos.integers(
                self.n_context_min,
                self.n_context_max + 1,
            )
        )
        prior_batch = self.prior.sample(
            batch_size,
            support_rng=self._streams.support,
            beta_rng=self._streams.beta,
        )
        context = self._sample_context(batch_size, single_eval_pos)
        query = self._streams.query.standard_normal(
            (batch_size, self.n_total - single_eval_pos, self.prior.d)
        )
        X = np.asarray(np.concatenate((context, query), axis=1), dtype=FLOAT_DTYPE)
        noise = self._streams.noise.standard_normal((batch_size, self.n_total))
        noise *= self.noise_std
        y = np.einsum(
            "bnd,bd->bn",
            X,
            prior_batch.beta,
            dtype=np.float64,
        ) + noise
        batch = EpisodeBatch(
            X=X,
            y=np.asarray(y, dtype=FLOAT_DTYPE),
            beta=prior_batch.beta,
            support_mask=prior_batch.support_mask,
            single_eval_pos=single_eval_pos,
        )
        self.validate_batch(batch)
        return batch

    def validate_batch(self, batch: EpisodeBatch) -> None:
        batch_size = batch.X.shape[0]
        expected_shapes = {
            "X": (batch_size, self.n_total, self.prior.d),
            "y": (batch_size, self.n_total),
            "beta": (batch_size, self.prior.d),
            "support_mask": (batch_size, self.prior.d),
        }
        for name, expected in expected_shapes.items():
            actual = getattr(batch, name).shape
            if actual != expected:
                raise RuntimeError(
                    f"generated {name} has shape {actual}, expected {expected}"
                )
        if batch.X.dtype != FLOAT_DTYPE or batch.y.dtype != FLOAT_DTYPE:
            raise RuntimeError("generated X and y must be float32")
        self.prior.validate_sample(
            PriorBatch(beta=batch.beta, support_mask=batch.support_mask)
        )
        if not np.isfinite(batch.X).all() or not np.isfinite(batch.y).all():
            raise RuntimeError("generated X/y contains NaN or infinite values")
        boundary = batch.single_eval_pos
        if not self.n_context_min <= boundary <= self.n_context_max:
            raise RuntimeError("single_eval_pos lies outside its configured range")
        context = batch.X[:, :boundary].astype(np.float64)
        gram = np.einsum("bnd,bne->bde", context, context) / boundary
        if not np.allclose(
            gram,
            np.eye(self.prior.d)[None, :, :],
            rtol=ORTHOGONAL_ATOL,
            atol=ORTHOGONAL_ATOL,
        ):
            raise RuntimeError("generated context design is not orthogonal")

    def metadata(self) -> dict[str, Any]:
        return {
            **self.prior.metadata(),
            "n_context_min": self.n_context_min,
            "n_context_max": self.n_context_max,
            "n_total": self.n_total,
            "context_length_prior": (
                "discrete_uniform_inclusive_per_generator_batch"
            ),
            "noise_std": self.noise_std,
            "noise_variance": self.noise_variance,
            "data_seed": self.seed,
            "rng_spawn_keys": _RandomStreams.spawn_keys(self.stream_index),
        }

    def _sample_context(self, batch_size: int, n_context: int) -> np.ndarray:
        raw = self._streams.context.standard_normal(
            (batch_size, n_context, self.prior.d)
        )
        q, r = np.linalg.qr(raw, mode="reduced")

        # Fix QR's arbitrary column signs for more portable deterministic replay.
        diagonal = np.diagonal(r, axis1=-2, axis2=-1)
        signs = np.where(diagonal < 0.0, -1.0, 1.0)
        q *= signs[:, None, :]
        return np.asarray(math.sqrt(n_context) * q, dtype=FLOAT_DTYPE)
