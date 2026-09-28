"""Exact-k Bayes posterior coefficients computed from context and prior."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def bayes_signal(coefficient: np.ndarray) -> float:
    """Population signal under standard Gaussian evaluation queries."""
    eta = np.asarray(coefficient, dtype=np.float64)
    return float(eta @ eta)


def bayes_prediction_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Finite-query squared error relative to the task's Bayes predictor."""
    return float(np.mean(np.square(
        np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    )))


@dataclass(frozen=True)
class ExactKBayesPredictor:
    """Compute the exact posterior mean without enumerating size-k supports."""

    d: int
    k: int
    signal_energy: float
    noise_std: float

    def __post_init__(self) -> None:
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")
        if not 1 <= self.k <= self.d:
            raise ValueError(
                f"k must satisfy 1 <= k <= d, got k={self.k}, d={self.d}"
            )
        if not math.isfinite(self.signal_energy) or self.signal_energy <= 0.0:
            raise ValueError(
                "signal_energy must be finite and > 0, got "
                f"{self.signal_energy}"
            )
        if not math.isfinite(self.noise_std) or self.noise_std <= 0.0:
            raise ValueError(
                f"noise_std must be finite and > 0, got {self.noise_std}"
            )

    def sufficient_statistic(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
    ) -> np.ndarray:
        """Return c = X.T y / n for one orthogonal context."""

        x = np.asarray(context_x, dtype=np.float64)
        y = np.asarray(context_y, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"context_x must have shape (n, {self.d}), got {x.shape}"
            )
        if y.shape != (x.shape[0],):
            raise ValueError(
                "context_y must have one value per context row, got "
                f"{y.shape} for {x.shape[0]} rows"
            )
        if x.shape[0] < 1:
            raise ValueError("context must contain at least one row")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("context contains non-finite values")
        return np.asarray(x.T @ y / x.shape[0], dtype=np.float64)

    def inclusion_probabilities(
        self,
        sufficient_statistic: np.ndarray,
        *,
        n_context: int,
    ) -> np.ndarray:
        """Return all posterior support-inclusion probabilities q_(k,j)."""

        c = np.asarray(sufficient_statistic, dtype=np.float64)
        if c.shape != (self.d,):
            raise ValueError(f"sufficient_statistic must have shape ({self.d},)")
        if n_context < 1:
            raise ValueError(f"n_context must be >= 1, got {n_context}")
        if not np.isfinite(c).all():
            raise ValueError("sufficient_statistic contains non-finite values")

        if self.k == self.d:
            return np.ones(self.d, dtype=np.float64)

        tau_squared = self.noise_std**2 / n_context
        theta = self.signal_energy / (
            2.0 * tau_squared * (self.signal_energy + self.k * tau_squared)
        )
        log_weights = theta * np.square(c)
        if not np.isfinite(log_weights).all():
            raise ValueError("posterior log weights contain non-finite values")

        if self.k == 1:
            shifted = log_weights - np.max(log_weights)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
            return probabilities

        probabilities = self._forward_backward_inclusions(log_weights)
        tolerance = 1e-10
        if np.any(probabilities < -tolerance) or np.any(
            probabilities > 1.0 + tolerance
        ):
            raise RuntimeError("posterior inclusion probability lies outside [0, 1]")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        if not math.isclose(
            float(probabilities.sum()),
            float(self.k),
            rel_tol=1e-10,
            abs_tol=1e-10 * self.k,
        ):
            raise RuntimeError("posterior inclusion probabilities do not sum to k")
        return probabilities

    def coefficients(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
    ) -> np.ndarray:
        """Return eta*_k(D) = E[beta | D] for one context."""

        c = self.sufficient_statistic(context_x, context_y)
        n_context = int(np.asarray(context_x).shape[0])
        tau_squared = self.noise_std**2 / n_context
        shrinkage = self.signal_energy / (
            self.signal_energy + self.k * tau_squared
        )
        inclusion = self.inclusion_probabilities(c, n_context=n_context)
        coefficients = shrinkage * inclusion * c
        if not np.isfinite(coefficients).all():
            raise RuntimeError("Bayes coefficients contain non-finite values")
        return np.asarray(coefficients, dtype=np.float64)

    def predict(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
        query_x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Bayes coefficients and predictions for one task."""

        queries = np.asarray(query_x, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != self.d:
            raise ValueError(
                f"query_x must have shape (n_queries, {self.d}), got {queries.shape}"
            )
        if queries.shape[0] < 1:
            raise ValueError("query_x must contain at least one query")
        if not np.isfinite(queries).all():
            raise ValueError("query_x contains non-finite values")
        coefficients = self.coefficients(context_x, context_y)
        predictions = queries @ coefficients
        if not np.isfinite(predictions).all():
            raise RuntimeError("Bayes predictions contain non-finite values")
        return coefficients, np.asarray(predictions, dtype=np.float64)

    def _forward_backward_inclusions(
        self,
        log_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute every w_j e_(k-1)(w_-j) / e_k(w) in O(dk)."""

        forward = np.full((self.d + 1, self.k + 1), -np.inf, dtype=np.float64)
        backward = np.full_like(forward, -np.inf)
        forward[0, 0] = 0.0
        for index in range(1, self.d + 1):
            forward[index, 0] = 0.0
            for selected in range(1, min(index, self.k) + 1):
                forward[index, selected] = np.logaddexp(
                    forward[index - 1, selected],
                    log_weights[index - 1] + forward[index - 1, selected - 1],
                )

        backward[self.d, 0] = 0.0
        for index in range(self.d - 1, -1, -1):
            backward[index, 0] = 0.0
            available = self.d - index
            for selected in range(1, min(available, self.k) + 1):
                backward[index, selected] = np.logaddexp(
                    backward[index + 1, selected],
                    log_weights[index] + backward[index + 1, selected - 1],
                )

        log_denominator = forward[self.d, self.k]
        probabilities = np.empty(self.d, dtype=np.float64)
        for feature in range(self.d):
            terms = np.asarray(
                [
                    forward[feature, left]
                    + backward[feature + 1, self.k - 1 - left]
                    for left in range(self.k)
                ],
                dtype=np.float64,
            )
            finite = terms[np.isfinite(terms)]
            if finite.size == 0:
                raise RuntimeError("cannot form an exact-k posterior support")
            maximum = float(np.max(finite))
            log_excluding_feature = maximum + math.log(
                float(np.exp(finite - maximum).sum())
            )
            probabilities[feature] = math.exp(
                log_weights[feature]
                + log_excluding_feature
                - log_denominator
            )
        return probabilities
