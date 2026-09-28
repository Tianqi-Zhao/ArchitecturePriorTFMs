"""Dependency-light helpers shared by model adapter implementations.

This module deliberately knows nothing about either experiment package.  The
legacy, feature-rich adapters and the small prediction adapters used by the new
experiment architecture both rely on the same fit-cache and compatibility
policies defined here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


def resolve_existing_checkpoint(path: str | Path, label: str) -> str:
    """Return an existing checkpoint path or raise a contextual error."""

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {resolved}")
    return str(resolved)


def construct_with_optional_kwargs(
    cls: Callable[..., Any],
    required: Mapping[str, Any],
    optional: Mapping[str, Any],
) -> Any:
    """Instantiate a version-varying external class.

    TabPFN, TabICL, and TabDPT have changed a few constructor keyword names
    between releases.  Required arguments are never discarded.  An optional
    argument is retried without only when the constructor's ``TypeError`` names
    that argument, preserving compatibility without hiding unrelated failures.
    """

    remaining = dict(optional)
    while True:
        try:
            return cls(**dict(required), **remaining)
        except TypeError as error:
            message = str(error)
            rejected = next((key for key in remaining if key in message), None)
            if rejected is None:
                raise
            remaining.pop(rejected)


def set_rebuilding_random_seed(adapter: Any, seed: int, prepare_device: Any) -> bool:
    """Install a constructor-bound seed and rebuild a prepared estimator.

    Returns whether the seed changed.  ``prepare`` owns fitted-cache
    invalidation, so all constructor-seeded backends follow one policy.
    """

    seed = int(seed)
    if seed == adapter.seed:
        return False
    adapter.seed = seed
    if adapter.est is not None:
        adapter.prepare(prepare_device)
    return True


def _array_equal_with_matching_nan(left: np.ndarray, right: np.ndarray) -> bool:
    if np.array_equal(left, right):
        return True
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        # NumPy's equal_nan path rejects some string/object arrays.
        for left_value, right_value in zip(left.flat, right.flat):
            try:
                if bool(left_value == right_value):
                    continue
            except (TypeError, ValueError):
                pass
            try:
                if bool(np.isnan(left_value)) and bool(np.isnan(right_value)):
                    continue
            except (TypeError, ValueError):
                pass
            return False
        return True


def same_fit_inputs(
    X: np.ndarray,
    y: np.ndarray,
    cached_X: np.ndarray | None,
    cached_y: np.ndarray | None,
) -> bool:
    """Whether cached training inputs represent the same fitted episode."""

    return (
        cached_X is not None
        and cached_y is not None
        and cached_X.shape == X.shape
        and cached_y.shape == y.shape
        and _array_equal_with_matching_nan(cached_X, X)
        and _array_equal_with_matching_nan(cached_y, y)
    )


__all__ = [
    "construct_with_optional_kwargs",
    "resolve_existing_checkpoint",
    "same_fit_inputs",
    "set_rebuilding_random_seed",
]
