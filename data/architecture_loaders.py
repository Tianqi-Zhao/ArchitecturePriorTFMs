"""Data-loader factory driven by an architecture's declared input policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.synthetic_data.datasets import PriorDumpDataLoader, PriorDumpDataLoader1D
from models.architecture_registry import ArchitectureSpec


def build_synthetic_loader(
    architecture: ArchitectureSpec,
    filename: str | Path,
    *,
    num_steps: int,
    batch_size: int,
    num_features: int,
    request_feature_shuffle: bool,
    device: Any,
):
    """Build the correct prior loader without branching on a model name."""

    shuffle = architecture.training_shuffle(request_feature_shuffle)
    common = dict(
        filename=str(filename),
        num_steps=num_steps,
        batch_size=batch_size,
        shuffle_features=shuffle,
        device=device,
    )
    if architecture.input_policy == "fixed_width":
        return PriorDumpDataLoader1D(num_features=num_features, **common)
    return PriorDumpDataLoader(**common)
