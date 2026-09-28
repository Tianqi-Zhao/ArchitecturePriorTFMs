"""Friedman generator implementation slot."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from experiments_arch.experiment_0_pretrained.data.base import (
    BaseDataset,
    DatasetGenerator,
    DatasetSource,
)


@dataclass(frozen=True)
class FriedmanConfig:
    num_tasks: int = 200
    n_context: int = 800
    n_query: int = 200
    relevant_features: int = 5
    noise_std: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_tasks < 1 or self.n_context < 1 or self.n_query < 1:
            raise ValueError("num_tasks, n_context, and n_query must be positive")
        if self.relevant_features != 5:
            raise ValueError("Friedman #1 requires exactly five relevant features")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


class FriedmanGenerator(DatasetGenerator):
    """Generate deterministic Friedman #1 regression tasks."""

    def __init__(self, config: FriedmanConfig) -> None:
        self.config = config

    def generate_base(self) -> Iterable[BaseDataset]:
        cfg = self.config
        for task_index in range(cfg.num_tasks):
            yield self.generate_task(task_index)

    def generate_task(self, task_index: int) -> BaseDataset:
        """Generate one indexed task without constructing the other tasks."""

        cfg = self.config
        task_index = int(task_index)
        if not 0 <= task_index < cfg.num_tasks:
            raise ValueError(f"task_index must lie in [0, {cfg.num_tasks})")
        feature_rng = np.random.default_rng(
            np.random.SeedSequence([cfg.seed, task_index, 0])
        )
        noise_rng = np.random.default_rng(
            np.random.SeedSequence([cfg.seed, task_index, 1])
        )
        n_total = cfg.n_context + cfg.n_query
        X = feature_rng.random((n_total, cfg.relevant_features), dtype=np.float32)
        y_signal = (
            10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20.0 * np.square(X[:, 2] - 0.5)
            + 10.0 * X[:, 3]
            + 5.0 * X[:, 4]
        )
        noise = noise_rng.normal(0.0, cfg.noise_std, size=n_total)
        y = (y_signal + noise).astype(np.float32)
        split = cfg.n_context
        dataset_id = f"friedman1-task-{task_index:03d}"
        return BaseDataset(
            dataset_id=dataset_id,
            dataset_name=dataset_id,
            source=DatasetSource.SYNTHETIC,
            X_context=X[:split],
            y_context=y[:split],
            X_query=X[split:],
            y_query=y[split:],
            context_row_id=np.arange(split, dtype=np.int64),
            query_row_id=np.arange(split, n_total, dtype=np.int64),
            relevant_feature_count=cfg.relevant_features,
            metadata={
                "generator": "friedman_1",
                "formula": "10*sin(pi*x0*x1)+20*(x2-0.5)^2+10*x3+5*x4",
                "task_index": task_index,
                "generator_seed": cfg.seed,
                "noise_std": cfg.noise_std,
            },
        )

