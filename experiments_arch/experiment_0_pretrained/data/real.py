"""OpenML generator implementation slot."""

from __future__ import annotations

from typing import Iterable

from experiment_core.openml import (
    OpenMLConfig,
    load_openml_regression_split,
    select_real_regression_datasets,
)
from experiments_arch.experiment_0_pretrained.data.base import (
    BaseDataset,
    DatasetGenerator,
    DatasetSource,
)


class OpenMLGenerator(DatasetGenerator):
    """Prepare the pinned real-regression tables as clean base datasets."""

    def __init__(self, config: OpenMLConfig) -> None:
        self.config = config

    def generate_base(self) -> Iterable[BaseDataset]:
        cfg = self.config
        for spec in select_real_regression_datasets(cfg.dataset_ids):
            split = load_openml_regression_split(
                spec,
                max_rows=cfg.max_rows,
                context_fraction=cfg.context_fraction,
                seed=cfg.seed,
                cache_dir=cfg.cache_dir,
            )
            yield BaseDataset(
                dataset_id=f"openml-{spec.dataset_id}",
                dataset_name=spec.name,
                source=DatasetSource.REAL,
                X_context=split.X_context,
                y_context=split.y_context,
                X_query=split.X_query,
                y_query=split.y_query,
                context_row_id=split.context_row_id,
                query_row_id=split.query_row_id,
                relevant_feature_count=None,
                metadata={
                    "generator": "openml_curated_regression",
                    "openml_dataset_id": spec.dataset_id,
                    "catalog_n_rows": spec.n_rows,
                    "catalog_n_features": spec.n_features,
                    "max_rows": cfg.max_rows,
                    "context_fraction": cfg.context_fraction,
                    "split_seed": cfg.seed,
                    "preprocessing": "fit_on_context_only",
                },
            )


__all__ = ["OpenMLConfig", "OpenMLGenerator"]
