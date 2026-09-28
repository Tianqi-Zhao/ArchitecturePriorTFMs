"""Curated OpenML regression data independent of any experiment package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from data.openml_preprocessing import (
    apply_openml_categorical_indicator,
    fit_transform_openml_features,
)


@dataclass(frozen=True)
class RealRegressionDatasetSpec:
    """Stable identity and raw dimensions of one pinned OpenML table."""

    dataset_id: int
    name: str
    n_rows: int
    n_features: int


@dataclass(frozen=True)
class OpenMLConfig:
    """JSON-safe settings for loading a curated OpenML subset."""

    dataset_ids: tuple[int, ...]
    max_rows: int = 1000
    context_fraction: float = 0.8
    cache_dir: str = ""
    seed: int = 0

    def __post_init__(self) -> None:
        ids = tuple(int(value) for value in self.dataset_ids)
        if not ids:
            raise ValueError("dataset_ids must not be empty")
        if len(ids) != len(set(ids)):
            raise ValueError("dataset_ids must not contain duplicates")
        if self.max_rows < 4:
            raise ValueError("max_rows must be at least four")
        if not 0.0 < self.context_fraction < 1.0:
            raise ValueError("context_fraction must lie in (0, 1)")
        object.__setattr__(self, "dataset_ids", ids)


@dataclass(frozen=True)
class OpenMLRegressionSplit:
    """A downloaded table after deterministic sampling and preprocessing."""

    spec: RealRegressionDatasetSpec
    X_context: np.ndarray
    y_context: np.ndarray
    X_query: np.ndarray
    y_query: np.ndarray
    context_row_id: np.ndarray
    query_row_id: np.ndarray

    def __post_init__(self) -> None:
        X_context = np.asarray(self.X_context, dtype=np.float32)
        X_query = np.asarray(self.X_query, dtype=np.float32)
        y_context = np.asarray(self.y_context, dtype=np.float32).reshape(-1)
        y_query = np.asarray(self.y_query, dtype=np.float32).reshape(-1)
        context_ids = np.asarray(self.context_row_id, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_row_id, dtype=np.int64).reshape(-1)
        if X_context.ndim != 2 or X_query.ndim != 2:
            raise ValueError("OpenML features must be two-dimensional")
        if X_context.shape[1] != X_query.shape[1] or X_context.shape[1] < 1:
            raise ValueError("OpenML context/query must share a positive width")
        if len(X_context) != len(y_context) or len(X_query) != len(y_query):
            raise ValueError("OpenML feature and target row counts do not match")
        if len(context_ids) != len(X_context) or len(query_ids) != len(X_query):
            raise ValueError("OpenML row ids do not match the sampled rows")
        arrays = (X_context, X_query, y_context, y_query)
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("OpenML split contains non-finite values")
        object.__setattr__(self, "X_context", X_context)
        object.__setattr__(self, "X_query", X_query)
        object.__setattr__(self, "y_context", y_context)
        object.__setattr__(self, "y_query", y_query)
        object.__setattr__(self, "context_row_id", context_ids)
        object.__setattr__(self, "query_row_id", query_ids)

    @property
    def n(self) -> int:
        return len(self.X_context) + len(self.X_query)

    @property
    def d(self) -> int:
        return int(self.X_context.shape[1])


OWN_REAL_REGRESSION_DATASETS: tuple[RealRegressionDatasetSpec, ...] = (
    RealRegressionDatasetSpec(46904, "airfoil_self_noise", 1503, 5),
    RealRegressionDatasetSpec(46907, "Another-Dataset-on-used-Fiat-500", 1538, 7),
    RealRegressionDatasetSpec(46917, "concrete_compressive_strength", 1030, 8),
    RealRegressionDatasetSpec(46923, "diamonds", 53940, 9),
    RealRegressionDatasetSpec(46928, "Food_Delivery_Time", 45451, 9),
    RealRegressionDatasetSpec(46931, "healthcare_insurance_expenses", 1338, 6),
    RealRegressionDatasetSpec(46934, "houses", 20640, 8),
    RealRegressionDatasetSpec(46949, "physiochemical_protein", 45730, 9),
    RealRegressionDatasetSpec(46954, "QSAR_fish_toxicity", 907, 6),
)

OPENML_CTR23_REAL_REGRESSION_DATASETS: tuple[RealRegressionDatasetSpec, ...] = (
    RealRegressionDatasetSpec(41021, "Moneyball", 1232, 14),
    RealRegressionDatasetSpec(44956, "abalone", 4177, 8),
    RealRegressionDatasetSpec(44957, "airfoil_self_noise", 1503, 5),
    RealRegressionDatasetSpec(44958, "auction_verification", 2043, 7),
    RealRegressionDatasetSpec(44959, "concrete_compressive_strength", 1030, 8),
    RealRegressionDatasetSpec(44960, "energy_efficiency", 768, 8),
    RealRegressionDatasetSpec(44962, "forest_fires", 517, 12),
    RealRegressionDatasetSpec(44963, "physiochemical_protein", 45730, 9),
    RealRegressionDatasetSpec(44964, "superconductivity", 21263, 81),
    RealRegressionDatasetSpec(44965, "geographical_origin_of_music", 1059, 116),
    RealRegressionDatasetSpec(44966, "solar_flare", 1066, 10),
    RealRegressionDatasetSpec(44967, "student_performance_por", 649, 30),
    RealRegressionDatasetSpec(44969, "naval_propulsion_plant", 11934, 14),
    RealRegressionDatasetSpec(44970, "QSAR_fish_toxicity", 908, 6),
    RealRegressionDatasetSpec(44971, "white_wine", 4898, 11),
    RealRegressionDatasetSpec(44972, "red_wine", 1599, 11),
    RealRegressionDatasetSpec(44973, "grid_stability", 10000, 12),
    RealRegressionDatasetSpec(44974, "video_transcoding", 68784, 18),
    RealRegressionDatasetSpec(44975, "wave_energy", 72000, 48),
    RealRegressionDatasetSpec(44976, "sarcos", 48933, 21),
    RealRegressionDatasetSpec(44977, "california_housing", 20640, 8),
    RealRegressionDatasetSpec(44978, "cpu_activity", 8192, 21),
    RealRegressionDatasetSpec(44979, "diamonds", 53940, 9),
    RealRegressionDatasetSpec(44980, "kin8nm", 8192, 8),
    RealRegressionDatasetSpec(44981, "pumadyn32nh", 8192, 32),
    RealRegressionDatasetSpec(44983, "miami_housing", 13932, 15),
    RealRegressionDatasetSpec(44984, "cps88wages", 28155, 6),
    RealRegressionDatasetSpec(44987, "socmob", 1156, 5),
    RealRegressionDatasetSpec(44989, "kings_county", 21613, 21),
    RealRegressionDatasetSpec(44990, "brazilian_houses", 10692, 9),
    RealRegressionDatasetSpec(44992, "fps_benchmark", 24624, 43),
    RealRegressionDatasetSpec(44993, "health_insurance", 22272, 11),
    RealRegressionDatasetSpec(44994, "cars", 804, 17),
    RealRegressionDatasetSpec(45012, "fifa", 19178, 28),
    RealRegressionDatasetSpec(45402, "space_ga", 3107, 6),
)

PRETRAINED_EXTRA_REAL_REGRESSION_DATASETS: tuple[
    RealRegressionDatasetSpec, ...
] = (
    RealRegressionDatasetSpec(196, "autoMpg", 398, 7),
    RealRegressionDatasetSpec(230, "machine_cpu", 209, 6),
    RealRegressionDatasetSpec(560, "bodyfat", 252, 14),
    RealRegressionDatasetSpec(42370, "yacht_hydrodynamics", 308, 6),
    RealRegressionDatasetSpec(44146, "medical_charges", 163065, 3),
    RealRegressionDatasetSpec(46286, "communities_and_crime", 1994, 122),
    RealRegressionDatasetSpec(46292, "servo", 167, 4),
)

PRETRAINED_REAL_REGRESSION_DATASETS = (
    *OPENML_CTR23_REAL_REGRESSION_DATASETS,
    *PRETRAINED_EXTRA_REAL_REGRESSION_DATASETS,
)
REAL_REGRESSION_DATASETS = (
    *OWN_REAL_REGRESSION_DATASETS,
    *PRETRAINED_REAL_REGRESSION_DATASETS,
)
DEFAULT_OWN_REAL_REGRESSION_DATASET_IDS = tuple(
    spec.dataset_id for spec in OWN_REAL_REGRESSION_DATASETS
)
DEFAULT_PRETRAINED_REAL_REGRESSION_DATASET_IDS = tuple(
    spec.dataset_id for spec in PRETRAINED_REAL_REGRESSION_DATASETS
)
DEFAULT_REAL_REGRESSION_DATASET_IDS_BY_ORIGIN = {
    "own": DEFAULT_OWN_REAL_REGRESSION_DATASET_IDS,
    "pretrained": DEFAULT_PRETRAINED_REAL_REGRESSION_DATASET_IDS,
    "tfm_prior": DEFAULT_OWN_REAL_REGRESSION_DATASET_IDS,
}
DEFAULT_REAL_REGRESSION_DATASET_IDS = DEFAULT_OWN_REAL_REGRESSION_DATASET_IDS
_SPECS_BY_ID = {spec.dataset_id: spec for spec in REAL_REGRESSION_DATASETS}


def select_real_regression_datasets(
    dataset_ids: Iterable[int] | None = None,
) -> tuple[RealRegressionDatasetSpec, ...]:
    """Resolve curated ids while preserving the caller's declared order."""

    ids = DEFAULT_REAL_REGRESSION_DATASET_IDS if dataset_ids is None else tuple(
        int(dataset_id) for dataset_id in dataset_ids
    )
    if not ids:
        raise ValueError("at least one real regression dataset id is required")
    unknown = [dataset_id for dataset_id in ids if dataset_id not in _SPECS_BY_ID]
    if unknown:
        raise ValueError(
            f"unknown curated real-regression dataset id(s) {unknown}; available: "
            f"{sorted(_SPECS_BY_ID)}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate real-regression dataset ids: {list(ids)}")
    return tuple(_SPECS_BY_ID[dataset_id] for dataset_id in ids)


def _as_feature_frame(X):
    import pandas as pd

    return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def make_openml_regression_split(
    spec: RealRegressionDatasetSpec,
    X,
    y,
    *,
    max_rows: int,
    context_fraction: float,
    seed: int,
) -> OpenMLRegressionSplit:
    """Create a deterministic, context-fitted split from downloaded data."""

    import pandas as pd

    if max_rows < 4:
        raise ValueError(f"real max_rows must be >= 4, got {max_rows}")
    if not 0.0 < context_fraction < 1.0:
        raise ValueError(
            f"real context_fraction must be in (0, 1), got {context_fraction}"
        )
    raw_rows, raw_features = int(len(y)), int(np.asarray(X).shape[1])
    if (raw_rows, raw_features) != (spec.n_rows, spec.n_features):
        raise ValueError(
            f"OpenML dataset {spec.dataset_id} ({spec.name}) changed dimensions: "
            f"got {raw_rows}x{raw_features}, expected {spec.n_rows}x{spec.n_features}"
        )

    target = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64)
    valid_target = np.isfinite(target)
    if int(valid_target.sum()) < 4:
        raise ValueError(f"real dataset {spec.name!r} has fewer than four finite targets")
    source_row_ids = np.flatnonzero(valid_target).astype(np.int64)
    frame = _as_feature_frame(X)
    if not valid_target.all():
        frame = frame.iloc[source_row_ids].reset_index(drop=True)
        target = target[valid_target]

    rng = np.random.default_rng([int(seed), int(spec.dataset_id)])
    count = min(int(max_rows), len(target))
    rows = rng.permutation(len(target))[:count]
    sampled_row_ids = source_row_ids[rows]
    frame = frame.iloc[list(rows)].reset_index(drop=True)
    target = np.asarray(target[rows], dtype=np.float32)
    split = min(max(int(round(context_fraction * count)), 2), count - 2)
    if float(np.std(target[split:])) < 1e-8:
        raise ValueError(f"real dataset {spec.name!r} has a constant query target")

    X_context, X_query = fit_transform_openml_features(
        frame.iloc[:split], frame.iloc[split:]
    )
    if X_context.shape[1] > spec.n_features:
        raise ValueError(
            f"real dataset {spec.name!r} has {X_context.shape[1]} usable columns after "
            f"preprocessing, expected at most {spec.n_features}"
        )
    return OpenMLRegressionSplit(
        spec=spec,
        X_context=X_context,
        y_context=target[:split],
        X_query=X_query,
        y_query=target[split:],
        context_row_id=sampled_row_ids[:split],
        query_row_id=sampled_row_ids[split:],
    )


def load_openml_regression_split(
    spec: RealRegressionDatasetSpec,
    *,
    max_rows: int,
    context_fraction: float,
    seed: int,
    cache_dir: str = "",
) -> OpenMLRegressionSplit:
    """Download one pinned OpenML table and return its prepared split."""

    import openml

    if cache_dir:
        openml.config.set_root_cache_directory(str(cache_dir))
    dataset = openml.datasets.get_dataset(
        int(spec.dataset_id),
        download_data=False,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    X = apply_openml_categorical_indicator(X, categorical_indicator)
    return make_openml_regression_split(
        spec,
        X,
        y,
        max_rows=max_rows,
        context_fraction=context_fraction,
        seed=seed,
    )


__all__ = [
    "DEFAULT_OWN_REAL_REGRESSION_DATASET_IDS",
    "DEFAULT_PRETRAINED_REAL_REGRESSION_DATASET_IDS",
    "DEFAULT_REAL_REGRESSION_DATASET_IDS",
    "DEFAULT_REAL_REGRESSION_DATASET_IDS_BY_ORIGIN",
    "OPENML_CTR23_REAL_REGRESSION_DATASETS",
    "OWN_REAL_REGRESSION_DATASETS",
    "OpenMLConfig",
    "OpenMLRegressionSplit",
    "PRETRAINED_EXTRA_REAL_REGRESSION_DATASETS",
    "PRETRAINED_REAL_REGRESSION_DATASETS",
    "REAL_REGRESSION_DATASETS",
    "RealRegressionDatasetSpec",
    "load_openml_regression_split",
    "make_openml_regression_split",
    "select_real_regression_datasets",
]
