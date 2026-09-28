"""Shared, train-fitted feature preprocessing for OpenML tables.

The transformer intentionally keeps one output column per usable input column:
numeric columns are median-imputed, categoricals are ordinal-encoded, and
query-only categories receive a finite sentinel code. All fitting decisions
come from the context/training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _as_feature_frame(X):
    import pandas as pd

    return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def apply_openml_categorical_indicator(X, categorical_indicator: Iterable[bool]):
    """Return a frame whose OpenML-declared categorical columns stay categorical."""

    frame = _as_feature_frame(X)
    indicators = tuple(bool(value) for value in categorical_indicator)
    if len(indicators) != len(frame.columns):
        raise ValueError(
            "OpenML categorical indicator has "
            f"{len(indicators)} entries for {len(frame.columns)} feature columns"
        )
    for column_name, is_categorical in zip(frame.columns, indicators):
        if is_categorical:
            frame[column_name] = frame[column_name].astype("category")
    return frame


@dataclass(frozen=True)
class _FittedColumn:
    name: object
    kind: str
    fill: float | None = None
    mapping: dict[str, int] | None = None
    unknown_code: float | None = None


@dataclass(frozen=True)
class OpenMLFeaturePreprocessor:
    """A numeric/categorical policy fitted on context or training rows."""

    columns: tuple[_FittedColumn, ...]

    @classmethod
    def fit(cls, X) -> "OpenMLFeaturePreprocessor":
        import pandas as pd

        frame = _as_feature_frame(X)
        numeric_columns: list[_FittedColumn] = []
        categorical_columns: list[_FittedColumn] = []
        for column_name in frame.columns:
            column = frame[column_name]
            if int(column.nunique(dropna=True)) <= 1:
                continue

            declared_categorical = isinstance(column.dtype, pd.CategoricalDtype)
            numeric = pd.to_numeric(column, errors="coerce")
            non_missing = int(column.notna().sum())
            if (
                not declared_categorical
                and int(numeric.notna().sum()) == non_missing
            ):
                values = numeric.to_numpy(dtype=np.float64, copy=True)
                finite = np.isfinite(values)
                fill = float(np.median(values[finite])) if finite.any() else 0.0
                numeric_columns.append(
                    _FittedColumn(name=column_name, kind="numeric", fill=fill)
                )
                continue

            strings = column.astype("string").fillna("<missing>")
            categories = sorted(str(value) for value in strings.unique())
            mapping = {value: index for index, value in enumerate(categories)}
            categorical_columns.append(
                _FittedColumn(
                    name=column_name,
                    kind="categorical",
                    mapping=mapping,
                    unknown_code=float(len(mapping)),
                )
            )

        columns = tuple(numeric_columns + categorical_columns)
        if not columns:
            raise ValueError("OpenML table has no non-constant feature columns")
        return cls(columns)

    def transform(self, X) -> np.ndarray:
        import pandas as pd

        frame = _as_feature_frame(X)
        columns: list[np.ndarray] = []
        for spec in self.columns:
            column = frame[spec.name]
            if spec.kind == "numeric":
                values = pd.to_numeric(column, errors="coerce").to_numpy(
                    dtype=np.float64, copy=True
                )
                values[~np.isfinite(values)] = float(spec.fill)
                columns.append(values)
                continue

            strings = column.astype("string").fillna("<missing>")
            mapping = spec.mapping or {}
            unknown = float(spec.unknown_code)
            columns.append(
                np.asarray(
                    [mapping.get(str(value), unknown) for value in strings],
                    dtype=np.float64,
                )
            )

        matrix = np.column_stack(columns).astype(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError("OpenML preprocessing produced non-finite features")
        return matrix


def fit_transform_openml_features(X_train, X_test) -> tuple[np.ndarray, np.ndarray]:
    """Fit on ``X_train`` and transform both splits with the same policy."""

    preprocessor = OpenMLFeaturePreprocessor.fit(X_train)
    return preprocessor.transform(X_train), preprocessor.transform(X_test)
