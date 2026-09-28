"""Regression scaling and metrics shared across experiment architectures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionTargetScaler:
    """Context-fitted target standardization used by pretrained models."""

    mean: float
    std: float

    @classmethod
    def fit(cls, y_context: np.ndarray, eps: float = 1e-8) -> "RegressionTargetScaler":
        y_context = np.asarray(y_context, dtype=np.float32)
        mean = float(np.mean(y_context))
        std = float(np.std(y_context))
        return cls(mean=mean, std=std if std >= eps else 1.0)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float32) - self.mean) / self.std

    def inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        return np.asarray(y_scaled, dtype=np.float64) * self.std + self.mean


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


__all__ = ["RegressionTargetScaler", "mae", "r2_score", "rmse"]
