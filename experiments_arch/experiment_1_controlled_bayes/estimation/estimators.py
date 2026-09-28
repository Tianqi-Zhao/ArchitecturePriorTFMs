"""Interchangeable coefficient estimators; only NumPy is required."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from ..artifacts.protocol import UnavailableError


@dataclass(frozen=True)
class Projection:
    x: np.ndarray
    prediction: np.ndarray
    gradient_mean: np.ndarray | None = None
    gradient_count: int = 0

    def validate(self):
        if self.x.ndim != 2 or not len(self.x):
            raise ValueError("projection queries must have shape (Qp, d), Qp > 0")
        if self.prediction.shape != (len(self.x),):
            raise ValueError("projection predictions must have shape (Qp,)")
        if not np.isfinite(self.x).all() or not np.isfinite(self.prediction).all():
            raise ValueError("projection contains nonfinite values")


@dataclass(frozen=True)
class Estimate:
    eta: np.ndarray
    zeta: float = 0.0


class EtaEstimator(ABC):
    name: str
    intercept_mode = "fixed_zero"
    version = "1"

    @property
    def configuration(self):
        return {"algorithm_version": self.version,
                "intercept_mode": self.intercept_mode}

    @abstractmethod
    def estimate(self, material: Projection) -> Estimate:
        pass


class MomentEstimator(EtaEstimator):
    name = "moment"

    def estimate(self, material):
        material.validate()
        x = np.asarray(material.x, dtype=np.float64)
        f = np.asarray(material.prediction, dtype=np.float64)
        return Estimate(x.T @ f / len(x))


class GradientEstimator(EtaEstimator):
    name = "gradient"

    def estimate(self, material):
        material.validate()
        if material.gradient_mean is None or material.gradient_count != len(material.x):
            raise UnavailableError("input gradients must cover all projection queries")
        eta = np.asarray(material.gradient_mean, dtype=np.float64)
        if eta.shape != (material.x.shape[1],) or not np.isfinite(eta).all():
            raise ValueError("invalid mean input gradient")
        return Estimate(eta)


class AffineBLPEstimator(EtaEstimator):
    name = "affine_blp"
    intercept_mode = "fitted"

    @property
    def configuration(self):
        return {**super().configuration, "solver": "numpy.linalg.lstsq",
                "rcond": None}

    def fit(self, material):
        material.validate()
        x = np.asarray(material.x, dtype=np.float64)
        if len(x) <= x.shape[1] + 1:
            raise ValueError("affine BLP requires Qp > d + 1")
        design = np.column_stack((np.ones(len(x)), x))
        coefficient, _, rank, singular = np.linalg.lstsq(
            design, np.asarray(material.prediction, dtype=np.float64), rcond=None)
        if rank != design.shape[1]:
            raise ValueError("affine BLP projection design is not full rank")
        if not np.isfinite(coefficient).all():
            raise ValueError("affine BLP returned nonfinite coefficients")
        return Estimate(coefficient[1:], float(coefficient[0])), rank, singular

    def estimate(self, material):
        return self.fit(material)[0]


def make_estimator(name):
    constructors = {"moment": MomentEstimator, "gradient": GradientEstimator,
                    "affine_blp": AffineBLPEstimator}
    if name not in constructors:
        raise ValueError("choose a method: moment, gradient, affine_blp")
    return constructors[name]()
