"""Derive IV.7 ratios from saved C/Z/N/E and sidecar L; do not recompute eta metrics."""

import numpy as np

from ..artifacts.nonlinear_protocol import RATIOS
from ..artifacts.protocol import EvaluationConfig
from .metrics import MeasurementCalculator


class NonlinearCalculator:
    def __init__(self, threshold=1e-12):
        self.measurements = MeasurementCalculator(EvaluationConfig(threshold))

    @staticmethod
    def ratio_components(metrics, name):
        numerator, denominators = RATIOS[name]
        value = np.asarray(metrics[numerator], dtype=np.float64)
        if name == "abs_gap_to_error":
            value = np.abs(value)
        return value, sum(np.asarray(metrics[key], dtype=np.float64) for key in denominators)

    def ratio(self, metrics, name):
        return self.measurements.ratio(*self.ratio_components(metrics, name))
