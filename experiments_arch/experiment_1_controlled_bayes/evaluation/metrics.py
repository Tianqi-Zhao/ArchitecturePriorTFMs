"""Offline exact-k formulas; all arithmetic uses float64."""

import numpy as np

from ..artifacts.protocol import EvaluationConfig


class MeasurementCalculator:
    def __init__(self, config=EvaluationConfig()):
        self.threshold = config.denominator_threshold

    def ratio(self, numerator, denominator):
        if not np.isfinite(denominator) or denominator <= self.threshold:
            return float("nan")
        return float(numerator / denominator)

    def public(self, prediction, x, y, bayes, baseline=None):
        f, x, y = (np.asarray(value, dtype=np.float64) for value in (prediction, x, y))
        error = f - x @ bayes
        observed = f - y
        mse = float(np.mean(observed ** 2))
        result = dict(E=float(np.mean(error ** 2)), mse=mse, rmse=float(np.sqrt(mse)),
                      mae=float(np.mean(np.abs(observed))),
                      r2=1 - self.ratio(np.sum(observed ** 2), np.sum((y - y.mean()) ** 2)))
        result.update(delta_E=np.nan, H_delta=np.nan, mean_prediction_shift=np.nan)
        if baseline is not None:
            baseline = np.asarray(baseline, dtype=np.float64)
            delta = f - baseline
            result.update(delta_E=result["E"] - float(np.mean((baseline - x @ bayes) ** 2)),
                          H_delta=float(np.mean(delta ** 2)),
                          mean_prediction_shift=float(delta.mean()))
        return result

    def coefficients(self, eta, zeta, bayes, support):
        difference = (np.asarray(eta, dtype=np.float64) - bayes) ** 2
        return dict(C=float(difference.sum()), Z=float(zeta ** 2),
                    C_active=float(difference[support].sum()),
                    C_inactive=float(difference[~support].sum()) if np.any(~support) else np.nan)

    @staticmethod
    def projection_design(x):
        design = np.column_stack((np.ones(len(x)), np.asarray(x, dtype=np.float64)))
        singular = np.linalg.svd(design, compute_uv=False)
        cutoff = np.finfo(np.float64).eps * max(design.shape)
        rank = int(np.sum(singular > cutoff * singular[0]))
        condition = float(singular[0] / singular[-1]) if rank == design.shape[1] else np.inf
        return dict(design_rank=rank, design_condition=condition)

    def method(self, eta, zeta, bayes, support, *, x=None, prediction=None,
               baseline_eta=None, baseline_zeta=0.0, baseline_prediction=None):
        result = self.coefficients(eta, zeta, bayes, support)
        result.update(N=np.nan, accounting_gap=np.nan, rho_lin=np.nan,
                      rho_delta=np.nan, mean_delta_residual=np.nan)
        for name in ("C", "Z", "N", "C_active", "C_inactive", "accounting_gap"):
            result[f"delta_{name}"] = np.nan
        residual = None
        if prediction is not None:
            x, f = np.asarray(x, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
            residual = f - zeta - x @ eta
            n = float(np.mean(residual ** 2))
            e = float(np.mean((f - x @ bayes) ** 2))
            result.update(N=n, accounting_gap=e - result["C"] - result["Z"] - n,
                          rho_lin=1 - self.ratio(np.sum(residual ** 2), np.sum((f - f.mean()) ** 2)))
        if baseline_eta is not None:
            base = self.coefficients(baseline_eta, baseline_zeta, bayes, support)
            for name in base:
                result[f"delta_{name}"] = result[name] - base[name]
            if residual is not None and baseline_prediction is not None:
                base_f = np.asarray(baseline_prediction, dtype=np.float64)
                base_r = base_f - baseline_zeta - x @ baseline_eta
                base_n = float(np.mean(base_r ** 2))
                base_e = float(np.mean((base_f - x @ bayes) ** 2))
                delta_f = f - base_f
                centered = delta_f - delta_f.mean()
                remainder = centered - (x - x.mean(axis=0)) @ (eta - baseline_eta)
                result.update(
                    delta_N=result["N"] - base_n,
                    delta_accounting_gap=result["accounting_gap"] - (
                        base_e - base["C"] - base["Z"] - base_n),
                    rho_delta=1 - self.ratio(np.sum(remainder ** 2), np.sum(centered ** 2)),
                    mean_delta_residual=float(np.mean(residual - base_r)))
        return result

    def response(self, matrix, groups):
        """Return own-feature strength and the equal-source mean specificity."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape[1] != len(groups) or not np.isfinite(matrix).all():
            return np.nan, np.nan
        magnitude = np.abs(matrix)
        own = np.array([magnitude[sorted({m["feature"] for m in members}), j].sum()
                        for j, members in enumerate(groups)], dtype=np.float64)
        total = magnitude.sum(axis=0)
        valid = np.isfinite(total) & (total > self.threshold)
        specificity = float(np.mean(own[valid] / total[valid])) if valid.any() else np.nan
        return float(own.sum()), specificity
