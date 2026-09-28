"""Evaluate saved exact-k predictions/eta without importing any model library."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from ..estimation.bayes import ExactKBayesPredictor
from ..artifacts.index import PredictionIndex
from ..artifacts.protocol import EVALUATION_VERSION, SCHEMA_VERSION, EvaluationConfig
from ..artifacts.storage import STRING, PredictionRun, SavedEstimate, new_directory, reference, sha256, write_json
from .metrics import MeasurementCalculator

PUBLIC_FIELDS = ("E", "mse", "rmse", "mae", "r2", "delta_E", "H_delta", "mean_prediction_shift")
METHOD_FIELDS = ("C", "Z", "N", "C_active", "C_inactive", "accounting_gap", "rho_lin",
                 "rho_delta", "mean_delta_residual", "delta_C", "delta_Z", "delta_N",
                 "delta_C_active", "delta_C_inactive", "delta_accounting_gap")


def array(group, name, shape, dtype=np.float64, fillvalue=np.nan):
    return group.create_dataset(name, shape=shape, dtype=dtype, fillvalue=fillvalue)


def _text(dataset):
    return np.asarray(dataset.asstr()[:])


def _write_text(group, name, values):
    group.create_dataset(name, data=np.array([str(value) for value in values], dtype=STRING))


class SavedEvaluator:
    def __init__(self, run, *, output_dir, estimates, config=EvaluationConfig()):
        self.saved = PredictionRun(run)
        self.output_dir = Path(output_dir).expanduser().resolve()
        sources = [SavedEstimate(path, self.saved) for path in estimates]
        self.estimates = {source.method: source for source in sources}
        if not sources or len(self.estimates) != len(sources):
            raise ValueError("provide one estimate artifact per distinct method")
        self.methods = tuple(self.estimates)
        self.protocol = self.saved.protocol
        self.config, self.calculator = config, MeasurementCalculator(config)
        self.dataset = self.saved.dataset()
        self.queries = None
        if config.projection_diagnostics:
            from ..artifacts.data import ProjectionQueries
            self.queries = ProjectionQueries(self.saved.input("projection_queries"), self.dataset)
        self.bayes = ExactKBayesPredictor(**asdict(self.dataset.prior))

    def _evaluate(self, destination, stack):
        dataset = self.dataset
        raw = stack.enter_context(self.saved.prediction())
        data_file = stack.enter_context(h5py.File(dataset.path, "r"))
        query_file = (stack.enter_context(h5py.File(self.queries.path, "r"))
                      if self.queries is not None else None)
        index = PredictionIndex.read(raw, dataset, self.protocol)
        t, s, d = len(index.task_ids), len(index.task_index), dataset.prior.d
        pred = raw["states"]
        eval_status = _text(pred["evaluation_status"])
        proj_status = _text(pred["projection_status"]) if query_file is not None else None
        eta_arrays = {}
        for name in self.methods:
            source = stack.enter_context(self.estimates[name].read())
            states = source["states"]
            if states["eta"].shape != (s, d) or states["status"].shape != (s,):
                raise ValueError("eta state rows do not align with predictions")
            status = _text(states["status"])
            if not set(status) <= {"ok", "failed", "unavailable"}:
                raise ValueError("eta file has invalid terminal states")
            if name == "affine_blp" and ("zeta" not in states or states["zeta"].shape != (s,)):
                raise ValueError("affine BLP requires one intercept per state")
            eta = np.asarray(states["eta"][:], dtype=np.float64)
            zeta = (np.asarray(states["zeta"][:], dtype=np.float64) if name == "affine_blp"
                    else np.zeros(s, dtype=np.float64))
            ok = status == "ok"
            if np.any(ok) and (not np.isfinite(eta[ok]).all() or not np.isfinite(zeta[ok]).all()):
                raise ValueError("eta marked ok contains invalid coefficients")
            eta_arrays[name] = (status, eta, zeta)

        bayes_eta = np.full((t, d), np.nan)
        bayes_signal = np.full(t, np.nan)
        support_mask = np.zeros((t, d), dtype=bool)
        public = {field: np.full(s, np.nan) for field in PUBLIC_FIELDS}
        public_status = np.full(s, "unavailable", dtype=object)
        public_reason = np.full(s, "", dtype=object)
        method_out = {}
        for name in self.methods:
            payload = {field: np.full(s, np.nan) for field in METHOD_FIELDS}
            if query_file is not None:
                payload["projection_mse"] = np.full(s, np.nan)
                if name == "affine_blp":
                    payload["design_rank"] = np.full(s, np.nan)
                    payload["design_condition"] = np.full(s, np.nan)
            payload["_status"] = np.full(s, "unavailable", dtype=object)
            payload["_reason"] = np.full(s, "", dtype=object)
            method_out[name] = payload

        for ti, task_id in enumerate(index.task_ids):
            task = dataset.read(int(task_id), data_file)
            eta_star = self.bayes.coefficients(task.context_x, task.context_y)
            bayes_eta[ti] = eta_star
            bayes_signal[ti] = eta_star @ eta_star
            support_mask[ti] = task.support
            baseline = int(index.baseline_ids[ti])
            rows = np.asarray(index.rows(ti), dtype=np.int64)
            length = len(task.evaluation_x)
            f_eval = np.asarray(pred["f_eval"][rows, :length], dtype=np.float64)
            row_eval_ok = eval_status[rows] == "ok"
            if np.any(row_eval_ok & ~np.isfinite(f_eval).all(axis=1)):
                raise ValueError("prediction marked ok contains nonfinite values")
            predictions = [f_eval[i] if row_eval_ok[i] else None for i in range(len(rows))]
            local_baseline = int(np.flatnonzero(rows == baseline)[0])
            base_prediction = predictions[local_baseline]
            projection_x, design, f_proj = None, None, None
            if query_file is not None:
                projection_x = np.asarray(
                    query_file["x_proj"][self.queries.index[int(task_id)]], dtype=np.float64)
                if not np.isfinite(projection_x).all():
                    raise ValueError(f"nonfinite projection queries for task {task_id}")
                if pred["f_proj"].shape[1] != len(projection_x):
                    raise ValueError("projection predictions and queries have different lengths")
                f_proj = np.asarray(pred["f_proj"][rows], dtype=np.float64)
                row_proj_ok = proj_status[rows] == "ok"
                if np.any(row_proj_ok & ~np.isfinite(f_proj).all(axis=1)):
                    raise ValueError("projection marked ok contains nonfinite values")
                if "affine_blp" in self.methods:
                    design = self.calculator.projection_design(projection_x)
            for local, row in enumerate(rows):
                prediction = predictions[local]
                is_baseline = row == baseline
                if prediction is not None:
                    values = self.calculator.public(
                        prediction, task.evaluation_x, task.evaluation_y, eta_star,
                        baseline=None if is_baseline else base_prediction)
                    for field, value in values.items():
                        public[field][row] = value
                    public_status[row] = "ok"
                    if np.isnan(values["r2"]):
                        public_reason[row] = "R2 denominator at or below threshold"
                else:
                    public_reason[row] = "evaluation prediction unavailable"
                for name, (status, eta, zeta) in eta_arrays.items():
                    output = method_out[name]
                    if design is not None and name == "affine_blp":
                        for field, value in design.items():
                            output[field][row] = value
                    if status[row] != "ok":
                        output["_reason"][row] = "eta estimate unavailable"
                        continue
                    estimate = (eta[row], float(zeta[row]))
                    base_estimate = ((eta[baseline], float(zeta[baseline]))
                                     if status[baseline] == "ok" else None)
                    values = self.calculator.method(
                        *estimate, eta_star, task.support, x=task.evaluation_x,
                        prediction=prediction,
                        baseline_eta=(base_estimate[0] if base_estimate and not is_baseline else None),
                        baseline_zeta=(base_estimate[1] if base_estimate else 0.0),
                        baseline_prediction=None if is_baseline else base_prediction)
                    for field, value in values.items():
                        output[field][row] = value
                    if projection_x is not None and proj_status[row] == "ok":
                        residual = f_proj[local] - estimate[1] - projection_x @ estimate[0]
                        output["projection_mse"][row] = float(np.mean(residual ** 2))
                    output["_status"][row] = "ok" if prediction is not None else "unavailable"
                    reasons = []
                    if prediction is None:
                        reasons.append("evaluation unavailable; coefficient measurements retained")
                    elif np.isnan(values["rho_lin"]):
                        reasons.append("rho_lin denominator at or below threshold")
                    if not is_baseline:
                        if base_estimate is None or base_prediction is None:
                            reasons.append("baseline unavailable for some changes")
                        elif prediction is not None and np.isnan(values["rho_delta"]):
                            reasons.append("rho_delta denominator at or below threshold")
                    if projection_x is not None and proj_status[row] != "ok":
                        reasons.append("projection prediction unavailable for fit diagnostics")
                    output["_reason"][row] = "; ".join(reasons)

        tasks = destination.create_group("tasks")
        tasks.create_dataset("task_id", data=index.task_ids)
        tasks.create_dataset("bayes_eta", data=bayes_eta)
        tasks.create_dataset("bayes_signal", data=bayes_signal)
        tasks.create_dataset("support_mask", data=support_mask)
        common = destination.create_group("states")
        _write_text(common, "kind", index.kind)
        for name, values in (("state_id", np.arange(s)), ("task_index", index.task_index),
                             ("layer", index.layer), ("source_token", index.source_token),
                             ("epsilon", index.epsilon)):
            common.create_dataset(name, data=values)
        for name in PUBLIC_FIELDS:
            common.create_dataset(name, data=public[name])
        _write_text(common, "status", public_status)
        _write_text(common, "reason", public_reason)
        for name, payload in method_out.items():
            group = destination.create_group(f"methods/{name}/states")
            for field in METHOD_FIELDS:
                group.create_dataset(field, data=payload[field])
            if query_file is not None:
                group.create_dataset("projection_mse", data=payload["projection_mse"])
                if name == "affine_blp":
                    group.create_dataset("design_rank", data=payload["design_rank"])
                    group.create_dataset("design_condition", data=payload["design_condition"])
            _write_text(group, "status", payload["_status"])
            _write_text(group, "reason", payload["_reason"])

        responses = list(index.responses(self.protocol))
        n = len(responses)
        max_sources = max((len(ids) for _, _, _, ids in responses), default=0)
        response_task = np.full(n, -1, dtype=np.int64)
        response_layer = np.full(n, -1, dtype=np.int64)
        response_eps = np.full(n, np.nan)
        response_ids = np.full((n, max_sources), -1, dtype=np.int64)
        response_out = {}
        for name in self.methods:
            response_out[name] = {
                "M": np.full((n, d, max_sources), np.nan),
                "strength": np.full(n, np.nan),
                "specificity": np.full(n, np.nan),
                "status": np.full(n, "unavailable", dtype=object),
                "reason": np.full(n, "", dtype=object),
            }
        for ri, (ti, layer, epsilon, state_ids) in enumerate(responses):
            response_task[ri], response_layer[ri] = ti, layer
            response_eps[ri] = epsilon
            response_ids[ri, :len(state_ids)] = state_ids
            baseline = int(index.baseline_ids[ti])
            for name, (status, eta, _zeta) in eta_arrays.items():
                group = response_out[name]
                base = eta[baseline] if status[baseline] == "ok" else None
                matrix = np.full((d, len(state_ids)), np.nan)
                for j, row in enumerate(state_ids):
                    if base is not None and row >= 0 and status[row] == "ok":
                        matrix[:, j] = (base - eta[row]) / epsilon
                group["M"][ri, :, :len(state_ids)] = matrix
                strength, specificity = self.calculator.response(
                    matrix, index.groups[ti][str(layer)])
                group["strength"][ri], group["specificity"][ri] = strength, specificity
                if np.isfinite(matrix).all():
                    group["status"][ri] = "ok"
                    if np.isnan(specificity):
                        group["reason"][ri] = "no source has a valid specificity denominator"
                else:
                    group["reason"][ri] = "baseline or source eta missing; matrix is incomplete"

        response = destination.create_group("responses")
        response.attrs["applicable"] = self.protocol.intervention_kind == "source"
        response.attrs["reason"] = ("" if self.protocol.intervention_kind == "source"
                                    else "source response metrics do not apply to feature_branch")
        response.create_dataset("task_index", data=response_task)
        response.create_dataset("layer", data=response_layer)
        response.create_dataset("epsilon", data=response_eps)
        response.create_dataset("state_ids", data=response_ids)
        for name, payload in response_out.items():
            group = destination.create_group(f"methods/{name}/responses")
            group.create_dataset("M", data=payload["M"])
            group.create_dataset("strength", data=payload["strength"])
            group.create_dataset("specificity", data=payload["specificity"])
            _write_text(group, "status", payload["status"])
            _write_text(group, "reason", payload["reason"])

    def run(self):
        with new_directory(self.output_dir) as directory:
            with ExitStack() as stack:
                output = stack.enter_context(h5py.File(directory / "metrics.h5", "x"))
                output.attrs.update(schema_version=SCHEMA_VERSION, complete=False)
                self._evaluate(output, stack)
                output.attrs["complete"] = True
            inputs = {"run": reference(self.saved.path, self.output_dir),
                      "predictions": reference(self.saved.predictions, self.output_dir)}
            for name, source in self.estimates.items():
                inputs[f"estimates/{name}"] = reference(source.path, self.output_dir)
                inputs[f"eta/{name}"] = reference(source.eta_path, self.output_dir)
            shared_names = ["manifest", "data"]
            if self.queries:
                shared_names.append("projection_queries")
            for name in shared_names:
                inputs[name] = reference(self.saved.input(name), self.output_dir)
            write_json(directory / "evaluation.json", {
                "schema_version": SCHEMA_VERSION, "evaluation_version": EVALUATION_VERSION,
                "inputs": inputs, "methods": list(self.methods), "config": asdict(self.config),
                "metrics": {"path": "metrics.h5", "sha256": sha256(directory / "metrics.h5")},
                "complete": True, "numpy_version": np.__version__, "statistics_unit": "task"})
        return self.output_dir / "metrics.h5"
