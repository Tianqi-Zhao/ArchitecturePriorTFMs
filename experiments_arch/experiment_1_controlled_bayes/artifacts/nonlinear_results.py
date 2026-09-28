"""Read sidecar L and join C/Z/N/E from evaluate_saved; ratios are derived in memory."""

from contextlib import contextmanager
import json
from pathlib import Path

import h5py
import numpy as np

from .data import id_index
from .nonlinear_protocol import (ALGORITHM_VERSION, ENERGY_SCHEMA, RATIOS, SAVED_METRICS,
                                 NonlinearEvaluationConfig)
from .protocol import Prior, SUPPORTED_SCHEMAS, TERMINAL, Protocol
from .storage import ReferenceResolver, sha256


class NonlinearEvaluation:
    """Join linear energy with evaluate_saved metrics; do not open raw predictions."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.config = json.loads(self.path.read_text())
        self.sha256 = sha256(self.path)
        self.resolver = ReferenceResolver(self.path.parent)
        c = self.config
        if (c.get("schema_version") != ENERGY_SCHEMA or c.get("complete") is not True
                or c.get("kind") != "linear_energy"
                or c.get("algorithm_version") != ALGORITHM_VERSION):
            raise ValueError("unsupported or incomplete linear energy artifact")
        self.prior = Prior(**c["prior"])
        self.metadata = {}
        for name in ("run", "estimate", "evaluation"):
            source = json.loads(self.resolver.resolve(c["inputs"][name]).read_text())
            if source.get("complete") is not True or source.get("schema_version") not in SUPPORTED_SCHEMAS:
                raise ValueError(f"unsupported or incomplete {name} metadata")
            self.metadata[name] = source
        run, estimate, evaluation = (self.metadata[name] for name in ("run", "estimate", "evaluation"))
        Protocol.from_run(run)
        for name in ("manifest", "data", "projection_queries"):
            if name in c["inputs"] and c["inputs"][name]["sha256"] != run["inputs"][name]["sha256"]:
                raise ValueError(f"result source identity differs: {name}")
        if (run["protocol"]["layers"] or estimate["method"] != "affine_blp"
                or "affine_blp" not in evaluation.get("methods", [])
                or estimate["inputs"]["run"]["sha256"] != c["inputs"]["run"]["sha256"]
                or evaluation["inputs"]["run"]["sha256"] != c["inputs"]["run"]["sha256"]
                or estimate["inputs"]["projection_queries"]["sha256"]
                != c["inputs"]["projection_queries"]["sha256"]):
            raise ValueError("linear energy does not describe one baseline affine_blp evaluation")
        self.evaluation_config = NonlinearEvaluationConfig(
            evaluation["config"].get("denominator_threshold", 1e-12))
        self.saved_evaluation_config = evaluation["config"]
        self.source_test_sha256 = c["inputs"]["data"]["sha256"]
        self.result_path = self.resolver.resolve(c["result"])
        evaluation_path = self.resolver.resolve(c["inputs"]["evaluation"])
        self.metrics_path = ReferenceResolver(evaluation_path.parent).resolve(evaluation["metrics"])
        with self.read() as raw, h5py.File(self.metrics_path, "r") as saved:
            self._load(raw, saved)

    @property
    def comparison_config(self):
        """Settings that must match across models; model identities deliberately differ."""
        return dict(
            source={name: self.config["inputs"][name]["sha256"]
                    for name in ("manifest", "data", "projection_queries")},
            prior=self.config["prior"], evaluation=self.saved_evaluation_config,
            estimator=self.metadata["estimate"]["configuration"],
            algorithm_version=self.config["algorithm_version"])

    def _load(self, raw, saved):
        dimensions = self.config["dimensions"]
        t, d = dimensions["num_tasks"], dimensions["d"]
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (t, d))
                or d != self.prior.d or dimensions["k"] != self.prior.k
                or dimensions.get("generator_batch_size", 0) < 1):
            raise ValueError("invalid linear energy dimensions")

        def array(source, name, shape, kind, axes="task"):
            field = source[name]
            if field.shape != shape or field.dtype.kind not in kind or field.attrs.get("axis_order") != axes:
                raise ValueError(f"invalid linear energy array: {name}")
            return field.asstr()[:] if kind == "OSU" else field[:]

        self.tasks = {name: array(raw, f"tasks/{name}", (t,), "iu") for name in (
            "task_id", "source_row", "original_batch_id", "baseline_state_id", "n_eval")}
        self.index = id_index(self.tasks["task_id"])
        for name in ("source_row", "baseline_state_id"):
            id_index(self.tasks[name])
        if (np.any(self.tasks["n_eval"] < 1)
                or not np.array_equal(self.tasks["original_batch_id"], self.tasks["source_row"]
                                      // dimensions["generator_batch_size"])):
            raise ValueError("invalid linear energy task lengths or batch IDs")
        self.status = array(raw, "tasks/status", (t,), "OSU")
        self.reason = array(raw, "tasks/reason", (t,), "OSU")
        if not set(self.status) <= TERMINAL:
            raise ValueError("linear energy contains invalid terminal states")
        self.counts = {name: int(np.sum(self.status == name)) for name in sorted(TERMINAL)}
        if self.counts != self.config["counts"]:
            raise ValueError("linear energy status counts differ from manifest")
        ok = self.status == "ok"
        values = array(raw, "metrics/L", (t,), "f")
        if not np.isfinite(values[ok]).all() or not np.isnan(values[~ok]).all():
            raise ValueError("linear energy values disagree with task status")
        self.metrics = {"L": np.asarray(values, dtype=np.float64)}
        self.metric_status = {"L": np.array(self.status, dtype=object)}
        self.metric_reason = {"L": np.array(self.reason, dtype=object)}
        self._join_saved(saved)
        self._derive_ratios()

    def _join_saved(self, saved):
        if not saved.attrs.get("complete"):
            raise ValueError("evaluate_saved metrics are incomplete")
        rows = self.tasks["baseline_state_id"]
        n_states = saved["states/E"].shape[0]
        if np.any(rows < 0) or np.any(rows >= n_states):
            raise ValueError("linear energy baseline states do not match evaluate_saved")
        task_ids = np.asarray(saved["tasks/task_id"][:])
        task_index = np.asarray(saved["states/task_index"][:])
        if not np.array_equal(task_ids[task_index[rows]], self.tasks["task_id"]):
            raise ValueError("linear energy tasks do not match evaluate_saved states")
        method = saved["methods/affine_blp/states"]
        public_status = saved["states/status"].asstr()[:][rows]
        method_status = method["status"].asstr()[:][rows]
        method_reason = method["reason"].asstr()[:][rows]
        public_reason = saved["states/reason"].asstr()[:][rows]
        mapping = dict(C="C", Z="Z", N="N", E="E", delta_IV2="accounting_gap")
        for name in SAVED_METRICS:
            source = saved["states/E"] if name == "E" else method[mapping[name]]
            values = np.asarray(source[:], dtype=np.float64)[rows]
            status_row = public_status if name == "E" else method_status
            reason_row = public_reason if name == "E" else method_reason
            status = np.where((status_row == "ok") & np.isfinite(values), "ok", "unavailable")
            reason = np.where(status == "ok", "", np.where(status_row != "ok", reason_row, "metric unavailable"))
            self.metrics[name] = values
            self.metric_status[name] = np.asarray(status, dtype=object)
            self.metric_reason[name] = np.asarray(reason, dtype=object)

    def _derive_ratios(self):
        threshold = self.evaluation_config.denominator_threshold
        self.ratios, self.ratio_status, self.ratio_reason = {}, {}, {}
        for name, (numerator, denominators) in RATIOS.items():
            components = (numerator, *denominators)
            values, denom = self.metrics[numerator], sum(self.metrics[key] for key in denominators)
            if name == "abs_gap_to_error":
                values = np.abs(values)
            derived = np.full(values.shape, np.nan)
            good = np.isfinite(denom) & (denom > threshold)
            np.divide(values, denom, out=derived, where=good)
            ok = np.array(good)
            reasons = np.full(len(derived), "", dtype=object)
            for component in components:
                missing = self.metric_status[component] != "ok"
                reasons = np.where(
                    missing & (reasons == ""),
                    np.array([f"{component}: {status}: {reason}" for status, reason
                              in zip(self.metric_status[component], self.metric_reason[component])],
                             dtype=object),
                    reasons)
                ok &= ~missing
            reasons = np.where(ok, "", np.where(reasons == "", "denominator at or below threshold", reasons))
            self.ratios[name] = np.where(ok, derived, np.nan)
            self.ratio_status[name] = np.where(ok, "ok", "unavailable")
            self.ratio_reason[name] = reasons

    @contextmanager
    def read(self):
        with h5py.File(self.result_path, "r") as raw:
            if (not raw.attrs.get("complete") or raw.attrs.get("schema_version") != ENERGY_SCHEMA
                    or raw.attrs.get("algorithm_version") != ALGORITHM_VERSION):
                raise ValueError("unsupported or incomplete linear energy arrays")
            yield raw
