"""Write ||eta||^2 beside an existing evaluate_saved result; do not rewrite C/Z/N/E."""

from dataclasses import asdict
from pathlib import Path
import json

import h5py
import numpy as np

from ..artifacts.index import PredictionIndex
from ..artifacts.nonlinear import Materials
from ..artifacts.nonlinear_protocol import ALGORITHM_VERSION, ENERGY_SCHEMA
from ..artifacts.nonlinear_storage import LinearEnergyWriter, analysis_provenance
from ..artifacts.protocol import TERMINAL, UnavailableError
from ..artifacts.storage import PredictionRun, SavedEstimate, new_directory, reference, sha256, write_json
from ..estimation.estimators import AffineBLPEstimator


class NonlinearEvaluator:
    def __init__(self, run, estimate, evaluation):
        self.saved = PredictionRun(run)
        self.materials = Materials(self.saved)
        self.estimate = SavedEstimate(estimate, self.saved)
        self.evaluation_path = Path(evaluation).expanduser().resolve()
        if (self.estimate.method != "affine_blp"
                or self.estimate.config["configuration"] != AffineBLPEstimator().configuration):
            raise ValueError("linear energy requires the supported affine_blp estimate")
        if (self.estimate.config["inputs"]["projection_queries"]["sha256"]
                != self.saved.config["inputs"]["projection_queries"]["sha256"]):
            raise ValueError("estimate and materials use different projection queries")
        payload = json.loads(self.evaluation_path.read_text())
        if (payload.get("complete") is not True or "affine_blp" not in payload.get("methods", [])
                or payload["inputs"]["run"]["sha256"] != sha256(self.saved.path)):
            raise ValueError("evaluation is not a complete affine_blp evaluate_saved result for this run")
        self.evaluation = payload
        self.dataset = self.materials.dataset
        self.counts = {}

    def _validate(self, predictions, estimates):
        dataset, m = self.dataset, self.materials
        index = PredictionIndex.read(predictions, dataset, self.saved.protocol)
        if set(map(int, index.task_ids)) != set(map(int, dataset.task_ids)):
            raise ValueError("prediction and material task sets differ")
        rows = {int(task_id): int(index.baseline_ids[i]) for i, task_id in enumerate(index.task_ids)}
        if (m.source_rows.shape != dataset.task_ids.shape or m.batch_ids.shape != dataset.task_ids.shape
                or m.source_rows.dtype.kind not in "iu" or np.any(m.source_rows < 0)
                or len(np.unique(m.source_rows)) != len(m.source_rows)
                or not np.array_equal(m.batch_ids, m.source_rows // m.batch_size)):
            raise ValueError("material source rows and batch IDs disagree")
        states, count = estimates["states"], len(index.task_index)
        for name, shape in dict(eta=(count, dataset.prior.d), zeta=(count,),
                                 status=(count,), reason=(count,)).items():
            if states[name].shape != shape:
                raise ValueError("estimate state rows do not align with predictions")
        if not set(states["status"].asstr()[:]) <= TERMINAL:
            raise ValueError("invalid estimate terminal status")
        return np.array([rows[int(task_id)] for task_id in dataset.task_ids], dtype=np.int64)

    @staticmethod
    def _linear_energy(row, estimates):
        if estimates["status"].asstr()[row] != "ok":
            raise UnavailableError(f"estimate: {estimates['reason'].asstr()[row]}")
        eta = np.asarray(estimates["eta"][row], dtype=np.float64)
        if eta.ndim != 1 or not np.isfinite(eta).all():
            raise ValueError("nonfinite or invalid affine_blp coefficients")
        return float(eta @ eta)

    def run(self, output_dir):
        destination = Path(output_dir).expanduser().resolve()
        m, saved = self.materials, self.saved
        with saved.prediction() as raw, self.estimate.read() as eta:
            baseline_ids = self._validate(raw, eta)
            with new_directory(destination) as directory:
                with LinearEnergyWriter(directory / "linear_energy.h5", m, baseline_ids) as writer:
                    for ti, _task_id in enumerate(self.dataset.task_ids):
                        try:
                            value = self._linear_energy(baseline_ids[ti], eta["states"])
                        except (ValueError, RuntimeError, FloatingPointError) as error:
                            status = "unavailable" if isinstance(error, UnavailableError) else "failed"
                            writer.write(ti, status=status, reason=str(error))
                        else:
                            writer.write(ti, value)
                    self.counts = writer.counts
                write_json(directory / "linear_energy.json", dict(
                    schema_version=ENERGY_SCHEMA, algorithm_version=ALGORITHM_VERSION,
                    kind="linear_energy", complete=True,
                    inputs={name: reference(path, destination) for name, path in dict(
                        run=saved.path, estimate=self.estimate.path, evaluation=self.evaluation_path,
                        manifest=saved.input("manifest"), data=self.dataset.path,
                        projection_queries=m.projection.path).items()},
                    prior=asdict(self.dataset.prior),
                    dimensions=dict(num_tasks=len(self.dataset.task_ids), d=self.dataset.prior.d,
                                    k=self.dataset.prior.k, generator_batch_size=m.batch_size),
                    counts=self.counts,
                    statistics_unit="task", cluster_unit="original_batch_id",
                    result=dict(path="linear_energy.h5", sha256=sha256(directory / "linear_energy.h5")),
                    provenance=analysis_provenance()))
        return destination / "linear_energy.json"
