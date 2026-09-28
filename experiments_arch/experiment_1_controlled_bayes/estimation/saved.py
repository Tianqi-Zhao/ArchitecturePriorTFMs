"""Estimate one eta method from immutable predictions and cached gradients."""

from pathlib import Path

import numpy as np

from ..artifacts.data import ProjectionQueries
from ..artifacts.index import PredictionIndex
from ..artifacts.protocol import CODE_VERSION, SCHEMA_VERSION, UnavailableError
from ..artifacts.storage import EstimateWriter, PredictionRun, new_directory, reference, sha256, write_json
from .estimators import Projection, make_estimator


class SavedEstimator:
    """Own one independent estimate artifact; never load a model or edit inputs."""

    def __init__(self, run, *, method, output_dir):
        self.saved = PredictionRun(run)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.estimator = make_estimator(method)
        self.protocol = self.saved.protocol
        self.dataset = self.saved.dataset()
        self.queries = ProjectionQueries(self.saved.input("projection_queries"), self.dataset)
        if (self.estimator.name == "gradient"
                and self.saved.config.get("gradient_capture") != "mean_projection_input_gradient"):
            raise ValueError(
                "gradient eta requires predictions collected with --gradients; "
                f"this run has gradient_capture={self.saved.config.get('gradient_capture')!r}")

    def _estimate(self, writer):
        dataset = self.dataset
        with self.saved.prediction() as raw:
            index = PredictionIndex.read(raw, dataset, self.protocol)
            writer.plan(len(index.task_index))
            states = raw["states"]
            for ti, task_id in enumerate(index.task_ids):
                x = self.queries.read(int(task_id))
                if states["f_proj"].shape[1] != len(x):
                    raise ValueError("projection predictions and queries have different lengths")
                for row in index.rows(ti):
                    try:
                        if states["projection_status"].asstr()[row] != "ok":
                            raise UnavailableError("saved projection predictions are unavailable")
                        mean, count = None, 0
                        if self.estimator.name == "gradient":
                            if states["gradient_status"].asstr()[row] != "ok":
                                raise UnavailableError("saved projection gradients are unavailable")
                            mean = states["gradient_mean"][row]
                            count = int(states["gradient_count"][row])
                        material = Projection(x, states["f_proj"][row], mean, count)
                        writer.write(row, self.estimator.estimate(material))
                    except (UnavailableError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                        writer.write(row, status=(
                            "unavailable" if isinstance(error, UnavailableError) else "failed"),
                            reason=str(error))

    def run(self):
        with new_directory(self.output_dir) as directory:
            writer = EstimateWriter(directory / "eta.h5", self.estimator,
                                    self.dataset.prior.d)
            try:
                self._estimate(writer)
                writer.close(self.saved.prediction_hash)
            finally:
                writer.file.close()
            write_json(directory / "estimate.json", {
                "schema_version": SCHEMA_VERSION, "estimation_version": CODE_VERSION,
                "complete": True, "method": self.estimator.name,
                "configuration": self.estimator.configuration,
                "inputs": {"run": reference(self.saved.path, self.output_dir),
                           "projection_queries": reference(self.queries.path, self.output_dir)},
                "eta": {"path": "eta.h5", "sha256": sha256(directory / "eta.h5")},
                "numpy_version": np.__version__})
        return self.output_dir / "estimate.json"
