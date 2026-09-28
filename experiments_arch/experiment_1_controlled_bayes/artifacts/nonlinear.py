"""Inputs and task identities for offline affine residual energy summaries."""

import h5py
import numpy as np

from .data import DatasetManifest, ProjectionQueries
from .storage import PredictionRun, sha256


def generator_batch_size(dataset, manifest=None):
    """Read the original generator batch size from the H5, then the manifest."""
    with h5py.File(dataset.path, "r") as source:
        size = source.attrs.get("generator_batch_size")
    payload = getattr(manifest, "payload", None) or {}
    if size is None:
        size = payload.get("generator_batch_size")
    if size is None:
        size = payload.get("command_arguments", {}).get("generator_batch_size")
    if size is None or int(size) < 1 or int(size) != size:
        raise ValueError("source data must specify its original generator_batch_size")
    return int(size)


class Materials:
    """Test split, projection queries and original batch identities from a run."""

    def __init__(self, run):
        self.run = run if isinstance(run, PredictionRun) else PredictionRun(run)
        if self.run.config["protocol"]["layers"]:
            raise ValueError("IV.7 requires a baseline-only prediction run")
        self.manifest = DatasetManifest(self.run.input("manifest"), self.run.resolver)
        self.dataset = self.manifest.dataset(self.run.input("data"))
        self.projection = ProjectionQueries(self.run.input("projection_queries"), self.dataset)
        if self.projection.count <= self.dataset.prior.d + 1:
            raise ValueError("affine BLP requires Qp > d + 1")
        for name, path in (("manifest", self.manifest.path), ("data", self.dataset.path),
                           ("projection_queries", self.projection.path)):
            if self.run.config["inputs"][name]["sha256"] != sha256(path):
                raise ValueError(f"prediction run and materials differ: {name}")
        self.batch_size = generator_batch_size(self.dataset, self.manifest)
        n = len(self.dataset.task_ids)
        with h5py.File(self.dataset.path, "r") as data:
            if "source_row" in data:
                self.source_rows = np.asarray(data["source_row"][:], dtype=np.int64)
                self.batch_ids = np.asarray(data["original_batch_id"][:], dtype=np.int64)
            else:
                self.source_rows = np.arange(n, dtype=np.int64)
                self.batch_ids = self.source_rows // self.batch_size
        if (self.source_rows.shape != self.dataset.task_ids.shape
                or self.batch_ids.shape != self.dataset.task_ids.shape
                or not np.array_equal(self.batch_ids, self.source_rows // self.batch_size)):
            raise ValueError("original batch IDs do not align with tasks")
