"""Shared task and projection-query inputs; no model dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np

from .storage import ReferenceResolver, sha256
from .protocol import Prior


def id_index(values):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in "iu"
            or np.any(values < 0) or len(np.unique(values)) != len(values)):
        raise ValueError("task IDs must be unique nonnegative integers")
    return {int(value): row for row, value in enumerate(values)}


@dataclass(frozen=True)
class Task:
    task_id: int
    context_x: np.ndarray
    context_y: np.ndarray
    evaluation_x: np.ndarray
    evaluation_y: np.ndarray
    support: np.ndarray
    beta: np.ndarray


class TaskDataset:
    """Keep indices in memory and stream task arrays from the shared HDF5."""

    def __init__(self, path, prior: Prior):
        self.path = Path(path).resolve()
        self.prior = prior
        self.sha256 = sha256(self.path)
        with h5py.File(self.path, "r") as source:
            self.task_ids = source["task_id"][:]
            self.index = id_index(self.task_ids)
            self.n_context = source["single_eval_pos"][:]
            self.n_total = source["num_datapoints"][:]
            if any(value.dtype.kind not in "iu" for value in (self.n_context, self.n_total)):
                raise ValueError("task lengths must be integers")
            self.n_eval = self.n_total - self.n_context
            t, d = len(self.task_ids), prior.d
            if not t or source["X"].ndim != 3 or source["X"].shape[::2] != (t, d):
                raise ValueError("task data has invalid X dimensions")
            capacity = source["X"].shape[1]
            for name, shape in {"y": (t, capacity), "beta": (t, d),
                                "support_mask": (t, d), "num_features": (t,),
                                "single_eval_pos": (t,), "num_datapoints": (t,)}.items():
                if source[name].shape != shape:
                    raise ValueError(f"invalid task data shape: {name}")
            if (np.any(self.n_context < d) or np.any(self.n_eval < 1)
                    or np.any(self.n_total > capacity)
                    or np.any(source["num_features"][:] != d)
                    or source["max_num_classes"][0] != 1):
                raise ValueError("invalid task lengths or regression metadata")
            for name, expected in (("d", prior.d), ("k", prior.k),
                                   ("noise_std", prior.noise_std),
                                   ("expected_signal_energy", prior.signal_energy)):
                if name in source.attrs and not np.isclose(source.attrs[name], expected):
                    raise ValueError(f"task data and manifest disagree on {name}")
        self.max_eval = int(max(self.n_eval))

    def read(self, task_id, source=None):
        if source is None:
            with h5py.File(self.path, "r") as handle:
                return self.read(task_id, handle)
        row = self.index[int(task_id)]
        n, total = int(self.n_context[row]), int(self.n_total[row])
        x, y = source["X"][row, :total], source["y"][row, :total]
        support, beta = source["support_mask"][row], source["beta"][row]
        if (not np.isfinite(x).all() or not np.isfinite(y).all()
                or not np.isfinite(beta).all() or not np.isin(support, (0, 1)).all()
                or support.sum() != self.prior.k or np.any(beta[support == 0] != 0)):
            raise ValueError(f"invalid task arrays: {task_id}")
        gram = x[:n].astype(np.float64).T @ x[:n].astype(np.float64) / n
        if not np.allclose(gram, np.eye(self.prior.d), atol=1e-4, rtol=1e-4):
            raise ValueError(f"exact-k Bayes requires an orthogonal context: {task_id}")
        return Task(int(task_id), x[:n], y[:n], x[n:], y[n:],
                    support.astype(bool), beta)

    def batches(self, task_ids=None, batch_size=32):
        ids = self.task_ids if task_ids is None else task_ids
        with h5py.File(self.path, "r") as source:
            for start in range(0, len(ids), batch_size):
                yield [self.read(int(t), source) for t in ids[start:start + batch_size]]


class DatasetManifest:
    def __init__(self, path, resolver=None):
        self.path = Path(path).resolve()
        self.payload = json.loads(self.path.read_text())
        if self.payload.get("complete") is not True:
            raise ValueError("dataset manifest must be complete")
        arguments = self.payload["command_arguments"]
        self.prior = Prior(**{key: arguments[key]
                              for key in ("d", "k", "signal_energy", "noise_std")})
        self.resolver = resolver or ReferenceResolver(self.path.parent)

    def dataset(self, path):
        """Read one explicitly selected dataset under this manifest's prior."""
        return TaskDataset(path, self.prior)

    def test_dataset(self):
        ref = self.payload["splits"]["test"]
        return TaskDataset(self.resolver.resolve(ref, relative_to=self.path.parent), self.prior)


class ProjectionQueries:
    """Independent Gaussian queries keyed by task ID, reusable across label changes."""

    def __init__(self, path, dataset: TaskDataset):
        self.path = Path(path).resolve()
        with h5py.File(self.path, "r") as source:
            if not bool(source.attrs.get("complete", False)):
                raise ValueError("projection queries are incomplete")
            self.index = id_index(source["task_id"][:])
            shape = source["x_proj"].shape
            if len(shape) != 3 or shape[0] != len(self.index) or shape[2] != dataset.prior.d:
                raise ValueError("projection query dimensions do not match data")
            if shape[1] < 1:
                raise ValueError("projection queries must not be empty")
            self.count = shape[1]
            self.seed = int(source.attrs["projection_seed"])
            if set(map(int, dataset.task_ids)) - self.index.keys():
                raise ValueError("projection queries are missing task IDs")

    def read(self, task_id):
        with h5py.File(self.path, "r") as source:
            x = source["x_proj"][self.index[int(task_id)]]
        if not np.isfinite(x).all():
            raise ValueError(f"nonfinite projection queries for task {task_id}")
        return x
