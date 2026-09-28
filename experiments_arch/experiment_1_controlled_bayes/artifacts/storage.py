"""File identity, input references, and the minimal HDF5 storage contract."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

from .protocol import METHOD_NAMES, SCHEMA_VERSION, SUPPORTED_SCHEMAS, TERMINAL, Protocol

STRING = h5py.string_dtype("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path, directory):
    path = Path(path).resolve()
    return {"path": os.path.relpath(path, Path(directory).resolve()),
            "sha256": sha256(path)}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def collection_log(directory):
    directory = Path(directory).resolve()
    path = directory.parent / "logs" / directory.name / "collection.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class ReferenceResolver:
    """Resolve and verify explicit artifact references."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self._verified = {}

    def resolve(self, ref, *, relative_to=None):
        path = Path(ref["path"]).expanduser()
        path = (Path(relative_to or self.directory) / path).resolve()
        expected = ref["sha256"]
        key = (path, expected)
        if key not in self._verified:
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"file missing or hash mismatch: {path}")
            self._verified[key] = True
        return path


@contextmanager
def new_directory(destination):
    """Publish a finished output atomically; refuse to overwrite previous work."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.",
                                     suffix=".partial", dir=destination.parent))
    try:
        yield temporary
        if destination.exists():
            raise FileExistsError(f"output already exists: {destination}")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def require_complete(source):
    if source.attrs.get("schema_version") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported schema: {source.file.filename}")
    if not bool(source.attrs.get("complete", False)):
        raise ValueError(f"incomplete artifact: {source.file.filename}")


def text_array(dataset):
    return dataset.asstr()[:]


class Table:
    """Append aligned rows to a small set of column datasets."""

    def __init__(self, group):
        self.group = group

    def column(self, name, dtype, tail=(), fillvalue=None):
        options = {"fillvalue": fillvalue} if fillvalue is not None else {}
        self.group.create_dataset(name, shape=(0, *tail), maxshape=(None, *tail),
                                  dtype=dtype, chunks=True, **options)

    def append(self, values):
        size = len(next(iter(values.values())))
        start = len(next(iter(self.group.values())))
        for dataset in self.group.values():
            dataset.resize(start + size, axis=0)
        for name, value in values.items():
            self.group[name][start:start + size] = value
        return start


class PredictionWriter:
    """Write state results, optionally coalescing one bounded inference batch."""

    def __init__(self, path, *, qp, qe, d, dtype):
        self.file = h5py.File(path, "x")
        self.file.attrs.update(schema_version=SCHEMA_VERSION, complete=False)
        self.qp = qp
        tasks, states = Table(self.file.create_group("tasks")), Table(self.file.create_group("states"))
        for name in ("task_id", "n_eval", "baseline_state_id"):
            tasks.column(name, np.int64)
        tasks.column("token_groups_json", STRING)
        for name in ("task_index", "layer", "source_token"):
            states.column(name, np.int64, fillvalue=-1)
        states.column("epsilon", np.float64, fillvalue=np.nan)
        states.column("kind", STRING)
        for name, width in (("f_proj", qp), ("f_eval", qe)):
            states.column(name, dtype, (width,), np.nan)
        states.column("gradient_mean", np.float64, (d,), np.nan)
        states.column("gradient_count", np.int64, fillvalue=0)
        for phase in ("projection", "evaluation", "gradient"):
            states.column(f"{phase}_status", STRING, fillvalue="pending")
        self.tasks, self.states = tasks, states
        self._pending = None

    @contextmanager
    def batch(self):
        """Combine state writes by column and flush once after a successful batch."""
        if self._pending is not None:
            raise RuntimeError("prediction write batches must not be nested")
        self._pending = {}
        try:
            yield
            for name, updates in self._pending.items():
                dataset = self.states.group[name]
                # Evaluation rows can have different valid query lengths. Keep
                # their unwritten padding, and sort HDF5's noncontiguous indices.
                groups = {}
                for row, value in sorted(updates.items()):
                    groups.setdefault(np.shape(value), []).append((row, value))
                for shape, entries in groups.items():
                    rows = np.asarray([row for row, _ in entries], dtype=np.int64)
                    selection = (slice(int(rows[0]), int(rows[-1]) + 1)
                                 if rows[-1] - rows[0] + 1 == len(rows) else rows)
                    values = np.asarray([value for _, value in entries], dtype=dataset.dtype)
                    if shape:
                        dataset[selection, :shape[0]] = values
                    else:
                        dataset[selection] = values
            if self._pending:
                self.file.flush()
        finally:
            self._pending = None

    def _write(self, name, row, value):
        if self._pending is not None:
            # Retain the value at write time even if the caller reuses a buffer.
            self._pending.setdefault(name, {})[row] = (
                np.asarray(value).copy() if np.ndim(value) else value)
        elif np.ndim(value):
            self.states.group[name][row, :len(value)] = value
        else:
            self.states.group[name][row] = value

    def plan(self, task, groups, gates):
        tasks, states = self.tasks, self.states
        index = len(tasks.group["task_id"])
        start = states.append({
            "task_index": np.full(len(gates), index, dtype=np.int64),
            "layer": [g.layer for g in gates],
            "source_token": [g.source_token for g in gates],
            "epsilon": [g.epsilon for g in gates],
            "kind": [g.state_kind for g in gates],
        })
        tasks.append({"task_id": [task.task_id], "n_eval": [len(task.evaluation_x)],
                      "baseline_state_id": [start],
                      "token_groups_json": [json.dumps(groups, sort_keys=True)]})
        return start

    def write(self, row, phase, values=None, status="ok"):
        if values is not None:
            field = "f_proj" if phase == "projection" else "f_eval"
            self._write(field, row, values)
        self._write(f"{phase}_status", row, status)
        if self._pending is None:
            self.file.flush()

    def write_gradient(self, row, mean=None, count=0, status="ok"):
        states = self.states.group
        if not isinstance(count, (int, np.integer)) or not 0 <= count <= self.qp:
            raise ValueError("invalid projection gradient count")
        if status == "ok":
            if (count != self.qp or np.shape(mean) != states["gradient_mean"].shape[1:]
                    or not np.isfinite(mean).all()):
                raise ValueError("mean gradients must be finite and cover all projection queries")
            self._write("gradient_mean", row, mean)
        self._write("gradient_count", row, count)
        self._write("gradient_status", row, status)
        if self._pending is None:
            self.file.flush()

    def close(self, complete=False):
        if self._pending is not None:
            raise RuntimeError("finish the prediction write batch before closing")
        if complete:
            for phase in ("projection", "evaluation", "gradient"):
                if not set(text_array(self.states.group[f"{phase}_status"])) <= TERMINAL:
                    raise ValueError("prediction file still has pending states")
            self.file.attrs["complete"] = True
        self.file.close()


class EstimateWriter:
    """Write one method's estimates without changing its prediction source."""

    def __init__(self, path, estimator, d):
        self.file = h5py.File(path, "x")
        self.file.attrs.update(schema_version=SCHEMA_VERSION, complete=False,
                               estimator=estimator.name)
        table = Table(self.file.create_group("states"))
        table.column("eta", np.float64, (d,), np.nan)
        if estimator.intercept_mode == "fitted":
            table.column("zeta", np.float64, fillvalue=np.nan)
        table.column("status", STRING, fillvalue="pending")
        table.column("reason", STRING, fillvalue="")
        self.table = table

    def plan(self, count):
        self.table.append({"status": ["pending"] * count})

    def write(self, row, estimate=None, status="ok", reason=""):
        table = self.table.group
        if estimate is not None:
            if (np.shape(estimate.eta) != table["eta"].shape[1:]
                    or not np.isfinite(estimate.eta).all() or not np.isfinite(estimate.zeta)):
                raise ValueError("eta estimate must have finite, correctly sized coefficients")
            table["eta"][row] = estimate.eta
            if "zeta" in table:
                table["zeta"][row] = estimate.zeta
        elif status in {"failed", "unavailable"}:
            table["eta"][row] = np.nan
            if "zeta" in table:
                table["zeta"][row] = np.nan
        table["status"][row] = status
        table["reason"][row] = reason
        self.file.flush()

    def close(self, prediction_hash):
        if not set(text_array(self.table.group["status"])) <= TERMINAL:
            raise ValueError("eta file still has pending states")
        self.file.attrs.update(prediction_sha256=prediction_hash, complete=True)
        self.file.close()


class PredictionRun:
    """Read one immutable prediction artifact for one explicitly selected dataset."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.directory = self.path.parent
        self.resolver = ReferenceResolver(self.directory)
        self.config = json.loads(self.path.read_text())
        self.protocol = Protocol.from_run(self.config)
        if self.config.get("complete") is not True:
            raise ValueError("incomplete prediction run")
        self.predictions = self.resolver.resolve(self.config["predictions"])
        with self.prediction():
            pass
        self.prediction_hash = self.config["predictions"]["sha256"]
        self.sha256 = sha256(self.path)

    def input(self, name):
        return self.resolver.resolve(self.config["inputs"][name])

    def validate_reuse(self, *, protocol, inputs, checkpoint, backend, version=None,
                       gradients=False, dtype="float32"):
        """Verify an existing campaign run against the requested experiment."""
        # Batching and check tolerances can vary without changing the experiment.
        for name in ("layers", "epsilons", "source_tokens", "inference_seed",
                     "row_scope", "intervention_kind"):
            if getattr(self.protocol, name) != getattr(protocol, name):
                raise ValueError(f"saved prediction protocol mismatch: {name}")
        for name, path in inputs.items():
            if sha256(path) != self.config["inputs"][name]["sha256"]:
                raise ValueError(f"saved prediction input mismatch: {name}")
        model = self.config["model"]
        if (model.get("adapter") != backend or model.get("dtype") != dtype
                or (model.get("checkpoint") or {}).get("sha256") != sha256(checkpoint)
                or backend == "tabpfn" and (model.get("version") != version
                                           or model.get("column_embeddings") != "native")):
            raise ValueError("saved model/checkpoint/preprocessing differs from requested prediction")
        if gradients and self.config.get("gradient_capture") != "mean_projection_input_gradient":
            raise ValueError("saved run has no collected projection gradients")

    @contextmanager
    def prediction(self):
        with h5py.File(self.predictions, "r") as source:
            require_complete(source)
            if source.attrs["schema_version"] != self.config["schema_version"]:
                raise ValueError("prediction schema differs from run schema")
            if source.attrs["schema_version"] == SCHEMA_VERSION and "kind" not in source["states"]:
                raise ValueError("prediction states are missing kind")
            yield source

    def dataset(self):
        from .data import DatasetManifest

        manifest = DatasetManifest(self.input("manifest"), self.resolver)
        return manifest.dataset(self.input("data"))


class SavedEstimate:
    """Read one independent eta artifact and verify its prediction provenance."""

    def __init__(self, path, run: PredictionRun):
        self.path = Path(path).expanduser().resolve()
        self.config = json.loads(self.path.read_text())
        if (self.config.get("schema_version") not in SUPPORTED_SCHEMAS
                or self.config.get("complete") is not True):
            raise ValueError("unsupported or incomplete eta artifact")
        resolver = ReferenceResolver(self.path.parent)
        source = self.config["inputs"]["run"]
        resolver.resolve(source)
        if source["sha256"] != run.sha256:
            raise ValueError("eta artifact belongs to a different prediction run")
        self.eta_path = resolver.resolve(self.config["eta"])
        self.method = self.config["method"]
        if self.method not in METHOD_NAMES:
            raise ValueError("unsupported eta method")
        with h5py.File(self.eta_path, "r") as raw:
            require_complete(raw)
            if raw.attrs.get("estimator") != self.method:
                raise ValueError("eta estimator identity mismatch")
            if raw.attrs.get("prediction_sha256") != run.prediction_hash:
                raise ValueError("eta/prediction hash mismatch")

    @contextmanager
    def read(self):
        with h5py.File(self.eta_path, "r") as raw:
            yield raw
