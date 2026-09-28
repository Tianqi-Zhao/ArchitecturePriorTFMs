"""Linear-energy writer and provenance for offline summaries."""

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from .nonlinear_protocol import ALGORITHM_VERSION, ENERGY_SCHEMA
from .protocol import TERMINAL
from .storage import STRING


def analysis_provenance():
    """Record the local implementation and runtime without requiring Git."""
    root = Path(__file__).resolve().parents[3]
    commit, dirty = None, None
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return dict(git_commit=commit, working_tree_dirty=dirty, argv=sys.argv,
                python_version=sys.version.split()[0], numpy_version=np.__version__,
                h5py_version=h5py.__version__)


class LinearEnergyWriter:
    """Write per-task L = ||eta||^2; C/Z/N/E stay in evaluate_saved."""

    def __init__(self, path, materials, baseline_ids):
        dataset = materials.dataset
        t = len(dataset.task_ids)
        self.n_eval = dataset.n_eval
        self.file = h5py.File(path, "x")
        self.file.attrs.update(schema_version=ENERGY_SCHEMA,
                               algorithm_version=ALGORITHM_VERSION, complete=False)
        for name, values in dict(task_id=dataset.task_ids, source_row=materials.source_rows,
                                 original_batch_id=materials.batch_ids,
                                 baseline_state_id=baseline_ids, n_eval=dataset.n_eval).items():
            self.file.create_dataset(f"tasks/{name}", data=values, dtype=np.int64).attrs["axis_order"] = "task"
        self._column("tasks/status", (t,), STRING, "pending")
        self._column("tasks/reason", (t,), STRING, "")
        self._column("metrics/L", (t,))

    def _column(self, name, shape, dtype=np.float64, fill=np.nan, axes="task"):
        field = self.file.create_dataset(name, shape, dtype=dtype, fillvalue=fill, chunks=True)
        field.attrs["axis_order"] = axes

    def write(self, row, value=None, *, status="ok", reason=""):
        if status not in TERMINAL or (value is None) != (status != "ok"):
            raise ValueError("linear energy must match its terminal status")
        if value is not None:
            if not np.isfinite(value):
                raise ValueError("linear energy must be finite")
            self.file["metrics/L"][row] = float(value)
        self.file["tasks/status"][row], self.file["tasks/reason"][row] = status, reason

    @property
    def counts(self):
        status = self.file["tasks/status"].asstr()[:]
        return {name: int(np.sum(status == name)) for name in sorted(TERMINAL)}

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None:
                if sum(self.counts.values()) != len(self.n_eval):
                    raise ValueError("linear energy still has pending tasks")
                self.file.attrs["complete"] = True
        finally:
            self.file.close()
