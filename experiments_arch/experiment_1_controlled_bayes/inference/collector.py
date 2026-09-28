"""Collect immutable predictions for one explicit dataset.

Mean projection input gradients are optional; enable them only when a later
gradient eta estimate will read the saved vectors.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np

from ..artifacts.data import DatasetManifest, ProjectionQueries
from ..artifacts.storage import (PredictionWriter, collection_log, new_directory,
                                 reference, sha256, write_json)
from ..artifacts.protocol import CODE_VERSION, Gate, Protocol, SCHEMA_VERSION
from .base import PredictionRequest


def state_plan(protocol, groups):
    gates = [Gate()]
    for layer in protocol.layers:
        if protocol.intervention_kind == "feature_branch":
            gates.extend(Gate(layer=layer, epsilon=epsilon, row_scope=protocol.row_scope,
                              kind="feature_branch") for epsilon in protocol.epsilons)
            continue
        count = len(groups[str(layer)])
        sources = protocol.source_tokens or tuple(range(count))
        if any(j >= count for j in sources):
            raise ValueError("source token lies outside the actual feature mapping")
        for epsilon in protocol.epsilons:
            gates.extend(Gate(layer, j, epsilon, row_scope=protocol.row_scope) for j in sources)
    return gates


class PredictionCollector:
    """Run a frozen model and write material; estimation is a separate stage."""

    def __init__(self, *, manifest, data, projection_queries, output_dir, adapter,
                 protocol=Protocol(), capture_gradients=False):
        self.manifest = DatasetManifest(manifest)
        self.dataset = self.manifest.dataset(data)
        self.queries = ProjectionQueries(projection_queries, self.dataset)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.adapter, self.protocol = adapter, protocol
        adapter.validate_protocol(protocol)
        self.capture_gradients = bool(capture_gradients)

    def _configuration(self):
        directory = self.output_dir
        return dict(
            schema_version=SCHEMA_VERSION,
            inputs={"manifest": reference(self.manifest.path, directory),
                    "data": reference(self.dataset.path, directory),
                    "projection_queries": reference(self.queries.path, directory)},
            model={**self.adapter.metadata, "collector_version": CODE_VERSION},
            protocol=self.protocol.to_dict(),
            gradient_capture=("mean_projection_input_gradient"
                              if self.capture_gradients else "none"))

    def _warn(self, message):
        """Persist each failure and make it immediately visible in the terminal."""
        self.log.write(message + "\n")
        self.log.flush()
        print(f"WARNING: {message}", file=sys.stderr, flush=True)

    def _save_gradient(self, writer, row, output, count, *, task_id):
        # A derivative failure must not discard a valid forward prediction.
        status = "ok"
        if output.gradient_error:
            status = "failed"
        elif output.gradient_mean is None or output.gradient_count != count:
            status = "unavailable"
        try:
            writer.write_gradient(row, output.gradient_mean,
                                  output.gradient_count, status)
        except ValueError as error:
            writer.write_gradient(row, status="failed")
            self._warn(f"task={task_id} state={row} gradient: {error}")
            return
        if status != "ok":
            self._warn(f"task={task_id} state={row} gradient: "
                       f"{output.gradient_error or 'incomplete projection gradients'}")

    def _collect_batch(self, tasks, predictions):
        p = self.protocol
        capacity = p.task_batch_size if self.adapter.supports_batch_prediction else 1
        plans = []
        # Reserve rows in the original task/state order before regrouping work.
        for task in tasks:
            groups = self.adapter.token_groups(p.layers)
            gates = state_plan(p, groups)
            start = predictions.plan(task, groups, gates)
            plans.append((task, gates, start, self.queries.read(task.task_id)))
        for phase in ("projection", "evaluation"):
            pending = {}
            # Fill with distinct tasks first; use independent states to fill
            # remaining slots when there are few tasks of a compatible shape.
            for offset in range(max(len(plan[1]) for plan in plans)):
                for task, gates, start, projection_x in plans:
                    if offset >= len(gates):
                        continue
                    x = projection_x if phase == "projection" else task.evaluation_x
                    key = (len(task.context_x), len(x))
                    jobs = pending.setdefault(key, [])
                    jobs.append((start + offset, PredictionRequest(
                        task, x, gates[offset], p.task_seed(task.task_id))))
                    if len(jobs) == capacity:
                        self._collect_requests(jobs, phase, predictions)
                        jobs.clear()
            for jobs in pending.values():
                if jobs:
                    self._collect_requests(jobs, phase, predictions)

    def _collect_requests(self, jobs, phase, writer):
        gradients = phase == "projection" and self.capture_gradients
        try:
            outputs = self.adapter.predict_batch(
                [request for _, request in jobs], gradients=gradients,
                batch_size=self.protocol.query_batch_size)
            if len(outputs) != len(jobs):
                raise ValueError("batch predictions must cover every request")
            for (_, request), output in zip(jobs, outputs):
                if (output.values.shape != (len(request.queries),)
                        or not np.isfinite(output.values).all()):
                    raise ValueError("predictions must be finite and cover all queries")
        except (RuntimeError, ValueError, AssertionError) as error:
            with writer.batch():
                for row, request in jobs:
                    writer.write(row, phase, status="failed")
                    if phase == "projection":
                        writer.write_gradient(row, status="unavailable")
                    self._warn(f"task={request.task.task_id} state={row} {phase}: {error}")
            return
        with writer.batch():
            for (row, request), output in zip(jobs, outputs):
                writer.write(row, phase, output.values)
                if phase == "projection":
                    if gradients:
                        self._save_gradient(writer, row, output, len(request.queries),
                                            task_id=request.task.task_id)
                    else:
                        writer.write_gradient(row, status="unavailable")

    def run(self):
        config = self._configuration()
        dtype = np.dtype(self.adapter.metadata["dtype"])
        with new_directory(self.output_dir) as directory, ExitStack() as stack:
            self.log = stack.enter_context(collection_log(self.output_dir).open("w"))
            predictions = PredictionWriter(
                directory / "predictions.h5",
                qp=self.queries.count, qe=self.dataset.max_eval, d=self.dataset.prior.d, dtype=dtype)
            stack.callback(lambda: predictions.file.close())
            for batch in self.dataset.batches(batch_size=self.protocol.task_batch_size):
                self._collect_batch(batch, predictions)
            predictions.close(complete=True)
            config["predictions"] = {"path": "predictions.h5",
                                     "sha256": sha256(directory / "predictions.h5")}
            config["complete"] = True
            write_json(directory / "run.json", config)
        return self.output_dir / "run.json"
