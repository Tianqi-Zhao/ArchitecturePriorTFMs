"""Validate saved prediction indices and align source-response states."""

from dataclasses import dataclass
from functools import cached_property
import numpy as np

from .protocol import SCHEMA_VERSION


@dataclass(frozen=True)
class PredictionIndex:
    task_ids: np.ndarray
    n_eval: np.ndarray
    baseline_ids: np.ndarray
    task_index: np.ndarray
    layer: np.ndarray
    source_token: np.ndarray
    epsilon: np.ndarray
    groups: list[dict]
    kind: np.ndarray

    @classmethod
    def read(cls, root, dataset, protocol):
        import json
        from .data import id_index

        tasks, states = root["tasks"], root["states"]
        ids = tasks["task_id"][:]
        id_index(ids)
        index = cls(ids, tasks["n_eval"][:], tasks["baseline_state_id"][:],
                    states["task_index"][:], states["layer"][:], states["source_token"][:],
                    states["epsilon"][:],
                    [json.loads(value) for value in tasks["token_groups_json"].asstr()[:]],
                    cls._read_kinds(root, protocol))
        t, s = len(ids), len(index.task_index)
        if t == 0 or s == 0:
            raise ValueError("prediction task/state tables must not be empty")
        if (index.kind.shape != (s,)
                or not set(index.kind) <= {"baseline", "source", "feature_branch"}):
            raise ValueError("invalid state intervention kinds")
        for values in (index.n_eval, index.baseline_ids):
            if values.shape != (t,) or values.dtype.kind not in "iu":
                raise ValueError("invalid task indices/lengths")
        for values in (index.task_index, index.layer, index.source_token):
            if values.shape != (s,) or values.dtype.kind not in "iu":
                raise ValueError("invalid state integer indices")
        if (index.epsilon.shape != (s,) or not np.isfinite(index.epsilon).all()
                or np.any(index.task_index < 0) or np.any(index.task_index >= t)
                or np.any(index.baseline_ids < 0) or np.any(index.baseline_ids >= s)):
            raise ValueError("invalid state/baseline index bounds")
        for name in ("f_proj", "f_eval"):
            if states[name].ndim != 2 or states[name].shape[0] != s:
                raise ValueError(f"invalid prediction shape: {name}")
        for name in ("projection_status", "evaluation_status", "gradient_status"):
            if states[name].shape != (s,) or not set(states[name].asstr()[:]) <= {
                    "ok", "failed", "unavailable"}:
                raise ValueError("invalid prediction terminal status")
        counts = states["gradient_count"][:]
        if (states["gradient_mean"].shape != (s, dataset.prior.d)
                or counts.shape != (s,) or counts.dtype.kind not in "iu"
                or np.any(counts < 0) or np.any(counts > states["f_proj"].shape[1])):
            raise ValueError("invalid saved gradient shape or count")
        ok = states["gradient_status"].asstr()[:] == "ok"
        if np.any(counts[ok] != states["f_proj"].shape[1]):
            raise ValueError("saved gradients do not cover all projection queries")
        seen = set()
        for ti, task_id in enumerate(ids):
            if int(task_id) not in dataset.index:
                raise ValueError("saved prediction task ID is absent from data")
            if index.n_eval[ti] != dataset.n_eval[dataset.index[int(task_id)]]:
                raise ValueError("saved evaluation query length differs from task")
            if index.n_eval[ti] > states["f_eval"].shape[1]:
                raise ValueError("saved evaluation capacity is too small")
            baseline = index.baseline_ids[ti]
            if (index.task_index[baseline] != ti or index.layer[baseline] != -1
                    or index.source_token[baseline] != -1 or index.epsilon[baseline] != 0
                    or index.kind[baseline] != "baseline"):
                raise ValueError("baseline does not belong to its task")
            if set(index.groups[ti]) != {str(layer) for layer in protocol.layers}:
                raise ValueError("token mapping does not match requested layers")
            for groups in index.groups[ti].values():
                if not groups:
                    raise ValueError("feature token mapping must not be empty")
                for members in groups:
                    if not members or any(not 0 <= m["feature"] < dataset.prior.d
                                          or m["column"] != m["feature"] or m["slot"] < 0
                                          for m in members):
                        raise ValueError("invalid raw feature mapping")
        for row in range(s):
            ti, layer, j, eps = (index.task_index[row], index.layer[row],
                                  index.source_token[row], index.epsilon[row])
            key = (ti, index.kind[row], layer, j, eps)
            if key in seen:
                raise ValueError("duplicate task/layer/source/epsilon state")
            seen.add(key)
            if row == index.baseline_ids[ti]:
                continue
            if (index.kind[row] != protocol.intervention_kind
                    or layer not in protocol.layers or eps not in protocol.epsilons
                    or not 0 < eps <= 1):
                raise ValueError("state differs from intervention protocol")
            if index.kind[row] == "feature_branch":
                if j != -1:
                    raise ValueError("feature_branch state must have source_token=-1")
            elif (not 0 <= j < len(index.groups[ti][str(layer)])
                  or protocol.source_tokens and j not in protocol.source_tokens):
                raise ValueError("state differs from intervention protocol")
        return index

    @staticmethod
    def _read_kinds(root, protocol):
        states = root["states"]
        if "kind" in states:
            return states["kind"].asstr()[:]
        if root.attrs.get("schema_version") == SCHEMA_VERSION:
            raise ValueError("prediction states are missing kind")
        if protocol.intervention_kind != "source":
            raise ValueError("legacy states only support source interventions")
        return np.where((states["layer"][:] == -1) & (states["source_token"][:] == -1)
                        & (states["epsilon"][:] == 0), "baseline", "source")

    @cached_property
    def task_rows(self):
        order = np.argsort(self.task_index, kind="stable")
        counts = np.bincount(self.task_index, minlength=len(self.task_ids))
        offsets = np.concatenate(([0], np.cumsum(counts)))
        return tuple(order[offsets[i]:offsets[i + 1]] for i in range(len(counts)))

    def rows(self, task_index):
        return self.task_rows[task_index]

    def responses(self, protocol):
        if protocol.intervention_kind != "source":
            return
        for ti, groups in enumerate(self.groups):
            rows = self.rows(ti)
            for layer in protocol.layers:
                for epsilon in protocol.epsilons:
                    state_ids = np.full(len(groups[str(layer)]), -1, dtype=np.int64)
                    selected = rows[(self.layer[rows] == layer) & (self.epsilon[rows] == epsilon)]
                    state_ids[self.source_token[selected]] = selected
                    yield ti, layer, epsilon, state_ids
