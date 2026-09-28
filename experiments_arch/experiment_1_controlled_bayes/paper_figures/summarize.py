"""Compute all paper plot inputs from saved evaluations and generated datasets."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from ..artifacts.storage import ReferenceResolver, new_directory, sha256, write_json
from .data import PRIORS, validate_bundle

METRICS = ("E", "C", "Z", "N", "C_active", "C_inactive")
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_DRAWS = 2000
GAP_LIMITS = {(10, 1): (-.25, 1.55), (10, 5): (-.04, .22), (10, 10): (-.025, .03),
              (20, 1): (-.4, 2.8), (20, 10): (-.04, .18), (20, 20): (-.035, .035)}


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


class Evaluation:
    """Read explicit, hash-checked references, independent of directory names."""

    def __init__(self, path, resolver=None):
        self.path = Path(path).resolve()
        self.payload = read(self.path)
        require(self.payload.get("complete"), f"Incomplete evaluation: {path}")
        self.resolver = resolver or ReferenceResolver(self.path.parent)
        self.run = read(self.resolve(self.payload["inputs"]["run"]))
        require(self.run.get("complete"), f"Incomplete prediction run: {path}")
        self.args = read(self.input("manifest"))["command_arguments"]
        self.d, self.k = self.args["d"], self.args["k"]
        self.model = self.run["model"]
        self.protocol = self.run["protocol"]
        self.metrics = self.resolve(self.payload["metrics"])

    def resolve(self, ref):
        return self.resolver.resolve(ref, relative_to=self.path.parent)

    def input(self, name):
        return self.resolve(self.payload["inputs"][name])

    def context(self, ids):
        """OLS context statistics, aligned by task ID (never positional pairing)."""
        with h5py.File(self.input("data"), "r") as f:
            stored = f["task_id"][:]
            require(len(np.unique(stored)) == len(stored), "Duplicate data task IDs")
            index = {int(t): i for i, t in enumerate(stored)}
            rows = np.array([index[int(t)] for t in ids])
            c = np.empty((len(ids), self.d), dtype=np.float64)
            n = np.empty(len(ids), dtype=int)
            support = np.empty_like(c, dtype=bool)
            # Sort reads for HDF5, then restore the requested order.
            order = np.argsort(rows)
            for start in range(0, len(rows), 64):
                dest = order[start:start + 64]
                ix = rows[dest]
                ns = f["single_eval_pos"][ix].astype(int)
                x = f["X"][ix, :max(ns)].astype(np.float64)
                y = f["y"][ix, :max(ns)].astype(np.float64)
                y = np.where(np.arange(x.shape[1])[None, :] < ns[:, None], y, 0.)
                c[dest] = np.einsum("bnd,bn->bd", x, y) / ns[:, None]
                n[dest], support[dest] = ns, f["support_mask"][ix].astype(bool)
        return c, n, support


def baseline(evaluation):
    with h5py.File(evaluation.metrics, "r") as f:
        ids = f["tasks/task_id"][:]
        order = np.argsort(ids)
        ids = ids[order]
        bayes = f["tasks/bayes_eta"][:][order]
        signal = f["tasks/bayes_signal"][:][order]
        require(np.all(signal > 1e-12), "Undefined task-normalized error")
        ti = f["states/task_index"][:]
        layer = f["states/layer"][:]
        rows = np.flatnonzero(layer == -1)
        require(len(rows) == len(ids) and np.array_equal(np.sort(ti[rows]), np.arange(len(ids))),
                "Expected exactly one baseline per task")
        rows = rows[np.argsort(ti[rows])][order]
        group = f["methods/affine_blp/states"]
        require(np.all(group["status"].asstr()[:][rows] == "ok"), "Failed baseline estimates")
        require(np.all(f["states/status"].asstr()[:][rows] == "ok"), "Failed baseline predictions")
        values = {}
        for metric in METRICS:
            raw = (f["states/E"][:] if metric == "E" else group[metric][:])[rows]
            if metric == "C_inactive" and evaluation.k == evaluation.d:
                raw = np.zeros(len(ids))
            require(np.isfinite(raw).all(), f"Nonfinite baseline {metric}")
            values[metric] = raw / signal
    return ids, bayes, signal, values


def baseline_summaries(evaluations):
    baselines, gaps, ols = [], [], []
    for d, k in sorted(PRIORS):
        runs = {(a, s): evaluations[d, k, a, s] for a in ("1d", "2d") for s in range(3)}
        arrays = {key: baseline(e) for key, e in runs.items()}
        ids, bayes, signal, _ = arrays["1d", 0]
        for other_ids, other_bayes, other_signal, _ in arrays.values():
            require(np.array_equal(ids, other_ids) and np.array_equal(bayes, other_bayes)
                    and np.array_equal(signal, other_signal), "Baseline tasks/Bayes targets differ")
        c, n, support = runs["1d", 0].context(ids)
        args = runs["1d", 0].args
        dense = args["signal_energy"] / (args["signal_energy"] + d * args["noise_std"] ** 2 / n)
        architectures = {a: dict(seeds=list(range(3)), tasks_per_seed=[len(ids)] * 3,
                                values={m: [float(arrays[a, s][3][m].mean()) for s in range(3)]
                                        for m in METRICS}) for a in ("1d", "2d")}
        baselines.append(dict(d=d, k=k, architectures=architectures,
                              dense_reference_gap=float(np.mean(np.sum((dense[:, None] * c - bayes) ** 2, axis=1)))))
        per_seed = np.array([arrays["1d", s][3]["E"] - arrays["2d", s][3]["E"] for s in range(3)])
        mean = per_seed.mean(axis=0)
        # Published display windows; summary statistics retain every tail task.
        limits = np.array(GAP_LIMITS[d, k])
        edges = np.linspace(*limits, 91)
        counts = [np.histogram(x, edges)[0].tolist() for x in (mean, *per_seed)]
        gaps.append(dict(d=d, k=k, tasks=len(ids), training_seeds=list(range(3)),
                         positive_fraction=float(np.mean(mean > 0)), mean=float(mean.mean()),
                         median=float(np.median(mean)), xlim=limits.tolist(),
                         outside_view_fraction=float(1 - sum(counts[0]) / len(ids)),
                         per_seed_mean=per_seed.mean(axis=1).tolist(), edges=edges.tolist(),
                         histogram_counts=counts))
        error = (c - bayes) ** 2 / signal[:, None]
        models = {"OLS": dict(active=float(np.sum(error * support, axis=1).mean()),
                              inactive=float(np.sum(error * ~support, axis=1).mean()),
                              total=float(error.sum(axis=1).mean()), training_seeds=None)}
        for arch, label in (("1d", "Row-token"), ("2d", "Alternating-axis")):
            seeds = [dict(active=architectures[arch]["values"]["C_active"][s],
                          inactive=architectures[arch]["values"]["C_inactive"][s],
                          total=architectures[arch]["values"]["C"][s]) for s in range(3)]
            models[label] = {key: float(np.mean([v[key] for v in seeds])) for key in seeds[0]}
            models[label]["training_seeds"] = seeds
        sample = np.random.default_rng(20260926).choice(len(ids), min(150, len(ids)), replace=False)
        ols.append(dict(d=d, k=k, tasks=len(ids), models=models,
                        n_context_range=[int(n.min()), int(n.max())],
                        prior={key: args[key] for key in ("signal_energy", "noise_std")},
                        scatter=dict(c=c[sample].tolist(), bayes=bayes[sample].tolist(),
                                     support=support[sample].tolist(), task_ids=ids[sample].tolist())))
    return {"baselines.json": {"conditions": baselines}, "task_gaps.json": {"conditions": gaps},
            "ols.json": {"conditions": ols}}


def bootstrap_indices(ids, args):
    offset = args["train_tasks"] + args["validation_tasks"]
    batches = (ids - offset) // args["generator_batch_size"]
    unique, inverse = np.unique(batches, return_inverse=True)
    # Match the paper's multinomial batch multiplicities exactly.
    draws = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=BOOTSTRAP_DRAWS)
    samples = [np.repeat(np.arange(len(ids)), draw[inverse]) for draw in draws]
    padded = np.full((len(samples), max(map(len, samples))), len(ids), dtype=int)
    for row, ix in enumerate(samples):
        padded[row, :len(ix)] = ix
    return padded, len(unique)


def stats(values, draws, mean=False):
    values = np.asarray(values, dtype=float)
    reduce = np.nanmean if mean else np.nanmedian
    count = int(np.isfinite(values).sum())
    if not count:
        return dict(n=0, center=None, lo=None, hi=None)
    sampled = np.append(values, np.nan)[draws]
    sampled = sampled[np.isfinite(sampled).any(axis=1)]
    samples = reduce(sampled, axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    return dict(n=count, center=float(reduce(values)), lo=float(lo), hi=float(hi))


def intervention_summary(e, model, scope):
    with h5py.File(e.metrics, "r") as f, h5py.File(e.input("predictions"), "r") as pf:
        stored_ids = f["tasks/task_id"][:]
        # Keep the published selection order even when evaluating a larger stored cohort.
        a = e.args
        ids = (np.random.default_rng(0).permutation(a["test_tasks"])[:128]
               + a["train_tasks"] + a["validation_tasks"])
        lookup = {int(t): i for i, t in enumerate(stored_ids)}
        require(all(int(t) in lookup for t in ids), "Missing fixed-cohort tasks")
        order = np.array([lookup[int(t)] for t in ids])
        require(np.array_equal(pf["tasks/task_id"][:], stored_ids), "Prediction task IDs differ")
        c, _, support = e.context(ids)
        bayes = f["tasks/bayes_eta"][:][order]
        groups = [json.loads(s) for s in pf["tasks/token_groups_json"].asstr()[:]]
        draws, clusters = bootstrap_indices(ids, a)
        task = f["responses/task_index"][:]
        layers, doses = f["responses/layer"][:], f["responses/epsilon"][:]
        state_ids = f["responses/state_ids"][:]
        delta_e = f["states/delta_E"][:]
        fidelity = f["methods/affine_blp/states/rho_delta"][:]
        state_status = f["states/status"].asstr()[:]
        estimate_status = f["methods/affine_blp/states/status"].asstr()[:]
        response_status = f["methods/affine_blp/responses/status"].asstr()[:]
        records, signed, matrices = [], None, []
        has_inactive = np.zeros(len(ids), dtype=bool)
        norms = np.linalg.norm(bayes, axis=1)
        median_norm = np.median(norms)
        distance = np.abs(norms - median_norm)
        tied = np.flatnonzero(np.isclose(distance, distance.min(), rtol=0,
                                        atol=8 * np.finfo(float).eps * max(1, median_norm)))
        example = int(tied[np.argmin(ids[tied])])
        for layer in e.protocol["layers"]:
            for dose in e.protocol["epsilons"]:
                role_values = {f"{m}_{r}": np.full(len(ids), np.nan)
                               for m in ("P", "w", "delta_E", "fidelity", "s", "a")
                               for r in ("active", "inactive")}
                for slot, ti in enumerate(order):
                    selected = np.flatnonzero((task == ti) & (layers == layer) & np.isclose(doses, dose))
                    require(len(selected) == 1, "Missing or duplicate response row")
                    row = int(selected[0])
                    require(response_status[row] == "ok", "Failed source response")
                    mapping = groups[ti][str(layer)]
                    states = state_ids[row, :len(mapping)]
                    require(np.all(states >= 0), "Missing source states")
                    require(np.all(state_status[states] == "ok") and
                            np.all(estimate_status[states] == "ok"), "Failed intervention measurements")
                    matrix = f["methods/affine_blp/responses/M"][row, :, :len(mapping)]
                    require(np.isfinite(matrix).all(), "Nonfinite response matrix")
                    total = np.abs(matrix).sum(axis=0)
                    own, signed_own, denom, active = [], [], [], []
                    for j, members in enumerate(mapping):
                        features = sorted({member["feature"] for member in members})
                        own.append(np.abs(matrix[features, j]).sum())
                        signed_own.append((np.sign(c[slot, features]) * matrix[features, j]).sum())
                        denom.append(np.abs(c[slot, features]).sum())
                        active.append(support[slot, features].any())
                    own, denom, active = np.array(own), np.array(denom), np.array(active)
                    has_inactive[slot] = (~active).any()
                    def ratio(num, den):
                        return np.divide(num, den, out=np.full(len(den), np.nan), where=den > 1e-12)
                    values = dict(P=ratio(own, total), w=ratio(own, denom),
                                  s=ratio(signed_own, denom), a=own,
                                  delta_E=delta_e[states], fidelity=fidelity[states])
                    for metric, values_for_sources in values.items():
                        for role, mask in (("active", active), ("inactive", ~active)):
                            valid = mask & np.isfinite(values_for_sources)
                            if valid.any():
                                role_values[f"{metric}_{role}"][slot] = values_for_sources[valid].mean()
                    if model == "nano" and e.k == 1 and scope != "both" and slot == example and np.isclose(dose, .1):
                        require(len(mapping) == e.d and all(
                            {m["feature"] for m in members} == {j} for j, members in enumerate(mapping)),
                            "Matrix display requires one source per raw feature")
                        matrices.append(matrix.tolist())
                record_stats = {key: stats(v, draws, mean=key.startswith("P_"))
                                for key, v in role_values.items()}
                records.append(dict(layer=int(layer) + 1, dose=float(dose), stats=record_stats))
                if e.k == 1 and scope != "both" and layer == max(e.protocol["layers"]) and np.isclose(dose, .1):
                    s = role_values["s_active"]
                    negative = s < 0
                    def negative_median(key):
                        return float(np.median(role_values[key][negative])) if negative.any() else None
                    signed = dict(model=model, d=e.d, scope=scope, layer=int(layer) + 1,
                                  dose=.1, n=len(ids), negative_count=int(negative.sum()),
                                  negative_fraction=float(negative.mean()), median_s=float(np.median(s)),
                                  median_w_negative=negative_median("w_active"),
                                  median_raw_own_response_negative=negative_median("a_active"),
                                  task_id=ids.tolist(), signed_sensitivity=s.tolist(),
                                  raw_own_response=role_values["a_active"].tolist())
        condition = dict(model=model, d=e.d, k=e.k, scope=scope, n=len(ids), n_clusters=clusters,
                         contexts_with_inactive=int(has_inactive.sum()),
                         task_ids=ids.tolist(), records=records)
        example_record = (dict(d=e.d, scope=scope, task_id=int(ids[example]),
                               active_source=int(np.flatnonzero(support[example])[0]) + 1,
                               J=matrices, median_norm=float(median_norm), selected_norm=float(norms[example]))
                          if matrices else None)
    return condition, signed, example_record


def summarize(paths, output_dir):
    resolver = ReferenceResolver(Path.cwd())
    baselines, interventions = {}, {}
    for path in paths:
        e = Evaluation(path, resolver)
        require((e.d, e.k) in PRIORS, f"Unexpected condition: {path}")
        require(e.args["seed"] == 1, "Paper summaries require final data seed 1")
        if not e.protocol["layers"]:
            require(e.model["adapter"] == "nano", "Baseline comparison uses Nano checkpoints")
            key = (e.d, e.k, e.model["architecture"], e.model["training_seed"])
            target = baselines
        else:
            model = "nano" if e.model["adapter"] == "nano" else e.model["version"]
            require(model in ("nano", "v2") and e.protocol["intervention_kind"] == "source",
                    "Expected Nano / TabPFN v2 source interventions")
            if model == "nano":
                require(e.model["architecture"] == "2d" and e.model["training_seed"] == 0,
                        "Paper interventions use the 2d seed-0 checkpoint")
            key = (e.d, e.k, model, e.protocol["row_scope"])
            target = interventions
        require(key not in target, f"Duplicate evaluation for {key}")
        target[key] = e
    expected_base = {(d, k, a, s) for d, k in PRIORS for a in ("1d", "2d") for s in range(3)}
    expected_iv = {(d, k, m, r) for d, k in PRIORS for m in ("nano", "v2")
                   for r in (("both", "context", "query") if k == 1 else ("both",))}
    require(set(baselines) == expected_base, f"Missing/unexpected baseline runs: {set(baselines) ^ expected_base}")
    require(set(interventions) == expected_iv, f"Missing/unexpected intervention runs: {set(interventions) ^ expected_iv}")
    bundle = baseline_summaries(baselines)
    conditions, signed, examples = [], [], []
    for (d, k, model, scope), e in sorted(interventions.items()):
        print(f"Summarizing {model} d={d} k={k} {scope}", flush=True)
        condition, sign, example = intervention_summary(e, model, scope)
        conditions.append(condition)
        if sign:
            signed.append(sign)
        if example:
            examples.append(example)
    bundle.update({"interventions.json": {"protocol": dict(n=128, bootstrap_draws=BOOTSTRAP_DRAWS,
        bootstrap_seed=BOOTSTRAP_SEED, resampling="generator-batch; shared draws within each prior",
        specificity="equal-source means within task, then mean across tasks",
        w_and_delta_E="equal-source means within task, then median across tasks"), "conditions": conditions},
        "signed.json": {"records": signed}, "response_matrices.json": {"records": examples}})
    validate_bundle(bundle)
    with new_directory(output_dir) as directory:
        for name, payload in bundle.items():
            write_json(directory / name, payload)
        write_json(directory / "manifest.json", dict(schema_version=1,
            purpose="Computed from saved evaluations and generated data; no external summaries.",
            files={name: sha256(directory / name) for name in bundle},
            evaluation_sha256=[sha256(path) for path in sorted(paths)]))
    return Path(output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted(args.results_dir.rglob("evaluation.json"))
    summarize(paths, args.output_dir)


if __name__ == "__main__":
    main()
