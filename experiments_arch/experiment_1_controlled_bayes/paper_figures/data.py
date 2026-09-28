"""Validate the paper plotting contract before creating any figure."""

import hashlib
import json
from pathlib import Path

import numpy as np

FILES = ("baselines.json", "task_gaps.json", "interventions.json", "signed.json",
         "response_matrices.json", "ols.json")
PRIORS = {(d, k) for d in (10, 20) for k in (1, d // 2, d)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keyed(records, fields, expected, label):
    result = {tuple(r[f] for f in fields): r for r in records}
    require(len(result) == len(records) and set(result) == expected,
            f"{label}: duplicate, missing, or extra paper conditions")
    return result


def close(a, b, label):
    require(np.allclose(a, b, rtol=1e-9, atol=1e-11), label)


def validate_bundle(bundle):
    baselines = keyed(bundle["baselines.json"]["conditions"], ("d", "k"), PRIORS, "baselines")
    for (d, k), c in baselines.items():
        require(set(c["architectures"]) == {"1d", "2d"}, "expected the two paper architectures")
        for a in c["architectures"].values():
            require(a["seeds"] == [0, 1, 2] and a["tasks_per_seed"] == [32000] * 3,
                    "baseline figures require three seeds and 32000 tasks per seed")
            for name, values in a["values"].items():
                v = np.asarray(values, dtype=float)
                require(v.shape == (3,) and np.isfinite(v).all() and (v >= 0).all(),
                        f"invalid baseline values: {d}/{k}/{name}")
            v = a["values"]
            close(np.array(v["C_active"]) + v["C_inactive"], v["C"], "active + inactive must equal C")
            if k == d:
                close(v["C_inactive"], 0, "dense inactive coefficient error is an empty sum")
        require(np.isfinite(c["dense_reference_gap"]) and c["dense_reference_gap"] >= 0,
                "invalid dense reference gap")
    gaps = keyed(bundle["task_gaps.json"]["conditions"], ("d", "k"), PRIORS, "task gaps")
    for key, c in gaps.items():
        edges = np.asarray(c["edges"])
        counts = np.asarray(c["histogram_counts"])
        require(edges.ndim == 1 and (np.diff(edges) > 0).all() and
                counts.shape == (4, len(edges) - 1) and (counts >= 0).all(), "invalid gap histograms")
        require(c["tasks"] == 32000 and c["training_seeds"] == [0, 1, 2], "invalid gap sample sizes")
        close(counts[0].sum() / c["tasks"], 1 - c["outside_view_fraction"], "gap density lost tail mass")
        v = baselines[key]["architectures"]
        close(c["per_seed_mean"], np.array(v["1d"]["values"]["E"]) - v["2d"]["values"]["E"],
              "gap histograms and baseline summaries describe different runs")
    expected = {(m, d, k, s) for m in ("nano", "v2") for d, k in PRIORS
                for s in (("both", "context", "query") if k == 1 else ("both",))}
    intervention = bundle["interventions.json"]
    require(intervention["protocol"]["n"] == 128 and intervention["protocol"]["bootstrap_draws"] == 2000,
            "expected the paper's 128-task / 2000-bootstrap summary, not a historical cohort")
    conditions = keyed(intervention["conditions"], ("model", "d", "k", "scope"), expected, "interventions")
    cohorts = {}
    for (model, d, k, scope), c in conditions.items():
        ids = c["task_ids"]
        require(c["n"] == len(ids) == len(set(ids)) == 128, "invalid 128-task IDs")
        require(all(isinstance(i, int) and i >= 0 for i in ids), "invalid task identity")
        previous = cohorts.setdefault((d, k), ids)
        require(ids == previous, "models and row scopes must use identical ordered task IDs")
        layers = range(1, 4 if model == "nano" else 13)
        records = keyed(c["records"], ("layer", "dose"),
                        {(l, dose) for l in layers for dose in (0.05, 0.1, 0.2)}, "layer/dose grid")
        for r in records.values():
            for metric in ("P", "w", "delta_E", "fidelity"):
                for role in ("active", "inactive"):
                    stats = r["stats"][f"{metric}_{role}"]
                    if k == d and role == "inactive":
                        require(stats["n"] == 0 and all(stats[s] is None for s in ("center", "lo", "hi")),
                                "dense inactive source statistics must remain undefined")
                        continue
                    values = np.array([stats[s] for s in ("center", "lo", "hi")], dtype=float)
                    require(np.isfinite(values).all() and values[1] <= values[2] and 0 < stats["n"] <= 128,
                            "invalid role statistic or interval")
                    if metric == "P":
                        require(((values >= 0) & (values <= 1)).all(), "specificity outside [0,1]")
    expected_signed = {(m, d, s) for m in ("nano", "v2") for d in (10, 20) for s in ("context", "query")}
    signed = keyed(bundle["signed.json"]["records"], ("model", "d", "scope"), expected_signed, "signed")
    for (m, d, s), r in signed.items():
        require(r["task_id"] == cohorts[d, 1], "signed task IDs do not match the intervention cohort")
        values = np.asarray(r["signed_sensitivity"], dtype=float)
        require(values.shape == (128,) and np.isfinite(values).all(), "invalid signed values")
        require(r["layer"] == (3 if m == "nano" else 12) and r["dose"] == .1,
                "signed results must use the final layer and dose 0.1")
        require(r["negative_count"] == int((values < 0).sum()), "incorrect signed negative count")
    examples = keyed(bundle["response_matrices.json"]["records"], ("d", "scope"),
                     {(d, s) for d in (10, 20) for s in ("context", "query")}, "matrix examples")
    for (d, scope), r in examples.items():
        require(r["task_id"] in cohorts[d, 1] and r["task_id"] == examples[d, "context"]["task_id"],
                "response matrix example must be shared across scopes and in the cohort")
        matrix = np.asarray(r["J"])
        require(matrix.shape == (3, d, d) and np.isfinite(matrix).all() and 1 <= r["active_source"] <= d,
                "invalid response matrix")
    ols = keyed(bundle["ols.json"]["conditions"], ("d", "k"), PRIORS, "OLS")
    for (d, k), c in ols.items():
        require(c["tasks"] == 32000, "OLS summaries require 32000 tasks")
        require(set(c["models"]) == {"OLS", "Row-token", "Alternating-axis"}, "unexpected OLS comparison models")
        for name, values in c["models"].items():
            numeric = np.array([values[key] for key in ("active", "inactive", "total")], dtype=float)
            require(np.isfinite(numeric).all() and (numeric >= 0).all(), "invalid OLS component error")
            close(values["active"] + values["inactive"], values["total"], "OLS support components disagree")
        for name, arch in (("Row-token", "1d"), ("Alternating-axis", "2d")):
            close(c["models"][name]["total"], np.mean(baselines[d, k]["architectures"][arch]["values"]["C"]),
                  "OLS and baseline learned-model summaries disagree")
        scatter = c["scatter"]
        shapes = [np.asarray(scatter[key]).shape for key in ("c", "bayes", "support")]
        require(shapes[0] == shapes[1] == shapes[2] and len(shapes[0]) == 2 and shapes[0][1] == d,
                "invalid OLS scatter arrays")
        require(all(np.isfinite(scatter[key]).all() for key in ("c", "bayes")) and
                np.isin(scatter["support"], (0, 1)).all(), "invalid OLS scatter values")
    return bundle


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["schema_version"] == 1 and set(manifest["files"]) == set(FILES),
            "unsupported plotting bundle manifest")
    bundle = {}
    for name in FILES:
        content = (directory / name).read_bytes()
        require(hashlib.sha256(content).hexdigest() == manifest["files"][name], f"checksum mismatch: {name}")
        bundle[name] = json.loads(content)
    return validate_bundle(bundle), manifest
