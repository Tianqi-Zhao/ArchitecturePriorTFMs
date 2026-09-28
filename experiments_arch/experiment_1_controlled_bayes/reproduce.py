"""Reproduce the controlled paper experiments from checkpoints and seeded data.

Each stage calls the same public Python entry points documented in the README.
Completed, matching artifacts are reused; incomplete or mismatched runs stop.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import h5py

from .artifacts.storage import PredictionRun, SavedEstimate, ReferenceResolver, sha256
from .artifacts.protocol import Protocol
from .paper_figures.data import PRIORS

PREFIX = "experiments_arch.experiment_1_controlled_bayes"


def read(path):
    return json.loads(Path(path).read_text())


def invoke(module, *args):
    command = [sys.executable, "-m", f"{PREFIX}.{module}", *map(str, args)]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def complete_json(path):
    if not path.exists():
        return False
    payload = read(path)
    if not payload.get("complete"):
        raise ValueError(f"Incomplete artifact: {path}; use a new output directory")
    return True


def complete_h5(path, **attributes):
    if not path.exists():
        return False
    with h5py.File(path, "r") as f:
        if not f.attrs.get("complete", False):
            raise ValueError(f"Incomplete artifact: {path}; use a new output directory")
        for name, value in attributes.items():
            if f.attrs.get(name) != value:
                raise ValueError(f"Existing material differs in {name}: {path}")
    return True


def measurements(manifest, data, queries, checkpoint, output, args, *, model="nano", scope=None):
    run = output / "predictions/run.json"
    layers = list(range(3 if model == "nano" else 12)) if scope else []
    if complete_json(run):
        saved = PredictionRun(run)
        saved.validate_reuse(
            protocol=Protocol(layers=layers, row_scope=scope or "both"),
            inputs=dict(manifest=manifest, data=data, projection_queries=queries),
            checkpoint=checkpoint, backend="nano" if model == "nano" else "tabpfn", version="v2")
    else:
        options = ["--manifest", manifest, "--data", data, "--projection-queries", queries,
                   "--checkpoint", checkpoint, "--device", args.device,
                   "--output-dir", run.parent]
        if model == "v2":
            options += ["--backend", "tabpfn", "--tabpfn-version", "v2"]
        if scope:
            options += ["--intervention-kind", "source", "--row-scope", scope,
                        "--layers", *layers, "--epsilons", .05, .1, .2]
        invoke("predict", *options)
    estimate = output / "estimates/estimate.json"
    if not complete_json(estimate):
        invoke("estimate_eta", "--run", run, "--method", "affine_blp", "--output-dir", estimate.parent)
    saved = PredictionRun(run)
    if SavedEstimate(estimate, saved).method != "affine_blp":
        raise ValueError(f"Expected affine estimates: {estimate}")
    evaluation = output / "evaluation/evaluation.json"
    if complete_json(evaluation):
        payload = read(evaluation)
        resolver = ReferenceResolver(evaluation.parent)
        for ref in [payload["metrics"], *payload["inputs"].values()]:
            resolver.resolve(ref)
        if (payload["inputs"]["run"]["sha256"] != sha256(run)
                or payload["inputs"]["estimates/affine_blp"]["sha256"] != sha256(estimate)
                or not payload["config"]["projection_diagnostics"]):
            raise ValueError(f"Evaluation does not match the current run: {evaluation}")
    else:
        invoke("evaluate_saved", "--run", run, "--estimates", estimate,
               "--projection-diagnostics", "--output-dir", evaluation.parent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data", "materials", "baselines", "interventions", "measurements", "all"), required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("paper"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("public_checkpoints"))
    parser.add_argument("--tabpfn-checkpoint", type=Path,
                        help="path to the frozen TabPFN v2 regression checkpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage in ("all", "measurements", "interventions"):
        if args.tabpfn_checkpoint is None or not args.tabpfn_checkpoint.is_file():
            parser.error("this stage requires --tabpfn-checkpoint pointing to a TabPFN v2 regression checkpoint")
    stages = (["data", "materials", "baselines", "interventions"] if args.stage == "all" else
              ["materials", "baselines", "interventions"] if args.stage == "measurements" else [args.stage])
    for stage in stages:
        for d, k in sorted(PRIORS):
            name = f"d{d}_k{k}"
            config = args.checkpoints_dir / "configs" / f"{name}_seed1.json"
            dataset = args.work_dir / "data" / f"{name}_seed1"
            manifest, data = dataset / "manifest.json", dataset / "test.h5"
            queries_dir = args.work_dir / "queries" / name
            queries = queries_dir / "projection_queries.h5"
            cohort_dir = args.work_dir / "cohorts" / name
            cohort = cohort_dir / "tasks.h5"
            cohort_queries = cohort_dir / "queries/projection_queries.h5"
            if stage == "data":
                expected = read(config)["command_arguments"]
                if complete_json(manifest):
                    actual = read(manifest)["command_arguments"]
                    if any(actual[key] != value for key, value in expected.items() if key != "output_dir"):
                        raise ValueError(f"Data configuration differs: {manifest}")
                    resolver = ReferenceResolver(manifest.parent)
                    for split in read(manifest)["splits"].values():
                        resolver.resolve(split)
                else:
                    options = []
                    for key, value in expected.items():
                        options.extend(["--" + key.replace("_", "-"), dataset if key == "output_dir" else value])
                    invoke("data_generation.generate", *options)
            elif stage == "materials":
                if not complete_h5(queries, projection_seed=0, num_projection_queries=1000,
                                   test_split_sha256=sha256(data), dataset_manifest_sha256=sha256(manifest)):
                    invoke("prepare_materials", "queries", "--manifest", manifest, "--data", data,
                           "--count", 1000, "--seed", 0, "--output-dir", queries_dir)
                if not complete_h5(cohort, d=d, k=k, data_seed=1, selection_seed=0,
                                   n_tasks=128, generator_batch_size=32):
                    invoke("prepare_materials", "cohort", "--manifest", config, "--count", 128,
                           "--seed", 0, "--output-dir", cohort_dir)
                if not complete_h5(cohort_queries, projection_seed=0, num_projection_queries=1000,
                                   test_split_sha256=sha256(cohort), dataset_manifest_sha256=sha256(config)):
                    invoke("prepare_materials", "queries", "--manifest", config, "--data", cohort,
                           "--count", 1000, "--seed", 0, "--output-dir", cohort_queries.parent)
            elif stage == "baselines":
                for arch in ("1d", "2d"):
                    for seed in range(3):
                        checkpoint = args.checkpoints_dir / "final" / name / f"arch_{arch}" / f"seed_{seed}/checkpoint.pt"
                        output = args.work_dir / "results" / name / arch / f"seed_{seed}"
                        measurements(manifest, data, queries, checkpoint, output, args)
            elif stage == "interventions":
                for model in ("nano", "v2"):
                    checkpoint = (args.checkpoints_dir / "final" / name / "arch_2d/seed_0/checkpoint.pt"
                                  if model == "nano" else args.tabpfn_checkpoint)
                    for scope in (("both", "context", "query") if k == 1 else ("both",)):
                        output = args.work_dir / "results" / name / "interventions" / model / scope
                        measurements(config, cohort, cohort_queries, checkpoint, output, args, model=model, scope=scope)
    if args.stage == "all":
        invoke("paper_figures.summarize", "--results-dir", args.work_dir / "results",
               "--output-dir", args.work_dir / "plot_data")
        invoke("paper_figures", "--data-dir", args.work_dir / "plot_data",
               "--output-dir", args.work_dir / "figures", "--formats", "pdf", "png")


if __name__ == "__main__":
    main()
