"""Command-line entry points for the Part I experiments.

The concrete composition root must create one ``DefaultPretrainedModelFactory``
from the loaded experiment seed/input-width policy and inject it into both the
Part I.1 prediction and Part I.2 functional-analysis implementations.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from experiment_core.pretrained import (
    PRETRAINED_MODEL_SPECS,
    DefaultPretrainedModelFactory,
)
from experiment_core.runtime import default_device
from experiments_arch.experiment_0_pretrained.config import load_config
from experiments_arch.experiment_0_pretrained.data.prepare import DefaultDatasetPreparer
from experiments_arch.experiment_0_pretrained.evaluate import (
    DefaultExpandedPartI1Evaluator,
    DefaultFriedmanSweepEvaluator,
    DefaultFriedmanSweepPartI2Evaluator,
    DefaultPartI2Evaluator,
    DefaultRawPredictionEvaluator,
)
from experiments_arch.experiment_0_pretrained.functional import DefaultPartI2Runner
from experiments_arch.experiment_0_pretrained.plot import (
    DefaultPublicationPlotter,
    DefaultTabDPTVersionPlotter,
)
from experiments_arch.experiment_0_pretrained.predict import DefaultPredictionRunner
from experiments_arch.experiment_0_pretrained.tabdpt_versions import (
    collect_plot_inputs as collect_tabdpt_version_plot_inputs,
    prepare_indexes as prepare_tabdpt_version_indexes,
    smoke_test as tabdpt_release_smoke_test,
    summarize_versions as summarize_tabdpt_versions,
)


_NANOTAB_ROOT = Path(__file__).resolve().parents[2]


def _comma_separated(value: str) -> tuple[str, ...] | None:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or None


def _comma_floats(value: str) -> tuple[float, ...] | None:
    values = _comma_separated(value)
    return None if values is None else tuple(float(item) for item in values)


def _project_checkpoint_dir(value: str) -> Path:
    """Resolve a portable checkpoint directory relative to ``NanoTab``."""

    path = Path(value)
    if path.is_absolute():
        raise ValueError("--checkpoint-dir must be relative to the NanoTab root")
    resolved = (_NANOTAB_ROOT / path).resolve()
    try:
        resolved.relative_to(_NANOTAB_ROOT)
    except ValueError as error:
        raise ValueError(
            "--checkpoint-dir must stay inside the NanoTab project"
        ) from error
    return resolved


def _checkpoint_paths(checkpoint_dir: Path, model_ids) -> dict[str, str]:
    return {
        model_id: str(
            checkpoint_dir / Path(PRETRAINED_MODEL_SPECS[model_id].default_model_path).name
        )
        for model_id in model_ids
        if model_id in PRETRAINED_MODEL_SPECS
    }


def _model_factory(config, checkpoint_dir: Path) -> DefaultPretrainedModelFactory:
    return DefaultPretrainedModelFactory(
        seed=config.run_seed,
        max_input_features=config.max_input_features,
        checkpoint_paths=_checkpoint_paths(
            checkpoint_dir, (model.model_id for model in config.models)
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 0: pretrained irrelevant-feature dose response"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="materialize unified datasets")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--out-dir", default="")

    predict = commands.add_parser("predict", help="write per-dataset raw predictions")
    predict.add_argument("--config", required=True)
    predict.add_argument("--data-index", required=True)
    predict.add_argument("--out-dir", default="")
    predict.add_argument("--models", default="", help="optional model-id subset")
    predict.add_argument("--device", default="")
    predict.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory relative to the NanoTab project root",
    )

    evaluate = commands.add_parser("evaluate", help="evaluate saved raw predictions")
    evaluate.add_argument("--predictions-dir", required=True)
    evaluate.add_argument("--out-dir", required=True)

    evaluate_sweeps = commands.add_parser(
        "evaluate-friedman-sweeps",
        help="aggregate saved Friedman sample-size/noise predictions",
    )
    evaluate_sweeps.add_argument("--config", required=True)
    evaluate_sweeps.add_argument("--predictions-dir", required=True)
    evaluate_sweeps.add_argument("--out-dir", required=True)

    evaluate_expanded = commands.add_parser(
        "evaluate-expanded-part-i1",
        help="combine equally weighted Friedman sweep settings with OpenML",
    )
    evaluate_expanded.add_argument("--config", required=True)
    evaluate_expanded.add_argument("--part-i1-evaluation-dir", required=True)
    evaluate_expanded.add_argument("--friedman-sweep-evaluation-dir", required=True)
    evaluate_expanded.add_argument("--out-dir", required=True)
    evaluate_expanded.add_argument("--tasks-per-setting", type=int, default=50)

    functional = commands.add_parser(
        "part-i2", help="run cached permutation and PDP functional suppression"
    )
    functional.add_argument("--config", required=True)
    functional.add_argument("--data-index", required=True)
    functional.add_argument("--out-dir", default="")
    functional.add_argument(
        "--models", default="", help="optional subset of part_i2.models"
    )
    functional.add_argument("--datasets", default="", help="optional dataset-id subset")
    functional.add_argument("--ratios", default="", help="optional ratio subset")
    functional.add_argument("--device", default="")
    functional.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory relative to the NanoTab project root",
    )

    evaluate_i2 = commands.add_parser(
        "evaluate-part-i2", help="evaluate saved Part I.2 functional artifacts"
    )
    evaluate_i2.add_argument("--run-dir", required=True)
    evaluate_i2.add_argument("--out-dir", required=True)

    evaluate_i2_sweeps = commands.add_parser(
        "evaluate-part-i2-friedman-sweeps",
        help="aggregate Part I.2 Friedman sample-size/noise sweeps",
    )
    evaluate_i2_sweeps.add_argument("--config", required=True)
    evaluate_i2_sweeps.add_argument("--run-dir", required=True)
    evaluate_i2_sweeps.add_argument("--out-dir", required=True)

    run = commands.add_parser(
        "run-part-i1", help="prepare, predict, and evaluate Part I.1"
    )
    run.add_argument("--config", required=True)
    run.add_argument("--out-dir", default="")
    run.add_argument("--models", default="", help="optional model-id subset")
    run.add_argument("--device", default="")
    run.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory relative to the NanoTab project root",
    )

    run_sweeps = commands.add_parser(
        "run-friedman-sweeps",
        help="prepare, predict, and aggregate Friedman sample-size/noise sweeps",
    )
    run_sweeps.add_argument("--config", required=True)
    run_sweeps.add_argument("--out-dir", default="")
    run_sweeps.add_argument("--device", default="")
    run_sweeps.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory relative to the NanoTab project root",
    )

    run_i2 = commands.add_parser(
        "run-part-i2", help="prepare/reuse data, run, and evaluate Part I.2"
    )
    run_i2.add_argument("--config", required=True)
    run_i2.add_argument(
        "--data-index", default="",
        help="existing prepared index; omit to run the cache-aware prepare stage",
    )
    run_i2.add_argument("--out-dir", default="")
    run_i2.add_argument(
        "--models", default="", help="optional subset of part_i2.models"
    )
    run_i2.add_argument("--datasets", default="", help="optional dataset-id subset")
    run_i2.add_argument("--ratios", default="", help="optional ratio subset")
    run_i2.add_argument("--device", default="")
    run_i2.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory relative to the NanoTab project root",
    )

    run_i2_sweeps = commands.add_parser(
        "run-part-i2-friedman-sweeps",
        help="prepare, run, and aggregate Part I.2 Friedman sweeps",
    )
    run_i2_sweeps.add_argument("--config", required=True)
    run_i2_sweeps.add_argument(
        "--data-index", default="",
        help="existing sweep prepared index; omit to run cache-aware prepare",
    )
    run_i2_sweeps.add_argument("--out-dir", default="")
    run_i2_sweeps.add_argument(
        "--models", default="", help="optional subset of part_i2.models"
    )
    run_i2_sweeps.add_argument("--device", default="")
    run_i2_sweeps.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory relative to the NanoTab project root",
    )

    publication = commands.add_parser(
        "plot-publication",
        help="build the compact median/IQR Part I publication bundle",
    )
    publication.add_argument("--artifact-root", type=Path, required=True)

    version_indexes = commands.add_parser(
        "prepare-tabdpt-version-indexes",
        help="reuse the Part I data caches for the TabDPT release replication",
    )
    version_indexes.add_argument("--friedman-index", type=Path, required=True)
    version_indexes.add_argument("--part-i1-index", type=Path, required=True)
    version_indexes.add_argument("--out-dir", type=Path, required=True)

    version_summary = commands.add_parser(
        "evaluate-tabdpt-versions",
        help="summarize the TabDPT release replication across versions",
    )
    for version in ("v1-0", "v1-1"):
        version_summary.add_argument(f"--friedman-{version}", type=Path, required=True)
        version_summary.add_argument(f"--openml-{version}", type=Path, required=True)
    version_summary.add_argument("--friedman-v1-3", type=Path)
    version_summary.add_argument("--openml-v1-3", type=Path)
    version_summary.add_argument(
        "--v1-2-effects", type=Path, action="append", default=[],
        help="existing paired_effects.csv files containing TabDPT v1.2",
    )
    version_summary.add_argument("--out-dir", type=Path, required=True)

    version_figures = commands.add_parser(
        "plot-tabdpt-versions",
        help="build the appendix figures and table for the TabDPT releases",
    )
    version_figures.add_argument("--versions-root", type=Path, required=True)

    version_smoke = commands.add_parser(
        "smoke-test-tabdpt-release",
        help="load one TabDPT release and run a tiny regression episode",
    )
    version_smoke.add_argument("--config", type=Path, required=True)
    version_smoke.add_argument("--model", required=True)
    version_smoke.add_argument("--checkpoint", type=Path, required=True)
    version_smoke.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-tabdpt-version-indexes":
        friedman, openml = prepare_tabdpt_version_indexes(
            args.friedman_index, args.part_i1_index, args.out_dir
        )
        print(json.dumps({
            "friedman_index": str(friedman), "openml_index": str(openml),
        }, indent=2))
        return
    if args.command == "evaluate-tabdpt-versions":
        if (args.friedman_v1_3 is None) != (args.openml_v1_3 is None):
            parser.error("--friedman-v1-3 and --openml-v1-3 must be given together")
        prediction_dirs = {
            "v1.0": (args.friedman_v1_0, args.openml_v1_0),
            "v1.1": (args.friedman_v1_1, args.openml_v1_1),
        }
        if args.friedman_v1_3 is not None:
            prediction_dirs["v1.3"] = (args.friedman_v1_3, args.openml_v1_3)
        output = summarize_tabdpt_versions(
            prediction_dirs, list(args.v1_2_effects), args.out_dir
        )
        print(json.dumps({
            "output_dir": str(output),
            "summary_path": str(output / "tabdpt_version_summary.csv"),
        }, indent=2))
        return
    if args.command == "plot-tabdpt-versions":
        versions_root = args.versions_root.resolve()
        reference_root = versions_root.parent / "experiment_0_pretrained"
        inputs = collect_tabdpt_version_plot_inputs(
            versions_root, reference_root,
        )
        result = DefaultTabDPTVersionPlotter().plot(
            part_i1_effects=inputs["part_i1"],
            part_i2_friedman=inputs["part_i2_friedman"],
            part_i2_openml=inputs["part_i2_openml"],
            output_root=versions_root / "publication",
        )
        print(json.dumps({
            "figures": [str(path) for path in result["figures"]],
            "table": str(result["table"]),
            "n_part_i1_rows": len(inputs["part_i1"]),
            "n_part_i2_friedman_rows": len(inputs["part_i2_friedman"]),
            "n_part_i2_openml_rows": len(inputs["part_i2_openml"]),
        }, indent=2))
        return
    if args.command == "smoke-test-tabdpt-release":
        print(json.dumps(tabdpt_release_smoke_test(
            args.config, args.model, args.checkpoint, args.device
        ), indent=2))
        return
    if args.command == "plot-publication":
        figures = DefaultPublicationPlotter().plot(
            artifact_root=args.artifact_root,
        )
        print(json.dumps({
            "part_i1_figures": [str(path) for path in figures["part_i1"]],
            "part_i2_figures": [str(path) for path in figures["part_i2"]],
        }, indent=2))
        return
    if args.command == "prepare":
        config = load_config(args.config)
        output_dir = Path(args.out_dir) if args.out_dir else config.artifact_root
        result = DefaultDatasetPreparer().prepare(config, output_dir)
        print(json.dumps({
            "data_run_id": result.data_run_id,
            "output_dir": str(result.output_dir),
            "index_path": str(result.index_path),
            "skips_path": str(result.skips_path),
            "num_datasets": result.num_datasets,
            "num_conditions": result.num_conditions,
            "num_skips": result.num_skips,
        }, indent=2))
        return
    if args.command == "predict":
        config = load_config(args.config)
        output_dir = Path(args.out_dir) if args.out_dir else config.artifact_root
        checkpoint_dir = _project_checkpoint_dir(args.checkpoint_dir)
        factory = _model_factory(config, checkpoint_dir)
        result = DefaultPredictionRunner(factory).run(
            config=config,
            data_index=Path(args.data_index),
            output_dir=output_dir,
            device=args.device or str(default_device()),
            model_ids=_comma_separated(args.models),
        )
        print(json.dumps({
            "prediction_run_id": result.prediction_run_id,
            "output_dir": str(result.output_dir),
            "num_files": result.num_files,
            "num_rows": result.num_rows,
            "num_failures": result.num_failures,
        }, indent=2))
        return
    if args.command == "evaluate":
        evaluation = DefaultRawPredictionEvaluator().evaluate(
            Path(args.predictions_dir), Path(args.out_dir)
        )
        print(json.dumps({
            "evaluation_run_id": evaluation.evaluation_run_id,
            "output_dir": str(evaluation.output_dir),
            "metrics_path": str(evaluation.metrics_path),
            "paired_effects_path": str(evaluation.paired_effects_path),
            "model_contrasts_path": str(evaluation.model_contrasts_path),
            "contrast_summary_path": str(evaluation.contrast_summary_path),
            "summary_path": str(evaluation.summary_path),
            "coverage_path": str(evaluation.coverage_path),
        }, indent=2))
        return
    if args.command == "evaluate-friedman-sweeps":
        config = load_config(args.config)
        evaluation = DefaultRawPredictionEvaluator().evaluate(
            Path(args.predictions_dir), Path(args.out_dir)
        )
        sweep = DefaultFriedmanSweepEvaluator().evaluate(evaluation, config)
        print(json.dumps({
            "evaluation_run_id": evaluation.evaluation_run_id,
            "friedman_sweep_evaluation_id": sweep.evaluation_run_id,
            "evaluation_dir": str(evaluation.output_dir),
            "sweep_dir": str(sweep.output_dir),
            "model_summary_path": str(sweep.model_summary_path),
            "contrast_summary_path": str(sweep.contrast_summary_path),
        }, indent=2))
        return
    if args.command == "evaluate-expanded-part-i1":
        evaluation = DefaultExpandedPartI1Evaluator().evaluate(
            part_i1_evaluation_dir=Path(args.part_i1_evaluation_dir),
            friedman_sweep_evaluation_dir=Path(
                args.friedman_sweep_evaluation_dir
            ),
            config=load_config(args.config),
            output_dir=Path(args.out_dir),
            tasks_per_setting=args.tasks_per_setting,
        )
        print(json.dumps({
            "evaluation_run_id": evaluation.evaluation_run_id,
            "output_dir": str(evaluation.output_dir),
            "paired_effects_path": str(evaluation.paired_effects_path),
            "model_contrasts_path": str(evaluation.model_contrasts_path),
            "contrast_summary_path": str(evaluation.contrast_summary_path),
            "summary_path": str(evaluation.summary_path),
            "manifest_path": str(evaluation.manifest_path),
        }, indent=2))
        return
    if args.command == "part-i2":
        config = load_config(args.config)
        output_dir = Path(args.out_dir) if args.out_dir else config.artifact_root
        checkpoint_dir = _project_checkpoint_dir(args.checkpoint_dir)
        result = DefaultPartI2Runner(
            _model_factory(config, checkpoint_dir)
        ).run(
            config=config,
            data_index=Path(args.data_index),
            output_dir=output_dir,
            device=args.device or str(default_device()),
            model_ids=_comma_separated(args.models),
            dataset_ids=_comma_separated(args.datasets),
            ratios=_comma_floats(args.ratios),
        )
        print(json.dumps({
            "data_run_id": result.data_run_id,
            "part_i2_run_id": result.part_i2_run_id,
            "output_dir": str(result.output_dir),
            "num_files": result.num_files,
            "num_metric_rows": result.num_metric_rows,
            "num_failures": result.num_failures,
            "cache_hits": result.cache_hits,
            "cache_misses": result.cache_misses,
            "elapsed_seconds": result.elapsed_seconds,
        }, indent=2))
        return
    if args.command == "evaluate-part-i2":
        evaluation = DefaultPartI2Evaluator().evaluate(
            Path(args.run_dir), Path(args.out_dir)
        )
        print(json.dumps({
            "evaluation_run_id": evaluation.evaluation_run_id,
            "output_dir": str(evaluation.output_dir),
            "task_effects_path": str(evaluation.task_effects_path),
            "summary_path": str(evaluation.summary_path),
            "model_contrasts_path": str(evaluation.contrasts_path),
            "pdp_curves_path": str(evaluation.pdp_curves_path),
            "dataset_dimensions_path": str(evaluation.dataset_dimensions_path),
        }, indent=2))
        return
    if args.command == "evaluate-part-i2-friedman-sweeps":
        evaluation = DefaultFriedmanSweepPartI2Evaluator().evaluate(
            Path(args.run_dir), load_config(args.config), Path(args.out_dir)
        )
        print(json.dumps({
            "evaluation_run_id": evaluation.evaluation_run_id,
            "output_dir": str(evaluation.output_dir),
            "task_effects_path": str(evaluation.task_effects_path),
            "summary_path": str(evaluation.summary_path),
            "model_contrasts_path": str(evaluation.contrasts_path),
            "coverage_path": str(evaluation.coverage_path),
        }, indent=2))
        return
    if args.command == "run-part-i1":
        config = load_config(args.config)
        output_dir = Path(args.out_dir) if args.out_dir else config.artifact_root
        checkpoint_dir = _project_checkpoint_dir(args.checkpoint_dir)
        device = args.device or str(default_device())
        model_ids = _comma_separated(args.models)
        prepared = DefaultDatasetPreparer().prepare(config, output_dir)
        prediction = DefaultPredictionRunner(
            _model_factory(config, checkpoint_dir)
        ).run(
            config=config,
            data_index=prepared.index_path,
            output_dir=output_dir,
            device=device,
            model_ids=model_ids,
        )
        evaluation = DefaultRawPredictionEvaluator().evaluate(
            prediction.output_dir, output_dir
        )
        print(json.dumps({
            "data_run_id": prepared.data_run_id,
            "prediction_run_id": prediction.prediction_run_id,
            "evaluation_run_id": evaluation.evaluation_run_id,
            "prepared_conditions": prepared.num_conditions,
            "prediction_files": prediction.num_files,
            "prediction_failures": prediction.num_failures,
            "evaluation_dir": str(evaluation.output_dir),
        }, indent=2))
        return
    if args.command == "run-friedman-sweeps":
        config = load_config(args.config)
        output_dir = Path(args.out_dir) if args.out_dir else config.artifact_root
        checkpoint_dir = _project_checkpoint_dir(args.checkpoint_dir)
        device = args.device or str(default_device())
        prepared = DefaultDatasetPreparer().prepare(config, output_dir)
        prediction = DefaultPredictionRunner(
            _model_factory(config, checkpoint_dir)
        ).run(
            config=config,
            data_index=prepared.index_path,
            output_dir=output_dir,
            device=device,
            model_ids=None,
        )
        evaluation = DefaultRawPredictionEvaluator().evaluate(
            prediction.output_dir, output_dir
        )
        sweep = DefaultFriedmanSweepEvaluator().evaluate(evaluation, config)
        print(json.dumps({
            "data_run_id": prepared.data_run_id,
            "prediction_run_id": prediction.prediction_run_id,
            "evaluation_run_id": evaluation.evaluation_run_id,
            "friedman_sweep_evaluation_id": sweep.evaluation_run_id,
            "prepared_conditions": prepared.num_conditions,
            "prediction_files": prediction.num_files,
            "prediction_failures": prediction.num_failures,
            "sweep_dir": str(sweep.output_dir),
        }, indent=2))
        return
    if args.command == "run-part-i2":
        config = load_config(args.config)
        output_dir = Path(args.out_dir) if args.out_dir else config.artifact_root
        checkpoint_dir = _project_checkpoint_dir(args.checkpoint_dir)
        device = args.device or str(default_device())
        if args.data_index:
            data_index = Path(args.data_index)
        else:
            prepared = DefaultDatasetPreparer().prepare(config, output_dir)
            data_index = prepared.index_path
        functional = DefaultPartI2Runner(
            _model_factory(config, checkpoint_dir)
        ).run(
            config=config,
            data_index=data_index,
            output_dir=output_dir,
            device=device,
            model_ids=_comma_separated(args.models),
            dataset_ids=_comma_separated(args.datasets),
            ratios=_comma_floats(args.ratios),
        )
        evaluation = DefaultPartI2Evaluator().evaluate(
            functional.output_dir, output_dir
        )
        print(json.dumps({
            "data_run_id": functional.data_run_id,
            "part_i2_run_id": functional.part_i2_run_id,
            "evaluation_run_id": evaluation.evaluation_run_id,
            "part_i2_files": functional.num_files,
            "part_i2_metric_rows": functional.num_metric_rows,
            "part_i2_failures": functional.num_failures,
            "cache_hits": functional.cache_hits,
            "cache_misses": functional.cache_misses,
            "elapsed_seconds": functional.elapsed_seconds,
            "evaluation_dir": str(evaluation.output_dir),
        }, indent=2))
        return
    if args.command == "run-part-i2-friedman-sweeps":
        config = load_config(args.config)
        output_dir = Path(args.out_dir) if args.out_dir else config.artifact_root
        checkpoint_dir = _project_checkpoint_dir(args.checkpoint_dir)
        device = args.device or str(default_device())
        if args.data_index:
            data_index = Path(args.data_index)
        else:
            prepared = DefaultDatasetPreparer().prepare(config, output_dir)
            data_index = prepared.index_path
        functional = DefaultPartI2Runner(
            _model_factory(config, checkpoint_dir)
        ).run(
            config=config,
            data_index=data_index,
            output_dir=output_dir,
            device=device,
            model_ids=_comma_separated(args.models),
        )
        evaluation = DefaultFriedmanSweepPartI2Evaluator().evaluate(
            functional.output_dir, config, output_dir
        )
        print(json.dumps({
            "data_run_id": functional.data_run_id,
            "part_i2_run_id": functional.part_i2_run_id,
            "evaluation_run_id": evaluation.evaluation_run_id,
            "part_i2_files": functional.num_files,
            "part_i2_metric_rows": functional.num_metric_rows,
            "part_i2_failures": functional.num_failures,
            "cache_hits": functional.cache_hits,
            "cache_misses": functional.cache_misses,
            "elapsed_seconds": functional.elapsed_seconds,
            "evaluation_dir": str(evaluation.output_dir),
        }, indent=2))
        return
    raise AssertionError(f"unhandled command: {args.command!r}")


if __name__ == "__main__":
    main()
