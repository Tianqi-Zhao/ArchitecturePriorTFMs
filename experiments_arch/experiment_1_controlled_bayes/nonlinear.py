"""Evaluate and summarize affine residual energies from saved artifacts."""

import argparse

from .artifacts.nonlinear_protocol import SummaryConfig


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="stage", required=True)
    evaluate = commands.add_parser("evaluate", help="write ||eta||^2 beside an existing evaluate_saved result")
    for name in ("run", "estimate", "evaluation", "output-dir"):
        evaluate.add_argument(f"--{name}", required=True)
    summarize = commands.add_parser("summarize", help="summarize task metrics and paired model differences")
    summarize.add_argument("--evaluations", nargs="+", required=True)
    summarize.add_argument("--labels", nargs="+", required=True)
    summarize.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"))
    summarize.add_argument("--bootstrap-repeats", type=int, default=2000)
    summarize.add_argument("--bootstrap-seed", type=int, default=0)
    summarize.add_argument("--confidence-level", type=float, default=0.95)
    summarize.add_argument("--output-dir", required=True)
    return root


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        if args.stage == "evaluate":
            from .evaluation.nonlinear import NonlinearEvaluator

            stage = NonlinearEvaluator(args.run, args.estimate, args.evaluation)
        else:
            from .evaluation.nonlinear_summary import NonlinearSummarizer

            stage = NonlinearSummarizer(
                args.evaluations, labels=args.labels, comparisons=args.compare,
                config=SummaryConfig(args.bootstrap_repeats, args.bootstrap_seed, args.confidence_level))
        result = stage.run(args.output_dir)
    except (ValueError, OSError) as error:
        arguments.error(str(error))
    print(result)
    if ((args.stage == "evaluate" and not stage.counts["ok"])
            or (args.stage == "summarize" and not stage.has_valid_results)):
        arguments.exit(1, "no valid tasks; diagnostic artifact saved\n")
    return result


if __name__ == "__main__":
    main()
