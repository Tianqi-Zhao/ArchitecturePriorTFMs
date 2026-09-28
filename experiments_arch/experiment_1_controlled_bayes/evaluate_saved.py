"""Evaluate saved exact-k predictions/eta without importing any model library."""

import argparse
from pathlib import Path

from .evaluation.evaluator import SavedEvaluator
from .artifacts.protocol import EvaluationConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimates", type=Path, nargs="+", required=True,
                        help="one estimate.json per method to evaluate")
    parser.add_argument("--denominator-threshold", type=float, default=1e-12)
    parser.add_argument("--projection-diagnostics", action="store_true",
                        help="also read projection queries to measure fit residuals and BLP rank/condition")
    args = parser.parse_args(argv)
    result = SavedEvaluator(
        args.run, output_dir=args.output_dir, estimates=args.estimates,
        config=EvaluationConfig(args.denominator_threshold, args.projection_diagnostics)).run()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
