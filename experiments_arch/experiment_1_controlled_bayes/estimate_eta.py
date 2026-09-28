"""Estimate one eta method from saved predictions and gradients into a new directory."""

import argparse
from pathlib import Path

from .estimation.saved import SavedEstimator
from .artifacts.protocol import METHOD_NAMES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--method", choices=METHOD_NAMES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(SavedEstimator(args.run, method=args.method,
                         output_dir=args.output_dir).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
