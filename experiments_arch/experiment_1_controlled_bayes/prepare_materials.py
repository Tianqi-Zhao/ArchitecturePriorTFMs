"""Reconstruct a fixed task cohort or prepare projection queries."""

import argparse
from pathlib import Path

from .prepare import CohortPreparer, QueryPreparer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    queries = commands.add_parser("queries")
    queries.add_argument("--manifest", type=Path, required=True)
    queries.add_argument("--output-dir", type=Path, required=True)
    queries.add_argument("--seed", type=int, default=0)
    queries.add_argument("--count", type=int, default=1000)
    queries.add_argument("--data", type=Path,
                         help="dataset whose task IDs receive queries (default: manifest test split)")
    cohort = commands.add_parser("cohort", help="replay test generation and retain a fixed subset")
    cohort.add_argument("--manifest", type=Path, required=True)
    cohort.add_argument("--output-dir", type=Path, required=True)
    cohort.add_argument("--seed", type=int, default=0, help="task-selection seed, not data-generation seed")
    cohort.add_argument("--count", type=int, default=128)
    args = parser.parse_args(argv)
    if args.command == "cohort":
        print(CohortPreparer(args.manifest, count=args.count, seed=args.seed).run(args.output_dir))
        return 0
    preparer = QueryPreparer(args.manifest, data=args.data, count=args.count, seed=args.seed)
    print(preparer.run(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
