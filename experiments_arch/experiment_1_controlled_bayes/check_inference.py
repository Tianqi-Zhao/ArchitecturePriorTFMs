"""Check a frozen model on a few tasks before running material collection."""

import argparse
import json
from pathlib import Path

from .inference.checks import check_task
from .artifacts.data import DatasetManifest, ProjectionQueries
from .artifacts.protocol import Protocol, ROW_SCOPES, INTERVENTION_KINDS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="one dataset H5 to check")
    parser.add_argument("--projection-queries", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("nano", "tabpfn"), default="nano")
    parser.add_argument("--tabpfn-version", choices=("v2", "v2.5", "v3"), default="v2.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--num-tasks", type=int, default=2,
                        help="check the first N tasks in --data (default: 2)")
    parser.add_argument("--layers", type=int, nargs="*", default=[])
    parser.add_argument("--source-tokens", type=int, nargs="+", default=None,
                        help="source tokens to check in each layer (default: 0)")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--intervention-kind", choices=INTERVENTION_KINDS, default="source")
    parser.add_argument("--row-scope", choices=ROW_SCOPES, default="both",
                        help="rows receiving interventions; context includes thinking rows")
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--gradients", action="store_true",
                        help="also check projection input gradients and their forward predictions")
    args = parser.parse_args(argv)
    if args.num_tasks < 1:
        parser.error("--num-tasks must be positive")
    if args.source_tokens is None:
        args.source_tokens = [0] if args.intervention_kind == "source" else []
    protocol = Protocol.from_args(args)
    dataset = DatasetManifest(args.manifest).dataset(args.data)
    queries = ProjectionQueries(args.projection_queries, dataset)
    from .inference.backends import load_adapter

    adapter = load_adapter(args.backend, args.checkpoint, d=dataset.prior.d, device=args.device,
                           dtype=args.dtype, tabpfn_version=args.tabpfn_version)
    results = []
    for task_id in dataset.task_ids[:args.num_tasks]:
        task = dataset.read(int(task_id))
        results.append(check_task(adapter, task, queries.read(task.task_id), protocol,
                                  gradients=args.gradients))
    print(json.dumps({"status": "ok", "tasks": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
