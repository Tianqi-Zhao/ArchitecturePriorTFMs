"""Save raw projection and evaluation predictions.

Mean projection input gradients are optional: pass ``--gradients`` only when a
later ``estimate_eta --method gradient`` run will need them.
"""

import argparse
from pathlib import Path

from .inference.collector import PredictionCollector
from .artifacts.data import DatasetManifest
from .artifacts.protocol import Protocol, ROW_SCOPES, INTERVENTION_KINDS


def parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="one dataset H5 to predict")
    parser.add_argument("--projection-queries", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("nano", "tabpfn"), default="nano")
    parser.add_argument("--tabpfn-version", choices=("v2", "v2.5", "v3"), default="v2.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--layers", type=int, nargs="*", default=[])
    parser.add_argument("--source-tokens", type=int, nargs="*", default=[])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--intervention-kind", choices=INTERVENTION_KINDS, default="source")
    parser.add_argument("--row-scope", choices=ROW_SCOPES, default="both",
                        help="rows receiving interventions; context includes thinking rows")
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--task-batch-size", type=int, default=32,
                        help="maximum task/state instances per Nano 1d/2d or TabPFN v2/v2.5/v3 forward; "
                             "1 uses serial collection; other backends only batch input reads")
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--gradients", action="store_true",
                        help="also write mean projection input gradients "
                             "(required later for estimate_eta --method gradient)")
    return parser


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    from .inference.backends import load_adapter

    try:
        prior = DatasetManifest(args.manifest).prior
        adapter = load_adapter(args.backend, args.checkpoint, d=prior.d, device=args.device,
                               dtype=args.dtype, tabpfn_version=args.tabpfn_version)
        protocol = Protocol.from_args(args)
        run = PredictionCollector(
            manifest=args.manifest, data=args.data, projection_queries=args.projection_queries,
            output_dir=args.output_dir, adapter=adapter, protocol=protocol,
            capture_gradients=args.gradients).run()
    except (ValueError, FileExistsError) as error:
        arguments.error(str(error))
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
