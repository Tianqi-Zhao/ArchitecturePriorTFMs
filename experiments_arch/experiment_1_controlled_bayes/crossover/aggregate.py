"""Aggregate task-level crossover artifacts across tasks and training seeds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from experiments_arch.experiment_1_controlled_bayes.crossover.evaluate import (
    CROSSOVER_SCHEMA_VERSION,
    TaskRelErrArtifact,
)


SUMMARY_COLUMNS = (
    "d",
    "k",
    "k_over_d",
    "signal_energy",
    "noise_std",
    "data_seed",
    "num_tasks",
    "num_seeds",
    "training_seeds",
    "row_task_relerr_mean",
    "row_task_relerr_std",
    "feature_task_relerr_mean",
    "feature_task_relerr_std",
    "g_k_mean",
    "g_k_std",
)


@dataclass(frozen=True)
class CrossoverAggregationConfig:
    """Explicit task-level artifacts and one new output directory."""

    results: tuple[Path, ...]
    output_dir: Path

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("at least one task_relerr.h5 is required")
        object.__setattr__(
            self,
            "results",
            tuple(Path(path).expanduser().resolve() for path in self.results),
        )
        object.__setattr__(
            self,
            "output_dir",
            Path(self.output_dir).expanduser().resolve(),
        )


class CrossoverAggregator:
    """Apply task averaging first, then equal-weight training-seed averaging."""

    def __init__(self, config: CrossoverAggregationConfig) -> None:
        self.config = config
        self.artifacts = tuple(
            TaskRelErrArtifact.load(path) for path in config.results
        )
        self._groups = self._validate_and_group()

    def aggregate(self) -> tuple[Path, Path]:
        output_dir = self.config.output_dir
        if output_dir.exists():
            raise FileExistsError(f"crossover summary already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.",
                suffix=".partial",
                dir=output_dir.parent,
            )
        )
        try:
            conditions = [
                self._aggregate_condition(d, k, artifacts)
                for (d, k), artifacts in sorted(self._groups.items())
            ]
            summary = {
                "schema_version": CROSSOVER_SCHEMA_VERSION,
                "complete": True,
                "signal_energy": self.artifacts[0].signal_energy,
                "noise_std": self.artifacts[0].noise_std,
                "data_seed": self.artifacts[0].data_seed,
                "seed_std_definition": "population standard deviation (ddof=0)",
                "inputs": [
                    {
                        "path": str(artifact.path),
                        "sha256": artifact.artifact_sha256,
                    }
                    for artifact in self.artifacts
                ],
                "conditions": conditions,
            }
            json_path = temporary_dir / "summary.json"
            json_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            csv_path = temporary_dir / "summary.csv"
            with csv_path.open("x", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=SUMMARY_COLUMNS)
                writer.writeheader()
                for condition in conditions:
                    writer.writerow(
                        {
                            name: self._csv_value(condition[name])
                            for name in SUMMARY_COLUMNS
                        }
                    )
            os.rename(temporary_dir, output_dir)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

        json_path = output_dir / "summary.json"
        csv_path = output_dir / "summary.csv"
        print(f"completed crossover aggregation: {json_path}")
        return json_path, csv_path

    def _validate_and_group(
        self,
    ) -> dict[tuple[int, int], tuple[TaskRelErrArtifact, ...]]:
        reference = self.artifacts[0]
        groups: dict[tuple[int, int], list[TaskRelErrArtifact]] = {}
        seen: set[tuple[int, int, int]] = set()
        for artifact in self.artifacts:
            if not math.isclose(
                artifact.signal_energy,
                reference.signal_energy,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("task RelErr artifacts use different signal_energy")
            if not math.isclose(
                artifact.noise_std,
                reference.noise_std,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("task RelErr artifacts use different noise_std")
            if artifact.data_seed != reference.data_seed:
                raise ValueError("task RelErr artifacts use different data seeds")
            identity = (artifact.d, artifact.k, artifact.training_seed)
            if identity in seen:
                raise ValueError(f"duplicate crossover result for {identity}")
            seen.add(identity)
            groups.setdefault((artifact.d, artifact.k), []).append(artifact)

        expected_seeds: set[int] | None = None
        validated: dict[tuple[int, int], tuple[TaskRelErrArtifact, ...]] = {}
        for condition, items in sorted(groups.items()):
            ordered = tuple(sorted(items, key=lambda item: item.training_seed))
            first = ordered[0]
            for artifact in ordered[1:]:
                if artifact.dataset_manifest_sha256 != first.dataset_manifest_sha256:
                    raise ValueError(
                        f"condition {condition} uses different dataset manifests"
                    )
                if not np.array_equal(artifact.task_id, first.task_id):
                    raise ValueError(f"condition {condition} uses different test tasks")
            seeds = {artifact.training_seed for artifact in ordered}
            if expected_seeds is None:
                expected_seeds = seeds
            elif seeds != expected_seeds:
                raise ValueError(
                    f"condition {condition} has training seeds {sorted(seeds)}, "
                    f"expected {sorted(expected_seeds)}"
                )
            validated[condition] = ordered
        return validated

    @staticmethod
    def _aggregate_condition(
        d: int,
        k: int,
        artifacts: tuple[TaskRelErrArtifact, ...],
    ) -> dict[str, Any]:
        per_seed = []
        for artifact in artifacts:
            row = artifact.row_task_relerr
            feature = artifact.feature_task_relerr
            gap = artifact.architecture_gap
            if not math.isclose(
                gap,
                row - feature,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"condition {(d, k)}, seed {artifact.training_seed} has "
                    "an inconsistent architecture gap"
                )
            per_seed.append(
                {
                    "training_seed": artifact.training_seed,
                    "row_task_relerr": row,
                    "feature_task_relerr": feature,
                    "g_k": gap,
                    "task_relerr_path": str(artifact.path),
                    "task_relerr_sha256": artifact.artifact_sha256,
                }
            )

        row_values = np.asarray(
            [item["row_task_relerr"] for item in per_seed], dtype=np.float64
        )
        feature_values = np.asarray(
            [item["feature_task_relerr"] for item in per_seed], dtype=np.float64
        )
        gap_values = np.asarray([item["g_k"] for item in per_seed], dtype=np.float64)
        gap_mean = float(np.mean(gap_values))
        row_mean = float(np.mean(row_values))
        feature_mean = float(np.mean(feature_values))
        if not math.isclose(
            gap_mean,
            row_mean - feature_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("aggregated G_k is inconsistent with TaskRelErr means")

        seeds = [artifact.training_seed for artifact in artifacts]
        return {
            "d": d,
            "k": k,
            "k_over_d": k / d,
            "signal_energy": artifacts[0].signal_energy,
            "noise_std": artifacts[0].noise_std,
            "data_seed": artifacts[0].data_seed,
            "dataset_manifest_sha256": artifacts[0].dataset_manifest_sha256,
            "num_tasks": artifacts[0].num_tasks,
            "num_seeds": len(artifacts),
            "training_seeds": seeds,
            "row_task_relerr_mean": row_mean,
            "row_task_relerr_std": float(np.std(row_values, ddof=0)),
            "feature_task_relerr_mean": feature_mean,
            "feature_task_relerr_std": float(np.std(feature_values, ddof=0)),
            "g_k_mean": gap_mean,
            "g_k_std": float(np.std(gap_values, ddof=0)),
            "per_seed": per_seed,
        }

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def config_from_args(args: argparse.Namespace) -> CrossoverAggregationConfig:
    return CrossoverAggregationConfig(
        results=tuple(args.results),
        output_dir=args.output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    CrossoverAggregator(config).aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
