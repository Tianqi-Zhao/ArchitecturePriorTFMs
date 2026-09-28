"""Generate one complete controlled Bayes data condition."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

from experiments_arch.experiment_1_controlled_bayes.data_generation.episodes import (
    MAX_SEED,
    ControlledBayesEpisodeGenerator,
)
from experiments_arch.experiment_1_controlled_bayes.data_generation.prior import (
    FixedKSparseGaussianPrior,
)
from experiments_arch.experiment_1_controlled_bayes.data_generation.storage import (
    GENERATOR_NAME,
    MAX_TASK_ID,
    SCHEMA_VERSION,
    H5SplitWriter,
)


@dataclass(frozen=True)
class _SplitSpec:
    name: str
    index: int
    n_tasks: int


@dataclass(frozen=True)
class GenerationConfig:
    """Resolved arguments for one ``(d, k, data_seed)`` generation call."""

    output_dir: Path
    d: int
    k: int
    n_context_min: int
    n_context_max: int
    n_total: int
    train_tasks: int
    validation_tasks: int
    test_tasks: int
    signal_energy: float
    noise_std: float
    seed: int
    generator_batch_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        self.validate()

    def validate(self) -> None:
        _ = self.prior  # Constructing it validates d, k, and signal energy.
        if self.n_context_min < self.d:
            raise ValueError(
                "n_context_min must be >= d for a full-rank orthogonal "
                "design, got "
                f"n_context_min={self.n_context_min}, d={self.d}"
            )
        if self.n_context_max < self.n_context_min:
            raise ValueError(
                "n_context_max must be >= n_context_min, got "
                f"{self.n_context_max} < {self.n_context_min}"
            )
        if self.n_total <= self.n_context_max:
            raise ValueError(
                "n_total must be > n_context_max so every task has a query, "
                f"got n_total={self.n_total}, n_context_max={self.n_context_max}"
            )
        if self.generator_batch_size < 1:
            raise ValueError(
                "generator_batch_size must be >= 1, got "
                f"{self.generator_batch_size}"
            )
        for split in self.splits:
            if split.n_tasks < 1:
                raise ValueError(
                    f"{split.name}_tasks must be >= 1, got {split.n_tasks}"
                )
            if split.n_tasks % self.generator_batch_size != 0:
                raise ValueError(
                    f"{split.name}_tasks must be divisible by "
                    f"generator_batch_size={self.generator_batch_size}, "
                    f"got {split.n_tasks}"
                )
        if sum(split.n_tasks for split in self.splits) > MAX_TASK_ID:
            raise ValueError("total task count exceeds the int64 task_id range")
        if not math.isfinite(self.noise_std) or self.noise_std <= 0.0:
            raise ValueError(
                f"noise_std must be finite and > 0, got {self.noise_std}"
            )
        if not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}], got {self.seed}")

    @property
    def prior(self) -> FixedKSparseGaussianPrior:
        return FixedKSparseGaussianPrior(
            d=self.d,
            k=self.k,
            signal_energy=self.signal_energy,
        )

    @property
    def splits(self) -> tuple[_SplitSpec, ...]:
        return (
            _SplitSpec("train", 0, self.train_tasks),
            _SplitSpec("validation", 1, self.validation_tasks),
            _SplitSpec("test", 2, self.test_tasks),
        )

    @property
    def noise_variance(self) -> float:
        return float(self.noise_std**2)

    def command_arguments(self) -> dict[str, Any]:
        arguments = asdict(self)
        arguments["output_dir"] = str(self.output_dir.resolve())
        return arguments


class ControlledBayesDatasetGenerator:
    """Generate, validate, and atomically publish one complete data condition."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.prior = config.prior

    def generate(self) -> Path:
        output_dir = self.config.output_dir.resolve()
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.",
                suffix=".partial",
                dir=output_dir.parent,
            )
        )

        try:
            splits = self._write_splits(staging_dir)
            self._validate_task_ranges(splits)
            self._write_json(self._manifest(splits), staging_dir / "manifest.json")
            os.rename(staging_dir, output_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
        return output_dir / "manifest.json"

    def _write_splits(self, staging_dir: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        task_id_start = 0
        for split in self.config.splits:
            episodes = ControlledBayesEpisodeGenerator(
                prior=self.prior,
                n_context_min=self.config.n_context_min,
                n_context_max=self.config.n_context_max,
                n_total=self.config.n_total,
                noise_std=self.config.noise_std,
                seed=self.config.seed,
                stream_index=split.index,
            )
            writer = H5SplitWriter(
                path=staging_dir / f"{split.name}.h5",
                split_name=split.name,
                n_tasks=split.n_tasks,
                episode_generator=episodes,
                task_id_start=task_id_start,
                generator_batch_size=self.config.generator_batch_size,
            )
            metadata[split.name] = writer.write()
            task_id_start += split.n_tasks
        return metadata

    def _manifest(self, splits: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "generator": GENERATOR_NAME,
            "command_arguments": self.config.command_arguments(),
            "distribution": {
                **self.prior.metadata(),
                "context_design": "orthogonal_gaussian_qr",
                "context_length_prior": (
                    "discrete_uniform_inclusive_per_generator_batch"
                ),
                "n_context_range": [
                    self.config.n_context_min,
                    self.config.n_context_max,
                ],
                "n_total": self.config.n_total,
                "generator_batch_size": self.config.generator_batch_size,
                "query_design": "standard_normal_iid",
                "label_noise": "gaussian_iid",
                "noise_std": self.config.noise_std,
                "noise_variance": self.config.noise_variance,
            },
            "splits": dict(splits),
        }

    @staticmethod
    def _validate_task_ranges(splits: Mapping[str, Any]) -> None:
        ranges = sorted(
            (metadata["task_id_range"] for metadata in splits.values()),
            key=lambda bounds: bounds[0],
        )
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise RuntimeError("train/validation/test task_id ranges overlap")

    @staticmethod
    def _write_json(value: Mapping[str, Any], path: Path) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(value, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one fixed-(d, k) controlled Bayes dataset."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n-context-min", type=int, required=True)
    parser.add_argument("--n-context-max", type=int, required=True)
    parser.add_argument("--n-total", type=int, required=True)
    parser.add_argument("--train-tasks", type=int, required=True)
    parser.add_argument("--validation-tasks", type=int, required=True)
    parser.add_argument("--test-tasks", type=int, required=True)
    parser.add_argument("--signal-energy", type=float, required=True)
    parser.add_argument("--noise-std", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generator-batch-size", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = ControlledBayesDatasetGenerator(
        GenerationConfig(**vars(args))
    ).generate()
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
