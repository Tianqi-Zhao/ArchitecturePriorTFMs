"""Select controlled-Bayes hyperparameters from explicit evaluation results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence
from uuid import uuid4


@dataclass(frozen=True)
class EvaluationRecord:
    """Validated selection view of one ``evaluation/metrics.json`` file."""

    path: Path
    checkpoint_path: str
    checkpoint_sha256: str
    dataset_manifest: str
    dataset_manifest_sha256: str
    d: int
    k: int
    architecture: str
    model_config: Mapping[str, Any]
    optimizer: str
    learning_rate: float
    training_seed: int
    training_config: Mapping[str, Any]
    metrics: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationRecord":
        resolved = Path(path).expanduser().resolve()
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read evaluation result {resolved}: {error}") from error
        if not isinstance(payload, Mapping) or payload.get("complete") is not True:
            raise ValueError(f"evaluation result is not marked complete: {resolved}")
        if payload.get("schema_version") != 1:
            raise ValueError(f"evaluation result has unsupported schema version: {resolved}")
        if payload.get("split") != "test":
            raise ValueError(f"evaluation result must describe the test split: {resolved}")

        model_config = cls._mapping(payload, "model_config", resolved)
        training_config = cls._mapping(payload, "training_config", resolved)
        metrics = cls._mapping(payload, "metrics", resolved)
        checkpoint_path = cls._string(payload, "checkpoint_path", resolved)
        checkpoint_sha256 = cls._string(payload, "checkpoint_sha256", resolved)
        dataset_manifest = cls._string(payload, "dataset_manifest", resolved)
        dataset_sha256 = cls._string(
            payload,
            "dataset_manifest_sha256",
            resolved,
        )
        architecture = cls._string(payload, "architecture", resolved)
        optimizer = cls._string(payload, "optimizer", resolved)
        d = cls._positive_integer(payload.get("d"), "d", resolved)
        k = cls._positive_integer(payload.get("k"), "k", resolved)
        if k > d:
            raise ValueError(f"evaluation result has k={k} greater than d={d}: {resolved}")
        learning_rate = cls._positive_float(
            payload.get("learning_rate"),
            "learning_rate",
            resolved,
        )
        training_seed = payload.get("training_seed")
        if not isinstance(training_seed, int) or isinstance(training_seed, bool):
            raise ValueError(
                f"evaluation result has invalid training_seed={training_seed!r}: "
                f"{resolved}"
            )
        return cls(
            path=resolved,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            dataset_manifest=dataset_manifest,
            dataset_manifest_sha256=dataset_sha256,
            d=d,
            k=k,
            architecture=architecture,
            model_config=dict(model_config),
            optimizer=optimizer,
            learning_rate=learning_rate,
            training_seed=training_seed,
            training_config=dict(training_config),
            metrics=dict(metrics),
        )

    def metric(self, name: str) -> float:
        value = self.metrics.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"evaluation result {self.path} has no finite metric {name!r}"
            )
        return float(value)

    @property
    def fixed_training_config(self) -> dict[str, Any]:
        ignored = {"optimizer_name", "learning_rate"}
        return {
            key: value
            for key, value in self.training_config.items()
            if key not in ignored
        }

    @staticmethod
    def _mapping(
        payload: Mapping[str, Any],
        name: str,
        path: Path,
    ) -> Mapping[str, Any]:
        value = payload.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"evaluation result has no {name} object: {path}")
        return value

    @staticmethod
    def _string(payload: Mapping[str, Any], name: str, path: Path) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"evaluation result has invalid {name}={value!r}: {path}")
        return value

    @staticmethod
    def _positive_integer(value: Any, name: str, path: Path) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"evaluation result has invalid {name}={value!r}: {path}")
        return value

    @staticmethod
    def _positive_float(value: Any, name: str, path: Path) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"evaluation result has invalid {name}={value!r}: {path}")
        return float(value)


@dataclass(frozen=True, order=True)
class HyperparameterKey:
    optimizer: str
    learning_rate: float


@dataclass(frozen=True)
class SelectionConfig:
    """Explicit evaluation files and objective for one selection operation."""

    metrics: tuple[Path, ...]
    output: Path
    metric: str
    mode: str

    def __post_init__(self) -> None:
        resolved = tuple(Path(path).expanduser().resolve() for path in self.metrics)
        object.__setattr__(self, "metrics", resolved)
        object.__setattr__(self, "output", Path(self.output).expanduser().resolve())
        if not resolved:
            raise ValueError("at least one evaluation result is required")
        if len(set(resolved)) != len(resolved):
            raise ValueError("evaluation result paths must be unique")
        if not self.metric:
            raise ValueError("metric must be non-empty")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")


class HyperparameterSelector:
    """Validate, aggregate, and select one optimizer/learning-rate pair."""

    def __init__(self, config: SelectionConfig) -> None:
        self.config = config
        self.records = tuple(EvaluationRecord.load(path) for path in config.metrics)
        self._validate_comparability()

    def select(self) -> Path:
        if self.config.output.exists():
            raise FileExistsError(f"selection output already exists: {self.config.output}")
        candidates = self._aggregate()
        best = min(
            candidates,
            key=lambda candidate: (
                candidate["metric_mean"]
                if self.config.mode == "min"
                else -candidate["metric_mean"],
                candidate["learning_rate"],
                candidate["optimizer"],
            ),
        )
        first = self.records[0]
        payload = {
            "schema_version": 1,
            "complete": True,
            "metric": self.config.metric,
            "mode": self.config.mode,
            "dataset_manifest": first.dataset_manifest,
            "dataset_manifest_sha256": first.dataset_manifest_sha256,
            "d": first.d,
            "k": first.k,
            "architecture": first.architecture,
            "model_config": dict(first.model_config),
            "fixed_training_config": first.fixed_training_config,
            "training_seeds": best["training_seeds"],
            "selected_hyperparameters": {
                "architecture": first.architecture,
                "model_config": dict(first.model_config),
                "optimizer": best["optimizer"],
                "learning_rate": best["learning_rate"],
            },
            "selection_evidence": {
                "candidate_count": len(candidates),
                "run_count": len(self.records),
                "tie_break": "learning_rate_then_optimizer",
                "selected_metric_mean": best["metric_mean"],
                "selected_metric_std": best["metric_std"],
                "candidates": candidates,
            },
        }
        self._write_json(payload, self.config.output)
        print(f"selected hyperparameters: {self.config.output}")
        return self.config.output

    def _validate_comparability(self) -> None:
        first = self.records[0]
        expected = self._comparison_signature(first)
        for record in self.records:
            record.metric(self.config.metric)
            if self._comparison_signature(record) != expected:
                raise ValueError(
                    "all evaluation results must use the same dataset manifest, "
                    "data condition, architecture, model config, and fixed training "
                    f"config; mismatch: {record.path}"
                )

        grouped: dict[HyperparameterKey, set[int]] = defaultdict(set)
        for record in self.records:
            key = HyperparameterKey(record.optimizer, record.learning_rate)
            if record.training_seed in grouped[key]:
                raise ValueError(
                    "duplicate training seed for hyperparameter candidate "
                    f"{key}: seed={record.training_seed}"
                )
            grouped[key].add(record.training_seed)
        seed_sets = list(grouped.values())
        if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            detail = {
                f"{key.optimizer}:{key.learning_rate}": sorted(seeds)
                for key, seeds in grouped.items()
            }
            raise ValueError(
                "every hyperparameter candidate must have the same training seeds: "
                f"{detail}"
            )

    def _aggregate(self) -> list[dict[str, Any]]:
        grouped: dict[HyperparameterKey, list[EvaluationRecord]] = defaultdict(list)
        for record in self.records:
            grouped[
                HyperparameterKey(record.optimizer, record.learning_rate)
            ].append(record)

        candidates = []
        for key in sorted(grouped):
            records = sorted(grouped[key], key=lambda record: record.training_seed)
            values = [record.metric(self.config.metric) for record in records]
            candidates.append(
                {
                    "optimizer": key.optimizer,
                    "learning_rate": key.learning_rate,
                    "training_seeds": [record.training_seed for record in records],
                    "metric_values": values,
                    "metric_mean": float(statistics.fmean(values)),
                    "metric_std": (
                        float(statistics.stdev(values)) if len(values) > 1 else None
                    ),
                    "evaluation_files": [str(record.path) for record in records],
                    "checkpoint_paths": [
                        record.checkpoint_path for record in records
                    ],
                    "checkpoint_sha256": [
                        record.checkpoint_sha256 for record in records
                    ],
                }
            )
        return candidates

    @staticmethod
    def _comparison_signature(record: EvaluationRecord) -> tuple[Any, ...]:
        return (
            record.dataset_manifest_sha256,
            record.d,
            record.k,
            record.architecture,
            HyperparameterSelector._canonical(record.model_config),
            HyperparameterSelector._canonical(record.fixed_training_config),
        )

    @staticmethod
    def _canonical(value: Mapping[str, Any]) -> str:
        try:
            return json.dumps(value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"configuration is not valid JSON: {error}") from error

    @staticmethod
    def _write_json(value: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(
                    value,
                    output,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--mode", choices=("min", "max"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def config_from_args(args: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        metrics=tuple(args.metrics),
        output=args.output,
        metric=args.metric,
        mode=args.mode,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    output = HyperparameterSelector(config).select()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
