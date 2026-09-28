"""Evaluate one row/feature pair against the exact-k Bayes predictor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from experiments_arch.experiment_1_controlled_bayes.estimation.bayes import (
    ExactKBayesPredictor,
    bayes_prediction_error,
    bayes_signal,
)
from experiments_arch.experiment_1_controlled_bayes.artifacts.storage import (
    PredictionRun,
)


CROSSOVER_SCHEMA_VERSION = 1
_FLOAT_DATASETS = (
    "bayes_signal",
    "row_error",
    "feature_error",
    "row_relerr",
    "feature_relerr",
    "g_k",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_reference(raw_path: str, *, relative_to: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"invalid {name}={value!r}")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"invalid {name}={value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"invalid {name}={value!r}")
    return number


def _attribute(source: h5py.File, name: str) -> Any:
    if name not in source.attrs:
        raise ValueError(f"H5 artifact is missing attribute {name!r}")
    value = source.attrs[name]
    return value.decode("utf-8") if isinstance(value, bytes) else value


@dataclass(frozen=True)
class ControlledBayesTestDataset:
    """Validated test split and prior parameters from one manifest."""

    manifest_path: Path
    manifest_sha256: str
    test_path: Path
    test_sha256: str | None
    d: int
    k: int
    signal_energy: float
    noise_std: float
    data_seed: int
    num_tasks: int

    @classmethod
    def load(cls, path: str | Path) -> "ControlledBayesTestDataset":
        manifest_path = Path(path).expanduser().resolve()
        try:
            content = manifest_path.read_bytes()
            payload = json.loads(content)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read dataset manifest {manifest_path}: {error}"
            ) from error
        if not isinstance(payload, Mapping) or payload.get("complete") is not True:
            raise ValueError("dataset manifest is not marked complete")

        arguments = payload.get("command_arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("dataset manifest has no command_arguments object")
        d = _positive_int(arguments.get("d"), "d")
        k = _positive_int(arguments.get("k"), "k")
        if k > d:
            raise ValueError(f"dataset manifest has k={k} greater than d={d}")
        signal_energy = _positive_float(
            arguments.get("signal_energy"), "signal_energy"
        )
        noise_std = _positive_float(arguments.get("noise_std"), "noise_std")
        seed = arguments.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"invalid data seed={seed!r}")

        splits = payload.get("splits")
        if not isinstance(splits, Mapping) or not isinstance(
            splits.get("test"), Mapping
        ):
            raise ValueError("dataset manifest has no test split")
        test = splits["test"]
        raw_test_path = test.get("path")
        if not isinstance(raw_test_path, str) or not raw_test_path:
            raise ValueError("dataset manifest test split has no path")
        test_path = _resolve_reference(
            raw_test_path,
            relative_to=manifest_path.parent,
        )
        if not test_path.is_file():
            raise ValueError(f"dataset test split does not exist: {test_path}")
        num_tasks = _positive_int(test.get("rows"), "test rows")
        test_sha256 = test.get("sha256")
        if test_sha256 is not None and (
            not isinstance(test_sha256, str) or not test_sha256
        ):
            raise ValueError("dataset manifest has an invalid test sha256")

        cls._inspect_test_h5(
            test_path,
            d=d,
            k=k,
            signal_energy=signal_energy,
            noise_std=noise_std,
            data_seed=int(seed),
            num_tasks=num_tasks,
        )
        return cls(
            manifest_path=manifest_path,
            manifest_sha256=hashlib.sha256(content).hexdigest(),
            test_path=test_path,
            test_sha256=test_sha256,
            d=d,
            k=k,
            signal_energy=signal_energy,
            noise_std=noise_std,
            data_seed=int(seed),
            num_tasks=num_tasks,
        )

    @staticmethod
    def _inspect_test_h5(
        path: Path,
        *,
        d: int,
        k: int,
        signal_energy: float,
        noise_std: float,
        data_seed: int,
        num_tasks: int,
    ) -> None:
        required = ("X", "y", "task_id", "single_eval_pos", "num_datapoints")
        try:
            with h5py.File(path, "r") as source:
                missing = [name for name in required if name not in source]
                if missing:
                    raise ValueError(f"test H5 is missing datasets: {missing}")
                expected_attributes = {
                    "d": d,
                    "k": k,
                    "expected_signal_energy": signal_energy,
                    "noise_std": noise_std,
                    "data_seed": data_seed,
                }
                for name, expected in expected_attributes.items():
                    actual = _attribute(source, name)
                    if isinstance(expected, float):
                        matches = math.isclose(
                            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                        )
                    else:
                        matches = actual == expected
                    if not matches:
                        raise ValueError(
                            f"test H5 attribute {name!r} differs from the manifest"
                        )
                if source["X"].ndim != 3 or source["X"].shape[2] != d:
                    raise ValueError(f"test H5 X does not have feature width d={d}")
                if source["y"].shape != source["X"].shape[:2]:
                    raise ValueError("test H5 X/y shapes are inconsistent")
                if source["X"].shape[0] != num_tasks:
                    raise ValueError("test H5 row count differs from the manifest")
                for name in required[2:]:
                    if source[name].shape != (num_tasks,):
                        raise ValueError(f"test H5 dataset {name!r} has wrong shape")
                n_context = np.asarray(source["single_eval_pos"][:], dtype=np.int64)
                n_datapoints = np.asarray(
                    source["num_datapoints"][:], dtype=np.int64
                )
                if np.any(n_context < 1):
                    raise ValueError("test H5 contains an empty context")
                if np.any(n_datapoints > source["X"].shape[1]):
                    raise ValueError("test H5 num_datapoints exceeds its array width")
                if np.any(n_datapoints <= n_context):
                    raise ValueError("test H5 contains a task without queries")
        except OSError as error:
            raise ValueError(f"cannot inspect test H5 {path}: {error}") from error


@dataclass(frozen=True)
class BaselineIndex:
    """Maps dataset task_id to the baseline evaluation row in predictions.h5."""

    task_index: dict[int, int]
    n_eval: np.ndarray
    baseline_ids: np.ndarray

    @classmethod
    def read(cls, source: h5py.File) -> "BaselineIndex":
        required = ("tasks/task_id", "tasks/n_eval", "tasks/baseline_state_id",
                    "states/layer", "states/source_token", "states/epsilon",
                    "states/f_eval", "states/evaluation_status")
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"prediction H5 is missing datasets: {missing}")
        task_ids = np.asarray(source["tasks/task_id"][:], dtype=np.int64)
        n_eval = np.asarray(source["tasks/n_eval"][:], dtype=np.int64)
        baseline_ids = np.asarray(source["tasks/baseline_state_id"][:], dtype=np.int64)
        if task_ids.ndim != 1 or task_ids.size == 0:
            raise ValueError("prediction H5 has no tasks")
        if n_eval.shape != task_ids.shape or baseline_ids.shape != task_ids.shape:
            raise ValueError("prediction task tables have inconsistent shapes")
        if len(np.unique(task_ids)) != task_ids.size:
            raise ValueError("prediction H5 contains duplicate task_id")
        layer = np.asarray(source["states/layer"][:], dtype=np.int64)
        source_token = np.asarray(source["states/source_token"][:], dtype=np.int64)
        epsilon = np.asarray(source["states/epsilon"][:], dtype=np.float64)
        status = source["states/evaluation_status"].asstr()[:]
        f_eval = source["states/f_eval"]
        if f_eval.ndim != 2 or f_eval.shape[0] != layer.size:
            raise ValueError("prediction H5 f_eval has the wrong shape")
        for i, baseline in enumerate(baseline_ids):
            row = int(baseline)
            if not 0 <= row < layer.size:
                raise ValueError("baseline_state_id is out of range")
            if layer[row] != -1 or source_token[row] != -1 or float(epsilon[row]) != 0.0:
                raise ValueError("baseline_state_id does not point to a baseline gate")
            if status[row] != "ok":
                raise ValueError("baseline evaluation is not marked ok")
            width = int(n_eval[i])
            if width < 1 or width > f_eval.shape[1]:
                raise ValueError("baseline n_eval exceeds f_eval width")
        return cls(
            task_index={int(task_id): i for i, task_id in enumerate(task_ids)},
            n_eval=n_eval,
            baseline_ids=baseline_ids,
        )

    def f_eval(
        self,
        source: h5py.File,
        task_id: int,
        n_queries: int,
    ) -> np.ndarray:
        index = self.task_index.get(int(task_id))
        if index is None:
            raise ValueError(f"prediction run is missing task_id={int(task_id)}")
        if int(self.n_eval[index]) != int(n_queries):
            raise ValueError(
                f"prediction n_eval for task_id={int(task_id)} does not match test H5"
            )
        row = int(self.baseline_ids[index])
        values = np.asarray(source["states/f_eval"][row, :n_queries], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"baseline f_eval for task_id={int(task_id)} contains non-finite values"
            )
        return values


@dataclass(frozen=True)
class SavedPredictionRun:
    """One complete prediction run.json and its predictions.h5."""

    run: PredictionRun
    architecture: str
    training_seed: int
    data_sha256: str
    manifest_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "SavedPredictionRun":
        run = PredictionRun(path)
        model = run.config.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("prediction run has no model object")
        architecture = model.get("architecture")
        if architecture not in {"1d", "2d"}:
            raise ValueError("prediction run has no architecture")
        training_seed = model.get("training_seed")
        if not isinstance(training_seed, int) or isinstance(training_seed, bool):
            raise ValueError("prediction run has no integer training seed")
        inputs = run.config.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("prediction run has no inputs object")
        data = inputs.get("data")
        manifest = inputs.get("manifest")
        if not isinstance(data, Mapping) or not isinstance(manifest, Mapping):
            raise ValueError("prediction run is missing data or manifest references")
        data_hash = data.get("sha256")
        manifest_hash = manifest.get("sha256")
        if not isinstance(data_hash, str) or not data_hash:
            raise ValueError("prediction run has no data sha256")
        if not isinstance(manifest_hash, str) or not manifest_hash:
            raise ValueError("prediction run has no manifest sha256")
        run.input("data")
        run.input("manifest")
        with run.prediction() as source:
            BaselineIndex.read(source)
        return cls(
            run=run,
            architecture=str(architecture),
            training_seed=int(training_seed),
            data_sha256=data_hash,
            manifest_sha256=manifest_hash,
        )


@dataclass(frozen=True)
class CrossoverEvaluationConfig:
    """Explicit inputs for one paired crossover evaluation."""

    dataset_manifest: Path
    row_run: Path
    feature_run: Path
    output_dir: Path
    batch_size: int = 256

    def __post_init__(self) -> None:
        for name in ("dataset_manifest", "row_run", "feature_run", "output_dir"):
            object.__setattr__(
                self,
                name,
                Path(getattr(self, name)).expanduser().resolve(),
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


class TaskRelErrWriter:
    """Write one fixed-array row per test task."""

    def __init__(
        self,
        path: Path,
        *,
        num_tasks: int,
        batch_size: int,
        attributes: Mapping[str, Any],
    ) -> None:
        self.num_tasks = num_tasks
        self._next_row = 0
        self._file = h5py.File(path, "x")
        for name, value in attributes.items():
            if value is not None:
                self._file.attrs[name] = value
        self._file.attrs["schema_version"] = CROSSOVER_SCHEMA_VERSION
        self._file.attrs["complete"] = False
        chunk_rows = min(batch_size, num_tasks)
        self._datasets = {
            "task_id": self._file.create_dataset(
                "task_id", shape=(num_tasks,), dtype=np.int64, chunks=(chunk_rows,)
            ),
            "n_context": self._file.create_dataset(
                "n_context", shape=(num_tasks,), dtype=np.int32, chunks=(chunk_rows,)
            ),
            "n_queries": self._file.create_dataset(
                "n_queries", shape=(num_tasks,), dtype=np.int32, chunks=(chunk_rows,)
            ),
        }
        for name in _FLOAT_DATASETS:
            self._datasets[name] = self._file.create_dataset(
                name, shape=(num_tasks,), dtype=np.float64, chunks=(chunk_rows,)
            )

    def __enter__(self) -> "TaskRelErrWriter":
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.close()
        elif self._file is not None:
            self._file.close()
            self._file = None

    def write(self, values: Mapping[str, np.ndarray]) -> None:
        if self._file is None:
            raise RuntimeError("task RelErr writer is already closed")
        if set(values) != set(self._datasets):
            raise ValueError("task RelErr batch has an unexpected dataset set")
        batch_size = len(values["task_id"])
        stop = self._next_row + batch_size
        if stop > self.num_tasks:
            raise RuntimeError("task RelErr output contains too many rows")
        rows = slice(self._next_row, stop)
        for name, dataset in self._datasets.items():
            array = np.asarray(values[name])
            if array.shape != (batch_size,):
                raise ValueError(f"task RelErr batch {name!r} has wrong shape")
            dataset[rows] = array
        self._next_row = stop

    def close(self) -> None:
        if self._file is None:
            return
        if self._next_row != self.num_tasks:
            self._file.close()
            self._file = None
            raise RuntimeError(
                f"wrote {self._next_row} task rows, expected {self.num_tasks}"
            )
        self._file.attrs["complete"] = True
        self._file.flush()
        self._file.close()
        self._file = None


@dataclass(frozen=True)
class TaskRelErrArtifact:
    """Validated in-memory view used by the crossover aggregator."""

    path: Path
    artifact_sha256: str
    d: int
    k: int
    signal_energy: float
    noise_std: float
    data_seed: int
    training_seed: int
    dataset_manifest_sha256: str
    task_id: np.ndarray
    row_relerr: np.ndarray
    feature_relerr: np.ndarray
    g_k: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "TaskRelErrArtifact":
        artifact_path = Path(path).expanduser().resolve()
        required = (
            "task_id",
            "n_context",
            "n_queries",
            *_FLOAT_DATASETS,
        )
        try:
            with h5py.File(artifact_path, "r") as source:
                if bool(_attribute(source, "complete")) is not True:
                    raise ValueError("task RelErr artifact is not marked complete")
                if int(_attribute(source, "schema_version")) != CROSSOVER_SCHEMA_VERSION:
                    raise ValueError("task RelErr artifact has wrong schema version")
                num_tasks = int(_attribute(source, "num_tasks"))
                missing = [name for name in required if name not in source]
                if missing:
                    raise ValueError(f"task RelErr artifact is missing: {missing}")
                for name in required:
                    if source[name].shape != (num_tasks,):
                        raise ValueError(f"task RelErr dataset {name!r} has wrong shape")

                task_id = np.asarray(source["task_id"][:], dtype=np.int64)
                n_context = np.asarray(source["n_context"][:], dtype=np.int64)
                n_queries = np.asarray(source["n_queries"][:], dtype=np.int64)
                values = {
                    name: np.asarray(source[name][:], dtype=np.float64)
                    for name in _FLOAT_DATASETS
                }
                if len(np.unique(task_id)) != num_tasks:
                    raise ValueError("task RelErr artifact contains duplicate task_id")
                if np.any(n_context < 1) or np.any(n_queries < 1):
                    raise ValueError("task RelErr artifact has invalid task lengths")
                if any(not np.isfinite(value).all() for value in values.values()):
                    raise ValueError("task RelErr artifact contains non-finite values")
                if np.any(values["bayes_signal"] <= 0.0):
                    raise ValueError("task RelErr artifact has non-positive Bayes signal")
                if np.any(values["row_error"] < 0.0) or np.any(
                    values["feature_error"] < 0.0
                ):
                    raise ValueError("task RelErr artifact has negative squared error")
                if not np.allclose(
                    values["row_relerr"],
                    values["row_error"] / values["bayes_signal"],
                ) or not np.allclose(
                    values["feature_relerr"],
                    values["feature_error"] / values["bayes_signal"],
                ):
                    raise ValueError("task RelErr values do not match error / signal")
                if not np.allclose(
                    values["g_k"],
                    values["row_relerr"] - values["feature_relerr"],
                ):
                    raise ValueError("task g_k does not match the paired RelErr difference")

                d = int(_attribute(source, "d"))
                k = int(_attribute(source, "k"))
                signal_energy = float(_attribute(source, "signal_energy"))
                noise_std = float(_attribute(source, "noise_std"))
                data_seed = int(_attribute(source, "data_seed"))
                training_seed = int(_attribute(source, "training_seed"))
                dataset_manifest_sha256 = str(
                    _attribute(source, "dataset_manifest_sha256")
                )
        except OSError as error:
            raise ValueError(
                f"cannot read task RelErr artifact {artifact_path}: {error}"
            ) from error
        if not 1 <= k <= d:
            raise ValueError(f"task RelErr artifact has invalid d={d}, k={k}")
        return cls(
            path=artifact_path,
            artifact_sha256=sha256(artifact_path),
            d=d,
            k=k,
            signal_energy=signal_energy,
            noise_std=noise_std,
            data_seed=data_seed,
            training_seed=training_seed,
            dataset_manifest_sha256=dataset_manifest_sha256,
            task_id=task_id,
            row_relerr=values["row_relerr"],
            feature_relerr=values["feature_relerr"],
            g_k=values["g_k"],
        )

    @property
    def num_tasks(self) -> int:
        return int(self.task_id.size)

    @property
    def row_task_relerr(self) -> float:
        return float(np.mean(self.row_relerr))

    @property
    def feature_task_relerr(self) -> float:
        return float(np.mean(self.feature_relerr))

    @property
    def architecture_gap(self) -> float:
        return float(np.mean(self.g_k))


class CrossoverEvaluator:
    """Compute task-level RelErr for one matched row/feature pair."""

    def __init__(self, config: CrossoverEvaluationConfig) -> None:
        self.config = config
        self.dataset = ControlledBayesTestDataset.load(config.dataset_manifest)
        self.row = SavedPredictionRun.load(config.row_run)
        self.feature = SavedPredictionRun.load(config.feature_run)
        self._validate_inputs()
        self.predictor = ExactKBayesPredictor(
            d=self.dataset.d,
            k=self.dataset.k,
            signal_energy=self.dataset.signal_energy,
            noise_std=self.dataset.noise_std,
        )

    def _validate_inputs(self) -> None:
        if self.row.architecture != "1d":
            raise ValueError("row prediction run must have architecture='1d'")
        if self.feature.architecture != "2d":
            raise ValueError("feature prediction run must have architecture='2d'")
        test_hash = sha256(self.dataset.test_path)
        if (
            self.dataset.test_sha256 is not None
            and self.dataset.test_sha256 != test_hash
        ):
            raise ValueError("dataset manifest test sha256 does not match test H5")
        for artifact in (self.row, self.feature):
            if artifact.manifest_sha256 != self.dataset.manifest_sha256:
                raise ValueError("prediction run uses a different dataset manifest")
            if artifact.data_sha256 != test_hash:
                raise ValueError("prediction run does not point at the manifest test split")
        if self.row.training_seed != self.feature.training_seed:
            raise ValueError("row and feature runs must use the same training seed")

        with (
            h5py.File(self.dataset.test_path, "r") as dataset,
            self.row.run.prediction() as row,
            self.feature.run.prediction() as feature,
        ):
            expected_task_id = np.asarray(dataset["task_id"][:], dtype=np.int64)
            expected_queries = (
                np.asarray(dataset["num_datapoints"][:], dtype=np.int64)
                - np.asarray(dataset["single_eval_pos"][:], dtype=np.int64)
            )
            if expected_task_id.size != self.dataset.num_tasks:
                raise ValueError("test H5 task count differs from the manifest")
            for name, source in (("row", row), ("feature", feature)):
                index = BaselineIndex.read(source)
                if set(index.task_index) != {int(v) for v in expected_task_id}:
                    raise ValueError(f"{name} prediction task_id does not match test H5")
                for task_id, n_queries in zip(
                    expected_task_id, expected_queries, strict=True
                ):
                    row_index = index.task_index[int(task_id)]
                    if int(index.n_eval[row_index]) != int(n_queries):
                        raise ValueError(
                            f"{name} prediction n_eval for task_id={int(task_id)} "
                            "does not match test H5"
                        )

    def evaluate(self) -> Path:
        output_dir = self.config.output_dir
        if output_dir.exists():
            raise FileExistsError(f"crossover output already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.",
                suffix=".partial",
                dir=output_dir.parent,
            )
        )
        temporary_path = temporary_dir / "task_relerr.h5"
        try:
            with (
                h5py.File(self.dataset.test_path, "r") as dataset,
                self.row.run.prediction() as row,
                self.feature.run.prediction() as feature,
                TaskRelErrWriter(
                    temporary_path,
                    num_tasks=self.dataset.num_tasks,
                    batch_size=self.config.batch_size,
                    attributes=self._output_attributes(),
                ) as writer,
            ):
                row_index = BaselineIndex.read(row)
                feature_index = BaselineIndex.read(feature)
                for start in range(0, self.dataset.num_tasks, self.config.batch_size):
                    stop = min(start + self.config.batch_size, self.dataset.num_tasks)
                    writer.write(
                        self._evaluate_batch(
                            dataset,
                            row,
                            feature,
                            row_index,
                            feature_index,
                            start=start,
                            stop=stop,
                        )
                    )
            TaskRelErrArtifact.load(temporary_path)
            os.rename(temporary_dir, output_dir)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

        result = output_dir / "task_relerr.h5"
        print(f"completed crossover evaluation: {result}")
        return result

    def _evaluate_batch(
        self,
        dataset: h5py.File,
        row: h5py.File,
        feature: h5py.File,
        row_index: BaselineIndex,
        feature_index: BaselineIndex,
        *,
        start: int,
        stop: int,
    ) -> dict[str, np.ndarray]:
        x = np.asarray(dataset["X"][start:stop], dtype=np.float64)
        y = np.asarray(dataset["y"][start:stop], dtype=np.float64)
        task_id = np.asarray(dataset["task_id"][start:stop], dtype=np.int64)
        n_context = np.asarray(
            dataset["single_eval_pos"][start:stop], dtype=np.int32
        )
        n_datapoints = np.asarray(
            dataset["num_datapoints"][start:stop], dtype=np.int32
        )
        n_queries = n_datapoints - n_context

        size = stop - start
        values = {
            "task_id": task_id,
            "n_context": n_context,
            "n_queries": n_queries,
            **{name: np.empty(size, dtype=np.float64) for name in _FLOAT_DATASETS},
        }
        for local_index in range(size):
            context_count = int(n_context[local_index])
            query_count = int(n_queries[local_index])
            total_count = int(n_datapoints[local_index])
            coefficients, bayes_prediction = self.predictor.predict(
                x[local_index, :context_count],
                y[local_index, :context_count],
                x[local_index, context_count:total_count],
            )
            current_task = int(task_id[local_index])
            row_values = row_index.f_eval(row, current_task, query_count)
            feature_values = feature_index.f_eval(feature, current_task, query_count)
            signal = bayes_signal(coefficients)
            if not math.isfinite(signal) or signal <= 0.0:
                raise ValueError(
                    f"task {current_task} has non-positive Bayes signal"
                )
            row_error = bayes_prediction_error(row_values, bayes_prediction)
            feature_error = bayes_prediction_error(feature_values, bayes_prediction)
            row_relerr = row_error / signal
            feature_relerr = feature_error / signal
            values["bayes_signal"][local_index] = signal
            values["row_error"][local_index] = row_error
            values["feature_error"][local_index] = feature_error
            values["row_relerr"][local_index] = row_relerr
            values["feature_relerr"][local_index] = feature_relerr
            values["g_k"][local_index] = row_relerr - feature_relerr
        return values

    def _output_attributes(self) -> dict[str, Any]:
        return {
            "d": self.dataset.d,
            "k": self.dataset.k,
            "signal_energy": self.dataset.signal_energy,
            "noise_std": self.dataset.noise_std,
            "data_seed": self.dataset.data_seed,
            "training_seed": self.row.training_seed,
            "num_tasks": self.dataset.num_tasks,
            "dataset_manifest": str(self.dataset.manifest_path),
            "dataset_manifest_sha256": self.dataset.manifest_sha256,
            "test_path": str(self.dataset.test_path),
            "test_split_sha256": self.dataset.test_sha256,
            "row_run": str(self.row.run.path),
            "row_run_sha256": self.row.run.sha256,
            "row_predictions": str(self.row.run.predictions),
            "row_predictions_sha256": self.row.run.prediction_hash,
            "feature_run": str(self.feature.run.path),
            "feature_run_sha256": self.feature.run.sha256,
            "feature_predictions": str(self.feature.run.predictions),
            "feature_predictions_sha256": self.feature.run.prediction_hash,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--row-run", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def config_from_args(args: argparse.Namespace) -> CrossoverEvaluationConfig:
    return CrossoverEvaluationConfig(
        dataset_manifest=args.dataset_manifest,
        row_run=args.row_run,
        feature_run=args.feature_run,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    print(CrossoverEvaluator(config).evaluate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
