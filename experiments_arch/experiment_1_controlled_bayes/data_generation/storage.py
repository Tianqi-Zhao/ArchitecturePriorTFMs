"""HDF5 persistence for controlled Bayes episode streams."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from experiments_arch.experiment_1_controlled_bayes.data_generation.episodes import (
    ControlledBayesEpisodeGenerator,
)
from experiments_arch.experiment_1_controlled_bayes.data_generation.prior import (
    FLOAT_DTYPE,
    SUPPORT_DTYPE,
)


SCHEMA_VERSION = 1
GENERATOR_NAME = "fixed_k_orthogonal_linear_regression"
INTEGER_DTYPE = np.dtype("int32")
TASK_ID_DTYPE = np.dtype("int64")
MAX_TASK_ID = np.iinfo(TASK_ID_DTYPE).max
_MAX_CHUNK_BYTES = 4 * 1024 * 1024


class H5SplitWriter:
    """Write and verify one split using the shared prior-dump schema."""

    def __init__(
        self,
        *,
        path: Path,
        split_name: str,
        n_tasks: int,
        episode_generator: ControlledBayesEpisodeGenerator,
        task_id_start: int,
        generator_batch_size: int,
    ) -> None:
        if generator_batch_size < 1:
            raise ValueError("generator_batch_size must be >= 1")
        if n_tasks % generator_batch_size != 0:
            raise ValueError("n_tasks must be divisible by generator_batch_size")
        self.path = path
        self.split_name = split_name
        self.n_tasks = n_tasks
        self.episode_generator = episode_generator
        self.task_id_start = task_id_start
        self.generator_batch_size = generator_batch_size

    @property
    def chunk_rows(self) -> int:
        generator = self.episode_generator
        bytes_per_x_row = (
            generator.n_total * generator.prior.d * FLOAT_DTYPE.itemsize
        )
        size_cap = max(1, _MAX_CHUNK_BYTES // bytes_per_x_row)
        return min(self.n_tasks, size_cap)

    def write(self) -> dict[str, Any]:
        with h5py.File(self.path, "x") as output:
            self._write_attributes(output)
            datasets = self._create_datasets(output)
            for start in range(0, self.n_tasks, self.generator_batch_size):
                stop = min(start + self.generator_batch_size, self.n_tasks)
                self._write_block(datasets, start, stop)
            output.flush()
        return self._inspect()

    def _write_attributes(self, output: h5py.File) -> None:
        parameters = {
            "schema_version": SCHEMA_VERSION,
            "generator": GENERATOR_NAME,
            "split": self.split_name,
            "n_tasks": self.n_tasks,
            "generator_batch_size": self.generator_batch_size,
            **self.episode_generator.metadata(),
        }
        for name, value in parameters.items():
            if isinstance(value, (dict, list, tuple)):
                output.attrs[name] = json.dumps(value, sort_keys=True)
            else:
                output.attrs[name] = value
        output.attrs["parameters_json"] = json.dumps(parameters, sort_keys=True)

    def _create_datasets(self, output: h5py.File) -> dict[str, h5py.Dataset]:
        generator = self.episode_generator
        d = generator.prior.d
        chunks = self.chunk_rows
        datasets = {
            "X": output.create_dataset(
                "X",
                shape=(self.n_tasks, generator.n_total, d),
                dtype=FLOAT_DTYPE,
                chunks=(chunks, generator.n_total, d),
            ),
            "y": output.create_dataset(
                "y",
                shape=(self.n_tasks, generator.n_total),
                dtype=FLOAT_DTYPE,
                chunks=(chunks, generator.n_total),
            ),
            "beta": output.create_dataset(
                "beta",
                shape=(self.n_tasks, d),
                dtype=FLOAT_DTYPE,
                chunks=(chunks, d),
            ),
            "support_mask": output.create_dataset(
                "support_mask",
                shape=(self.n_tasks, d),
                dtype=SUPPORT_DTYPE,
                chunks=(chunks, d),
            ),
            "task_id": output.create_dataset(
                "task_id",
                shape=(self.n_tasks,),
                dtype=TASK_ID_DTYPE,
                chunks=(chunks,),
            ),
            "num_features": output.create_dataset(
                "num_features",
                shape=(self.n_tasks,),
                dtype=INTEGER_DTYPE,
                chunks=(chunks,),
            ),
            "num_datapoints": output.create_dataset(
                "num_datapoints",
                shape=(self.n_tasks,),
                dtype=INTEGER_DTYPE,
                chunks=(chunks,),
            ),
            "single_eval_pos": output.create_dataset(
                "single_eval_pos",
                shape=(self.n_tasks,),
                dtype=INTEGER_DTYPE,
                chunks=(chunks,),
            ),
        }
        output.create_dataset(
            "max_num_classes",
            data=np.asarray([1], dtype=INTEGER_DTYPE),
        )
        datasets["X"].attrs["context_boundary_dataset"] = "single_eval_pos"
        datasets["beta"].attrs["prior"] = (
            "N(0, signal_energy / k) on support"
        )
        datasets["support_mask"].attrs["semantics"] = (
            "1 iff coefficient is active"
        )
        datasets["task_id"].attrs["semantics"] = (
            "unique episode identifier within the generated condition"
        )
        return datasets

    def _write_block(
        self,
        datasets: Mapping[str, h5py.Dataset],
        start: int,
        stop: int,
    ) -> None:
        generator = self.episode_generator
        batch = generator.sample(stop - start)
        datasets["X"][start:stop] = batch.X
        datasets["y"][start:stop] = batch.y
        datasets["beta"][start:stop] = batch.beta
        datasets["support_mask"][start:stop] = batch.support_mask
        datasets["task_id"][start:stop] = np.arange(
            self.task_id_start + start,
            self.task_id_start + stop,
            dtype=TASK_ID_DTYPE,
        )
        datasets["num_features"][start:stop] = generator.prior.d
        datasets["num_datapoints"][start:stop] = generator.n_total
        datasets["single_eval_pos"][start:stop] = batch.single_eval_pos

    def _inspect(self) -> dict[str, Any]:
        generator = self.episode_generator
        d = generator.prior.d
        expected = {
            "X": ((self.n_tasks, generator.n_total, d), FLOAT_DTYPE),
            "y": ((self.n_tasks, generator.n_total), FLOAT_DTYPE),
            "beta": ((self.n_tasks, d), FLOAT_DTYPE),
            "support_mask": ((self.n_tasks, d), SUPPORT_DTYPE),
            "task_id": ((self.n_tasks,), TASK_ID_DTYPE),
            "num_features": ((self.n_tasks,), INTEGER_DTYPE),
            "num_datapoints": ((self.n_tasks,), INTEGER_DTYPE),
            "single_eval_pos": ((self.n_tasks,), INTEGER_DTYPE),
            "max_num_classes": ((1,), INTEGER_DTYPE),
        }
        with h5py.File(self.path, "r") as source:
            self._validate_attributes(source)
            for name, (shape, dtype) in expected.items():
                if name not in source:
                    raise RuntimeError(f"{self.path} is missing dataset {name!r}")
                dataset = source[name]
                if dataset.shape != shape or np.dtype(dataset.dtype) != dtype:
                    raise RuntimeError(
                        f"{self.path}:{name} has {dataset.shape}/{dataset.dtype}; "
                        f"expected {shape}/{dtype}"
                    )
            self._validate_rows(source)
            datasets = {
                name: {
                    "shape": [int(value) for value in source[name].shape],
                    "dtype": np.dtype(source[name].dtype).name,
                }
                for name in sorted(expected)
            }
        return {
            "path": self.path.name,
            "rows": self.n_tasks,
            "task_id_range": [
                self.task_id_start,
                self.task_id_start + self.n_tasks,
            ],
            "datasets": datasets,
            "sha256": self._sha256(),
        }

    def _validate_attributes(self, source: h5py.File) -> None:
        prior = self.episode_generator.prior
        if int(source.attrs["schema_version"]) != SCHEMA_VERSION:
            raise RuntimeError(f"{self.path} has the wrong schema_version")
        if source.attrs["split"] != self.split_name:
            raise RuntimeError(f"{self.path} has the wrong split metadata")
        if int(source.attrs["d"]) != prior.d or int(source.attrs["k"]) != prior.k:
            raise RuntimeError(f"{self.path} has inconsistent d/k metadata")

    def _validate_rows(self, source: h5py.File) -> None:
        generator = self.episode_generator
        for start in range(0, self.n_tasks, self.generator_batch_size):
            stop = min(start + self.generator_batch_size, self.n_tasks)
            if not np.all(source["num_features"][start:stop] == generator.prior.d):
                raise RuntimeError(f"{self.path} has inconsistent num_features")
            if not np.all(source["num_datapoints"][start:stop] == generator.n_total):
                raise RuntimeError(f"{self.path} has inconsistent num_datapoints")
            boundaries = source["single_eval_pos"][start:stop]
            if not np.all(boundaries == boundaries[0]):
                raise RuntimeError(
                    f"{self.path} contains mixed single_eval_pos in one batch"
                )
            if np.any(boundaries < generator.n_context_min) or np.any(
                boundaries > generator.n_context_max
            ):
                raise RuntimeError(
                    f"{self.path} contains a context boundary outside its range"
                )
            expected_ids = np.arange(
                self.task_id_start + start,
                self.task_id_start + stop,
                dtype=TASK_ID_DTYPE,
            )
            if not np.array_equal(source["task_id"][start:stop], expected_ids):
                raise RuntimeError(f"{self.path} has inconsistent task_id values")
            support = source["support_mask"][start:stop]
            if not np.all(support.sum(axis=1) == generator.prior.k):
                raise RuntimeError(f"{self.path} contains a support whose size is not k")
            beta = source["beta"][start:stop]
            if np.any(beta[support == 0] != 0.0):
                raise RuntimeError(f"{self.path} contains a nonzero inactive coefficient")
        if int(source["max_num_classes"][0]) != 1:
            raise RuntimeError(f"{self.path} has invalid regression metadata")

    def _sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
