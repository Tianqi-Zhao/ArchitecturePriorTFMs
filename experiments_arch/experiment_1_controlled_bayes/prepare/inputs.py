"""Prepare shared projection queries for test or intervention tasks."""

from pathlib import Path

import h5py
import numpy as np

from ..artifacts.data import DatasetManifest, ProjectionQueries
from ..artifacts.storage import new_directory, sha256


class QueryPreparer:
    def __init__(self, manifest, *, data=None, count=1000, seed=0):
        self.manifest = DatasetManifest(manifest)
        self.dataset = self.manifest.dataset(data) if data is not None else self.manifest.test_dataset()
        self.count, self.seed = count, seed
        if count < 1 or seed < 0:
            raise ValueError("query count must be positive and seed nonnegative")

    def run(self, output_dir):
        destination = Path(output_dir).expanduser().resolve()
        data = self.dataset
        with new_directory(destination) as directory:
            path = directory / "projection_queries.h5"
            with h5py.File(path, "x") as output:
                output.attrs.update(schema_version=1, complete=False, d=data.prior.d,
                                    k=data.prior.k, projection_seed=self.seed,
                                    num_tasks=len(data.task_ids), num_projection_queries=self.count,
                                    dataset_manifest_sha256=sha256(self.manifest.path),
                                    test_split_sha256=data.sha256)
                output.create_dataset("task_id", data=data.task_ids)
                queries = output.create_dataset(
                    "x_proj", shape=(len(data.task_ids), self.count, data.prior.d),
                    dtype=np.float32, chunks=(1, min(self.count, 256), data.prior.d))
                for row, task_id in enumerate(data.task_ids):
                    t = int(task_id)
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [self.seed & 0xFFFFFFFF, self.seed >> 32, t & 0xFFFFFFFF, t >> 32]))
                    for start in range(0, self.count, 256):
                        n = min(self.count - start, 256)
                        queries[row, start:start + n] = rng.standard_normal((n, data.prior.d))
                output.attrs["complete"] = True
            ProjectionQueries(path, data)
        return destination / "projection_queries.h5"



class CohortPreparer:
    """Replay the test stream and retain a seeded subset in selection order."""

    def __init__(self, manifest, *, count=128, seed=0):
        self.manifest = DatasetManifest(manifest)
        self.arguments = self.manifest.payload["command_arguments"]
        self.count, self.seed = count, seed
        if not 1 <= count <= self.arguments["test_tasks"] or seed < 0:
            raise ValueError("cohort count must fit the test split and seed must be nonnegative")

    def run(self, output_dir):
        from ..data_generation.episodes import ControlledBayesEpisodeGenerator
        from ..data_generation.prior import FixedKSparseGaussianPrior

        a = self.arguments
        batch_size = a["generator_batch_size"]
        selected = np.random.default_rng(self.seed).permutation(a["test_tasks"])[:self.count]
        slots = {int(row): slot for slot, row in enumerate(selected)}
        generator = ControlledBayesEpisodeGenerator(
            prior=FixedKSparseGaussianPrior(d=a["d"], k=a["k"], signal_energy=a["signal_energy"]),
            n_context_min=a["n_context_min"], n_context_max=a["n_context_max"],
            n_total=a["n_total"], noise_std=a["noise_std"], seed=a["seed"], stream_index=2)
        values = {
            "X": np.empty((self.count, a["n_total"], a["d"]), dtype=np.float32),
            "y": np.empty((self.count, a["n_total"]), dtype=np.float32),
            "beta": np.empty((self.count, a["d"]), dtype=np.float32),
            "support_mask": np.empty((self.count, a["d"]), dtype=np.uint8),
            "single_eval_pos": np.empty(self.count, dtype=np.int32),
        }
        # Every preceding batch must consume its random streams, even if no row is retained.
        for start in range(0, int(selected.max()) + 1, batch_size):
            batch = generator.sample(min(batch_size, a["test_tasks"] - start))
            for row in range(start, start + len(batch.X)):
                if row not in slots:
                    continue
                slot = slots[row]
                for name in ("X", "y", "beta", "support_mask"):
                    values[name][slot] = getattr(batch, name)[row - start]
                values["single_eval_pos"][slot] = batch.single_eval_pos
        values.update(
            task_id=(selected + a["train_tasks"] + a["validation_tasks"]).astype(np.int64),
            num_features=np.full(self.count, a["d"], dtype=np.int32),
            num_datapoints=np.full(self.count, a["n_total"], dtype=np.int32),
            max_num_classes=np.array([1], dtype=np.int32))
        destination = Path(output_dir).expanduser().resolve()
        with new_directory(destination) as directory:
            with h5py.File(directory / "tasks.h5", "x") as output:
                output.attrs.update(complete=False, d=a["d"], k=a["k"],
                                    noise_std=a["noise_std"], expected_signal_energy=a["signal_energy"],
                                    data_seed=a["seed"], selection_seed=self.seed,
                                    generator_batch_size=batch_size, n_tasks=self.count)
                for name, value in values.items():
                    output.create_dataset(name, data=value)
                output.attrs["complete"] = True
        return destination / "tasks.h5"
