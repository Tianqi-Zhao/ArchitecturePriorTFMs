"""TabDPT release-version replication: index preparation, summary, smoke test.

Part I is run for the official TabDPT v1.0/v1.1/v1.3 releases next to the
existing v1.2 results.  Every release uses its package-default inference
settings except for one departure, the full training context (retrieval off),
and each release runs in its own virtual environment because their Python APIs
differ.  Results live under ``artifacts/experiment_0_tabdpt_versions``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_core.adapters import ModelConfig
from experiment_core.pretrained import DefaultPretrainedModelFactory
from experiments_arch.experiment_0_pretrained.config import load_config
from experiments_arch.experiment_0_pretrained.data.store import INDEX_COLUMNS
from experiments_arch.experiment_0_pretrained.evaluate import (
    PAIRED_EFFECT_COLUMNS,
    PART_I2_TASK_EFFECT_COLUMNS,
    DefaultRawPredictionEvaluator,
    require_benchmark_sizes,
)

MODEL_IDS = {
    "v1.0": "tabdpt_v1_0_retrieval_off",
    "v1.1": "tabdpt_v1_1_retrieval_off",
    "v1.2": "tabdpt_retrieval_off",
    "v1.3": "tabdpt_v1_3_retrieval_off",
}
VERSIONS = ("v1.0", "v1.1", "v1.2", "v1.3")
# The replication runs live under one directory per release; v1.2 comes from
# the existing Part I evaluations instead.
VERSION_DIRS = {"v1.0": "v1_0", "v1.1": "v1_1", "v1.3": "v1_3"}
BASELINE_MODEL_ID = "tabpfn_v2"
SUMMARY_COLUMNS = (
    "version", "source", "requested_irrelevant_ratio", "n_datasets",
    "n_complete_pairs", "n_transform_repeats", "r2_drop_mean", "r2_drop_std",
    "r2_drop_median", "r2_drop_q25", "r2_drop_q75",
)


# --- cached index preparation -------------------------------------------------

def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INDEX_COLUMNS:
            raise ValueError(f"prepared index schema mismatch: {path}")
        return list(reader)


def _write_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _validate_paths(rows: list[dict[str, str]]) -> None:
    missing = sorted({
        row["data_path"] for row in rows if not Path(row["data_path"]).is_file()
    })
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} cached data files are missing:\n{preview}"
        )


def prepare_indexes(
    friedman_index: Path, part_i1_index: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Reuse the immutable Part I data caches without regenerating datasets."""

    friedman = [
        row for row in _read_index(friedman_index) if row["source"] == "synthetic"
    ]
    openml = [row for row in _read_index(part_i1_index) if row["source"] == "real"]
    if len(friedman) != 12_250:
        raise ValueError(
            f"expected 12,250 Friedman conditions, found {len(friedman)}"
        )
    if len(openml) != 1_470:
        raise ValueError(f"expected 1,470 OpenML conditions, found {len(openml)}")
    logical_friedman = {
        (row["pair_id"].split(":", 1)[0], row["dataset_id"]) for row in friedman
    }
    if len(logical_friedman) != 350:
        raise ValueError(
            f"expected 350 Friedman tasks, found {len(logical_friedman)}"
        )
    if len({row["dataset_id"] for row in openml}) != 42:
        raise ValueError("OpenML cache does not contain exactly 42 datasets")
    for label, rows in (("Friedman", friedman), ("OpenML", openml)):
        if len({row["data_run_id"] for row in rows}) != 1:
            raise ValueError(f"{label} index contains multiple data_run_id values")
        _validate_paths(rows)
    friedman_path = output_dir / "friedman350.csv"
    openml_path = output_dir / "openml42.csv"
    _write_index(friedman_path, friedman)
    _write_index(openml_path, openml)
    return friedman_path, openml_path


# --- cross-version summary ----------------------------------------------------

def _read_effects(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PAIRED_EFFECT_COLUMNS:
            raise ValueError(f"paired-effect schema mismatch: {path}")
        return list(reader)


def _logical_dataset(row: dict[str, str]) -> str:
    if row["source"] == "synthetic":
        source_id = row["pair_id"].split(":", 1)[0]
        return f"{source_id}/{row['dataset_id']}"
    return row["dataset_id"]


def _summaries(
    versioned_rows: list[tuple[str, dict[str, str]]]
) -> list[dict[str, object]]:
    dataset_groups: dict[tuple[str, str, float, str], list[dict[str, str]]] = {}
    for version, row in versioned_rows:
        key = (
            version, row["source"], float(row["requested_irrelevant_ratio"]),
            _logical_dataset(row),
        )
        dataset_groups.setdefault(key, []).append(row)

    aggregated: dict[tuple[str, str, float], list[tuple[float, int]]] = {}
    for (version, source, ratio, _), rows in dataset_groups.items():
        value = float(np.mean([float(row["r2_drop"]) for row in rows]))
        aggregated.setdefault((version, source, ratio), []).append(
            (value, len(rows))
        )

    result = []
    for (version, source, ratio), values_and_counts in sorted(aggregated.items()):
        values = np.asarray([item[0] for item in values_and_counts], dtype=np.float64)
        repeats = {
            int(row["transform_repeat"])
            for row_version, row in versioned_rows
            if row_version == version
            and row["source"] == source
            and float(row["requested_irrelevant_ratio"]) == ratio
        }
        result.append({
            "version": version,
            "source": source,
            "requested_irrelevant_ratio": ratio,
            "n_datasets": len(values),
            "n_complete_pairs": sum(item[1] for item in values_and_counts),
            "n_transform_repeats": len(repeats),
            "r2_drop_mean": float(np.mean(values)),
            "r2_drop_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "r2_drop_median": float(np.median(values)),
            "r2_drop_q25": float(np.quantile(values, 0.25)),
            "r2_drop_q75": float(np.quantile(values, 0.75)),
        })
    return result


def _write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_versions(
    prediction_dirs: dict[str, tuple[Path, Path]],
    v1_2_effects: list[Path],
    output_dir: Path,
) -> Path:
    """Summarize per-release prediction runs plus existing v1.2 paired effects."""

    output = output_dir.resolve()
    versioned_rows: list[tuple[str, dict[str, str]]] = []
    evaluator = DefaultRawPredictionEvaluator()
    for version in VERSIONS:
        if version not in prediction_dirs:
            continue
        for source, predictions in zip(
            ("friedman", "openml"), prediction_dirs[version]
        ):
            evaluation = evaluator.evaluate(
                predictions, output / "intermediate" / version / source
            )
            rows = _read_effects(evaluation.paired_effects_path)
            expected = MODEL_IDS[version]
            if {row["model_id"] for row in rows} != {expected}:
                raise ValueError(
                    f"{version}/{source} does not contain exactly {expected}"
                )
            versioned_rows.extend((version, row) for row in rows)
    for path in v1_2_effects:
        rows = [
            row for row in _read_effects(path)
            if row["model_id"] == MODEL_IDS["v1.2"]
        ]
        versioned_rows.extend(("v1.2", row) for row in rows)

    for version in VERSIONS:
        require_benchmark_sizes(
            _with_logical_dataset(
                [row for row_version, row in versioned_rows if row_version == version]
            ),
            f"TabDPT {version} paired effects",
        )
    summaries = _summaries(versioned_rows)
    _write_summary(summaries, output / "tabdpt_version_summary.csv")
    return output


# --- inputs for the appendix version figures ---------------------------------

def _read_task_effects(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PART_I2_TASK_EFFECT_COLUMNS:
            raise ValueError(f"Part I.2 task-effect schema mismatch: {path}")
        return list(reader)


def _only(paths: list[Path], what: str, *, hint: str = "") -> Path:
    if len(paths) != 1:
        message = f"expected exactly one {what}, found {len(paths)}"
        raise ValueError(message + hint if not paths else message)
    return paths[0]


def _with_logical_dataset(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Give Friedman tasks the ``<setting>::<task>`` id used by the sweeps.

    The same 50 task seeds appear in every Friedman setting, so a bare task id
    would silently merge tasks from different settings.  The expanded Part I.1
    evaluation already stores the qualified id; per-release runs keep the
    setting in ``pair_id`` only.
    """
    out = []
    for row in rows:
        row = dict(row)
        if row["source"] == "synthetic" and "::" not in row["dataset_id"]:
            setting = row["pair_id"].split(":", 1)[0]
            row["dataset_id"] = f"{setting}::{row['dataset_id']}"
        out.append(row)
    return out


def collect_plot_inputs(
    versions_root: Path,
    reference_root: Path,
) -> dict[str, list[dict[str, str]]]:
    """Gather the four releases plus the TabPFN v2 reference for plotting."""

    reference_root = Path(reference_root).resolve()
    part_i1_dir = reference_root / "evaluations_part_i1"
    part_i2_dir = reference_root / "evaluations_part_i2"
    keep = {MODEL_IDS["v1.2"], BASELINE_MODEL_ID}
    part_i1 = [
        row for row in _read_effects(part_i1_dir / "paired_effects.csv")
        if row["model_id"] in keep
    ]
    part_i2_rows = _read_task_effects(part_i2_dir / "task_effects.csv")
    part_i2_friedman = [
        row for row in part_i2_rows
        if row["model_id"] in keep and row["source"] == "synthetic"
    ]
    part_i2_openml = [
        row for row in part_i2_rows
        if row["model_id"] in keep and row["source"] == "real"
    ]

    for version, directory in VERSION_DIRS.items():
        expected = MODEL_IDS[version]
        intermediate = versions_root / "evaluations" / "intermediate" / version
        for source in ("friedman", "openml"):
            part_i1_candidates = sorted((
                intermediate / source / "evaluations_part_i1"
            ).glob("*/paired_effects.csv"))
            if not part_i1_candidates:
                # Compatibility with release runs produced before the two
                # canonical evaluation roots were introduced.
                part_i1_candidates = sorted((
                    intermediate / source / "evaluations"
                ).glob("*/paired_effects.csv"))
            effects = _only(
                part_i1_candidates,
                f"{version}/{source} paired_effects.csv",
                hint=(
                    "; these per-release effects are written by"
                    " evaluate-tabdpt-versions, which must run first"
                ),
            )
            rows = _read_effects(effects)
            if {row["model_id"] for row in rows} != {expected}:
                raise ValueError(f"{effects} does not contain exactly {expected}")
            part_i1.extend(rows)
        friedman_candidates = sorted((
            versions_root / directory / "friedman"
            / "evaluations_part_i2" / "friedman_sweeps"
        ).glob("*/task_effects.csv"))
        if not friedman_candidates:
            # Compatibility with already completed legacy-release runs.
            friedman_candidates = sorted((
                versions_root / directory / "friedman"
                / "evaluations_part_i2_friedman_sweeps"
            ).glob("*/task_effects.csv"))
        friedman = _only(
            friedman_candidates,
            f"{version} Friedman Part I.2 task_effects.csv",
        )
        openml = _only(
            sorted((
                versions_root / directory / "openml" / "evaluations_part_i2"
            ).glob("*/task_effects.csv")),
            f"{version} OpenML Part I.2 task_effects.csv",
        )
        part_i2_friedman.extend(
            row for row in _read_task_effects(friedman)
            if row["source"] == "synthetic"
        )
        part_i2_openml.extend(
            row for row in _read_task_effects(openml) if row["source"] == "real"
        )

    part_i1 = _with_logical_dataset(part_i1)
    require_benchmark_sizes(part_i1, "Part I.1 paired effects")
    require_benchmark_sizes(part_i2_friedman, "Part I.2 Friedman task effects")
    require_benchmark_sizes(part_i2_openml, "Part I.2 OpenML task effects")
    return {
        "part_i1": part_i1,
        "part_i2_friedman": part_i2_friedman,
        "part_i2_openml": part_i2_openml,
    }


# --- single-release smoke test -----------------------------------------------

def smoke_test(
    config_path: Path, model_id: str, checkpoint: Path, device: str
) -> dict[str, object]:
    """Load one historical release and run a tiny regression episode."""

    config = load_config(config_path)
    configured = {model.model_id: model for model in config.models}
    if model_id not in configured:
        raise ValueError(f"model is not configured: {model_id}")
    options = dict(configured[model_id].options)
    options.update({
        "model_path": str(checkpoint.resolve()),
        "n_ensembles": 1,
        "compile": False,
        "use_flash": False,
        "verbose": False,
    })
    model = DefaultPretrainedModelFactory(
        seed=config.run_seed,
        max_input_features=config.max_input_features,
    ).build(ModelConfig(model_id, options))

    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(40, 5)).astype(np.float32)
    y = (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    ).astype(np.float32)
    model.prepare(device)
    prediction = np.asarray(model.fit_predict(X[:32], y[:32], X[32:]))
    if prediction.shape != (8,) or not np.isfinite(prediction).all():
        raise RuntimeError(f"invalid smoke prediction: shape={prediction.shape}")
    return {
        "model_id": model_id,
        "model_version": model.descriptor.model_version,
        "prediction_shape": list(prediction.shape),
        "prediction_mean": float(prediction.mean()),
    }


__all__ = [
    "BASELINE_MODEL_ID",
    "MODEL_IDS",
    "SUMMARY_COLUMNS",
    "VERSIONS",
    "VERSION_DIRS",
    "collect_plot_inputs",
    "prepare_indexes",
    "smoke_test",
    "summarize_versions",
]
