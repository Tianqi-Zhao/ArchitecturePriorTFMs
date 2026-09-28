"""Dependency-light runtime helpers shared by all experiment architectures."""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def default_device():
    """Return the best available torch device without importing torch eagerly."""

    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def csv_has_data_rows(path: Path) -> bool:
    """Return whether a CSV exists and contains at least one data row."""

    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        return any(csv.DictReader(handle))


def discard_empty_csvs(*paths: Path) -> None:
    """Delete header-only CSVs so interrupted work can be retried."""

    for path in paths:
        if path.is_file() and not csv_has_data_rows(path):
            path.unlink(missing_ok=True)


def require_csv_rows(*paths: Path, label: str) -> None:
    """Require every listed CSV to contain at least one data row."""

    empty = [path for path in paths if not csv_has_data_rows(path)]
    if not empty:
        return
    discard_empty_csvs(*empty)
    joined = ", ".join(str(path) for path in empty)
    raise SystemExit(f"{label} produced no data rows ({joined})")


def require_any_csv_rows(*paths: Path, label: str) -> None:
    """Require at least one listed CSV to contain a data row."""

    if any(csv_has_data_rows(path) for path in paths):
        return
    discard_empty_csvs(*paths)
    joined = ", ".join(str(path) for path in paths)
    raise SystemExit(f"{label} produced no data rows ({joined})")


def staging_csv_path(path: Path) -> Path:
    """Return the unpublished sidecar path used during a CSV write."""

    return path.with_name(f".{path.name}.partial")


@contextlib.contextmanager
def guarded_csv_outputs(
    *paths: Path,
    label: str,
    required: Iterable[Path] | None = None,
    require_any: bool = False,
):
    """Publish CSVs atomically after their required rows are validated."""

    staging = {path: staging_csv_path(path) for path in paths}
    for staged in staging.values():
        staged.unlink(missing_ok=True)
    check = tuple(required) if required is not None else paths
    try:
        yield staging
        staged_check = tuple(staging[path] for path in check)
        if require_any:
            require_any_csv_rows(*staged_check, label=label)
        else:
            require_csv_rows(*staged_check, label=label)
        for final, staged in staging.items():
            if staged.is_file():
                staged.replace(final)
    except BaseException:
        for staged in staging.values():
            staged.unlink(missing_ok=True)
        raise


def subsample_indices(total: int, count: int, seed: int) -> list[int]:
    if count <= 0 or count >= total:
        return list(range(total))
    rng = np.random.default_rng([seed, 424242])
    return sorted(int(i) for i in rng.choice(total, size=count, replace=False))


class OnceWarner:
    """Print each warning category at most once."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def warn(self, key: str, message: str) -> None:
        if key not in self._seen:
            self._seen.add(key)
            print(f"WARNING: {message}")


def reraise_episode_failure(
    error: BaseException,
    *,
    label: str,
    dataset_idx: object,
    detail: str = "",
) -> None:
    """Re-raise a leaf failure so partial outputs cannot be published."""

    extra = f" {detail}" if detail else ""
    raise RuntimeError(
        f"{label} failed on dataset_idx={dataset_idx}{extra}: "
        f"{type(error).__name__}: {error}"
    ) from error


__all__ = [
    "OnceWarner",
    "csv_has_data_rows",
    "default_device",
    "discard_empty_csvs",
    "guarded_csv_outputs",
    "require_any_csv_rows",
    "require_csv_rows",
    "reraise_episode_failure",
    "staging_csv_path",
    "subsample_indices",
]
