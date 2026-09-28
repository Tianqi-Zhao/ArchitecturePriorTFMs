"""Canonical task-type names shared by data, training, and evaluation."""


def normalize_task_type(task_type: str) -> str:
    if task_type in {"binary", "classification", "binary_classification"}:
        return "classification"
    if task_type == "regression":
        return "regression"
    raise ValueError(f"unknown task_type: {task_type!r}")
