"""Small metric-logging boundary for shared training code."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ScalarMetricLogger(Protocol):
    """Consumer of step-indexed scalar training metrics."""

    def log_scalar(self, name: str, value: float, step: int) -> None: ...

    def flush(self) -> None: ...


class TensorBoardMetricLogger:
    """Write scalar metrics through PyTorch's TensorBoard integration."""

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard logging requires the 'tensorboard' package; "
                "install requirements.txt or pass --no-tensorboard"
            ) from error
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, global_step=step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> "TensorBoardMetricLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
