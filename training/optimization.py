"""Optimizer lifecycle shared by NanoTab training entry points."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import schedulefree
import torch
from torch import nn


OptimizerName = Literal["schedulefree-adamw", "adamw"]


class OptimizerController:
    """Give regular and schedule-free optimizers one small common interface."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler

    @classmethod
    def build(
        cls,
        parameters: Iterable[nn.Parameter],
        *,
        name: OptimizerName,
        learning_rate: float,
        weight_decay: float,
        adam_beta2: float,
        total_steps: int,
    ) -> "OptimizerController":
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if not 0.0 <= adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must be in [0, 1)")
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")

        if name == "schedulefree-adamw":
            optimizer = schedulefree.AdamWScheduleFree(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            return cls(optimizer)
        if name == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, adam_beta2),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
            )
            return cls(optimizer, scheduler)
        raise ValueError(
            f"unknown optimizer {name!r}; expected 'schedulefree-adamw' or 'adamw'"
        )

    def train(self) -> None:
        callback = getattr(self.optimizer, "train", None)
        if callback is not None:
            callback()

    def eval(self) -> None:
        callback = getattr(self.optimizer, "eval", None)
        if callback is not None:
            callback()

    def step(self) -> None:
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    @property
    def scheduler_name(self) -> str:
        return "cosine" if self.scheduler is not None else "none"
