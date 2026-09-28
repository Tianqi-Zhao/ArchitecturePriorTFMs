"""Dimension and irrelevant-feature extension points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from enum import Enum
import hashlib

import numpy as np

from experiments_arch.experiment_0_pretrained.data.base import (
    BaseDataset,
    DimensionInfo,
    IrrelevantFeatureBank,
    PreparedDataset,
)


class RoundingMode(str, Enum):
    CEIL = "ceil"
    HALF_UP = "half_up"


def quantize_feature_count(value: Decimal, mode: RoundingMode) -> int:
    """Quantize a non-negative exact decimal without binary float surprises."""

    if not value.is_finite():
        raise ValueError("feature count must be finite")
    if value < 0:
        raise ValueError("feature count must be non-negative")
    rounding = ROUND_CEILING if mode is RoundingMode.CEIL else ROUND_HALF_UP
    return int(value.to_integral_value(rounding=rounding))


@dataclass(frozen=True)
class GrowingWidthStrategy:
    """Keep the base task fixed while increasing the total feature width."""

    rounding: RoundingMode = RoundingMode.CEIL

    def __post_init__(self) -> None:
        if not isinstance(self.rounding, RoundingMode):
            object.__setattr__(self, "rounding", RoundingMode(str(self.rounding)))

    def resolve(self, base: BaseDataset, requested_ratio: float) -> DimensionInfo:
        ratio = Decimal(str(float(requested_ratio)))
        denominator = base.ratio_denominator
        count = quantize_feature_count(Decimal(denominator) * ratio, self.rounding)
        return DimensionInfo(
            requested_irrelevant_ratio=float(ratio),
            actual_irrelevant_ratio=count / denominator,
            ratio_denominator=denominator,
            d_base=base.d_base,
            d_irrelevant=count,
            d_total=base.d_base + count,
            relevant_feature_count=base.relevant_feature_count,
            rounding=self.rounding.value,
        )

class IrrelevantFeatureAdder(ABC):
    """Collaborator boundary for generating and mixing irrelevant features."""

    name: str

    @abstractmethod
    def build_bank(
        self,
        base: BaseDataset,
        max_irrelevant_features: int,
        *,
        repeat: int,
        seed: int,
    ) -> IrrelevantFeatureBank:
        """Return the shared bank from which every dose takes a prefix."""

    @abstractmethod
    def materialize(
        self,
        base: BaseDataset,
        dimensions: DimensionInfo,
        bank: IrrelevantFeatureBank,
        *,
        condition_id: str,
        pair_id: str,
        repeat: int,
        seed: int,
    ) -> PreparedDataset:
        """Return one shuffled condition with complete feature provenance.

        The collaborator owns the generation and shuffle algorithm. The result
        must use a prefix of ``bank`` for nested doses and identify every stable
        base feature through ``feature_role`` and ``source_feature_index``.
        Predict and PDP consume the returned arrays without reordering them.
        """

        raise NotImplementedError


def _derived_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


class DefaultIrrelevantFeatureAdder(IrrelevantFeatureAdder):
    """Build the nested maximum-dose bank specified by Part I.1.

    Synthetic tasks receive independent U(0, 1) columns. Real tasks receive
    whole-row-permuted copies: one row permutation is shared by every column
    in a copy block, while context and query are permuted independently.
    """

    name = "part_i_1_irrelevant_features_v1"

    def build_bank(
        self,
        base: BaseDataset,
        max_irrelevant_features: int,
        *,
        repeat: int,
        seed: int,
    ) -> IrrelevantFeatureBank:
        width = int(max_irrelevant_features)
        if width < 0:
            raise ValueError("max_irrelevant_features must be non-negative")
        bank_ids = np.arange(width, dtype=np.int32)
        if width == 0:
            return IrrelevantFeatureBank(
                X_context=np.empty((len(base.X_context), 0), dtype=np.float32),
                X_query=np.empty((len(base.X_query), 0), dtype=np.float32),
                source_feature_index=np.empty(0, dtype=np.int32),
                copy_block_id=np.empty(0, dtype=np.int32),
                feature_bank_id=bank_ids,
                metadata={"algorithm": self.name, "repeat": int(repeat)},
            )

        if base.source.value == "synthetic":
            context_rng = np.random.default_rng(
                _derived_seed(seed, base.dataset_id, repeat, "synthetic", "context")
            )
            query_rng = np.random.default_rng(
                _derived_seed(seed, base.dataset_id, repeat, "synthetic", "query")
            )
            context = context_rng.random((len(base.X_context), width), dtype=np.float32)
            query = query_rng.random((len(base.X_query), width), dtype=np.float32)
            sources = np.full(width, -1, dtype=np.int32)
            blocks = (bank_ids // base.d_base + 1).astype(np.int32)
            metadata = {
                "algorithm": self.name,
                "distribution": "independent_uniform_0_1",
                "context_query_streams": "separate",
                "repeat": int(repeat),
            }
        else:
            selection_rng = np.random.default_rng(
                _derived_seed(seed, base.dataset_id, repeat, "real", "column_order")
            )
            source_order = selection_rng.permutation(base.d_base).astype(np.int32)
            context_blocks: list[np.ndarray] = []
            query_blocks: list[np.ndarray] = []
            source_blocks: list[np.ndarray] = []
            copy_ids: list[np.ndarray] = []
            remaining = width
            block_id = 1
            while remaining:
                take = min(base.d_base, remaining)
                selected = source_order[:take]
                context_rng = np.random.default_rng(
                    _derived_seed(
                        seed, base.dataset_id, repeat, "real", block_id, "context"
                    )
                )
                query_rng = np.random.default_rng(
                    _derived_seed(
                        seed, base.dataset_id, repeat, "real", block_id, "query"
                    )
                )
                context_perm = context_rng.permutation(len(base.X_context))
                query_perm = query_rng.permutation(len(base.X_query))
                context_blocks.append(base.X_context[context_perm][:, selected])
                query_blocks.append(base.X_query[query_perm][:, selected])
                source_blocks.append(selected)
                copy_ids.append(np.full(take, block_id, dtype=np.int32))
                remaining -= take
                block_id += 1
            context = np.concatenate(context_blocks, axis=1).astype(np.float32, copy=False)
            query = np.concatenate(query_blocks, axis=1).astype(np.float32, copy=False)
            sources = np.concatenate(source_blocks)
            blocks = np.concatenate(copy_ids)
            metadata = {
                "algorithm": self.name,
                "distribution": "whole_row_permuted_base_copy",
                "within_copy_column_permutation": "shared",
                "context_query_permutations": "separate",
                "column_selection_order": source_order.tolist(),
                "repeat": int(repeat),
            }

        return IrrelevantFeatureBank(
            X_context=context,
            X_query=query,
            source_feature_index=sources,
            copy_block_id=blocks,
            feature_bank_id=bank_ids,
            metadata=metadata,
        )

    def materialize(
        self,
        base: BaseDataset,
        dimensions: DimensionInfo,
        bank: IrrelevantFeatureBank,
        *,
        condition_id: str,
        pair_id: str,
        repeat: int,
        seed: int,
    ) -> PreparedDataset:
        if dimensions.d_base != base.d_base:
            raise ValueError("dimensions and base dataset disagree on d_base")
        if dimensions.d_irrelevant > bank.width:
            raise ValueError("requested dose exceeds the maximum irrelevant bank")

        n_irrelevant = dimensions.d_irrelevant
        context = np.concatenate(
            [base.X_context, bank.X_context[:, :n_irrelevant]], axis=1
        )
        query = np.concatenate(
            [base.X_query, bank.X_query[:, :n_irrelevant]], axis=1
        )
        roles = np.concatenate(
            [
                np.full(base.d_base, 0, dtype=np.uint8),
                np.full(n_irrelevant, 1, dtype=np.uint8),
            ]
        )
        sources = np.concatenate(
            [
                np.arange(base.d_base, dtype=np.int32),
                bank.source_feature_index[:n_irrelevant],
            ]
        )
        blocks = np.concatenate(
            [np.zeros(base.d_base, dtype=np.int32), bank.copy_block_id[:n_irrelevant]]
        )
        bank_ids = np.concatenate(
            [np.arange(base.d_base, dtype=np.int32), bank.feature_bank_id[:n_irrelevant]]
        )

        # Each feature receives a width-independent priority. Sorting the
        # currently present features therefore preserves the relative order of
        # every shared column as the dose grows.
        priorities = np.fromiter(
            (
                _derived_seed(
                    seed,
                    base.dataset_id,
                    repeat,
                    "column_shuffle",
                    int(role),
                    int(feature_id),
                )
                for role, feature_id in zip(roles, bank_ids)
            ),
            dtype=np.uint64,
            count=len(roles),
        )
        order = np.argsort(priorities, kind="stable")
        return PreparedDataset(
            dataset_id=base.dataset_id,
            dataset_name=base.dataset_name,
            source=base.source,
            condition_id=condition_id,
            pair_id=pair_id,
            dimensions=dimensions,
            transform_name=self.name,
            transform_repeat=repeat,
            transform_seed=seed,
            column_layout="shuffle",
            X_context=context[:, order],
            y_context=base.y_context,
            X_query=query[:, order],
            y_query=base.y_query,
            context_row_id=base.context_row_id,
            query_row_id=base.query_row_id,
            feature_role=roles[order],
            source_feature_index=sources[order],
            copy_block_id=blocks[order],
            feature_bank_id=bank_ids[order],
            metadata={"bank": dict(bank.metadata), "nested_prefix": True},
        )
