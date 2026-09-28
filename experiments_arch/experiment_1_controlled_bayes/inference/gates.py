"""Row ownership and branch scaling shared by all torch backends."""

from dataclasses import dataclass

import torch

from ..artifacts.protocol import validate_row_scope


@dataclass(frozen=True)
class RowLayout:
    """One forward's [thinking, context, query] rows before batch flattening.

    Thinking rows belong to the context side. Every batch member has the same
    layout, but may choose its own intervention scope.
    """

    n_context: int
    n_query: int
    n_prefix: int = 0

    def __post_init__(self):
        if self.n_context < 1 or self.n_query < 1 or self.n_prefix < 0:
            raise ValueError("row layout requires context/query rows and a nonnegative prefix")

    @property
    def context_end(self):
        return self.n_prefix + self.n_context

    @property
    def rows(self):
        return self.context_end + self.n_query

    def select(self, scope):
        validate_row_scope(scope)
        return slice(self.context_end if scope == "query" else 0,
                     self.context_end if scope == "context" else self.rows)

    def check_batch(self, flat_rows, batch_count):
        if flat_rows != batch_count * self.rows:
            raise ValueError("feature attention row layout does not align with the request batch")

    def indices(self, slots, scope, *, device):
        """Flatten selected (batch slot, row) pairs, preserving slot order."""
        selected = self.select(scope)
        rows = torch.arange(selected.start, selected.stop, device=device)
        return (torch.as_tensor(slots, device=device)[:, None] * self.rows + rows).flatten()


@dataclass(frozen=True)
class FeatureBranchGate:
    """Scale selected updates; the backend identifies the residual boundary.

    Entries map row scopes to (batch slot, epsilon) pairs for one layer.
    Both flattened and unflattened table batches use the same tensor operation.
    """

    layout: RowLayout
    batch_count: int
    entries: dict

    def apply(self, output, *, residual=None):
        shape = output.shape
        flat = output.reshape(-1, *shape[-2:])
        self.layout.check_batch(len(flat), self.batch_count)
        base = None if residual is None else residual.reshape_as(flat)
        result = flat.clone()
        for scope, entries in self.entries.items():
            slots, epsilons = zip(*entries)
            ids = self.layout.indices(slots, scope, device=output.device)
            rows = len(ids) // len(slots)
            factor = 1 - output.new_tensor(epsilons).repeat_interleave(rows)[:, None, None]
            selected = flat.index_select(0, ids)
            if base is None:
                scaled = factor * selected
            else:
                identity = base.index_select(0, ids)
                scaled = identity + factor * (selected - identity)
            result.index_copy_(0, ids, scaled)
        return result.reshape(shape)
