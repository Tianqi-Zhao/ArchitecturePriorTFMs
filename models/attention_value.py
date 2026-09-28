"""Exact per-key value paths for PyTorch multi-head attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def project_attention_value_paths(
    attention: nn.MultiheadAttention,
    value_heads: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    query_slice: slice = slice(None),
    key_slice: slice = slice(None),
) -> torch.Tensor:
    """Project each selected ``alpha * V`` term through ``W_O`` separately.

    ``value_heads`` has shape ``(batch, heads, keys, head_dim)`` and
    ``attention_weights`` has shape ``(batch, heads, queries, keys)``.  The
    returned tensor keeps the selected query and key axes as
    ``(batch, queries, keys, embedding_dim)``.  The output-projection bias is
    deliberately omitted because it belongs to the summed attention output,
    not to any individual key.
    """

    if value_heads.ndim != 4 or attention_weights.ndim != 4:
        raise ValueError(
            "value heads and attention weights must both be four-dimensional"
        )
    batch, heads, keys, head_dim = value_heads.shape
    if attention_weights.shape[:2] != (batch, heads):
        raise ValueError(
            "value heads and attention weights must share batch/head axes"
        )
    if attention_weights.shape[3] != keys:
        raise ValueError(
            "value heads and attention weights must share the key axis"
        )
    if heads * head_dim != int(attention.embed_dim):
        raise ValueError(
            "value-head width does not match the attention embedding dimension"
        )

    weights = attention_weights[:, :, query_slice, key_slice]
    values = value_heads[:, :, key_slice, :]
    weighted = weights[..., None].to(values.dtype) * values[:, :, None, :, :]
    # (B, H, Q, K, Dh) -> (B, Q, K, H*Dh), preserving one vector per edge.
    weighted = weighted.permute(0, 2, 3, 1, 4).reshape(
        batch,
        weights.shape[2],
        weights.shape[3],
        attention.embed_dim,
    )
    return F.linear(weighted, attention.out_proj.weight, bias=None)


def multihead_attention_value_paths(
    attention: nn.MultiheadAttention,
    key_value: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    query_slice: slice = slice(None),
    key_slice: slice = slice(None),
) -> torch.Tensor:
    """Recover ``V`` from a self-attention input and project its value paths."""

    if not attention._qkv_same_embed_dim or attention.in_proj_weight is None:
        raise ValueError("value-path capture requires packed same-width Q/K/V")
    if key_value.ndim != 3 or key_value.shape[-1] != attention.embed_dim:
        raise ValueError(
            "key/value input must have shape (batch, keys, embedding_dim)"
        )
    _, _, values = F._in_projection_packed(
        key_value,
        key_value,
        key_value,
        attention.in_proj_weight,
        attention.in_proj_bias,
    )
    value_heads = values.reshape(
        values.shape[0], values.shape[1], attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    return project_attention_value_paths(
        attention,
        value_heads,
        attention_weights,
        query_slice=query_slice,
        key_slice=key_slice,
    )
