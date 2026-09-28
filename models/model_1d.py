"""
1D attention model for tabular data, inspired by TabDPT architecture.

Unlike NanoTabPFN's 2D attention (separate attention over features and samples),
this model treats each sample as a single token by projecting all features into
one embedding vector, then applies standard transformer attention across samples.

Architecture (binary classification):
    1. FeatureEncoder1D: (B, R, num_features) -> (B, R, E)
       Projects all features of each sample into a single embedding.
    2. TargetEncoder1D: (B, R_train, 1) -> (B, R_train, E)
       Encodes context labels into embeddings.
    3. Combine: context_token = feature_emb + label_emb; query_token = feature_emb only.
    4. TransformerEncoderLayer1D: 1D attention across samples.
       Context attends to context; query attends to context only.
    5. Decoder1D: query embeddings -> class logits.

Feature alignment: input features are expected to be aligned to a fixed dimension
(num_features) via zero-padding (if fewer) or PCA (if more).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm

from models.classifier_contract import NanoTabClassifierBase


class NanoTabDPTModel(nn.Module):
    def __init__(self, num_features: int, embedding_size: int, num_attention_heads: int,
                 mlp_hidden_size: int, num_layers: int, num_outputs: int):
        """
        Initializes the 1D attention model.

        Args:
            num_features: fixed number of input features (aligned via padding/PCA)
            embedding_size: dimension of token embeddings
            num_attention_heads: number of attention heads
            mlp_hidden_size: hidden dimension of feed-forward blocks
            num_layers: number of transformer layers
            num_outputs: number of output classes
        """
        super().__init__()
        self.num_features = num_features
        self.feature_encoder = FeatureEncoder1D(num_features, embedding_size)
        self.target_encoder = TargetEncoder1D(embedding_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderLayer1D(embedding_size, num_attention_heads, mlp_hidden_size)
            for _ in range(num_layers)
        ])
        self.decoder = Decoder1D(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int,
                padding_mask: torch.Tensor = None,
                return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[dict]]:
        """
        Args:
            src: tuple of (x, y_train)
                x: (B, R, C) all samples (train + test, possibly with padding rows)
                y_train: (B, R_train) or (B, R_train, 1) context labels
            train_test_split_index: number of training samples
            padding_mask: (B, R) optional bool tensor, True for padding rows.
                          Used to prevent attention to padding positions and should also
                          be used externally to mask the loss on padding test rows.
            return_attention: if True, also return per-layer attention dicts with keys
                              "datapoint_train_attn" (B, H, train, train) and
                              "datapoint_test_attn"  (B, H, test, train).
        """
        x_src, y_src = src
        # x_src: (B, R, C), y_src: (B, R_train) or (B, R_train, 1)
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)

        B, R, C = x_src.shape

        # align features to fixed dimension (zero-padding for fewer features)
        # NOTE: if C > num_features, data should be pre-aligned via PCA in the Classifier
        # before reaching forward(). PCA cannot be placed here because:
        #   1. PCA (SVD-based) is not differentiable — breaks gradient computation during training
        #   2. PCA requires a separate fit step on training data — cannot be done per forward call
        #   3. During training with synthetic data, C == num_features always holds
        assert C <= self.num_features, (
            f"Input has {C} features but model expects at most {self.num_features}. "
            f"Use NanoTabDPTClassifier which handles PCA alignment automatically, "
            f"or preprocess your data to have <= {self.num_features} features."
        )
        if C < self.num_features:
            x_src = F.pad(x_src, (0, self.num_features - C))

        # from here on B=Batches, R=Rows(samples), E=embedding size
        # encode features: each sample's features -> single embedding (B,R,E)
        x_enc = self.feature_encoder(x_src, train_test_split_index)
        # encode context labels: (B, R_train, 1) -> (B, R_train, E)
        y_enc = self.target_encoder(y_src)

        # context tokens = feature embedding + label embedding
        x_ctx = x_enc[:, :train_test_split_index] + y_enc   # (B, R_train, E)
        # query tokens = feature embedding only (no label info)
        x_qry = x_enc[:, train_test_split_index:]            # (B, R_test, E)

        # concatenate context and query tokens
        src = torch.cat([x_ctx, x_qry], dim=1)  # (B, R, E)

        # zero out padding rows' embeddings so they don't produce extreme values
        # from normalization of zero features (0 → (0-mean)/std → non-zero).
        # This ensures padding rows stay inert throughout the transformer layers.
        if padding_mask is not None:
            src = src.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # apply transformer layers with 1D attention
        all_attentions: list[dict] = []
        for block in self.transformer_blocks:
            if return_attention:
                src, attn_dict = block(src, train_test_split_index=train_test_split_index,
                                       padding_mask=padding_mask, return_attention=True)
                all_attentions.append(attn_dict)
            else:
                src = block(src, train_test_split_index=train_test_split_index,
                            padding_mask=padding_mask)

        # extract query outputs and decode to class logits
        output = src[:, train_test_split_index:]  # (B, R_test, E)
        output = self.decoder(output)             # (B, R_test, num_outputs)
        if return_attention:
            return output, all_attentions
        return output


class FeatureEncoder1D(nn.Module):
    def __init__(self, num_features: int, embedding_size: int):
        """
        Creates the linear layer that projects all features into embedding space.
        Unlike the 2D model which embeds each feature independently (1->E),
        this embeds all features together (num_features->E).
        """
        super().__init__()
        self.num_features = num_features
        self.linear_layer = nn.Linear(num_features, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Normalizes features based on training data statistics, then projects to embedding space.

        Args:
            x: (batch_size, num_rows, num_features)
            train_test_split_index: number of training samples
        Returns:
            (batch_size, num_rows, embedding_size)
        """
        # normalize using training data statistics (per-feature)
        x_ctx = x[:, :train_test_split_index]
        mean = torch.mean(x_ctx, dim=1, keepdim=True)
        var = ((x_ctx - mean) ** 2).mean(dim=1, keepdim=True)
        std = torch.sqrt(var + 1e-20)
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder1D(nn.Module):
    def __init__(self, embedding_size: int):
        """Creates the linear layer that embeds context labels."""
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_ctx: torch.Tensor) -> torch.Tensor:
        """
        Encodes context labels into embeddings.

        Args:
            y_ctx: (batch_size, num_train_samples, 1) - context labels
        Returns:
            (batch_size, num_train_samples, embedding_size)
        """
        return self.linear_layer(y_ctx.to(dtype=self.linear_layer.weight.dtype))


class TransformerEncoderLayer1D(nn.Module):
    """
    Transformer encoder layer with 1D attention across samples.

    Attention pattern (same as the between-datapoints attention in the 2D model):
        - Context samples attend to all context samples (self-attention)
        - Query samples attend only to context samples (cross-attention)
        - Query samples do NOT attend to each other (ensures independent predictions)
    """
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attention = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int,
                padding_mask: torch.Tensor = None,
                return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Applies 1D attention followed by a feed-forward MLP.

        Args:
            src: (batch_size, num_rows, embedding_size) embeddings for all samples
            train_test_split_index: number of context (training) samples
            padding_mask: (batch_size, num_rows) optional bool tensor, True for padding rows.
                          Splits into context key_padding_mask to prevent attending to padding
                          context rows in both self-attention and cross-attention.
            return_attention: if True, also return a dict with raw attention weights of this block:
                "datapoint_train_attn": (B, H, train, train)
                "datapoint_test_attn":  (B, H, test, train)
        Returns:
            (batch_size, num_rows, embedding_size), optionally with attn_dict
        """
        # extract context key_padding_mask from the full padding_mask
        ctx_key_padding_mask = None
        if padding_mask is not None:
            ctx_key_padding_mask = padding_mask[:, :train_test_split_index]  # (B, R_train)

        # context self-attention: training samples attend to each other
        src_ctx, train_attn_w = self.self_attention(
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            key_padding_mask=ctx_key_padding_mask,
            need_weights=return_attention, average_attn_weights=False,
        )
        # query cross-attention: test samples attend to training samples only
        src_qry, test_attn_w = self.self_attention(
            src[:, train_test_split_index:],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            key_padding_mask=ctx_key_padding_mask,
            need_weights=return_attention, average_attn_weights=False,
        )
        src = torch.cat([src_ctx, src_qry], dim=1) + src  # residual connection
        src = self.norm1(src)
        # feed-forward MLP
        src = self.linear2(F.gelu(self.linear1(src))) + src  # residual connection
        src = self.norm2(src)
        # re-zero padding rows after residual connections and layer norms,
        # which would otherwise turn zero embeddings into non-zero values
        if padding_mask is not None:
            src = src.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if return_attention:
            attn_dict = {
                "datapoint_train_attn": train_attn_w,  # (B, H, train, train)
                "datapoint_test_attn": test_attn_w,    # (B, H, test, train)
            }
            return src, attn_dict
        return src


class Decoder1D(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        """Initializes the classification decoder (no regression head)."""
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies an MLP to query embeddings to get class logits.

        Args:
            x: (batch_size, num_query_samples, embedding_size)
        Returns:
            (batch_size, num_query_samples, num_outputs)
        """
        return self.linear2(F.gelu(self.linear1(x)))


class NanoTabDPTClassifier(NanoTabClassifierBase):
    """
    scikit-learn like interface for the 1D attention model.
    Handles feature alignment via PCA (dimensionality reduction) and zero-padding.
    """
    def __init__(self, model: NanoTabDPTModel, device: torch.device):
        super().__init__(
            model,
            device,
            num_outputs=model.decoder.linear2.out_features,
        )
        self.pca = None

    def fit(self, X_train: np.array, y_train: np.array):
        """
        Stores training data and applies feature alignment if needed.
        If X_train has more features than the model expects, PCA is used to reduce dimensionality.
        If fewer, zero-padding is applied during prediction.
        """
        super().fit(X_train, y_train)
        num_features = self.model.num_features

        if self.X_train.shape[1] > num_features:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=num_features)
            self.pca.fit(self.X_train)
        else:
            self.pca = None
        return self

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """Aligns feature dimension: PCA if too many, zero-padding if too few."""
        num_features = self.model.num_features
        if self.pca is not None:
            X = self.pca.transform(X)
        if X.shape[1] < num_features:
            X = np.pad(X, ((0, 0), (0, num_features - X.shape[1])))
        return X
