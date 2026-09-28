import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm

from models.attention_value import multihead_attention_value_paths
from models.classifier_contract import NanoTabClassifierBase

class NanoTabPFNModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int,
                 feature_std_correction: int | None = None,
                 feature_std_eps: float = 1e-20):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        super().__init__()
        self.feature_encoder = FeatureEncoder(
            embedding_size,
            std_correction=feature_std_correction,
            std_eps=feature_std_eps,
        )
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size))
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int,
                return_attention: bool = False,
                return_feature_value: bool = False,
                padding_mask: torch.Tensor = None,
                feature_padding_mask: torch.Tensor = None,
                feature_attn_masks: list | None = None) -> torch.Tensor | tuple[torch.Tensor, list[dict]]:
        return_evidence = return_attention or return_feature_value
        x_src, y_src = src
        # we expect the labels to look like (batches, num_train_datapoints, 1),
        # so we add the last dimension if it is missing
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        x_src = self.feature_encoder(
            x_src,
            train_test_split_index,
            padding_mask=padding_mask,
            feature_padding_mask=feature_padding_mask,
        )
        num_rows = x_src.shape[1]
        # padds the y_train up to y by using the mean,
        # then converts scalar values to embeddings (B,R,1,E)
        train_padding_mask = padding_mask[:, :train_test_split_index] if padding_mask is not None else None
        y_src = self.target_encoder(y_src, num_rows, padding_mask=train_padding_mask)
        # concatenates the feature embeddings with the target embeddings
        # to give us the full table of embeddings (B,R,C,E))
        src = torch.cat([x_src, y_src], 2)
        # repeatedly applies the transformer block on (B,R,C,E)
        # feature_attn_masks (optional) is a length-num_layers list; entry i is
        # an additive (C,C) attention mask for layer i's feature attention, or
        # None to leave that layer unmasked (used by the attention knockout)
        all_evidence = []
        for i, block in enumerate(self.transformer_blocks):
            block_feature_attn_mask = feature_attn_masks[i] if feature_attn_masks is not None else None
            if return_evidence:
                src, layer_evidence = block(src, train_test_split_index=train_test_split_index,
                                       return_attention=return_attention, padding_mask=padding_mask,
                                       return_feature_value=return_feature_value,
                                       feature_padding_mask=feature_padding_mask,
                                       feature_attn_mask=block_feature_attn_mask)
                all_evidence.append(layer_evidence)
            else:
                src = block(src, train_test_split_index=train_test_split_index,
                            padding_mask=padding_mask,
                            feature_padding_mask=feature_padding_mask,
                            feature_attn_mask=block_feature_attn_mask)
        # selects the target embeddings (B,num_targets,1,E)
        output = src[:, train_test_split_index:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        if return_evidence:
            return output, all_evidence
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int, std_correction: int | None = None,
                 std_eps: float = 1e-20):
        """ Creates the linear layer that we will use to embed our features. """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)
        self.std_correction = std_correction
        self.std_eps = float(std_eps)

    def forward(self, x: torch.Tensor, train_test_split_index: int,
                padding_mask: torch.Tensor = None,
                feature_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Normalizes all the features based on the mean and std of the features of the training data,
        clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
            train_test_split_index: (int) the number of datapoints in X_train
            padding_mask: (torch.Tensor) optional (batch_size, num_rows) bool tensor, True for padding rows
            feature_padding_mask: (torch.Tensor) optional (batch_size, num_features) bool tensor,
                                  True for padded feature columns
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
                           the embeddings of the features
        """
        x = x.unsqueeze(-1)  # (B, R, C, 1)
        x_ctx = x[:, :train_test_split_index]

        if padding_mask is not None:
            ctx_mask = padding_mask[:, :train_test_split_index].unsqueeze(-1).unsqueeze(-1)
            valid = (~ctx_mask).float()  # (B, T, 1, 1)
            count = valid.sum(dim=1, keepdim=True).clamp(min=1)
            mean = (x_ctx * valid).sum(dim=1, keepdim=True) / count
            correction = (
                0 if self.std_correction is None else int(self.std_correction)
            )
            denominator = (count - correction).clamp(min=1)
            var = (
                ((x_ctx - mean) ** 2 * valid).sum(dim=1, keepdim=True)
                / denominator
            )
            if self.std_correction is None:
                std = torch.sqrt(var + self.std_eps)
            else:
                std = torch.sqrt(var) + self.std_eps
        elif self.std_correction is None:
            mean = torch.mean(x_ctx, dim=1, keepdim=True)
            var = ((x_ctx - mean) ** 2).mean(dim=1, keepdim=True)
            std = torch.sqrt(var + self.std_eps)
        else:
            # TFM-Playground checkpoints were trained with torch.std's sample
            # correction and an epsilon added after the square root.  Keep the
            # existing NanoTab population-variance path as the default while
            # allowing those public weights to run under their exact forward.
            mean = torch.mean(x_ctx, dim=1, keepdim=True)
            std = torch.std(
                x_ctx,
                dim=1,
                keepdim=True,
                correction=int(self.std_correction),
            ) + self.std_eps

        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        embedded = self.linear_layer(x)
        if feature_padding_mask is not None:
            embedded = embedded.masked_fill(
                feature_padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
        return embedded

class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """ Creates the linear layer that we will use to embed our targets. """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Padds up y_train to the full length of y using the mean per dataset and then embeds it using a linear layer

        Args:
            y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
            num_rows: (int) the full length of y
            padding_mask: (torch.Tensor) optional (batch_size, num_train_datapoints) bool, True for padding
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, 1, embedding_size), representing
                           the embeddings of the targets
        """
        y_train = y_train.to(dtype=self.linear_layer.weight.dtype)
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).to(dtype=y_train.dtype)  # (B, T, 1)
            count = valid.sum(dim=1, keepdim=True).clamp(min=1)
            mean = (y_train * valid).sum(dim=1, keepdim=True) / count
            y_train = y_train * valid + mean * (1 - valid)
        else:
            mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)

class TransformerEncoderLayer(nn.Module):
    """
    Modified version of older version of https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630
    """
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)
        self.self_attention_between_features = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int,
                return_attention: bool = False,
                return_feature_value: bool = False,
                padding_mask: torch.Tensor = None,
                feature_padding_mask: torch.Tensor = None,
                feature_attn_mask: torch.Tensor = None) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Takes the embeddings of the table as input and applies self-attention between features and self-attention between datapoints
        followed by a simple 2 layer MLP.

        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                                for all the cells in the table
            train_test_split_index: (int) the length of X_train
            return_attention: (bool) if True, also return attention weight maps
            return_feature_value: (bool) if True, return projected per-feature
                                  value paths without requiring attention maps
            padding_mask: (torch.Tensor) optional (batch_size, num_rows) bool, True for padding rows
            feature_padding_mask: (torch.Tensor) optional (batch_size, num_x_features) bool, True for
                                  padded X columns. The concatenated target column is never padded.
            feature_attn_mask: optional additive (C, C) mask for feature attention
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
            If either evidence flag is True, also returns a dict. Attention
            capture adds the following keys:
                "feature_attn": (batch_size, num_rows, num_features, num_features)
                "datapoint_train_attn": (batch_size, num_features, num_train, num_train)
                "datapoint_test_attn": (batch_size, num_features, num_test, num_train)
        """
        return_evidence = return_attention or return_feature_value
        batch_size, rows_size, col_size, embedding_size = src.shape

        # build context key_padding_mask for datapoint attention: (B*C, train_len)
        ctx_key_padding_mask = None
        if padding_mask is not None:
            ctx_mask = padding_mask[:, :train_test_split_index]                       # (B, T)
            ctx_key_padding_mask = ctx_mask.unsqueeze(1).expand(-1, col_size, -1)     # (B, C, T)
            ctx_key_padding_mask = ctx_key_padding_mask.reshape(batch_size * col_size, -1)

        feature_key_padding_mask = None
        if feature_padding_mask is not None:
            if feature_padding_mask.shape != (batch_size, col_size - 1):
                raise ValueError(
                    "feature_padding_mask must have shape "
                    f"{(batch_size, col_size - 1)}, got {tuple(feature_padding_mask.shape)}"
                )
            target_col = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=src.device
            )
            full_col_mask = torch.cat([feature_padding_mask, target_col], dim=1)
            feature_key_padding_mask = (
                full_col_mask.unsqueeze(1)
                .expand(-1, rows_size, -1)
                .reshape(batch_size * rows_size, col_size)
            )

        # attention between features
        src = src.reshape(batch_size*rows_size, col_size, embedding_size)
        feature_input = src
        # attn_mask (optional) is an additive (C,C) mask: -inf entries drop those
        # key-columns before softmax and the remaining row is renormalized to 1
        feat_out, feat_attn_weights = self.self_attention_between_features(
            src, src, src,
            need_weights=return_evidence,
            average_attn_weights=False,
            attn_mask=feature_attn_mask,
            key_padding_mask=feature_key_padding_mask,
        )
        feature_values = None
        if return_feature_value:
            feature_values = multihead_attention_value_paths(
                self.self_attention_between_features,
                feature_input,
                feat_attn_weights,
                query_slice=slice(-1, None),
                key_slice=slice(None, -1),
            ).reshape(
                batch_size,
                rows_size,
                1,
                col_size - 1,
                embedding_size,
            )
        src = feat_out + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size*col_size, rows_size, embedding_size)
        # training data attends to itself
        src_left, dp_train_attn_weights = self.self_attention_between_datapoints(
            src[:,:train_test_split_index], src[:,:train_test_split_index], src[:,:train_test_split_index],
            key_padding_mask=ctx_key_padding_mask,
            need_weights=return_attention, average_attn_weights=False)
        # test data attends to the training data
        src_right, dp_test_attn_weights = self.self_attention_between_datapoints(
            src[:,train_test_split_index:], src[:,:train_test_split_index], src[:,:train_test_split_index],
            key_padding_mask=ctx_key_padding_mask,
            need_weights=return_attention, average_attn_weights=False)
        src = torch.cat([src_left, src_right], dim=1)+src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # MLP after attention
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)

        # re-zero padding rows so residual connections / layer norms don't leak
        if padding_mask is not None:
            src = src.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        if feature_padding_mask is not None:
            padded_features = src[:, :, :-1].masked_fill(
                feature_padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            src = torch.cat([padded_features, src[:, :, -1:]], dim=2)

        if return_evidence:
            evidence = {}
            if return_attention:
                # feat_attn_weights: (B*R, H, C, C) -> (B, R, H, C, C)
                feat_attn = feat_attn_weights.reshape(batch_size, rows_size, -1, col_size, col_size)
                # dp_train_attn_weights: (B*C, H, train, train) -> (B, C, H, train, train)
                dp_train_attn = dp_train_attn_weights.reshape(batch_size, col_size, -1, train_test_split_index, train_test_split_index)
                # dp_test_attn_weights: (B*C, H, test, train) -> (B, C, H, test, train)
                num_test = rows_size - train_test_split_index
                dp_test_attn = dp_test_attn_weights.reshape(batch_size, col_size, -1, num_test, train_test_split_index)
                evidence.update({
                    "feature_attn": feat_attn,
                    "datapoint_train_attn": dp_train_attn,
                    "datapoint_test_attn": dp_test_attn,
                })
            if feature_values is not None:
                evidence["feature_value"] = feature_values
            return src, evidence
        return src

class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        """ Initializes the linear layers for use in the forward """
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies an MLP to the embeddings to get the logits

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, embedding_size)
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_outputs)
        """
        return self.linear2(F.gelu(self.linear1(x)))

class NanoTabPFNClassifier(NanoTabClassifierBase):
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel, device: torch.device):
        super().__init__(
            model,
            device,
            num_outputs=model.decoder.linear2.out_features,
        )


def full_support_bar_mean(logits: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """Mean of the FullSupportBarDistribution used by TFM-Playground.

    Interior bars are uniform.  The first and last bars have half-normal
    unbounded tails whose median lies one edge-bucket width from the adjacent
    finite border.  This is the exact point prediction made by the official
    ``pfns.FullSupportBarDistribution.mean`` without making ``pfns`` a runtime
    dependency of NanoTab's experiment adapters.
    """

    borders = torch.as_tensor(borders, dtype=logits.dtype, device=logits.device)
    if borders.ndim != 1 or borders.numel() != logits.shape[-1] + 1:
        raise ValueError(
            f"bar borders must have shape ({logits.shape[-1] + 1},), got "
            f"{tuple(borders.shape)}"
        )
    widths = borders[1:] - borders[:-1]
    if not bool(torch.isfinite(borders).all() and (widths > 0).all()):
        raise ValueError("bar borders must be finite and strictly increasing")
    means = borders[:-1] + widths / 2
    # HalfNormal(scale=1): median=sqrt(2)*erfinv(0.5), mean=sqrt(2/pi).
    unit_median = math.sqrt(2.0) * torch.erfinv(
        torch.tensor(0.5, dtype=logits.dtype, device=logits.device)
    )
    unit_mean = math.sqrt(2.0 / math.pi)
    means = means.clone()
    means[0] = borders[1] - widths[0] * unit_mean / unit_median
    means[-1] = borders[-2] + widths[-1] * unit_mean / unit_median
    return torch.softmax(logits, dim=-1) @ means


class TFMPlaygroundNanoTabPFNRegressor:
    """Numeric-table inference wrapper for the public TFM-Playground regressor.

    Experiment 4 already supplies finite numeric arrays and requires raw-column
    identity to survive every intervention.  We therefore run the checkpoint's
    model-native normalization directly and deliberately omit the optional
    sklearn dataframe preprocessor from TFM-Playground's general-purpose API.
    """

    def __init__(self, model: NanoTabPFNModel, borders: torch.Tensor,
                 device: torch.device, target_std_ddof: int = 1):
        self.model = model.to(device)
        self.device = device
        self.target_std_ddof = int(target_std_ddof)
        self.borders = torch.as_tensor(
            borders, dtype=torch.float32, device=device
        ).reshape(-1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        if X_train.ndim != 2 or y_train.shape != (len(X_train),):
            raise ValueError(
                f"TFM-Playground fit expects X=(n,d), y=(n,), got "
                f"{X_train.shape} and {y_train.shape}"
            )
        if len(X_train) < 2:
            raise ValueError("TFM-Playground regression needs at least two context rows")
        if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
            raise ValueError("TFM-Playground experiment adapter needs finite numeric data")
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_mean = float(np.mean(y_train))
        self.y_train_std = float(
            np.std(y_train, ddof=self.target_std_ddof) + 1e-8
        )
        self.y_train_normalized = (
            (y_train - self.y_train_mean) / self.y_train_std
        ).astype(np.float32)
        return self

    def predict_values(self, X_test: np.ndarray, return_attention: bool = False,
                       feature_attn_masks: list | None = None):
        X_test = np.asarray(X_test, dtype=np.float32)
        if X_test.ndim != 2 or X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"TFM-Playground predict expects (*,{self.X_train.shape[1]}), "
                f"got {X_test.shape}"
            )
        if not np.all(np.isfinite(X_test)):
            raise ValueError("TFM-Playground experiment adapter needs finite numeric data")
        X = np.concatenate([self.X_train, X_test], axis=0)
        with torch.no_grad():
            x_tensor = torch.as_tensor(
                X, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            y_tensor = torch.as_tensor(
                self.y_train_normalized,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            result = self.model(
                (x_tensor, y_tensor),
                train_test_split_index=len(self.X_train),
                return_attention=return_attention,
                feature_attn_masks=feature_attn_masks,
            )
            if return_attention:
                logits, all_attentions = result
            else:
                logits = result
            prediction = full_support_bar_mean(
                logits.squeeze(0), self.borders
            )
            prediction = prediction * self.y_train_std + self.y_train_mean
        values = prediction.detach().cpu().numpy()
        if not return_attention:
            return values
        attentions = [
            {key: value.detach().cpu() for key, value in layer.items()}
            for layer in all_attentions
        ]
        return values, attentions

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.predict_values(X_test)
