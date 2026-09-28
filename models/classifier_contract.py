"""Shared implementation for the repository's sklearn-like model wrappers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def prepare_fit_data(
    X_train,
    y_train,
    *,
    task_type: str,
    num_outputs: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalize fit arrays and validate zero-based classification labels."""

    X = np.asarray(X_train, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X_train must have shape (n, d), got {X.shape}")

    y = np.asarray(y_train)
    if y.shape != (len(X),):
        raise ValueError(
            f"y_train must have shape ({len(X)},), got {y.shape}"
        )
    if task_type != "classification":
        return X, y.astype(np.float32, copy=False), 1
    if len(y) == 0:
        raise ValueError("classification fit requires at least one target")

    try:
        numeric = y.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("classification labels must be finite integers") from error
    if not np.all(np.isfinite(numeric)):
        raise ValueError("classification labels must be finite integers")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError("classification labels must be integers")

    labels = rounded.astype(np.int64)
    classes = np.unique(labels)
    expected = np.arange(len(classes), dtype=np.int64)
    if not np.array_equal(classes, expected):
        raise ValueError(
            "classification labels must be contiguous and zero-based; "
            f"got {classes.tolist()}"
        )
    num_classes = int(len(classes))
    if num_classes > int(num_outputs):
        raise ValueError(
            f"classification data has {num_classes} classes, but the model has "
            f"only {int(num_outputs)} outputs"
        )
    return X, labels.astype(np.float32), num_classes


class NanoTabClassifierBase:
    """Common fit/predict contract for all in-repository NanoTab models.

    Subclasses only need to customize feature transformation.  The models all
    consume the same ``(x, y_context)`` input and return ``(B, query, output)``
    tensors, so tensor conversion, task dispatch, output shaping and attention
    conversion belong in one place.
    """

    def __init__(self, model, device: torch.device, *, num_outputs: int) -> None:
        self.model = model.to(device)
        self.device = device
        self.num_outputs = int(num_outputs)
        self.task_type = "regression" if self.num_outputs == 1 else "classification"

    def fit(self, X_train, y_train):
        self.X_train, self.y_train, self.num_classes = prepare_fit_data(
            X_train,
            y_train,
            task_type=self.task_type,
            num_outputs=self.num_outputs,
        )
        return self

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """Architecture-specific feature alignment hook."""

        return X

    def _prepare_test_features(self, X_test) -> np.ndarray:
        X = np.asarray(X_test, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X_test must have shape (n, d), got {X.shape}")
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"X_test has {X.shape[1]} features, expected {self.X_train.shape[1]}"
            )
        return X

    def _forward(
        self,
        X_test,
        *,
        return_attention: bool,
        return_feature_value: bool = False,
        feature_attn_masks=None,
    ):
        X_test = self._prepare_test_features(X_test)
        X_train = self._transform_features(self.X_train)
        X_test = self._transform_features(X_test)
        x = np.concatenate((X_train, X_test), axis=0)
        with torch.no_grad():
            x_tensor = torch.as_tensor(
                x, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            y_tensor = torch.as_tensor(
                self.y_train, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            kwargs = {"return_attention": return_attention}
            if return_feature_value:
                kwargs["return_feature_value"] = True
            if feature_attn_masks is not None:
                kwargs["feature_attn_masks"] = feature_attn_masks
            return self.model(
                (x_tensor, y_tensor),
                train_test_split_index=len(self.X_train),
                **kwargs,
            )

    @staticmethod
    def _split_result(result, return_evidence: bool):
        if not return_evidence:
            return result, None
        output, evidence = result
        converted = [
            {key: value.detach().cpu() for key, value in layer.items()}
            for layer in evidence
        ]
        return output, converted

    def predict_values(
        self,
        X_test,
        return_attention: bool = False,
        return_feature_value: bool = False,
        feature_attn_masks=None,
    ):
        if self.task_type != "regression":
            raise RuntimeError("predict_values is only available for regression models")
        result = self._forward(
            X_test,
            return_attention=return_attention,
            return_feature_value=return_feature_value,
            feature_attn_masks=feature_attn_masks,
        )
        return_evidence = return_attention or return_feature_value
        output, evidence = self._split_result(result, return_evidence)
        values = output[0, :, 0].detach().cpu().numpy()
        return (values, evidence) if return_evidence else values

    def predict_proba(
        self,
        X_test,
        return_attention: bool = False,
        return_feature_value: bool = False,
        feature_attn_masks=None,
    ):
        if self.task_type != "classification":
            raise RuntimeError("predict_proba is only available for classification models")
        result = self._forward(
            X_test,
            return_attention=return_attention,
            return_feature_value=return_feature_value,
            feature_attn_masks=feature_attn_masks,
        )
        return_evidence = return_attention or return_feature_value
        output, evidence = self._split_result(result, return_evidence)
        probabilities = (
            F.softmax(output[0, :, :self.num_classes], dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        return (probabilities, evidence) if return_evidence else probabilities

    def predict(self, X_test) -> np.ndarray:
        if self.task_type == "classification":
            return self.predict_proba(X_test).argmax(axis=1)
        return self.predict_values(X_test)
