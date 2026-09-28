"""Reusable pretrained regression adapters and their default factory.

The adapters in this module intentionally expose only the prediction boundary
shared by the paper experiments. Optional packages and checkpoints are loaded
lazily in ``prepare``; model identity, checkpoint defaults, fit caching, target
scaling, and ordinary inference belong to this shared layer.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from experiment_core.adapters import (
    DEFAULT_MAX_INPUT_FEATURES_BY_ORIGIN,
    MODEL_ORIGIN_PRETRAINED,
    ModelConfig,
    ModelDescriptor,
    PredictiveModel,
    PretrainedModelFactory,
)
from experiment_core.metrics import RegressionTargetScaler
from experiment_core.model_runtime import (
    construct_with_optional_kwargs,
    resolve_existing_checkpoint,
    same_fit_inputs,
    set_rebuilding_random_seed,
)


MODEL_GROUP_1D = "1d_arch"
MODEL_GROUP_2D = "2d_arch"
MODEL_GROUP_1DICL = "1dicl_arch"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINTS_DIR = _PROJECT_ROOT / "checkpoints"

DEFAULT_TABPFN_V2_PATH = str(
    _CHECKPOINTS_DIR / "tabpfn-v2-regressor.ckpt"
)
DEFAULT_TABPFN_V25_PATH = str(
    _CHECKPOINTS_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt"
)
DEFAULT_TABPFN_V3_PATH = str(
    _CHECKPOINTS_DIR / "tabpfn-v3-regressor-v3_default.ckpt"
)
DEFAULT_TABICL_V2_PATH = str(
    _CHECKPOINTS_DIR / "tabicl-regressor-v2-20260212.ckpt"
)
DEFAULT_TABDPT_PATH = str(_CHECKPOINTS_DIR / "tabdpt1_2.safetensors")
DEFAULT_TABDPT_V11_PATH = str(_CHECKPOINTS_DIR / "tabdpt1_1.safetensors")
DEFAULT_TABDPT_V10_PATH = str(_CHECKPOINTS_DIR / "tabdpt_76M.ckpt")
DEFAULT_TABDPT_V13_PATH = str(_CHECKPOINTS_DIR / "tabdpt1_3.safetensors")

TABDPT_RETRIEVAL_OFF_MODEL_IDS = frozenset({
    "tabdpt_retrieval_off",
    "tabdpt_v1_1_retrieval_off",
    "tabdpt_v1_0_retrieval_off",
    "tabdpt_v1_3_retrieval_off",
})

TABPFN_DEFAULT_PATHS = MappingProxyType({
    "tabpfn_v2": DEFAULT_TABPFN_V2_PATH,
    "tabpfn_v2.5": DEFAULT_TABPFN_V25_PATH,
    "tabpfn_v3": DEFAULT_TABPFN_V3_PATH,
})
TABICL_DEFAULT_PATHS = MappingProxyType({
    "tabicl_v2": DEFAULT_TABICL_V2_PATH,
})


@dataclass(frozen=True)
class PretrainedModelSpec:
    """Stable project metadata for one external checkpoint family."""

    model_id: str
    model_group: str
    package: str
    default_model_path: str


PRETRAINED_MODEL_SPECS = MappingProxyType({
    "tabpfn_v2": PretrainedModelSpec(
        "tabpfn_v2", MODEL_GROUP_2D, "tabpfn", DEFAULT_TABPFN_V2_PATH
    ),
    "tabpfn_v2.5": PretrainedModelSpec(
        "tabpfn_v2.5", MODEL_GROUP_2D, "tabpfn", DEFAULT_TABPFN_V25_PATH
    ),
    "tabpfn_v3": PretrainedModelSpec(
        "tabpfn_v3", MODEL_GROUP_1DICL, "tabpfn", DEFAULT_TABPFN_V3_PATH
    ),
    "tabicl_v2": PretrainedModelSpec(
        "tabicl_v2", MODEL_GROUP_1DICL, "tabicl", DEFAULT_TABICL_V2_PATH
    ),
    "tabdpt": PretrainedModelSpec(
        "tabdpt", MODEL_GROUP_1D, "tabdpt", DEFAULT_TABDPT_PATH
    ),
    "tabdpt_retrieval_off": PretrainedModelSpec(
        "tabdpt_retrieval_off",
        MODEL_GROUP_1D,
        "tabdpt",
        DEFAULT_TABDPT_PATH,
    ),
    "tabdpt_v1_1_retrieval_off": PretrainedModelSpec(
        "tabdpt_v1_1_retrieval_off",
        MODEL_GROUP_1D,
        "tabdpt",
        DEFAULT_TABDPT_V11_PATH,
    ),
    "tabdpt_v1_0_retrieval_off": PretrainedModelSpec(
        "tabdpt_v1_0_retrieval_off",
        MODEL_GROUP_1D,
        "tabdpt",
        DEFAULT_TABDPT_V10_PATH,
    ),
    "tabdpt_v1_3_retrieval_off": PretrainedModelSpec(
        "tabdpt_v1_3_retrieval_off",
        MODEL_GROUP_1D,
        "tabdpt",
        DEFAULT_TABDPT_V13_PATH,
    ),
})

DEFAULT_PRETRAINED_REGRESSION_MODEL_IDS = (
    "tabdpt_retrieval_off",
    "tabpfn_v2",
    "tabpfn_v2.5",
    "tabicl_v2",
    "tabpfn_v3",
)


def pretrained_model_groups() -> dict[str, str]:
    """Return a mutable copy of the shared external model/group catalog."""

    return {
        model_id: spec.model_group
        for model_id, spec in PRETRAINED_MODEL_SPECS.items()
    }


def _package_checkpoint_version(package: str, checkpoint: str) -> str:
    try:
        package_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    return f"{package}=={package_version};checkpoint={Path(checkpoint).name}"


def identity_preprocess_transforms(model_path: str, version: str):
    """Preserve the raw-column to feature-token identity used by old adapters."""

    from tabpfn.model_loading import load_model_criterion_config

    _, _, _, inference_config = load_model_criterion_config(
        model_path,
        check_bar_distribution_criterion=True,
        cache_trainset_representation=False,
        which="regressor",
        version=version,
        download_if_not_exists=False,
    )
    transforms = inference_config.PREPROCESS_TRANSFORMS
    if not transforms:
        raise RuntimeError(
            f"{version} checkpoint {model_path} resolved no "
            "PREPROCESS_TRANSFORMS; cannot preserve raw feature identity"
        )
    return [
        dataclasses.replace(
            transform,
            global_transformer_name=None,
            append_original=False,
        )
        for transform in transforms
    ]


def _validate_model_inputs(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    *,
    max_input_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_context = np.asarray(X_context)
    y_context = np.asarray(y_context)
    X_query = np.asarray(X_query)
    if X_context.ndim != 2 or X_query.ndim != 2:
        raise ValueError("X_context and X_query must be two-dimensional")
    if y_context.ndim != 1:
        raise ValueError("y_context must be one-dimensional")
    if X_context.shape[0] != y_context.shape[0]:
        raise ValueError(
            "X_context and y_context must contain the same number of rows"
        )
    if X_context.shape[0] < 1 or X_query.shape[0] < 1:
        raise ValueError("context and query must each contain at least one row")
    if X_context.shape[1] != X_query.shape[1]:
        raise ValueError("context and query feature widths must match")
    if X_context.shape[1] < 1:
        raise ValueError("model input must contain at least one feature")
    if X_context.shape[1] > max_input_features:
        raise ValueError(
            f"model accepts at most {max_input_features} features, got "
            f"{X_context.shape[1]}"
        )
    if not np.all(np.isfinite(y_context)):
        raise ValueError("y_context must contain only finite values")
    return X_context, y_context, X_query


class TargetScaledPredictiveModel(PredictiveModel):
    """Adapt a same-target-space backend to the public model contract.

    A backend only needs ``prepare(device)`` and
    ``fit_predict(X_context, y_context, X_query)``.  This wrapper owns target
    standardization, prediction validation, input-width policy, and descriptor
    publication.  Consequently a feature-rich legacy adapter can also be
    injected as a backend without introducing an import from core to legacy
    experiments.
    """

    def __init__(
        self,
        backend: Any,
        descriptor: ModelDescriptor,
        *,
        scale_targets: bool = True,
    ) -> None:
        self.backend = backend
        self.descriptor = descriptor
        self.scale_targets = bool(scale_targets)
        self._prepared = False

    def prepare(self, device: str) -> None:
        self.backend.prepare(device)
        version = getattr(self.backend, "model_version", None)
        if version:
            self.descriptor = dataclasses.replace(
                self.descriptor,
                model_version=str(version),
            )
        self._prepared = True

    def set_random_seed(self, seed: int) -> None:
        setter = getattr(self.backend, "set_random_seed", None)
        if callable(setter):
            setter(int(seed))

    def fit_predict(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        if not self._prepared:
            raise RuntimeError(
                f"model {self.descriptor.model_id!r} must be prepared before prediction"
            )
        X_context, y_context, X_query = _validate_model_inputs(
            X_context,
            y_context,
            X_query,
            max_input_features=self.descriptor.max_input_features,
        )
        scaler = RegressionTargetScaler.fit(y_context)
        backend_y = scaler.transform(y_context) if self.scale_targets else y_context
        prediction = np.asarray(
            self.backend.fit_predict(X_context, backend_y, X_query),
            dtype=np.float64,
        )
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            prediction = prediction[:, 0]
        if prediction.shape != (X_query.shape[0],):
            raise RuntimeError(
                f"model {self.descriptor.model_id!r} returned prediction shape "
                f"{prediction.shape}; expected ({X_query.shape[0]},)"
            )
        if self.scale_targets:
            prediction = scaler.inverse(prediction)
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError(
                f"model {self.descriptor.model_id!r} returned non-finite predictions"
            )
        return np.asarray(prediction, dtype=np.float64)


class TabPFNRegressionBackend:
    """Shared deterministic TabPFN fit/predict implementation."""

    def __init__(
        self,
        name: str,
        model_path: str,
        *,
        n_estimators: int,
        seed: int,
        preserve_feature_identity: bool,
        ignore_pretraining_limits: bool = False,
    ) -> None:
        self.name = name
        self.model_path = str(model_path)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.preserve_feature_identity = bool(preserve_feature_identity)
        self.ignore_pretraining_limits = bool(ignore_pretraining_limits)
        self.est = None
        self._device = "cpu"
        self._fit_X = None
        self._fit_y = None
        self.model_version = "unprepared"

    def _invalidate_fitted_state(self) -> None:
        self._fit_X = None
        self._fit_y = None

    def _after_fit(self) -> None:
        """Hook for feature-rich legacy adapters with fitted-module caches."""

    def prepare(self, device: str) -> None:
        from tabpfn import TabPFNRegressor

        self._device = str(device)
        resolved_path = resolve_existing_checkpoint(
            self.model_path or TABPFN_DEFAULT_PATHS[self.name],
            self.name,
        )
        required = {"random_state": self.seed, "model_path": resolved_path}
        optional: dict[str, Any] = {
            "device": self._device,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            # Explicitly pin the official no-progress default. Functional
            # analyses make many predict calls and must not flood logs.
            "show_progress_bar": False,
        }
        if self.preserve_feature_identity:
            if self.n_estimators != 1:
                print(
                    f"WARNING: {self.name!r}: forcing n_estimators=1 to preserve "
                    f"raw feature identity (was {self.n_estimators})"
                )
            version = {
                "tabpfn_v2": "v2",
                "tabpfn_v2.5": "v2.5",
                "tabpfn_v3": "v3",
            }[self.name]
            optional.update({
                "n_estimators": 1,
                "fit_mode": "fit_preprocessors",
                "inference_config": {
                    "FEATURE_SHIFT_METHOD": None,
                    "FINGERPRINT_FEATURE": False,
                    "POLYNOMIAL_FEATURES": "no",
                    "SUBSAMPLE_SAMPLES": None,
                    "PREPROCESS_TRANSFORMS": identity_preprocess_transforms(
                        resolved_path, version
                    ),
                },
            })
        elif self.n_estimators > 0:
            optional["n_estimators"] = self.n_estimators
        self.est = construct_with_optional_kwargs(
            TabPFNRegressor, required, optional
        )
        self._invalidate_fitted_state()
        self.model_version = _package_checkpoint_version("tabpfn", resolved_path)

    def set_random_seed(self, seed: int) -> None:
        set_rebuilding_random_seed(self, seed, self._device)

    def _ensure_fit(self, X_context, y_context) -> None:
        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.float32)
        if not same_fit_inputs(X_context, y_context, self._fit_X, self._fit_y):
            self.est.fit(X_context, y_context)
            self._fit_X = X_context.copy()
            self._fit_y = y_context.copy()
            self._after_fit()

    def fit_predict(self, X_context, y_context, X_query):
        self._ensure_fit(X_context, y_context)
        return np.asarray(self.est.predict(np.asarray(X_query)), dtype=np.float64)


class TabICLBackend:
    """Shared official TabICL fit/predict implementation."""

    def __init__(
        self,
        name: str,
        model_path: str,
        *,
        seed: int,
        n_estimators: int = 1,
        norm_methods: str | list[str] | None = "none",
        feat_shuffle_method: str = "none",
        batch_size: int | None = 1,
        kv_cache: bool | str = False,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = False,
        offload_mode: bool | str = False,
        verbose: bool = False,
        task_type: str = "regression",
    ) -> None:
        self.name = name
        self.model_path = str(model_path)
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.norm_methods = norm_methods
        self.feat_shuffle_method = str(feat_shuffle_method)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.kv_cache = kv_cache
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.verbose = bool(verbose)
        self.task_type = str(task_type)
        self.est = None
        self._device = "cpu"
        self._fit_X = None
        self._fit_y = None
        self.model_version = "unprepared"

    def prepare(self, device: str) -> None:
        from tabicl import TabICLClassifier, TabICLRegressor

        self._device = str(device)
        if not self.supports_task(self.task_type):
            raise RuntimeError(
                f"{self.name} has no official {self.task_type} checkpoint"
            )
        resolved_path = resolve_existing_checkpoint(
            self.model_path or TABICL_DEFAULT_PATHS[self.name],
            self.name,
        )
        required = {"model_path": resolved_path}
        optional = {
            "allow_auto_download": False,
            "device": self._device,
            "random_state": self.seed,
            "n_estimators": self.n_estimators,
            "norm_methods": self.norm_methods,
            "feat_shuffle_method": self.feat_shuffle_method,
            "batch_size": self.batch_size,
            "kv_cache": self.kv_cache,
            "use_amp": self.use_amp,
            "use_fa3": self.use_fa3,
            "offload_mode": self.offload_mode,
            "verbose": self.verbose,
        }
        estimator_class = (
            TabICLRegressor
            if self.task_type == "regression"
            else TabICLClassifier
        )
        self.est = construct_with_optional_kwargs(estimator_class, required, optional)
        self._fit_X = None
        self._fit_y = None
        self.model_version = _package_checkpoint_version("tabicl", resolved_path)

    def set_random_seed(self, seed: int) -> None:
        set_rebuilding_random_seed(self, seed, self._device)

    def supports_task(self, task_type: str) -> bool:
        return task_type in {"classification", "regression"}

    def _ensure_fit(self, X_context, y_context) -> None:
        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context)
        if not same_fit_inputs(X_context, y_context, self._fit_X, self._fit_y):
            self.est.fit(X_context, y_context)
            self._fit_X = X_context.copy()
            self._fit_y = y_context.copy()

    def fit_predict(self, X_context, y_context, X_query):
        self._ensure_fit(X_context, y_context)
        return np.asarray(self.est.predict(np.asarray(X_query)), dtype=np.float64)

    def fit_predict_proba(self, X_context, y_context, X_query):
        self._ensure_fit(X_context, y_context)
        return np.asarray(
            self.est.predict_proba(np.asarray(X_query)), dtype=np.float64
        )


class TabDPTBackend:
    """Shared deterministic TabDPT fit/predict implementation."""

    def __init__(
        self,
        model_path: str,
        *,
        context_size: int,
        seed: int,
        context_reduction: str = "subsample",
        feature_reduction: str | None = None,
        n_ensembles: int = 1,
        compile: bool = False,
        use_flash: bool = False,
        verbose: bool = False,
    ) -> None:
        self.name = "tabdpt"
        self.model_path = str(model_path)
        self.context_size = int(context_size)
        self.context_reduction = str(context_reduction)
        self.feature_reduction = (
            None if feature_reduction is None else str(feature_reduction)
        )
        self.n_ensembles = int(n_ensembles)
        self.compile = bool(compile)
        self.use_flash = bool(use_flash)
        self.verbose = bool(verbose)
        self.seed = int(seed)
        self.est = None
        self._fit_X = None
        self._fit_y = None
        self._fit_seed = None
        self._legacy_full_context = False
        self.model_version = "unprepared"

    def prepare(self, device: str) -> None:
        from tabdpt import TabDPTRegressor

        resolved_path = resolve_existing_checkpoint(
            self.model_path or DEFAULT_TABDPT_PATH,
            "tabdpt",
        )
        constructor = inspect.signature(TabDPTRegressor).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in constructor.values()
        )
        if "model_weight_path" in constructor or accepts_kwargs:
            required = {"model_weight_path": resolved_path}
            self._legacy_full_context = False
        elif "path" in constructor:
            required = {"path": resolved_path}
            self._legacy_full_context = True
        else:
            raise RuntimeError(
                "unsupported TabDPT constructor: expected model_weight_path or path"
            )
        optional = {
            "device": str(device),
            "context_reduction": self.context_reduction,
            "compile": self.compile,
            "use_flash": self.use_flash,
            "verbose": self.verbose,
        }
        if self.feature_reduction is not None:
            optional["feature_reduction"] = self.feature_reduction
        self.est = construct_with_optional_kwargs(
            TabDPTRegressor,
            required,
            optional,
        )
        self._invalidate_fitted_state()
        self.model_version = _package_checkpoint_version("tabdpt", resolved_path)

    def _invalidate_fitted_state(self) -> None:
        self._fit_X = None
        self._fit_y = None
        self._fit_seed = None

    def set_random_seed(self, seed: int) -> None:
        seed = int(seed)
        if seed != self.seed:
            self.seed = seed
            self._invalidate_fitted_state()

    def _ensure_fit(self, X_context, y_context) -> None:
        import torch

        X_context = np.asarray(X_context)
        y_context = np.asarray(y_context)
        if (
            self._fit_seed == self.seed
            and same_fit_inputs(
                X_context, y_context, self._fit_X, self._fit_y
            )
        ):
            return
        device = torch.device(str(getattr(self.est, "device", "cpu")))
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                torch.cuda.current_device()
                if device.index is None
                else device.index
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(self.seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(self.seed)
            if self._legacy_full_context:
                max_features = int(
                    getattr(self.est, "max_features", X_context.shape[1])
                )
                # v1.0 always applies its release-native PCA (after its own
                # imputation and standardization) above the checkpoint width.
                if (
                    X_context.shape[1] > max_features
                    and self.feature_reduction != "pca"
                ):
                    raise RuntimeError(
                        "TabDPT v1.0 input exceeds its checkpoint width; "
                        "configure feature_reduction='pca'"
                    )
            self.est.fit(X_context, y_context)
        self._fit_X = X_context.copy()
        self._fit_y = y_context.copy()
        self._fit_seed = self.seed

    def _context_size(self, num_context: int) -> int | None:
        if self.context_size == 0:
            # TabDPT v1.0 has no explicit retrieval-off sentinel.  Passing the
            # full fitted context prevents its retrieval branch from running.
            return int(num_context) if self._legacy_full_context else None
        return min(self.context_size, int(num_context))

    def _legacy_predict(self, X_query, context_size: int | None):
        """Seeded v1.0 prediction.

        v1.0 fits its randomized PCA (``torch.pca_lowrank``) inside
        ``predict``, so every call runs under the fitted seed to keep the
        feature basis reproducible and batch-invariant.
        """
        import torch

        device = torch.device(str(getattr(self.est, "device", "cpu")))
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                torch.cuda.current_device()
                if device.index is None
                else device.index
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(self.seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(self.seed)
            return self.est.predict(
                np.asarray(X_query),
                context_size=context_size,
            )

    def _predict(self, X_query, context_size: int | None):
        if self._legacy_full_context:
            return self._legacy_predict(X_query, context_size)
        try:
            return self.est.predict(
                np.asarray(X_query),
                n_ensembles=self.n_ensembles,
                context_size=context_size,
                seed=self.seed,
            )
        except TypeError as error:
            raise RuntimeError(
                "the installed TabDPT predict API cannot select a reproducible "
                "ensemble; expected predict(..., n_ensembles=..., "
                "context_size=..., seed=...)"
            ) from error

    def _predict_single(self, X_query, context_size: int | None):
        """Compatibility path for legacy gradient adapters."""

        if self._legacy_full_context:
            return self._legacy_predict(X_query, context_size)
        try:
            return self.est.predict(
                np.asarray(X_query),
                n_ensembles=1,
                context_size=context_size,
                seed=self.seed,
            )
        except TypeError as error:
            raise RuntimeError(
                "the installed TabDPT predict API cannot select a deterministic "
                "single estimator; expected predict(..., n_ensembles=1, "
                "context_size=..., seed=...)"
            ) from error

    def fit_predict(self, X_context, y_context, X_query):
        X_context = np.asarray(X_context)
        self._ensure_fit(X_context, y_context)
        prediction = self._predict(
            X_query, self._context_size(len(X_context))
        )
        return np.asarray(prediction, dtype=np.float64)


class TabPFNPredictiveModel(TargetScaledPredictiveModel):
    def __init__(
        self,
        *,
        model_id: str,
        model_path: str,
        n_estimators: int,
        seed: int,
        preserve_feature_identity: bool,
        ignore_pretraining_limits: bool,
        max_input_features: int,
        scale_targets: bool,
    ) -> None:
        spec = PRETRAINED_MODEL_SPECS[model_id]
        options = {
            "model_path": str(model_path),
            "n_estimators": int(n_estimators),
            "seed": int(seed),
            "preserve_feature_identity": bool(preserve_feature_identity),
            "ignore_pretraining_limits": bool(ignore_pretraining_limits),
            "target_scaling": bool(scale_targets),
        }
        super().__init__(
            TabPFNRegressionBackend(
                model_id,
                model_path,
                n_estimators=n_estimators,
                seed=seed,
                preserve_feature_identity=preserve_feature_identity,
                ignore_pretraining_limits=ignore_pretraining_limits,
            ),
            ModelDescriptor(
                model_id=model_id,
                model_group=spec.model_group,
                model_origin=MODEL_ORIGIN_PRETRAINED,
                model_version="unprepared",
                max_input_features=max_input_features,
                options=options,
            ),
            scale_targets=scale_targets,
        )


class TabICLPredictiveModel(TargetScaledPredictiveModel):
    def __init__(
        self,
        *,
        model_id: str,
        model_path: str,
        seed: int,
        n_estimators: int,
        norm_methods: str | list[str] | None,
        feat_shuffle_method: str,
        batch_size: int | None,
        kv_cache: bool | str,
        use_amp: bool | str,
        use_fa3: bool | str,
        offload_mode: bool | str,
        verbose: bool,
        max_input_features: int,
        scale_targets: bool,
    ) -> None:
        spec = PRETRAINED_MODEL_SPECS[model_id]
        options = {
            "model_path": str(model_path),
            "seed": int(seed),
            "n_estimators": int(n_estimators),
            "norm_methods": norm_methods,
            "feat_shuffle_method": str(feat_shuffle_method),
            "batch_size": batch_size,
            "kv_cache": kv_cache,
            "use_amp": use_amp,
            "use_fa3": use_fa3,
            "offload_mode": offload_mode,
            "verbose": bool(verbose),
            "target_scaling": bool(scale_targets),
        }
        super().__init__(
            TabICLBackend(
                model_id,
                model_path,
                seed=seed,
                n_estimators=n_estimators,
                norm_methods=norm_methods,
                feat_shuffle_method=feat_shuffle_method,
                batch_size=batch_size,
                kv_cache=kv_cache,
                use_amp=use_amp,
                use_fa3=use_fa3,
                offload_mode=offload_mode,
                verbose=verbose,
            ),
            ModelDescriptor(
                model_id=model_id,
                model_group=spec.model_group,
                model_origin=MODEL_ORIGIN_PRETRAINED,
                model_version="unprepared",
                max_input_features=max_input_features,
                options=options,
            ),
            scale_targets=scale_targets,
        )


class TabDPTPredictiveModel(TargetScaledPredictiveModel):
    def __init__(
        self,
        *,
        model_id: str,
        model_path: str,
        context_size: int,
        context_reduction: str,
        feature_reduction: str | None,
        n_ensembles: int,
        compile: bool,
        use_flash: bool,
        verbose: bool,
        seed: int,
        max_input_features: int,
        scale_targets: bool,
    ) -> None:
        spec = PRETRAINED_MODEL_SPECS[model_id]
        options = {
            "model_path": str(model_path),
            "context_size": int(context_size),
            "context_reduction": str(context_reduction),
            "n_ensembles": int(n_ensembles),
            "compile": bool(compile),
            "use_flash": bool(use_flash),
            "verbose": bool(verbose),
            "seed": int(seed),
            "target_scaling": bool(scale_targets),
        }
        if feature_reduction is not None:
            options["feature_reduction"] = str(feature_reduction)
        super().__init__(
            TabDPTBackend(
                model_path,
                context_size=context_size,
                context_reduction=context_reduction,
                feature_reduction=feature_reduction,
                n_ensembles=n_ensembles,
                compile=compile,
                use_flash=use_flash,
                verbose=verbose,
                seed=seed,
            ),
            ModelDescriptor(
                model_id=model_id,
                model_group=spec.model_group,
                model_origin=MODEL_ORIGIN_PRETRAINED,
                model_version="unprepared",
                max_input_features=max_input_features,
                options=options,
            ),
            scale_targets=scale_targets,
        )


ModelBuilder = Callable[[ModelConfig], PredictiveModel]


class DefaultPretrainedModelFactory(PretrainedModelFactory):
    """Build the project's supported pretrained regression adapters.

    Factory defaults are deterministic and can be overridden per model through
    JSON-safe ``ModelConfig.options``.  Custom builders are primarily useful for
    tests and project-local extensions; their ids are kept outside the built-in
    catalog and still have to return the neutral ``PredictiveModel`` contract.
    """

    _BUILTIN_MODEL_IDS = frozenset(DEFAULT_PRETRAINED_REGRESSION_MODEL_IDS) | {
        "tabdpt",
        "tabdpt_v1_1_retrieval_off",
        "tabdpt_v1_0_retrieval_off",
        "tabdpt_v1_3_retrieval_off",
    }

    def __init__(
        self,
        *,
        seed: int = 0,
        max_input_features: int | None = None,
        checkpoint_paths: Mapping[str, str] | None = None,
        builders: Mapping[str, ModelBuilder] | None = None,
    ) -> None:
        self.seed = int(seed)
        policy_limit = DEFAULT_MAX_INPUT_FEATURES_BY_ORIGIN[
            MODEL_ORIGIN_PRETRAINED
        ]
        self.max_input_features = (
            policy_limit
            if max_input_features is None
            else int(max_input_features)
        )
        if not 1 <= self.max_input_features <= policy_limit:
            raise ValueError(
                f"max_input_features must be in [1, {policy_limit}]"
            )
        self.checkpoint_paths = dict(checkpoint_paths or {})
        unknown_paths = set(self.checkpoint_paths) - set(PRETRAINED_MODEL_SPECS)
        if unknown_paths:
            raise ValueError(
                f"checkpoint paths configured for unknown models: "
                f"{sorted(unknown_paths)}"
            )
        self.builders = dict(builders or {})
        collisions = set(self.builders) & self._BUILTIN_MODEL_IDS
        if collisions:
            raise ValueError(
                f"custom builders cannot replace built-in models: {sorted(collisions)}"
            )

    @property
    def supported_model_ids(self) -> tuple[str, ...]:
        return (
            tuple(DEFAULT_PRETRAINED_REGRESSION_MODEL_IDS)
            + (
                "tabdpt",
                "tabdpt_v1_1_retrieval_off",
                "tabdpt_v1_0_retrieval_off",
                "tabdpt_v1_3_retrieval_off",
            )
            + tuple(sorted(self.builders))
        )

    @staticmethod
    def _checked_options(
        config: ModelConfig,
        allowed: set[str],
    ) -> dict[str, Any]:
        options = dict(config.options)
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(
                f"model {config.model_id!r} received unknown options: "
                f"{sorted(unknown)}"
            )
        return options

    def _common_values(
        self,
        config: ModelConfig,
        options: Mapping[str, Any],
    ) -> tuple[str, int, int, bool]:
        spec = PRETRAINED_MODEL_SPECS[config.model_id]
        model_path = str(
            options.get(
                "model_path",
                self.checkpoint_paths.get(config.model_id, spec.default_model_path),
            )
        )
        seed = int(options.get("seed", self.seed))
        max_features = int(
            options.get("max_input_features", self.max_input_features)
        )
        if not 1 <= max_features <= self.max_input_features:
            raise ValueError(
                f"model {config.model_id!r} max_input_features must be in "
                f"[1, {self.max_input_features}]"
            )
        scale_targets = options.get("target_scaling", False)
        if not isinstance(scale_targets, bool):
            raise TypeError("target_scaling must be a boolean")
        return model_path, seed, max_features, scale_targets

    def build(self, config: ModelConfig) -> PredictiveModel:
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        custom = self.builders.get(config.model_id)
        if custom is not None:
            model = custom(config)
            if not isinstance(model, PredictiveModel):
                raise TypeError(
                    f"builder for {config.model_id!r} did not return PredictiveModel"
                )
            if model.descriptor.model_id != config.model_id:
                raise ValueError(
                    f"builder for {config.model_id!r} returned descriptor for "
                    f"{model.descriptor.model_id!r}"
                )
            return model
        if config.model_id not in self._BUILTIN_MODEL_IDS:
            raise ValueError(
                f"unknown pretrained regression model {config.model_id!r}; "
                f"expected one of {sorted(self._BUILTIN_MODEL_IDS)}"
            )

        common = {
            "model_path",
            "seed",
            "max_input_features",
            "target_scaling",
        }
        if config.model_id in TABPFN_DEFAULT_PATHS:
            options = self._checked_options(
                config,
                common
                | {
                    "n_estimators",
                    "preserve_feature_identity",
                    "ignore_pretraining_limits",
                },
            )
            model_path, seed, max_features, scale_targets = self._common_values(
                config, options
            )
            n_estimators = int(options.get("n_estimators", 8))
            if n_estimators < 1:
                raise ValueError("n_estimators must be positive")
            preserve_identity = options.get("preserve_feature_identity", False)
            if not isinstance(preserve_identity, bool):
                raise TypeError("preserve_feature_identity must be a boolean")
            if preserve_identity and n_estimators != 1:
                raise ValueError(
                    "preserve_feature_identity requires n_estimators=1"
                )
            ignore_pretraining_limits = options.get(
                "ignore_pretraining_limits", False
            )
            if not isinstance(ignore_pretraining_limits, bool):
                raise TypeError("ignore_pretraining_limits must be a boolean")
            return TabPFNPredictiveModel(
                model_id=config.model_id,
                model_path=model_path,
                n_estimators=n_estimators,
                seed=seed,
                preserve_feature_identity=preserve_identity,
                ignore_pretraining_limits=ignore_pretraining_limits,
                max_input_features=max_features,
                scale_targets=scale_targets,
            )

        if config.model_id == "tabicl_v2":
            options = self._checked_options(
                config,
                common
                | {
                    "n_estimators",
                    "norm_methods",
                    "feat_shuffle_method",
                    "batch_size",
                    "kv_cache",
                    "use_amp",
                    "use_fa3",
                    "offload_mode",
                    "verbose",
                },
            )
            model_path, seed, max_features, scale_targets = self._common_values(
                config, options
            )
            n_estimators = int(options.get("n_estimators", 8))
            if n_estimators < 1:
                raise ValueError("n_estimators must be positive")
            norm_methods = options.get("norm_methods")
            if norm_methods is not None and not isinstance(
                norm_methods, (str, list)
            ):
                raise TypeError("norm_methods must be null, a string, or a list")
            batch_size_value = options.get("batch_size", 8)
            batch_size = (
                None if batch_size_value is None else int(batch_size_value)
            )
            if batch_size is not None and batch_size < 1:
                raise ValueError("batch_size must be positive or null")
            return TabICLPredictiveModel(
                model_id=config.model_id,
                model_path=model_path,
                seed=seed,
                n_estimators=n_estimators,
                norm_methods=norm_methods,
                feat_shuffle_method=str(
                    options.get("feat_shuffle_method", "latin")
                ),
                batch_size=batch_size,
                kv_cache=options.get("kv_cache", False),
                use_amp=options.get("use_amp", "auto"),
                use_fa3=options.get("use_fa3", "auto"),
                offload_mode=options.get("offload_mode", "auto"),
                verbose=options.get("verbose", False),
                max_input_features=max_features,
                scale_targets=scale_targets,
            )

        options = self._checked_options(
            config,
            common
            | {
                "context_size",
                "context_reduction",
                "feature_reduction",
                "n_ensembles",
                "compile",
                "use_flash",
                "verbose",
            },
        )
        model_path, seed, max_features, scale_targets = self._common_values(
            config, options
        )
        context_size = int(options.get("context_size", 0))
        if context_size < 0:
            raise ValueError("context_size must be non-negative")
        if (
            config.model_id in TABDPT_RETRIEVAL_OFF_MODEL_IDS
            and context_size != 0
        ):
            raise ValueError(
                f"{config.model_id} requires context_size=0 (the full context)"
            )
        context_reduction = str(options.get("context_reduction", "subsample"))
        if context_reduction not in {"retrieval", "subsample", "subsample-balanced"}:
            raise ValueError(f"invalid context_reduction: {context_reduction!r}")
        if (
            config.model_id in TABDPT_RETRIEVAL_OFF_MODEL_IDS
            and context_reduction != "subsample"
        ):
            raise ValueError(
                f"{config.model_id} requires context_reduction='subsample'"
            )
        feature_reduction_value = options.get("feature_reduction")
        feature_reduction = (
            None
            if feature_reduction_value is None
            else str(feature_reduction_value)
        )
        if feature_reduction not in {None, "pca", "subsample"}:
            raise ValueError(f"invalid feature_reduction: {feature_reduction!r}")
        if (
            config.model_id == "tabdpt_v1_0_retrieval_off"
            and feature_reduction != "pca"
        ):
            raise ValueError(
                "TabDPT v1.0 requires feature_reduction='pca' to cover all "
                "configured OpenML doses above its 100-feature limit"
            )
        default_n_ensembles = (
            1 if config.model_id == "tabdpt_v1_0_retrieval_off" else 8
        )
        n_ensembles = int(options.get("n_ensembles", default_n_ensembles))
        if n_ensembles < 1:
            raise ValueError("n_ensembles must be positive")
        if config.model_id == "tabdpt_v1_0_retrieval_off" and n_ensembles != 1:
            raise ValueError(
                "TabDPT v1.0 exposes no ensemble API and requires n_ensembles=1"
            )
        compile_model = options.get("compile", True)
        use_flash = options.get("use_flash", True)
        verbose = options.get("verbose", True)
        for name, value in {
            "compile": compile_model,
            "use_flash": use_flash,
            "verbose": verbose,
        }.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        return TabDPTPredictiveModel(
            model_id=config.model_id,
            model_path=model_path,
            context_size=context_size,
            context_reduction=context_reduction,
            feature_reduction=feature_reduction,
            n_ensembles=n_ensembles,
            compile=compile_model,
            use_flash=use_flash,
            verbose=verbose,
            seed=seed,
            max_input_features=max_features,
            scale_targets=scale_targets,
        )


__all__ = [
    "DEFAULT_PRETRAINED_REGRESSION_MODEL_IDS",
    "DEFAULT_TABDPT_PATH",
    "DEFAULT_TABDPT_V10_PATH",
    "DEFAULT_TABDPT_V11_PATH",
    "DEFAULT_TABDPT_V13_PATH",
    "DEFAULT_TABICL_V2_PATH",
    "DEFAULT_TABPFN_V2_PATH",
    "DEFAULT_TABPFN_V25_PATH",
    "DEFAULT_TABPFN_V3_PATH",
    "DefaultPretrainedModelFactory",
    "MODEL_GROUP_1D",
    "MODEL_GROUP_1DICL",
    "MODEL_GROUP_2D",
    "PRETRAINED_MODEL_SPECS",
    "PretrainedModelSpec",
    "TABICL_DEFAULT_PATHS",
    "TABDPT_RETRIEVAL_OFF_MODEL_IDS",
    "TABPFN_DEFAULT_PATHS",
    "TabDPTPredictiveModel",
    "TabDPTBackend",
    "TabICLBackend",
    "TabICLPredictiveModel",
    "TabPFNRegressionBackend",
    "TabPFNPredictiveModel",
    "TargetScaledPredictiveModel",
    "identity_preprocess_transforms",
    "pretrained_model_groups",
]
