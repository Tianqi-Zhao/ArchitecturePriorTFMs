"""Registry for the two controlled architectures used in the paper.

Training and evaluation code depend on :class:`ArchitectureSpec` rather than
on concrete model classes. Each registered architecture defines construction,
checkpoint reconstruction, classifier wrapping, and input-width policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

from torch import nn

from models.model_1d import NanoTabDPTClassifier, NanoTabDPTModel
from models.model_2d import NanoTabPFNClassifier, NanoTabPFNModel


InputPolicy = Literal["raw_features", "fixed_width"]
ShufflePolicy = Literal["never", "requested"]
ModelBuilder = Callable[[Mapping[str, Any]], nn.Module]

ARCHITECTURE_REGISTRY_VERSION = 1


@dataclass(frozen=True)
class ArchitectureSpec:
    """Runtime contract for one paper architecture."""

    name: str
    builder: ModelBuilder = field(repr=False)
    model_class: type[nn.Module]
    classifier_class: type
    defaults: Mapping[str, Any]
    input_policy: InputPolicy
    shuffle_policy: ShufflePolicy
    permutation_invariant: bool
    display_order: int = 0

    def build(self, overrides: Mapping[str, Any] | None = None) -> "BuiltArchitecture":
        config = dict(self.defaults)
        if overrides:
            unknown = set(overrides) - set(self.defaults)
            if unknown:
                raise ValueError(
                    f"architecture {self.name!r} received unknown config keys: "
                    f"{sorted(unknown)}"
                )
            config.update(
                key_value
                for key_value in overrides.items()
                if key_value[1] is not None
            )
        return BuiltArchitecture(
            spec=self,
            model=self.builder(config),
            config=config,
        )

    def training_shuffle(self, requested: bool) -> bool:
        return bool(requested) if self.shuffle_policy == "requested" else False

    def build_from_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        num_outputs: int | None = None,
    ) -> "BuiltArchitecture":
        nested = checkpoint.get("model_config")
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise ValueError("checkpoint model_config must be a mapping")
            overrides = {key: nested[key] for key in self.defaults if key in nested}
        else:
            # Older paper checkpoints stored model hyperparameters at the top
            # level instead of under ``model_config``.
            overrides = {
                key: checkpoint[key]
                for key in self.defaults
                if key in checkpoint and checkpoint[key] is not None
            }
        if num_outputs is not None:
            overrides["num_outputs"] = int(num_outputs)
        return self.build(overrides)


@dataclass(frozen=True)
class BuiltArchitecture:
    """A concrete model and the serializable configuration that built it."""

    spec: ArchitectureSpec
    model: nn.Module
    config: Mapping[str, Any]

    @property
    def classifier_class(self) -> type:
        return self.spec.classifier_class

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "architecture_registry_version": ARCHITECTURE_REGISTRY_VERSION,
            "arch": self.spec.name,
            "model_config": dict(self.config),
            **dict(self.config),
        }


_ARCHITECTURES: dict[str, ArchitectureSpec] = {}


def register_architecture(spec: ArchitectureSpec) -> None:
    if not spec.name:
        raise ValueError("architecture name must be non-empty")
    if spec.name in _ARCHITECTURES:
        raise ValueError(f"architecture {spec.name!r} is already registered")
    _ARCHITECTURES[spec.name] = spec


def get_architecture(name: str) -> ArchitectureSpec:
    try:
        return _ARCHITECTURES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown architecture {name!r}; available: {list_architectures()}"
        ) from error


def iter_architecture_specs() -> tuple[ArchitectureSpec, ...]:
    specs = sorted(
        _ARCHITECTURES.values(),
        key=lambda spec: (spec.display_order, spec.name),
    )
    return tuple(specs)


def list_architectures() -> tuple[str, ...]:
    return tuple(spec.name for spec in iter_architecture_specs())


def classifier_class_for_model(model: nn.Module) -> type:
    matches = [
        spec.classifier_class
        for spec in _ARCHITECTURES.values()
        if isinstance(model, spec.model_class)
    ]
    unique = list(dict.fromkeys(matches))
    if not unique:
        raise ValueError(f"model type {type(model).__name__!r} is not registered")
    if len(unique) != 1:
        raise ValueError(
            f"model type {type(model).__name__!r} maps to multiple classifier wrappers"
        )
    return unique[0]


_COMMON_DEFAULTS = {
    "embedding_size": 96,
    "num_attention_heads": 4,
    "mlp_hidden_size": 192,
    "num_outputs": 1,
}


register_architecture(ArchitectureSpec(
    name="1d",
    builder=lambda config: NanoTabDPTModel(**dict(config)),
    model_class=NanoTabDPTModel,
    classifier_class=NanoTabDPTClassifier,
    defaults={**_COMMON_DEFAULTS, "num_features": 5, "num_layers": 5},
    input_policy="fixed_width",
    shuffle_policy="requested",
    permutation_invariant=False,
    display_order=10,
))

register_architecture(ArchitectureSpec(
    name="2d",
    builder=lambda config: NanoTabPFNModel(**dict(config)),
    model_class=NanoTabPFNModel,
    classifier_class=NanoTabPFNClassifier,
    defaults={**_COMMON_DEFAULTS, "num_layers": 3},
    input_policy="raw_features",
    shuffle_policy="never",
    permutation_invariant=True,
    display_order=20,
))
