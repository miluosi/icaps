"""Validated, lazy value-function registry used by CLI entry points."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ValueFunctionSpec:
    module: str
    class_name: str = "PyTorchChargingValueFunction"
    public: bool = True

    def load(self) -> type:
        module = import_module(self.module)
        value_function_class = getattr(module, self.class_name)
        if not isinstance(value_function_class, type):
            raise TypeError(
                f"{self.module}.{self.class_name} is not a value-function class"
            )
        return value_function_class


VALUE_FUNCTION_REGISTRY: dict[str, ValueFunctionSpec] = {
    "optimization_anchored_residual": ValueFunctionSpec(
        "src.ValueFunction_optimization_anchored_residual"
    ),
    "integrated_directq": ValueFunctionSpec("src.ValueFunction_integrated_directq"),
    # Internal ADP=0 compatibility sentinel. It is deliberately not a public
    # learning choice; public CLIs expose only the two learners above.
    "none": ValueFunctionSpec("src.ValueFunction_pytorch_bayes", public=False),
}


VALUE_FUNCTION_CHOICES = tuple(
    key for key, spec in VALUE_FUNCTION_REGISTRY.items() if spec.public
)
DEFAULT_VALUE_FUNCTION = "optimization_anchored_residual"


def resolve_value_function_mode(
    learner_variant: str | None,
    distribution_mode: str | None,
) -> str:
    """Resolve the compatibility flags to one of the two retained learners."""
    learner = None if learner_variant is None else str(learner_variant).strip().lower()
    distribution = (
        None if distribution_mode is None else str(distribution_mode).strip().lower()
    )
    for label, value in (("learner variant", learner), ("distribution mode", distribution)):
        if value is not None and value not in VALUE_FUNCTION_CHOICES:
            raise ValueError(
                f"unknown {label} {value!r}; choose one of "
                f"{', '.join(VALUE_FUNCTION_CHOICES)}"
            )
    if learner is not None and distribution is not None and learner != distribution:
        raise ValueError(
            "--learner-variant and --distribution-mode must select the same learner"
        )
    return learner or distribution or DEFAULT_VALUE_FUNCTION


def get_value_function_class(distribution_mode: str) -> type:
    key = str(distribution_mode or "none").strip().lower()
    try:
        spec = VALUE_FUNCTION_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown distribution mode {distribution_mode!r}; "
            f"choose one of {', '.join(VALUE_FUNCTION_CHOICES)}"
        ) from exc
    return spec.load()


def validate_value_function_registry() -> dict[str, type]:
    public = {key: spec for key, spec in VALUE_FUNCTION_REGISTRY.items() if spec.public}
    loaded = {key: spec.load() for key, spec in public.items()}
    if set(VALUE_FUNCTION_CHOICES) != set(public):
        raise AssertionError("public registry choices are inconsistent")
    return loaded
