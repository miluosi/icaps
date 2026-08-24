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
    "bayes": ValueFunctionSpec("src.ValueFunction_pytorch_bayes"),
    "st_masac_gat": ValueFunctionSpec("src.ValueFunction_st_masac_gat"),
    "st_masac_gat_frozen": ValueFunctionSpec("src.ValueFunction_st_masac_gat"),
    "st_masac_gat_neighbour_frozen": ValueFunctionSpec("src.ValueFunction_st_masac_gat"),
    "st_masac_gat_post_demand": ValueFunctionSpec(
        "src.ValueFunction_st_masac_gat_post_demand"
    ),
    "st_masac_gat_post_demand_direct": ValueFunctionSpec(
        "src.ValueFunction_st_masac_gat_post_demand_direct"
    ),
    "optimization_anchored_residual": ValueFunctionSpec(
        "src.ValueFunction_optimization_anchored_residual"
    ),
    "integrated_directq": ValueFunctionSpec("src.ValueFunction_integrated_directq"),
    # `none` is a control condition.  The trainer never constructs this class
    # when ADP is disabled, but mapping it keeps registry validation total.
    "none": ValueFunctionSpec("src.ValueFunction_pytorch_bayes"),
}


VALUE_FUNCTION_CHOICES = tuple(
    key for key, spec in VALUE_FUNCTION_REGISTRY.items() if spec.public
)


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
    loaded = {key: spec.load() for key, spec in VALUE_FUNCTION_REGISTRY.items()}
    if set(VALUE_FUNCTION_CHOICES) != set(VALUE_FUNCTION_REGISTRY):
        raise AssertionError("every registry entry must be deliberately public or internal")
    return loaded

