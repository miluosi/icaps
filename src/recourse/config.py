"""Canonical assignment-method registry and experiment-axis configuration."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


LEADER_CREDITS = ("uncoupled", "macro_realized", "nested_follower")
TARGET_SOLVER_POLICIES = (
    "same_as_rollout_exact",
    "fixed_primal_dual_exact",
    "exact_oracle_for_approximate_rollout",
)
OBJECTIVE_POLICY = "execution_reward_with_separate_service_loss_metrics"


@dataclass(frozen=True)
class RecourseMethod:
    operating_mode: str
    variant: str
    repair_policy: str
    leader_credit: str

    def __post_init__(self) -> None:
        if self.leader_credit not in LEADER_CREDITS:
            raise ValueError(f"invalid leader credit: {self.leader_credit}")


# Ordered, publication-facing registry. Aliases must never be inserted here.
PAPER_METHODS = (
    "no_repair",
    "evfirst_no_rejection",
    "evfirst_no_repair",
    "evfirst_no_repair_structured",
    "repair_only",
    "repair_learning",
    "recourse_macro",
    "recourse_nested_q2",
    "samitha",
)
METHODS = {
    "no_repair": RecourseMethod("integrated", "legacy", "none", "uncoupled"),
    "evfirst_no_rejection": RecourseMethod("evfirst", "r0", "none", "uncoupled"),
    "evfirst_no_repair": RecourseMethod("evfirst", "r1", "none", "uncoupled"),
    "evfirst_no_repair_structured": RecourseMethod(
        "evfirst", "r1_structured", "structured", "uncoupled"
    ),
    "repair_only": RecourseMethod("evfirst", "r2", "structured", "uncoupled"),
    "repair_learning": RecourseMethod("evfirst", "r3", "learned", "uncoupled"),
    "recourse_macro": RecourseMethod(
        "evfirst", "recourse_macro", "learned", "macro_realized"
    ),
    "recourse_nested_q2": RecourseMethod(
        "evfirst", "r4", "learned", "nested_follower"
    ),
    "samitha": RecourseMethod(
        "integrated_repair", "legacy", "structured", "macro_realized"
    ),
}
if tuple(METHODS) != PAPER_METHODS:
    raise RuntimeError("PAPER_METHODS and METHODS must have identical order")

METHOD_ALIASES = {
    "recourse_aware": "recourse_macro",
    "recourse-aware": "recourse_macro",
    "integrated": "no_repair",
    "r0": "evfirst_no_rejection",
    "r1": "evfirst_no_repair",
    "structured_r1": "evfirst_no_repair_structured",
    "r1_structured": "evfirst_no_repair_structured",
    "r2": "repair_only",
    "r3": "repair_learning",
    "macro": "recourse_macro",
    "r4": "recourse_nested_q2",
}
VARIANT_ALIASES = {
    name: spec.variant for name, spec in METHODS.items()
    if spec.operating_mode == "evfirst"
}
VARIANT_ALIASES.update(
    {alias: METHODS[name].variant for alias, name in METHOD_ALIASES.items()
     if name in METHODS and METHODS[name].operating_mode == "evfirst"}
)
VARIANT_CHOICES = (
    "legacy", "r0", "r1", "r1_structured", "r2", "r3", "r4",
    "recourse_macro", *PAPER_METHODS,
)


@dataclass(frozen=True)
class AssignmentOracleConfig:
    """Serialized relationship between rollout and target assignment solvers."""

    solver_family: str = "exact"
    backend: str = "primal_dual"
    graph_reduction: bool = True
    verify: bool = True
    cost_scale: int = 10_000
    target_policy: str = "same_as_rollout_exact"
    strict: bool = True

    def __post_init__(self) -> None:
        if self.target_policy not in TARGET_SOLVER_POLICIES:
            raise ValueError(f"invalid target solver policy: {self.target_policy}")
        if int(self.cost_scale) <= 0:
            raise ValueError("assignment cost_scale must be positive")

    @classmethod
    def from_environment(cls, env, *, target_policy: str | None = None):
        rollout_backend = str(getattr(env, "mcmf_backend", "primal_dual"))
        policy = str(
            target_policy
            or getattr(env, "target_solver_policy", "same_as_rollout_exact")
        )
        backend = "primal_dual" if policy == "fixed_primal_dual_exact" else rollout_backend
        if policy == "exact_oracle_for_approximate_rollout":
            backend = str(getattr(env, "target_oracle_backend", "primal_dual"))
        return cls(
            solver_family=str(getattr(env, "mcmf_solver", "exact") or "exact"),
            backend=backend,
            graph_reduction=bool(getattr(env, "mcmf_graph_reduction", True)),
            verify=bool(getattr(env, "mcmf_verify", True)),
            cost_scale=max(1, int(getattr(env, "mcmf_cost_scale", 10_000) or 10_000)),
            target_policy=policy,
            strict=bool(getattr(env, "mcmf_strict", True)),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def canonical_method(name: str) -> str:
    value = str(name).strip().lower().replace("-", "_")
    value = METHOD_ALIASES.get(value, value)
    if value not in METHODS:
        raise ValueError(f"unknown recourse method: {name}")
    return value


def canonical_variant(variant):
    value = str(variant or "legacy").strip().lower().replace("-", "_")
    return VARIANT_ALIASES.get(value, value)


def target_family(variant, mode="ev_first"):
    variant = canonical_variant(variant)
    if mode == "integrated_repair" or variant == "recourse_macro":
        return "macro_realized"
    return "nested_follower" if variant == "r4" else "uncoupled"


def validate_joint_learner(mode, variant, value_function_class):
    """Fail closed instead of silently using legacy edge TD for recourse."""
    requires_joint = mode == "integrated_repair" or canonical_variant(variant) in {
        "r1_structured", "r2", "r3", "r4", "recourse_macro",
    }
    if requires_joint and not callable(getattr(value_function_class, "_train_joint_step", None)):
        raise ValueError(
            "Repair/Macro learning requires a solver-consistent joint critic; "
            "use --learner-variant optimization_anchored_residual"
        )


def method_metadata(mode, variant):
    if isinstance(mode, (list, tuple)):
        mode = mode[0] if len(mode) == 1 else "multiple"
    mode = {"ev_first": "evfirst", "aev_first": "aevfirst"}.get(mode, mode)
    variant = canonical_variant(variant)
    name = next((name for name in PAPER_METHODS
                 if METHODS[name].operating_mode == mode
                 and METHODS[name].variant == variant), variant)
    spec = METHODS.get(name)
    family = spec.leader_credit if spec else target_family(variant, mode)
    return dict(
        paper_variant_name=name,
        operating_mode=mode,
        repair_policy=spec.repair_policy if spec else "legacy",
        leader_credit=family,
        recourse_target_family=family,
        follower_learning=bool(spec and spec.repair_policy == "learned"),
        leader_recourse_credit=family != "uncoupled",
    )


def add_method_arguments(parser):
    parser.add_argument(
        "--recourse-method", choices=(*PAPER_METHODS, *METHOD_ALIASES), default=None,
        help="Named architecture/repair/credit preset; aliases resolve before execution",
    )
    parser.add_argument("--integrated-repair-policy", choices=["limited_hold"], default="limited_hold")
    parser.add_argument(
        "--integrated-repair-hold-enabled", action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Samitha repair-reserve edges; disable only for controlled comparisons",
    )
    parser.add_argument(
        "--target-solver-policy", choices=TARGET_SOLVER_POLICIES,
        default="same_as_rollout_exact",
    )


def resolve_method_arguments(args):
    if args.recourse_method is not None:
        args.recourse_method = canonical_method(args.recourse_method)
        spec = METHODS[args.recourse_method]
        args.transportation_mode = [spec.operating_mode]
        args.recourse_variant = spec.variant
        if args.learner_variant == "legacy" and spec.repair_policy != "none":
            args.learner_variant = "optimization_anchored_residual"
        if args.all_modes:
            raise ValueError("--recourse-method cannot be combined with --all-modes")
    args.recourse_variant = canonical_variant(args.recourse_variant)
    if (args.recourse_variant in {"r1_structured", "r2", "r3", "r4", "recourse_macro"}
            and args.learner_variant == "legacy" and args.distribution_mode in {None, "none"}):
        args.learner_variant = "optimization_anchored_residual"
    if ("integrated_repair" in args.transportation_mode and args.learner_variant == "legacy"
            and args.distribution_mode in {None, "none"}):
        args.learner_variant = "optimization_anchored_residual"
    return args
