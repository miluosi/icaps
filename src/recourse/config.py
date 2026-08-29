"""Paper names separate physical repair from leader Bellman credit."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RecourseMethod:
    operating_mode: str
    variant: str
    repair_policy: str
    leader_credit: str


METHODS = {
    "no_repair": RecourseMethod("integrated", "legacy", "none", "uncoupled"),
    "evfirst_no_repair": RecourseMethod("evfirst", "r1", "none", "uncoupled"),
    "repair_only": RecourseMethod("evfirst", "r2", "structured", "uncoupled"),
    "repair_learning": RecourseMethod("evfirst", "r3", "learned", "uncoupled"),
    "recourse_macro": RecourseMethod("evfirst", "recourse_macro", "learned", "macro_realized"),
    "recourse_nested_q2": RecourseMethod("evfirst", "r4", "learned", "nested_follower"),
    "samitha": RecourseMethod("integrated_repair", "legacy", "structured", "macro_realized"),
}
METHODS["recourse_aware"] = METHODS["recourse_macro"]
VARIANT_ALIASES = {name: spec.variant for name, spec in METHODS.items() if spec.operating_mode == "evfirst"}
VARIANT_CHOICES = ("legacy", "r0", "r1", "r2", "r3", "r4", *VARIANT_ALIASES)


def canonical_variant(variant):
    value = str(variant or "legacy").strip().lower()
    return VARIANT_ALIASES.get(value, value)


def target_family(variant, mode="ev_first"):
    if mode == "integrated_repair" or canonical_variant(variant) == "recourse_macro":
        return "macro_realized"
    return "nested_follower" if canonical_variant(variant) == "r4" else "uncoupled"


def validate_joint_learner(mode, variant, value_function_class):
    """Fail closed instead of silently using legacy edge TD for recourse."""
    requires_joint = mode == "integrated_repair" or canonical_variant(variant) in {
        "r2", "r3", "r4", "recourse_macro",
    }
    if requires_joint and not callable(getattr(value_function_class, "_train_joint_step", None)):
        raise ValueError(
            "Repair learning requires a solver-consistent joint critic; "
            "use --learner-variant optimization_anchored_residual"
        )


def method_metadata(mode, variant):
    if isinstance(mode, (list, tuple)):
        mode = mode[0] if len(mode) == 1 else "multiple"
    mode = {"ev_first": "evfirst", "aev_first": "aevfirst"}.get(mode, mode)
    variant = canonical_variant(variant)
    name = next((name for name, spec in METHODS.items()
                 if spec.operating_mode == mode and spec.variant == variant), variant)
    spec = METHODS.get(name)
    return dict(paper_variant_name=name, operating_mode=mode,
                repair_policy=spec.repair_policy if spec else "legacy",
                recourse_target_family=target_family(variant, mode),
                follower_learning=bool(spec and spec.repair_policy == "learned"),
                leader_recourse_credit=target_family(variant, mode) != "uncoupled")


def add_method_arguments(parser):
    parser.add_argument("--recourse-method", choices=tuple(METHODS), default=None,
                        help="Named architecture/repair/credit preset; R1 remains evfirst_no_repair")
    parser.add_argument("--integrated-repair-policy", choices=["limited_hold"], default="limited_hold")
    parser.add_argument("--integrated-repair-hold-enabled", action="store_true", default=True,
                        help="Integrated-repair always includes explicit eligible AEV hold edges")


def resolve_method_arguments(args):
    if args.recourse_method is not None:
        spec = METHODS[args.recourse_method]
        args.transportation_mode = [spec.operating_mode]
        args.recourse_variant = spec.variant
        if args.learner_variant == "legacy":
            args.learner_variant = "optimization_anchored_residual"
        if args.all_modes:
            raise ValueError("--recourse-method cannot be combined with --all-modes")
    args.recourse_variant = canonical_variant(args.recourse_variant)
    if (args.recourse_variant in {'r2', 'r3', 'r4', 'recourse_macro'}
            and args.learner_variant == 'legacy' and args.distribution_mode in {None, 'none'}):
        args.learner_variant = 'optimization_anchored_residual'
    if ('integrated_repair' in args.transportation_mode and args.learner_variant == 'legacy'
            and args.distribution_mode in {None, 'none'}):
        args.learner_variant = 'optimization_anchored_residual'
    return args
