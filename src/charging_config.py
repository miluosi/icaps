"""CLI and checkpoint semantics for the optional conservative charging model."""
import argparse


def add_conservative_charging_argument(parser, *, default=False, test=False):
    name = "--test-conservative-charging" if test else "--conservative-charging"
    parser.add_argument(
        name, action=argparse.BooleanOptionalAction, default=default,
        help=("Override the testing charging model; omitted inherits training/checkpoint."
              if test or default is None else
              "Use the all-potential-arrivals conservative charging model (default: off)."),
    )


def charging_checkpoint_suffix(suffix, conservative):
    """Keep legacy directories unchanged and isolate the new training model."""
    suffix = suffix or ""
    return "_".join(part for part in (suffix, "charge-conservative") if part) if conservative else suffix


def resolve_charging_model(requested, checkpoint_identities=()):
    """Resolve inheritance; missing metadata denotes the legacy current model."""
    identities = list(checkpoint_identities)
    saved = {bool(row.get("conservative_charging", False)) for row in identities}
    if len(saved) > 1:
        raise ValueError("EV/AEV checkpoints use different conservative charging models")
    trained = next(iter(saved)) if saved else None
    effective = bool(requested) if requested is not None else bool(trained)
    source = "explicit" if requested is not None else "checkpoint" if identities else "legacy_default"
    return effective, trained, source


def phase_charging_model(args, *, training):
    trained = bool(getattr(args, "conservative_charging", False))
    override = getattr(args, "test_conservative_charging", None)
    return trained if training or override is None else bool(override)
