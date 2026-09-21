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


def charging_checkpoint_suffix(suffix, conservative=False, *, charging_model=None):
    """Isolate implementations while preserving the existing two namespaces."""
    suffix = suffix or ""
    mode = requested_charging_model(charging_model, conservative)
    tag = {'current': '', 'conservative': 'charge-conservative',
           'real_conservative': 'charge-real-conservative'}[mode]
    existing = set(suffix.split('_')) & {'charge-conservative', 'charge-real-conservative'}
    if existing and existing != {tag}:
        raise ValueError(f'Charging namespace conflicts with selected model {mode}: {suffix}')
    return '_'.join(part for part in (suffix, tag if tag not in existing else '') if part)


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


CHARGING_MODELS = ('current', 'conservative', 'real_conservative')


def add_nyc_charging_model_argument(parser, *, checkpoint=False):
    name = '--checkpoint-charging-model' if checkpoint else '--charging-model'
    parser.add_argument(name, choices=CHARGING_MODELS, default=None,
                        help=('Training checkpoint namespace to read (default: current).'
                              if checkpoint else
                              'NYC charging implementation; overrides the legacy conservative flag. '
                              'Omitted on model evaluation inherits the checkpoint.'))


def requested_charging_model(mode=None, conservative=None):
    if mode is not None:
        if mode not in CHARGING_MODELS:
            raise ValueError(f'Unknown charging model: {mode}')
        return mode
    return None if conservative is None else 'conservative' if conservative else 'current'


def checkpoint_charging_model(identity):
    return requested_charging_model(identity.get('charging_model'),
                                    identity.get('conservative_charging', False))


def resolve_nyc_charging_model(requested, identities=(), checkpoint_hint=None):
    saved = {checkpoint_charging_model(row) for row in identities}
    if len(saved) > 1:
        raise ValueError('EV/AEV checkpoints use different NYC charging models')
    trained = next(iter(saved)) if saved else None
    if trained is not None and checkpoint_hint is not None and trained != checkpoint_hint:
        raise ValueError(f'Checkpoint namespace {checkpoint_hint} disagrees with saved charging model {trained}')
    mode = requested or trained or checkpoint_hint or 'current'
    return mode, trained, 'explicit' if requested is not None else 'checkpoint' if saved else 'legacy_default'


def apply_nyc_charging_model(env, mode, trained=None, source='explicit'):
    if mode == 'real_conservative':
        if not hasattr(env, 'enable_real_conservative'):
            raise ValueError('real_conservative checkpoint requires --checkpoint-charging-model real_conservative')
        if not env.aev_charging_station_ids:
            raise ValueError('real_conservative requires dedicated AEV charging centers (3, 4 or 5)')
    if hasattr(env, 'real_conservative_active'):
        env.real_conservative_active = mode == 'real_conservative'
    env.charging_model = mode
    env.conservative_charging = mode != 'current'
    env.checkpoint_charging_model = trained
    env.checkpoint_conservative_charging = None if trained is None else trained != 'current'
    env.charging_model_source = source
