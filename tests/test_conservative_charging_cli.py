from types import SimpleNamespace
import inspect

import pytest
import torch

import run_nyctrainer
import run_recourse_audit
import run_recourse_day
import test_all_nyc_models
import test_nyc_model
from src.ADPtrainer import ADPTrainer
from src.charging_config import charging_checkpoint_suffix, phase_charging_model, resolve_charging_model


def test_cli_preserves_legacy_training_and_inherits_on_evaluation():
    assert run_nyctrainer.parse_args([]).conservative_charging is False
    assert test_nyc_model.parse_args([]).conservative_charging is None
    args = test_nyc_model.parse_args(['--checkpoint-conservative-charging', '--no-conservative-charging'])
    assert args.checkpoint_conservative_charging is True
    assert args.conservative_charging is False
    assert run_nyctrainer.parse_args(['--conservative-charging']).conservative_charging is True


@pytest.mark.parametrize('requested,saved,expected', [
    (None, [], (False, None, 'legacy_default')),
    (None, [{}, {}], (False, False, 'checkpoint')),
    (None, [{'conservative_charging': True}] * 2, (True, True, 'checkpoint')),
    (False, [{'conservative_charging': True}] * 2, (False, True, 'explicit')),
    (True, [{}] * 2, (True, False, 'explicit')),
])
def test_checkpoint_inheritance_and_independent_override(requested, saved, expected):
    assert resolve_charging_model(requested, saved) == expected


def test_mixed_checkpoint_pair_is_rejected_even_with_override():
    with pytest.raises(ValueError, match='different conservative charging'):
        resolve_charging_model(True, [{}, {'conservative_charging': True}])


def test_training_dispatch_keeps_conservative_checkpoints_separate(monkeypatch):
    calls = []
    monkeypatch.setattr(run_nyctrainer, 'run_nyc_training', lambda **kw:
                        (calls.append(kw) or {}, SimpleNamespace(parquet_path=())))
    run_nyctrainer.main(['--methods', 'r1', '--episodes', '1', '--start-date', '2025-12-18',
                        '--conservative-charging', '--checkpoint-suffix', 'experiment'])
    assert calls[0]['conservative_charging'] is True
    assert 'experiment_charge-conservative_method-r1' in calls[0]['checkpoint_suffix']
    assert charging_checkpoint_suffix('experiment', False) == 'experiment'


def test_nyc_factory_forwards_flag_to_real_constructor_contract(monkeypatch):
    received = []
    monkeypatch.setattr(run_nyctrainer, 'NYCEnvironment', lambda **kw:
                        (received.append(kw) or SimpleNamespace(configure_recourse_experiment=lambda *a, **k: None)))
    signature = inspect.signature(run_nyctrainer._create_nyc_environment)
    kwargs = {name: None for name, parameter in signature.parameters.items()
              if parameter.default is inspect.Parameter.empty}
    kwargs.update(start_hour=0, stop_hour=1, epoch_length=30, conservative_charging=True)
    run_nyctrainer._create_nyc_environment(**kwargs)
    assert received[0]['conservative_charging'] is True


def test_checkpoint_save_and_identity_record_model(tmp_path):
    vf = SimpleNamespace(env=SimpleNamespace(conservative_charging=True),
                         network=torch.nn.Linear(1, 1), target_network=torch.nn.Linear(1, 1),
                         optimizer=None)
    paths = ADPTrainer._save_q_network_checkpoint(vf, 1, str(tmp_path))
    assert paths
    identity = ADPTrainer._checkpoint_identity(paths['full_state'])
    assert identity['conservative_charging'] is True
    payload = torch.load(paths['full_state'], map_location='cpu', weights_only=False)
    del payload['conservative_charging']
    torch.save(payload, paths['full_state'])
    assert ADPTrainer._checkpoint_identity(paths['full_state'])['conservative_charging'] is False


def test_train_test_flags_survive_subprocess_roundtrip_and_control_env(monkeypatch):
    args = test_all_nyc_models.parse_args(['train-test', '--conservative-charging',
                                         '--no-test-conservative-charging', '--dry-run'])
    settings = run_recourse_day.parse_args(test_all_nyc_models.engine_arguments(args))
    assert settings.conservative_charging is True
    assert settings.test_conservative_charging is False
    monkeypatch.setattr(run_recourse_audit, 'make_environment', lambda *a:
                        SimpleNamespace(configure_recourse_experiment=lambda *a, **k: None))
    monkeypatch.setattr(run_recourse_audit, 'configure_acceptance_feature', lambda *a, **k: None)
    train_env = run_recourse_audit.build_env(settings, settings.seed, 'recourse_macro', training=True)
    test_env = run_recourse_audit.build_env(settings, settings.test_seed, 'recourse_macro', training=False)
    assert train_env.conservative_charging is True
    assert test_env.conservative_charging is False
    settings.test_conservative_charging = None
    assert phase_charging_model(settings, training=False) is True


def test_test_only_override_does_not_default_to_false():
    args = test_all_nyc_models.parse_args(['test-only', '--source-dir', 'unused'])
    assert args.conservative_charging is None
    args = test_all_nyc_models.parse_args(['test-only', '--source-dir', 'unused', '--no-conservative-charging'])
    assert args.conservative_charging is False
