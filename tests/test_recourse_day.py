from types import SimpleNamespace

import pytest

import run_recourse_audit as audit
from run_recourse_day import parse_args


def test_day_runner_defaults_cover_canonical_methods_and_disjoint_full_days():
    args = parse_args([])
    from src.recourse.config import PAPER_METHODS
    assert tuple(args.methods) == PAPER_METHODS
    assert args.train_date != args.test_date and args.seed != args.test_seed
    assert args.num_vehicles == 200 and args.num_ev == 100
    assert (args.start_hour, args.stop_hour, args.max_steps) == (0., 24., None)
    assert round(86400 / args.epoch_length) == 2880


@pytest.mark.parametrize('arguments', [
    ['--test-date', '2025-12-18'], ['--test-seed', '71'],
    ['--joint-replay-capacity', '1'], ['--smoke-steps', '1'],
])
def test_day_runner_rejects_leakage_and_invalid_budgets(arguments):
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_environment_routes_training_and_testing_to_distinct_dates(monkeypatch):
    seen = []

    def make(settings, seed):
        seen.append((settings.date, settings.start_hour, settings.stop_hour, settings.parquet_path, seed))
        return SimpleNamespace(configure_recourse_experiment=lambda *a, **k: None,
                               set_request_generation_seed=lambda seed: None)

    monkeypatch.setattr(audit, 'make_environment', make)
    monkeypatch.setattr(audit, 'configure_acceptance_feature', lambda *a, **k: None)
    args = parse_args([])
    audit.build_env(args, args.seed, 'recourse_macro', training=True)
    audit.build_env(args, args.test_seed, 'recourse_macro', training=False)
    assert seen[0][:3] == ('2025-12-18', 0., 24.)
    assert seen[1][:3] == ('2025-12-19', 0., 24.)
    assert seen[0][3] == seen[1][3] == args.parquet_path
