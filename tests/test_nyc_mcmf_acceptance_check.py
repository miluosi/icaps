import numpy as np
import pytest
from scipy.special import expit
from types import SimpleNamespace

from check_nyc_mcmf_acceptance import (
    environment_configuration, loss_trace, parse_args, rejection_metrics, select_rejection_threshold,
    verify_offer_response,
)
from src.acceptance_model import BinaryAcceptanceModel, FEATURE_NAMES, FEATURE_VERSION
from src.NYCEnvironment import NYCEnvironment


def samples(count, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(count, 3))
    p = expit(0.8 + x @ np.array([-0.9, -0.5, 1.2]))
    y = rng.binomial(1, p)
    return [dict({name: 0.0 for name in FEATURE_NAMES}, feature_version=FEATURE_VERSION,
                 feature_schema='nyc_minutes', idle_time=row[0], pickup_time=row[1],
                 surge_bonus=row[2], accepted=int(label), oracle_acceptance_probability=float(probability))
            for row, label, probability in zip(x, y, p)]


def test_rejection_is_the_positive_class_and_average_precision_handles_ties():
    rows = [dict(accepted=y) for y in [0, 0, 1, 1]]
    m = rejection_metrics(rows, [0.1, 0.6, 0.2, 0.9])
    assert [m[k] for k in ['tp', 'fp', 'fn', 'tn']] == [1, 1, 1, 1]
    assert m['rejection_recall'] == m['rejection_precision'] == m['accuracy'] == 0.5
    assert m['rejection_average_precision'] == pytest.approx(5 / 6)
    constant = rejection_metrics(rows, [0.8] * 4)
    assert constant['rejection_average_precision'] == 0.5
    assert constant['rejection_precision'] is None
    assert constant['rejection_recall'] == 0
    assert constant['predicted_rejections'] == 0


def test_validation_threshold_can_identify_rare_rejections_below_point_five():
    rows = [dict(accepted=y) for y in [0, 1, 0, 1]]
    p = [0.7, 0.9, 0.8, 0.95]
    threshold, metrics = select_rejection_threshold(rows, p)
    assert threshold == pytest.approx(0.2)
    assert metrics['rejection_f1'] == 1
    assert metrics['tp'] == 2 and metrics['fp'] == 0
    assert rejection_metrics(rows, p)['rejection_recall'] == 0


def test_real_optimizer_loss_history_and_checkpoint_compatibility(tmp_path):
    train, validation = samples(1400, 42), samples(500, 43)
    model = BinaryAcceptanceModel(l2=0.001).fit(train, validation_rows=validation)
    trace = loss_trace(model, validation)
    assert len(trace) > 2
    assert len(trace) == model.epochs_run + 1
    assert trace[-1]['objective'] < trace[0]['objective']
    assert trace[model.selected_epoch]['objective'] == pytest.approx(model.fit_loss)
    for row in trace:
        assert row['objective'] == pytest.approx(row['binary_cross_entropy'] + row['l2_penalty'])
    assert trace[0]['l2_penalty'] > 0
    path = tmp_path / 'model.json'
    model.save(path)
    restored = BinaryAcceptanceModel.load(path)
    assert 'loss_history' not in restored.to_dict()
    np.testing.assert_array_equal(restored.predict_proba(validation), model.predict_proba(validation))


@pytest.mark.parametrize('p', [[0.5], [0.2, np.nan], [0.1, 1.1]])
def test_metrics_fail_closed_for_bad_inputs(p):
    with pytest.raises(ValueError):
        rejection_metrics([dict(accepted=0), dict(accepted=1)], p)


@pytest.mark.parametrize('argv', [
    ['--train-seeds', '1', '--test-seeds', '1'],
    ['--workers', '0'], ['--max-steps', '0'], ['--num-vehicles', '10', '--num-ev', '20'],
    ['--expected-assignment-range-km', 'nan'], ['--expected-assignment-range-km', '0'], ['--nn-epochs', '0'],
])
def test_standalone_cli_rejects_invalid_experiments(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


@pytest.mark.parametrize('uniform,rejected', [(0.1, True), (0.9, False)])
def test_actual_stochastic_response_can_reject_below_half(uniform, rejected):
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.reject_uniform, env.ifreject, env.recourse_variant = True, True, 'legacy'
    env._last_offer_realizations = {}
    env._epoch_id = lambda: 0
    env._calculate_rejection_probabilityreal = lambda *args: .2
    env._acceptance_uniform = lambda *args: uniform
    response = env._should_reject_request(0, SimpleNamespace(request_id=1))
    assert bool(response) is rejected
    configuration = dict(reject_uniform=True, ifreject=True, use_range_requests=True, assignmentrange=2.0)
    diagnostic = verify_offer_response(env._last_offer_realizations[(0, 0, 1)], response, configuration, 1.8)
    assert diagnostic['response_uniform'] == uniform
    with pytest.raises(AssertionError, match='outside'):
        verify_offer_response(env._last_offer_realizations[(0, 0, 1)], response, configuration, 2.1)


def test_experiment_guard_checks_configuration_without_overriding_it():
    env = SimpleNamespace(reject_uniform=True, ifreject=True, ride_acceptance_noise_std=0.0,
        rejection_logit_shift=0.0, use_range_requests=True, assignmentrange=2.0,
        ride_acceptance_asc=1.810, ride_acceptance_beta_idle_min=-.017,
        ride_acceptance_beta_pickup_min=-.050, ride_acceptance_beta_surge=.101)
    args = parse_args(['--require-random-rejection', '--expected-assignment-range-km', '2'])
    assert environment_configuration(env, args)['assignmentrange'] == 2
    env.reject_uniform = False
    with pytest.raises(AssertionError, match='stochastic'):
        environment_configuration(env, args)
    assert env.reject_uniform is False
