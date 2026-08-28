from types import SimpleNamespace

import numpy as np
import pytest

from check_nyc_deterministic_rejection import (
    coverage, deterministic_rows, evaluate, fit_if_identifiable, freeze_thresholds, parse_args,
)
from src.NYCEnvironment import NYCEnvironment
from src.acceptance_model import FEATURE_NAMES, FEATURE_VERSION


@pytest.mark.parametrize('q,expected', [(0.0, False), (0.49, False), (0.5, False), (0.51, True), (0.99, True)])
def test_default_deterministic_rejection_uses_no_uniform(q, expected):
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.reject_uniform = False
    env.ifreject = True
    env.recourse_variant = 'legacy'
    env._last_offer_realizations = {}
    env._epoch_id = lambda: 0
    env._calculate_rejection_probabilityreal = lambda vehicle_id, request: q

    def forbidden(*args):
        raise AssertionError('Uniform must not be drawn')

    env._acceptance_uniform = forbidden
    result = env._should_reject_request(0, SimpleNamespace(request_id=1))
    assert bool(result) is expected
    assert env._last_offer_realizations[(0, 0, 1)]['uniform'] == 0.5


def test_latent_score_is_not_confused_with_deterministic_response_probability():
    rows = deterministic_rows([dict(accepted=y, oracle_acceptance_probability=p)
                               for y, p in [(1, 0.8), (1, 0.5), (0, 0.4)]])
    assert [r['oracle_acceptance_probability'] for r in rows] == [1.0, 1.0, 0.0]
    assert [r['latent_acceptance_score'] for r in rows] == [0.8, 0.5, 0.4]
    assert coverage(rows)['rejected'] == 1
    assert coverage(rows)['acceptance_score_below_half'] == 1


@pytest.mark.parametrize('p,y', [(np.nan, 1), (1.1, 1), (-0.1, 0), (0.8, 0), (0.3, 1), (0.5, 0)])
def test_invalid_or_stochastic_rows_fail_closed(p, y):
    with pytest.raises(ValueError):
        deterministic_rows([dict(accepted=y, oracle_acceptance_probability=p)])


def test_single_class_data_does_not_produce_a_fake_trained_detector():
    rows = deterministic_rows([dict(accepted=1, oracle_acceptance_probability=0.8)] * 20)
    model, status, histories = fit_if_identifiable(rows, rows)
    assert model is None and histories == {}
    assert status['status'] == 'not_trainable_single_class'
    assert status['observed_training_labels'] == [1]


def test_two_class_deterministic_data_can_train_and_predict_the_boundary():
    def rows(count, seed):
        x = np.random.default_rng(seed).uniform(-2, 2, (count, 3))
        y = (x[:, 0] - 0.7 * x[:, 1] + 0.3 * x[:, 2] >= 0).astype(int)
        return [dict({name: 0.0 for name in FEATURE_NAMES}, feature_version=FEATURE_VERSION,
                     feature_schema='nyc_minutes', idle_time=row[0] + 2,
                     pickup_time=row[1] + 2, surge_bonus=row[2] + 2,
                     accepted=int(label), oracle_acceptance_probability=float(label))
                for row, label in zip(x, y)]

    train, validation, test = rows(1000, 41), rows(400, 42), rows(400, 43)
    model, status, histories = fit_if_identifiable(train, validation)
    assert status['status'] == 'trained'
    assert histories
    assert np.mean((model.predict_proba(test) >= 0.5) == [r['accepted'] for r in test]) > 0.97


def test_no_rejections_means_recall_undefined_and_no_tuned_threshold():
    class FixedModel:
        def predict_proba(self, rows):
            return np.full(len(rows), 0.8)

    rows = deterministic_rows([dict(accepted=1, oracle_acceptance_probability=0.8)] * 20)
    model = FixedModel()
    choices = freeze_thresholds(rows, {'current': model})['current']
    assert choices['validation_f1'] is None
    metrics = evaluate(rows, model.predict_proba(rows), choices)['classifications']['default']
    assert metrics['accuracy'] == 1
    assert metrics['rejection_recall'] is None
    assert metrics['tp'] == metrics['fp'] == metrics['fn'] == 0


@pytest.mark.parametrize('argv', [
    ['--train-seeds', '1', '--test-seeds', '1'], ['--workers', '0'],
    ['--num-ev', '201'], ['--max-steps', '0'],
])
def test_cli_rejects_invalid_experiments(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
