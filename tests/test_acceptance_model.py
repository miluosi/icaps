from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import expit

from src.acceptance_model import (
    BinaryAcceptanceModel, FEATURE_NAMES, FEATURE_VERSION, collect_offers, offer_features, probability_metrics,
)
from train_acceptance_model import parse_args


def samples(count=12000, seed=17):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(count, 3))
    probabilities = expit(1.5 - 2.0 * x[:, 0] - 1.0 * x[:, 1] + x[:, 2])
    labels = rng.binomial(1, probabilities)
    return [dict({name: 0.0 for name in FEATURE_NAMES}, feature_version=FEATURE_VERSION,
                 feature_schema="synthetic_steps", idle_time=features[0],
                 pickup_time=features[1], surge_bonus=features[2],
                 accepted=int(label), oracle_acceptance_probability=float(probability))
            for features, label, probability in zip(x, labels, probabilities)]


def test_binary_model_recovers_probabilities_from_labels_and_roundtrips(tmp_path):
    train, test = samples(), samples(3000, seed=18)
    model = BinaryAcceptanceModel(l2=1e-4).fit(train, validation_rows=samples(2000, seed=19))
    predictions = model.predict_proba(test)
    metrics = probability_metrics(test, predictions)
    constant = probability_metrics(test, np.full(len(test), model.training_acceptance_rate))
    assert metrics["oracle_probability_mae"] < 0.045
    assert metrics["log_loss"] < constant["log_loss"]
    assert metrics["brier_score"] < constant["brier_score"]
    assert metrics["roc_auc"] > 0.6
    path = tmp_path / "model.json"
    model.save(path)
    restored = BinaryAcceptanceModel.load(path)
    np.testing.assert_array_equal(predictions, restored.predict_proba(test))
    assert restored.predict_proba([]).shape == (0,)


def test_oracle_draw_ids_and_outcomes_cannot_leak_into_features():
    rows = samples(2500)
    contaminated = [{**row, "oracle_acceptance_probability": 1 - row["accepted"],
                     "random_draw": row["accepted"], "vehicle_id": row["accepted"],
                     "request_id": index, "post_response_battery": row["accepted"]}
                    for index, row in enumerate(rows)]
    model = BinaryAcceptanceModel().fit(rows)
    other = BinaryAcceptanceModel().fit(contaminated)
    assert model.to_dict() == other.to_dict()
    tampered_labels = [{**row, "accepted": 1 - row["accepted"]} for row in contaminated]
    np.testing.assert_array_equal(model.predict_proba(rows), model.predict_proba(tampered_labels))


def test_natural_class_ratio_is_preserved():
    rows = [dict({name: 0.0 for name in FEATURE_NAMES}, feature_version=FEATURE_VERSION,
                 feature_schema="synthetic_steps", idle_time=1, pickup_time=2,
                 surge_bonus=0, accepted=int(index < 850), oracle_acceptance_probability=0.85)
            for index in range(1000)]
    model = BinaryAcceptanceModel().fit(rows)
    assert model.predict_proba(rows).mean() == pytest.approx(0.85, abs=1e-5)


def fake_env(*, nyc=False):
    env = SimpleNamespace(vehicles={0: {"type": 1, "idle_timer": 12, "location": 3},
                                   1: {"type": 2, "idle_timer": 0, "location": 3}},
                          current_time=5, EPOCH_LENGTH=30,
                          grid_size=10, simulation_period=100,
                          zone_coords={3: (40.7, -74.0), 7: (40.8, -73.9)},
                          get_distance_km=lambda a, b: abs(b - a),
                          _manhattan_distance_loc=lambda a, b: abs(b - a),
                          _request_surge_bonus=lambda request: 2.0,
                          _epoch_id=lambda: 5, _last_offer_realizations={}, draws=0)
    if nyc:
        env.get_travel_time_minutes = lambda a, b: 7.5
    return env


def fake_request():
    return SimpleNamespace(pickup=7, dropoff=3, request_id=8, value=10.0, final_value=12.0,
                           created_time=0.0, pickup_deadline=20.0, dropoff_deadline=30.0, travel_time=10.0)


def test_feature_units_follow_each_environment():
    request = fake_request()
    synthetic = offer_features(fake_env(), 0, request)
    nyc = offer_features(fake_env(nyc=True), 0, request)
    for row, schema, idle, pickup in [(synthetic, 'synthetic_steps', 12, 4), (nyc, 'nyc_minutes', 6, 7.5)]:
        assert row['feature_schema'] == schema and row['idle_time'] == idle
        assert row['pickup_time'] == pickup and row['surge_bonus'] == 2
        assert set(row) == set(FEATURE_NAMES) | {'feature_schema', 'feature_version'}
    assert synthetic['trip_time'] == 10 and nyc['trip_time'] == 5


def test_collector_preserves_response_and_snapshots_before_mutation():
    env = fake_env()
    request = fake_request()

    def original(vid, req):
        env.draws += 1
        env.vehicles[vid]["idle_timer"] = 0
        env._last_offer_realizations[(5, vid, req.request_id)] = {
            "acceptance_probability": 0.8, "rejected": False}
        return False

    env._should_reject_request = original
    with collect_offers(env, episode_id="train:1", seed=1) as rows:
        assert env._should_reject_request(0, request) is False
        assert env._should_reject_request(1, request) is False
    assert env.draws == 2
    assert env._should_reject_request is original
    assert len(rows) == 1
    assert rows[0]["idle_time"] == 12
    assert rows[0]["accepted"] == 1
    assert rows[0]["oracle_acceptance_probability"] == 0.8


def test_live_prediction_and_batch_prediction_agree():
    model = BinaryAcceptanceModel().fit(samples(1000))
    env = fake_env()
    request = fake_request()
    expected = model.predict_proba([offer_features(env, 0, request)])[0]
    assert model.predict_acceptance_probability(env, 0, request) == expected
    assert model.predict_rejection_probability(env, 0, request) == 1 - expected
    assert model.predict_acceptance_probability(env, 1, request) == 1
    assert model.predict_rejection_probability(env, 1, request) == 0
    with pytest.raises(ValueError, match="schema"):
        model.predict_acceptance_probability(fake_env(nyc=True), 0, request)


def test_metrics_do_not_confuse_acceptance_with_rejection():
    rows = [dict(accepted=0, oracle_acceptance_probability=0.1),
            dict(accepted=1, oracle_acceptance_probability=0.9)]
    metrics = probability_metrics(rows, [0.1, 0.9])
    assert metrics["roc_auc"] == 1
    assert metrics["brier_score"] == pytest.approx(0.01)
    assert metrics["oracle_probability_mae"] == 0
    assert probability_metrics(rows, [0.5, 0.5])["roc_auc"] == 0.5


def test_split_validation_rejects_overlap_and_single_class_fit():
    with pytest.raises(SystemExit):
        parse_args(["--train-seeds", "1", "--test-seeds", "1"])
    with pytest.raises(ValueError, match="both accepted and rejected"):
        BinaryAcceptanceModel().fit([{**row, "accepted": 1} for row in samples(10)])
    with pytest.raises(RuntimeError, match="not been trained"):
        BinaryAcceptanceModel().predict_proba(samples(1))


def test_neural_architecture_rng_isolation_and_legacy_checkpoint_rejection():
    before = torch.random.get_rng_state().clone()
    model = BinaryAcceptanceModel(max_epochs=5).fit(samples(300))
    restored = BinaryAcceptanceModel.from_dict(model.to_dict())
    restored.predict_proba(samples(10))
    assert torch.equal(before, torch.random.get_rng_state())
    layers = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [(30, 64), (64, 32), (32, 1)]
    assert not any(p.requires_grad for p in restored.network.parameters())
    with pytest.raises(ValueError, match='Legacy logistic-regression'):
        BinaryAcceptanceModel.from_dict({'version': 1})
    with pytest.raises(ValueError, match='recollect'):
        model.predict_proba([{k: v for k, v in samples(1)[0].items() if k != 'feature_version'}])
    with pytest.raises(ValueError, match='Missing required'):
        model.predict_proba([{k: v for k, v in samples(1)[0].items() if k != 'battery_level'}])


def test_network_can_learn_nonlinear_interactions():
    def nonlinear(count, seed):
        rows = samples(count, seed)
        rng = np.random.default_rng(seed + 10)
        for row in rows:
            probability = float(expit(24 * (row['idle_time'] - .5) * (row['pickup_time'] - .5)))
            row['accepted'] = int(rng.random() < probability)
            row['oracle_acceptance_probability'] = probability
        return rows
    model = BinaryAcceptanceModel().fit(nonlinear(4000, 81), validation_rows=nonlinear(1000, 82))
    test = nonlinear(1000, 83)
    assert probability_metrics(test, model.predict_proba(test))['log_loss'] < 0.55


@pytest.mark.parametrize('corruption', ['nonfinite_weight', 'wrong_layer_size', 'old_features', 'zero_scale'])
def test_neural_checkpoint_rejects_corrupt_or_incompatible_parameters(corruption):
    model = BinaryAcceptanceModel(max_epochs=2).fit(samples(100))
    state = model.to_dict()
    if corruption == 'nonfinite_weight':
        state['network_state']['0.weight'][0][0] = float('nan')
    elif corruption == 'wrong_layer_size':
        state['network_state']['0.weight'].pop()
    elif corruption == 'old_features':
        state['feature_names'] = ['idle_time', 'pickup_time', 'surge_bonus']
    else:
        state['scale'][0] = 0
    with pytest.raises(ValueError):
        BinaryAcceptanceModel.from_dict(state)
