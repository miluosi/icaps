"""Executable acceptance criteria for the two rejection/residual change notes.

These tests exercise one-stage training only. They do not claim convergence or
cross-date generalization from a same-day NYC smoke experiment.
"""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.acceptance_inputs import offer_features, FEATURE_NAMES, CONTEXT_FEATURE_NAMES
from src.acceptance_model import EVRejectionProbabilityModel, collect_offers
from src.acceptance_features import configure_acceptance_feature
from src.rejection_anchor import expected_structured_score, rejection_score
from src.rejection_collection import mixed_feasible_offers, parse_mixture
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.recourse.target_builder import RecourseTargetBuilder, TargetComponents
from src.recourse.replay import PrioritizedJointReplayBuffer
from src.recourse.types import ActionType, JointActionSnapshot
from src.NYCEnvironment import NYCEnvironment
from train_acceptance_model import parse_args
from test_acceptance_model import samples, fake_env, fake_request
from test_acceptance_learning import env, make_value


def test_restricted_information_does_not_read_hidden_offer_or_market_fields():
    request = SimpleNamespace(pickup=7)  # No dropoff, fare, deadline, trip time.
    environment = fake_env()
    environment.active_requests = object()  # Not even an iterable demand view.
    row = offer_features(environment, 0, request)
    assert set(row) == set(FEATURE_NAMES) | {'feature_schema', 'feature_version', 'feature_variant'}
    assert row['feature_variant'] == 'driver_offer_core'


def test_context_ablation_is_separate_and_mismatched_schema_fails():
    environment, request = fake_env(), fake_request()
    row = offer_features(environment, 0, request, feature_variant='platform_context')
    assert set(CONTEXT_FEATURE_NAMES).issubset(row)
    rows = [dict(row, rejected=i % 2) for i in range(80)]
    model = EVRejectionProbabilityModel(feature_variant='platform_context', max_epochs=2).fit(rows)
    assert model.network[0].in_features == 30
    with pytest.raises(ValueError, match='variant'):
        model.predict_proba([offer_features(environment, 0, request)])


@pytest.mark.parametrize('method', ['temperature', 'platt'])
def test_validation_calibration_and_saved_inference(method):
    train, validation = samples(600, 301), samples(200, 302)
    model = EVRejectionProbabilityModel(max_epochs=8, calibration=method).fit(train, validation_rows=validation)
    calibration = model.calibration
    assert calibration['fitted'] and calibration['a'] > 0
    assert calibration['validation_nll_after'] <= calibration['validation_nll_before'] + 1e-12
    z = model.predict_logits(validation)
    y = np.asarray([row['rejected'] for row in validation])
    calibrated_logits = calibration['a'] * z + calibration['b']
    loss = np.mean(np.logaddexp(0, calibrated_logits) - y * calibrated_logits)
    assert loss == pytest.approx(calibration['validation_nll_after'])
    restored = EVRejectionProbabilityModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(model.predict_proba(validation), restored.predict_proba(validation))
    assert model.predictor_hash == restored.predictor_hash
    # Held-out data cannot influence the normalization statistics.
    np.testing.assert_allclose(model.mean, np.array([[r[n] for n in FEATURE_NAMES] for r in train]).mean(0))


def test_nyc_minutes_live_and_snapshot_core_are_identical():
    environment = fake_env(nyc=True)
    request = fake_request()
    snapshot = SimpleNamespace(current_time=5, vehicles=(SimpleNamespace(
        vehicle_id=0, vehicle_type=1, idle_time=12, location=3),))
    row = offer_features(environment, 0, request)
    assert row == offer_features(environment, 0, request, snapshot=snapshot)
    environment.vehicles[0].update(idle_timer=999, location=99)
    assert row == offer_features(environment, 0, request, snapshot=snapshot)
    assert row['idle_time'] == 6 and row['pickup_time'] == 7.5


def test_q_endpoints_monotonicity_masks_and_branch_monte_carlo():
    success, rejected = 20., -5.
    assert expected_structured_score(success, rejected, 0., True) == success
    assert expected_structured_score(success, rejected, 1., True) == rejected
    scores = [expected_structured_score(success, rejected, q, True) for q in np.linspace(0, 1, 21)]
    assert np.all(np.diff(scores) < 0)
    assert expected_structured_score(success, rejected, 0., False) == success
    with pytest.raises(ValueError):
        expected_structured_score(success, rejected, .2, False)
    # Monte Carlo of the SAME option-score branches, not epoch cash rewards.
    q = .3
    observed = np.where(np.random.default_rng(71).random(200000) < q, rejected, success)
    assert observed.mean() == pytest.approx(expected_structured_score(success, rejected, q, True), abs=.08)


@pytest.mark.parametrize('ratio', [.25, None])
def test_nyc_execution_and_anchor_share_exact_rejection_penalty(ratio):
    environment = SimpleNamespace(vehicles={0: {'location': 1}},
        rejection_penalty_final_value_ratio=ratio, rejection_penalty_base=4., rejection_penalty_per_km=.35,
        get_distance_km=lambda a, b: 2.)
    request = SimpleNamespace(pickup=3, final_value=20.)
    actual = NYCEnvironment._calculate_rejection_reward(environment, 0, request)
    assert actual == rejection_score(environment, request_value=20., pickup_distance=2.)
    assert actual == pytest.approx(-5. if ratio is not None else -4.7)
    assert rejection_score(SimpleNamespace(), request_value=20., pickup_distance=2.) == 0.


def graph_for(env, value):
    env.value_function = env.value_function_ev = value
    ids = [next(k for k, v in env.vehicles.items() if v['type'] == kind) for kind in (1, 2)]
    env._last_matrix_request_ids = [991]
    env._last_matrix_charge_station_ids = []
    env._last_matrix_zone_indices = []
    env._last_matrix_zone_target_ids = []
    state = StateSnapshotBuilder.build(env)
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(env, ids, np.ones((2, 2)),
        np.zeros((2, 2)), np.full((2, 2), 999.), num_requests=1, num_stations=0, num_zones=0,
        stage_id=0, solver_backend='primal_dual', state=state)
    return graph


def live_score(value, graph, edge, q=None):
    vehicle = next(v for v in graph.state.vehicles if v.vehicle_id == edge.vehicle_id)
    return value.batch_get_mixed_q_values(vehicle_ids=[edge.vehicle_id],
        vehicle_locations=[vehicle.location], target_locations=[edge.target_location],
        current_times=[graph.state.current_time], other_vehicles=[len(graph.state.vehicles) - 1],
        num_requests=[len(graph.state.requests)], battery_levels=[vehicle.battery],
        request_values=[edge.request_value], target_distances=[edge.target_distance],
        target_zoneids=[edge.target_zoneid], vehicle_idle_times=[vehicle.idle_time],
        action_type_ids=[int(edge.action_type)], post_action_distances=[edge.post_action_distance],
        post_action_durations=[edge.post_action_duration], post_action_locations=[edge.post_action_location],
        request_ids=[edge.request_id or -1], rejection_probabilities=None if q is None else [q])[0]


def test_live_snapshot_replay_anchor_and_beta_zero_risk_effect(env):
    value = make_value(env, 'optimization_anchored_residual')
    graph = graph_for(env, value)
    ev = next(e for e in graph.edges if e.vehicle_type == 1 and e.action_type == ActionType.SERVICE)
    assert value._beta() == 0.
    for edge in graph.edges:
        if not dict(edge.metadata).get('continuing', False):
            assert live_score(value, graph, edge) == pytest.approx(edge.structured_score, abs=1e-5)
        else:
            # Continuing edges are not new dispatch decisions; their existing
            # remaining-action surrogate is retained, with no response risk.
            assert edge.structured_score == edge.success_structured_score
        if not (edge.vehicle_type == 1 and edge.action_type == ActionType.SERVICE):
            assert not edge.human_response_mask and edge.rejection_probability == 0.
    assert live_score(value, graph, ev, q=0.) > live_score(value, graph, ev, q=1.)
    before = value._edge_experience(graph, ev, state_variant=value.state_variant)
    tensor, _, score = value._edge_tensor_from_experience(before)
    assert score == pytest.approx(ev.structured_score)
    assert tensor[0, value.rejection_input_index + 1] == 1
    # Mutating live time, demand and driver state must not change old replay.
    env.current_time += 100
    env.vehicles[ev.vehicle_id]['assigned_request'] = 991
    env.vehicles[ev.vehicle_id]['idle_timer'] = 9999
    env.active_requests.clear()
    repeated, _, repeated_score = value._edge_tensor_from_experience(before)
    assert torch.equal(tensor, repeated) and repeated_score == score
    with pytest.raises(ValueError, match='hash'):
        value._graph_edge_scores(replace(graph, edges=(replace(ev, response_model_hash='bad'),)), target_context=True)


def test_continuing_human_service_has_no_second_response_mixture(env):
    value = make_value(env, 'optimization_anchored_residual')
    vid = next(k for k, v in env.vehicles.items() if v['type'] == 1)
    env.vehicles[vid]['assigned_request'] = 991
    assert value.rejection_for_live_edges([vid], [2], [991])[0] == 0.
    assert value.response_masks_for_live_edges([vid], [2])[0] == 0.
    state = StateSnapshotBuilder.build(env)
    exp = dict(vehicle_id=vid, vehicle_type=1, action_type='assign_991', state_snapshot=state)
    assert value.response_from_experience(exp) == (0., 0.)
    edges = []
    StateSnapshotBuilder._append_continuing_edges(env, state, edges, matrix_vehicle_ids=set(), stage_id=0)
    continuing = next(e for e in edges if e.vehicle_id == vid)
    assert not continuing.human_response_mask and continuing.rejection_probability == 0.
    assert continuing.structured_score == continuing.success_structured_score


def test_bellman_uses_realized_reward_and_different_current_next_q():
    current = expected_structured_score(20., -5., .1, True)
    next_anchor = expected_structured_score(30., -7.5, .8, True)
    components = TargetComponents(('next',), next_anchor, next_anchor, 3.)
    # A rejected offer's actual reward remains -5, not an expectation or a
    # second q-weighted penalty. This is still one ordinary temporal backup.
    args = dict(reward=-5., discount=.95**2, next_components=components, current_structured_value=current)
    expected = -5. + .95**2 * (next_anchor + 3.)
    assert RecourseTargetBuilder.correction_bellman_target(**args) == pytest.approx(expected - current)
    assert RecourseTargetBuilder.correction_bellman_target(**args, direct_q=True) == pytest.approx(expected)
    assert RecourseTargetBuilder.correction_bellman_target(**{**args, 'next_components': None}) == -5. - current


def test_frozen_predictor_q_mask_gradients_actor_and_checkpoint_modes(env):
    value = make_value(env, 'optimization_anchored_residual')
    x = torch.randn(5, value.edge_dim)
    index = value.rejection_input_index
    x[:, index:index + 2] = torch.tensor([.3, 1.])
    value.network(x).sum().backward()
    network = getattr(value.network, 'base', value.network)
    assert torch.all(network.net[0].weight.grad[:, index:index + 2].abs().sum(0) > 0)
    predictor_ids = {id(p) for p in value.response_model.network.parameters()}
    assert not any(id(p) in predictor_ids for group in value.optimizer.param_groups for p in group['params'])
    assert all(p.grad is None and not p.requires_grad for p in value.response_model.network.parameters())
    other = x.clone()
    other[:, index:index + 2] = 0.
    assert torch.equal(value.actor(value._actor_features(x)), value.actor(value._actor_features(other)))
    state = value.extra_checkpoint_state()
    configure_acceptance_feature(env, 'predicted', model_state=env.ev_acceptance_model.to_dict(), anchor='off')
    feature_only = make_value(env, 'optimization_anchored_residual')
    with pytest.raises(ValueError, match='mismatch'):
        feature_only.load_extra_checkpoint_state(state)
    assert feature_only.acceptance_input_enabled and not feature_only.response_anchor_enabled
    configure_acceptance_feature(env, 'predicted', model_state=env.ev_acceptance_model.to_dict(), critic_input='none')
    anchor_only = make_value(env, 'optimization_anchored_residual')
    assert not anchor_only.acceptance_input_enabled and anchor_only.response_anchor_enabled


@pytest.mark.parametrize('version', [1, 2])
def test_legacy_probability_and_replay_schema_fail_closed(version):
    with pytest.raises(ValueError, match='Legacy'):
        EVRejectionProbabilityModel.from_dict({'version': version})
    replay = PrioritizedJointReplayBuffer()
    with pytest.raises(ValueError, match='schema'):
        replay.load_state_dict({'schema_version': version})


def test_date_splits_and_support_diagnostics():
    with pytest.raises(SystemExit):
        parse_args(['--train-dates', '2025-12-18', '--validation-dates', '2025-12-18', '--test-dates', '2025-12-20'])
    model = EVRejectionProbabilityModel(max_epochs=2).fit(samples(100))
    row = dict(samples(1)[0], idle_time=1000.)
    report = model.support_diagnostics([row])
    assert report['outside_any_count'] == report['per_feature']['idle_time'] == 1


def test_training_and_inference_share_v3_checkpoint_namespace(env, tmp_path):
    from src.acceptance_features import acceptance_checkpoint_suffix, add_acceptance_arguments
    import argparse
    path = tmp_path / 'model.json'
    env.ev_acceptance_model.save(path)
    auto = acceptance_checkpoint_suffix('predicted', path)
    feature_only = acceptance_checkpoint_suffix('predicted', path, anchor='off')
    anchor_only = acceptance_checkpoint_suffix('predicted', path, critic_input='none')
    assert len({auto, feature_only, anchor_only}) == 3
    parser = argparse.ArgumentParser()
    add_acceptance_arguments(parser)
    new = parser.parse_args(['--ev-response-feature', 'predicted', '--ev-response-model', str(path)])
    old_alias = parser.parse_args(['--ev-acceptance-feature', 'predicted', '--ev-acceptance-model', str(path)])
    assert vars(new) == vars(old_alias)
    # Guard the two real trainer paths as well as inference lookup paths.
    root = Path(__file__).resolve().parents[1]
    for filename in ('src/ADPtrainer.py', 'src/NYCtrainer.py', 'test_model.py', 'test_nyc_model.py'):
        source = (root / filename).read_text()
        assert 'acceptance_checkpoint_suffix(' in source
        assert '_evaccept-' not in source


def test_three_integration_modes_have_the_intended_beta_zero_scores(env):
    state = env.ev_acceptance_model.to_dict()
    result = {}
    for name, anchor, critic_input in (('feature_only', 'off', 'q_mask'),
                                      ('anchor_only', 'auto', 'none'),
                                      ('both', 'auto', 'q_mask')):
        configure_acceptance_feature(env, 'predicted', model_state=state, anchor=anchor, critic_input=critic_input)
        value = make_value(env, 'optimization_anchored_residual')
        graph = graph_for(env, value)
        edge = next(e for e in graph.edges if e.vehicle_type == 1 and e.action_type == ActionType.SERVICE)
        result[name] = [live_score(value, graph, edge, q=q) for q in (0., .5, 1.)]
    assert len(set(result['feature_only'])) == 1  # Zero-initialized input columns; beta=0.
    assert result['anchor_only'] == result['both']
    assert result['both'][0] > result['both'][1] > result['both'][2]


@pytest.mark.parametrize('mixture', [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
def test_mixed_collection_uses_feasible_unique_proposals_not_candidate_responses(mixture):
    environment = fake_env()
    environment.vehicles[1]['type'] = 1
    requests = [fake_request(), SimpleNamespace(**{**vars(fake_request()), 'request_id': 9, 'pickup': 8})]
    seen = []
    def solve(vids, reqs, feasible, scores, **kwargs):
        seen.append(feasible.copy())
        return {}
    optimizer = SimpleNamespace(_np_vehicle_rebalancing_network=solve, _np_vehicle_rebalancing_network_ev=solve,
        _get_matrix_action_layout=lambda reqs, cols: dict(requests=reqs, num_requests=2))
    environment.gurobi_optimizer = optimizer
    feasible = np.array([[1, 1, 1], [0, 1, 1]])
    def answer(vid, req):
        environment.draws += 1
        environment._last_offer_realizations[(5, vid, req.request_id)] = dict(acceptance_probability=.8)
        return False
    environment._should_reject_request = answer
    before = np.random.get_state()
    with mixed_feasible_offers(environment, seed=42, mixture=mixture) as stats, collect_offers(environment, episode_id='unit', seed=42) as rows:
        optimizer._np_vehicle_rebalancing_network([0, 1], requests, feasible, np.ones_like(feasible))
        assert environment.draws == 0 and rows == []
        assert len(stats['feasible_rows']) == 3
        assert np.all(seen[0] <= feasible)
        if mixture[0] == 0:
            assert np.all(seen[0][:, :2].sum(0) <= 1)
        environment._should_reject_request(0, requests[0])
        assert len(rows) == 1 and rows[0]['rejected'] == 0
        assert rows[0]['candidate_count'] == 2
        assert 'behavior_policy_id' in rows[0]
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert optimizer._np_vehicle_rebalancing_network is solve
    with pytest.raises(ValueError):
        parse_mixture('0.2,0.3,0.7')
