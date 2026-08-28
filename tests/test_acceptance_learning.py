from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.acceptance_features import (
    acceptance_checkpoint_suffix, configure_acceptance_feature, predicted_rejection,
)
from src.acceptance_model import BinaryAcceptanceModel, offer_features
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.value_function_registry import VALUE_FUNCTION_CHOICES, get_value_function_class
from train_acceptance_model import make_environment, parse_args


@pytest.fixture
def env():
    args = parse_args(['--num-vehicles', '12', '--num-ev', '6', '--simulation-period', '20'])
    env = make_environment(args, 881)
    env.active_requests[991] = SimpleNamespace(request_id=991, pickup=100, dropoff=120,
        value=10.0, final_value=13.0, surge_bonus=7.0, created_time=0.0,
        pickup_deadline=30.0, dropoff_deadline=50.0, travel_time=4.0)
    vid = next(k for k, v in env.vehicles.items() if v['type'] == 1)
    features = offer_features(env, vid, env.active_requests[991])
    rows = [dict(features, idle_time=features['idle_time'] + i % 10,
                 pickup_time=features['pickup_time'] + i % 4,
                 surge_bonus=features['surge_bonus'] + i % 7, rejected=int(i % 5 == 0))
            for i in range(200)]
    model = BinaryAcceptanceModel(max_epochs=30).fit(rows)
    configure_acceptance_feature(env, 'predicted', model_state=model.to_dict())
    return env


def make_value(env, mode):
    return get_value_function_class(mode)(env=env, num_vehicles=12, grid_size=env.grid_size,
                                          episode_length=env.episode_length, zone_distribution_mode=mode)


@pytest.mark.parametrize('mode', VALUE_FUNCTION_CHOICES)
def test_every_registered_learner_receives_probability_and_roundtrips(env, mode):
    value = make_value(env, mode)
    vid = next(k for k, v in env.vehicles.items() if v['type'] == 1)
    req = next(iter(env.active_requests.values()))
    expected = predicted_rejection(env, vid, req)
    seen = []
    handle = value.network.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append((args, kwargs)), with_kwargs=True)
    out = value.batch_get_assignment_q_value([dict(
        vehicle_id=vid, target_id=req.request_id, vehicle_location=env.vehicles[vid]['location'],
        target_location=req.pickup, request_value=req.final_value, current_time=env.current_time,
        battery_level=env.vehicles[vid]['battery'], vehicle_idle_time=env.vehicles[vid]['idle_timer'])])
    handle.remove()
    assert len(out) == 1 and np.isfinite(out).all()
    args, kwargs = seen[-1]
    actual = (args[0][0, value.acceptance_input_index].item() if args
              else kwargs['rejection_probability'].item())
    assert actual == pytest.approx(expected)
    assert 0 < actual < 1
    state = value.extra_checkpoint_state()
    value.load_extra_checkpoint_state(state)
    bad = {**state, 'ev_response': {'mode': 'off', 'model': None}}
    with pytest.raises(ValueError, match='mismatch'):
        value.load_extra_checkpoint_state(bad)


@pytest.mark.parametrize('mode', ['integrated_directq', 'optimization_anchored_residual', 'bayes', 'time-only'])
def test_replay_probability_is_pre_offer_not_live_or_outcome(env, mode):
    value = make_value(env, mode)
    state = StateSnapshotBuilder.build(env)
    vid = next(v.vehicle_id for v in state.vehicles if v.vehicle_type == 1)
    req = state.requests[0]
    row = dict(vehicle_id=vid, vehicle_type=1, action_type=f'assign_{req.request_id}', state_snapshot=state)
    before = value.rejection_from_experience(row)
    env.vehicles[vid]['idle_timer'] = 1000
    env.vehicles[vid]['location'] = (env.vehicles[vid]['location'] + 40) % (env.grid_size ** 2)
    env.active_requests.clear()
    env.current_time += 100
    for vehicle in env.vehicles.values():
        vehicle['battery'] = .01
        vehicle['is_online'] = False
    row.update(accepted=0, acceptance_outcome='rejected', reward=-9999)
    assert value.rejection_from_experience(row) == before
    row['next_state_snapshot'] = replace(state, vehicles=tuple(
        replace(v, idle_time=123) if v.vehicle_id == vid else v for v in state.vehicles))
    next_value = value.rejection_from_experience(row, {'action_type': row['action_type']}, next_state=True)
    assert next_value != before
    with pytest.raises(ValueError, match='snapshot'):
        value.rejection_from_experience({**row, 'state_snapshot': None})


def test_probability_masks_and_schema_validation(env):
    value = make_value(env, 'integrated_directq')
    ev = next(k for k, v in env.vehicles.items() if v['type'] == 1)
    aev = next(k for k, v in env.vehicles.items() if v['type'] == 2)
    req = next(iter(env.active_requests))
    result = value.rejection_for_live_edges([ev, ev, aev], [2, 3, 2], [req, -1, req])
    assert 0 < result[0] < 1 and list(result[1:]) == [0, 0]
    with pytest.raises(ValueError, match='request_ids'):
        value.rejection_for_live_edges([ev], [2])
    with pytest.raises(ValueError, match='trained'):
        configure_acceptance_feature(env, 'predicted')
    model = env.ev_acceptance_model.to_dict()
    with pytest.raises(ValueError, match='units'):
        configure_acceptance_feature(SimpleNamespace(get_travel_time_minutes=lambda a, b: 1),
                                     'predicted', model_state=model)


@pytest.mark.parametrize('mode', ['integrated_directq', 'optimization_anchored_residual'])
def test_probability_column_has_td_gradient_and_preserves_initialization(env, mode):
    torch.manual_seed(81)
    on = make_value(env, mode)
    configure_acceptance_feature(env, 'off')
    torch.manual_seed(81)
    off = make_value(env, mode)
    x = torch.randn(4, off.edge_dim)
    p = torch.full((4, 2), .7)
    idx = on.acceptance_input_index
    expanded = torch.cat([x[:, :idx], p, x[:, idx:]], 1)
    assert torch.allclose(on.network(expanded), off.network(x), atol=1e-5)
    on.network(expanded).sum().backward()
    base = getattr(on.network, 'base', on.network)
    assert base.net[0].weight.grad[:, idx].abs().sum() > 0
    for key, tensor in on.graph_encoder.state_dict().items():
        assert torch.equal(tensor, off.graph_encoder.state_dict()[key])


@pytest.mark.parametrize('mode', ['bayes', 'time-only', 'st_masac_gat_queue_demand_gurobi'])
def test_legacy_learning_end_to_end_with_probability(env, mode):
    model_state = env.ev_acceptance_model.to_dict()
    env = make_environment(parse_args(['--num-vehicles', '20', '--num-ev', '10']), 883)
    configure_acceptance_feature(env, 'predicted', model_state=model_state)
    cls = get_value_function_class(mode)
    pair = [cls(env=env, num_vehicles=20, grid_size=env.grid_size,
                episode_length=env.episode_length, zone_distribution_mode=mode) for _ in range(2)]
    for vf in pair:
        vf.learner_variant = 'legacy'
    env.set_value_function(pair[0])
    env.set_value_function_ev(pair[1])
    env.adp_value = 1
    env.evaluatemode = False
    for _ in range(45):
        actions, stored, stored_ev = env.simulate_motion(
            agents=[], current_requests=list(env.active_requests.values()), rebalance=True)
        env.step(actions, stored, stored_ev)
    before = [p.detach().clone() for p in pair[1].network.parameters()]
    # Legacy residual beta starts at zero; subsequent updates activate TD
    # gradients on the residual critic.
    for _ in range(3):
        loss = pair[1].train_step(batch_size=8, ifEV=True)
    assert np.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, pair[1].network.parameters()))


def test_nyc_dispatch_uses_startup_features_and_joint_training_counter():
    from src.NYCEnvironment import NYCEnvironment
    ready = NYCEnvironment._value_function_ready_for_dispatch
    assert not ready(None)
    assert not ready(SimpleNamespace(training_step=0))
    assert ready(SimpleNamespace(training_step=0, joint_training_step=2))
    assert ready(SimpleNamespace(training_step=0, acceptance_input_enabled=True))
    assert ready(SimpleNamespace(training_step=0, learner_variant='integrated_directq'))
    assert ready(SimpleNamespace(training_step=0, learner_variant='optimization_anchored_residual'))


def test_checkpoint_namespace_and_predictor_identity(env, tmp_path):
    value = make_value(env, 'integrated_directq')
    model_path = tmp_path / 'acceptance.json'
    env.ev_acceptance_model.save(model_path)
    suffix = acceptance_checkpoint_suffix('predicted', model_path)
    assert suffix.startswith('_evreject-v3-')
    assert acceptance_checkpoint_suffix('off', None) == ''
    assert acceptance_checkpoint_suffix('predicted', model_path) == suffix
    state = value.extra_checkpoint_state()
    state['ev_response'] = {**state['ev_response'], 'model': None}
    with pytest.raises(ValueError, match='predictor differs'):
        value.load_acceptance_checkpoint_state(state)


def test_full_feature_vector_matches_snapshot_and_survives_live_mutation(env):
    state = StateSnapshotBuilder.build(env)
    vid = next(v.vehicle_id for v in state.vehicles if v.vehicle_type == 1)
    request = env.active_requests[991]
    live = offer_features(env, vid, request)
    saved_request = next(r for r in state.requests if r.request_id == 991)
    assert live == offer_features(env, vid, saved_request, snapshot=state)
    env.current_time += 100
    env.active_requests.clear()
    for vehicle in env.vehicles.values():
        vehicle.update(battery=.01, is_online=False, idle_timer=500, penalty_timer=10)
    assert live == offer_features(env, vid, saved_request, snapshot=state)


def test_bayes_probability_adapter_preserves_supplied_edge_features(env, monkeypatch):
    value = make_value(env, 'bayes')
    vid = next(k for k, v in env.vehicles.items() if v['type'] == 1)
    req = next(iter(env.active_requests.values()))
    captured = {}
    def forward(**kwargs):
        captured.update(kwargs)
        return [0.0]
    monkeypatch.setattr(value, 'batch_get_mixed_q_values', forward)
    value.batch_get_assignment_q_value([dict(
        vehicle_id=vid, target_id=req.request_id, vehicle_location=env.vehicles[vid]['location'],
        target_location=req.pickup, pickup_dist=7.5, pick_zone=3, vehicle_idle_time=4.0,
        post_action_distance=12.0, post_action_duration=18.0, post_action_zoneid=5)])
    assert captured['target_distances'] == [7.5]
    assert captured['target_zoneids'] == [3]
    assert captured['vehicle_idle_times'] == [4.0]
    assert captured['post_action_distances'] == [12.0]
    assert captured['post_action_durations'] == [18.0]
    assert captured['post_action_zoneids'] == [5]
