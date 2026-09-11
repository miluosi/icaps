"""Feasibility fallback and learning-only battery shaping regressions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark_conservative_charging import make_case, solve_adapter
from src.Action import ChargingAction, IdleAction
from src.NYCEnvironment import NYCEnvironment
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction
from test_recourse_must_fix import _graph, _transition


def idle_env(battery=.1):
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.EPOCH_LENGTH = 30.
    env.current_time = 2.
    env.idle_penalty = 4. / 120.
    env.vehicles = {1: dict(type=2, location=0, battery=battery,
                           is_online=True, is_stationary=True, idle_target=None,
                           assigned_request=None, idle_timer=0)}
    action = IdleAction([], (0., 0.), (0., 0.), 0, battery)
    action.current_time = 0.
    action.target_location = action.vehicle_loc_post = 0
    action.dur_time = 2.
    env.storeactions = {1: action}
    env.storeactions_ev = {}
    env.active_requests = {}
    env.get_distance_km = lambda a, b: abs(a - b)
    env.get_travel_time = lambda a, b: abs(a - b)
    env.get_zone_embedding_id = lambda a: a
    env.filter_aev_experiences_for_aev_value_function = lambda rows: rows
    return env, action


@pytest.mark.parametrize('conservative', [False, True])
def test_low_soc_competition_has_real_wait_fallback(conservative):
    env, ids, scores, _ = make_case(2, 1, 1, 'synchronized', 0)
    env.conservative_charging = conservative
    for vehicle in env.vehicles.values():
        vehicle['battery'] = .1
    charge = env.generate_vehicle_chargerange(ids)
    wait = env.generate_vehicle_wait(ids, charge_feasibility=charge)
    assert wait.tolist() == [[1.], [1.]]
    assignments, _ = solve_adapter(env, ids, np.column_stack((charge, wait)), scores, True)
    assert assignments is not None
    assert len(assignments) == 2
    assert sum(action == 'charge_100' for action in assignments.values()) == 1
    assert sum(action == 'waiting' for action in assignments.values()) == 1


@pytest.mark.parametrize('battery,expected', [(.8, 0.), (.3, 0.), (.2, 8/120), (.1, 32/120), (0., 72/120)])
def test_battery_shaping_and_epoch_units(battery, expected):
    env, _ = idle_env(battery)
    assert env.soc_wait_learning_penalty(battery) == pytest.approx(expected)
    assert env.soc_wait_learning_penalty(battery, 2.) == pytest.approx(2 * expected)
    env.EPOCH_LENGTH = 60.
    assert env.soc_wait_learning_penalty(battery) == pytest.approx(2 * expected)


def test_only_online_aev_wait_gets_shaping():
    env, action = idle_env()
    assert env._action_soc_wait_learning_penalty(1, action) > 0
    assert env._action_soc_wait_learning_penalty(1, ChargingAction([], 100)) == 0
    action.learning_action_type = 'reloc'
    assert env._action_soc_wait_learning_penalty(1, action) == 0
    action.learning_action_type = 'idle'
    env.vehicles[1]['type'] = 1
    assert env._action_soc_wait_learning_penalty(1, action) == 0
    env.vehicles[1].update(type=2, is_online=False)
    assert env._action_soc_wait_learning_penalty(1, action) == 0


@pytest.mark.parametrize('battery', [.1, .8])
def test_legacy_learning_changes_without_changing_executed_reward(battery):
    env, action = idle_env(battery)
    experiences = []
    env.value_function = SimpleNamespace(experience_buffer=[],
                                        store_experience=lambda **kw: experiences.append(kw))
    env.value_function_ev = None
    assert env._execute_action(1, action)[0] == pytest.approx(-env.idle_penalty)
    assert env._execute_action(1, action)[0] == pytest.approx(-env.idle_penalty)
    physical = action.dur_reward
    env._update_q_learning({1: action})
    assert len(experiences) == 1
    assert experiences[0]['reward'] == pytest.approx(
        physical - env.soc_wait_learning_penalty(battery, 2.))
    assert action.dur_reward == pytest.approx(-2 * env.idle_penalty)


@pytest.mark.parametrize('variant', ['r1', 'r2', 'r3', 'macro'])
def test_joint_learning_payload_keeps_economic_rewards_separate(variant):
    ev = _graph('ev', stage=1, vehicle_id=0, vehicle_type=1)
    aev = _graph('aev', stage=2, vehicle_id=1, vehicle_type=2)
    row = _transition(ev_graph=ev, aev_graph=aev, recourse_variant=variant)
    shaped = replace(row, aev_soc_wait_learning_penalty=.25)
    vf = PyTorchChargingValueFunction.__new__(PyTorchChargingValueFunction)
    leader = vf._joint_stage_payload(shaped, ifEV=True)
    follower = vf._joint_stage_payload(shaped, ifEV=False)
    assert leader[2] == pytest.approx(row.reward_system - .25 if variant == 'macro' else row.reward_ev)
    if variant == 'r2':
        assert follower is None
    else:
        assert follower[2] == pytest.approx(row.reward_aev - .25)
    assert shaped.reward_system == row.reward_system
    assert shaped.reward_aev == row.reward_aev
    assert row.learning_reward_system == row.reward_system  # older/unshaped replay


@pytest.mark.parametrize('variant', ['r3', 'macro'])
def test_real_nyc_step_freezes_shaping_in_joint_replay(variant):
    from run_recourse_audit import build_pair
    from train_acceptance_model import make_environment, parse_args
    env = make_environment(parse_args([
        '--environment', 'nyc', '--num-vehicles', '2', '--num-ev', '1',
        '--stop-hour', '8.05',
    ]), 901)
    env.evaluatemode, env.adp_value, env.episode_length = False, 1., 1
    env.configure_recourse_experiment(variant, common_random_numbers=True)
    env.state_variant = 'joint_state_separate_critics'
    env.learner_variant = 'optimization_anchored_residual'
    env._should_consider_ev_charging = lambda vid: False
    env._charging_station_ids_for_vehicle = lambda vid: []
    env.active_requests = {}
    aev = next(vid for vid, v in env.vehicles.items() if v['type'] == 2)
    for v in env.vehicles.values():
        v.update(battery=.1, assigned_request=None, passenger_onboard=None,
                 charging_station=None, charging_target=None, target_location=None,
                 idle_target=None, penalty_timer=0, is_online=True)
    pair = build_pair(env)
    actions, stores, stores_ev = env.simulate_motion_evfirst()
    assert isinstance(actions[aev], IdleAction)
    expected = env.soc_wait_learning_penalty(.1)
    env.step(actions, stores, stores_ev)
    row = list(pair[0].joint_replay_buffer)[-1]
    assert row.aev_soc_wait_learning_penalty == pytest.approx(expected)
    assert row.reward_aev == pytest.approx(-env.idle_penalty)
    assert row.reward_ledger.system == pytest.approx(row.reward_system)
    assert row.learning_reward_aev == pytest.approx(row.reward_aev - expected)
    env._epoch_soc_wait_learning_penalties.clear()
    assert row.aev_soc_wait_learning_penalty == pytest.approx(expected)
    if variant == 'macro':
        assert pair[1]._joint_stage_payload(row, ifEV=True)[2] == pytest.approx(row.reward_system - expected)
