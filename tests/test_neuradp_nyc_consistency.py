"""Independent simulator/reward checks for the electric NeurADP adaptation."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.Action import IdleAction
from src.ADPtrainer import ADPTrainer
from src.NYCEnvironment import NYCEnvironment
from src.ValueFunction_neuradp import PyTorchChargingValueFunction as NeurADP
from src.real_conservative_charging import RealConservativeNYCEnvironment
from src.recourse.manifest import write_experiment_manifest
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.recourse.types import ActionType


@pytest.fixture
def nyc_factory(monkeypatch, tmp_path):
    empty = pd.DataFrame({'pickup_date': pd.to_datetime([]),
                          'pickup_second_of_day': pd.Series(dtype=float),
                          'PULocationID': pd.Series(dtype=int), 'DOLocationID': pd.Series(dtype=int)})
    monkeypatch.setattr(NYCEnvironment, '_load_demand_data', lambda self: empty.copy())
    monkeypatch.setenv('ICAPS_REPLAY_STORAGE', 'disk')
    monkeypatch.setenv('ICAPS_REPLAY_DIR', str(tmp_path/'replay'))

    def build(mode='current', soc=.2, scale=1., training=False, **options):
        cls = RealConservativeNYCEnvironment if mode == 'real_conservative' else NYCEnvironment
        env = cls(num_vehicles=2, ev_num_vehicles=1, random_seed=301,
                  start_date='2025-12-15', end_date='2025-12-15', start_hour=8, stop_hour=12,
                  aev_charging_center_count=3, usemcmf=True, mcmf_solver='exact',
                  mcmf_backend='ortools', daily_drop_off=False, ifreject=False,
                  charge_duration_scale=scale)
        env.reset()
        env.evaluatemode = not training
        env.adp_value = 1.
        env.mcmf_graph_reduction = True
        env.learner_variant = 'neuradp'
        env.state_variant = 'joint_state_shared_critic'
        env.configure_recourse_experiment('legacy')
        vf = NeurADP(grid_size=env.grid_size, num_vehicles=2, device='cpu',
                     episode_length=env.episode_length, env=env, neuradp_noise_scale=0., **options)
        vf.state_variant = env.state_variant
        vf.recourse_variant = 'legacy'
        env.set_value_function(vf)
        env.set_value_function_ev(vf)
        for vid, vehicle in env.vehicles.items():
            if env._is_ev(vid):
                vehicle['is_online'] = False
        vid = next(vid for vid in env.vehicles if not env._is_ev(vid))
        sid = env.aev_charging_station_ids[0]
        loc = env.charging_manager.stations[sid].location
        env.vehicles[vid].update(location=loc, zone_id=loc, coordinates=env.zone_coords[loc],
                                 battery=soc, target_location=None, idle_target=None)
        return env, vf, vid
    return build


def prefer_charge(vf):
    # Real network: positive value for SoC, making a charge selectable without
    # bypassing its features, exact solver, reservations or physical executor.
    with torch.no_grad():
        for parameter in vf.network.parameters():
            parameter.zero_()
        scalar_start = vf.network.state_embedding[0].in_features - 10
        vf.network.state_embedding[0].weight[0, scalar_start] = 1.
        vf.network.state_embedding[2].weight[0, 0] = 1.
        vf.network.output.weight[0, 0] = 1000.


@pytest.mark.parametrize('mode', ['current', 'real_conservative'])
@pytest.mark.parametrize('soc,scale', [(.2, 1.), (.5, 1.), (.7, 1.), (.85, 1.), (.2, .5)])
def test_charge_feature_matches_actual_finished_session(nyc_factory, mode, soc, scale):
    env, vf, vid = nyc_factory(mode, soc, scale)
    prefer_charge(vf)
    actions, stores, stores_ev = env.simulate_motion_integrated_control()
    graph = env._last_integrated_repair_graphs[0]
    chosen = next(e for e in graph.edges if e.vehicle_id == vid and e.edge_id in graph.selected_edge_ids)
    assert chosen.action_type == ActionType.CHARGE
    predicted_soc = vf._post_rows(graph, [chosen])[3][0, 0]
    env.step(actions, stores, stores_ev)
    assert predicted_soc == pytest.approx(env.vehicles[vid]["battery"], abs=1e-6)
    assert predicted_soc == pytest.approx(soc+env.chargeincrease_per_epoch, abs=1e-6)
    for _ in range(250):
        if env.vehicles[vid].get('completed_charging_durations_minutes'):
            break
        actions, stores, stores_ev = env.simulate_motion_integrated_control()
        env.step(actions, stores, stores_ev)
    vehicle = env.vehicles[vid]
    assert len(vehicle['completed_charging_durations_minutes']) == 1
    if scale == 1 and soc in (.5, .7):
        assert vehicle['battery'] == pytest.approx(.8)
    if scale == .5:
        assert vehicle['battery'] < .8  # session timeout remains physical
    if mode == 'real_conservative':
        assert not env._is_vehicle_waiting_for_charger(vid)


def test_wait_shaping_matches_execution_and_does_not_change_system_reward(nyc_factory):
    env, vf, vid = nyc_factory(soc=.2)
    for net in (vf.network, vf.target_network):
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env, [vid], np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)),
        num_requests=0, num_stations=0, num_zones=0, stage_id=0, solver_backend='ortools')
    edge = next(e for e in graph.edges if e.vehicle_id == vid)
    # Independent physical rates: 4 dollars idle/hour + 8 dollars learning/hour at 20% SoC.
    expected = -(4+8)*30/3600
    for target in (False, True):
        assert vf._neuradp_scores(graph, target=target)[edge.edge_id] == pytest.approx(expected)
    assert vf._saved_graph_targets(graph)[vid] == pytest.approx(expected)
    env.learning_soc_wait_penalty_per_hour = 80
    assert vf._neuradp_scores(graph, target=False)[edge.edge_id] == pytest.approx(expected)  # frozen reward
    fresh = StateSnapshotBuilder.feasible_graph_from_matrix(
        env, [vid], np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)),
        num_requests=0, num_stations=0, num_zones=0, stage_id=0, solver_backend='ortools')
    assert vf._neuradp_scores(fresh, target=False)[fresh.edges[0].edge_id] == pytest.approx(-(4+80)*30/3600)
    env.learning_soc_wait_penalty_per_hour = 8
    loc = env.vehicles[vid]['location']
    action = IdleAction([], env.zone_coords[loc], env.zone_coords[loc], loc, .2)
    action.learning_action_type = 'wait'
    env.vehicles[vid]['is_stationary'] = True
    _, rewards, _, _, info = env.step({vid: action}, {}, {})
    assert rewards[vid] == pytest.approx(-4*30/3600)
    assert env._epoch_soc_wait_learning_penalties[vid] == pytest.approx(8*30/3600)
    # A historical replay state must not pick up the current car's SoC.
    env.vehicles[vid]['battery'] = .9
    assert vf._neuradp_scores(graph, target=False)[edge.edge_id] == pytest.approx(expected)


def test_continuing_charge_features_are_frozen_and_do_not_top_up_again(nyc_factory):
    env, vf, vid = nyc_factory(soc=.85)
    prefer_charge(vf)
    actions, stores, stores_ev = env.simulate_motion_integrated_control()
    env.step(actions, stores, stores_ev)
    env.simulate_motion_integrated_control()
    graph = env._last_integrated_repair_graphs[0]
    edge = next(e for e in graph.edges if e.vehicle_id == vid)
    assert dict(edge.metadata)['continuing']
    before = vf._post_rows(graph, [edge])
    assert before[3][0, 0] == pytest.approx(.85+2*env.chargeincrease_per_epoch)
    env.vehicles[vid].update(battery=.01, charge_target_soc=.05, charging_time_left=1)
    after = vf._post_rows(graph, [edge])
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_nondefault_checkpoint_restores_scores_and_hyperparameters(nyc_factory, tmp_path):
    env, source, vid = nyc_factory(neuradp_gamma=.83, neuradp_target_tau=.2,
                                  neuradp_learning_rate=.0007, neuradp_fit_batch_size=3,
                                  neuradp_samples_per_update=2, neuradp_response_aware=True)
    env.learning_soc_wait_penalty_per_hour = 12.
    env.simulate_motion_integrated_control()
    graph = env._last_integrated_repair_graphs[0]
    expected = source._graph_edge_scores(graph, target_context=False)
    path = ADPTrainer._save_inference_checkpoint(source, 1, tmp_path/'model')['full_state']
    env.learning_soc_wait_penalty_per_hour = 99.
    restored = NeurADP(grid_size=env.grid_size, num_vehicles=2, env=env, device='cpu',
                       episode_length=env.episode_length)
    restored.state_variant = source.state_variant
    restored.recourse_variant = 'legacy'
    assert ADPTrainer.__new__(ADPTrainer).load_checkpoint(restored, path)
    assert restored.neuradp_config == source.neuradp_config
    assert restored.gamma == .83 and restored.tau == .2
    assert restored.optimizer.param_groups[0]['lr'] == .0007
    assert env.learning_soc_wait_penalty_per_hour == 12.
    assert restored._graph_edge_scores(graph, target_context=False) == pytest.approx(expected)
    assert restored._graph_edge_scores(graph, target_context=True) == pytest.approx(expected)


def test_bounded_inference_preserves_scores_and_retries_oom(nyc_factory, monkeypatch):
    import src.ValueFunction_neuradp as module
    env, vf, vid = nyc_factory()
    env.simulate_motion_integrated_control()
    graph = env._last_integrated_repair_graphs[0]
    edge = next(e for e in graph.edges if e.vehicle_id == vid)
    edges = [replace(edge, edge_id=str(i)) for i in range(11)]
    expected = vf._values(graph, edges, target=False).numpy()
    sizes = []
    forward = vf.network.forward
    def guarded(*tensors):
        sizes.append(len(tensors[0]))
        if len(tensors[0]) > 2:
            raise torch.cuda.OutOfMemoryError('simulated NeurADP batch allocation')
        return forward(*tensors)
    bounded = module.bounded_edge_inference
    monkeypatch.setattr(module, 'bounded_edge_inference',
                        lambda n, batch, evaluate, device: bounded(n, batch, evaluate, device='cuda'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(vf.network, 'forward', guarded)
    vf.qvalue_inference_batch_size = 8
    actual = vf._values(graph, edges, target=False).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert vf._last_qvalue_inference_stats['oom_retries'] == 2
    assert sizes[:3] == [8, 4, 2]


def test_training_td_error_is_bounded_and_frozen_evaluation_cannot_update(nyc_factory, monkeypatch):
    env, vf, vid = nyc_factory(training=True, neuradp_fit_batch_size=2)
    for _ in range(5):
        actions, stores, stores_ev = env.simulate_motion_integrated_control()
        env.step(actions, stores, stores_ev)
    forward = vf.network.forward
    sizes = []
    def guarded(*tensors):
        sizes.append(len(tensors[0]))
        assert len(tensors[0]) <= 2
        return forward(*tensors)
    vf.qvalue_inference_batch_size = 2
    monkeypatch.setattr(vf.network, 'forward', guarded)
    assert vf.train_step(batch_size=3) > 0
    assert sizes and vf._last_neuradp_td_inference_stats['max_batch_edges'] <= 2
    before = {key: value.clone() for key, value in vf.network.state_dict().items()}
    steps = vf.optimizer_steps_total
    env.evaluatemode = True
    assert vf.train_step(batch_size=3) == 0
    assert vf.optimizer_steps_total == steps
    assert all(torch.equal(v, before[k]) for k, v in vf.network.state_dict().items())


def test_memory_replay_manifest_includes_neuradp_config(tmp_path):
    vf = NeurADP(num_vehicles=2, device='cpu')
    assert vf.joint_replay_buffer.storage == 'memory'
    path = write_experiment_manifest(tmp_path/'manifest.json', arguments={}, results={}, value_functions=[vf])
    saved = json.loads(path.read_text())
    assert saved['effective_replay_hyperparameters'][0]['storage'] == 'memory'
    assert 'cache_rows' not in saved['effective_replay_hyperparameters'][0]
    assert saved['effective_model_hyperparameters'][0]['neuradp_config'] == vf.neuradp_config


def test_continuous_days_keep_training_clock_and_duration_normalization(nyc_factory):
    env, vf, vid = nyc_factory()
    env.simulate_motion_integrated_control()
    graph = env._last_integrated_repair_graphs[0]
    edge = next(e for e in graph.edges if e.vehicle_id == vid)
    before = vf._post_rows(graph, [edge])
    env.end_date = '2025-12-17'
    assert env.configure_continuous_evaluation() == 3
    for offset in (0, 2880, 5760):
        later = replace(graph, state=replace(graph.state, current_time=graph.state.current_time+offset))
        after = vf._post_rows(later, [edge])
        for actual, expected in zip(after, before):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize('mode', ['current', 'real_conservative'])
def test_actual_trainer_and_frozen_loaded_policy_choose_charge(nyc_factory, tmp_path, monkeypatch, mode):
    import run_nyctrainer as runner
    from test_real_conservative_nyc_cli import base_kwargs
    demand = pd.DataFrame([
        dict(pickup_date=pd.Timestamp('2025-12-15'), pickup_second_of_day=second,
             PULocationID=161, DOLocationID=162, fare_amount=15., trip_distance=.1)
        for second in range(0, 900, 30)
    ])
    monkeypatch.setattr(NYCEnvironment, '_load_demand_data', lambda self: demand.copy())
    monkeypatch.setattr(runner, 'resolve_nyc_parquet_paths', lambda *a, **k: [])
    monkeypatch.setattr(ADPTrainer, '_save_episode_stats_to_excel', lambda *a, **k: (None, None))
    reset = NYCEnvironment.reset
    def low_soc_reset(env):
        result = reset(env)
        sid = env.aev_charging_station_ids[0]
        location = env.charging_manager.stations[sid].location
        for vid, vehicle in env.vehicles.items():
            if not env._is_ev(vid):
                vehicle.update(battery=.1, location=location, zone_id=location,
                               coordinates=env.zone_coords[location], idle_target=None, target_location=None)
        return result
    monkeypatch.setattr(NYCEnvironment, 'reset', low_soc_reset)
    monkeypatch.chdir(tmp_path)
    kwargs = base_kwargs()
    kwargs.update(transportation_mode='integrated', recourse_variant='legacy',
                  learner_variant='neuradp', zone_distribution_mode='neuradp',
                  state_variant='joint_state_separate_critics', mcmf_backend='ortools',
                  mcmf_graph_reduction=True, charging_model=mode, stop_hour=.25,
                  neuradp_options={'neuradp_noise_scale': 0.})
    trained, training_env = runner.run_nyc_training(**kwargs)
    train_starts = sum(v['charging_count'] for v in training_env.vehicles.values() if v['type'] == 2)
    assert train_starts > 0 and trained['optimizer_budget']['optimizer_steps_total'] > 0
    checkpoints = list((tmp_path/'checkpoints').rglob('best_full_state_episode_1.pth'))
    assert len(checkpoints) == 2
    checkpoint = torch.load(next(p for p in checkpoints if p.parent.name.endswith('_aev')), weights_only=False)
    assert checkpoint['extra_value_function_state']['neuradp_implementation_version'] == 3
    kwargs.update(trainnetwork=False, ifloadcheckpoint=True, charging_model=None,
                  conservative_charging=None, checkpoint_charging_model=mode, checkpoint_selection='best_reward')
    evaluated, env = runner.run_nyc_training(**kwargs)
    test_starts = sum(v['charging_count'] for v in env.vehicles.values() if v['type'] == 2)
    assert test_starts > 0
    assert evaluated['optimizer_budget']['optimizer_steps_total'] == 0
    assert not env.value_function.joint_replay_buffer
    assert env.value_function.neuradp_config == training_env.value_function.neuradp_config
    assert all(torch.equal(v.cpu(), checkpoint['network_state_dict'][k])
               for k, v in env.value_function.network.state_dict().items())
    (tmp_path/'charge_verification.json').write_text(json.dumps({
        'mode': mode, 'vehicles': 6, 'aevs': 4, 'steps': 30,
        'initial_aev_soc': .1, 'training_starts': train_starts, 'test_starts': test_starts,
        'training_optimizer_steps': trained['optimizer_budget']['optimizer_steps_total'],
        'test_optimizer_steps': evaluated['optimizer_budget']['optimizer_steps_total'],
        'controlled_network_weights': False,
    }, indent=2))


@pytest.mark.parametrize('mode', ['current', 'real_conservative'])
@pytest.mark.parametrize('case', ['pickup', 'pickup_moving', 'trip_moving',
                                 'dropoff', 'relocate', 'wait', 'charge_travel', 'charge_start'])
def test_frozen_epoch_matches_independent_actual_nyc_step(nyc_factory, mode, case):
    """Use actual dispatch/execution as oracle, not another copy of the preview."""
    import pickle
    import random
    from src.Action import ServiceAction
    from src.Request import Request
    from src.neuradp_nyc_transition import preview_epoch, vehicle_record
    from src.recourse.integrated_repair import execute_commits
    from src.recourse.types import FeasibleEdgeSnapshot, FeasibleGraphSnapshot, VehicleSnapshot
    env, vf, vid = nyc_factory(mode, soc=.7)
    env.movement_reward_noise_std = env.service_event_reward_noise_std = 0.
    env.charging_reward_noise = 0.
    v = env.vehicles[vid]
    v.update(location=161, zone_id=161, coordinates=env.zone_coords[161])
    req = Request(501, 161, 162, env.current_time, 5, value=37.5, final_value=37.5, trip_distance_km=.9)
    env.active_requests[501] = req
    sid, rid, continuing = None, None, case in ('trip_moving', 'dropoff')
    if case in ('pickup', 'pickup_moving', 'trip_moving', 'dropoff'):
        kind, target, rid = ActionType.SERVICE, 161, 501
        if case == 'pickup_moving':
            v.update(location=162, zone_id=162, coordinates=env.zone_coords[162])
        if continuing:
            loc = 162 if case == 'dropoff' else 161
            v.update(location=loc, zone_id=loc, coordinates=env.zone_coords[loc],
                     passenger_onboard=501, assigned_request=None,
                     passenger_trip_distance_total=.9,
                     passenger_trip_distance_remaining=0. if case == 'dropoff' else .9,
                     passenger_trip_distance_travelled=.9 if case == 'dropoff' else 0.,
                     passenger_trip_start_coordinates=env.zone_coords[161])
    elif case.startswith('charge'):
        kind, sid = ActionType.CHARGE, env.aev_charging_station_ids[0]
        target = env.charging_manager.stations[sid].location
        loc = target if case == 'charge_start' else min(
            (z for z in env.zone_coords if env.get_distance_km(z, target) > .5),
            key=lambda z: env.get_distance_km(z, target))
        v.update(location=loc, zone_id=loc, coordinates=env.zone_coords[loc])
        env.generate_vehicle_chargerange([vid])
    else:
        kind = ActionType.WAIT if case == 'wait' else ActionType.RELOCATE
        target = v['location'] if case == 'wait' else 162
    def core():
        return pickle.dumps((env.vehicles, {r: vars(q) for r,q in env.active_requests.items()},
            {s:(x.current_vehicles, x.charging_queue, x.charging_queue_notarrived, x.available_slots)
             for s,x in env.charging_manager.stations.items()}, env.current_time,
            env.idle_charging_num, getattr(env, 'real_reservations', {})))
    before, numpy_before, random_before = core(), np.random.get_state(), random.getstate()
    record, physical, shaping = preview_epoch(env, vehicle_id=vid, action_type=kind,
        target_location=target, request_id=rid, station_id=sid, continuing=continuing)
    assert core() == before
    assert random.getstate() == random_before
    numpy_after = np.random.get_state()
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    if continuing:
        actions = {vid: ServiceAction([], 501, v['location'], v['battery'])}
    else:
        edge = FeasibleEdgeSnapshot(edge_id='test', vehicle_id=vid, vehicle_type=2,
            action_type=kind, action_id='test', target_location=target,
            post_action_location=target, request_id=rid, station_id=sid)
        graph = FeasibleGraphSnapshot('test', 0, 0, 'ortools', StateSnapshotBuilder.build(env),
                                      (edge,), selected_edge_ids=('test',))
        actions = {}
        execute_commits(env, graph, actions, {}, {})
    _, rewards, _, _, _ = env.step(actions, {}, {})
    rid_after = v['passenger_onboard'] if v['passenger_onboard'] is not None else v['assigned_request']
    actual = VehicleSnapshot.from_vehicle(vid, v, request=env.active_requests.get(rid_after),
                                          env=env, current_time=env.current_time)
    assert record == pytest.approx(vehicle_record(actual))
    assert physical == pytest.approx(rewards[vid], abs=1e-10)
    assert shaping == pytest.approx(env._epoch_soc_wait_learning_penalties[vid])
    if case == 'pickup':
        assert physical == .5  # fare belongs to completion, not assignment
    if case == 'dropoff':
        assert physical == 37.5


def test_reactive_queue_release_is_in_the_one_epoch_outcome(nyc_factory):
    from src.Action import ChargingAction
    from src.neuradp_nyc_transition import preview_epoch
    env, vf, vid = nyc_factory(soc=.2)
    station_id = env.aev_charging_station_ids[0]
    station = env.charging_manager.stations[station_id]
    station.max_capacity = station.available_slots = 1
    station.start_charging('0')
    env.vehicles[0].update(is_online=True, location=station.location, battery=.5,
        charging_station=station_id, charging_time_left=1, charge_target_soc=.8)
    record, reward, _ = preview_epoch(env, vehicle_id=vid, action_type=ActionType.CHARGE,
                                      target_location=station.location, station_id=station_id)
    action = ChargingAction([], station_id, env.charge_duration, station.location, .2)
    env._move_vehicle_to_charging_station(vid, station_id)
    _, rewards, _, _, _ = env.step({vid: action}, {}, {})
    assert reward == pytest.approx(rewards[vid])
    assert record[1] == env.vehicles[vid]['battery'] == .2  # newly promoted: no retroactive charge
    assert record[11] == env.vehicles[vid]['charging_station'] == station_id


def test_saved_current_graph_backup_does_not_need_a_successor(nyc_factory, monkeypatch):
    env, vf, vid = nyc_factory(training=True)
    actions, stores, stores_ev = env.simulate_motion_integrated_control()
    env.step(actions, stores, stores_ev)
    row = next(iter(vf.joint_replay_buffer))
    from src.recourse.replay import PrioritizedJointReplayBuffer
    vf.joint_replay_buffer = PrioritizedJointReplayBuffer(capacity=8)
    row = replace(row, next_transition_id=None, done=False)
    vf.joint_replay_buffer.add(row)
    monkeypatch.setattr(vf, '_temporal_successor_graph', lambda *a: pytest.fail('shifted backup'))
    visited, current_soc = [], []
    resolve, forward = vf._saved_graph_targets, vf.network.forward
    def capture(graph):
        visited.append(graph.graph_id)
        return resolve(graph)
    def inputs(*arrays):
        if torch.is_grad_enabled():
            current_soc.extend(arrays[3][:, 0].detach().tolist())
        return forward(*arrays)
    monkeypatch.setattr(vf, '_saved_graph_targets', capture)
    monkeypatch.setattr(vf.network, 'forward', inputs)
    assert vf.train_step(batch_size=1) > 0
    assert visited == [row.ev_stage_graph.graph_id]
    expected = [v.battery for v in row.ev_stage_graph.state.vehicles if v.vehicle_id == vid]
    assert current_soc == pytest.approx(expected)


def test_terminal_backup_keeps_last_epoch_reward_and_rejects_old_weights(nyc_factory):
    env, vf, vid = nyc_factory(soc=.2)
    env.current_time = env.episode_length-1
    for net in (vf.network, vf.target_network):
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
            net.output.bias.fill_(123.)
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env, [vid], np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)),
        num_requests=0, num_stations=0, num_zones=0, stage_id=0, solver_backend='ortools')
    assert vf._saved_graph_targets(graph)[vid] == pytest.approx(-.1)  # no 123 bootstrap, no zero target
    state = vf.inference_checkpoint_state()
    state['neuradp_implementation_version'] = 2
    with pytest.raises(ValueError, match='retrain'):
        vf.load_extra_checkpoint_state(state)


@pytest.mark.parametrize('mode', ['current', 'real_conservative'])
def test_public_train_and_test_entrypoints_share_v3_model_paths(nyc_factory, tmp_path, monkeypatch, mode):
    import run_nyctrainer as train
    import test_nyc_model as evaluate
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, 'apply_paper_parameter_preset', lambda args: args)
    monkeypatch.setattr(evaluate, 'apply_paper_parameter_preset', lambda args: args)
    monkeypatch.setattr(train, 'resolve_nyc_parquet_paths', lambda *a, **k: [])
    monkeypatch.setattr(ADPTrainer, '_save_episode_stats_to_excel', lambda *a, **k: (None, None))
    common = ['--methods', 'neuradp', '--num-vehicles', '6', '--num-ev', '2',
              '--start-date', '2025-12-15', '--end-date', '2025-12-15', '--episodes', '1',
              '--start-hour', '0', '--stop-hour', '.1', '--mcmf-backend', 'ortools',
              '--batch-size', '2',
              '--neuradp-noise-scale', '0', '--state-variant', 'joint_state_separate_critics']
    train.main(common+['--charging-model', mode, '--training-frequency', '1', '--random-seed', '7'])
    checkpoints = list((tmp_path/'checkpoints').rglob('best_full_state_episode_1.pth'))
    assert len(checkpoints) == 2
    assert all('neuradp-v3' in str(p) for p in checkpoints)
    checkpoint = torch.load(next(p for p in checkpoints if p.parent.name.endswith('_aev')), weights_only=False)
    assert checkpoint['extra_value_function_state']['neuradp_implementation_version'] == 3
    observed = []
    original = evaluate.run_nyc_training
    def run(**kwargs):
        result, env = original(**kwargs)
        observed.append((result, env))
        return result, env
    monkeypatch.setattr(evaluate, 'run_nyc_training', run)
    output = tmp_path/'test_output'
    evaluate.main(common+['--strategies', 'ADP-MCMF', '--seeds', '7',
        '--checkpoint-charging-model', mode, '--load-model-start-date', '2025-12-15',
        '--load-model-end-date', '2025-12-15', '--checkpoint-selection', 'best-reward',
        '--test-steps-per-episode', '12', '--output-dir', str(output)])
    assert len(observed) == 1
    result, env = observed[0]
    assert result['charging_model'] == mode
    assert result['optimizer_budget']['optimizer_steps_total'] == 0
    assert not env.value_function.joint_replay_buffer
    assert env.mcmf_backend == 'ortools' and env.mcmf_graph_reduction
    assert env.value_function.target_builder.backend == 'ortools'
    assert env.value_function.target_builder.graph_reduction
    assert all(torch.equal(v.cpu(), checkpoint['network_state_dict'][k])
               for k,v in env.value_function.network.state_dict().items())
    exported = np.load(next(output.glob('*.npy')), allow_pickle=True)
    assert exported[0]['neuradp_implementation_version'] == 3
    assert exported[0]['neuradp_backup'] == 'saved_current_state_one_nyc_epoch'


def test_rejected_hev_epoch_matches_actual_relocation_and_reward(nyc_factory, monkeypatch):
    from src.Request import Request
    from src.neuradp_nyc_transition import preview_epoch, rejection_probability, vehicle_record
    from src.recourse.types import VehicleSnapshot
    env, vf, _ = nyc_factory(soc=.7)
    vid, v = 0, env.vehicles[0]
    v.update(is_online=True, battery=.7, location=161, coordinates=env.zone_coords[161], zone_id=161)
    req = Request(501, 161, 162, env.current_time, 5, value=37.5, final_value=37.5)
    env.active_requests[501] = req
    env.ifreject, env.reject_uniform = True, False
    env.ride_acceptance_noise_std = 0.
    env.ride_acceptance_asc = -100.
    env.movement_reward_noise_std = env.service_event_reward_noise_std = 0.
    assert rejection_probability(env, vid, req) == 1.
    monkeypatch.setattr(env, '_handle_ev_rejection_relocation', lambda vid: (162, {162: 1.}))
    record, reward, shaping = preview_epoch(env, vehicle_id=vid,
        action_type=ActionType.SERVICE, target_location=161, request_id=501, rejected_target=162)
    actions = {}
    env._process_integrated_assignments({vid: req}, actions, {}, {})
    assert actions[vid].was_rejected
    _, rewards, _, _, _ = env.step(actions, {}, {})
    actual = VehicleSnapshot.from_vehicle(vid, v, env=env, current_time=env.current_time)
    assert record == pytest.approx(vehicle_record(actual))
    assert reward == pytest.approx(rewards[vid])
    assert reward < 0 and shaping == 0


@pytest.mark.parametrize('mode', ['current', 'real_conservative'])
def test_neuradp_epoch_carries_charging_across_midnight(nyc_factory, monkeypatch, mode):
    from src.neuradp_nyc_transition import preview_epoch, vehicle_record
    from src.recourse.types import VehicleSnapshot
    env, vf, vid = nyc_factory(mode, soc=.2)
    env.end_date = '2025-12-17'
    env.configure_continuous_evaluation()
    env.current_time = (86400-env.START_EPOCH)/env.EPOCH_LENGTH-1
    # No demand reads or draws are needed to test the physical day boundary.
    monkeypatch.setattr(env, 'generate_requests', lambda: [])
    env.generate_vehicle_chargerange([vid])
    sid = env.aev_charging_station_ids[0]
    station = env.charging_manager.stations[sid]
    env.charging_reward_noise = 0.
    record, reward, _ = preview_epoch(env, vehicle_id=vid, action_type=ActionType.CHARGE,
                                      target_location=station.location, station_id=sid)
    from src.Action import ChargingAction
    env._move_vehicle_to_charging_station(vid, sid)
    action = ChargingAction([], sid, env.charge_duration, station.location, .2)
    _, rewards, _, _, _ = env.step({vid: action}, {}, {})
    actual = VehicleSnapshot.from_vehicle(vid, env.vehicles[vid], env=env, current_time=env.current_time)
    assert record == pytest.approx(vehicle_record(actual))
    assert reward == pytest.approx(rewards[vid])
    assert vf._time_norm(env.current_time) == 0.
