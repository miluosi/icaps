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
        scalar_start = vf.network.state_embedding[0].in_features - 9
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
    for _ in range(250):
        if env.vehicles[vid].get('completed_charging_durations_minutes'):
            break
        actions, stores, stores_ev = env.simulate_motion_integrated_control()
        env.step(actions, stores, stores_ev)
    vehicle = env.vehicles[vid]
    assert len(vehicle['completed_charging_durations_minutes']) == 1
    assert predicted_soc == pytest.approx(vehicle['battery'], abs=1e-6)
    if scale == 1 and soc in (.5, .7):
        assert predicted_soc == pytest.approx(.8)
    if scale == .5:
        assert predicted_soc < .8  # timeout, rather than an assumed full top-up
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
    assert vf._next_epoch_targets(graph)[vid] == pytest.approx(expected)
    env.learning_soc_wait_penalty_per_hour = 80
    assert vf._neuradp_scores(graph, target=False)[edge.edge_id] == pytest.approx(-(4+80)*30/3600)
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
    assert before[3][0, 0] == pytest.approx(.9)
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
    assert checkpoint['extra_value_function_state']['neuradp_implementation_version'] == 2
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
