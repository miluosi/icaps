"""Memory changes must preserve histories, rewards and charging feasibility."""
import gc
import pickle
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from src.Action import IdleAction
from src.NYCEnvironment import NYCEnvironment
from src.compact_history import PositionHistory, ChargingEventHistory
from src.expected_charging import build_charge_action_epoch_expansion


def action_env():
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.vehicles = {0: dict(type=1, location=0, battery=.8, is_stationary=True)}
    env.storeactions = {0: None}
    env.storeactions_ev = {0: None}
    env.current_time = 0
    env.idle_penalty = .08
    env._attach_bootstrap_candidates = lambda *a: None
    return env


@pytest.mark.parametrize('evaluation', [False, True])
def test_ev_actions_and_attached_graphs_are_not_rooted_in_aev_cache(evaluation):
    env = action_env()
    env.evaluatemode = evaluation
    refs = []
    for step in range(8640):  # three full days of 30-second decisions
        env.current_time = step
        action = IdleAction([], (0., 0.), (0., 0.), 0, .8)
        # A placeholder stands for the graph attached by joint collection.
        action.metadata.feasible_graph_snapshot = IdleAction([], (), (), 0, .8)
        refs.append(weakref.ref(action))
        previous = dict(env.storeactions_ev)
        had_previous = previous[0] is not None
        env._update_storeaction(0, action, previous, is_ev=True)
        if had_previous:
            assert previous[0].next_action is action  # training still sees successor
        assert env._execute_action(0, action) == (-.08, -.08)
        assert env.storeactions_ev[0].dur_reward == -.08
        assert env.storeactions[0] is None
    del previous
    gc.collect()
    assert sum(ref() is not None for ref in refs) == 1


@pytest.mark.parametrize('cls,row', [
    (PositionHistory, dict(zone=161, time=2880., coordinates=(40.712345678901, -73.999123456789))),
    (ChargingEventHistory, dict(vehicle_id=2999, station_id=9000001, duration=173.125, time=8640.)),
])
def test_compact_histories_preserve_every_row_and_pickle(cls, row):
    rows = [dict(row, time=i + .125) for i in range(1000)]
    packed = cls(rows)
    assert len(packed) == len(rows)
    assert list(packed) == rows
    assert packed[-1] == rows[-1]
    assert packed[2:5] == rows[2:5]
    assert list(pickle.loads(pickle.dumps(packed))) == rows
    assert len(packed._data) == 32 * len(rows)
    copy = cls()
    copy.extend(packed)
    assert list(copy) == rows


def test_compact_expansion_keeps_exact_solver_metadata_without_dense_tensor():
    kwargs = dict(vehicle_ids=[1], station_ids=[100], feasibility=np.ones((1, 1)),
                  station_schedules={100: dict(capacity=1, occupancy=[1, 0, 0])},
                  candidate_windows={(1, 100): dict(travel_epochs=1, charging_duration=2)})
    dense = build_charge_action_epoch_expansion(**kwargs)
    compact = build_charge_action_epoch_expansion(**kwargs, materialize_epoch_arrays=False)
    assert all(dense[k] == value for k, value in compact.items())
    assert 'action_epoch_mask' not in compact
    # Large future horizons must not allocate a per-vehicle tensor.
    kwargs['candidate_windows'][1, 100]['charging_duration'] = 10**9
    compact = build_charge_action_epoch_expansion(**kwargs, materialize_epoch_arrays=False)
    assert compact['horizon'] == 10**9 + 1


def test_frozen_evaluation_does_not_collect_replay():
    env = action_env()
    env.evaluatemode = True
    def forbidden(*args, **kwargs):
        raise AssertionError('Frozen evaluation attempted to collect replay')
    env.recourse_coordinator = SimpleNamespace(finalize=forbidden)
    assert env._finalize_joint_collection({}, False) is None
    assert env._update_q_learning({}, False) is None


@pytest.mark.parametrize('mode', ['current', 'real_conservative'])
def test_three_day_frozen_neural_rollout_matches_old_storage(tmp_path, monkeypatch, mode):
    import pandas as pd
    import run_nyctrainer as runner
    from src.ADPtrainer import ADPTrainer
    from test_real_conservative_nyc_cli import base_kwargs

    demand = pd.DataFrame([dict(pickup_date=pd.Timestamp(day), pickup_second_of_day=second,
                               PULocationID=161, DOLocationID=162, fare_amount=15., trip_distance=.1)
                           for day in ['2025-12-15', '2025-12-16', '2025-12-17']
                           for second in range(0, 86400, 3600)])
    monkeypatch.setattr(NYCEnvironment, '_load_demand_data', lambda self: demand.copy())
    monkeypatch.setattr(runner, 'resolve_nyc_parquet_paths', lambda *a, **k: [])
    monkeypatch.setattr(ADPTrainer, '_save_episode_stats_to_excel', lambda *a, **k: (None, None))
    monkeypatch.chdir(tmp_path)
    kwargs = base_kwargs()
    kwargs.update(charging_model=mode, epoch_length=3600, stop_hour=24)
    # Build real checkpoints, then compare two frozen-model 72-step rollouts.
    trained, training_env = runner.run_nyc_training(**kwargs)
    del trained, training_env
    gc.collect()
    kwargs.update(trainnetwork=False, ifloadcheckpoint=True, num_episodes=3,
                  end_date='2025-12-17', load_checkpoint_start_date='2025-12-15',
                  load_checkpoint_end_date='2025-12-15',
                  checkpoint_charging_model=mode, continuous_evaluation=True)
    execute = NYCEnvironment._execute_action
    def old_storage(self, vehicle_id, action):
        # Reproduce the original extra root without changing execution.
        if self.storeactions.get(vehicle_id) is None:
            self.storeactions[vehicle_id] = action
            action.dur_reward = 0
            action.current_time = self.current_time
        return execute(self, vehicle_id, action)
    with monkeypatch.context() as old:
        old.setattr(NYCEnvironment, '_execute_action', old_storage)
        old.setattr('src.NYCEnvironment.PositionHistory', list)
        old.setattr('src.NYCtrainer.ChargingEventHistory', list)
        baseline, baseline_env = runner.run_nyc_training(**kwargs)
    result, env = runner.run_nyc_training(**kwargs)
    assert result['evaluation_days'] == 3
    assert result['optimizer_budget']['optimizer_steps_total'] == 0
    assert len(env.value_function.joint_replay_buffer) == 0
    assert len(env.value_function_ev.joint_replay_buffer) == 0
    assert result['episode_rewards'] == baseline['episode_rewards']
    assert result['daily_evaluation'] == baseline['daily_evaluation']
    assert list(result['charging_events']) == baseline['charging_events']
    assert {vid: list(rows) for vid, rows in env.vehicle_position_history.items()} == baseline_env.vehicle_position_history
    for key in ('completed_orders', 'rejected_requests', 'recourse_requests', 'lost_requests', 'avg_wait'):
        assert result['episode_detailed_stats'][0][key] == baseline['episode_detailed_stats'][0][key]


@pytest.mark.parametrize('legacy_serialization', [False, True])
def test_evaluation_loading_old_full_checkpoint_skips_training_payload(tmp_path, monkeypatch, legacy_serialization):
    import torch
    from src.ADPtrainer import ADPTrainer
    from test_nyc_inference_checkpoints import make_value
    from src.memory_lifecycle import load_checkpoint_on_cpu
    source = make_value()
    source.joint_training_step = 37
    source.queue_predictor_trained = True
    source.recent_station_waits = {2: 7.5}
    payload = dict(network_state_dict=source.network.state_dict(),
                   extra_value_function_state=source.extra_checkpoint_state(),
                   optimizer_state_dict={'training_only': True},
                   training_losses=[123.])
    path = tmp_path / 'full_state_episode_1.pth'
    torch.save(payload, path, _use_new_zipfile_serialization=not legacy_serialization)
    assert load_checkpoint_on_cpu(path)['training_losses'] == [123.]
    restored = make_value()
    restored.env = SimpleNamespace(evaluatemode=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('Frozen evaluation restored training-only state')
    monkeypatch.setattr(restored.optimizer, 'load_state_dict', forbidden)
    monkeypatch.setattr(restored.queue_optimizer, 'load_state_dict', forbidden)
    monkeypatch.setattr(restored.joint_replay_buffer, 'load_state_dict', forbidden)
    assert ADPTrainer.__new__(ADPTrainer).load_checkpoint(restored, str(path))
    for name in ('network', 'critic2', 'graph_encoder', 'mixer', 'actor', 'queue_predictor'):
        for key, tensor in getattr(source, name).state_dict().items():
            assert torch.equal(tensor, getattr(restored, name).state_dict()[key])
    assert restored.joint_training_step == 37
    assert restored.queue_predictor_trained
    assert restored.recent_station_waits == {2: 7.5}
    assert len(restored.joint_replay_buffer) == 0
    assert not restored.training_losses
