"""Calendar continuity, deployed model clocks and exported metric denominators."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.NYCEnvironment import NYCEnvironment
from src.ValueFunction_pytorch_bayes import PyTorchChargingValueFunction as LegacyValue
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction as GATValue
from src.recourse.lifecycle import RequestLifecycleTracker
from test_recourse_architecture import _request, _vehicle
import test_nyc_model as evaluation


def calendar_env(start=0, stop=24):
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.START_EPOCH, env.STOP_EPOCH, env.EPOCH_LENGTH = start * 3600, stop * 3600, 30
    env.simulation_period = env.episode_length = int((stop - start) * 120)
    env.start_date, env.end_date = '2025-12-15', '2025-12-17'
    env._available_demand_dates = [pd.Timestamp('2025-12-15'), pd.Timestamp('2025-12-17')]
    env._ensure_demand_loaded = lambda: None
    env.current_time = 0
    env.episode_day_index = 0
    return env


@pytest.mark.parametrize('start,stop', [(0, 24), (12, 17)])
def test_each_day_reuses_training_time_features_without_rewinding_physics(start, stop):
    env = calendar_env(start, stop)
    assert env.configure_continuous_evaluation() == 3
    assert env.episode_length == 5760 + (stop-start)*120
    assert len(env._available_demand_dates) == 3  # retain the empty middle day
    value = SimpleNamespace(env=env, episode_length=(stop-start)*120,
                            aligned_inference_episode_length=2880,
                            aligned_inference_time_offset=start*120)
    times = np.array([120, 3000, 5880], dtype=float)
    aligned, length = LegacyValue._get_aligned_time_array(value, times)
    assert aligned.tolist() == [start*120+120]*3
    assert length == 2880
    assert [LegacyValue._get_aligned_time_scalar(value, t)[0] for t in times] == aligned.tolist()
    assert [GATValue._time_norm(value, t) for t in times] == pytest.approx([(start+1)/24]*3)
    for day, t in enumerate(times):
        assert env._current_date_label(t) == pd.Timestamp('2025-12-15') + pd.Timedelta(days=day)
    env.current_time = env.episode_length
    assert env.generate_requests() == []
    assert env.generate_requests_time() == []


def test_default_training_clock_unchanged():
    env = calendar_env()
    value = SimpleNamespace(env=env, episode_length=2880)
    assert LegacyValue._get_aligned_time_scalar(value, 2881) == (2880, 2880)
    assert env.get_hour_of_day(121) == 1


def test_queue_counts_deduplicate_and_exclude_occupants():
    env = calendar_env()
    env.charging_manager = SimpleNamespace(stations={1: SimpleNamespace(
        location=10, max_capacity=50, current_vehicles=['1'],
        charging_queue=[1, '2', 2, 3], charging_queue_notarrived=['3', '4', 4, 5])})
    row = env._compute_zone_charge_station_counts()[10]
    assert row['queue_vehicle_count'] == 4
    assert row['waiting_vehicle_count'] == row['reservation_vehicle_count'] == 2
    env.hourly_zone_charge_station_snapshots = {}
    env._record_hourly_zone_charge_station_snapshot()
    assert next(iter(env.hourly_zone_charge_station_snapshots.values()))['reservation_vehicle_count_sum'] == 2


def test_hourly_queue_means_weight_by_sample_count():
    rows = [dict(date='2025-12-15', hour=0, zone_id=10, snapshot_count=n,
                 mean_queue_vehicle_count=q, mean_waiting_vehicle_count=w,
                 mean_reservation_vehicle_count=q-w) for n,q,w in [(1,4,1),(3,8,2)]]
    combined, _ = evaluation._aggregate_hourly_zone_charge_station_counts([
        {'hourly_zone_charge_station_counts': [row]} for row in rows])
    assert combined[0]['mean_queue_vehicle_count'] == 7
    assert combined[0]['mean_waiting_vehicle_count'] == 1.75
    assert combined[0]['mean_reservation_vehicle_count'] == 5.25


def test_recourse_success_follows_assigned_aev_across_midnight():
    tracker = RequestLifecycleTracker()
    assert tracker.metrics()['recourse_success_rate'] is None
    for rid in (10,11,12):
        tracker.record_offer(transition_id='day1', epoch_id=2879, request=_request(rid),
                             ev_id=0, vehicle=_vehicle(1), acceptance_probability=.2,
                             acceptance_uniform=.9, accepted=False)
        tracker.mark_residual(rid, epoch_id=2879, category='rejected', eligible=True)
        tracker.record_aev_assignment(rid, vehicle_id=1, epoch_id=2879)
    tracker.record_completion(10, epoch_id=2882, vehicle_id=1, vehicle_type=2)
    tracker.record_completion(11, epoch_id=2883, vehicle_id=2, vehicle_type=2)  # different AEV
    metrics = tracker.metrics()
    assert metrics['recourse_assigned_requests'] == 3
    assert metrics['recourse_completed_requests'] == 1
    assert metrics['recourse_uncompleted_requests'] == 2
    assert metrics['recourse_success_rate'] == pytest.approx(1/3)


@pytest.mark.parametrize('adp,structured,expected', [(0,False,[0,1,1]),(1,True,[1,1,1]),(1,False,[1,1,1])])
def test_myopic_wait_strictly_below_minimum(adp, structured, expected):
    env = calendar_env()
    env.adp_value, env._structured_only_planning, env.min_battery_level = adp, structured, .2
    env.vehicles = {i: {'battery':soc} for i,soc in enumerate([.199,.2,.3])}
    assert env.generate_vehicle_wait([0,1,2]).ravel().tolist() == expected


def test_excel_uses_daily_reward_range_counts_and_pooled_success(tmp_path, monkeypatch):
    def run(**kwargs):
        assert kwargs['continuous_evaluation'] is True
        stats = dict(whole_req_num=100, completed_orders=60, completed_ev_orders=40,
                     completed_aev_orders=20, episode_ev_reward=200, episode_aev_reward=100,
                     recourse_requests=20, recourse_assigned_requests=20, recourse_completed_requests=15,
                     recourse_uncompleted_requests=5, lost_requests=30,
                     avg_queue_length_including_reservations=7, avg_queue_length_waiting=2,
                     avg_queue_length_reservations=5, finished_charge=10)
        result = dict(continuous_evaluation=True, evaluation_days=2, episode_rewards=[300],
                      episode_detailed_stats=[stats], daily_evaluation=[])
        env = SimpleNamespace(conservative_charging=False, checkpoint_conservative_charging=False,
                              charging_model_source='test')
        return result, env
    monkeypatch.setattr(evaluation, 'run_nyc_training', run)
    evaluation.main(['--strategies','MCMF','--methods','r1','--start-date','2025-12-15',
                     '--end-date','2025-12-16','--seeds','1','2','--output-dir',str(tmp_path)])
    file = next(tmp_path.glob('*continuous*.xlsx'))
    detail = pd.read_excel(file, sheet_name='detail')
    summary = pd.read_excel(file, sheet_name='summary').iloc[0]
    assert detail['avg_reward'].tolist() == [150,150]
    assert detail['mean_ev_completed_orders'].tolist() == [20,20]
    assert detail['complete'].tolist() == [60,60]
    assert detail['episodes'].tolist() == [2,2]
    assert detail['rollouts'].tolist() == [1,1]
    assert summary['recourse_assigned_requests'] == 40
    assert summary['recourse_success_rate'] == .75
    assert summary['avg_queue_length_including_reservations'] == 7


def test_real_environment_midnight_retains_charge_queue_and_inbound(monkeypatch):
    empty_demand = pd.DataFrame({'pickup_date': pd.to_datetime([]),
                                'pickup_second_of_day': pd.Series(dtype=float),
                                'PULocationID': pd.Series(dtype=int),
                                'DOLocationID': pd.Series(dtype=int)})
    monkeypatch.setattr(NYCEnvironment, '_load_demand_data', lambda self: empty_demand.copy())
    env = NYCEnvironment(num_vehicles=3, ev_num_vehicles=0, random_seed=7,
                         start_date='2025-12-15', end_date='2025-12-16',
                         aev_charging_center_count=3, daily_drop_off=False,
                         ifreject=False, start_hour=0, stop_hour=24)
    env.configure_continuous_evaluation()
    env.reset()
    env.evaluatemode = True
    env.current_time = 2879
    sid = env.aev_charging_station_ids[0]
    station = env.charging_manager.stations[sid]
    station.current_vehicles = ['0']
    station.charging_queue = ['1']
    station.charging_queue_notarrived = ['2']
    station.available_slots = station.max_capacity - 1
    for vid in range(3):
        env.vehicles[vid].update(battery=.15, idle_timer=9,
                                 location=station.location, coordinates=(40.7,-74.0))
    env.vehicles[0].update(charging_station=sid, charging_time_left=10, charge_target_soc=.8)
    env.vehicles[1].update(charging_target=sid)
    env.vehicles[2].update(charging_target=sid)
    active = SimpleNamespace(pickup_deadline=2890)
    env.active_requests[100] = active
    env._update_environment()
    assert env.current_time == 2880
    assert env.get_hour_of_day() == 0
    assert str(env._current_date_label().date()) == '2025-12-16'
    assert env.vehicles[0]['charging_time_left'] == 9
    assert env.vehicles[0]['battery'] == pytest.approx(.15 + env.chargeincrease_per_epoch)
    assert env.vehicles[1]['battery'] == .15
    assert env.vehicles[2]['location'] == station.location
    assert station.current_vehicles == ['0']
    assert station.charging_queue == ['1']
    assert station.charging_queue_notarrived == ['2']
    assert env.active_requests[100] is active
    assert env.vehicles[2]['idle_timer'] == 10


def test_two_day_runner_resets_once_and_keeps_daily_reward_ledger(tmp_path, monkeypatch):
    import run_nyctrainer as runner
    from src.ADPtrainer import ADPTrainer
    empty = pd.DataFrame({'pickup_date': pd.to_datetime([]),
                          'pickup_second_of_day': pd.Series(dtype=float),
                          'PULocationID': pd.Series(dtype=int), 'DOLocationID': pd.Series(dtype=int)})
    monkeypatch.setattr(NYCEnvironment, '_load_demand_data', lambda self: empty.copy())
    env = NYCEnvironment(num_vehicles=2, ev_num_vehicles=0, random_seed=7,
                         start_date='2025-12-15', end_date='2025-12-16',
                         aev_charging_center_count=3, daily_drop_off=False,
                         ifreject=False, start_hour=0, stop_hour=24,
                         epoch_length_sec=3600, episode_length=24)
    env.checkpoint_conservative_charging = False
    env.charging_model_source = 'test'
    resets = []
    original_reset = env.reset
    def reset():
        resets.append(True)
        return original_reset()
    env.reset = reset
    monkeypatch.setattr(runner, '_create_nyc_environment', lambda **kw: env)
    monkeypatch.setattr(runner, 'resolve_nyc_parquet_paths', lambda *a, **kw: [])
    monkeypatch.setattr(ADPTrainer, '_save_episode_stats_to_excel', lambda *a, **kw: (None, None))
    monkeypatch.chdir(tmp_path)
    results, _ = runner.run_nyc_training(
        adpvalue=0, num_episodes=2, use_intense_requests=True,
        assignmentgurobi=True, batch_size=4, num_vehicles=2, num_ev=0,
        heuristic_battery_threshold=.5, transportation_mode='evfirst',
        start_training_episode=0, usemcmf=True, knownreject=False,
        mcmf_use_gpu=False, useauction=False, auction_use_gpu=False,
        auction_epsilon=.001, auction_max_rounds=None, auction_top_k=None,
        ifloadcheckpoint=False, trainnetwork=False, random_seed=7,
        parquet_path=None, start_year_month='2025-12', end_year_month='2025-12',
        start_date='2025-12-15', end_date='2025-12-16', coord_csv=None, station_csv=None,
        start_hour=0, stop_hour=24, epoch_length=3600,
        aev_charging_center_count=3)
    assert len(resets) == 1
    assert results['continuous_evaluation']
    assert results['evaluation_days'] == 2
    assert len(results['episode_rewards']) == 1
    assert env.current_time == 48
    assert len(results['daily_evaluation']) == 2
    assert sum(r['reward'] for r in results['daily_evaluation']) == pytest.approx(sum(results['episode_rewards']))
