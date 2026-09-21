"""Real conservative model namespaces and NYC training/evaluation round trips."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import run_nyctrainer as runner
import test_nyc_model as evaluation
from src.ADPtrainer import ADPTrainer
from src.NYCEnvironment import NYCEnvironment
from src.real_conservative_charging import RealConservativeNYCEnvironment
from src.charging_config import charging_checkpoint_suffix, resolve_nyc_charging_model


def test_real_namespace_and_legacy_metadata():
    assert charging_checkpoint_suffix('test', charging_model='real_conservative') == 'test_charge-real-conservative'
    assert charging_checkpoint_suffix('test_charge-real-conservative', charging_model='real_conservative') == 'test_charge-real-conservative'
    with pytest.raises(ValueError, match='conflicts'):
        charging_checkpoint_suffix('test_charge-conservative', charging_model='real_conservative')
    assert resolve_nyc_charging_model(None, [{}]*2) == ('current','current','checkpoint')
    assert resolve_nyc_charging_model(None, [{'conservative_charging':True}]*2)[0] == 'conservative'
    identity = {'charging_model':'real_conservative','conservative_charging':True}
    assert resolve_nyc_charging_model(None, [identity]*2, 'real_conservative')[0] == 'real_conservative'
    assert resolve_nyc_charging_model('current', [identity]*2, 'real_conservative')[0] == 'current'
    with pytest.raises(ValueError, match='different NYC'):
        resolve_nyc_charging_model(None, [identity, {'conservative_charging':True}])
    with pytest.raises(ValueError, match='namespace'):
        resolve_nyc_charging_model(None, [identity]*2, 'conservative')


def test_training_cli_and_checkpoint_lookup_use_real_namespace(monkeypatch, capsys):
    calls=[]
    monkeypatch.setattr(runner, 'run_nyc_training', lambda **kw: (calls.append(kw) or {}, SimpleNamespace(parquet_path=())))
    runner.main(['--methods','macro','--episodes','1','--charging-model','real_conservative'])
    assert calls[0]['charging_model'] == 'real_conservative'
    assert 'charge-real-conservative_aev-centers-3_method-macro' in calls[0]['checkpoint_suffix']
    with pytest.raises(SystemExit):
        evaluation.main(['--methods','macro','--strategies','ADP-MCMF','--checkpoints-only',
                         '--checkpoint-charging-model','real_conservative'])
    output = capsys.readouterr().out
    assert 'checkpoints/charge-real-conservative/q_networks' in output
    assert 'aev-centers-3_method-macro' in output


def base_kwargs():
    return dict(adpvalue=1, num_episodes=1, use_intense_requests=True,
                assignmentgurobi=True, batch_size=2, num_vehicles=6, num_ev=2,
                heuristic_battery_threshold=.5, transportation_mode='evfirst',
                start_training_episode=0, usemcmf=True, knownreject=False,
                mcmf_use_gpu=False, useauction=False, auction_use_gpu=False,
                auction_epsilon=.001, auction_max_rounds=None, auction_top_k=None,
                ifloadcheckpoint=False, trainnetwork=True, random_seed=7,
                parquet_path=None, start_year_month='2025-12', end_year_month='2025-12',
                start_date='2025-12-15', end_date='2025-12-15', coord_csv=None, station_csv=None,
                start_hour=0, stop_hour=.1, epoch_length=30,
                aev_charging_center_count=3, charging_model='real_conservative',
                recourse_variant='recourse_macro', checkpoint_suffix='smoke',
                prestep=0, training_frequency=1)


def test_train_save_reload_real_model_with_passenger_demand(tmp_path, monkeypatch):
    demand = pd.DataFrame([dict(pickup_date=pd.Timestamp(date), pickup_second_of_day=second,
                               PULocationID=161, DOLocationID=162, fare_amount=15., trip_distance=.1)
                           for date in ['2025-12-15','2025-12-16'] for second in range(0,360,30)])
    monkeypatch.setattr(NYCEnvironment, '_load_demand_data', lambda self: demand.copy())
    monkeypatch.setattr(runner, 'resolve_nyc_parquet_paths', lambda *a, **k: [])
    monkeypatch.setattr(ADPTrainer, '_save_episode_stats_to_excel', lambda *a, **k: (None,None))
    monkeypatch.chdir(tmp_path)
    # Exercise the actual factory, solver, training and checkpoint serialization.
    result, env = runner.run_nyc_training(**base_kwargs())
    assert isinstance(env, RealConservativeNYCEnvironment)
    assert result['charging_model'] == 'real_conservative'
    assert env.real_invariant_checks > 0
    assert result['episode_detailed_stats'][0]['whole_req_num'] > 0
    assert result['optimizer_budget']['optimizer_steps_total'] > 0
    paths = list((tmp_path/'checkpoints'/'charge-real-conservative').glob('*/best_full_state_episode_1.pth'))
    assert len(paths) == 2
    for path in paths:
        assert ADPTrainer._checkpoint_identity(str(path))['charging_model'] == 'real_conservative'
    assert not list((tmp_path/'checkpoints').glob('*charge-conservative*'))
    kwargs = base_kwargs()
    kwargs.update(trainnetwork=False, ifloadcheckpoint=True, charging_model=None,
                  conservative_charging=None, checkpoint_charging_model='real_conservative',
                  checkpoint_suffix='smoke', checkpoint_selection='best_reward')
    loaded, restored = runner.run_nyc_training(**kwargs)
    assert isinstance(restored, RealConservativeNYCEnvironment)
    assert loaded['charging_model'] == loaded['checkpoint_charging_model'] == 'real_conservative'
    assert loaded['charging_model_source'] == 'checkpoint'
    assert restored.real_invariant_checks > 0
    assert loaded['optimizer_budget']['optimizer_steps_total'] == 0
    saved = torch.load(next(p for p in paths if p.parent.name.endswith('_aev')), weights_only=False)
    for name, tensor in restored.value_function.network.state_dict().items():
        assert torch.equal(tensor.cpu(), saved['network_state_dict'][name].cpu())

    # Full two-stage dispatch, with a controlled charging preference to ensure
    # the test exercises actual arrival/start/completion rather than zero events.
    restored.end_date = "2025-12-16"
    restored.configure_continuous_evaluation()
    restored.reset()
    restored.current_time = 2879  # first charge spans midnight
    restored.adp_value = 0
    restored.generate_requests = lambda: []
    sid = restored.aev_charging_station_ids[0]
    station = restored.charging_manager.stations[sid]
    station.max_capacity = station.available_slots = 1
    for vid, vehicle in restored.vehicles.items():
        if not restored._is_ev(vid):
            vehicle.update(location=station.location, zone_id=station.location,
                           coordinates=tuple(restored.zone_coords[station.location]), battery=.79)
    original_scores = restored.generate_vehicle_qvalue_withoutqnetwork
    def prefer_charge(ids):
        scores = original_scores(ids)
        nr = restored._last_matrix_num_requests
        ns = restored._last_matrix_num_stations
        for row, vid in enumerate(ids):
            scores[row, nr:nr+ns] = 1000 if restored.vehicles[vid]['battery'] < .799 else -1000
        return scores
    restored.generate_vehicle_qvalue_withoutqnetwork = prefer_charge
    for _ in range(10):
        actions, aev, ev = restored.simulate_motion_evfirst(agents=[], current_requests=[], rebalance=True)
        restored.step(actions, aev, ev)
    arrivals = [row for row in restored.real_execution_events if row['event']=='arrival_start']
    assert arrivals and all(row['wait'] == 0 for row in arrivals)
    assert any(row['event']=='complete' and row['epoch'] >= 2880 for row in restored.real_execution_events)
    assert str(restored._current_date_label().date()) == '2025-12-16'
    assert restored.get_hour_of_day() < .1
    assert all(not restored.charging_manager.stations[s].charging_queue for s in restored.aev_charging_station_ids)
    import json
    (tmp_path/'smoke_summary.json').write_text(json.dumps({
        'vehicles': 6, 'ev': 2, 'aev': 4, 'training_steps': 12,
        'training_optimizer_updates': result['optimizer_budget']['optimizer_steps_total'],
        'evaluation_optimizer_updates': loaded['optimizer_budget']['optimizer_steps_total'],
        'checkpoint_pairs_saved_and_loaded': 1,
        'charging_model': loaded['charging_model'],
        'weights_match_checkpoint': True,
        'controlled_charging_dispatch_steps': 10,
        'actual_arrivals': len(arrivals),
        'actual_completions': sum(row['event']=='complete' for row in restored.real_execution_events),
        'max_arrival_wait_epochs': max(row['wait'] for row in arrivals),
        'final_aev_physical_queue': sum(len(restored.charging_manager.stations[s].charging_queue) for s in restored.aev_charging_station_ids),
        'midnight_reservation_continuity': True,
    }, indent=2))


