"""EV charging commitments survive dispatch, queues and a daily boundary."""
from copy import deepcopy

import pandas as pd
import pytest

from src.Action import ChargingAction
from src.NYCEnvironment import NYCEnvironment
from src.real_conservative_charging import RealConservativeNYCEnvironment


@pytest.fixture
def env(monkeypatch):
    empty = pd.DataFrame({'pickup_date': pd.to_datetime([]),
                         'pickup_second_of_day': pd.Series(dtype=float),
                         'PULocationID': pd.Series(dtype=int), 'DOLocationID': pd.Series(dtype=int)})
    monkeypatch.setattr(NYCEnvironment, '_load_demand_data', lambda self: empty.copy())
    e = NYCEnvironment(num_vehicles=2, ev_num_vehicles=2, random_seed=17,
                       start_date='2025-12-15', end_date='2025-12-16',
                       aev_charging_center_count=3, daily_drop_off=False, ifreject=False)
    e.configure_continuous_evaluation()
    e.reset()
    e.evaluatemode = True
    e.adp_value = 0
    e.configure_recourse_experiment('recourse_macro')
    e.current_time = 2878
    e.chargeincrease_per_epoch = .02
    sid = e.public_charging_station_ids[0]
    station = e.charging_manager.stations[sid]
    station.max_capacity = station.available_slots = 1
    station.log_queue_arrivals = False
    origin = min((z for z in e.manhattan_zone_ids if z != station.location),
                 key=lambda z: e.get_distance_km(z, station.location))
    for v in e.vehicles.values():
        v.update(location=origin, zone_id=origin, coordinates=e.zone_coords[origin],
                 battery=.2, assigned_request=None, passenger_onboard=None,
                 charging_target=None, charging_station=None, idle_target=None,
                 target_location=None, is_stationary=False,
                 no_charge_cooldown_until=100000, needs_emergency_charging=False)
    # P=0 isolates the safety rule: SOC at its service floor must still charge.
    monkeypatch.setattr(e, 'compute_ev_charge_probability', lambda vid:(0., {sid:1.}))
    e.test_station_id = sid
    return e


def test_update_does_not_make_unexecuted_charging_decisions(env, monkeypatch):
    def forbidden(*args):
        raise AssertionError('update must not choose a driver action')
    monkeypatch.setattr(env, '_ev_charging_phase', forbidden)
    before = deepcopy(env.vehicles)
    env._update_environment()
    for vid in env.vehicles:
        for key in ('charging_target', 'target_location', 'idle_target', 'no_charge_cooldown_until'):
            assert env.vehicles[vid][key] == before[vid][key]


def test_low_soc_interrupts_empty_relocation_and_commits_station(env, monkeypatch):
    v = env.vehicles[0]
    v.update(idle_target=161, target_location=161, is_stationary=True)
    actions = {}
    env._ev_charging_phase(actions, dict(env.storeactions_ev))
    assert isinstance(actions[0], ChargingAction)
    assert v['charging_target'] == env.test_station_id
    assert v['idle_target'] is None and not v['is_stationary']
    assert 0 not in env._build_vehicles_to_rebalance([0])
    def forbidden(*args):
        raise AssertionError('a committed trip must not resample the driver choice')
    monkeypatch.setattr(env, 'compute_ev_charge_probability', forbidden)
    env._ev_charging_phase({}, dict(env.storeactions_ev))


@pytest.mark.parametrize('mode', ['current', 'conservative', 'real_conservative'])
@pytest.mark.parametrize('occupied', [False, True])
def test_ev_reaches_queues_and_finishes_across_midnight(env, mode, occupied):
    if mode == 'conservative':
        env.conservative_charging = True
    elif mode == 'real_conservative':
        # Use the real model's execution/update hooks; it controls AEVs only.
        env.__class__ = RealConservativeNYCEnvironment
        env._reset_reservations()
        env.real_conservative_active = True
        env.conservative_charging = True
    sid = env.test_station_id
    station = env.charging_manager.stations[sid]
    occupant = env.vehicles[0]
    if occupied:
        assert station.start_charging('0')
        occupant.update(location=station.location, zone_id=station.location,
                        coordinates=env.zone_coords[station.location], battery=.2,
                        charging_station=sid, charging_time_left=50,
                        charge_target_soc=.8, charging_session_start_time=env.current_time)
    else:
        occupant['is_online'] = False
    initial_distance = env.vehicles[1]['total_distance']
    saw_queue = False
    traction = 0.
    movement = 0.
    finished = False
    for _ in range(80):
        actions, aev, ev = env.simulate_motion_evfirst(agents=[], current_requests=[], rebalance=True)
        if not finished:
            assert isinstance(actions[1], ChargingAction)
            assert actions[1].charging_station_id == sid
        _, _, _, _, info = env.step(actions, aev, ev)
        saw_queue |= '1' in station.charging_queue
        traction += info['physical_motion']['EV']['traction_soc']
        movement += info['physical_motion']['EV']['distance_km']
        v = env.vehicles[1]
        if v.get('completed_charging_durations_minutes'):
            finished = True
            break
    assert finished
    assert saw_queue == occupied
    assert env.vehicles[1]['battery'] >= .8 - 1e-6
    assert env.vehicles[1]['total_distance'] > initial_distance
    assert traction == pytest.approx(movement * env.battery_consum, abs=1e-9)
    assert env.vehicles[1]['charging_target'] is None
    assert '1' not in station.charging_queue and '1' not in station.current_vehicles
    assert str(env._current_date_label().date()) == '2025-12-16'
    assert env.get_hour_of_day() < 1


def test_real_movement_consumes_battery_but_local_wait_does_not(env):
    v = env.vehicles[1]
    sid = env.test_station_id
    station = env.charging_manager.stations[sid]
    battery = v['battery']
    distance = v['total_distance']
    coordinates = tuple(v['coordinates'])
    moved = env._move_vehicle_one_step(1, station.location)
    assert moved > 0 and tuple(v['coordinates']) != coordinates
    assert v['total_distance'] - distance == pytest.approx(moved)
    assert battery - v['battery'] == pytest.approx(moved * env.battery_consum)
    # At the service floor the EV's nominal relocation is an actual local wait.
    v['battery'] = .2
    target = env._sample_ev_default_relocation_target(1)
    assert target == v['location']
    before = deepcopy(v)
    env._execute_movement_towards_idle(1, target)
    assert v['battery'] == before['battery']
    assert v['total_distance'] == before['total_distance']
    assert v['coordinates'] == before['coordinates']
