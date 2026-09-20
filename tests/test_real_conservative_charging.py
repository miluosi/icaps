from argparse import Namespace
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import pytest

from benchmark_nyc_charging_rollout import run_case
from src.charging_station import ChargingStation
from src.real_conservative_charging import (
    RealConservativeNYCEnvironment, Reservation, charging_motion_step,
)


def empty_environment():
    env = RealConservativeNYCEnvironment.__new__(RealConservativeNYCEnvironment)
    env._reset_reservations()
    env.real_conservative_active = True
    env.current_time = 0
    env.aev_charging_station_ids = [1]
    env.charging_manager = SimpleNamespace(stations={1: ChargingStation(1, 1, 1)})
    env._is_ev = lambda _: False
    env.enable_real_conservative()
    return env


def test_binding_calendar_is_immutable_and_not_rescheduled():
    env = empty_environment()
    r = Reservation(10, 1, 0, 5, 5, 12)
    env._retain(r)
    with pytest.raises(FrozenInstanceError):
        r.end = 15
    env.current_time = 2
    s = env._build_expected_charging_occupancy(1)
    assert s['intervals'][0]['start_offset'] == 3
    assert s['intervals'][0]['end_offset'] == 10
    assert env.real_reservations[10] is r
    with pytest.raises(RuntimeError, match='overlaps'):
        env._retain(Reservation(11, 1, 0, 11, 11, 15))
    env._retain(Reservation(11, 1, 0, 12, 12, 15))  # half-open boundary


def test_unadmitted_and_redirected_vehicles_are_rejected():
    env = empty_environment()
    with pytest.raises(RuntimeError, match='admission certificate'):
        env._commit_selected(10, 1)
    env._real_admission_epoch = 0
    env._last_conservative_charge = {'virtual_windows': {}}
    with pytest.raises(RuntimeError, match='bypassed'):
        env._commit_selected(10, 1)
    env._retain(Reservation(10, 1, 0, 5, 5, 12))
    with pytest.raises(RuntimeError, match='redirect'):
        env._commit_selected(10, 2)
    with pytest.raises(RuntimeError, match='re-enter'):
        env.generate_vehicle_chargerange([10])


def test_existing_queue_and_unrecorded_arrivals_fail_closed():
    env = empty_environment()
    env.charging_manager.stations[1].charging_queue.append('10')
    with pytest.raises(RuntimeError, match='nonempty'):
        env._check_real_state()
    env.charging_manager.stations[1].charging_queue.clear()
    env.charging_manager.stations[1].charging_queue_notarrived.append('10')
    with pytest.raises(RuntimeError, match='Unrecorded'):
        env._check_real_state()


def test_motion_does_not_arrive_before_reaching_station_coordinate():
    destination = (40.72, -73.99)
    coords, moved, arrived = charging_motion_step((40.70, -73.99), destination, .1)
    assert not arrived and coords != destination and moved == pytest.approx(.1)
    coords, moved, arrived = charging_motion_step(coords, destination, 10.)
    assert arrived and coords == destination


@pytest.mark.parametrize('scale,minimum,battery,expected', [
    (1., 1, .3, 50), (2., 1, .3, 50), (.5, 1, .3, 25),
    (1., 10, .795, 1), (1., 1, .9, 5),
])
def test_service_prediction_matches_timer_or_soc_completion(scale, minimum, battery, expected):
    env = empty_environment()
    env.charge_target_soc = .8
    env.charge_topup_soc = .05
    env.chargeincrease_per_epoch = .01
    env.charge_duration_scale = scale
    env.min_charging_session_epochs = minimum
    env.max_charging_session_epochs = 200
    assert env._service_epochs(battery) == expected
    target = env._charge_target_for_battery(battery)
    timer = env._charge_duration_for_battery(battery)
    # Independent replay of the inherited physical progress loop.
    steps = 0
    while timer > 0:
        steps += 1
        timer -= 1
        battery = min(target, 1., battery + env.chargeincrease_per_epoch)
        if battery >= target - 1e-6:
            timer = 0
    assert steps == expected


@pytest.mark.parametrize('scenario', ['burst', 'staggered'])
def test_real_rollout_arrival_start_and_completion_match_binding_calendar(tmp_path, scenario):
    # Covers plug reuse and simultaneous release/arrival across a full busy horizon.
    args = Namespace(vehicles=1000, hev=500, centers=3, hours=4.)
    rows = run_case(args, 0, scenario, 'real_conservative', tmp_path)
    assert rows[0]['queued_arrival_count'] == 0
    assert rows[0]['mean_wait_minutes_all_arrivals'] == 0
    assert rows[0]['arrival_count'] > 300  # cannot pass by withholding everyone
    audit = json.loads((tmp_path/'reservation_audit.json').read_text())
    reservations = {r['vehicle_id']: r for r in audit['reservations']}
    starts = [e for e in audit['execution_events'] if e['event']=='arrival_start']
    completed = [e for e in audit['execution_events'] if e['event']=='complete']
    assert len(starts) == rows[0]['arrival_count']
    assert len(completed) == rows[0]['completion_count']
    assert all(e['epoch']==reservations[e['vehicle_id']]['arrival'] for e in starts)
    assert all(e['epoch']==reservations[e['vehicle_id']]['end'] for e in completed)
    end_slots = {(r['station_id'], r['plug_index'], r['end']) for r in audit['reservations']}
    assert any((r['station_id'], r['plug_index'], r['start']) in end_slots
               for r in audit['reservations'] if not r['initial'])
