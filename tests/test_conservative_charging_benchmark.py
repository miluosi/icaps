import numpy as np
import pytest

from benchmark_conservative_charging import (
    fixed_graph_ssg_check, make_case, measure_plan, replay_station, solve_adapter,
)


def test_nyc_opt_in_filters_only_charge_edges_and_retains_all_waits():
    env, ids, _, _ = make_case(20, 2, 3, "synchronized", 0)
    for vehicle in env.vehicles.values():
        vehicle["battery"] = 0.1
    current = env.generate_vehicle_chargerange(ids)
    current_wait = env.generate_vehicle_wait(ids, charge_feasibility=current)
    env.conservative_charging = True
    reduced = env.generate_vehicle_chargerange(ids)
    reduced_wait = env.generate_vehicle_wait(ids, charge_feasibility=reduced)
    assert current.sum() == 40
    assert reduced.sum() == 6
    assert np.all(reduced <= current)
    assert current_wait.sum() == 20
    assert reduced_wait.sum() == 20
    assert all(not station.charging_queue_notarrived
               for station in env.charging_manager.stations.values())


def test_paired_capacity_loss_belongs_to_new_mask_not_ssg():
    env, ids, scores, info = make_case(20, 2, 3, "synchronized", 0)
    outcomes = {}
    for conservative in (False, True):
        env.conservative_charging = conservative
        charge = env.generate_vehicle_chargerange(ids)
        mask = np.column_stack((charge, np.ones((len(ids), 1))))
        assert fixed_graph_ssg_check(mask, scores) == 0
        assignments, stats = solve_adapter(env, ids, mask, scores, True)
        assert assignments is not None
        rows, _ = measure_plan(env._last_expected_charge_expansion, assignments, info, str(conservative))
        assert sum(r["new_queue_vehicle_count"] for r in rows) == 0
        outcomes[conservative] = (sum(r["assigned_charge_vehicles"] for r in rows),
                                 sum(r["idle_slot_epochs"] for r in rows), stats)
    assert outcomes[False][0] == 6
    assert outcomes[True][0] == 3
    assert outcomes[True][1] > outcomes[False][1]
    assert outcomes[True][2]["cut_iterations"] == 0


def test_replay_detects_wait_and_respects_half_open_release_boundary():
    hard = [{"start_offset": 0, "end_offset": 2}]
    safe, _ = replay_station(1, hard, [(0, 2, 3)], 6)
    conflict, _ = replay_station(1, hard, [(0, 1, 3)], 6)
    assert safe["new_queue_vehicle_count"] == 0
    assert conflict["new_queue_vehicle_count"] == 1
    assert conflict["new_queue_wait_epochs"] == 1


def test_replay_detects_delaying_an_existing_reservation():
    hard = [{"start_offset": 2, "end_offset": 3}]
    result, _ = replay_station(1, hard, [(0, 1, 5)], 8)
    assert result["new_queue_vehicle_count"] == 0
    assert result["committed_reservation_delay_epochs"] == 4


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Existing NYC epoch-order gap: movement admits before end-of-epoch charging release")
def test_nyc_movement_boundary_is_not_fixed_by_an_admission_mask(conservative):
    """Known physical-timing limitation, separate from fixed-calendar safety."""
    env, _, _, _ = make_case(2, 1, 1, "synchronized", 0)
    station = env.charging_manager.stations[100]
    station.current_vehicles = ["1"]
    station.available_slots = 0
    env.vehicles[1]["charging_time_left"] = 1
    env.vehicles[1]["location"] = station.location
    env.conservative_charging = conservative
    assert env.generate_vehicle_chargerange([0]).tolist() == [[1.0]]
    # Reproduce a deterministic one-epoch movement ending at the station.
    def arrive(vid, target):
        env.vehicles[vid]["location"] = target
        return 0.0
    env._move_vehicle_one_step = arrive
    env._mark_charging_queue_arrival = lambda *args: None
    env.charging_wait_steps = 0
    env.charging_wait_penalty_total = 0.0
    env._execute_movement_towards_charging_station(0, 100)
    # NYC.step releases vehicle 1 only afterwards, in _update_environment.
    assert station.charging_queue == []
