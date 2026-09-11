"""Safety and over-conservatism of the experimental admission filter."""

from copy import deepcopy

import numpy as np
import pytest

from src.conservative_charging import conservative_charge_mask
from src.expected_charging import (
    build_expected_station_schedule,
    has_complete_charging_window,
)


def _schedule(capacity, current=(), waiting=()):
    return build_expected_station_schedule(
        capacity=capacity, current_jobs=current, waiting_jobs=waiting
    )


def _run(windows, schedule, feasibility=None, station_ids=(7,), vehicle_ids=None):
    if vehicle_ids is None:
        vehicle_ids = sorted({vehicle for vehicle, _ in windows})
    schedules = {station: deepcopy(schedule) for station in station_ids}
    if feasibility is None:
        feasibility = np.array([
            [
                (vehicle, station) in windows and has_complete_charging_window(
                    schedules[station],
                    arrival_offset=windows[vehicle, station]["travel_epochs"],
                    charging_duration=windows[vehicle, station]["charging_duration"],
                )
                for station in station_ids
            ]
            for vehicle in vehicle_ids
        ])
    return conservative_charge_mask(
        vehicle_ids=vehicle_ids,
        station_ids=station_ids,
        feasibility=feasibility,
        candidate_windows=windows,
        station_schedules=schedules,
    )


def test_completion_at_arrival_reuses_slot_without_wait():
    result = _run(
        {(1, 7): {"travel_epochs": 3, "charging_duration": 2}},
        _schedule(1, current=[{"duration": 3}]),
    )
    assert result["feasibility"].tolist() == [[True]]
    assert result["virtual_windows"][1, 7]["start_offset"] == 3


def test_future_commitment_cannot_be_delayed_by_potential_candidate():
    schedule = _schedule(1, waiting=[{"release_offset": 3, "duration": 3}])
    result = _run(
        {
            (1, 7): {"travel_epochs": 0, "charging_duration": 2},
            (2, 7): {"travel_epochs": 2, "charging_duration": 2},
            (3, 7): {"travel_epochs": 6, "charging_duration": 1},
        },
        schedule,
    )
    # Vehicle 2 is background-infeasible but still occupies virtual [6, 8).
    assert [result["virtual_windows"][j, 7]["start_offset"] for j in (1, 2, 3)] == [0, 6, 8]
    assert result["feasibility"].tolist() == [[True], [False], [False]]
    assert schedule["intervals"][0]["start_offset"] == 3


def test_current_queue_and_confirmed_inbound_reservations_remain_fixed():
    schedule = _schedule(
        1,
        current=[{"vehicle_id": 80, "duration": 2}],
        waiting=[
            {"vehicle_id": 81, "release_offset": 0, "duration": 2},
            {"vehicle_id": 82, "release_offset": 6, "duration": 2},
        ],
    )
    result = _run(
        {
            (1, 7): {"travel_epochs": 4, "charging_duration": 2},
            (2, 7): {"travel_epochs": 5, "charging_duration": 2},
        },
        schedule,
    )
    assert result["virtual_windows"][1, 7]["start_offset"] == 4
    assert result["virtual_windows"][2, 7]["start_offset"] == 8
    assert result["station_diagnostics"][7]["committed_jobs"] == 3


def test_same_vehicles_claim_every_station_causing_idle_capacity():
    # Six vehicles can jointly fill six slots, but identical per-station
    # all-potential ordering retains only vehicles 0 and 1 at every station.
    stations = (7, 8, 9)
    result = _run(
        {
            (vehicle, station): {"travel_epochs": 0, "charging_duration": 3}
            for vehicle in range(6) for station in stations
        },
        _schedule(2),
        station_ids=stations,
    )
    assert result["feasibility"].sum(axis=0).tolist() == [2, 2, 2]
    assert np.count_nonzero(result["feasibility"].any(axis=1)) == 2
    # Even an optimal assignment on this filtered graph can occupy at most
    # two of the six slots.  This loss happens before any SSG reduction.


def test_ties_use_vehicle_id_not_input_order_and_slot_id():
    result = _run(
        {(j, 7): {"travel_epochs": 0, "charging_duration": 2} for j in (9, 2, 4)},
        _schedule(2),
        vehicle_ids=(9, 2, 4),
    )
    assert result["feasibility"].tolist() == [[False], [True], [True]]
    assert result["virtual_windows"][2, 7]["plug_index"] == 0
    assert result["virtual_windows"][4, 7]["plug_index"] == 1


def test_inputs_not_mutated_and_mask_does_not_alias_original():
    schedule = _schedule(1, waiting=[{"release_offset": 3, "duration": 2}])
    windows = {(1, 7): {"travel_epochs": 0, "charging_duration": 2}}
    original_schedule = deepcopy(schedule)
    original_windows = deepcopy(windows)
    feasibility = np.ones((1, 1), dtype=float)
    result = conservative_charge_mask(
        vehicle_ids=(1,), station_ids=(7,), feasibility=feasibility,
        candidate_windows=windows, station_schedules={7: schedule},
    )
    result["feasibility"][0, 0] = False
    assert feasibility[0, 0] == 1.0
    assert schedule == original_schedule
    assert windows == original_windows


def test_fifty_slots_and_three_thousand_candidates_have_expected_batch_waits():
    result = _run(
        {(j, 7): {"travel_epochs": 0, "charging_duration": 4} for j in range(3000)},
        _schedule(50),
    )
    assert result["feasibility"].sum() == 50
    assert result["virtual_windows"][50, 7]["start_offset"] == 4
    assert result["virtual_windows"][2999, 7]["start_offset"] == 236
    assert result["station_diagnostics"][7]["virtual_completion_offset"] == 240


def test_retained_virtual_intervals_do_not_conflict_with_hard_or_each_other():
    rng = np.random.default_rng(193)
    schedule = _schedule(
        5,
        current=[{"duration": j + 1} for j in range(3)],
        waiting=[
            {"release_offset": int(rng.integers(0, 20)), "duration": int(rng.integers(1, 8))}
            for _ in range(15)
        ],
    )
    result = _run(
        {
            (j, 7): {"travel_epochs": int(rng.integers(0, 40)), "charging_duration": int(rng.integers(1, 8))}
            for j in range(100)
        },
        schedule,
    )
    for plug in range(5):
        intervals = [
            (item["start_offset"], item["end_offset"])
            for item in schedule["intervals"] if item["plug_index"] == plug
        ]
        for item in result["virtual_windows"].values():
            if item["kept"]:
                assert item["start_offset"] == item["arrival_offset"]
            if item["plug_index"] == plug:
                intervals.append((item["start_offset"], item["end_offset"]))
        intervals.sort()
        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))


def test_zero_capacity_and_missing_candidate_windows_cannot_create_edges():
    result = _run(
        {(1, 7): {"travel_epochs": 0, "charging_duration": 2}},
        _schedule(0), feasibility=np.ones((2, 1)), vehicle_ids=(1, 2),
    )
    assert result["feasibility"].tolist() == [[False], [False]]
    assert result["virtual_windows"][1, 7]["start_offset"] is None


def test_occupancy_without_plug_reservations_is_rejected():
    with pytest.raises(ValueError, match="per-plug"):
        _run(
            {(1, 7): {"travel_epochs": 0, "charging_duration": 1}},
            {"capacity": 1, "occupancy": (1, 0)},
        )
