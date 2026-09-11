from types import SimpleNamespace

import numpy as np
import pytest

from src.GurobiOptimizer import GurobiOptimizer
from src.NYCEnvironment import NYCEnvironment


def _nyc_expected_environment(*, current_remaining=2, inbound=False):
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.current_time = 10.0
    env.battery_consum = 0.01
    env.min_battery_level = 0.2
    env.chargeincrease_per_epoch = 0.1
    env.charge_duration_scale = 1.0
    env.charge_target_soc = 0.8
    env.charge_topup_soc = 0.05
    env.min_charging_session_epochs = 1
    env.max_charging_session_epochs = 20
    env.charge_duration = 6
    env.charge_action_range_km = None
    env.charge_top_k = None
    env.charge_wait_bool = True
    env.vehicles = {
        0: {"type": 2, "location": 0, "battery": 0.50},
        1: {
            "type": 2,
            "location": 1,
            "battery": 0.50,
            "charging_time_left": current_remaining,
        },
        2: {"type": 2, "location": 2, "battery": 0.50},
    }
    station = SimpleNamespace(
        id=7,
        location=1,
        max_capacity=1,
        current_vehicles=["1"],
        charging_queue=[],
        charging_queue_notarrived=["2"] if inbound else [],
    )
    env.charging_manager = SimpleNamespace(stations={7: station})
    env.station_zone_ids = np.asarray([1], dtype=np.int32)
    env.distance_matrix = np.zeros((3, 3), dtype=np.float32)
    env.distance_matrix[0, 1] = 1.0
    env.distance_matrix[2, 1] = 1.0
    env.get_distance_km = lambda origin, destination: float(
        env.distance_matrix[int(origin), int(destination)]
    )
    env.get_travel_time = lambda origin, destination: {
        0: 3.0,
        1: 0.0,
        2: 4.0,
    }[int(origin)]
    env._is_ev = lambda vehicle_id: env.vehicles[vehicle_id]["type"] == 1
    return env


def test_expected_arrival_and_completion_use_vehicle_specific_future_epoch():
    env = _nyc_expected_environment(current_remaining=2)

    expected = env.calculate_expected_entercharge_station_time_battery(0, 7)

    assert expected["travel_epochs"] == 3
    assert expected["expected_entercharge_station_time"] == 13.0
    assert expected["expected_charge_completion_time"] == 17.0
    assert expected["charging_duration"] == 4


def test_default_charge_matrix_checks_only_arrival_and_retains_physical_duration():
    env = _nyc_expected_environment(current_remaining=2)

    charge_matrix = env.generate_vehicle_chargerange([0])
    expansion = env._last_expected_charge_expansion

    assert charge_matrix.shape == (1, 1)
    assert charge_matrix.tolist() == [[1.0]]
    assert expansion["action_epoch_mask"].shape == (1, 1, 7)
    assert expansion["action_epoch_mask"][0, 0].tolist() == [
        0, 0, 0, 1, 0, 0, 0
    ]
    assert expansion["capacity_scope"] == "arrival"
    assert expansion["candidate_windows"][(0, 7)]["charging_duration"] == 4
    assert expansion["base_occupancy"][0, :2].tolist() == [1, 1]


def test_default_rejects_current_or_booked_occupancy_at_arrival():
    env = _nyc_expected_environment(current_remaining=5)

    assert env.generate_vehicle_chargerange([0]).tolist() == [[0.0]]

    env = _nyc_expected_environment(current_remaining=2, inbound=True)
    env.get_travel_time = lambda origin, destination: 3.0
    assert env.generate_vehicle_chargerange([0]).tolist() == [[0.0]]


def test_later_booking_does_not_block_default_but_blocks_conservative():
    env = _nyc_expected_environment(current_remaining=2, inbound=True)
    # Candidate arrives at 3; the committed inbound vehicle starts at 4.
    assert env.generate_vehicle_chargerange([0]).tolist() == [[1.0]]
    env.conservative_charging = True
    assert env.generate_vehicle_chargerange([0]).tolist() == [[0.0]]
    assert env._last_expected_charge_expansion["capacity_scope"] == "full_window"


def test_default_releases_completed_charging_and_bookings_at_arrival():
    env = _nyc_expected_environment(current_remaining=3)
    assert env.generate_vehicle_chargerange([0]).tolist() == [[1.0]]
    env = _nyc_expected_environment(current_remaining=1, inbound=True)
    env.get_travel_time = lambda origin, destination: {0: 6.0, 2: 2.0}[origin]
    assert env.generate_vehicle_chargerange([0]).tolist() == [[1.0]]


def test_default_does_not_reserve_other_potential_candidates():
    env = _nyc_expected_environment(current_remaining=2)
    station = env.charging_manager.stations[7]
    assert env.generate_vehicle_chargerange([0, 2]).tolist() == [[1.0], [1.0]]
    intervals = env._last_expected_charge_expansion["station_schedules"][7]["intervals"]
    assert [item["vehicle_id"] for item in intervals] == [1]
    assert station.charging_queue_notarrived == []


def test_default_includes_committed_already_arrived_queue():
    env = _nyc_expected_environment(current_remaining=2)
    env.charging_manager.stations[7].charging_queue = ["2"]
    assert env.generate_vehicle_chargerange([0]).tolist() == [[0.0]]


def test_aev_wait_remains_feasible_for_every_epoch_and_soc():
    env = _nyc_expected_environment(current_remaining=20)
    env.vehicles[0]["battery"] = 0.01

    assert env.generate_vehicle_wait([0]).tolist() == [[1.0]]


def test_low_battery_wait_is_independent_of_charge_window():
    env = _nyc_expected_environment(current_remaining=2)
    env.min_battery_level = 0.50
    charge_matrix = env.generate_vehicle_chargerange([0])
    assert charge_matrix.tolist() == [[1.0]]
    assert env.generate_vehicle_wait(
        [0], charge_feasibility=charge_matrix
    ).tolist() == [[1.0]]

    env = _nyc_expected_environment(current_remaining=5)
    env.min_battery_level = 0.50
    charge_matrix = env.generate_vehicle_chargerange([0])
    assert charge_matrix.tolist() == [[0.0]]
    assert env.generate_vehicle_wait(
        [0], charge_feasibility=charge_matrix
    ).tolist() == [[1.0]]


def test_mcmf_capacity_cut_distinguishes_overlapping_candidate_windows():
    env = SimpleNamespace(
        vehicles={0: {"type": 2}, 1: {"type": 2}},
        _last_expected_charge_expansion={
            "vehicle_ids": (0, 1),
            "candidate_windows": {
                (0, 7): {"travel_epochs": 3, "charging_duration": 4},
                (1, 7): {"travel_epochs": 4, "charging_duration": 2},
            },
            "station_schedules": {
                7: {"capacity": 1, "occupancy": (0, 0, 0, 0, 0, 0, 0)},
            },
        },
    )
    optimizer = GurobiOptimizer.__new__(GurobiOptimizer)
    optimizer.env = env
    layout = {
        "num_requests": 0,
        "num_charging": 1,
        "charge_station_ids": [7],
    }
    feasibility = np.asarray([[1, 1], [1, 1]], dtype=bool)
    q_values = np.asarray([[10.0, 0.0], [9.0, 8.5]])

    disabled = optimizer._expected_charge_edges_to_disable(
        np.asarray([0, 0]),
        [0, 1],
        layout,
        feasibility,
        q_values,
    )

    assert disabled == [(1, 0)]


@pytest.mark.parametrize("scope,second_arrival,second_action,cuts", [
    ("full_window", 4, "waiting", 1),
    ("arrival", 4, "charge_7", 0),
    ("arrival", 3, "waiting", 1),
])
def test_exact_mcmf_enforces_active_capacity_scope(
    scope, second_arrival, second_action, cuts,
):
    station = SimpleNamespace(
        id=7,
        max_capacity=1,
        current_vehicles=[],
        charging_queue=[],
        charging_queue_notarrived=[],
        available_slots=1,
    )
    env = SimpleNamespace(
        current_time=0.0,
        vehicles={0: {"type": 2}, 1: {"type": 2}},
        charging_manager=SimpleNamespace(stations={7: station}),
        station_queue_capacity=0,
        reserve_inbound_charging_capacity=False,
        mcmf_solver="ortools",
        mcmf_backend="ortools",
        mcmf_strict=True,
        mcmf_cost_scale=10_000,
        mcmf_graph_reduction=True,
        mcmf_verify=True,
        mip_backend="docplex",
        _last_matrix_num_requests=0,
        _last_matrix_num_stations=1,
        _last_matrix_num_zones=0,
        _last_matrix_charge_station_ids=[7],
        _last_matrix_zone_indices=[],
        _last_matrix_zone_target_ids=[],
        _last_expected_charge_expansion={
            "capacity_scope": scope,
            "current_time": 0.0,
            "vehicle_ids": (0, 1),
            "candidate_windows": {
                (0, 7): {"travel_epochs": 3, "charging_duration": 4},
                (1, 7): {"travel_epochs": second_arrival, "charging_duration": 2},
            },
            "station_schedules": {
                7: {"capacity": 1, "occupancy": (0, 0, 0, 0, 0, 0, 0)},
            },
        },
    )
    optimizer = GurobiOptimizer(env, num_threads=1)
    legacy_matrix = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    q_values = np.asarray([[10.0, 0.0], [9.0, 8.5]])

    assignments = optimizer._exact_vehicle_rebalancing_network(
        [0, 1], [], legacy_matrix, q_values
    )

    assert assignments == {0: "charge_7", 1: second_action}
    assert env.expected_charge_capacity_cut_iterations == cuts
