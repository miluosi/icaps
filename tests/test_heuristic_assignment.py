import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.GurobiOptimizer import GurobiOptimizer
from src.NYCEnvironment import NYCEnvironment


def make_request(request_id, pickup, dropoff, value):
    return SimpleNamespace(
        request_id=request_id,
        pickup=pickup,
        dropoff=dropoff,
        final_value=value,
    )


def make_vehicle(vehicle_type, location, battery=1.0):
    return {
        'type': vehicle_type,
        'location': location,
        'battery': battery,
        'salary_ratio': 0.0 if vehicle_type == 1 else 10000.0,
        'assigned_request': None,
        'passenger_onboard': None,
    }


class FakeEnvironment:
    heuristic_battery_threshold = 0.5
    battery_consum = 0.01
    min_battery_level = 0.1
    battery_first = False
    heuevfirst = False
    knownreject = False

    def __init__(self, vehicles, requests, distances):
        self.vehicles = vehicles
        self.active_requests = {request.request_id: request for request in requests}
        self.distances = distances
        self.relocation_target_ids = sorted({
            vehicle['location'] for vehicle in vehicles.values()
        } | {
            request.pickup for request in requests
        })

    def get_distance_km(self, source, target):
        if source == target:
            return 0.0
        return self.distances.get((source, target), 1.0)


class HeuristicAssignmentTest(unittest.TestCase):
    def test_nyc_adp_heuristic_receives_action_matrix_and_qvalues(self):
        vehicle_action_matrix = np.array([[1.0, 1.0]])
        batch_q_value = np.array([[3.0, 0.5]])
        qvalue_heuristic = Mock(return_value={7: "waiting"})
        env = SimpleNamespace(
            adp_value=1.0,
            value_function=object(),
            value_function_ev=object(),
            usemcmf=False,
            assignmentgurobi=False,
            gurobi_network=True,
            heuristic_use_scale=True,
            charging_manager=SimpleNamespace(stations={}),
            active_requests={},
            current_time=0,
            time_stats=defaultdict(list),
            gurobi_optimizer=SimpleNamespace(
                _heuristic_assignment_fastqvalue=qvalue_heuristic,
            ),
            generate_whole_matrix=Mock(
                return_value=(vehicle_action_matrix, 0, 0, 1)
            ),
            generate_vehicle_qvalue=Mock(return_value=batch_q_value),
            generate_vehicle_qvalue_withoutqnetwork=Mock(),
            _should_log_timing=Mock(return_value=False),
        )

        assignments = NYCEnvironment._solve_rebalancing(
            env,
            [7],
            [],
            onlyev=False,
        )

        self.assertEqual(assignments, {7: "waiting"})
        qvalue_heuristic.assert_called_once_with(
            [7],
            [],
            vehicle_action_matrix,
            batch_q_value,
        )

    def test_uses_environment_distance_for_nyc_zone_ids(self):
        requests = [
            make_request(1, 101, 101, 10.0),
            make_request(2, 200, 200, 10.0),
        ]
        env = FakeEnvironment(
            {0: make_vehicle(1, 100)},
            requests,
            {(100, 101): 10.0, (100, 200): 0.5},
        )

        assignments = GurobiOptimizer(env)._heuristic_assignment_with_reject(
            [0], requests)

        self.assertEqual(assignments[0].request_id, 2)

    def test_known_reject_multiplies_ev_request_value(self):
        requests = [
            make_request(1, 101, 101, 100.0),
            make_request(2, 102, 102, 50.0),
        ]
        env = FakeEnvironment({0: make_vehicle(1, 100)}, requests, {})
        env.knownreject = True
        env._calculate_known_rejection_probability = (
            lambda _vehicle_id, request: 0.9 if request.request_id == 1 else 0.0
        )

        assignments = GurobiOptimizer(env)._heuristic_assignment_with_reject(
            [0], requests)

        self.assertEqual(assignments[0].request_id, 2)

    def test_integrated_order_does_not_starve_aev(self):
        requests = [
            make_request(1, 100, 100, 10.0),
            make_request(2, 100, 100, 9.0),
        ]
        vehicles = {
            0: make_vehicle(1, 100),
            1: make_vehicle(1, 100),
            2: make_vehicle(2, 100),
        }
        env = FakeEnvironment(vehicles, requests, {})

        assignments = GurobiOptimizer(env)._heuristic_assignment_with_reject(
            [0, 1, 2], requests)

        self.assertIn(2, assignments)
        self.assertNotIsInstance(assignments[2], str)
        self.assertEqual(assignments[1], "reloc")

    def test_charging_uses_real_distance_and_reserves_capacity(self):
        vehicles = {
            0: make_vehicle(2, 100, battery=0.2),
            1: make_vehicle(2, 100, battery=0.2),
        }
        env = FakeEnvironment(
            vehicles,
            [],
            {(100, 101): 10.0, (100, 200): 0.5},
        )
        stations = [
            SimpleNamespace(id=1, location=101, available_slots=1),
            SimpleNamespace(id=2, location=200, available_slots=1),
        ]

        assignments = GurobiOptimizer(env)._heuristic_assignment_with_reject(
            [0, 1], [], stations)

        self.assertEqual(assignments[0], "charge_2")
        self.assertEqual(assignments[1], "charge_1")

    def test_aev_relocation_returns_real_hotspot_index(self):
        request = make_request(1, 200, 200, 10.0)
        env = FakeEnvironment(
            {0: make_vehicle(2, 100, battery=0.4)},
            [request],
            {(100, 200): 1.0},
        )

        assignments = GurobiOptimizer(env)._heuristic_assignment_with_reject(
            [0], [request])

        self.assertEqual(env.relocation_target_ids, [100, 200])
        self.assertEqual(assignments[0], "idle_at_1")


if __name__ == '__main__':
    unittest.main()
