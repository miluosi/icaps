import numpy as np
import pytest
import random
from collections import deque
from types import SimpleNamespace

from src.Action import IdleAction, ServiceAction
from src.Environment import ChargingIntegratedEnvironment
from src.GurobiOptimizer import GurobiOptimizer
from src.Request import Request


def _minimal_environment():
    env = ChargingIntegratedEnvironment.__new__(ChargingIntegratedEnvironment)
    env.grid_size = 20
    env.num_vehicles = 200
    env.num_zones = 4
    env.simulation_period = 50
    env.episode_length = 50
    env.synthetic_demand_profile = "predictive"
    env.synthetic_pickup_patience_steps = 10
    env.critical_charging_battery = 0.15
    env.queue_forecast_aev_capacity_share = 0.25
    env.lowrequest = False
    env.loc_to_zone = {
        y * env.grid_size + x: (y >= 10) * 2 + (x >= 10)
        for y in range(env.grid_size)
        for x in range(env.grid_size)
    }
    env.active_requests = {}
    env.vehicles = {}
    env._ev_request_q_source_reported = False
    return env


def test_post_demand_target_counts_the_destination_zone_not_exact_cell():
    env = _minimal_environment()
    env.active_requests = {
        1: Request(1, 21, 0, 0, 1),
        2: Request(2, 42, 0, 0, 1),
        3: Request(3, 219, 0, 0, 1),
    }
    env.vehicles = {
        1: {'assigned_request': 2, 'passenger_onboard': None},
    }

    assert env._active_request_count_at_location(63) == 1
    assert env._active_request_count_at_location(250) == 1


def test_predictive_profile_moves_the_peak_between_zones():
    env = _minimal_environment()

    env.current_time = 22
    morning = env._request_generation_settings()
    env.current_time = 40
    evening = env._request_generation_settings()

    assert morning[0] == "morning_peak"
    assert evening[0] == "evening_peak"
    assert morning[4].index(max(morning[4])) == 2
    assert evening[4].index(max(evening[4])) == 1
    assert morning[2] >= 12 and evening[2] >= 14


def test_predictive_dropoffs_follow_noisy_commuting_flow():
    env = _minimal_environment()
    env.hotspot_locations = [(5, 5), (15, 5), (5, 15), (15, 15)]
    random.seed(9)

    destination_zones = [
        env.get_zone_id(y * env.grid_size + x)
        for x, y in (
            env._predictive_dropoff_coordinates("morning_peak")
            for _ in range(200)
        )
    ]

    assert destination_zones.count(3) >= 130


def test_request_priority_caps_idle_but_preserves_charge_score():
    env = _minimal_environment()
    env.request_priority_margin = 1e-3
    q_values = np.array([[4.0, 9.0, 12.0, 20.0, 7.0]])
    action_matrix = np.ones_like(q_values)

    result = env._enforce_predictive_request_priority(
        q_values,
        action_matrix,
        num_requests=2,
        nonservice_columns=[3, 4],
    )

    assert result[0, 2] == 12.0
    assert result[0, 3] < result[0, 1]
    assert result[0, 4] < result[0, 1]


def test_predictive_requests_use_grid_scaled_pickup_patience():
    env = _minimal_environment()
    env.current_time = 7
    request = Request(1, 21, 42, env.current_time, 3)

    env._apply_predictive_request_patience(request)

    assert request.pickup_deadline == 17


def test_predictive_initial_battery_avoids_free_off_peak_heuristic_charge():
    env = _minimal_environment()
    random.seed(3)

    low_cohort = [env._initial_vehicle_battery(vehicle_id) for vehicle_id in (0, 3, 6)]

    assert all(0.35 <= battery <= 0.45 for battery in low_cohort)
    assert all(battery > 0.30 for battery in low_cohort)


def test_queue_forecast_filters_slower_station_actions():
    env = _minimal_environment()
    env.current_time = 5
    env.active_requests = {}
    env.proactive_charging_max_battery = 0.15
    env.battery_consum = 0.001
    env.chargeincrease_whole = 1.0
    env.min_battery_level = 0.2
    env.chargeassignnum = 3
    env.queue_forecast_filter_margin = 0.5
    env.queue_forecast_optional_wait_limit = 2.0
    env.queue_forecast_filtered_actions = 0
    env.queue_forecast_deferred_charges = 0
    env.queue_forecast_reservation_filtered_actions = 0
    env.use_queue_forecast_action_filter = True
    env.station_queue_capacity = 0
    env.vehicles = {0: {"type": 2, "battery": 0.1, "location": 0}}
    stations = {
        index: SimpleNamespace(
            id=index,
            location=index,
            current_vehicles=[],
            charging_queue=[],
            charging_queue_notarrived=[],
            max_capacity=4,
        )
        for index in (1, 2, 3)
    }
    env.charging_manager = SimpleNamespace(stations=stations)

    class QueuePredictor:
        queue_predictor_trained = True

        @staticmethod
        def predict_queue_waits(**kwargs):
            return np.array([3.0, 0.0, 2.0])

    env.value_function = QueuePredictor()

    result = env.generate_vehicle_chargerange([0])

    assert result.tolist() == [[0.0, 1.0, 0.0]]
    assert env.queue_forecast_filtered_actions == 2

    env.value_function.zone_distribution_mode = (
        "st_masac_gat_former2_queue_feature"
    )
    env.queue_forecast_filtered_actions = 0

    feature_only_result = env.generate_vehicle_chargerange([0])

    assert feature_only_result.tolist() == [[1.0, 1.0, 1.0]]
    assert env.queue_forecast_filtered_actions == 0

    stations[1].charging_queue_notarrived[:] = [10, 11, 12, 13]
    capacity_limited_result = env.generate_vehicle_chargerange([0])

    assert capacity_limited_result.tolist() == [[0.0, 1.0, 1.0]]


def test_aev_queue_admission_keeps_full_station_as_a_decision_candidate():
    env = _minimal_environment()
    env.station_queue_capacity = 2
    env.proactive_charging_max_battery = 0.40
    env.battery_consum = 0.001
    env.chargeincrease_whole = 1.0
    env.min_battery_level = 0.2
    env.chargeassignnum = 1
    env.use_queue_forecast_action_filter = False
    env.vehicles = {
        100: {
            "type": 2,
            "battery": 0.30,
            "location": 0,
            "coordinates": (0, 0),
        }
    }
    station = SimpleNamespace(
        id=1,
        location=1,
        current_vehicles=["1", "2"],
        charging_queue=[],
        charging_queue_notarrived=[],
        max_capacity=2,
    )
    env.charging_manager = SimpleNamespace(stations={1: station})
    env.value_function = None

    assert env.generate_vehicle_chargerange([100]).tolist() == [[1.0]]

    station.charging_queue[:] = ["3", "4"]
    assert env.generate_vehicle_chargerange([100]).tolist() == [[0.0]]


def test_critical_aev_cannot_wait_when_queue_admission_is_reachable():
    env = _minimal_environment()
    env.station_queue_capacity = 2
    env.battery_consum = 0.001
    env.vehicles = {
        100: {
            "type": 2,
            "battery": 0.10,
            "location": 0,
            "coordinates": (0, 0),
        }
    }
    env.charging_manager = SimpleNamespace(
        stations={
            1: SimpleNamespace(
                id=1,
                location=1,
                current_vehicles=["1", "2"],
                charging_queue=[],
                charging_queue_notarrived=[],
                max_capacity=2,
            )
        }
    )

    assert env.generate_vehicle_wait([100], rebalance_num=200).tolist() == [[0.0]]


def test_predictive_aev_initial_battery_scale_creates_charge_demand():
    env = _minimal_environment()
    env.ev_num_vehicles = 100
    env.aev_initial_battery_scale = 0.72
    random.seed(3)

    ev_battery = env._initial_vehicle_battery(0)
    random.seed(3)
    aev_battery = env._initial_vehicle_battery(102)

    assert aev_battery == pytest.approx(ev_battery * 0.72)


def test_no_network_q_values_match_former2_myopic_scores():
    env = _minimal_environment()
    env.current_time = 0
    env.knownreject = False
    env.vehicles = {
        0: {
            "type": 2,
            "location": 0,
            "assigned_request": None,
            "passenger_onboard": None,
        }
    }
    env.active_requests = {
        7: SimpleNamespace(
            request_id=7,
            pickup=2,
            dropoff=5,
            final_value=8.0,
        )
    }
    env.charging_manager = SimpleNamespace(
        stations={
            1: SimpleNamespace(id=1, location=4),
        }
    )
    env.hotspot_locations = [(3, 0)]
    env.generate_whole_matrix = lambda *args, **kwargs: (
        np.ones((1, 4), dtype=np.float32),
        1,
        1,
        1,
    )
    env._calculate_rejection_probability = lambda *args, **kwargs: 0.0

    result = env.generate_vehicle_qvalue_withoutqnetwork([0])

    # Myopic scores use the same immediate-reward scale as the environment:
    # request value minus movement cost; charging/relocation/wait are costs.
    np.testing.assert_allclose(result, [[7.975, -0.02, -0.015, -0.005]])


def test_synthetic_ev_request_assignment_uses_learned_value_function():
    env = _minimal_environment()
    env.current_time = 0
    env.knownreject = False
    env.multi_gpu_devices = []
    env.synthetic_ev_myopic_request_q = False
    env.vehicles = {
        0: {
            "type": 1,
            "location": 0,
            "battery": 1.0,
            "idle_timer": 0,
            "assigned_request": None,
            "passenger_onboard": None,
            "charging_station": None,
        }
    }
    env.active_requests = {
        7: SimpleNamespace(
            request_id=7,
            pickup=2,
            dropoff=5,
            final_value=8.0,
        )
    }
    env.generate_whole_matrix = lambda *args, **kwargs: (
        np.ones((1, 2), dtype=np.float32),
        1,
        0,
        0,
    )
    env._manhattan_distance_loc = lambda source, target: abs(target - source)
    env.get_zone_id = lambda location: 0
    env._sample_ev_default_relocation_target = lambda vehicle_id: 1
    env._enforce_predictive_request_priority = (
        lambda q_values, *args, **kwargs: q_values
    )

    class LearnedEVValue:
        def __init__(self):
            self.calls = []

        def batch_get_mixed_q_values(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["action_type_ids"] == [2]:
                assert kwargs["request_values"] == [8.0]
                return [123.0]
            assert kwargs["action_type_ids"] == [1]
            assert kwargs["target_locations"] == [1]
            assert kwargs["target_distances"] == [1.0]
            return [8.0]

    learned_ev = LearnedEVValue()
    env.value_function = SimpleNamespace()
    env.value_function_ev = learned_ev

    result = env.generate_vehicle_qvalue([0], onlyev=True)

    assert len(learned_ev.calls) == 2
    np.testing.assert_allclose(result, [[123.0, 8.0]])


def _synthetic_ev_replay_environment():
    env = _minimal_environment()
    env.current_time = 7
    env.episode_length = 50
    env.done = False
    env.active_requests = {}
    env.vehicles = {
        0: {
            "type": 1,
            "location": 5,
            "battery": 0.8,
            "idle_timer": 0,
            "assigned_request": None,
        }
    }
    env.value_function = SimpleNamespace(experience_buffer=deque())
    env._get_request_final_value = lambda request_id, fallback: 8.0
    env.get_zone_embedding_id = lambda location: int(location)
    env._get_ev_dropout_state_features = lambda vehicle_id: (0.0, 0.0, 0.0)
    return env


def test_synthetic_ev_replay_keeps_request_geometry_and_start_time():
    env = _synthetic_ev_replay_environment()

    class EVReplay:
        def __init__(self):
            self.experience_buffer = deque()

        def store_experience(self, **experience):
            self.experience_buffer.append(experience)

    replay = EVReplay()
    env.value_function_ev = replay
    action = ServiceAction([], 7, 0, 1.0, req_num=4)
    action.target_location = 2
    action.vehicle_loc_post = 5
    action.current_time = 3
    action.dur_time = 4
    action.dur_reward = 8.0
    action.request_value = 8.0
    action.next_action = IdleAction([], (0, 0), (0, 0), 5, 0.8)

    env._update_q_learning({0: action}, ifev=True)

    assert len(replay.experience_buffer) == 1
    experience = replay.experience_buffer[0]
    assert experience["current_time"] == 3.0
    assert experience["dur_time"] == 4.0
    assert experience["target_distance"] == 2.0
    assert experience["post_action_distance"] == 5.0
    assert experience["post_action_location"] == 5


def test_synthetic_ev_post_demand_replay_has_no_duplicate_action_fields():
    env = _synthetic_ev_replay_environment()

    class EVReplay:
        uses_post_demand_feature = True

        def __init__(self):
            self.experience_buffer = deque()

        def store_experience(self, **experience):
            self.experience_buffer.append(experience)

    replay = EVReplay()
    env.value_function_ev = replay
    action = ServiceAction([], 7, 0, 1.0, req_num=4)
    action.target_location = 2
    action.vehicle_loc_post = 5
    action.current_time = 3
    action.dur_time = 4
    action.dur_reward = 8.0
    action.request_value = 8.0
    action.next_action = IdleAction([], (0, 0), (0, 0), 5, 0.8)

    env._update_q_learning({0: action}, ifev=True)

    assert len(replay.experience_buffer) == 1
    experience = replay.experience_buffer[0]
    assert experience["post_action_location"] == 5
    assert experience["post_action_duration"] == 4.0
    assert experience["post_demand_current_time"] == 3.0
    assert "observed_post_demand" in experience


def test_synthetic_ev_relocation_is_written_to_masac_replay():
    env = _synthetic_ev_replay_environment()

    class EVReplay:
        def __init__(self):
            self.experience_buffer = deque()

        def store_experience(self, **experience):
            self.experience_buffer.append(experience)

    replay = EVReplay()
    env.value_function_ev = replay
    action = IdleAction([], (0, 0), (1, 0), 0, 0.8, req_num=4)
    action.learning_action_type = "reloc"
    action.target_location = 1
    action.post_action_location = 1
    action.post_action_distance = 1.0
    action.post_action_duration = 1.0
    action.current_time = 3
    action.dur_time = 1
    action.dur_reward = -0.25
    env.vehicles[0]["location"] = 1
    env.vehicles[0]["battery"] = 0.79

    env._update_q_learning({0: action}, ifev=True)

    assert len(replay.experience_buffer) == 1
    experience = replay.experience_buffer[0]
    assert experience["action_type"] == "reloc"
    assert experience["target_location"] == 1
    assert experience["target_distance"] == 1.0
    assert experience["post_action_location"] == 1
    assert experience["reward"] == -0.25


def test_synthetic_ev_rejection_enters_request_critic_replay():
    env = _synthetic_ev_replay_environment()

    class EVReplay:
        def __init__(self):
            self.experience_buffer = deque()
            self.rejections = []

        def store_experience(self, **experience):
            self.experience_buffer.append(experience)

        def store_rejection_experience(self, **experience):
            self.rejections.append(experience)

    replay = EVReplay()
    env.value_function_ev = replay
    action = ServiceAction([], 7, 0, 1.0, req_num=4)
    action.target_location = 2
    action.vehicle_loc_post = 0
    action.current_time = 3
    action.dur_time = 1
    action.dur_reward = -4.0
    action.was_rejected = True

    env._update_q_learning({0: action}, ifev=True)

    assert len(replay.experience_buffer) == 1
    critic_row = replay.experience_buffer[0]
    assert critic_row["action_type"] == "assign_7"
    assert critic_row["was_rejected"] is True
    assert critic_row["reward"] == -4.0
    assert critic_row["is_system_done"] is False
    assert len(replay.rejections) == 1


def test_solver_vacancy_reserves_inbound_synthetic_charging_capacity():
    optimizer = GurobiOptimizer.__new__(GurobiOptimizer)
    optimizer.env = SimpleNamespace(reserve_inbound_charging_capacity=True)
    station = SimpleNamespace(
        max_capacity=4,
        current_vehicles=[1],
        charging_queue=[2],
        charging_queue_notarrived=[3],
    )

    assert optimizer._charging_station_vacancy(station) == 1

    optimizer.env = SimpleNamespace()
    assert optimizer._charging_station_vacancy(station) == 3


def test_queue_forecast_defers_optional_charge_when_every_station_is_slow():
    env = _minimal_environment()
    env.current_time = 5
    env.active_requests = {}
    env.critical_charging_battery = 0.15
    env.proactive_charging_max_battery = 0.30
    env.battery_consum = 0.001
    env.chargeincrease_whole = 1.0
    env.min_battery_level = 0.2
    env.chargeassignnum = 3
    env.queue_forecast_filter_margin = 0.0
    env.queue_forecast_optional_wait_limit = 2.0
    env.queue_forecast_filtered_actions = 0
    env.queue_forecast_deferred_charges = 0
    env.queue_forecast_reservation_filtered_actions = 0
    env.use_queue_forecast_action_filter = True
    env.vehicles = {0: {"type": 2, "battery": 0.2, "location": 0}}
    env.charging_manager = SimpleNamespace(
        stations={
            1: SimpleNamespace(id=1, location=1, current_vehicles=[], max_capacity=4),
            2: SimpleNamespace(id=2, location=2, current_vehicles=[], max_capacity=4),
        }
    )

    class QueuePredictor:
        queue_predictor_trained = True

        @staticmethod
        def predict_queue_waits(**kwargs):
            return np.array([4.0, 3.0])

    env.value_function = QueuePredictor()

    result = env.generate_vehicle_chargerange([0])

    assert result.tolist() == [[0.0, 0.0]]
    assert env.queue_forecast_filtered_actions == 2
    assert env.queue_forecast_deferred_charges == 1


def test_queue_forecast_never_defers_critical_charge():
    env = _minimal_environment()
    env.current_time = 5
    env.active_requests = {}
    env.critical_charging_battery = 0.15
    env.proactive_charging_max_battery = 0.30
    env.battery_consum = 0.001
    env.chargeincrease_whole = 1.0
    env.min_battery_level = 0.2
    env.chargeassignnum = 2
    env.queue_forecast_filter_margin = 0.0
    env.queue_forecast_optional_wait_limit = 2.0
    env.queue_forecast_filtered_actions = 0
    env.queue_forecast_deferred_charges = 0
    env.queue_forecast_reservation_filtered_actions = 0
    env.use_queue_forecast_action_filter = True
    env.vehicles = {0: {"type": 2, "battery": 0.1, "location": 0}}
    env.charging_manager = SimpleNamespace(
        stations={
            1: SimpleNamespace(id=1, location=1, current_vehicles=[], max_capacity=4),
            2: SimpleNamespace(id=2, location=2, current_vehicles=[], max_capacity=4),
        }
    )

    class QueuePredictor:
        queue_predictor_trained = True

        @staticmethod
        def predict_queue_waits(**kwargs):
            return np.array([4.0, 3.0])

    env.value_function = QueuePredictor()

    result = env.generate_vehicle_chargerange([0])

    assert result.tolist() == [[0.0, 1.0]]
    assert env.queue_forecast_deferred_charges == 0


def test_queue_forecast_reserves_only_the_aev_share_of_station_slots():
    env = _minimal_environment()
    env.current_time = 5
    env.active_requests = {}
    env.proactive_charging_max_battery = 0.40
    env.battery_consum = 0.001
    env.chargeincrease_whole = 1.0
    env.min_battery_level = 0.2
    env.chargeassignnum = 1
    env.queue_forecast_filter_margin = 0.0
    env.queue_forecast_optional_wait_limit = 0.0
    env.queue_forecast_filtered_actions = 0
    env.queue_forecast_deferred_charges = 0
    env.queue_forecast_reservation_filtered_actions = 0
    env.use_queue_forecast_action_filter = True
    env.vehicles = {
        vehicle_id: {"type": 2, "battery": 0.2 + vehicle_id * 0.01, "location": 0}
        for vehicle_id in range(4)
    }
    env.charging_manager = SimpleNamespace(
        stations={
            1: SimpleNamespace(
                id=1,
                location=1,
                current_vehicles=[],
                charging_queue=[],
                charging_queue_notarrived=[],
                max_capacity=4,
            )
        }
    )

    class QueuePredictor:
        queue_predictor_trained = True

        @staticmethod
        def predict_queue_waits(**kwargs):
            return np.zeros(len(kwargs["vehicle_ids"]))

    env.value_function = QueuePredictor()

    result = env.generate_vehicle_chargerange([0, 1, 2, 3])

    assert int(result.sum()) == 1
    assert result[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_demand_forecast_keeps_the_best_relocation_zone():
    env = _minimal_environment()
    env.current_time = 10
    env.hotspot_locations = [(5, 5), (15, 5), (5, 15), (15, 15)]
    env.hotspot_locations_num = 4
    env.battery_consum = 0.001
    env.min_battery_level = 0.2
    env.demand_forecast_filter_margin = 0.01
    env.demand_forecast_filtered_actions = 0
    env.vehicles = {0: {"battery": 0.8, "location": 10 * 20 + 10}}

    class DemandPredictor:
        post_demand_predictor_trained = True

        @staticmethod
        def predict_post_action_demand(**kwargs):
            return np.array([0.1, 0.8, 0.2, 0.3])

    env.value_function = DemandPredictor()

    result = env.generate_vehicle_zone([0], distance_threshold=20)

    assert result.tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert env.demand_forecast_filtered_actions == 3


def test_optional_charging_stays_below_a_feasible_request():
    env = _minimal_environment()
    env.request_priority_margin = 1e-3
    env.critical_charging_battery = 0.15
    env.vehicles = {0: {"battery": 0.25}}
    q_values = np.array([[8.0, 12.0, 7.0]])
    action_matrix = np.ones_like(q_values)

    result = env._enforce_predictive_proactive_charge_priority(
        q_values,
        action_matrix,
        vehicle_ids=[0],
        num_requests=1,
        num_stations=2,
    )

    assert result[0, 1] < result[0, 0]
    assert result[0, 2] == 7.0
