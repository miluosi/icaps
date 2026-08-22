from types import SimpleNamespace

import pytest

from src.Environment import ChargingIntegratedEnvironment
from src.NYCEnvironment import NYCEnvironment


def _vehicle(vehicle_type):
    return {
        "type": vehicle_type,
        "assigned_request": None,
        "passenger_onboard": None,
        "penalty_timer": 0,
        "is_online": True,
        "rejected_requests": 0,
        "idle_target": None,
        "is_stationary": True,
        "stationary_duration": 1,
        "coordinates": (0, 0),
        "location": 0,
    }


def _minimal_environment(environment_class):
    env = environment_class.__new__(environment_class)
    env.current_time = 4.0
    env.vehicles = {1: _vehicle(1), 2: _vehicle(2)}
    env.active_requests = {
        101: SimpleNamespace(request_id=101, pickup=0),
        202: SimpleNamespace(request_id=202, pickup=0),
    }
    env.rejected_requests = []
    env.ev_requests = []
    env.ev_rejected_request_ids = set()
    env.ev_rejection_times = {}
    env.ev_rejected_recovered_same_epoch_ids = set()
    env._should_reject_request = lambda vehicle_id, request: vehicle_id == 1
    env._record_ev_rejection = lambda vehicle_id: None
    env._record_ev_acceptance = lambda vehicle_id: None
    env._clear_ev_charge_trigger = lambda vehicle_id: None
    env._is_ev = lambda vehicle_id: env.vehicles[vehicle_id]["type"] == 1
    return env


@pytest.mark.parametrize(
    "environment_class",
    [ChargingIntegratedEnvironment, NYCEnvironment],
)
def test_same_epoch_ev_rejection_then_aev_acceptance_is_recourse(
    environment_class,
):
    env = _minimal_environment(environment_class)

    assert env._assign_request_to_vehicle(1, 101) is False
    env.vehicles[1]["assigned_request"] = None
    assert env._assign_request_to_vehicle(2, 101) is True

    assert env.ev_rejected_recovered_same_epoch_ids == {101}
    assert env._record_same_epoch_recourse_if_applicable(2, 101) is False


@pytest.mark.parametrize(
    "environment_class",
    [ChargingIntegratedEnvironment, NYCEnvironment],
)
def test_aev_acceptance_in_a_later_epoch_is_not_recourse(environment_class):
    env = _minimal_environment(environment_class)

    assert env._assign_request_to_vehicle(1, 202) is False
    env.vehicles[1]["assigned_request"] = None
    env.current_time = 5.0
    assert env._assign_request_to_vehicle(2, 202) is True

    assert env.ev_rejected_recovered_same_epoch_ids == set()
