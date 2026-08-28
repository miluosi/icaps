"""Regression tests for the Human EV charge-decision clock (not its formula)."""

from src.NYCEnvironment import NYCEnvironment, charge_decision_interval_epochs


def _environment(probability=0.0, station_probs=None):
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.current_time = 0.0
    env.human_ev_charge_decision_interval_epochs = 30
    env.rebalance_battery_threshold = 0.15
    env.vehicles = {
        0: {
            "type": 1,
            "battery": 0.8,
            "location": 1,
            "is_online": True,
            "charging_station": None,
            "assigned_request": None,
            "passenger_onboard": None,
            "idle_target": None,
            "target_location": None,
            "no_charge_cooldown_until": 0,
        }
    }
    env._is_ev = lambda vid: True
    env._reachable_charging_station_probs = lambda vid, probs: probs
    env.decision_times = []

    def probability_model(vid):
        env.decision_times.append(env.current_time)
        return probability, (station_probs or {})

    env.compute_ev_charge_probability = probability_model
    env._move_vehicle_to_charging_station = lambda vid, sid: None
    env._charge_duration_for_vehicle = lambda vid: 5
    env._update_storeaction = lambda *args, **kwargs: None
    return env


def test_interval_is_independent_of_epoch_length():
    assert charge_decision_interval_epochs(120, 120) == 60
    assert charge_decision_interval_epochs(120, 30) == 240


def test_skipped_decisions_do_not_extend_the_cooldown():
    env = _environment()
    env._ev_charging_phase({}, {})
    assert env.vehicles[0]["no_charge_cooldown_until"] == 30
    for now in (1, 10, 29):
        env.current_time = now
        env._ev_charging_phase({}, {})
    assert env.decision_times == [0]
    assert env.vehicles[0]["no_charge_cooldown_until"] == 30
    env.current_time = 30
    env._ev_charging_phase({}, {})
    assert env.decision_times == [0, 30]
    assert env.vehicles[0]["no_charge_cooldown_until"] == 60


def test_positive_charge_decision_also_starts_the_interval():
    env = _environment(probability=1.0, station_probs={2: 1.0})
    actions = {}
    env._ev_charging_phase(actions, {})
    assert 0 in actions
    assert env.vehicles[0]["no_charge_cooldown_until"] == 30


def test_low_soc_still_bypasses_the_interval():
    env = _environment()
    env._ev_charging_phase({}, {})
    env.vehicles[0]["battery"] = 0.1
    env.current_time = 1
    env._ev_charging_phase({}, {})
    assert env.decision_times == [0, 1]
    assert env.vehicles[0]["needs_emergency_charging"]
