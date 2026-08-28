import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.charging_metrics import charging_session_metrics
from src.NYCEnvironment import (
    NYCEnvironment,
    charge_decision_interval_epochs,
    initial_battery_bounds,
)
from test_charging_sensitivity import (
    DEFAULT_CONSUMPTION_RATIOS,
    build_summary,
    build_trend_summary,
    summarize_run,
    validate_consumption_ratios,
)


def test_charging_session_metrics_are_type_specific_and_day_normalized():
    vehicles = {
        0: {
            "type": 1,
            "charging_count": 2,
            "completed_charging_durations_minutes": [30.0, 60.0],
        },
        1: {
            "type": 1,
            "charging_count": 4,
            "completed_charging_durations_minutes": [45.0],
        },
        2: {
            "type": 2,
            "charging_count": 6,
            "completed_charging_durations_minutes": [20.0, 40.0, 60.0],
        },
    }

    metrics = charging_session_metrics(vehicles, simulated_days=2.0)

    assert metrics["human_ev_charging_sessions"] == 6
    assert metrics["aev_charging_sessions"] == 6
    assert metrics["all_vehicle_charging_sessions"] == 12
    assert metrics["avg_daily_charging_sessions_per_human_ev"] == pytest.approx(1.5)
    assert metrics["avg_daily_charging_sessions_per_aev"] == pytest.approx(3.0)
    assert metrics["avg_daily_charging_sessions_per_vehicle"] == pytest.approx(2.0)
    assert metrics["avg_charging_session_duration_minutes_human_ev"] == pytest.approx(45.0)
    assert metrics["avg_charging_session_duration_minutes_aev"] == pytest.approx(40.0)
    assert metrics["avg_charging_session_duration_minutes_all"] == pytest.approx(42.5)

def test_consumption_ratio_grid_matches_requested_multipliers():
    assert validate_consumption_ratios(DEFAULT_CONSUMPTION_RATIOS) == pytest.approx(
        (0.8, 0.9025, 0.95, 1.0, 1.05, 1.1025, 1.2)
    )


def test_initial_battery_mean_preserves_legacy_and_supports_lower_soc():
    assert initial_battery_bounds(0.875) == pytest.approx((0.8, 0.95))
    assert initial_battery_bounds(0.45) == pytest.approx((0.375, 0.525))
    assert initial_battery_bounds(0.10) == pytest.approx((0.025, 0.175))


def test_initial_battery_mean_rejects_invalid_values():
    with pytest.raises(ValueError):
        initial_battery_bounds(0.0)
    with pytest.raises(ValueError):
        initial_battery_bounds(1.01)


def test_human_ev_charge_decision_interval_uses_real_time():
    assert charge_decision_interval_epochs(60.0, 120.0) == 30
    assert charge_decision_interval_epochs(180.0, 120.0) == 90
    assert charge_decision_interval_epochs(60.0, 30.0) == 120


def test_human_ev_charge_decision_interval_rejects_invalid_values():
    with pytest.raises(ValueError):
        charge_decision_interval_epochs(-1.0, 120.0)
    with pytest.raises(ValueError):
        charge_decision_interval_epochs(60.0, 0.0)


def test_ratio_summary_refuses_mixed_decision_intervals():
    with pytest.raises(ValueError, match="Do not pool"):
        build_summary(pd.DataFrame({
            "human_ev_charge_decision_interval_minutes": [60.0, 120.0],
        }))


def _wait_test_environment(charge_wait_bool=True):
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.charge_wait_bool = charge_wait_bool
    env.charge_action_range_km = 5.0
    env.battery_consum = 0.01
    env.vehicles = {
        1: {"type": 1, "location": 10, "battery": 0.50},
        2: {"type": 2, "location": 10, "battery": 0.20},
        3: {"type": 2, "location": 30, "battery": 0.20},
        4: {"type": 2, "location": 10, "battery": 0.30},
    }
    station = SimpleNamespace(
        location=20,
        max_capacity=2,
        current_vehicles=["9"],
        charging_queue_notarrived=[],
    )
    env.charging_manager = SimpleNamespace(stations={0: station})
    env.get_distance_km = lambda origin, destination: {
        (10, 20): 4.0,
        (30, 20): 8.0,
    }[(origin, destination)]
    return env


def test_wait_feasibility_uses_reachable_current_charging_capacity():
    env = _wait_test_environment(charge_wait_bool=True)
    wait = env.generate_vehicle_wait([1, 2, 3, 4])
    np.testing.assert_array_equal(
        wait,
        np.asarray([[1.0], [0.0], [1.0], [1.0]], dtype=np.float32),
    )
    assert env.generate_capacity_charge(env.vehicles[2]) == 1
    assert env.generate_capacity_charge(env.vehicles[3]) == 0


def test_disabling_charge_wait_gate_restores_all_one_wait_column():
    env = _wait_test_environment(charge_wait_bool=False)
    np.testing.assert_array_equal(
        env.generate_vehicle_wait([1, 2, 3]),
        np.ones((3, 1), dtype=np.float32),
    )


def test_notarrived_aev_reservations_reduce_current_charging_capacity():
    env = _wait_test_environment(charge_wait_bool=True)
    station = env.charging_manager.stations[0]
    station.charging_queue_notarrived.append("88")
    assert env.generate_capacity_charge(env.vehicles[2]) == 0
    np.testing.assert_array_equal(
        env.generate_vehicle_wait([2]),
        np.ones((1, 1), dtype=np.float32),
    )


def test_sensitivity_summary_converts_positive_wait_steps_to_minutes():
    stats = [{
        "charging_observation_days": 1.0,
        "human_ev_vehicle_count": 100,
        "aev_vehicle_count": 100,
        "all_vehicle_count": 200,
        "human_ev_charging_sessions": 20,
        "aev_charging_sessions": 40,
        "all_vehicle_charging_sessions": 60,
        "completed_charging_sessions_with_duration_human_ev": 2,
        "completed_charging_sessions_with_duration_aev": 2,
        "completed_charging_sessions_with_duration_all": 4,
        "avg_charging_session_duration_minutes_human_ev": 30.0,
        "avg_charging_session_duration_minutes_aev": 40.0,
        "avg_charging_session_duration_minutes_all": 35.0,
        "avg_wait": 3.0,
        "waiting_vehicle_count": 2,
        "completed_orders": 10,
        "service_ratio": 0.5,
        "avg_battery_level": 0.6,
    }]
    row = summarize_run(stats, ratio=1.0, seed=256, epoch_length_sec=30.0)
    assert row["avg_daily_charging_sessions_per_human_ev"] == pytest.approx(0.2)
    assert row["avg_daily_charging_sessions_per_aev"] == pytest.approx(0.4)
    assert row["avg_wait_minutes_waiting_charging_vehicles"] == pytest.approx(1.5)


def test_build_summary_reports_change_from_unit_ratio():
    detail = pd.DataFrame([
        {
            "battery_consumption_ratio": ratio,
            "effective_consumption_wh_per_mile": 230.0 * ratio,
            "effective_consumption_kwh_per_km": 0.1429 * ratio,
            "seed": 256,
            "avg_daily_charging_sessions_per_human_ev": human,
            "avg_daily_charging_sessions_per_aev": human,
            "avg_daily_charging_sessions_per_vehicle": human,
            "avg_charging_session_duration_minutes_human_ev": 30.0,
            "avg_charging_session_duration_minutes_aev": 30.0,
            "avg_charging_session_duration_minutes_all": 30.0,
            "avg_wait_steps_waiting_charging_vehicles": 2.0,
            "avg_wait_minutes_waiting_charging_vehicles": 1.0,
            "waiting_charging_vehicle_count": 2.0,
            "mean_completed_orders": 10.0,
            "mean_service_ratio": 0.5,
            "mean_final_battery_soc": 0.6,
        }
        for ratio, human in ((0.95, 0.9), (1.0, 1.0), (1.05, 1.1))
    ])
    summary = build_summary(detail)
    high = summary[np.isclose(summary["battery_consumption_ratio"], 1.05)].iloc[0]
    assert high[
        "pct_change_vs_baseline_avg_daily_charging_sessions_per_human_ev"
    ] == pytest.approx(10.0)


def test_trend_summary_reports_endpoint_change_and_linear_fit():
    detail = pd.DataFrame([
        {
            "battery_consumption_ratio": ratio,
            "effective_consumption_wh_per_mile": 230.0 * ratio,
            "effective_consumption_kwh_per_km": 0.1429 * ratio,
            "seed": 256,
            "avg_daily_charging_sessions_per_human_ev": value,
            "avg_daily_charging_sessions_per_aev": 0.0,
            "avg_daily_charging_sessions_per_vehicle": value / 2.0,
            "avg_charging_session_duration_minutes_human_ev": value * 10.0,
            "avg_charging_session_duration_minutes_aev": 0.0,
            "avg_charging_session_duration_minutes_all": value * 10.0,
            "avg_wait_steps_waiting_charging_vehicles": value,
            "avg_wait_minutes_waiting_charging_vehicles": value * 2.0,
            "waiting_charging_vehicle_count": 2.0,
            "mean_completed_orders": 100.0 - value,
            "mean_service_ratio": 1.0 - value / 100.0,
            "mean_final_battery_soc": 0.6,
        }
        for ratio, value in ((0.8, 8.0), (1.0, 10.0), (1.2, 12.0))
    ])
    trend = build_trend_summary(build_summary(detail)).set_index("metric")
    charging = trend.loc["avg_daily_charging_sessions_per_human_ev"]
    assert charging["endpoint_change_percent"] == pytest.approx(50.0)
    assert charging["linear_r_squared"] == pytest.approx(1.0)



def test_charging_session_metrics_avoid_division_by_zero():
    metrics = charging_session_metrics(
        {0: {"type": 1, "charging_count": 3}},
        simulated_days=0.0,
    )

    assert metrics["human_ev_charging_sessions"] == 3
    assert metrics["avg_daily_charging_sessions_per_human_ev"] == 0.0
    assert metrics["avg_charging_session_duration_minutes_all"] == 0.0
