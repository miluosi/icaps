import pytest

from src.charging_metrics import charging_session_metrics


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


def test_charging_session_metrics_avoid_division_by_zero():
    metrics = charging_session_metrics(
        {0: {"type": 1, "charging_count": 3}},
        simulated_days=0.0,
    )

    assert metrics["human_ev_charging_sessions"] == 3
    assert metrics["avg_daily_charging_sessions_per_human_ev"] == 0.0
    assert metrics["avg_charging_session_duration_minutes_all"] == 0.0
