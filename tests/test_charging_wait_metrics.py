import pytest

from src.charging_wait_metrics import (
    aggregate_wait_metrics,
    positive_wait_metrics,
)


def test_positive_wait_excludes_immediate_starts_and_includes_ongoing_queue():
    metrics = positive_wait_metrics(
        [
            {"observed_wait": 0.0},
            {"observed_wait": 2.0},
            {"observed_wait": 4.0},
        ],
        active_arrivals={
            (7, 1): {"arrival_time": 8.0},
            (8, 1): {"arrival_time": 10.0},
        },
        current_time=10.0,
    )

    # Positive waits are [2, 4] and censored ongoing waits are [2, 1].
    assert metrics["avg_wait"] == pytest.approx(2.25)
    assert metrics["waiting_vehicle_count"] == 4
    assert metrics["completed_waiting_vehicle_count"] == 2
    assert metrics["ongoing_waiting_vehicle_count"] == 2


def test_aggregate_wait_is_weighted_by_waiting_vehicle_count():
    metrics = aggregate_wait_metrics(
        [
            {"avg_wait": 10.0, "waiting_vehicle_count": 1},
            {"avg_wait": 2.0, "waiting_vehicle_count": 3},
            {"avg_wait": 0.0, "waiting_vehicle_count": 0},
        ]
    )

    assert metrics["avg_wait"] == pytest.approx(4.0)
    assert metrics["waiting_vehicle_count"] == pytest.approx(4.0)
    assert metrics["mean_waiting_vehicle_count"] == pytest.approx(4.0 / 3.0)


def test_ongoing_wait_preserves_zero_arrival_timestamp():
    metrics = positive_wait_metrics(
        [],
        active_arrivals={(1, 2): {"arrival_time": 0.0}},
        current_time=12.0,
    )

    assert metrics["avg_wait"] == pytest.approx(12.0)
    assert metrics["waiting_vehicle_count"] == 1
