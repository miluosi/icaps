"""Auditable charging-session statistics shared by the simulators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _nonnegative_session_count(vehicle: Mapping[str, Any]) -> int:
    try:
        return max(0, int(vehicle.get("charging_count", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _valid_durations_minutes(vehicle: Mapping[str, Any]) -> list[float]:
    durations = []
    for raw_duration in vehicle.get("completed_charging_durations_minutes", []) or []:
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            continue
        if duration >= 0.0:
            durations.append(duration)
    return durations


def charging_session_metrics(
    vehicles: Mapping[Any, Mapping[str, Any]],
    simulated_days: float,
) -> dict[str, float | int]:
    """Return type-specific session counts, per-vehicle-day rates, and durations.

    ``charging_count`` is incremented only when a vehicle actually starts a
    station charging session.  This intentionally differs from a count of
    ``ChargingAction`` objects, which may include travel and queueing steps.
    """

    days = max(0.0, float(simulated_days))
    human_evs = [vehicle for vehicle in vehicles.values() if vehicle.get("type") == 1]
    aevs = [vehicle for vehicle in vehicles.values() if vehicle.get("type") == 2]
    all_vehicles = list(vehicles.values())

    human_ev_sessions = sum(_nonnegative_session_count(vehicle) for vehicle in human_evs)
    aev_sessions = sum(_nonnegative_session_count(vehicle) for vehicle in aevs)
    all_sessions = sum(_nonnegative_session_count(vehicle) for vehicle in all_vehicles)

    human_ev_durations = [
        duration
        for vehicle in human_evs
        for duration in _valid_durations_minutes(vehicle)
    ]
    aev_durations = [
        duration
        for vehicle in aevs
        for duration in _valid_durations_minutes(vehicle)
    ]
    all_durations = [
        duration
        for vehicle in all_vehicles
        for duration in _valid_durations_minutes(vehicle)
    ]

    def per_vehicle_day(session_count: int, vehicle_count: int) -> float:
        denominator = float(vehicle_count) * days
        return float(session_count) / denominator if denominator > 0.0 else 0.0

    def mean_duration(durations: list[float]) -> float:
        return sum(durations) / len(durations) if durations else 0.0

    return {
        "charging_observation_days": days,
        "human_ev_vehicle_count": len(human_evs),
        "aev_vehicle_count": len(aevs),
        "all_vehicle_count": len(all_vehicles),
        "human_ev_charging_sessions": human_ev_sessions,
        "aev_charging_sessions": aev_sessions,
        "all_vehicle_charging_sessions": all_sessions,
        "avg_daily_charging_sessions_per_human_ev": per_vehicle_day(
            human_ev_sessions,
            len(human_evs),
        ),
        "avg_daily_charging_sessions_per_aev": per_vehicle_day(
            aev_sessions,
            len(aevs),
        ),
        "avg_daily_charging_sessions_per_vehicle": per_vehicle_day(
            all_sessions,
            len(all_vehicles),
        ),
        "completed_charging_sessions_with_duration_human_ev": len(human_ev_durations),
        "completed_charging_sessions_with_duration_aev": len(aev_durations),
        "completed_charging_sessions_with_duration_all": len(all_durations),
        "avg_charging_session_duration_minutes_human_ev": mean_duration(
            human_ev_durations
        ),
        "avg_charging_session_duration_minutes_aev": mean_duration(aev_durations),
        "avg_charging_session_duration_minutes_all": mean_duration(all_durations),
    }
