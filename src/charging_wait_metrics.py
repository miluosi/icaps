"""Shared charging-queue congestion metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def positive_wait_metrics(
    observations: Iterable[Mapping[str, Any]] | None,
    *,
    active_arrivals: Mapping[Any, Mapping[str, Any]] | None = None,
    current_time: float = 0.0,
) -> dict[str, float | int]:
    """Average wait among vehicles that actually waited.

    Completed zero-wait charging starts are excluded.  Active arrivals are
    right-censored waits still present at episode end; each such vehicle has
    waited at least one simulation step even when its arrival timestamp equals
    the final timestamp.
    """

    completed_waits = []
    for observation in observations or ():
        wait = max(0.0, float(observation.get("observed_wait", 0.0) or 0.0))
        if wait > 0.0:
            completed_waits.append(wait)

    ongoing_waits = []
    for arrival in (active_arrivals or {}).values():
        raw_arrival_time = arrival.get("arrival_time", current_time)
        arrival_time = float(
            current_time if raw_arrival_time is None else raw_arrival_time
        )
        ongoing_waits.append(max(1.0, float(current_time) - arrival_time))

    waits = completed_waits + ongoing_waits
    return {
        "avg_wait": sum(waits) / len(waits) if waits else 0.0,
        "waiting_vehicle_count": len(waits),
        "completed_waiting_vehicle_count": len(completed_waits),
        "ongoing_waiting_vehicle_count": len(ongoing_waits),
    }


def aggregate_wait_metrics(
    rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, float]:
    """Vehicle-weighted aggregation of episode/day ``avg_wait`` values."""

    total_count = 0.0
    total_wait = 0.0
    row_count = 0
    for row in rows or ():
        row_count += 1
        count = max(0.0, float(row.get("waiting_vehicle_count", 0.0) or 0.0))
        total_count += count
        total_wait += max(0.0, float(row.get("avg_wait", 0.0) or 0.0)) * count
    return {
        "avg_wait": total_wait / total_count if total_count > 0.0 else 0.0,
        "waiting_vehicle_count": total_count,
        "mean_waiting_vehicle_count": total_count / row_count if row_count else 0.0,
    }
