"""Deterministic charging-station occupancy helpers.

The action matrix uses discrete simulator epochs.  NYC's default admission
checks committed occupancy at arrival; conservative admission checks a full
charging window.  Waiting-room capacity is not physical charging capacity.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def epoch_offset(value: float) -> int:
    """Round a non-negative continuous duration up to simulator epochs."""

    return max(0, int(math.ceil(max(0.0, float(value)) - 1e-12)))


def build_expected_station_schedule(
    *,
    capacity: int,
    current_jobs: Iterable[Mapping],
    waiting_jobs: Iterable[Mapping],
) -> dict:
    """Schedule committed jobs and return relative per-epoch occupancy.

    ``current_jobs`` already occupy separate plugs at offset zero.  Remaining
    jobs are processed by release time and input order, matching station FIFO
    behavior.  Each job starts on the first plug that becomes available no
    earlier than its release offset.
    """

    capacity = max(0, int(capacity))
    intervals: list[dict] = []
    plug_available = [0 for _ in range(capacity)]

    for index, raw_job in enumerate(current_jobs):
        duration = max(1, epoch_offset(raw_job.get("duration", 1)))
        interval = {
            **dict(raw_job),
            "start_offset": 0,
            "end_offset": duration,
            "plug_index": index if index < capacity else None,
        }
        intervals.append(interval)
        if index < capacity:
            plug_available[index] = duration

    ordered_waiting_jobs = sorted(
        enumerate(waiting_jobs),
        key=lambda item: (
            epoch_offset(item[1].get("release_offset", 0)),
            int(item[1].get("priority", 1)),
            item[0],
        ),
    )
    for _input_index, raw_job in ordered_waiting_jobs:
        release_offset = epoch_offset(raw_job.get("release_offset", 0))
        duration = max(1, epoch_offset(raw_job.get("duration", 1)))
        if capacity <= 0:
            intervals.append({
                **dict(raw_job),
                "start_offset": release_offset,
                "end_offset": release_offset + duration,
                "plug_index": None,
                "unscheduled": True,
            })
            continue
        plug_index = min(range(capacity), key=lambda idx: (plug_available[idx], idx))
        start_offset = max(release_offset, plug_available[plug_index])
        end_offset = start_offset + duration
        plug_available[plug_index] = end_offset
        intervals.append({
            **dict(raw_job),
            "start_offset": start_offset,
            "end_offset": end_offset,
            "plug_index": plug_index,
        })

    horizon = max((int(item["end_offset"]) for item in intervals), default=0)
    occupancy = [0 for _ in range(horizon)]
    for interval in intervals:
        if interval.get("unscheduled"):
            continue
        for offset in range(
            max(0, int(interval["start_offset"])),
            max(0, int(interval["end_offset"])),
        ):
            if offset >= len(occupancy):
                occupancy.extend([0] * (offset + 1 - len(occupancy)))
            occupancy[offset] += 1

    return {
        "capacity": capacity,
        "occupancy": tuple(occupancy),
        "intervals": tuple(intervals),
        "plug_available_offsets": tuple(plug_available),
    }


def has_complete_charging_window(
    schedule: Mapping,
    *,
    arrival_offset: int,
    charging_duration: int,
) -> bool:
    """Return whether a plug is free for every epoch in the candidate window."""

    capacity = max(0, int(schedule.get("capacity", 0)))
    if capacity <= 0:
        return False
    start = epoch_offset(arrival_offset)
    duration = max(1, epoch_offset(charging_duration))
    occupancy = tuple(schedule.get("occupancy", ()))
    for offset in range(start, start + duration):
        if offset < len(occupancy) and int(occupancy[offset]) >= capacity:
            return False
    return True


def build_charge_action_epoch_expansion(
    *,
    vehicle_ids,
    station_ids,
    feasibility,
    station_schedules: Mapping,
    candidate_windows: Mapping,
    capacity_scope: str = "full_window",
) -> dict:
    """Expand the legacy 2-D charge matrix into per-epoch resource masks.

    The returned tensor does not replace the public vehicle-by-station matrix.
    It records the epochs constrained by the admission policy.  ``arrival``
    uses only the arrival epoch; ``full_window`` uses the charging interval.
    Candidate windows always retain the physical charging duration.
    """

    vehicle_ids = tuple(int(vehicle_id) for vehicle_id in vehicle_ids)
    station_ids = tuple(int(station_id) for station_id in station_ids)
    feasibility = np.asarray(feasibility)
    horizon = max(
        [len(schedule.get("occupancy", ())) for schedule in station_schedules.values()]
        + [
            int(window.get("travel_epochs", 0))
            + int(window.get("charging_duration", 0))
            for window in candidate_windows.values()
            if window
        ]
        + [0]
    )
    action_epoch_mask = np.zeros(
        (len(vehicle_ids), len(station_ids), horizon), dtype=np.uint8
    )
    base_occupancy = np.zeros((len(station_ids), horizon), dtype=np.int32)
    remaining_capacity = np.zeros((len(station_ids), horizon), dtype=np.int32)

    for station_col, station_id in enumerate(station_ids):
        schedule = station_schedules.get(station_id, {})
        occupancy = np.asarray(schedule.get("occupancy", ()), dtype=np.int32)
        if occupancy.size:
            base_occupancy[station_col, :occupancy.size] = occupancy
        capacity = max(0, int(schedule.get("capacity", 0)))
        remaining_capacity[station_col, :] = np.maximum(
            0, capacity - base_occupancy[station_col, :]
        )

        for vehicle_row, vehicle_id in enumerate(vehicle_ids):
            if feasibility[vehicle_row, station_col] == 0:
                continue
            window = candidate_windows.get((vehicle_id, station_id))
            if not window:
                continue
            epochs = charging_capacity_epochs(window, capacity_scope)
            action_epoch_mask[vehicle_row, station_col, epochs.start:epochs.stop] = 1

    return {
        "vehicle_ids": vehicle_ids,
        "capacity_scope": capacity_scope,
        "station_ids": station_ids,
        "horizon": int(horizon),
        "epoch_offsets": tuple(range(horizon)),
        "action_epoch_mask": action_epoch_mask,
        "base_occupancy": base_occupancy,
        "remaining_capacity": remaining_capacity,
        "candidate_windows": dict(candidate_windows),
        "station_schedules": dict(station_schedules),
    }


def charging_capacity_epochs(window: Mapping, capacity_scope: str) -> range:
    """Epochs charged to an admission quota, independent of physical duration."""
    if capacity_scope not in {"arrival", "full_window"}:
        raise ValueError(f"Unknown charging capacity scope: {capacity_scope}")
    start = epoch_offset(window.get("travel_epochs", 0))
    duration = (1 if capacity_scope == "arrival" else
                max(1, epoch_offset(window.get("charging_duration", 1))))
    return range(start, start + duration)
