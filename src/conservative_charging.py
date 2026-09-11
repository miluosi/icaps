"""Experimental all-potential charging admission, without simulator mutation.

This is an additional conservative feasibility filter, not an SSG reduction.
Every physical vehicle--station candidate is virtually sent to that station,
including candidates already rejected by the background-only feasibility test.
Only candidates that can start immediately in this virtual schedule survive.
Existing charging and committed arrival reservations retain their original
plug and service interval.  The real station queues are never updated here.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Mapping

import numpy as np

from .expected_charging import epoch_offset


def _hard_calendars(schedule: Mapping, capacity: int) -> list[list[tuple[int, int]]]:
    """Copy and validate fixed reservations, retaining their assigned plugs."""

    calendars: list[list[tuple[int, int]]] = [[] for _ in range(capacity)]
    intervals = schedule.get("intervals", ())
    if not intervals and any(int(x) for x in schedule.get("occupancy", ())):
        raise ValueError("Nonempty background occupancy requires per-plug intervals")
    for interval in intervals:
        if capacity == 0:
            continue
        plug = interval.get("plug_index")
        if interval.get("unscheduled") or plug is None:
            raise ValueError("A committed job has no fixed plug reservation")
        plug = int(plug)
        start = int(interval["start_offset"])
        end = int(interval["end_offset"])
        if not 0 <= plug < capacity or start < 0 or end <= start:
            raise ValueError("Invalid committed charging interval")
        calendars[plug].append((start, end))
    for calendar in calendars:
        calendar.sort()
        if any(a[1] > b[0] for a, b in zip(calendar, calendar[1:])):
            raise ValueError("Committed intervals overlap on the same plug")
    return calendars


def conservative_charge_mask(
    *,
    vehicle_ids,
    station_ids,
    feasibility,
    candidate_windows: Mapping,
    station_schedules: Mapping,
) -> dict:
    """Return an all-potential, no-wait charge mask and its virtual schedule.

    ``candidate_windows[(vehicle_id, station_id)]`` uses the existing NYC
    ``travel_epochs`` and ``charging_duration`` keys.  Its presence denotes a
    physical candidate even when its input feasibility entry is false.  The
    background schedules are from ``build_expected_station_schedule`` and
    must contain valid per-plug ``intervals`` for all committed jobs.  Candidate
    jobs and committed jobs must represent distinct outstanding obligations.

    Per station, candidates are ordered by arrival offset, then vehicle ID.
    A candidate is appended to the earliest available virtual plug, finding a
    complete gap around that plug's immutable committed reservations.  Virtual
    jobs assigned earlier to a plug are not reordered to fill gaps.  A candidate
    is retained exactly when it was originally feasible and its virtual start
    equals its arrival.  Deleted candidates still occupy their virtual slots:
    this is deliberately the user's all-potential pessimistic assumption.

    With P_c candidates, S_c plugs and H_c committed intervals at station c,
    time is O(N*C + sum_c(H_c log(1+H_c) + P_c log(1+P_c)
    + P_c*S_c*log(2+H_c) + B_c)), where B_c counts reservations inspected
    while looking for complete gaps.  B_c can be O(P_c*H_c) in the worst case;
    this is not merely a heap/sort complexity bound.  Space is
    O(N*C + sum_c(P_c + H_c + S_c)), including returned virtual windows.

    Returns a dict containing the bool array ``feasibility``, tuple IDs,
    ``virtual_windows`` keyed by (vehicle, station), and ``station_diagnostics``.
    Intervals are half open [start_offset, end_offset), so release at arrival
    is allowed.  No caller-owned arrays, mappings, reservations or queues mutate.
    The guarantee concerns these deterministic intervals, not simulator timing
    error, uncertain travel times, or feasibility of mandatory charging for all
    vehicles after this extra admission filter.
    """

    vehicle_ids = tuple(int(value) for value in vehicle_ids)
    station_ids = tuple(int(value) for value in station_ids)
    if len(set(vehicle_ids)) != len(vehicle_ids):
        raise ValueError("vehicle_ids must be unique")
    if len(set(station_ids)) != len(station_ids):
        raise ValueError("station_ids must be unique")
    original = np.asarray(feasibility, dtype=bool)
    if original.shape != (len(vehicle_ids), len(station_ids)):
        raise ValueError("feasibility shape must match vehicle_ids and station_ids")
    admitted = np.zeros(original.shape, dtype=bool)
    virtual_windows: dict[tuple[int, int], dict] = {}
    diagnostics: dict[int, dict] = {}

    for station_col, station_id in enumerate(station_ids):
        schedule = station_schedules.get(station_id, {})
        capacity = max(0, int(schedule.get("capacity", 0)))
        calendars = _hard_calendars(schedule, capacity)
        hard_ends = [tuple(end for _, end in calendar) for calendar in calendars]
        virtual_next_free = [0] * capacity
        candidates = []
        for vehicle_row, vehicle_id in enumerate(vehicle_ids):
            window = candidate_windows.get((vehicle_id, station_id))
            if not window:
                continue
            arrival = epoch_offset(window.get("travel_epochs", 0))
            duration = max(1, epoch_offset(window.get("charging_duration", 1)))
            candidates.append((arrival, vehicle_id, duration, vehicle_row))
        candidates.sort(key=lambda item: (item[0], item[1]))
        gap_checks = 0
        delayed = 0
        total_wait = 0
        max_wait = 0

        for arrival, vehicle_id, duration, vehicle_row in candidates:
            best_start = None
            best_plug = None
            for plug_index, calendar in enumerate(calendars):
                start = max(arrival, virtual_next_free[plug_index])
                reservation_index = bisect_right(hard_ends[plug_index], start)
                while reservation_index < len(calendar):
                    hard_start, hard_end = calendar[reservation_index]
                    gap_checks += 1
                    if start + duration <= hard_start:
                        break
                    start = hard_end
                    reservation_index += 1
                if best_start is None or start < best_start:
                    best_start = start
                    best_plug = plug_index
                # Plugs are visited by ID: immediate service is unbeatable,
                # and the first such plug also satisfies deterministic ties.
                if best_start == arrival:
                    break

            if best_plug is None:
                end = None
                wait = None
                kept = False
            else:
                end = best_start + duration
                wait = best_start - arrival
                virtual_next_free[best_plug] = end
                kept = bool(original[vehicle_row, station_col] and wait == 0)
                delayed += int(wait > 0)
                total_wait += wait
                max_wait = max(max_wait, wait)
            admitted[vehicle_row, station_col] = kept
            virtual_windows[vehicle_id, station_id] = {
                "arrival_offset": arrival,
                "start_offset": best_start,
                "end_offset": end,
                "charging_duration": duration,
                "wait_offset": wait,
                "plug_index": best_plug,
                "original_feasible": bool(original[vehicle_row, station_col]),
                "kept": kept,
            }

        original_count = int(np.count_nonzero(original[:, station_col]))
        kept_count = int(np.count_nonzero(admitted[:, station_col]))
        diagnostics[station_id] = {
            "capacity": capacity,
            "committed_jobs": sum(len(calendar) for calendar in calendars),
            "potential_edges": len(candidates),
            "background_feasible_edges": original_count,
            "kept_edges": kept_count,
            "rejected_background_feasible_edges": original_count - kept_count,
            "delayed_potential_edges": delayed,
            "unscheduled_potential_edges": len(candidates) if capacity == 0 else 0,
            "virtual_wait_epochs": total_wait,
            "max_virtual_wait_epochs": max_wait,
            "virtual_completion_offset": max(virtual_next_free, default=0),
            "hard_reservation_gap_checks": gap_checks,
        }

    return {
        "vehicle_ids": vehicle_ids,
        "station_ids": station_ids,
        "feasibility": admitted,
        "virtual_windows": virtual_windows,
        "station_diagnostics": diagnostics,
    }
