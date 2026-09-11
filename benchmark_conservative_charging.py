"""Paired single-epoch NYC charge-admission diagnostics (no training).

Uses the real NYC charge/wait generators and current OR-Tools adapter on
synthetic, reproducible states.  All-potential admission is a NEW model;
SSG exactness is checked within that model.  The current adapter's temporal
repair is a comparator, not a globally optimal interval-scheduling oracle.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import ortools

from src.NYCEnvironment import NYCEnvironment
from src.GurobiOptimizer import GurobiOptimizer
from src.charging_station import ChargingStation
from src.exact_mcmf import build_reduced_problem, solve_exact

SCENARIOS = ("synchronized", "staggered", "committed", "mixed_demand")


def make_case(total_vehicles, station_count, slots, scenario, seed):
    """Total includes busy, committed inbound, and decision vehicles."""
    rng = np.random.default_rng(seed)
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.current_time = 0.0
    env.battery_consum = 0.0001
    env.min_battery_level = 0.2
    env.chargeincrease_per_epoch = 0.1
    env.charge_duration_scale = 1.0
    env.charge_target_soc = 0.8
    env.charge_topup_soc = 0.05
    env.min_charging_session_epochs = 1
    env.max_charging_session_epochs = 20
    env.charge_duration = 6
    env.charge_action_range_km = None
    env.charge_top_k = None
    env.charge_wait_bool = True
    env.conservative_charging = False
    env.mcmf_solver = "exact"
    env.mcmf_backend = "ortools"
    env.mcmf_strict = True
    env.mcmf_cost_scale = 10_000
    env.mcmf_graph_reduction = True
    env.mcmf_verify = True
    env.mcmf_fallback_value = None
    env.reserve_inbound_charging_capacity = False
    env.station_queue_capacity = 0
    env.record_time = False
    stations = {100 + c: ChargingStation(100 + c, c, slots)
                for c in range(station_count)}
    env.charging_manager = SimpleNamespace(stations=stations)
    station_ids = sorted(stations)
    env._charging_station_ids_for_vehicle = lambda vid: station_ids
    env._is_ev = lambda vid: False
    env.distance_matrix = np.zeros((station_count + total_vehicles, station_count),
                                   dtype=np.float32)
    travel = np.zeros_like(env.distance_matrix)
    if scenario == "synchronized":
        travel[station_count:] = 1
        batteries = np.full(total_vehicles, 0.3)
    else:
        travel[station_count:] = rng.integers(1, 25, (total_vehicles, station_count))
        batteries = rng.uniform(0.15, 0.65, total_vehicles)
        env.distance_matrix[station_count:] = travel[station_count:] * 0.2
    env.get_distance_km = lambda origin, destination: float(env.distance_matrix[int(origin), int(destination)])
    env.get_travel_time = lambda origin, destination: float(travel[int(origin), int(destination)])
    env.vehicles = {
        v: {"type": 2, "location": station_count + v, "battery": float(batteries[v])}
        for v in range(total_vehicles)
    }
    current_count = min(30, slots * 3 // 5) if scenario == "committed" else (10 if scenario == "mixed_demand" else 0)
    inbound_count = min(20, slots - current_count) if scenario == "committed" else (10 if scenario == "mixed_demand" else 0)
    busy_count = station_count * (current_count + inbound_count)
    if busy_count >= total_vehicles:
        raise ValueError("This scenario needs more vehicles than its committed background")
    next_id = total_vehicles - busy_count
    for sid, station in stations.items():
        for _ in range(current_count):
            vid = next_id
            next_id += 1
            station.current_vehicles.append(str(vid))
            env.vehicles[vid]["charging_time_left"] = int(rng.integers(8, 16))
            env.vehicles[vid]["location"] = station.location
        station.available_slots = slots - current_count
        for _ in range(inbound_count):
            vid = next_id
            next_id += 1
            station.charging_queue_notarrived.append(str(vid))
            travel[station_count + vid, station.location] = int(rng.integers(1, 8))
    vehicle_ids = list(range(total_vehicles - busy_count))
    n = len(vehicle_ids)
    # Synthetic fixed assignment scores, not realized NYC dollar rewards.
    q = np.zeros((n, station_count + 1), dtype=np.float64)
    q[:, :station_count] = rng.uniform(80, 120, (n, station_count))
    if scenario == "mixed_demand":
        wants_charge = rng.random(n) < 0.25
        q[~wants_charge, -1] = 150.0
    q = np.round(q, 4)
    env._last_matrix_num_requests = 0
    env._last_matrix_num_stations = station_count
    env._last_matrix_num_zones = 0
    env._last_matrix_charge_station_ids = station_ids
    env._last_matrix_zone_indices = []
    env._last_matrix_zone_target_ids = []
    return env, vehicle_ids, q, {
        "total_vehicles": total_vehicles, "decision_vehicles": n,
        "current_charging_vehicles": station_count * current_count,
        "committed_inbound_vehicles": station_count * inbound_count,
        "station_count": station_count, "slots_per_station": slots,
        "scenario": scenario, "seed": seed,
        "travel_epochs": travel[station_count:].astype(int).tolist(),
        "batteries": batteries.tolist(),
        "vehicles": env.vehicles,
    }


def make_optimizer(env):
    # Exercise the real adapter while avoiding unrelated CPLEX initialization.
    optimizer = GurobiOptimizer.__new__(GurobiOptimizer)
    optimizer.env = env
    optimizer.num_threads = 1
    optimizer.mcmf_backend = "ortools"
    optimizer.mcmf_cost_scale = 10_000
    optimizer.mcmf_graph_reduction = True
    optimizer.mcmf_verify = True
    optimizer.mip_backend = "docplex"
    optimizer.available = False
    optimizer._reported_exact_mcmf_backend = True
    return optimizer


def solve_adapter(env, vehicle_ids, mask, scores, ssg):
    env.mcmf_graph_reduction = bool(ssg)
    env.expected_charge_capacity_cut_iterations = 0
    started = time.perf_counter()
    try:
        assignments = make_optimizer(env)._exact_vehicle_rebalancing_network(
            vehicle_ids, [], mask.copy(), scores.copy())
        return assignments, {
            "status": "optimal_on_final_graph",
            "assignment_wall_seconds": time.perf_counter() - started,
            "objective_int": int(env.mcmf_last_result["objective_int"]),
            "assignment_score": float(env.mcmf_last_result["objective_q"]),
            "cut_iterations": int(env.expected_charge_capacity_cut_iterations),
            "backend_build_seconds": float(env.mcmf_last_result["build_time"]),
            "backend_solve_seconds": float(env.mcmf_last_result["solve_time"]),
        }
    except (RuntimeError, ValueError) as error:
        return None, {"status": "infeasible_or_solver_error", "error": str(error),
                      "assignment_wall_seconds": time.perf_counter() - started}


def fixed_graph_ssg_check(mask, scores):
    """Check only identical 2-D coefficients/constraints, outside timings."""
    objectives = []
    capacities = np.full(mask.shape[1], mask.shape[0], dtype=np.int64)
    for ssg in (False, True):
        problem = build_reduced_problem(mask, scores, capacities,
                                        cost_scale=10_000, graph_reduction=ssg)
        result = solve_exact(problem, backend="ortools", verify=True, num_threads=1)
        objectives.append(int(result.objective_int))
    if objectives[0] != objectives[1]:
        raise AssertionError("SSG changed the fixed input-graph optimum")
    return objectives[1] - objectives[0]


def replay_station(capacity, hard_intervals, chosen, horizon):
    """Replay fixed intended starts with the actual ChargingStation queue API.

    Releases precede admissions at the same integer epoch, matching [start,end).
    This is a calendar-consistent single-decision replay, NOT NYC.step's full
    movement/update loop. Existing scheduled waits are measured separately.
    """
    station = ChargingStation(0, 0, capacity)
    starts = {}
    active = {}
    wanted = {}
    durations = {}
    new_ids = set()
    for idx, interval in enumerate(hard_intervals):
        key = f"hard_{idx}"
        start, end = int(interval["start_offset"]), int(interval["end_offset"])
        starts.setdefault(start, []).append(key)
        wanted[key], durations[key] = start, end - start
    for vid, arrival, duration in chosen:
        key = f"new_{vid}"
        starts.setdefault(arrival, []).append(key)
        wanted[key], durations[key] = arrival, duration
        new_ids.add(key)
    actual_start = {}
    actual_occupancy = np.zeros(horizon, dtype=np.int32)
    end_limit = max(horizon, max(starts, default=0)) + sum(durations.values()) + 1
    with contextlib.redirect_stdout(io.StringIO()):
        for epoch in range(end_limit):
            for key in sorted([key for key, end in active.items() if end == epoch]):
                station.stop_charging(key)
                del active[key]
            # stop_charging can immediately start an existing queued vehicle.
            for key in station.current_vehicles:
                if key not in active:
                    actual_start[key] = epoch
                    active[key] = epoch + durations[key]
            for key in sorted(starts.get(epoch, [])):
                if station.start_charging(key):
                    actual_start[key] = epoch
                    active[key] = epoch + durations[key]
            if epoch < horizon:
                actual_occupancy[epoch] = len(station.current_vehicles)
            if epoch >= max(starts, default=0) and not active and not station.charging_queue:
                break
    waits = [actual_start[key] - wanted[key] for key in new_ids]
    return {"new_queue_vehicle_count": sum(wait > 0 for wait in waits),
            "new_queue_wait_epochs": sum(waits),
            "new_queue_max_wait_epochs": max(waits, default=0),
            "committed_reservation_delay_epochs": sum(actual_start[key] - wanted[key]
                                                       for key in wanted if key not in new_ids)}, actual_occupancy


def measure_plan(expansion, assignments, scenario_info, method):
    windows = expansion["candidate_windows"]
    schedules = expansion["station_schedules"]
    # Common, unfiltered immediate-window horizon; virtual queues never extend it.
    horizon = max([len(s["occupancy"]) for s in schedules.values()] +
                  [int(w["travel_epochs"] + w["charging_duration"]) for w in windows.values()] + [1])
    rows, arrays = [], {}
    for sid, schedule in schedules.items():
        base = np.zeros(horizon, dtype=np.int32)
        base[:len(schedule["occupancy"])] = schedule["occupancy"]
        added = np.zeros(horizon, dtype=np.int32)
        chosen = []
        for vid, action in assignments.items():
            if action != f"charge_{sid}":
                continue
            window = windows[vid, sid]
            start, duration = int(window["travel_epochs"]), int(window["charging_duration"])
            chosen.append((vid, start, duration))
            added[start:start + duration] += 1
        total = base + added
        capacity = int(schedule["capacity"])
        replay, observed = replay_station(capacity, schedule["intervals"], chosen, horizon)
        hard_wait = sum(max(0, int(w["start_offset"]) - int(w.get("release_offset", 0)))
                        for w in schedule["intervals"])
        rows.append({**{k: scenario_info[k] for k in ("scenario", "seed", "station_count", "total_vehicles")},
                     "method": method, "station_id": sid, "horizon_epochs": horizon,
                     "slots": capacity, "assigned_charge_vehicles": len(chosen),
                     "peak_occupied_slots": int(total.max()),
                     "base_occupied_slot_epochs": int(base.sum()),
                     "new_occupied_slot_epochs": int(added.sum()),
                     "idle_slot_epochs": int((capacity - total).sum()),
                     "idle_slots_at_epoch_1": int(capacity - total[min(1, horizon - 1)]),
                     "overcapacity_slot_epochs": int(np.maximum(0, total - capacity).sum()),
                     "background_scheduled_wait_epochs": hard_wait, **replay})
        arrays[f"station_{sid}_base"] = base
        arrays[f"station_{sid}_new"] = added
        arrays[f"station_{sid}_replay"] = observed
    return rows, arrays


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    output = Path(args.output_dir or (Path(__file__).resolve().parent / "results" /
                  "conservative_charging" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    metadata = {"python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "ortools": ortools.__version__, "args": vars(args),
                "scope": f"synthetic {args.vehicles}-vehicle single-decision NYC charge API comparison; not training or NYC.step rollout",
                "queue_replay": "ChargingStation API, fixed arrivals/durations, releases before arrivals",
                "scores": "synthetic quantized assignment scores, not realized monetary reward",
                "current_comparator": "existing background-only mask plus existing greedy temporal repair; not a global interval oracle",
                "slot_epochs": "one free slot for one simulator epoch; unfiltered common horizon, no virtual queue tail"}
    metadata["source_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                 for name in ("benchmark_conservative_charging.py", "src/conservative_charging.py",
                                              "src/NYCEnvironment.py", "src/GurobiOptimizer.py", "src/expected_charging.py",
                                              "src/exact_mcmf.py")}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Output: {output}", flush=True)
    # Keep one-time OR-Tools initialization outside the paired measurements.
    warm_env, warm_ids, warm_scores, _ = make_case(20, 1, 5, "synchronized", 0)
    warm_mask = np.column_stack((warm_env.generate_vehicle_chargerange(warm_ids), np.ones((20, 1))))
    for ssg in (False, True):
        solve_adapter(warm_env, warm_ids, warm_mask, warm_scores, ssg)
    warm_env.conservative_charging = True
    warm_env.generate_vehicle_chargerange(warm_ids)
    all_rows, station_rows = [], []
    for scenario in args.scenarios:
        for count in args.stations:
            for seed in args.seeds:
                env, ids, scores, info = make_case(args.vehicles, count, args.slots, scenario, seed)
                case_dir = output / "cases" / f"{scenario}_{count}stations_seed{seed}"
                case_dir.mkdir()
                (case_dir / "input.json").write_text(json.dumps(info, indent=2))
                np.savez_compressed(case_dir / "scores.npz", scores=scores, vehicle_ids=ids)
                masks = {}
                pair = []
                for conservative in ((False, True) if seed % 2 == 0 else (True, False)):
                    method = "conservative" if conservative else "current"
                    env.conservative_charging = conservative
                    started = time.perf_counter()
                    charge = env.generate_vehicle_chargerange(ids)
                    preprocess = time.perf_counter() - started
                    expansion = env._last_expected_charge_expansion
                    native_wait = env.generate_vehicle_wait(ids, charge_feasibility=charge)
                    wait = native_wait if args.wait_policy == "nyc" else np.ones((len(ids), 1))
                    mask = np.column_stack((charge, wait))
                    masks[method] = mask
                    row = {k: info[k] for k in ("scenario", "seed", "station_count", "slots_per_station",
                                                "total_vehicles", "decision_vehicles", "current_charging_vehicles",
                                                "committed_inbound_vehicles")}
                    row.update(method=method, wait_policy=args.wait_policy,
                               charge_edges=int(charge.sum()), vehicles_with_charge=int(np.any(charge, axis=1).sum()),
                               native_wait_forbidden=int((native_wait == 0).sum()),
                               preprocess_seconds=preprocess)
                    assignments, stats = solve_adapter(env, ids, mask, scores, True)
                    row.update(stats)
                    off_assignments, unscaled = solve_adapter(env, ids, mask, scores, False)
                    row["ssg_off_status"] = unscaled["status"]
                    row["ssg_off_cut_iterations"] = unscaled.get("cut_iterations")
                    row["fixed_input_graph_ssg_gap_int"] = fixed_graph_ssg_check(mask, scores)
                    if "objective_int" in stats and "objective_int" in unscaled:
                        row["ssg_on_off_objective_gap_int"] = stats["objective_int"] - unscaled["objective_int"]
                    row["end_to_end_seconds"] = preprocess + stats["assignment_wall_seconds"]
                    if assignments is not None:
                        details, profiles = measure_plan(expansion, assignments, info, method)
                        station_rows.extend(details)
                        for metric in ("assigned_charge_vehicles", "idle_slot_epochs", "idle_slots_at_epoch_1",
                                       "new_occupied_slot_epochs", "overcapacity_slot_epochs",
                                       "new_queue_vehicle_count", "new_queue_wait_epochs", "committed_reservation_delay_epochs"):
                            row[metric] = sum(item[metric] for item in details)
                        row["horizon_epochs"] = details[0]["horizon_epochs"]
                        row["idle_capacity_fraction"] = row["idle_slot_epochs"] / (count * args.slots * row["horizon_epochs"])
                        np.savez_compressed(case_dir / f"{method}_occupancy.npz", **profiles)
                        (case_dir / f"{method}_assignments.json").write_text(json.dumps(assignments, indent=2))
                    if off_assignments is not None:
                        off_details, _ = measure_plan(expansion, off_assignments, info, method)
                        row["ssg_off_new_queue_vehicle_count"] = sum(r["new_queue_vehicle_count"] for r in off_details)
                        row["ssg_off_overcapacity_slot_epochs"] = sum(r["overcapacity_slot_epochs"] for r in off_details)
                        (case_dir / f"{method}_ssg_off_assignments.json").write_text(json.dumps(off_assignments, indent=2))
                    if conservative:
                        diagnostic = env._last_conservative_charge
                        (case_dir / "conservative_virtual_windows.json").write_text(json.dumps(
                            [{"vehicle_id": vid, "station_id": sid, **window}
                             for (vid, sid), window in diagnostic["virtual_windows"].items()]))
                        (case_dir / "conservative_station_diagnostics.json").write_text(json.dumps(diagnostic["station_diagnostics"], indent=2))
                        if assignments is None or row.get("ssg_on_off_objective_gap_int") != 0:
                            raise AssertionError(f"Conservative fixed-graph solve failed: {row}")
                        if (row["cut_iterations"] or row["overcapacity_slot_epochs"] or row["new_queue_vehicle_count"]
                                or row["committed_reservation_delay_epochs"]
                                or row["ssg_off_cut_iterations"] or row["ssg_off_new_queue_vehicle_count"]
                                or row["ssg_off_overcapacity_slot_epochs"]):
                            raise AssertionError(f"Conservative calendar safety check failed: {row}")
                    all_rows.append(row)
                    pair.append(row)
                    with (output / "raw.jsonl").open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                if not np.all(masks["conservative"][:, :-1] <= masks["current"][:, :-1]):
                    raise AssertionError("Conservative filter invented a charge edge")
                np.savez_compressed(case_dir / "masks.npz", **masks)
                write_csv(output / "raw.csv", all_rows)
                write_csv(output / "station_metrics.csv", station_rows)
                print(f"{scenario:13s} C={count} seed={seed}: " + " | ".join(
                    f"{r['method']} charge={r.get('assigned_charge_vehicles', 'FAIL')} "
                    f"queue={r.get('new_queue_vehicle_count', 'NA')}" for r in pair), flush=True)
    summarize(output, all_rows)
    print(f"Complete: {output}", flush=True)
    return output


def summarize(output, rows):
    pairs = []
    lookup = {(r["scenario"], r["station_count"], r["seed"], r["method"]): r for r in rows}
    for key, current in lookup.items():
        if key[-1] != "current":
            continue
        conservative = lookup.get((*key[:-1], "conservative"))
        if not conservative:
            continue
        pair = dict(zip(("scenario", "station_count", "seed"), key[:-1]))
        pair["both_successful"] = "objective_int" in current and "objective_int" in conservative
        for metric in ("assigned_charge_vehicles", "assignment_score", "idle_slot_epochs", "idle_slots_at_epoch_1",
                       "idle_capacity_fraction", "preprocess_seconds", "assignment_wall_seconds", "end_to_end_seconds"):
            if metric in current and metric in conservative:
                pair[metric + "_delta_conservative_minus_current"] = conservative[metric] - current[metric]
        pairs.append(pair)
    write_csv(output / "paired.csv", pairs)
    summary = []
    metrics = ("charge_edges", "vehicles_with_charge", "assigned_charge_vehicles", "assignment_score",
               "idle_slot_epochs", "idle_slots_at_epoch_1", "idle_capacity_fraction", "new_queue_vehicle_count",
               "overcapacity_slot_epochs", "preprocess_seconds", "assignment_wall_seconds", "end_to_end_seconds")
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        for count in sorted({r["station_count"] for r in rows}):
            for method in ("current", "conservative"):
                group = [r for r in rows if (r["scenario"], r["station_count"], r["method"]) == (scenario, count, method)]
                if not group:
                    continue
                item = {"scenario": scenario, "station_count": count, "method": method,
                        "num_seeds": len(group), "successful_runs": sum("objective_int" in r for r in group)}
                for metric in metrics:
                    values = [r[metric] for r in group if metric in r]
                    item[metric + "_mean"] = float(np.mean(values)) if values else ""
                    item[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                summary.append(item)
    write_csv(output / "summary.csv", summary)
    lines = ["# Conservative charging: single-epoch comparison", "",
             "500 total vehicles by default; each scenario has 10 paired seeds, 3/5 stations and 50 slots/station.",
             "Synthetic states use real NYC charge/wait generation and the current OR-Tools adapter.",
             "SSG is exact relative to each fixed new mask; the current temporal repair is not a global oracle.",
             "Queue results are a fixed-calendar ChargingStation replay, not a full NYC.step rollout.", "",
             "| Scenario | Stations | Method | Charge count | Idle slot-epochs | Idle slots at epoch 1 | New queued |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for item in summary:
        vals = [item.get(metric + "_mean", "") for metric in
                ("assigned_charge_vehicles", "idle_slot_epochs", "idle_slots_at_epoch_1", "new_queue_vehicle_count")]
        formatted = [f"{v:.2f}" if isinstance(v, float) else "NA" for v in vals]
        lines.append(f"| {item['scenario']} | {item['station_count']} | {item['method']} | " + " | ".join(formatted) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicles", type=int, default=500)
    parser.add_argument("--stations", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--slots", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--wait-policy", choices=["nyc", "available"], default="nyc",
                        help="nyc uses the real low-SOC gate; available holds a non-charge alternative feasible for mask-only comparisons")
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
