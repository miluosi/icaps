"""Benchmark CPLEX, MCMF, SSG(CPLEX), and SSG(MCMF) on identical graphs.

This is the ICAPS counterpart of the ``adp_trainer`` assignment-time
experiment behind ``results/assignment_runtime_reward_comparison.xlsx`` and
``plot.ipynb``.  The old figure called its reduced formulation
``Scaled graph + Gurobi``.  Here ``SSG`` denotes that same exact
scaled/sparse action-graph idea, implemented with ICAPS' objective-preserving
action-graph reduction.

The four labels isolate two choices:

* CPLEX / MCMF: the complete feasible action graph;
* SSG(CPLEX) / SSG(MCMF): the same instance after exact graph reduction;
* CPLEX: ICAPS' DOcplex/CPLEX network-flow solver;
* MCMF: the explicitly selected single-threaded CPU flow backend.

The default suite grows from 100 to 6000 vehicles, with ten paired seeds at
each scale in a control, relocation-rich, and high-AEV joint-growth scenario.
All scenarios retain the ADP generator's reward distributions. Inputs and
incremental measurements are saved locally; plot them in
``plot_cplex_mcmf_ssg.ipynb``. Explicit ``--scales`` selects a single custom
scenario and retains the old generator/backend options.
No Python primal-dual backend is used. Every method receives the same quantized
Q-values, feasibility edges, action capacities, and zero-valued outside
action. All four methods are exact and their objectives are checked before
a case is marked complete; partial diagnostics are retained on failure.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
# Apply before importing NumPy when launched as a server CLI. The callable
# API additionally limits already-loaded BLAS runtimes in run_benchmark.
if __name__ == "__main__":
    for _thread_variable in THREAD_VARIABLES:
        os.environ[_thread_variable] = "1"

import numpy as np

from src.exact_mcmf import (
    ReducedMCMFProblem, build_reduced_problem, solve_docplex_network, solve_ortools,
)
from src.qvalue_precision import round_qvalue_matrix


ROOT = Path(__file__).resolve().parent
METHODS = ("cplex", "mcmf", "ssg_cplex", "ssg_mcmf")
METHOD_LABELS = {
    "cplex": "CPLEX",
    "mcmf": "MCMF (OR-Tools)",
    "ssg_cplex": "CPLEX + SSG",
    "ssg_mcmf": "MCMF (OR-Tools) + SSG",
}
DEFAULT_VEHICLE_COUNTS = (100, 500, 1000, 2000, 3000, 6000)
SCENARIOS = {
    "adp_control": {
        "title": "ADP control", "aev_ratio": 0.5, "charging_ratio": 0.2,
        "relocation_ratio": 0.1,
        "description": "Original large-scale action ratios; matched fixed station capacity 4.",
    },
    "reloc_rich": {
        "title": "Relocation-rich", "aev_ratio": 0.5, "charging_ratio": 0.2,
        "relocation_ratio": 2.0,
        "description": "More feasible nonbinding relocation actions; default ADP density floor.",
    },
    "aev_joint": {
        "title": "High AEV, stations and relocation", "aev_ratio": 0.9,
        "charging_ratio": 1.0, "relocation_ratio": 1.0,
        "description": "90% AEV with stations and relocation regions each equal to fleet size.",
    },
    "fixed_candidates": {
        "title": "Fixed relocation candidate budget", "aev_ratio": 0.5,
        "charging_ratio": 0.2, "relocation_ratio": 2.0,
        "description": "Control: relocation density 4/Z plus a guaranteed candidate; about five candidates per AEV.",
    },
}
DEFAULT_SCENARIOS = ("adp_control", "reloc_rich", "aev_joint")
DEFAULT_SCALES = (
    "100:500:50:10",
    "500:2500:100:50",
    "1000:5000:200:100",
    "2000:10000:400:200",
    "3000:15000:600:300",
)


@dataclass(frozen=True)
class Scale:
    vehicles: int
    requests: int
    charging: int
    relocation: int

    @property
    def label(self) -> str:
        return f"{self.vehicles}V/{self.requests}R"


@dataclass
class AssignmentCase:
    scale: Scale
    seed: int
    feasibility: np.ndarray
    q_values: np.ndarray
    capacities: np.ndarray
    fallback_values: np.ndarray | None
    feasible_edges: int
    generation_seconds: float


@dataclass(frozen=True)
class SolverResult:
    objective_int: int
    objective_q: float
    flow: int
    status: str
    backend: str


def parse_scale(value: str) -> Scale:
    """Parse ``vehicles:requests[:charging:relocation]``."""

    try:
        parts = [int(part) for part in value.replace(",", ":").split(":")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid scale {value!r}") from exc
    if len(parts) == 2:
        vehicles, requests = parts
        charging = max(1, vehicles // 5)
        relocation = max(1, vehicles // 10)
    elif len(parts) == 4:
        vehicles, requests, charging, relocation = parts
    else:
        raise argparse.ArgumentTypeError(
            "scale must be V:R or V:R:C:Z, for example 200:1000:40:20"
        )
    if min(parts) <= 0:
        raise argparse.ArgumentTypeError("all scale dimensions must be positive")
    return Scale(vehicles, requests, charging, relocation)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=parse_scale,
        default=None,
        metavar="V:R[:C:Z]",
        help=(
            "explicit custom scenario; V:R derives C=V/5 and Z=V/10; "
            "mutually exclusive with --scenarios and --vehicle-counts"
        ),
    )
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=None)
    parser.add_argument("--vehicle-counts", nargs="+", type=int, default=None,
                        help="fleet sizes for scenario suite; default 100 500 1000 2000 3000 6000")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument("--repeats", type=int, default=10)
    seeds_group.add_argument("--seeds", nargs="+", type=int, default=None,
                             help="explicit distinct seeds; default seed through seed+9")
    parser.add_argument(
        "--case-distribution", choices=("adp", "sparse"), default="adp",
        help="adp matches adp_trainer/test_alg_time.py matrices; sparse retains the original fixed candidate generator",
    )
    parser.add_argument("--seed", type=int, default=2101)
    fleet_group = parser.add_mutually_exclusive_group()
    fleet_group.add_argument("--ev-ratio", type=float, default=None, help="human-driven EV fraction")
    fleet_group.add_argument("--aev-ratio", type=float, default=None, help="autonomous EV fraction")
    parser.add_argument("--request-candidates", type=int, default=8)
    parser.add_argument("--charge-candidates", type=int, default=4)
    parser.add_argument("--relocation-candidates", type=int, default=4)
    parser.add_argument("--station-capacity", type=int, default=4)
    parser.add_argument("--request-density", type=float, default=None)
    parser.add_argument("--charge-density", type=float, default=None)
    parser.add_argument("--reloc-density", type=float, default=None)
    parser.add_argument("--fixed-charge-capacity", type=int, default=None)
    parser.add_argument("--cost-scale", type=int, default=10_000)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument(
        "--mcmf-backend", choices=("legacy", "ortools"), default="ortools",
        help="legacy reproduces Python SPFA; ortools uses compiled cost-scaling MCMF",
    )
    parser.add_argument(
        "--warmup", action=argparse.BooleanOptionalAction, default=True,
        help="initialize selected backends on a small problem before timing",
    )
    parser.add_argument(
        "--plot-metric",
        choices=("end_to_end_seconds", "solve_seconds"),
        default="end_to_end_seconds",
        help="metric drawn in the comparison plot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: results/cplex_mcmf_ssg/<timestamp>",
    )
    parser.add_argument("--no-plot", action="store_true", default=True)
    parser.add_argument("--plot", dest="no_plot", action="store_false",
                        help="optional legacy PNG export; normally use the local notebook")
    parser.add_argument("--save-inputs", action=argparse.BooleanOptionalAction, default=True,
                        help="save each exact input as a lossless sparse NPZ outside the timed region")
    parser.add_argument("--resume", action="store_true",
                        help="continue --output-dir, skipping completed paired cases with identical settings and solver sources")
    parser.add_argument(
        "--allow-objective-mismatch",
        action="store_true",
        help="save diagnostics instead of failing if exact objectives differ",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    for name in ("ev_ratio", "aev_ratio"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.aev_ratio is not None:
        args.ev_ratio = 1.0 - args.aev_ratio
    if args.scales is not None and (args.scenarios is not None or args.vehicle_counts is not None):
        parser.error("--scales is a custom scenario; do not combine it with --scenarios/--vehicle-counts")
    if args.scales is None:
        if args.case_distribution != "adp":
            parser.error("--case-distribution sparse requires explicit --scales")
        args.scenarios = args.scenarios or list(DEFAULT_SCENARIOS)
        args.vehicle_counts = sorted(args.vehicle_counts or DEFAULT_VEHICLE_COUNTS)
        if min(args.vehicle_counts) <= 0 or len(set(args.vehicle_counts)) != len(args.vehicle_counts):
            parser.error("--vehicle-counts must be positive and distinct")
    else:
        args.scenarios = ["custom"]
        args.scales = sorted(args.scales, key=lambda scale: (scale.vehicles, scale.requests, scale.charging, scale.relocation))
        if len(set(args.scales)) != len(args.scales):
            parser.error("--scales must be distinct")
    if len(set(args.methods)) != len(args.methods) or len(set(args.scenarios)) != len(args.scenarios):
        parser.error("methods and scenarios must be distinct")
    args.seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.repeats))
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be nonnegative and distinct")
    args.repeats = len(args.seeds)
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    for name in ("request_candidates", "charge_candidates", "relocation_candidates"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.station_capacity <= 0 or args.cost_scale <= 0 or args.cplex_threads <= 0:
        parser.error("capacity, cost scale, and CPLEX thread count must be positive")
    if args.cplex_threads != 1:
        parser.error("this benchmark requires --cplex-threads 1")
    for name in ("request_density", "charge_density", "reloc_density"):
        value = getattr(args, name)
        if value is not None and not 0 < value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.fixed_charge_capacity is not None and args.fixed_charge_capacity < 0:
        parser.error("--fixed-charge-capacity must be nonnegative")
    return args


def _choice(
    rng: np.Generator,
    population: int,
    count: int,
) -> np.ndarray:
    if population <= 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(
        rng.choice(population, size=min(population, count), replace=False),
        dtype=np.int64,
    )


def build_case(
    scale: Scale,
    seed: int,
    *,
    ev_ratio: float,
    request_candidates: int,
    charge_candidates: int,
    relocation_candidates: int,
    station_capacity: int,
) -> AssignmentCase:
    """Build one sparse-feasibility ICAPS assignment instance.

    The matrices remain in the production ``(vehicle, action)`` format.
    Request actions have unit capacity, charging actions have the configured
    station capacity, and relocation actions are nonbinding.  Every vehicle
    also has a private zero-value outside action represented by
    ``fallback_values``.
    """

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    num_actions = scale.requests + scale.charging + scale.relocation
    feasibility = np.zeros((scale.vehicles, num_actions), dtype=np.bool_)
    q_values = np.zeros((scale.vehicles, num_actions), dtype=np.float32)
    charge_start = scale.requests
    relocation_start = charge_start + scale.charging

    num_ev = int(round(scale.vehicles * ev_ratio))
    vehicle_is_ev = np.zeros(scale.vehicles, dtype=np.bool_)
    vehicle_is_ev[:num_ev] = True
    rng.shuffle(vehicle_is_ev)

    for row in range(scale.vehicles):
        request_actions = _choice(rng, scale.requests, request_candidates)
        feasibility[row, request_actions] = True
        q_values[row, request_actions] = rng.uniform(
            8.0, 35.0, size=request_actions.size
        ).astype(np.float32)

        # Human-driven EVs receive requests plus the outside action.  AEVs
        # additionally receive charging and relocation actions, matching the
        # mixed-fleet first-step action layout.
        if not vehicle_is_ev[row]:
            charge_actions = charge_start + _choice(
                rng, scale.charging, charge_candidates
            )
            feasibility[row, charge_actions] = True
            q_values[row, charge_actions] = rng.uniform(
                1.0, 12.0, size=charge_actions.size
            ).astype(np.float32)

            relocation_actions = relocation_start + _choice(
                rng, scale.relocation, relocation_candidates
            )
            feasibility[row, relocation_actions] = True
            q_values[row, relocation_actions] = rng.uniform(
                0.0, 8.0, size=relocation_actions.size
            ).astype(np.float32)

    capacities = np.empty(num_actions, dtype=np.int64)
    capacities[:scale.requests] = 1
    capacities[charge_start:relocation_start] = int(station_capacity)
    capacities[relocation_start:] = scale.vehicles
    fallback_values = np.zeros(scale.vehicles, dtype=np.float64)
    return AssignmentCase(
        scale=scale,
        seed=seed,
        feasibility=feasibility,
        q_values=q_values,
        capacities=capacities,
        fallback_values=fallback_values,
        feasible_edges=int(np.count_nonzero(feasibility)),
        generation_seconds=time.perf_counter() - started,
    )


def build_adp_case(
    scale: Scale, seed: int, *, ev_ratio: float = 0.5,
    request_density: float | None = None,
    charge_density: float | None = None,
    reloc_density: float | None = None,
    fixed_charge_capacity: int | None = None,
) -> AssignmentCase:
    """Match adp_trainer/test_alg_time.py build_matrix's RNG order and inputs.

    This uses Bernoulli masks with the same guaranteed candidate per row,
    random charger capacities, explicit zero-value wait column, invalid-Q
    sentinel, and four-decimal rounding. It does not import the other project.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    n, requests, charging, relocation = (
        scale.vehicles, scale.requests, scale.charging, scale.relocation,
    )
    num_actions = requests + charging + relocation + 1
    feasible = np.zeros((n, num_actions), dtype=bool)
    values = np.full((n, num_actions), -1.0e6, dtype=np.float32)
    vehicle_types = np.full(n, 2, dtype=np.int8)
    vehicle_types[:int(round(n * ev_ratio))] = 1
    rng.shuffle(vehicle_types)
    request_density = request_density or min(0.25, max(8.0 / max(requests, 1), 0.01))
    charge_density = charge_density or min(0.4, max(4.0 / max(charging, 1), 0.05))
    reloc_density = reloc_density or min(0.5, max(4.0 / max(relocation, 1), 0.1))
    request_mask = rng.random((n, requests)) < request_density
    if requests:
        best = rng.integers(0, requests, size=n)
        request_mask[np.arange(n), best] = True
        feasible[:, :requests] = request_mask
        rewards = rng.uniform(8.0, 35.0, size=(n, requests))
        values[:, :requests] = np.where(request_mask, rewards, -1.0e6)
    aev_rows = vehicle_types == 2
    for offset, count, density, low, high in (
        (requests, charging, charge_density, 1.0, 12.0),
        (requests + charging, relocation, reloc_density, 0.0, 8.0),
    ):
        if count and np.any(aev_rows):
            mask = rng.random((n, count)) < density
            mask[~aev_rows, :] = False
            indices = np.flatnonzero(aev_rows)
            best = rng.integers(0, count, size=len(indices))
            mask[indices, best] = True
            feasible[:, offset:offset + count] = mask
            rewards = rng.uniform(low, high, size=(n, count))
            values[:, offset:offset + count] = np.where(mask, rewards, -1.0e6)
    feasible[:, -1] = True
    values[:, -1] = 0.0
    values = round_qvalue_matrix(values, 10_000)
    capacities = np.full(num_actions, n, dtype=np.int64)
    capacities[:requests] = 1
    if fixed_charge_capacity is not None:
        capacities[requests:requests + charging] = fixed_charge_capacity
    else:
        base = max(1, math.ceil(n / max(charging, 1) / 2))
        capacities[requests:requests + charging] = rng.integers(1, base + 2, size=charging)
    return AssignmentCase(
        scale=scale, seed=seed, feasibility=feasible, q_values=values,
        capacities=capacities, fallback_values=None,
        feasible_edges=int(np.count_nonzero(feasible)),
        generation_seconds=time.perf_counter() - started,
    )


def solve_ssp_mcmf(problem: ReducedMCMFProblem) -> SolverResult:
    """Solve a canonical ICAPS flow graph with the production CPU SSP code."""

    # Import lazily so ``--help`` and CPLEX-only runs do not initialize the
    # simulation/Gym stack.  The actual implementation is the same class used
    # by GurobiOptimizer's ``mcmf_solver='legacy'`` production path.
    from src.GurobiOptimizer import GurobiOptimizer

    solver = GurobiOptimizer._MCMFSolver(problem.num_nodes)
    handles: list[tuple[int, int, int]] = []
    for tail, head, capacity, cost in zip(
        problem.tails, problem.heads, problem.capacities, problem.costs
    ):
        u = int(tail)
        edge_index = len(solver.graph[u])
        initial_capacity = int(capacity)
        solver.add_edge(u, int(head), initial_capacity, int(cost))
        handles.append((u, edge_index, initial_capacity))

    flow_raw, min_cost_raw = solver.solve(problem.source, problem.sink, time_limit=None)
    flow = int(round(flow_raw))
    min_cost = int(round(min_cost_raw))
    if flow != problem.target_flow:
        raise RuntimeError(f"MCMF sent {flow}/{problem.target_flow} units")

    action_by_vehicle: dict[int, int] = {}
    decoded_cost = 0
    for (u, edge_index, initial_capacity), cost, meta in zip(
        handles, problem.costs, problem.arc_meta
    ):
        residual_capacity = int(round(solver.graph[u][edge_index][1]))
        arc_flow = initial_capacity - residual_capacity
        decoded_cost += arc_flow * int(cost)
        if arc_flow <= 0 or meta.kind not in {"baseline", "shared"}:
            continue
        if meta.vehicle_index is None or meta.action_index is None:
            raise AssertionError("selected assignment arc is missing metadata")
        if meta.vehicle_index in action_by_vehicle:
            raise AssertionError("a vehicle received more than one action")
        action_by_vehicle[int(meta.vehicle_index)] = int(meta.action_index)
    if decoded_cost != min_cost:
        raise AssertionError(f"decoded cost {decoded_cost} != solver cost {min_cost}")
    if len(action_by_vehicle) != problem.target_flow:
        raise AssertionError(
            f"decoded {len(action_by_vehicle)}/{problem.target_flow} vehicle actions"
        )

    loads = np.zeros(len(problem.action_capacities), dtype=np.int64)
    recomputed_objective = 0
    for vehicle, action in action_by_vehicle.items():
        if action < 0:
            if problem.fallback_value is None:
                raise AssertionError("fallback selected without a fallback value")
            recomputed_objective += int(problem.fallback_value[vehicle])
            continue
        if not problem.feasibility[vehicle, action]:
            raise AssertionError("MCMF selected an infeasible action")
        loads[action] += 1
        recomputed_objective += int(problem.q_values_int[vehicle, action])
    if np.any(loads > problem.action_capacities):
        raise AssertionError("MCMF assignment violates an action capacity")

    objective_int = int(problem.baseline_sum - min_cost)
    if recomputed_objective != objective_int:
        raise AssertionError(
            f"decoded objective {recomputed_objective} != {objective_int}"
        )
    return SolverResult(
        objective_int=objective_int,
        objective_q=objective_int / float(problem.cost_scale),
        flow=flow,
        status="OPTIMAL",
        backend="production_cpu_ssp",
    )


def solve_method(
    method: str, problem: ReducedMCMFProblem, threads: int,
    mcmf_backend: str = "ortools",
) -> SolverResult:
    if method in {"cplex", "ssg_cplex"}:
        result = solve_docplex_network(problem, num_threads=threads)
        return SolverResult(
            objective_int=int(result.objective_int),
            objective_q=float(result.objective_q),
            flow=int(result.flow),
            status=str(result.status),
            backend="docplex_network_cplex",
        )
    if method in {"mcmf", "ssg_mcmf"}:
        if mcmf_backend == "ortools":
            result = solve_ortools(problem)
            return SolverResult(
                objective_int=result.objective_int,
                objective_q=result.objective_q,
                flow=result.flow,
                status=result.status,
                backend="ortools_cost_scaling",
            )
        if mcmf_backend != "legacy":
            raise ValueError(f"unknown MCMF backend: {mcmf_backend}")
        return solve_ssp_mcmf(problem)
    raise ValueError(f"unknown method: {method}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_case_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resolve every effective parameter, including preset overrides, for replay."""
    specs = []
    for scenario in args.scenarios:
        preset = SCENARIOS.get(scenario)
        scales = args.scales if preset is None else [
            Scale(n, 5*n, max(1, int(n*preset["charging_ratio"])),
                  max(1, int(n*preset["relocation_ratio"])))
            for n in args.vehicle_counts
        ]
        ev_ratio = args.ev_ratio
        if ev_ratio is None:
            ev_ratio = 0.5 if preset is None else 1.0 - preset["aev_ratio"]
        fixed_capacity = args.fixed_charge_capacity
        if fixed_capacity is None and preset is not None:
            fixed_capacity = 4
        for index, scale in enumerate(scales):
            generator = dict(
                ev_ratio=ev_ratio,
                request_density=args.request_density or min(0.25, max(8/scale.requests, 0.01)),
                charge_density=args.charge_density or min(0.4, max(4/scale.charging, 0.05)),
                reloc_density=args.reloc_density or (
                    min(1.0, 4/scale.relocation) if scenario == "fixed_candidates"
                    else min(0.5, max(4/scale.relocation, 0.1))),
                fixed_charge_capacity=fixed_capacity,
            )
            specs.append(dict(
                scenario=scenario,
                scenario_title=preset["title"] if preset else "Custom scenario",
                scenario_description=preset["description"] if preset else "Explicit command-line dimensions",
                scale_index=index, scale=asdict(scale), generator=generator,
            ))
    return specs


def save_case_input(case: AssignmentCase, path: Path, *, cost_scale: int,
                    case_distribution: str) -> str:
    """Losslessly archive these generators' dense inputs using feasible COO entries.

    All infeasible entries have the documented constant fill (ADP -1e6,
    sparse 0). Raw feasible floats and their dtype are retained, rather than
    quantizing again for storage. Archive work is outside the timing scope.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, actions = np.nonzero(case.feasibility)
    info = dict(format_version=1, scale=asdict(case.scale), seed=case.seed,
                shape=list(case.feasibility.shape), q_dtype=case.q_values.dtype.str,
                infeasible_q=-1.0e6 if case_distribution == "adp" else 0.0,
                case_distribution=case_distribution, cost_scale=cost_scale,
                generation_seconds=case.generation_seconds,
                has_fallback=case.fallback_values is not None)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, metadata=np.asarray(json.dumps(info)),
            rows=rows.astype(np.int32), actions=actions.astype(np.int32),
            q_values=case.q_values[rows, actions], capacities=case.capacities,
            fallback_values=(case.fallback_values if case.fallback_values is not None else np.empty(0)),
        )
    temporary.replace(path)
    return file_sha256(path)


def load_case_input(path: Path) -> AssignmentCase:
    """Restore an archived benchmark matrix, without needing a solver installed."""
    with np.load(path, allow_pickle=False) as data:
        info = json.loads(str(data["metadata"]))
        if info["format_version"] != 1:
            raise ValueError("unsupported input archive version")
        feasible = np.zeros(info["shape"], dtype=bool)
        q = np.full(info["shape"], info["infeasible_q"], dtype=np.dtype(info["q_dtype"]))
        rows, actions = data["rows"], data["actions"]
        feasible[rows, actions] = True
        q[rows, actions] = data["q_values"]
        return AssignmentCase(
            Scale(**info["scale"]), info["seed"], feasible, q,
            data["capacities"].copy(),
            data["fallback_values"].copy() if info["has_fallback"] else None,
            len(rows), info["generation_seconds"],
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("scenario", "custom")),
            int(row["vehicles"]),
            int(row["requests"]),
            int(row["charging"]),
            int(row["relocation"]),
            str(row["method"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        scenario, vehicles, requests, charging, relocation, method = key
        solve_times = [float(row["solve_seconds"]) for row in group]
        end_to_end = [float(row["end_to_end_seconds"]) for row in group]
        objectives = [float(row["objective_q"]) for row in group]
        summary_rows.append({
            "scenario": scenario,
            "scenario_title": group[0].get("scenario_title", scenario),
            "vehicles": vehicles,
            "requests": requests,
            "charging": charging,
            "relocation": relocation,
            "scale_label": f"{vehicles}V/{requests}R",
            "method": method,
            "method_label": group[0]["method_label"],
            "runs": len(group),
            "seeds": ";".join(str(row["seed"]) for row in group),
            "graph_build_mean_seconds": statistics.fmean(float(row["graph_build_seconds"]) for row in group),
            "solve_mean_seconds": statistics.fmean(solve_times),
            "solve_median_seconds": statistics.median(solve_times),
            "solve_std_seconds": statistics.stdev(solve_times) if len(solve_times) > 1 else 0.0,
            "end_to_end_mean_seconds": statistics.fmean(end_to_end),
            "end_to_end_median_seconds": statistics.median(end_to_end),
            "end_to_end_std_seconds": statistics.stdev(end_to_end) if len(end_to_end) > 1 else 0.0,
            "objective_q_mean": statistics.fmean(objectives),
            "graph_edges_mean": statistics.fmean(float(row["graph_edges"]) for row in group),
            "edge_reduction_ratio_mean": statistics.fmean(
                float(row["edge_reduction_ratio"]) for row in group
            ),
        })
    return sorted(summary_rows, key=lambda row: (row["scenario"], row["vehicles"], row["requests"], row["charging"], row["relocation"], METHODS.index(row["method"])))


def paired_speedups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[row["case_id"]][row["method"]] = row
    paired = []
    for case_id, methods in groups.items():
        for full, reduced in (("cplex", "ssg_cplex"), ("mcmf", "ssg_mcmf")):
            if full not in methods or reduced not in methods:
                continue
            a, b = methods[full], methods[reduced]
            paired.append({
                "case_id": case_id, "scenario": a["scenario"], "vehicles": a["vehicles"],
                "requests": a["requests"], "charging": a["charging"], "relocation": a["relocation"],
                "seed": a["seed"], "comparison": f"{full}_over_{reduced}",
                "objective_match": a["objective_int"] == b["objective_int"],
                "end_to_end_speedup": a["end_to_end_seconds"] / b["end_to_end_seconds"],
                "solve_speedup": a["solve_seconds"] / b["solve_seconds"],
                "time_saved_seconds": a["end_to_end_seconds"] - b["end_to_end_seconds"],
                "ssg_faster": a["end_to_end_seconds"] > b["end_to_end_seconds"],
                "edge_reduction_ratio": b["edge_reduction_ratio"],
            })
    return paired


def save_plot(
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    metric: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suffix = "end_to_end" if metric == "end_to_end_seconds" else "solve"
    mean_key = f"{suffix}_mean_seconds"
    std_key = f"{suffix}_std_seconds"
    scales = sorted(
        {(int(row["vehicles"]), str(row["scale_label"])) for row in summary_rows}
    )
    labels = [label for _vehicles, label in scales]
    vehicles = [vehicle_count for vehicle_count, _label in scales]
    dimensions = {(row["vehicles"], row["requests"], row["charging"], row["relocation"]) for row in summary_rows}
    if len({entry[0] for entry in dimensions}) != len(dimensions):
        raise ValueError("plot one R/C/Z configuration per vehicle count and scenario")
    plot_labels = {row["method"]: row["method_label"] for row in summary_rows}
    colors = {
        "cplex": "#E63946",
        "mcmf": "#1D3557",
        "ssg_cplex": "#2A9D8F",
        "ssg_mcmf": "#8D99AE",
    }
    lookup = {
        (int(row["vehicles"]), str(row["method"])): row for row in summary_rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=200)
    x = np.arange(len(scales))
    width = 0.18
    for index, method in enumerate(METHODS):
        if not any(row["method"] == method for row in summary_rows):
            continue
        means = [float(lookup[(v, method)][mean_key]) for v in vehicles]
        stds = [float(lookup[(v, method)][std_key]) for v in vehicles]
        offset = (index - (len(METHODS) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=plot_labels[method],
            color=colors[method],
            alpha=0.9,
            capsize=3,
            edgecolor="white",
            linewidth=0.6,
        )
        axes[1].plot(
            vehicles,
            means,
            marker="o",
            markersize=7,
            linewidth=2.2,
            label=plot_labels[method],
            color=colors[method],
        )

    positive = [
        float(row[mean_key]) for row in summary_rows if float(row[mean_key]) > 0.0
    ]
    if positive:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
    if vehicles and min(vehicles) > 0:
        axes[1].set_xscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Problem Scale (Vehicles / Requests)", fontweight="bold")
    axes[0].set_ylabel("Time (s, log scale)", fontweight="bold")
    axes[0].set_title("ICAPS Assignment Time vs Problem Scale", fontweight="bold")
    axes[0].legend(ncol=2, loc="upper left")
    axes[0].grid(True, alpha=0.3, axis="y", which="both")
    axes[1].set_xticks(vehicles)
    axes[1].set_xticklabels([str(value) for value in vehicles])
    axes[1].set_xlabel("Number of Vehicles (log scale)", fontweight="bold")
    axes[1].set_ylabel("Time (s, log scale)", fontweight="bold")
    axes[1].set_title("Scaling Behavior of ICAPS Solvers", fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3, which="both")
    fig.suptitle(
        "CPLEX vs MCMF, with and without exact SSG reduction\n"
        f"metric: {metric.replace('_', ' ')}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _check_objectives(
    rows: list[dict[str, Any]],
    *,
    allow_mismatch: bool,
) -> None:
    objectives = [int(row["objective_int"]) for row in rows]
    gap = max(objectives) - min(objectives)
    for row in rows:
        row["objective_gap_int"] = int(int(row["objective_int"]) - max(objectives))
        row["objective_match"] = gap == 0
    if gap and not allow_mismatch:
        details = {str(row["method"]): int(row["objective_int"]) for row in rows}
        raise AssertionError(f"exact solver objective mismatch: {details}")


def _runtime_metadata() -> dict[str, Any]:
    from threadpoolctl import threadpool_info

    versions = {}
    for name in ("numpy", "cplex", "docplex", "ortools", "threadpoolctl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "processor": platform.processor(),
        "numpy": np.__version__, "solver_versions": versions,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "threadpools": threadpool_info(),
    }


def run_benchmark(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    from threadpoolctl import threadpool_limits

    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    with threadpool_limits(limits=1):
        return _run_benchmark_serial(args)


class OutputDirectoryError(ValueError):
    """An output path needs a different new-run/resume choice."""


def prepare_output_directory(output_dir: Path, *, resume: bool) -> None:
    """Accept a new or pre-created empty directory, preserving existing data."""
    if resume:
        if not (output_dir / "metadata.json").is_file():
            raise OutputDirectoryError(
                f"Cannot resume {output_dir}: metadata.json is missing. "
                "Use an interrupted experiment directory, or omit --output-dir "
                "and --resume to start in a new timestamped directory."
            )
        return
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        return
    except FileExistsError:
        if output_dir.is_dir() and not any(output_dir.iterdir()):
            return
    if (output_dir / "metadata.json").is_file():
        advice = (
            "To continue an interrupted experiment, repeat its original command "
            "with --resume. To start a separate experiment, omit --output-dir "
            "for a new timestamped directory or choose a new path."
        )
    else:
        advice = (
            "This path is not an empty directory and has no experiment metadata.json. "
            "Omit --output-dir for a new timestamped directory or choose a new path."
        )
    raise OutputDirectoryError(f"Output path already exists: {output_dir}. {advice} Existing files were not changed.")


def _run_benchmark_serial(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    specs = resolve_case_specs(args)
    settings = {
        "case_specs": specs, "seeds": args.seeds, "methods": args.methods,
        **{key: getattr(args, key) for key in (
            "case_distribution", "mcmf_backend", "cplex_threads", "cost_scale",
            "warmup", "save_inputs", "request_candidates", "charge_candidates",
            "relocation_candidates", "station_capacity", "allow_objective_mismatch")},
    }
    source_hashes = {name: file_sha256(ROOT / name) for name in (
        "benchmark_cplex_mcmf_ssg.py", "src/exact_mcmf.py", "src/qvalue_precision.py")}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (args.output_dir or ROOT / "results" / "cplex_mcmf_ssg" / timestamp).resolve()
    runtime = _runtime_metadata()
    rows: list[dict[str, Any]] = []
    completed_cases: set[str] = set()
    labels = dict(METHOD_LABELS)
    if args.mcmf_backend == "legacy":
        labels.update(mcmf="MCMF (Python SPFA)", ssg_mcmf="MCMF (Python SPFA) + SSG")

    prepare_output_directory(output_dir, resume=args.resume)
    if args.resume:
        metadata = json.loads((output_dir / "metadata.json").read_text())
        if metadata["experiment_settings"] != settings or metadata["source_sha256"] != source_hashes:
            raise ValueError("Resume settings or solver sources differ; use the original command/code or a new output directory")
        for key in ("python", "platform", "processor", "solver_versions"):
            if metadata["runtime"][key] != runtime[key]:
                raise ValueError(f"Resume runtime differs ({key}); use a new output directory")
        if (output_dir / "raw_results.json").exists():
            saved = json.loads((output_dir / "raw_results.json").read_text())
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in saved:
                grouped[row["case_id"]].append(row)
            for case_id, group in grouped.items():
                if (len(group) == len(args.methods)
                        and {row["method"] for row in group} == set(args.methods)
                        and all(row["case_complete"] for row in group)):
                    if args.save_inputs:
                        path = output_dir / group[0]["input_file"]
                        if not path.is_file() or file_sha256(path) != group[0]["input_sha256"]:
                            raise ValueError(f"Missing or changed input archive for {case_id}")
                    rows.extend(group)
                    completed_cases.add(case_id)
    else:
        metadata = {
            "schema_version": 2, "created_at": datetime.now().isoformat(),
            "experiment_settings": settings, "source_sha256": source_hashes,
            "methods": labels, "primal_dual_used": False,
            "method_definition": {
                "cplex": "full graph + DOcplex/CPLEX continuous network LP (lpmethod=auto)",
                "mcmf": f"full graph + {args.mcmf_backend} MCMF",
                "ssg_cplex": "exact reduced graph + same CPLEX continuous network LP",
                "ssg_mcmf": f"exact reduced graph + same {args.mcmf_backend} MCMF",
            },
            "input_contract": {
                "distribution": args.case_distribution,
                "adp_matrix_generation_matched": args.case_distribution == "adp",
                "adp_parameters_modified_by_scenarios": args.scales is None,
                "seed_schedule": "identical explicit seed list at every scale and scenario",
                "reward_ranges": {"request": [8, 35], "charging": [1, 12], "relocation": [0, 8], "wait": 0},
                "timing_scope": "graph_build_seconds + solve_seconds; excludes input generation, archive I/O, and explicit inter-method GC",
                "solve_seconds": "backend model creation + optimization + decode and exact assignment checks; not optimizer-only time",
                "graph_build_seconds": "dense validation and quantization + optional exact SSG + flow graph construction",
                "shared_build": "Each full/reduced graph is constructed once per case; its full build time is attributed to each backend using it",
                "execution_order": "full/reduced group order alternates; engine order independently alternates on a four-repeat cycle",
                "input_archive": "lossless sparse NPZ: raw feasible Q values, constant infeasible fill, capacities, fallback, shape and dtype",
            },
            "args": {**{key: value for key, value in vars(args).items() if key not in {"scales", "output_dir"}},
                     "scales": [asdict(scale) for scale in args.scales] if args.scales is not None else None,
                     "output_dir": str(output_dir)},
            "runtime": runtime, "sessions": [],
        }
    metadata["sessions"].append({"started_at": datetime.now().isoformat(), "resume": args.resume})
    metadata["planned_cases"] = len(specs) * len(args.seeds)
    metadata["planned_rows"] = metadata["planned_cases"] * len(args.methods)
    metadata["run_status"] = "running"
    metadata.pop("failure", None)

    def checkpoint() -> None:
        complete = [row for row in rows if row["case_complete"]]
        _write_json(output_dir / "raw_results.json", rows)
        _write_csv(output_dir / "raw_results.csv", rows)
        _write_csv(output_dir / "summary.csv", summarize(complete))
        _write_csv(output_dir / "paired_speedups.csv", paired_speedups(complete))
        metadata["completed_cases"] = len({row["case_id"] for row in complete})
        metadata["completed_rows"] = len(complete)
        metadata["all_objectives_match"] = bool(complete) and all(row["objective_match"] for row in complete)
        metadata["updated_at"] = datetime.now().isoformat()
        _write_json(output_dir / "metadata.json", metadata)

    checkpoint()
    print(f"Output: {output_dir}\nCases: {metadata['planned_cases']}; seeds: {args.seeds}; methods: {args.methods}", flush=True)
    try:
        if args.warmup and len(completed_cases) < metadata["planned_cases"]:
            warmup_problem = build_reduced_problem(
                np.ones((2, 2), dtype=bool), np.array([[3.0, 2.0], [4.0, 1.0]]),
                np.ones(2, dtype=np.int64), fallback_values=np.zeros(2), graph_reduction=False)
            for method in args.methods:
                solve_method(method, warmup_problem, args.cplex_threads, args.mcmf_backend)
            del warmup_problem

        for spec_index, spec in enumerate(specs):
            scale = Scale(**spec["scale"])
            for repeat, seed in enumerate(args.seeds):
                case_id = f"{spec['scenario']}/v{scale.vehicles}_r{scale.requests}_c{scale.charging}_z{scale.relocation}/seed{seed}"
                if case_id in completed_cases:
                    print(f"Skip complete: {case_id}", flush=True)
                    continue
                if args.case_distribution == "adp":
                    case = build_adp_case(scale, seed, **spec["generator"])
                else:
                    case = build_case(
                        scale, seed, ev_ratio=spec["generator"]["ev_ratio"],
                        request_candidates=args.request_candidates,
                        charge_candidates=args.charge_candidates,
                        relocation_candidates=args.relocation_candidates,
                        station_capacity=args.station_capacity)
                input_file, input_hash = "", ""
                if args.save_inputs:
                    input_file = f"inputs/{case_id}.npz"
                    input_hash = save_case_input(case, output_dir / input_file,
                                                cost_scale=args.cost_scale,
                                                case_distribution=args.case_distribution)
                r, c, z = scale.requests, scale.charging, scale.relocation
                input_counts = {
                    "request_edges": int(np.count_nonzero(case.feasibility[:, :r])),
                    "charge_edges": int(np.count_nonzero(case.feasibility[:, r:r+c])),
                    "relocation_edges": int(np.count_nonzero(case.feasibility[:, r+c:r+c+z])),
                }
                print(f"[{case_id}] repeat={repeat+1}/{len(args.seeds)} feasible_edges={case.feasible_edges}", flush=True)
                case_rows = []
                groups = [(False, ("cplex", "mcmf")), (True, ("ssg_cplex", "ssg_mcmf"))]
                cycle = repeat + spec_index
                if cycle % 2:
                    groups.reverse()
                execution_index = 0
                for group_index, (reduced, methods) in enumerate(groups):
                    selected = [method for method in methods if method in args.methods]
                    if not selected:
                        continue
                    if (cycle // 2) % 2:
                        selected.reverse()
                    gc.collect()
                    started = time.perf_counter()
                    problem = build_reduced_problem(
                        case.feasibility, case.q_values, case.capacities,
                        cost_scale=args.cost_scale, fallback_values=case.fallback_values,
                        preserve_zero_gain_ties=not reduced, graph_reduction=reduced)
                    graph_build_seconds = time.perf_counter() - started
                    reduction_ratio = 1.0 - problem.reduced_edges / problem.original_edges if problem.original_edges else 0.0
                    # Diagnostics are outside the timer; metadata arrays avoid
                    # constructing one Python object per arc.
                    action_ids = problem.arc_meta.actions[problem.arc_meta.kinds == 2]
                    retained = {
                        "retained_request_edges": int(np.count_nonzero(action_ids < r)),
                        "retained_charge_edges": int(np.count_nonzero((action_ids >= r) & (action_ids < r+c))),
                        "retained_relocation_edges": int(np.count_nonzero((action_ids >= r+c) & (action_ids < r+c+z))),
                    }
                    del action_ids
                    for method in selected:
                        gc.collect()
                        started = time.perf_counter()
                        result = solve_method(method, problem, args.cplex_threads, args.mcmf_backend)
                        solve_seconds = time.perf_counter() - started
                        if result.flow != scale.vehicles:
                            raise AssertionError(f"incomplete flow: {result.flow}/{scale.vehicles}")
                        row = {
                            "case_id": case_id, "case_complete": False,
                            "scenario": spec["scenario"], "scenario_title": spec["scenario_title"],
                            "scale_index": spec["scale_index"], "scale_label": scale.label,
                            "repeat": repeat, "seed": seed, "execution_index": execution_index,
                            "graph_group_index": group_index, "case_distribution": args.case_distribution,
                            "method": method, "method_label": labels[method], **asdict(scale),
                            "aev_vehicles": scale.vehicles - int(round(scale.vehicles * spec["generator"]["ev_ratio"])),
                            "aev_ratio": 1.0 - spec["generator"]["ev_ratio"],
                            **{key: spec["generator"][key] for key in ("request_density", "charge_density", "reloc_density", "fixed_charge_capacity")},
                            "feasible_edges": case.feasible_edges, **input_counts, **retained,
                            "graph_nodes": problem.num_nodes, "graph_edges": problem.reduced_edges,
                            "original_graph_edges": problem.original_edges,
                            "edge_reduction_ratio": reduction_ratio, "reduction_rounds": problem.reduction_rounds,
                            "generation_seconds": case.generation_seconds,
                            "graph_build_seconds": graph_build_seconds, "solve_seconds": solve_seconds,
                            "end_to_end_seconds": graph_build_seconds + solve_seconds,
                            "time_ms": 1000.0 * (graph_build_seconds + solve_seconds),
                            "reward": result.objective_q, "objective_int": result.objective_int,
                            "objective_q": result.objective_q, "flow": result.flow,
                            "status": result.status, "backend": result.backend,
                            "mcmf_backend": args.mcmf_backend, "warmed_up": args.warmup,
                            "cost_scale": args.cost_scale, "uses_primal_dual": False,
                            "objective_gap_int": None, "objective_match": None,
                            "input_file": input_file, "input_sha256": input_hash,
                        }
                        execution_index += 1
                        case_rows.append(row)
                        rows.append(row)
                        # Preserve finished methods even if a later native solver
                        # terminates the process. The canonical CSV is checkpointed
                        # after the cross-method objective comparison.
                        with (output_dir / "method_events.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row) + "\n")
                        print(f"  {labels[method]:26s} graph={problem.reduced_edges:9d} build={graph_build_seconds:.4f}s solve={solve_seconds:.4f}s objective={result.objective_q:.4f}", flush=True)
                    del problem
                _check_objectives(case_rows, allow_mismatch=args.allow_objective_mismatch)
                for row in case_rows:
                    row["case_complete"] = True
                # Stable output order; execution_index preserves actual timed order.
                rows[-len(case_rows):] = sorted(case_rows, key=lambda row: METHODS.index(row["method"]))
                checkpoint()
                del case, case_rows
                gc.collect()
        metadata["run_status"] = "completed"
        checkpoint()
    except BaseException as exc:
        metadata["run_status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        checkpoint()
        raise

    if not args.no_plot:
        summary_rows = summarize(rows)
        for scenario in args.scenarios:
            save_plot([row for row in summary_rows if row["scenario"] == scenario],
                      output_dir / f"{scenario}_{args.plot_metric}.png", args.plot_metric)
    print(f"Saved inputs, raw results, summary, paired speedups and metadata: {output_dir}", flush=True)
    return output_dir, rows


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run_benchmark(args)
    except OutputDirectoryError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
