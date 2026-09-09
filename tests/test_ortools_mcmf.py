"""Exactness checks for the optional single-threaded OR-Tools flow backend."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from src.exact_mcmf import build_reduced_problem, solve_exact


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("with_fallback", [False, True])
def test_ortools_full_and_ssg_match_enumeration(seed, with_fallback):
    pytest.importorskip("ortools.graph.python.min_cost_flow")
    rng = np.random.default_rng(seed)
    n, m = 4, 5
    feasible = rng.random((n, m)) < 0.65
    # Give required rows a real feasible action while retaining shared conflicts.
    feasible[:, -1] = True
    capacities = rng.integers(0, n + 1, size=m)
    capacities[-1] = n
    values = rng.integers(-30, 31, size=(n, m)) / 10.0
    fallback = rng.integers(-20, 1, size=n) / 10.0 if with_fallback else None
    choices = [
        ([-1] if with_fallback else [])
        + np.flatnonzero(feasible[i] & (capacities > 0)).tolist()
        for i in range(n)
    ]
    best = None
    for assignment in itertools.product(*choices):
        load = np.bincount([a for a in assignment if a >= 0], minlength=m)
        if np.any(load > capacities):
            continue
        objective = sum(
            int(round((fallback[i] if a == -1 else values[i, a]) * 10))
            for i, a in enumerate(assignment)
        )
        best = objective if best is None else max(best, objective)
    for reduced in (False, True):
        problem = build_reduced_problem(
            feasible, values, capacities, fallback_values=fallback,
            cost_scale=10, graph_reduction=reduced,
        )
        result = solve_exact(problem, backend="ortools")
        assert result.objective_int == best
        assert result.flow == n
        assert result.optimal and not result.solver_fallback_used


def test_ortools_rejects_infeasible_required_assignment():
    pytest.importorskip("ortools.graph.python.min_cost_flow")
    problem = build_reduced_problem(
        np.ones((3, 1), dtype=bool), np.ones((3, 1)), np.array([1]),
    )
    with pytest.raises(RuntimeError, match="not OPTIMAL"):
        solve_exact(problem, backend="ortools")


def test_ortools_keeps_forced_rows_and_negative_values_exact():
    pytest.importorskip("ortools.graph.python.min_cost_flow")
    for reduced in (False, True):
        problem = build_reduced_problem(
            np.ones((2, 2), dtype=bool),
            np.array([[-2.0, -5.0], [-3.0, -1.0]]), np.array([1, 1]),
            graph_reduction=reduced,
        )
        assert not problem.baseline_available.any()
        result = solve_exact(problem, backend="ortools")
        assert result.objective_int == -30_000
        assert result.action_by_vehicle == {0: 0, 1: 1}
        assert not result.fallback_used


def test_production_assignment_routes_to_ortools_with_exact_ssg():
    pytest.importorskip("ortools.graph.python.min_cost_flow")
    from src.GurobiOptimizer import GurobiOptimizer

    requests = [SimpleNamespace(request_id=100), SimpleNamespace(request_id=101)]
    env = SimpleNamespace(
        mip_backend="docplex", mcmf_solver="exact", mcmf_backend="ortools",
        mcmf_graph_reduction=True, mcmf_strict=True,
        vehicles={10: {"type": 1}, 20: {"type": 2}},
        num_zones=0, hotspot_locations=[],
        charging_manager=SimpleNamespace(stations={}), station_queue_capacity=0,
        reserve_inbound_charging_capacity=False, record_time=False,
        synthetic_demand_profile="predictive",
    )
    optimizer = GurobiOptimizer(env, num_threads=1)
    assignments = optimizer._np_vehicle_rebalancing_network(
        [10, 20], requests, np.ones((2, 3), dtype=np.int8),
        np.array([[5.00006, 1.0, 0.0], [4.0, 6.0, 1.0]]),
    )
    assert assignments == {10: requests[0], 20: requests[1]}
    assert env.mcmf_last_result["backend"] == "ortools"
    assert env.mcmf_last_result["objective_int"] == 110_001
    assert not env.mcmf_last_result["solver_fallback_used"]
