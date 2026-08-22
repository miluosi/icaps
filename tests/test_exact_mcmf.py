from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.exact_mcmf import (
    build_reduced_problem,
    solve_gurobi_network,
    solve_primal_dual,
)
from src.qvalue_precision import (
    decimal_places_for_scale,
    qvalue_rounding_diagnostics,
    round_qvalue_matrix,
)


def test_shared_qvalue_precision_uses_four_decimal_grid():
    original = np.asarray(
        [[1.23456, -2.34566, 0.00004, -0.0]],
        dtype=np.float32,
    )
    rounded = round_qvalue_matrix(original, 10_000)
    diagnostics = qvalue_rounding_diagnostics(original, rounded, 10_000)

    assert rounded.dtype == np.float64
    assert rounded.tolist() == [[1.2346, -2.3457, 0.0, 0.0]]
    assert np.array_equal(
        np.rint(rounded * 10_000).astype(np.int64),
        np.array([[12346, -23457, 0, 0]], dtype=np.int64),
    )
    assert decimal_places_for_scale(10_000) == 4
    assert diagnostics["qvalue_decimal_places"] == 4
    assert diagnostics["qvalue_rounding_max_abs"] <= 0.5 / 10_000


def _brute_force(feasible, q_int, capacities, fallback_int):
    choices = []
    for row in range(feasible.shape[0]):
        choices.append([-1, *np.flatnonzero(feasible[row]).tolist()])
    best = None
    for actions in itertools.product(*choices):
        loads = np.zeros(feasible.shape[1], dtype=np.int64)
        objective = 0
        valid = True
        for row, action in enumerate(actions):
            if action == -1:
                objective += int(fallback_int[row])
            else:
                loads[action] += 1
                if loads[action] > capacities[action]:
                    valid = False
                    break
                objective += int(q_int[row, action])
        if valid and (best is None or objective > best):
            best = objective
    return best


def _brute_force_raw(feasible, q_values, capacities, fallback_values):
    choices = []
    for row in range(feasible.shape[0]):
        choices.append([-1, *np.flatnonzero(feasible[row]).tolist()])
    best = -np.inf
    for actions in itertools.product(*choices):
        loads = np.zeros(feasible.shape[1], dtype=np.int64)
        values = []
        valid = True
        for row, action in enumerate(actions):
            if action == -1:
                values.append(float(fallback_values[row]))
            else:
                loads[action] += 1
                if loads[action] > capacities[action]:
                    valid = False
                    break
                values.append(float(q_values[row, action]))
        if valid:
            best = max(best, sum(values))
    return float(best)


def _brute_force_required(feasible, q_int, capacities):
    choices = [np.flatnonzero(feasible[row]).tolist() for row in range(len(feasible))]
    best = None
    for actions in itertools.product(*choices):
        loads = np.zeros(feasible.shape[1], dtype=np.int64)
        objective = 0
        valid = True
        for row, action in enumerate(actions):
            loads[action] += 1
            if loads[action] > capacities[action]:
                valid = False
                break
            objective += int(q_int[row, action])
        if valid and (best is None or objective > best):
            best = objective
    return best


@pytest.mark.parametrize("seed", range(40))
def test_reduced_and_full_graph_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    vehicles = int(rng.integers(1, 6))
    actions = int(rng.integers(1, 7))
    feasible = rng.random((vehicles, actions)) < 0.55
    q_values = rng.integers(-20, 31, size=(vehicles, actions)) / 10.0
    capacities = rng.integers(0, vehicles + 1, size=actions, dtype=np.int64)
    fallback = rng.integers(-30, 1, size=vehicles) / 10.0
    scale = 10

    effective_feasible = feasible & (capacities[np.newaxis, :] > 0)
    effective_capacities = np.minimum(
        capacities,
        effective_feasible.sum(axis=0, dtype=np.int64),
    )
    q_int = np.rint(q_values * scale).astype(np.int64)
    fallback_int = np.rint(fallback * scale).astype(np.int64)
    expected = _brute_force(
        effective_feasible, q_int, effective_capacities, fallback_int
    )

    reduced = build_reduced_problem(
        feasible,
        q_values,
        capacities,
        cost_scale=scale,
        fallback_values=fallback,
        graph_reduction=True,
    )
    full = build_reduced_problem(
        feasible,
        q_values,
        capacities,
        cost_scale=scale,
        fallback_values=fallback,
        graph_reduction=False,
    )
    reduced_result = solve_primal_dual(reduced, verify=True)
    full_result = solve_primal_dual(full, verify=True)

    assert reduced_result.flow == vehicles
    assert reduced_result.objective_int == expected
    assert full_result.objective_int == expected
    assert reduced.reduced_edges <= full.reduced_edges


def test_zero_capacity_action_is_never_used_as_private_baseline():
    problem = build_reduced_problem(
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[100.0, 2.0], [100.0, 3.0]]),
        np.array([0, 2]),
        fallback_values=np.array([-10.0, -10.0]),
    )
    result = solve_primal_dual(problem, verify=True)
    assert result.action_by_vehicle == {0: 1, 1: 1}


def test_explicit_fallback_makes_full_flow_feasible():
    problem = build_reduced_problem(
        np.zeros((3, 2), dtype=bool),
        np.zeros((3, 2), dtype=float),
        np.array([1, 1]),
        fallback_values=np.array([-1.0, -2.0, -3.0]),
        cost_scale=100,
    )
    result = solve_primal_dual(problem, verify=True)
    assert result.flow == 3
    assert result.fallback_used
    assert result.objective_int == -600


def test_strict_eagr_witnesses_remain_capacity_feasible():
    feasible = np.array(
        [
            [0, 0, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    q_values = np.array(
        [
            [-10, -7.6, 2.8, 9.3, 3.2],
            [8.7, -4.3, -7, -6.6, 3.3],
            [8.2, -4.8, -4.7, 0.7, 3.4],
            [2.5, 6.5, 4.3, -5.8, 4],
            [7.5, -1.2, 6.5, 5.7, -0.6],
            [8, 1.2, 3.2, 2.6, -8.6],
        ]
    )
    problem = build_reduced_problem(
        feasible,
        q_values,
        np.array([4, 1, 1, 2, 6]),
        fallback_values=np.array([-2.7, -2.4, -4.7, -6.4, -9.6, -4.5]),
        cost_scale=10,
    )
    result = solve_primal_dual(problem, verify=True)
    loads = np.bincount(
        [action for action in result.action_by_vehicle.values() if action >= 0],
        minlength=5,
    )
    assert np.all(loads <= problem.action_capacities)


def test_eagr_folds_zero_gain_ties_without_replacing_strict_witnesses():
    # Action 0 is the nonbinding outside action.  Action 1 has capacity one
    # and ties both baselines.  TeX EAGR counts only strict improvements, so
    # action 1 folds in round two; strict witness updates keep both vehicles
    # on action 0 and therefore preserve capacity feasibility.
    problem = build_reduced_problem(
        np.ones((2, 2), dtype=bool),
        np.zeros((2, 2), dtype=float),
        np.array([2, 1]),
        fallback_values=None,
    )
    result = solve_primal_dual(problem, verify=True)

    assert problem.reduction_rounds == 2
    assert problem.shared_actions.size == 0
    assert problem.baseline_action.tolist() == [0, 0]
    assert result.action_by_vehicle == {0: 0, 1: 0}
    assert result.objective_int == 0


def test_eagr_keeps_rows_without_outside_actions_in_exact_shared_core():
    problem = build_reduced_problem(
        np.ones((2, 2), dtype=bool),
        np.array([[2.0, -5.0], [-3.0, 1.0]]),
        np.array([1, 1]),
        cost_scale=10,
        fallback_values=None,
    )
    result = solve_primal_dual(problem, verify=True)

    assert problem.baseline_available.tolist() == [False, False]
    assert not any(meta.kind == "baseline" for meta in problem.arc_meta)
    assert result.action_by_vehicle == {0: 0, 1: 1}
    assert result.objective_int == 30
    assert not result.fallback_used


@pytest.mark.parametrize("seed", range(20))
def test_forced_shared_core_matches_required_assignment_brute_force(seed):
    rng = np.random.default_rng(20_000 + seed)
    vehicles = int(rng.integers(2, 6))
    actions = int(rng.integers(2, 6))
    feasible = np.ones((vehicles, actions), dtype=bool)
    capacities = rng.integers(1, vehicles, size=actions, dtype=np.int64)
    while capacities.sum() < vehicles:
        index = int(rng.integers(0, actions))
        if capacities[index] < vehicles - 1:
            capacities[index] += 1
    q_int = rng.integers(-30, 31, size=(vehicles, actions), dtype=np.int64)
    expected = _brute_force_required(feasible, q_int, capacities)

    problem = build_reduced_problem(
        feasible,
        q_int.astype(float),
        capacities,
        cost_scale=1,
        fallback_values=None,
    )
    result = solve_primal_dual(problem, verify=True)

    assert result.objective_int == expected
    assert all(action >= 0 for action in result.action_by_vehicle.values())
    assert not result.fallback_used


def test_eagr_mixes_real_baselines_with_forced_shared_core_exactly():
    feasible = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=bool,
    )
    q_values = np.array(
        [
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 9.0],
            [0.0, 8.0, 7.0],
        ]
    )
    problem = build_reduced_problem(
        feasible,
        q_values,
        np.array([2, 1, 1]),
        cost_scale=1,
        fallback_values=None,
    )
    result = solve_primal_dual(problem, verify=True)

    assert problem.baseline_available.tolist() == [True, True, False]
    assert result.objective_int == 17
    assert result.action_by_vehicle[2] in {1, 2}
    assert all(action >= 0 for action in result.action_by_vehicle.values())
    assert not result.fallback_used


def test_gurobi_network_solves_forced_shared_core_to_proven_optimum():
    gp = pytest.importorskip("gurobipy")
    from gurobipy import GRB

    problem = build_reduced_problem(
        np.ones((2, 2), dtype=bool),
        np.array([[2.0, -5.0], [-3.0, 1.0]]),
        np.array([1, 1]),
        cost_scale=10,
        fallback_values=None,
    )
    result = solve_gurobi_network(problem, gp=gp, grb=GRB, num_threads=1)

    assert result.status == "OPTIMAL"
    assert result.optimal
    assert result.action_by_vehicle == {0: 0, 1: 1}
    assert result.objective_int == 30
    assert not result.fallback_used


def test_gurobi_eagr_optimizes_the_shared_four_decimal_q_grid():
    gp = pytest.importorskip("gurobipy")
    from gurobipy import GRB

    # The input values straddle the four-decimal rounding boundary.  Every
    # exact backend must optimize the same canonical integer grid.
    problem = build_reduced_problem(
        np.ones((2, 2), dtype=bool),
        np.array(
            [
                [1.00006, 1.0],
                [1.00004, 1.0],
            ]
        ),
        np.array([1, 2]),
        cost_scale=10_000,
        fallback_values=None,
    )
    result = solve_gurobi_network(problem, gp=gp, grb=GRB, num_threads=1)

    assert result.status == "OPTIMAL"
    assert result.optimal
    assert result.objective_mode == "rounded_q"
    assert result.action_by_vehicle == {0: 0, 1: 1}
    assert result.objective_q == pytest.approx(2.0001, abs=1e-12)
    assert result.objective_int == 20_001
    assert result.qvalue_scale == 10_000
    assert result.qvalue_rounded_entries == 2
    assert result.qvalue_rounding_max_abs <= 0.5 / 10_000


@pytest.mark.parametrize("seed", range(20))
def test_rounded_q_eagr_matches_full_brute_force(seed):
    gp = pytest.importorskip("gurobipy")
    from gurobipy import GRB

    rng = np.random.default_rng(10_000 + seed)
    vehicles = int(rng.integers(1, 5))
    actions = int(rng.integers(1, 6))
    feasible = rng.random((vehicles, actions)) < 0.6
    capacities = rng.integers(0, vehicles + 1, size=actions, dtype=np.int64)
    q_values = rng.normal(0.0, 2.0, size=(vehicles, actions))
    fallback = rng.normal(-1.0, 0.5, size=vehicles)
    effective_feasible = feasible & (capacities[np.newaxis, :] > 0)
    effective_capacities = np.minimum(
        capacities,
        effective_feasible.sum(axis=0, dtype=np.int64),
    )
    problem = build_reduced_problem(
        feasible,
        q_values,
        capacities,
        cost_scale=10,
        fallback_values=fallback,
    )
    full_problem = build_reduced_problem(
        feasible,
        q_values,
        capacities,
        cost_scale=10,
        fallback_values=fallback,
        graph_reduction=False,
    )
    expected = _brute_force_raw(
        effective_feasible,
        problem.q_values_raw,
        effective_capacities,
        problem.fallback_value_raw,
    )
    result = solve_gurobi_network(problem, gp=gp, grb=GRB, num_threads=1)
    full_result = solve_gurobi_network(
        full_problem,
        gp=gp,
        grb=GRB,
        num_threads=1,
    )

    assert result.status == "OPTIMAL"
    assert result.optimal
    assert result.backend == "gurobi_network"
    assert full_result.status == "OPTIMAL"
    assert full_result.optimal
    assert full_result.backend == "gurobi_network"
    assert result.objective_mode == "rounded_q"
    assert result.objective_q == pytest.approx(expected, abs=1e-8)
    assert full_result.objective_q == pytest.approx(expected, abs=1e-8)
    assert problem.reduced_edges <= full_problem.reduced_edges


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_q_values_are_rejected(bad_value):
    with pytest.raises(ValueError, match="NaN or infinity"):
        build_reduced_problem(
            np.ones((1, 1), dtype=bool),
            np.array([[bad_value]]),
            np.array([1]),
            fallback_values=np.array([0.0]),
        )
