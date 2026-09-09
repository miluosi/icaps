"""Check accelerated EAGR against the original monotone reduction and precision policy."""

import numpy as np
import pytest

from src import exact_mcmf as mcmf
from src.qvalue_precision import quantize_qvalues


@pytest.mark.parametrize("seed", range(50))
def test_batched_eagr_matches_incremental_witnesses_and_fixed_point(seed):
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(1, 20)), int(rng.integers(1, 25))
    feasible = rng.random((n, m)) < 0.6
    capacities = rng.integers(0, n + 1, size=m)
    feasible &= capacities > 0
    capacities = np.minimum(capacities, feasible.sum(axis=0))
    values = rng.integers(-8, 9, size=(n, m)).astype(float)
    baseline = rng.integers(-8, 5, size=n).astype(float)
    baseline[rng.random(n) < 0.5] = -np.inf
    rows, actions = np.nonzero(feasible)
    actual = mcmf._reduce_eagr(
        feasible, values, capacities, baseline, rows, actions, values[rows, actions],
    )
    expected = mcmf._reduce_eagr_incremental(feasible, values, capacities, baseline)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_deep_cascade_finishes_exactly_after_batched_work_budget(monkeypatch):
    n = 9
    feasible = np.eye(n, dtype=bool)
    values = np.diag(np.arange(1, n + 1, dtype=float))
    for action in range(1, n):
        feasible[action - 1, action] = True
        values[action - 1, action] = action - 0.5
    original = mcmf._reduce_eagr_incremental
    calls = []

    def record(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(mcmf, "_reduce_eagr_incremental", record)
    problem = mcmf.build_reduced_problem(feasible, values, np.ones(n, dtype=int))
    assert len(calls) == 1
    assert problem.reduction_rounds == n
    assert not problem.shared_actions.size
    assert problem.baseline_action.tolist() == list(range(n))
    result = mcmf.solve_primal_dual(problem, verify=True)
    assert result.objective_int == 10_000 * n * (n + 1) // 2


def test_block_quantization_matches_full_policy_and_owns_output():
    rng = np.random.default_rng(719)
    # Multiple blocks; non-contiguous input, signed zero, and half-grid cases.
    original = rng.normal(size=(301, 2200))[:, ::2]
    original[0, :5] = [0.0, -0.0, 1.00005, -1.00005, 2.00015]
    before = original.copy()
    expected_q, expected_int = quantize_qvalues(original, 10_000)
    q, integers, rounded, max_delta = mcmf._quantize_problem_values(original, 10_000)
    np.testing.assert_array_equal(q, expected_q)
    np.testing.assert_array_equal(integers, expected_int)
    delta = np.abs(before - expected_q)
    assert rounded == np.count_nonzero(delta)
    assert max_delta == np.max(delta)
    assert not np.shares_memory(q, original)
    original[:] = 77.0
    np.testing.assert_array_equal(q, expected_q)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_block_quantization_validates_infeasible_entries_in_later_blocks(invalid):
    values = np.zeros((301, 1100))
    values[-1, -1] = invalid
    with pytest.raises(ValueError, match="NaN or infinity"):
        mcmf.build_reduced_problem(
            np.zeros_like(values, dtype=bool), values, np.ones(1100, dtype=int),
            fallback_values=np.zeros(301),
        )


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
def test_empty_graphs_and_explicit_outside_actions(shape):
    n, m = shape
    problem = mcmf.build_reduced_problem(
        np.zeros(shape, dtype=bool), np.zeros(shape), np.ones(m, dtype=int),
        fallback_values=np.arange(n, dtype=float),
    )
    result = mcmf.solve_primal_dual(problem, verify=True)
    assert result.objective_int == 10_000 * sum(range(n))
    assert result.flow == n


def test_compact_metadata_preserves_indexing_and_slice_contract():
    problem = mcmf.build_reduced_problem(
        np.ones((2, 2), dtype=bool), np.array([[3.0, 2.0], [2.0, 4.0]]),
        np.ones(2, dtype=int), fallback_values=np.zeros(2), graph_reduction=False,
    )
    metadata = list(problem.arc_meta)
    assert len(metadata) == problem.reduced_edges
    assert problem.arc_meta[-1] == metadata[-1]
    assert problem.arc_meta[1::2] == metadata[1::2]
    assert {meta.kind for meta in metadata} == {"source", "baseline", "shared", "sink"}
    # A negative scaled value at the int64 boundary must not evade overflow
    # checking because abs(int64_min) itself overflows.
    with pytest.raises(OverflowError):
        mcmf.build_reduced_problem(
            np.ones((1, 1), dtype=bool), np.array([[-float(2**63)]]),
            np.ones(1, dtype=int), cost_scale=1,
        )
