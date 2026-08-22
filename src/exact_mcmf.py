"""Exact action-graph reduction and one-period assignment solvers.

EAGR first canonicalizes every finite Q-value onto the configured integer
grid (10,000 by default, i.e. four decimal places).  Its strict dominance,
nonbinding-capacity certificates, and every exact backend then optimize that
same precision-controlled objective.

The builder applies only objective-preserving reductions:

* capacity-nonbinding actions are collapsed into a per-vehicle baseline;
* shared actions that do not improve that baseline are removed; and
* actions that become nonbinding after pruning are collapsed to a fixed point.

If a vehicle has no capacity-nonbinding outside action, EAGR leaves that row
in the shared core instead of inventing a fallback.  Such a row must receive a
real capacitated action, so feasibility and the rounded-Q optimum are preserved.

OR-Tools and Gurobi are optional exact backends.  The bundled integer
primal-dual solver is always available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from typing import Optional

import numpy as np

from src.qvalue_precision import quantize_qvalues


@dataclass(frozen=True)
class ArcMeta:
    vehicle_index: Optional[int] = None
    action_index: Optional[int] = None
    kind: str = "internal"


@dataclass
class ReducedMCMFProblem:
    num_nodes: int
    source: int
    sink: int
    target_flow: int
    tails: np.ndarray
    heads: np.ndarray
    capacities: np.ndarray
    costs: np.ndarray
    raw_costs: np.ndarray
    arc_meta: list[ArcMeta]
    baseline_action: np.ndarray
    baseline_available: np.ndarray
    baseline_value: np.ndarray
    baseline_value_raw: np.ndarray
    shared_actions: np.ndarray
    q_values_int: np.ndarray
    q_values_raw: np.ndarray
    feasibility: np.ndarray
    action_capacities: np.ndarray
    fallback_value: Optional[np.ndarray]
    fallback_value_raw: Optional[np.ndarray]
    cost_scale: int
    qvalue_entries: int
    qvalue_rounded_entries: int
    qvalue_rounding_max_abs: float
    original_edges: int
    reduction_rounds: int

    @property
    def baseline_sum(self) -> int:
        return sum(int(value) for value in self.baseline_value)

    @property
    def baseline_sum_raw(self) -> float:
        return math.fsum(float(value) for value in self.baseline_value_raw)

    @property
    def reduced_edges(self) -> int:
        return int(len(self.tails))


@dataclass(frozen=True)
class ExactMCMFResult:
    action_by_vehicle: dict[int, int]
    objective_int: int
    objective_q: float
    objective_mode: str
    flow: int
    backend: str
    status: str
    optimal: bool
    fallback_used: bool
    solver_fallback_used: bool
    original_edges: int
    reduced_edges: int
    reduction_rounds: int
    qvalue_scale: int
    qvalue_entries: int
    qvalue_rounded_entries: int
    qvalue_rounding_max_abs: float

    @property
    def edge_reduction_ratio(self) -> float:
        if self.original_edges <= 0:
            return 0.0
        return 1.0 - self.reduced_edges / self.original_edges


def quantize_values(values: np.ndarray, scale: int) -> np.ndarray:
    """Compatibility wrapper returning the shared scaled integer grid."""

    _, scaled = quantize_qvalues(values, scale)
    return scaled


def build_reduced_problem(
    feasibility: np.ndarray,
    q_values: np.ndarray,
    action_capacities: np.ndarray,
    *,
    cost_scale: int = 10_000,
    fallback_values: Optional[np.ndarray] = None,
    preserve_zero_gain_ties: bool = False,
    graph_reduction: bool = True,
) -> ReducedMCMFProblem:
    """Build the exact EAGR core on the precision-controlled Q grid.

    Rows satisfying the manuscript's outside-action assumption use the normal
    baseline reduction.  A row without such an action remains forced into the
    shared capacitated core; no synthetic wait/fallback action is introduced.
    """

    feasible = np.asarray(feasibility, dtype=bool)
    q_input = np.asarray(q_values, dtype=np.float64)
    capacities = np.asarray(action_capacities, dtype=np.int64).reshape(-1)

    if feasible.ndim != 2 or q_input.ndim != 2:
        raise ValueError("feasibility and q_values must be rank-2 arrays")
    if feasible.shape != q_input.shape:
        raise ValueError("feasibility and q_values shapes must match")
    n_vehicles, n_actions = feasible.shape
    if capacities.shape != (n_actions,):
        raise ValueError("action_capacities length must match action count")
    if np.any(capacities < 0):
        raise ValueError("action capacities must be nonnegative")

    # A zero-capacity action is never a private/unlimited option.
    feasible = feasible & (capacities[np.newaxis, :] > 0)
    feasible_degree = feasible.sum(axis=0, dtype=np.int64)
    capacities = np.minimum(capacities, feasible_degree)
    q, q_int = quantize_qvalues(q_input, cost_scale)
    q_rounding_delta = np.abs(q_input - q)
    qvalue_rounded_entries = int(np.count_nonzero(q_rounding_delta))
    qvalue_rounding_max_abs = float(np.max(q_rounding_delta, initial=0.0))

    fallback_int = None
    if fallback_values is not None:
        fallback_input = np.asarray(fallback_values, dtype=np.float64).reshape(-1)
        if fallback_input.shape != (n_vehicles,):
            raise ValueError("fallback_values must have one value per vehicle")
        fallback_values, fallback_int = quantize_qvalues(
            fallback_input,
            cost_scale,
        )
        fallback_delta = np.abs(fallback_input - fallback_values)
        qvalue_rounded_entries += int(np.count_nonzero(fallback_delta))
        qvalue_rounding_max_abs = max(
            qvalue_rounding_max_abs,
            float(np.max(fallback_delta, initial=0.0)),
        )

    candidate_abs = [abs(int(value)) for value in q_int[feasible]]
    if fallback_int is not None:
        candidate_abs.extend(abs(int(value)) for value in fallback_int)
    max_abs_value = max(candidate_abs, default=0)
    # Keep both accumulated int64 objectives and Gurobi's double-precision
    # integer coefficients exact.  A gain is the difference of two Q values,
    # hence the conservative 2**51 per-value limit.
    safe_limit = min(
        np.iinfo(np.int64).max // max(4 * max(n_vehicles, 1), 4),
        2**51,
    )
    if max_abs_value > safe_limit:
        raise OverflowError(
            "scaled Q-values are too large for exact integer objective arithmetic"
        )

    active_actions = (capacities > 0) & (feasible_degree > 0)
    # Count the unreduced canonical graph.  An explicit fallback value is a
    # vehicle-specific direct-to-sink action and therefore contributes one
    # additional arc per vehicle only when it is actually configured.
    original_edges = int(
        n_vehicles
        + feasible.sum(dtype=np.int64)
        + active_actions.sum()
        + (n_vehicles if fallback_values is not None else 0)
    )

    baseline_action = np.full(n_vehicles, -1, dtype=np.int64)
    baseline_value_raw = (
        np.asarray(fallback_values, dtype=np.float64).copy()
        if fallback_values is not None
        else np.full(n_vehicles, -np.inf, dtype=np.float64)
    )
    reduction_rounds = 0

    if graph_reduction:
        # Exact action-graph reduction (EAGR).  Baselines and their witness
        # actions are monotone: a witness changes only under a strict
        # precision-controlled Q
        # improvement.  This is the strict fixed-point algorithm stated in
        # the manuscript, not a top-K or proximity approximation.
        shared = active_actions.copy()
        incident_vehicles: list[list[int]] = [[] for _ in range(n_actions)]
        row_actions: list[np.ndarray] = []
        row_values: list[np.ndarray] = []
        row_threshold = np.zeros(n_vehicles, dtype=np.int64)
        positive_degree = np.zeros(n_actions, dtype=np.int64)

        for vehicle in range(n_vehicles):
            actions = np.flatnonzero(feasible[vehicle] & shared)
            if actions.size:
                # Values are primary and action ids are the deterministic
                # tie-break.  The monotone pointer then deletes every
                # dominated edge at most once as the baseline rises.
                order = np.lexsort((actions, q[vehicle, actions]))
                actions = actions[order].astype(np.int64, copy=False)
                values = q[vehicle, actions]
                pointer = int(np.searchsorted(
                    values,
                    baseline_value_raw[vehicle],
                    side="right",
                ))
                row_threshold[vehicle] = pointer
                for action in actions:
                    incident_vehicles[int(action)].append(vehicle)
                for action in actions[pointer:]:
                    positive_degree[int(action)] += 1
            else:
                values = np.empty(0, dtype=np.float64)
            row_actions.append(actions)
            row_values.append(values)

        foldable = np.flatnonzero(
            shared & (positive_degree <= capacities)
        ).astype(np.int64)
        while foldable.size:
            reduction_rounds += 1
            # All actions in the current EAGR batch are folded together.
            # Marking them first reproduces K <- K \ F exactly.
            shared[foldable] = False
            updated_vehicles: set[int] = set()
            for action_raw in foldable:
                action = int(action_raw)
                for vehicle in incident_vehicles[action]:
                    value = float(q[vehicle, action])
                    if value > float(baseline_value_raw[vehicle]):
                        baseline_value_raw[vehicle] = value
                        baseline_action[vehicle] = action
                        updated_vehicles.add(vehicle)

            next_foldable: set[int] = set()
            for vehicle in updated_vehicles:
                actions = row_actions[vehicle]
                values = row_values[vehicle]
                pointer = int(row_threshold[vehicle])
                baseline = float(baseline_value_raw[vehicle])
                while pointer < len(actions) and float(values[pointer]) <= baseline:
                    action = int(actions[pointer])
                    if shared[action]:
                        positive_degree[action] -= 1
                        if positive_degree[action] <= capacities[action]:
                            next_foldable.add(action)
                    pointer += 1
                row_threshold[vehicle] = pointer
            foldable = np.asarray(sorted(next_foldable), dtype=np.int64)

    else:
        if fallback_int is None or fallback_values is None:
            raise ValueError("unreduced exact graph requires explicit fallback values")
        shared = active_actions.copy()
        baseline_action.fill(-1)
        baseline_value_raw[:] = np.asarray(fallback_values, dtype=np.float64)
        row_actions = [
            np.flatnonzero(feasible[vehicle] & shared).astype(
                np.int64,
                copy=False,
            )
            for vehicle in range(n_vehicles)
        ]

    # The TeX EAGR assumption gives every row a real capacity-nonbinding
    # baseline.  In NYC, an AEV row can lack one when charger availability
    # disallows wait while insufficient battery independently disallows
    # relocation.  Keep such rows in the exact shared core: a zero reference
    # is only an objective offset, not an available action.
    baseline_available = np.isfinite(baseline_value_raw)
    baseline_value_raw[~baseline_available] = 0.0
    baseline_value = np.empty(n_vehicles, dtype=np.int64)
    for vehicle, action_raw in enumerate(baseline_action):
        action = int(action_raw)
        if action >= 0:
            baseline_value[vehicle] = int(q_int[vehicle, action])
        elif fallback_int is not None:
            baseline_value[vehicle] = int(fallback_int[vehicle])
        elif not baseline_available[vehicle]:
            baseline_value[vehicle] = 0
        else:
            raise AssertionError("real EAGR witness is missing after reduction")

    shared_actions = np.flatnonzero(shared).astype(np.int64)
    action_to_local = {int(action): j for j, action in enumerate(shared_actions)}

    source = 0
    vehicle_offset = 1
    action_offset = vehicle_offset + n_vehicles
    sink = action_offset + len(shared_actions)

    tails: list[int] = []
    heads: list[int] = []
    arc_caps: list[int] = []
    costs: list[int] = []
    raw_costs: list[float] = []
    metadata: list[ArcMeta] = []

    def add_arc(
        u: int,
        v: int,
        cap: int,
        cost: int,
        raw_cost: float,
        meta: ArcMeta,
    ) -> None:
        tails.append(u)
        heads.append(v)
        arc_caps.append(cap)
        costs.append(cost)
        raw_costs.append(raw_cost)
        metadata.append(meta)

    for i in range(n_vehicles):
        vehicle_node = vehicle_offset + i
        add_arc(source, vehicle_node, 1, 0, 0.0, ArcMeta(i, None, "source"))
        if baseline_available[i]:
            add_arc(
                vehicle_node,
                sink,
                1,
                0,
                0.0,
                ArcMeta(i, int(baseline_action[i]), "baseline"),
            )
        for action_raw in row_actions[i]:
            action = int(action_raw)
            if not shared[action]:
                continue
            raw_gain = float(q[i, action] - baseline_value_raw[i])
            if baseline_available[i] and (
                raw_gain < 0
                or (raw_gain == 0.0 and not preserve_zero_gain_ties)
            ):
                continue
            gain = int(q_int[i, action] - baseline_value[i])
            add_arc(
                vehicle_node,
                action_offset + action_to_local[action],
                1,
                -gain,
                -raw_gain,
                ArcMeta(i, action, "shared"),
            )

    for local, action in enumerate(shared_actions):
        action = int(action)
        add_arc(
            action_offset + local,
            sink,
            int(capacities[action]),
            0,
            0.0,
            ArcMeta(None, action, "sink"),
        )

    return ReducedMCMFProblem(
        num_nodes=sink + 1,
        source=source,
        sink=sink,
        target_flow=n_vehicles,
        tails=np.asarray(tails, dtype=np.int32),
        heads=np.asarray(heads, dtype=np.int32),
        capacities=np.asarray(arc_caps, dtype=np.int64),
        costs=np.asarray(costs, dtype=np.int64),
        raw_costs=np.asarray(raw_costs, dtype=np.float64),
        arc_meta=metadata,
        baseline_action=baseline_action,
        baseline_available=baseline_available,
        baseline_value=baseline_value,
        baseline_value_raw=baseline_value_raw,
        shared_actions=shared_actions,
        q_values_int=q_int,
        q_values_raw=q,
        feasibility=feasible,
        action_capacities=capacities,
        fallback_value=fallback_int,
        fallback_value_raw=(
            np.asarray(fallback_values, dtype=np.float64).copy()
            if fallback_values is not None else None
        ),
        cost_scale=cost_scale,
        qvalue_entries=int(q_input.size),
        qvalue_rounded_entries=qvalue_rounded_entries,
        qvalue_rounding_max_abs=qvalue_rounding_max_abs,
        original_edges=original_edges,
        reduction_rounds=reduction_rounds,
    )


class PrimalDualMinCostFlow:
    """Integer successive-shortest-path solver with Johnson potentials."""

    def __init__(self, num_nodes: int):
        self.num_nodes = int(num_nodes)
        self.graph: list[list[list[int]]] = [[] for _ in range(self.num_nodes)]

    def add_edge(self, u: int, v: int, capacity: int, cost: int) -> tuple[int, int]:
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        forward_index = len(self.graph[u])
        reverse_index = len(self.graph[v])
        self.graph[u].append([v, int(capacity), int(cost), reverse_index])
        self.graph[v].append([u, 0, -int(cost), forward_index])
        return u, forward_index

    def _initial_feasible_potential(self) -> list[int]:
        # Bellman-Ford from a zero-cost super-source connected to every node.
        # This is exact integer arithmetic and runs once before Dijkstra.
        distance = [0] * self.num_nodes
        for _ in range(self.num_nodes):
            changed = False
            for u, edges in enumerate(self.graph):
                distance_u = distance[u]
                for v, capacity, cost, _ in edges:
                    if capacity > 0 and distance[v] > distance_u + cost:
                        distance[v] = distance_u + cost
                        changed = True
            if not changed:
                return distance
        raise ValueError("negative-cost cycle exists in the initial residual graph")

    def solve(
        self,
        source: int,
        sink: int,
        target_flow: int,
        *,
        verify_optimality: bool = False,
    ) -> tuple[int, int]:
        infinity = 10**40
        potential = self._initial_feasible_potential()
        total_flow = 0
        total_cost = 0

        while total_flow < target_flow:
            distance = [infinity] * self.num_nodes
            parent_node = [-1] * self.num_nodes
            parent_edge = [-1] * self.num_nodes
            distance[source] = 0
            heap: list[tuple[int, int]] = [(0, source)]

            while heap:
                current_distance, u = heappop(heap)
                if current_distance != distance[u]:
                    continue
                for edge_index, edge in enumerate(self.graph[u]):
                    v, capacity, cost, _ = edge
                    if capacity <= 0:
                        continue
                    reduced_cost = cost + potential[u] - potential[v]
                    if reduced_cost < 0:
                        raise AssertionError(
                            f"negative reduced cost {reduced_cost} on ({u}, {v})"
                        )
                    candidate = current_distance + reduced_cost
                    if candidate < distance[v]:
                        distance[v] = candidate
                        parent_node[v] = u
                        parent_edge[v] = edge_index
                        heappush(heap, (candidate, v))

            if distance[sink] == infinity:
                break

            for node in range(self.num_nodes):
                if distance[node] < infinity:
                    potential[node] += distance[node]

            pushed = target_flow - total_flow
            node = sink
            while node != source:
                parent = parent_node[node]
                if parent < 0:
                    raise RuntimeError("incomplete augmenting path")
                edge_index = parent_edge[node]
                pushed = min(pushed, self.graph[parent][edge_index][1])
                node = parent

            node = sink
            while node != source:
                parent = parent_node[node]
                edge_index = parent_edge[node]
                reverse_index = self.graph[parent][edge_index][3]
                total_cost += pushed * self.graph[parent][edge_index][2]
                self.graph[parent][edge_index][1] -= pushed
                self.graph[node][reverse_index][1] += pushed
                node = parent
            total_flow += pushed

        if total_flow != target_flow:
            raise ValueError(
                f"infeasible full assignment: sent {total_flow}/{target_flow} units"
            )
        if verify_optimality and not self.has_no_negative_residual_cycle():
            raise AssertionError("negative residual cycle found; flow is not optimal")
        return total_flow, total_cost

    def solve_while_negative(
        self,
        source: int,
        sink: int,
        max_flow: int,
        *,
        verify_optimality: bool = False,
    ) -> tuple[int, int]:
        """Send only strictly improving flow and stop at the exact optimum.

        This is the variable-cardinality form of successive shortest path.  At
        every iteration the current flow is minimum-cost for its cardinality;
        the marginal augmenting-path costs are nondecreasing.  Therefore the
        first nonnegative marginal cost certifies that adding another shared
        assignment cannot improve the objective.
        """

        infinity = 10**40
        potential = self._initial_feasible_potential()
        total_flow = 0
        total_cost = 0

        while total_flow < max_flow:
            distance = [infinity] * self.num_nodes
            parent_node = [-1] * self.num_nodes
            parent_edge = [-1] * self.num_nodes
            distance[source] = 0
            heap: list[tuple[int, int]] = [(0, source)]

            while heap:
                current_distance, u = heappop(heap)
                if current_distance != distance[u]:
                    continue
                for edge_index, edge in enumerate(self.graph[u]):
                    v, capacity, cost, _ = edge
                    if capacity <= 0:
                        continue
                    reduced_cost = cost + potential[u] - potential[v]
                    if reduced_cost < 0:
                        raise AssertionError(
                            f"negative reduced cost {reduced_cost} on ({u}, {v})"
                        )
                    candidate = current_distance + reduced_cost
                    if candidate < distance[v]:
                        distance[v] = candidate
                        parent_node[v] = u
                        parent_edge[v] = edge_index
                        heappush(heap, (candidate, v))

            if distance[sink] == infinity:
                break
            marginal_cost = distance[sink] - potential[source] + potential[sink]
            if marginal_cost >= 0:
                break

            for node in range(self.num_nodes):
                if distance[node] < infinity:
                    potential[node] += distance[node]

            pushed = max_flow - total_flow
            node = sink
            while node != source:
                parent = parent_node[node]
                if parent < 0:
                    raise RuntimeError("incomplete augmenting path")
                edge_index = parent_edge[node]
                pushed = min(pushed, self.graph[parent][edge_index][1])
                node = parent

            node = sink
            while node != source:
                parent = parent_node[node]
                edge_index = parent_edge[node]
                reverse_index = self.graph[parent][edge_index][3]
                total_cost += pushed * self.graph[parent][edge_index][2]
                self.graph[parent][edge_index][1] -= pushed
                self.graph[node][reverse_index][1] += pushed
                node = parent
            total_flow += pushed

        if verify_optimality and not self.has_no_negative_residual_cycle():
            raise AssertionError("negative residual cycle found; flow is not optimal")
        return total_flow, total_cost

    def has_no_negative_residual_cycle(self) -> bool:
        distance = [0] * self.num_nodes
        for _ in range(self.num_nodes):
            changed = False
            for u, edges in enumerate(self.graph):
                distance_u = distance[u]
                for v, capacity, cost, _ in edges:
                    if capacity > 0 and distance[v] > distance_u + cost:
                        distance[v] = distance_u + cost
                        changed = True
            if not changed:
                return True
        return False


def _decode_arc_flows(
    problem: ReducedMCMFProblem,
    arc_flows: list[int],
) -> tuple[dict[int, int], int]:
    action_by_vehicle: dict[int, int] = {}
    min_cost = 0
    for flow, cost, meta in zip(arc_flows, problem.costs, problem.arc_meta):
        flow = int(flow)
        min_cost += flow * int(cost)
        if flow <= 0 or meta.kind not in {"baseline", "shared"}:
            continue
        if meta.vehicle_index is None or meta.action_index is None:
            raise AssertionError("selected assignment arc is missing metadata")
        if meta.vehicle_index in action_by_vehicle:
            raise AssertionError("vehicle received more than one action")
        action_by_vehicle[meta.vehicle_index] = meta.action_index
    if len(action_by_vehicle) != problem.target_flow:
        raise AssertionError("not every vehicle received one decoded action")
    return action_by_vehicle, min_cost


def _verify_assignment(
    problem: ReducedMCMFProblem,
    action_by_vehicle: dict[int, int],
    objective_int: int,
) -> None:
    if set(action_by_vehicle) != set(range(problem.target_flow)):
        raise AssertionError("assignment does not contain every vehicle exactly once")
    loads = np.zeros(len(problem.action_capacities), dtype=np.int64)
    recomputed = 0
    for vehicle, action in action_by_vehicle.items():
        if action == -1:
            if problem.fallback_value is None:
                raise AssertionError("fallback selected without an explicit value")
            recomputed += int(problem.fallback_value[vehicle])
            continue
        if action < 0 or action >= problem.feasibility.shape[1]:
            raise AssertionError(f"invalid action index {action}")
        if not problem.feasibility[vehicle, action]:
            raise AssertionError("selected action is infeasible")
        loads[action] += 1
        recomputed += int(problem.q_values_int[vehicle, action])
    if np.any(loads > problem.action_capacities):
        raise AssertionError("selected assignment violates an action capacity")
    if recomputed != objective_int:
        raise AssertionError(
            f"objective mismatch: decoded={recomputed}, reported={objective_int}"
        )


def _assignment_objective_raw(
    problem: ReducedMCMFProblem,
    action_by_vehicle: dict[int, int],
) -> float:
    values: list[float] = []
    for vehicle in range(problem.target_flow):
        action = int(action_by_vehicle[vehicle])
        if action >= 0:
            values.append(float(problem.q_values_raw[vehicle, action]))
        elif problem.fallback_value_raw is not None:
            values.append(float(problem.fallback_value_raw[vehicle]))
        else:
            raise AssertionError("fallback selected without a raw fallback value")
    return math.fsum(values)


def solve_primal_dual(
    problem: ReducedMCMFProblem,
    *,
    verify: bool = False,
) -> ExactMCMFResult:
    if not np.all(problem.baseline_available):
        # Some rows have no real outside action and therefore cannot be left
        # unassigned to the shared core.  Solve the canonical reduced network
        # as a full-flow integer SSP problem.  This is the same integral
        # min-cost-flow model used by OR-Tools/Gurobi, not a fallback heuristic.
        solver = PrimalDualMinCostFlow(problem.num_nodes)
        handles: list[tuple[int, int, int]] = []
        for tail, head, capacity, cost in zip(
            problem.tails,
            problem.heads,
            problem.capacities,
            problem.costs,
        ):
            u, edge_index = solver.add_edge(
                int(tail),
                int(head),
                int(capacity),
                int(cost),
            )
            handles.append((u, edge_index, int(capacity)))
        flow, min_cost = solver.solve(
            problem.source,
            problem.sink,
            problem.target_flow,
            verify_optimality=verify,
        )
        arc_flows = [
            capacity - int(solver.graph[u][edge_index][1])
            for u, edge_index, capacity in handles
        ]
        action_by_vehicle, decoded_min_cost = _decode_arc_flows(
            problem,
            arc_flows,
        )
        if decoded_min_cost != min_cost:
            raise AssertionError(
                f"decoded full-flow cost {decoded_min_cost} differs from {min_cost}"
            )
        objective_int = problem.baseline_sum - min_cost
        _verify_assignment(problem, action_by_vehicle, objective_int)
        return _make_result(
            problem,
            action_by_vehicle,
            objective_int,
            flow,
            "primal_dual",
            objective_mode="rounded_q",
        )

    # Every vehicle already owns its best capacity-nonbinding baseline action.
    # Solve only the positive-gain capacitated b-matching.  Compared with the
    # canonical full-flow graph this removes n mandatory baseline augmentations
    # while preserving the exact integer objective.
    num_vehicles = problem.target_flow
    num_shared = len(problem.shared_actions)
    source = 0
    action_offset = 1
    vehicle_offset = action_offset + num_shared
    sink = vehicle_offset + num_vehicles
    solver = PrimalDualMinCostFlow(sink + 1)
    action_to_local = {
        int(action): local for local, action in enumerate(problem.shared_actions)
    }

    total_shared_capacity = 0
    for local, action in enumerate(problem.shared_actions):
        capacity = int(problem.action_capacities[int(action)])
        if capacity <= 0:
            continue
        solver.add_edge(source, action_offset + local, capacity, 0)
        total_shared_capacity += capacity
    for vehicle in range(num_vehicles):
        solver.add_edge(vehicle_offset + vehicle, sink, 1, 0)

    shared_handles: list[tuple[int, int, int, int, int]] = []
    for cost, meta in zip(problem.costs, problem.arc_meta):
        if meta.kind != "shared":
            continue
        if meta.vehicle_index is None or meta.action_index is None:
            raise AssertionError("shared assignment arc is missing metadata")
        cost = int(cost)
        if cost >= 0:
            # Zero gain can be left on the baseline without changing the exact
            # rounded-Q objective, and negative gain arcs were removed by the
            # builder.
            continue
        action = int(meta.action_index)
        vehicle = int(meta.vehicle_index)
        u, edge_index = solver.add_edge(
            action_offset + action_to_local[action],
            vehicle_offset + vehicle,
            1,
            cost,
        )
        shared_handles.append((vehicle, action, u, edge_index, cost))

    _improvement_flow, min_cost = solver.solve_while_negative(
        source,
        sink,
        min(num_vehicles, total_shared_capacity),
        verify_optimality=verify,
    )
    action_by_vehicle = {
        vehicle: int(problem.baseline_action[vehicle])
        for vehicle in range(num_vehicles)
    }
    decoded_min_cost = 0
    selected_vehicles: set[int] = set()
    for vehicle, action, u, edge_index, cost in shared_handles:
        selected = 1 - int(solver.graph[u][edge_index][1])
        if selected <= 0:
            continue
        if selected != 1 or vehicle in selected_vehicles:
            raise AssertionError("invalid shared assignment flow")
        selected_vehicles.add(vehicle)
        action_by_vehicle[vehicle] = action
        decoded_min_cost += cost
    if decoded_min_cost != min_cost:
        raise AssertionError(
            f"decoded improvement cost {decoded_min_cost} differs from {min_cost}"
        )
    objective_int = problem.baseline_sum - min_cost
    _verify_assignment(problem, action_by_vehicle, objective_int)
    return _make_result(
        problem,
        action_by_vehicle,
        objective_int,
        problem.target_flow,
        "primal_dual",
        objective_mode="rounded_q",
    )


def solve_ortools(problem: ReducedMCMFProblem) -> ExactMCMFResult:
    from ortools.graph.python import min_cost_flow

    solver = min_cost_flow.SimpleMinCostFlow()
    solver.add_arcs_with_capacity_and_unit_cost(
        problem.tails, problem.heads, problem.capacities, problem.costs
    )
    solver.set_node_supply(problem.source, problem.target_flow)
    solver.set_node_supply(problem.sink, -problem.target_flow)
    status = solver.solve()
    if status != solver.OPTIMAL:
        raise RuntimeError(f"OR-Tools exact MCMF status is {status}, not OPTIMAL")
    arc_flows = [int(solver.flow(index)) for index in range(solver.num_arcs())]
    action_by_vehicle, min_cost = _decode_arc_flows(problem, arc_flows)
    objective_int = problem.baseline_sum - min_cost
    _verify_assignment(problem, action_by_vehicle, objective_int)
    return _make_result(
        problem,
        action_by_vehicle,
        objective_int,
        problem.target_flow,
        "ortools",
        objective_mode="rounded_q",
    )


def solve_gurobi_network(
    problem: ReducedMCMFProblem,
    *,
    gp,
    grb,
    num_threads: int = 1,
) -> ExactMCMFResult:
    """Solve the reduced network LP and accept only a proven optimum."""

    model = gp.Model("exact_reduced_mcmf")
    model.Params.OutputFlag = 0
    model.Params.Threads = max(1, int(num_threads))
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol = 1e-9
    model.Params.NumericFocus = 2
    if hasattr(model.Params, "NetworkAlg"):
        model.Params.NetworkAlg = 1

    variables = []
    for index, (capacity, cost) in enumerate(
        zip(problem.capacities, problem.costs)
    ):
        variables.append(
            model.addVar(
                lb=0.0,
                ub=float(capacity),
                obj=float(cost),
                vtype=grb.CONTINUOUS,
                name=f"arc_{index}",
            )
        )
    model.ModelSense = grb.MINIMIZE
    outgoing: list[list[int]] = [[] for _ in range(problem.num_nodes)]
    incoming: list[list[int]] = [[] for _ in range(problem.num_nodes)]
    for index, (tail, head) in enumerate(zip(problem.tails, problem.heads)):
        outgoing[int(tail)].append(index)
        incoming[int(head)].append(index)
    for node in range(problem.num_nodes):
        supply = (
            problem.target_flow
            if node == problem.source
            else -problem.target_flow if node == problem.sink else 0
        )
        model.addConstr(
            gp.quicksum(variables[index] for index in outgoing[node])
            - gp.quicksum(variables[index] for index in incoming[node])
            == supply,
            name=f"flow_{node}",
        )
    model.optimize()
    if model.Status != grb.OPTIMAL or model.SolCount <= 0:
        raise RuntimeError(
            "Gurobi exact network did not return a proven optimum: "
            f"status={model.Status}, sol_count={model.SolCount}"
        )
    arc_flows = [int(round(variable.X)) for variable in variables]
    action_by_vehicle, integer_min_cost = _decode_arc_flows(problem, arc_flows)
    objective_int = problem.baseline_sum - integer_min_cost
    _verify_assignment(problem, action_by_vehicle, objective_int)
    decoded_q = _assignment_objective_raw(problem, action_by_vehicle)
    objective_q = objective_int / float(problem.cost_scale)
    tolerance = 1e-12 * max(1.0, abs(decoded_q), abs(objective_q))
    if not math.isclose(decoded_q, objective_q, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"rounded Q objective mismatch: decoded={decoded_q}, scaled={objective_q}"
        )
    return _make_result(
        problem,
        action_by_vehicle,
        objective_int,
        problem.target_flow,
        "gurobi_network",
        objective_mode="rounded_q",
    )


def _make_result(
    problem: ReducedMCMFProblem,
    action_by_vehicle: dict[int, int],
    objective_int: int,
    flow: int,
    backend: str,
    *,
    objective_mode: str,
    solver_fallback_used: bool = False,
) -> ExactMCMFResult:
    return ExactMCMFResult(
        action_by_vehicle=action_by_vehicle,
        objective_int=int(objective_int),
        objective_q=int(objective_int) / float(problem.cost_scale),
        objective_mode=str(objective_mode),
        flow=int(flow),
        backend=backend,
        status="OPTIMAL",
        optimal=True,
        fallback_used=any(action == -1 for action in action_by_vehicle.values()),
        solver_fallback_used=solver_fallback_used,
        original_edges=problem.original_edges,
        reduced_edges=problem.reduced_edges,
        reduction_rounds=problem.reduction_rounds,
        qvalue_scale=problem.cost_scale,
        qvalue_entries=problem.qvalue_entries,
        qvalue_rounded_entries=problem.qvalue_rounded_entries,
        qvalue_rounding_max_abs=problem.qvalue_rounding_max_abs,
    )


def solve_exact(
    problem: ReducedMCMFProblem,
    *,
    backend: str = "auto",
    verify: bool = False,
    gp=None,
    grb=None,
    num_threads: int = 1,
) -> ExactMCMFResult:
    """Dispatch to an exact backend; ``auto`` only falls back to exact solvers."""

    normalized = str(backend or "auto").strip().lower().replace("-", "_")
    aliases = {
        "exact": "auto",
        "primaldual": "primal_dual",
        "python": "primal_dual",
        "gurobi": "gurobi_network",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "primal_dual":
        return solve_primal_dual(problem, verify=verify)
    if normalized == "ortools":
        return solve_ortools(problem)
    if normalized == "gurobi_network":
        if gp is None or grb is None:
            raise RuntimeError("Gurobi exact backend requested but Gurobi is unavailable")
        return solve_gurobi_network(
            problem, gp=gp, grb=grb, num_threads=num_threads
        )
    if normalized != "auto":
        raise ValueError(f"unknown exact MCMF backend: {backend}")

    errors: list[str] = []
    # Prefer Gurobi when it is available.  Every backend optimizes the same
    # precision-controlled integer Q grid.
    if gp is not None and grb is not None:
        try:
            result = solve_gurobi_network(
                problem, gp=gp, grb=grb, num_threads=num_threads
            )
            return replace(result, solver_fallback_used=bool(errors))
        except Exception as exc:  # license/runtime errors still permit exact Python
            errors.append(f"gurobi_network={exc}")
    try:
        result = solve_ortools(problem)
        return replace(result, solver_fallback_used=bool(errors))
    except Exception as exc:
        errors.append(f"ortools={exc}")
    result = solve_primal_dual(problem, verify=verify)
    return replace(result, solver_fallback_used=bool(errors))
