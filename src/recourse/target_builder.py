"""Explicit R0--R4 targets and a shared joint feasible projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping
import time

import numpy as np

from .types import FeasibleEdgeSnapshot, FeasibleGraphSnapshot


VALID_RECOURSE_VARIANTS = {"legacy", "r0", "r1", "r2", "r3", "r4"}


@dataclass(frozen=True)
class TargetComponents:
    selected_edge_ids: tuple[str, ...]
    online_full_value: float
    target_structured_value: float
    target_correction_value: float
    solver_status: str = "optimal"
    fallback_used: bool = False
    solver_runtime_seconds: float = 0.0

    @property
    def target_full_value(self) -> float:
        return float(self.target_structured_value + self.target_correction_value)

    # Compatibility names for older diagnostics.  Both now deliberately
    # refer to full deployed values rather than raw residuals.
    @property
    def online_selection_value(self) -> float:
        return self.online_full_value

    @property
    def target_evaluation_value(self) -> float:
        return self.target_full_value


@dataclass(frozen=True)
class RecourseVariantPolicy:
    """Executable R0--R4 semantics used by both simulators and tests."""

    rejection_enabled: bool
    same_epoch_repair: bool
    structured_only_follower: bool
    learned_follower: bool
    stage_coupled_leader: bool


RECOURSE_VARIANT_POLICIES = {
    "legacy": RecourseVariantPolicy(True, True, False, True, False),
    "r0": RecourseVariantPolicy(False, False, False, True, False),
    "r1": RecourseVariantPolicy(True, False, False, False, False),
    "r2": RecourseVariantPolicy(True, True, True, False, False),
    "r3": RecourseVariantPolicy(True, True, False, True, False),
    "r4": RecourseVariantPolicy(True, True, False, True, True),
}


class RecourseTargetBuilder:
    VERSION = "solver_consistent_v2"

    def __init__(self) -> None:
        self.last_solver_status = "not_run"
        self.last_fallback_used = False
        self.last_solver_runtime_seconds = 0.0

    @staticmethod
    def variant_policy(variant: str) -> RecourseVariantPolicy:
        variant = str(variant or "legacy").lower()
        try:
            return RECOURSE_VARIANT_POLICIES[variant]
        except KeyError as exc:
            raise ValueError(f"invalid recourse variant: {variant}") from exc

    @staticmethod
    def validate_variant(variant: str, transportation_mode: str) -> str:
        variant = str(variant or "legacy").lower()
        if variant not in RECOURSE_VARIANT_POLICIES:
            raise ValueError(
                "recourse variant must be legacy or one of r0, r1, r2, r3, r4"
            )
        normalized_mode = str(transportation_mode).replace("-", "_").lower()
        if variant != "legacy" and normalized_mode not in {"evfirst", "ev_first"}:
            raise ValueError(
                f"recourse variant {variant} is defined only for transportation_mode=evfirst"
            )
        return variant

    @staticmethod
    def leader_target(
        *,
        variant: str,
        reward_ev: float,
        follower_value: float,
        temporal_value: float,
        done: bool,
        gamma: float = 0.95,
        elapsed_epochs: float = 1.0,
        within_epoch_gamma: float = 1.0,
    ) -> float:
        variant = str(variant).lower()
        policy = RecourseTargetBuilder.variant_policy(variant)
        ordinary_bootstrap = 0.0 if done else (gamma ** elapsed_epochs) * temporal_value
        if policy.stage_coupled_leader:
            # Rejection is an observed within-epoch outcome, not a terminal
            # mask.  The follower bootstrap is retained even when the EV offer
            # was rejected.
            return float(reward_ev + within_epoch_gamma * follower_value)
        return float(reward_ev + ordinary_bootstrap)

    @staticmethod
    def follower_target(
        *,
        reward_aev: float,
        temporal_value: float,
        done: bool,
        gamma: float = 0.95,
        elapsed_epochs: float = 1.0,
    ) -> float:
        bootstrap = 0.0 if done else (gamma ** elapsed_epochs) * temporal_value
        return float(reward_aev + bootstrap)

    @staticmethod
    def integrated_target(
        *,
        reward_system: float,
        temporal_value: float,
        done: bool,
        gamma: float = 0.95,
        elapsed_epochs: float = 1.0,
    ) -> float:
        bootstrap = 0.0 if done else (gamma ** elapsed_epochs) * temporal_value
        return float(reward_system + bootstrap)

    @staticmethod
    def correction_bellman_target(
        *,
        reward: float,
        discount: float,
        next_components: TargetComponents | None,
        current_structured_value: float,
        direct_q: bool = False,
    ) -> float:
        next_full = (
            0.0
            if next_components is None
            else float(next_components.target_full_value)
        )
        full_target = float(reward) + float(discount) * next_full
        return (
            full_target
            if direct_q
            else full_target - float(current_structured_value)
        )

    def project(
        self,
        graph: FeasibleGraphSnapshot,
        scores: Mapping[str, float] | Callable[[FeasibleEdgeSnapshot], float] | None = None,
    ) -> tuple[str, ...]:
        """Solve the same additive resource-constrained projection as rollout.

        SciPy's HiGHS MILP backend is used for target selection.  Constraints
        are derived from the serialized graph, so request exclusivity and
        charging capacities cannot silently differ from collection.
        """

        edges = graph.edges
        if not edges:
            self.last_solver_status = "empty"
            self.last_fallback_used = False
            self.last_solver_runtime_seconds = 0.0
            return ()
        self._validate_resource_capacities(edges)
        self.last_solver_runtime_seconds = 0.0
        score_values = np.asarray(
            [self._score(edge, scores) for edge in edges], dtype=np.float64
        )
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp

            rows = []
            lower = []
            upper = []
            for vehicle_id in sorted({edge.vehicle_id for edge in edges}):
                rows.append([1.0 if edge.vehicle_id == vehicle_id else 0.0 for edge in edges])
                # Rollout sends one unit of flow per represented vehicle.  A
                # real wait/continue edge supplies the outside option.
                lower.append(1.0)
                upper.append(1.0)
            resources = sorted(
                {
                    (edge.resource_type, edge.resource_id)
                    for edge in edges
                    if edge.resource_type is not None and edge.resource_id is not None
                }
            )
            for resource in resources:
                resource_edges = [edge for edge in edges if (edge.resource_type, edge.resource_id) == resource]
                rows.append([
                    1.0 if (edge.resource_type, edge.resource_id) == resource else 0.0
                    for edge in edges
                ])
                lower.append(0.0)
                upper.append(float(resource_edges[0].resource_capacity))
            constraints = LinearConstraint(
                np.asarray(rows, dtype=np.float64),
                np.asarray(lower, dtype=np.float64),
                np.asarray(upper, dtype=np.float64),
            )
            solve_start = time.perf_counter()
            result = milp(
                c=-score_values,
                integrality=np.ones(len(edges), dtype=np.int8),
                bounds=Bounds(np.zeros(len(edges)), np.ones(len(edges))),
                constraints=constraints,
                options={"disp": False},
            )
            self.last_solver_runtime_seconds = time.perf_counter() - solve_start
            if result.success and result.x is not None:
                self.last_solver_status = "optimal"
                self.last_fallback_used = False
                return tuple(
                    edge.edge_id
                    for edge, selected in zip(edges, result.x)
                    if float(selected) > 0.5
                )
            message = getattr(result, "message", "unknown target MILP failure")
            self.last_solver_status = f"failed:{message}"
            raise RuntimeError(
                "target projection is infeasible or failed; every represented "
                f"vehicle requires a real action: {message}"
            )
        except ImportError as exc:
            self.last_solver_status = "missing_scipy_milp"
            self.last_solver_runtime_seconds = 0.0
            raise RuntimeError(
                "solver-consistent target projection requires scipy.optimize.milp"
            ) from exc

    def double_q_target(
        self,
        graph: FeasibleGraphSnapshot,
        *,
        online_scores: Mapping[str, float] | Callable[[FeasibleEdgeSnapshot], float],
        target_scores: Mapping[str, float] | Callable[[FeasibleEdgeSnapshot], float],
        structured_only: bool = False,
        direct_q: bool = False,
    ) -> TargetComponents:
        if structured_only:
            selection_scores = {edge.edge_id: edge.structured_score for edge in graph.edges}
            correction_scores = {edge.edge_id: 0.0 for edge in graph.edges}
        else:
            selection_scores = online_scores
            correction_scores = target_scores
        selected = self.project(graph, selection_scores)
        selected_set = set(selected)
        structured_value = (
            0.0
            if direct_q
            else sum(
                float(edge.structured_score)
                for edge in graph.edges
                if edge.edge_id in selected_set
            )
        )
        return TargetComponents(
            selected_edge_ids=selected,
            online_full_value=sum(
                self._score(edge, selection_scores)
                for edge in graph.edges
                if edge.edge_id in selected_set
            ),
            target_structured_value=structured_value,
            target_correction_value=sum(
                self._score(edge, correction_scores)
                for edge in graph.edges
                if edge.edge_id in selected_set
            ),
            solver_status=self.last_solver_status,
            fallback_used=self.last_fallback_used,
            solver_runtime_seconds=self.last_solver_runtime_seconds,
        )

    @staticmethod
    def verify_feasible(
        graph: FeasibleGraphSnapshot,
        selected_edge_ids: tuple[str, ...] | list[str],
    ) -> None:
        selected_set = set(selected_edge_ids)
        selected = [edge for edge in graph.edges if edge.edge_id in selected_set]
        if len(selected) != len(selected_set):
            raise AssertionError("selected action contains an edge absent from the graph")
        vehicle_counts: dict[int, int] = {}
        resource_counts: dict[tuple[str, int], int] = {}
        resource_capacities: dict[tuple[str, int], int] = {}
        for edge in selected:
            vehicle_counts[edge.vehicle_id] = vehicle_counts.get(edge.vehicle_id, 0) + 1
            if edge.resource_type is not None and edge.resource_id is not None:
                key = (edge.resource_type, edge.resource_id)
                resource_counts[key] = resource_counts.get(key, 0) + 1
                existing = resource_capacities.setdefault(key, edge.resource_capacity)
                if existing != edge.resource_capacity:
                    raise AssertionError(
                        f"resource {key} has inconsistent capacities: "
                        f"{existing} and {edge.resource_capacity}"
                    )
        represented_vehicles = {edge.vehicle_id for edge in graph.edges}
        invalid_vehicle_counts = {
            vehicle_id: vehicle_counts.get(vehicle_id, 0)
            for vehicle_id in represented_vehicles
            if vehicle_counts.get(vehicle_id, 0) != 1
        }
        if invalid_vehicle_counts:
            raise AssertionError(
                "joint action must select exactly one real edge per represented "
                f"vehicle: {invalid_vehicle_counts}"
            )
        for key, count in resource_counts.items():
            if count > resource_capacities[key]:
                vehicles = [
                    edge.vehicle_id
                    for edge in selected
                    if (edge.resource_type, edge.resource_id) == key
                ]
                raise AssertionError(
                    f"joint action exceeds {key} capacity: "
                    f"count={count}, capacity={resource_capacities[key]}, "
                    f"vehicles={vehicles}"
                )

    @staticmethod
    def _score(
        edge: FeasibleEdgeSnapshot,
        scores: Mapping[str, float] | Callable[[FeasibleEdgeSnapshot], float] | None,
    ) -> float:
        if scores is None:
            return float(edge.collection_score)
        if callable(scores):
            return float(scores(edge))
        return float(scores[edge.edge_id])

    @staticmethod
    def _validate_resource_capacities(
        edges: tuple[FeasibleEdgeSnapshot, ...],
    ) -> None:
        capacities: dict[tuple[str, int], int] = {}
        for edge in edges:
            if edge.resource_type is None or edge.resource_id is None:
                continue
            key = (edge.resource_type, edge.resource_id)
            capacity = int(edge.resource_capacity)
            if capacity < 0:
                raise ValueError(f"resource {key} has negative capacity {capacity}")
            previous = capacities.setdefault(key, capacity)
            if previous != capacity:
                raise ValueError(
                    f"resource {key} has inconsistent capacities: "
                    f"{previous} and {capacity}"
                )
