"""Explicit R0--R4 targets and a shared joint feasible projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .types import FeasibleEdgeSnapshot, FeasibleGraphSnapshot


VALID_RECOURSE_VARIANTS = {"legacy", "r0", "r1", "r2", "r3", "r4"}


@dataclass(frozen=True)
class TargetComponents:
    selected_edge_ids: tuple[str, ...]
    online_selection_value: float
    target_evaluation_value: float


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
    VERSION = "solver_consistent_v1"

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
            return ()
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
                lower.append(0.0)
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
                upper.append(float(max(edge.resource_capacity for edge in resource_edges)))
            constraints = LinearConstraint(
                np.asarray(rows, dtype=np.float64),
                np.asarray(lower, dtype=np.float64),
                np.asarray(upper, dtype=np.float64),
            )
            result = milp(
                c=-score_values,
                integrality=np.ones(len(edges), dtype=np.int8),
                bounds=Bounds(np.zeros(len(edges)), np.ones(len(edges))),
                constraints=constraints,
                options={"disp": False},
            )
            if result.success and result.x is not None:
                return tuple(
                    edge.edge_id
                    for edge, selected in zip(edges, result.x)
                    if float(selected) > 0.5
                )
        except (ImportError, ValueError, RuntimeError):
            pass
        return self._greedy_projection(edges, score_values)

    def double_q_target(
        self,
        graph: FeasibleGraphSnapshot,
        *,
        online_scores: Mapping[str, float] | Callable[[FeasibleEdgeSnapshot], float],
        target_scores: Mapping[str, float] | Callable[[FeasibleEdgeSnapshot], float],
        structured_only: bool = False,
    ) -> TargetComponents:
        if structured_only:
            selection_scores = {edge.edge_id: edge.structured_score for edge in graph.edges}
            evaluation_scores = selection_scores
        else:
            selection_scores = online_scores
            evaluation_scores = target_scores
        selected = self.project(graph, selection_scores)
        selected_set = set(selected)
        return TargetComponents(
            selected_edge_ids=selected,
            online_selection_value=sum(
                self._score(edge, selection_scores)
                for edge in graph.edges
                if edge.edge_id in selected_set
            ),
            target_evaluation_value=sum(
                self._score(edge, evaluation_scores)
                for edge in graph.edges
                if edge.edge_id in selected_set
            ),
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
                resource_capacities[key] = edge.resource_capacity
        if any(count > 1 for count in vehicle_counts.values()):
            raise AssertionError("joint action selects more than one edge for a vehicle")
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
    def _greedy_projection(
        edges: tuple[FeasibleEdgeSnapshot, ...],
        scores: np.ndarray,
    ) -> tuple[str, ...]:
        selected = []
        used_vehicles = set()
        resource_usage: dict[tuple[str, int], int] = {}
        for index in np.argsort(-scores):
            if float(scores[int(index)]) <= 0.0:
                continue
            edge = edges[int(index)]
            if edge.vehicle_id in used_vehicles:
                continue
            if edge.resource_type is not None and edge.resource_id is not None:
                key = (edge.resource_type, edge.resource_id)
                if resource_usage.get(key, 0) >= edge.resource_capacity:
                    continue
                resource_usage[key] = resource_usage.get(key, 0) + 1
            used_vehicles.add(edge.vehicle_id)
            selected.append(edge.edge_id)
        return tuple(selected)
