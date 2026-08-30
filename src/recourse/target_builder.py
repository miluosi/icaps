"""Explicit R0--R4 targets and a shared joint feasible projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping
from hashlib import sha256
import time

import numpy as np

from src.exact_mcmf import build_reduced_problem, solve_exact

from .types import FeasibleEdgeSnapshot, FeasibleGraphSnapshot
from .config import (
    AssignmentOracleConfig,
    LEADER_CREDITS,
    TARGET_SOLVER_POLICIES,
    canonical_variant,
)


VALID_RECOURSE_VARIANTS = {
    "legacy", "r0", "r1", "r1_structured", "r2", "r3", "r4",
    "recourse_macro",
}


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
    leader_credit: str

    def __post_init__(self) -> None:
        if self.leader_credit not in LEADER_CREDITS:
            raise ValueError(f"invalid leader credit: {self.leader_credit}")

    @property
    def stage_coupled_leader(self) -> bool:
        """Deprecated compatibility view; use ``leader_credit`` in new code."""
        return self.leader_credit == "nested_follower"


RECOURSE_VARIANT_POLICIES = {
    "legacy": RecourseVariantPolicy(True, True, False, True, "uncoupled"),
    "r0": RecourseVariantPolicy(False, False, False, True, "uncoupled"),
    "r1": RecourseVariantPolicy(True, False, False, True, "uncoupled"),
    "r1_structured": RecourseVariantPolicy(True, False, True, False, "uncoupled"),
    "r2": RecourseVariantPolicy(True, True, True, False, "uncoupled"),
    "r3": RecourseVariantPolicy(True, True, False, True, "uncoupled"),
    "r4": RecourseVariantPolicy(True, True, False, True, "nested_follower"),
    "recourse_macro": RecourseVariantPolicy(True, True, False, True, "macro_realized"),
}


class RecourseTargetBuilder:
    VERSION = "solver_consistent_v3"

    def __init__(
        self,
        *,
        backend: str | None = None,
        solver_family: str | None = None,
        graph_reduction: bool | None = None,
        verify: bool | None = None,
        cost_scale: int | None = None,
        target_policy: str | None = None,
        gp=None,
        grb=None,
        environment=None,
        strict: bool | None = None,
    ) -> None:
        if target_policy is not None and target_policy not in TARGET_SOLVER_POLICIES:
            raise ValueError(f"invalid target solver policy: {target_policy}")
        self.backend = backend
        self.solver_family = solver_family
        self.graph_reduction = graph_reduction
        self.verify = verify
        self.cost_scale = cost_scale
        self.target_policy = target_policy
        self.gp, self.grb = gp, grb
        self.environment = environment
        self.strict = strict
        self.last_solver_status = "not_run"
        self.last_fallback_used = False
        self.last_solver_runtime_seconds = 0.0
        self.last_solver_diagnostics: dict[str, object] = {}

    @classmethod
    def from_environment(cls, env) -> "RecourseTargetBuilder":
        config = AssignmentOracleConfig.from_environment(env)
        optimizer = getattr(env, "gurobi_optimizer", None)
        return cls(
            **config.as_dict(),
            gp=getattr(optimizer, "gp", getattr(env, "gp", None)),
            grb=getattr(optimizer, "GRB", getattr(env, "GRB", None)),
            environment=env,
        )

    @staticmethod
    def variant_policy(variant: str) -> RecourseVariantPolicy:
        variant = canonical_variant(variant)
        try:
            return RECOURSE_VARIANT_POLICIES[variant]
        except KeyError as exc:
            raise ValueError(f"invalid recourse variant: {variant}") from exc

    @staticmethod
    def validate_variant(variant: str, transportation_mode: str) -> str:
        variant = canonical_variant(variant)
        if variant not in RECOURSE_VARIANT_POLICIES:
            raise ValueError(
                "recourse variant must be legacy, r0, r1, r1_structured, "
                "r2, r3, recourse_macro, or r4"
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
        reward_aev: float = 0.0,
    ) -> float:
        variant = canonical_variant(variant)
        policy = RecourseTargetBuilder.variant_policy(variant)
        ordinary_bootstrap = 0.0 if done else (gamma ** elapsed_epochs) * temporal_value
        if policy.leader_credit == "macro_realized":
            return float(reward_ev + reward_aev + ordinary_bootstrap)
        if policy.leader_credit == "nested_follower":
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

        The serialized graph is reduced to the same integer-grid exact MCMF
        problem used by rollout.  This keeps quantization and deterministic
        zero-gain tie handling identical for online and target selection.
        """

        edges = graph.edges
        if not edges:
            self.last_solver_status = "empty"
            self.last_fallback_used = False
            self.last_solver_runtime_seconds = 0.0
            self.last_solver_diagnostics = self.solver_config_for_graph(graph)
            return ()
        self._validate_resource_capacities(edges)
        self.last_solver_runtime_seconds = 0.0
        vehicle_ids = sorted({edge.vehicle_id for edge in edges})
        vehicle_index = {
            vehicle_id: index for index, vehicle_id in enumerate(vehicle_ids)
        }

        def action_key(edge: FeasibleEdgeSnapshot):
            if edge.resource_type is not None and edge.resource_id is not None:
                return ("resource", edge.resource_type, int(edge.resource_id))
            if bool(dict(edge.metadata).get("continuing", False)):
                return ("continuing", edge.edge_id)
            return (
                "action",
                int(edge.action_type),
                edge.action_id,
                int(edge.target_location),
            )

        action_keys = []
        for edge in edges:
            key = action_key(edge)
            if key not in action_keys:
                action_keys.append(key)
        action_index = {key: index for index, key in enumerate(action_keys)}
        feasibility = np.zeros(
            (len(vehicle_ids), len(action_keys)), dtype=bool
        )
        q_values = np.zeros_like(feasibility, dtype=np.float64)
        capacities = np.full(
            len(action_keys), len(vehicle_ids), dtype=np.int64
        )
        edge_by_vehicle_action: dict[tuple[int, int], FeasibleEdgeSnapshot] = {}
        for edge in edges:
            row = vehicle_index[edge.vehicle_id]
            column = action_index[action_key(edge)]
            feasibility[row, column] = True
            q_values[row, column] = self._score(edge, scores)
            edge_by_vehicle_action[(row, column)] = edge
            if edge.resource_type is not None and edge.resource_id is not None:
                capacities[column] = int(edge.resource_capacity)

        build_start = time.perf_counter()
        config = self.solver_config_for_graph(graph)
        if str(config["backend"]) == "gurobi_network" and self.gp is None:
            optimizer = getattr(self.environment, "gurobi_optimizer", None)
            self.gp = getattr(optimizer, "gp", getattr(self.environment, "gp", None))
            self.grb = getattr(optimizer, "GRB", getattr(self.environment, "GRB", None))
        problem = build_reduced_problem(
            feasibility,
            q_values,
            capacities,
            cost_scale=int(config["cost_scale"]),
            graph_reduction=bool(config["graph_reduction"]),
        )
        result = solve_exact(
            problem,
            backend=str(config["backend"]),
            verify=bool(config["verify"]),
            gp=self.gp,
            grb=self.grb,
        )
        if bool(config["strict"]) and result.fallback_used:
            raise RuntimeError(
                "strict target assignment forbids backend fallback"
            )
        self.last_solver_runtime_seconds = time.perf_counter() - build_start
        self.last_solver_status = result.status.lower()
        self.last_fallback_used = bool(result.fallback_used)
        self.last_solver_diagnostics = {
            **config,
            "status": self.last_solver_status,
            "fallback_used": self.last_fallback_used,
            "runtime_seconds": self.last_solver_runtime_seconds,
            "objective_q": float(result.objective_q),
        }
        selected = tuple(
            edge_by_vehicle_action[(row, int(column))].edge_id
            for row, column in sorted(result.action_by_vehicle.items())
        )
        self.last_solver_diagnostics["selected_edge_trace_hash"] = sha256(
            "\n".join(selected).encode()
        ).hexdigest()
        self.verify_feasible(graph, selected)
        return selected

    def solver_config_for_graph(self, graph: FeasibleGraphSnapshot) -> dict[str, object]:
        policy = str(
            self.target_policy
            or getattr(graph, "target_solver_policy", "fixed_primal_dual_exact")
        )
        solver_family = str(
            self.solver_family or getattr(graph, "solver_family", "exact")
        )
        if policy == "same_as_rollout_exact" and solver_family != "exact":
            raise ValueError(
                "same_as_rollout_exact requires an exact rollout; use "
                "exact_oracle_for_approximate_rollout for approximate solvers"
            )
        backend = str(self.backend or graph.solver_backend)
        metadata_migrated = False
        if backend.startswith("mcmf:"):
            backend = backend.split(":", 1)[1] or "primal_dual"
            metadata_migrated = True
        elif backend in {"", "unknown", "test", "mcmf"}:
            # Replay schema <=3 used descriptive placeholders here while the
            # target operator was always exact primal-dual. Preserve those
            # checkpoints explicitly; arbitrary new backend names still fail.
            backend = "primal_dual"
            metadata_migrated = True
        if policy == "fixed_primal_dual_exact":
            backend = "primal_dual"
        elif policy == "exact_oracle_for_approximate_rollout" and self.backend is None:
            # This policy is the one explicit exception allowed to decouple
            # targets from an approximate rollout backend.
            backend = "primal_dual"
        return {
            "backend": backend,
            "graph_reduction": bool(
                getattr(graph, "graph_reduction", True)
                if self.graph_reduction is None else self.graph_reduction
            ),
            "verify": bool(
                getattr(graph, "solver_verify", True)
                if self.verify is None else self.verify
            ),
            "cost_scale": max(1, int(
                graph.objective_cost_scale if self.cost_scale is None else self.cost_scale
            )),
            "target_policy": policy,
            "rollout_backend": str(graph.solver_backend),
            "rollout_solver_family": solver_family,
            "target_solver_family": "exact",
            "strict": bool(
                getattr(graph, "solver_strict", True)
                if self.strict is None else self.strict
            ),
            "legacy_solver_metadata_migrated": metadata_migrated,
        }

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
