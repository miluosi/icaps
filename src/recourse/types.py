"""Immutable records used by integrated and two-stage recourse learning.

The simulator contains mutable request/vehicle dictionaries.  None of those
objects are safe replay payloads: they continue changing after collection.
This module deliberately stores only primitive values and tuples so a replay
row always describes the state in which it was collected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from typing import Any, Iterable, Mapping


REPLAY_SCHEMA_VERSION = 2
STATE_VARIANTS = (
    "joint_state_shared_critic",
    "joint_state_separate_critics",
    "fleet_local_separate_critics",
    "fleet_local_shared_critic",
)
LEARNER_VARIANTS = (
    "legacy",
    "integrated_directq",
    "optimization_anchored_residual",
)


class ActionType(IntEnum):
    """Canonical action identifiers shared by simulators and learners."""

    WAIT = 0
    RELOCATE = 1
    SERVICE = 2
    CHARGE = 3


@dataclass(frozen=True)
class RequestSnapshot:
    request_id: int
    pickup: int
    dropoff: int
    value: float
    final_value: float
    created_time: float
    pickup_deadline: float
    dropoff_deadline: float
    travel_time: float
    trip_distance_km: float | None = None
    pickup_zone_id: int | None = None

    @classmethod
    def from_request(cls, request: Any) -> "RequestSnapshot":
        pickup = int(getattr(request, "pickup", 0))
        return cls(
            request_id=int(getattr(request, "request_id")),
            pickup=pickup,
            dropoff=int(getattr(request, "dropoff", pickup)),
            value=float(getattr(request, "value", 0.0) or 0.0),
            final_value=float(
                getattr(request, "final_value", getattr(request, "value", 0.0))
                or 0.0
            ),
            created_time=float(getattr(request, "created_time", 0.0) or 0.0),
            pickup_deadline=float(
                getattr(request, "pickup_deadline", float("inf"))
            ),
            dropoff_deadline=float(
                getattr(request, "dropoff_deadline", float("inf"))
            ),
            travel_time=float(getattr(request, "travel_time", 0.0) or 0.0),
            trip_distance_km=(
                None
                if getattr(request, "trip_distance_km", None) is None
                else float(getattr(request, "trip_distance_km"))
            ),
            pickup_zone_id=(
                None
                if getattr(request, "pickup_zone_id", None) is None
                else int(getattr(request, "pickup_zone_id"))
            ),
        )


@dataclass(frozen=True)
class VehicleSnapshot:
    vehicle_id: int
    vehicle_type: int
    location: int
    battery: float
    idle_time: float
    online: bool
    assigned_request: int | None
    passenger_onboard: int | None
    charging_station: int | None
    charging_target: int | None
    target_location: int | None

    @classmethod
    def from_vehicle(cls, vehicle_id: int, vehicle: Mapping[str, Any]) -> "VehicleSnapshot":
        target = vehicle.get("target_location")
        if not isinstance(target, (int, float)):
            target = None
        return cls(
            vehicle_id=int(vehicle_id),
            vehicle_type=int(vehicle.get("type", 1)),
            location=int(vehicle.get("location", 0)),
            battery=float(vehicle.get("battery", 1.0) or 0.0),
            idle_time=float(vehicle.get("idle_timer", 0.0) or 0.0),
            online=bool(vehicle.get("is_online", True)),
            assigned_request=_optional_int(vehicle.get("assigned_request")),
            passenger_onboard=_optional_int(vehicle.get("passenger_onboard")),
            charging_station=_optional_int(vehicle.get("charging_station")),
            charging_target=_optional_int(vehicle.get("charging_target")),
            target_location=_optional_int(target),
        )


@dataclass(frozen=True)
class StationSnapshot:
    station_id: int
    location: int
    capacity: int
    occupied: int
    inbound: int
    queued: int
    physical_capacity: int = 0
    queue_admission_capacity: int = 0
    reserve_inbound_capacity: bool = False
    remaining_admission_capacity: int = 0


@dataclass(frozen=True)
class SystemSnapshot:
    epoch_id: int
    current_time: float
    zone_ids: tuple[int, ...]
    vehicles: tuple[VehicleSnapshot, ...]
    requests: tuple[RequestSnapshot, ...]
    stations: tuple[StationSnapshot, ...]
    request_labels: tuple[tuple[int, str], ...] = ()
    exogenous_context: tuple[tuple[str, float | int | str | bool | None], ...] = ()

    def request_label(self, request_id: int) -> str | None:
        return dict(self.request_labels).get(int(request_id))

    def masked(self, variant: str, vehicle_type: int | None = None) -> "SystemSnapshot":
        """Create deterministic state-ablation views from one raw snapshot."""

        variant = str(variant or "joint_state_shared_critic")
        if variant in {
            "joint_state_shared_critic",
            "joint_state_separate_critics",
            "s0",
            "s1",
        }:
            return self
        if variant not in {
            "fleet_local_separate_critics",
            "fleet_local_shared_critic",
            "s2",
            "s3",
        }:
            raise ValueError(f"unknown state variant: {variant}")
        if vehicle_type not in {1, 2}:
            raise ValueError("fleet-local state masks require vehicle_type 1 or 2")
        masked_vehicles = tuple(
            vehicle
            if vehicle.vehicle_type == int(vehicle_type)
            else replace(
                vehicle,
                location=0,
                battery=0.0,
                idle_time=0.0,
                online=False,
                assigned_request=None,
                passenger_onboard=None,
                charging_station=None,
                charging_target=None,
                target_location=None,
            )
            for vehicle in self.vehicles
        )
        return replace(self, vehicles=masked_vehicles)


@dataclass(frozen=True)
class FeasibleEdgeSnapshot:
    edge_id: str
    vehicle_id: int
    vehicle_type: int
    action_type: ActionType
    action_id: str
    target_location: int
    post_action_location: int
    request_id: int | None = None
    station_id: int | None = None
    resource_type: str | None = None
    resource_id: int | None = None
    resource_capacity: int = 1
    structured_score: float = 0.0
    collection_score: float = 0.0
    request_value: float = 0.0
    target_distance: float = 0.0
    post_action_distance: float = 0.0
    post_action_duration: float = 0.0
    target_zoneid: int = 0
    post_action_zoneid: int = 0
    queue_features: tuple[float, ...] = ()
    post_demand_feature: float | None = None
    metadata: tuple[tuple[str, float | int | str | bool | None], ...] = ()

    def candidate_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_id,
            "target_location": self.target_location,
            "post_action_location": self.post_action_location,
            "request_value": self.request_value,
            "target_distance": self.target_distance,
            "post_action_distance": self.post_action_distance,
            "post_action_duration": self.post_action_duration,
            "target_zoneid": self.target_zoneid,
            "post_action_zoneid": self.post_action_zoneid,
            "target_station_id": self.station_id,
            "queue_features": self.queue_features,
            "post_demand_feature": self.post_demand_feature,
        }


@dataclass(frozen=True)
class FeasibleGraphSnapshot:
    graph_id: str
    epoch_id: int
    stage_id: int
    solver_backend: str
    state: SystemSnapshot
    edges: tuple[FeasibleEdgeSnapshot, ...]
    selected_edge_ids: tuple[str, ...] = ()
    solver_status: str = "unknown"
    solver_objective: float = 0.0
    solver_runtime_seconds: float = 0.0

    def with_selected(
        self,
        edge_ids: Iterable[str],
        *,
        status: str = "selected",
        objective: float | None = None,
    ) -> "FeasibleGraphSnapshot":
        selected = tuple(str(edge_id) for edge_id in edge_ids)
        if objective is None:
            selected_set = set(selected)
            objective = sum(
                edge.collection_score
                for edge in self.edges
                if edge.edge_id in selected_set
            )
        return replace(
            self,
            selected_edge_ids=selected,
            solver_status=status,
            solver_objective=float(objective),
        )


@dataclass(frozen=True)
class JointActionSnapshot:
    stage_id: int
    selected_edge_ids: tuple[str, ...]
    structured_value: float
    collection_value: float

    @classmethod
    def from_graph(
        cls,
        graph: FeasibleGraphSnapshot | None,
        edge_ids: Iterable[str] = (),
    ) -> "JointActionSnapshot | None":
        if graph is None:
            return None
        selected = tuple(edge_ids) or graph.selected_edge_ids
        selected_set = set(selected)
        return cls(
            stage_id=graph.stage_id,
            selected_edge_ids=selected,
            structured_value=sum(
                edge.structured_score
                for edge in graph.edges
                if edge.edge_id in selected_set
            ),
            collection_value=sum(
                edge.collection_score
                for edge in graph.edges
                if edge.edge_id in selected_set
            ),
        )


@dataclass(frozen=True)
class OfferAttempt:
    offer_id: str
    transition_id: str
    epoch_id: int
    attempt_index: int
    request_id: int
    ev_id: int
    selected_by_stage1: bool
    acceptance_probability: float
    acceptance_uniform: float
    accepted: bool
    rejected: bool
    rejection_reason: str | None
    request_snapshot: RequestSnapshot
    vehicle_snapshot: VehicleSnapshot


@dataclass(frozen=True)
class ResidualObservation:
    request_id: int
    epoch_id: int
    category: str
    eligible: bool


@dataclass(frozen=True)
class RejectionOutcomeSnapshot:
    offer_attempts: tuple[OfferAttempt, ...] = ()

    @property
    def rejected_request_ids(self) -> tuple[int, ...]:
        return tuple(sorted({offer.request_id for offer in self.offer_attempts if offer.rejected}))


@dataclass(frozen=True)
class RecourseEvent:
    request_id: int
    epoch_id: int
    residual_category: str
    eligible: bool
    assigned: bool
    picked_up: bool
    completed: bool
    assigned_vehicle_id: int | None = None
    assignment_epoch_id: int | None = None
    pickup_epoch_id: int | None = None
    completion_epoch_id: int | None = None
    expired: bool = False
    cancelled: bool = False
    residual_observations: tuple[ResidualObservation, ...] = ()
    first_rejected_epoch: int | None = None
    assigned_vehicle_type: int | None = None
    pickup_vehicle_id: int | None = None
    pickup_vehicle_type: int | None = None
    completion_vehicle_id: int | None = None
    completion_vehicle_type: int | None = None
    same_epoch_recourse_link: bool = False


@dataclass(frozen=True)
class OutcomeSummary:
    events: tuple[RecourseEvent, ...] = ()

    def count(self, field_name: str) -> int:
        return sum(bool(getattr(event, field_name)) for event in self.events)


@dataclass(frozen=True)
class PlannerMetadata:
    backend: str = "unknown"
    target_backend: str = "same_as_execution"
    objective: float = 0.0
    status: str = "unknown"
    runtime_seconds: float = 0.0


@dataclass(frozen=True)
class RecourseTransition:
    transition_id: str
    episode_id: int
    day_id: str
    seed: int
    epoch_id: int
    mode: str
    recourse_variant: str
    state_variant: str
    learner_variant: str
    solver_backend: str
    pre_state: SystemSnapshot
    ev_stage_graph: FeasibleGraphSnapshot | None
    ev_joint_action: JointActionSnapshot | None
    rejection_outcome: RejectionOutcomeSnapshot
    residual_state: SystemSnapshot | None
    aev_stage_graph: FeasibleGraphSnapshot | None
    aev_joint_action: JointActionSnapshot | None
    reward_ev: float
    reward_aev: float
    reward_system: float
    next_state: SystemSnapshot
    elapsed_epochs: float
    done: bool
    planner_metadata: PlannerMetadata = field(default_factory=PlannerMetadata)
    outcome_summary: OutcomeSummary = field(default_factory=OutcomeSummary)
    schema_version: int = REPLAY_SCHEMA_VERSION
    target_builder_version: str = "solver_consistent_v2"
    run_id: str = ""
    cumulative_episode_id: int = 0
    transition_sequence_index: int = 0
    previous_transition_id: str | None = None
    next_transition_id: str | None = None
    request_generation_seed: int = 0
    vehicle_initialization_seed: int = 0
    reward_scope: str = "selected_epoch_actions"
    rewarded_vehicle_ids: tuple[int, ...] = ()
    continuing_action_edge_ids: tuple[str, ...] = ()
    online_model_step: int = 0
    target_model_step: int = 0
    target_solver_backend: str = ""
    target_solver_status: str = "not_built"
    target_selected_edge_ids: tuple[str, ...] = ()
    target_structured_value: float = 0.0
    target_correction_value: float = 0.0
    target_full_value: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay schema {self.schema_version}; expected {REPLAY_SCHEMA_VERSION}"
            )
        if self.mode not in {"integrated", "ev_first", "aev_first"}:
            raise ValueError(f"invalid transition mode: {self.mode}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EdgeTransition:
    parent_transition_id: str
    stage_id: int
    vehicle_id: int
    vehicle_type: int
    action_id: str
    action_type: ActionType
    selected: bool
    state: SystemSnapshot
    graph: FeasibleGraphSnapshot
    edge: FeasibleEdgeSnapshot
    realized_reward: float
    acceptance_outcome: str | None
    residual_category: str | None
    recourse_value_target: float | None
    terminal: bool
    schema_version: int = REPLAY_SCHEMA_VERSION


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
