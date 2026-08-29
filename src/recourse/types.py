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


REPLAY_SCHEMA_VERSION = 3
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
    surge_bonus: float | None = None

    @classmethod
    def from_request(cls, request: Any) -> "RequestSnapshot":
        pickup = int(getattr(request, "pickup", 0))
        return cls(
            request_id=int(getattr(request, "request_id")),
            surge_bonus=(None if getattr(request, "surge_bonus", None) is None
                         else float(request.surge_bonus)),
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
    service_phase: str | None = None
    service_request_id: int | None = None
    service_request_value: float = 0.0
    service_pickup: int | None = None
    service_dropoff: int | None = None
    remaining_pickup_time: float = 0.0
    remaining_trip_time: float = 0.0
    remaining_service_distance: float = 0.0
    service_deadline_remaining: float = 0.0
    charging_time_left: float = 0.0
    relocation_target: int | None = None
    remaining_relocation_time: float = 0.0
    stationary_duration_left: float = 0.0
    penalty_timer: float = 0.0

    @classmethod
    def from_vehicle(
        cls,
        vehicle_id: int,
        vehicle: Mapping[str, Any],
        *,
        request: Any | None = None,
        env: Any | None = None,
        current_time: float = 0.0,
    ) -> "VehicleSnapshot":
        target = vehicle.get("target_location")
        if not isinstance(target, (int, float)):
            target = None
        assigned_request = _optional_int(vehicle.get("assigned_request"))
        passenger_onboard = _optional_int(vehicle.get("passenger_onboard"))
        service_request_id = (
            passenger_onboard
            if passenger_onboard is not None
            else assigned_request
        )
        service_phase = (
            "passenger_onboard"
            if passenger_onboard is not None
            else ("to_pickup" if assigned_request is not None else None)
        )
        pickup = _optional_int(getattr(request, "pickup", None))
        dropoff = _optional_int(getattr(request, "dropoff", None))
        location = int(vehicle.get("location", 0))
        remaining_pickup = (
            _environment_metric(env, "get_travel_time", location, pickup)
            if service_phase == "to_pickup" and pickup is not None
            else 0.0
        )
        remaining_trip = (
            _environment_metric(env, "get_travel_time", location, dropoff)
            if service_phase == "passenger_onboard" and dropoff is not None
            else (
                float(getattr(request, "travel_time", 0.0) or 0.0)
                if service_phase == "to_pickup"
                else 0.0
            )
        )
        remaining_service_distance = 0.0
        if service_phase == "to_pickup" and pickup is not None:
            remaining_service_distance += _environment_distance(
                env, location, pickup
            )
            if dropoff is not None:
                remaining_service_distance += _environment_distance(
                    env, pickup, dropoff
                )
        elif service_phase == "passenger_onboard" and dropoff is not None:
            remaining_service_distance = _environment_distance(
                env, location, dropoff
            )
        deadline = (
            getattr(request, "dropoff_deadline", current_time)
            if service_phase == "passenger_onboard"
            else getattr(request, "pickup_deadline", current_time)
        )
        charging_station = _optional_int(vehicle.get("charging_station"))
        charging_target = _optional_int(
            vehicle.get(
                "charging_target", vehicle.get("target_charging_station")
            )
        )
        relocation_target = None
        if (
            service_request_id is None
            and charging_station is None
            and charging_target is None
        ):
            relocation_target = _optional_int(
                vehicle.get("idle_target", target)
            )
        return cls(
            vehicle_id=int(vehicle_id),
            vehicle_type=int(vehicle.get("type", 1)),
            location=location,
            battery=float(vehicle.get("battery", 1.0) or 0.0),
            idle_time=float(vehicle.get("idle_timer", 0.0) or 0.0),
            penalty_timer=float(vehicle.get("penalty_timer", 0.0) or 0.0),
            online=bool(vehicle.get("is_online", True)),
            assigned_request=assigned_request,
            passenger_onboard=passenger_onboard,
            charging_station=charging_station,
            charging_target=charging_target,
            target_location=_optional_int(target),
            service_phase=service_phase,
            service_request_id=service_request_id,
            service_request_value=float(
                getattr(request, "final_value", getattr(request, "value", 0.0))
                or 0.0
            ),
            service_pickup=pickup,
            service_dropoff=dropoff,
            remaining_pickup_time=max(0.0, remaining_pickup),
            remaining_trip_time=max(0.0, remaining_trip),
            remaining_service_distance=max(0.0, remaining_service_distance),
            service_deadline_remaining=max(
                0.0, float(deadline or current_time) - float(current_time)
            ),
            charging_time_left=max(
                0.0, float(vehicle.get("charging_time_left", 0.0) or 0.0)
            ),
            relocation_target=relocation_target,
            remaining_relocation_time=(
                max(
                    0.0,
                    float(vehicle.get("relocation_time_left", 0.0) or 0.0),
                )
                if vehicle.get("relocation_time_left") is not None
                else (
                    _environment_metric(
                        env, "get_travel_time", location, relocation_target
                    )
                    if relocation_target is not None
                    else 0.0
                )
            ),
            stationary_duration_left=max(
                0.0,
                float(
                    vehicle.get(
                        "stationary_duration_left",
                        vehicle.get("stationary_duration", 0.0),
                    )
                    or 0.0
                ),
            ),
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
                service_phase=None,
                service_request_id=None,
                service_request_value=0.0,
                service_pickup=None,
                service_dropoff=None,
                remaining_pickup_time=0.0,
                remaining_trip_time=0.0,
                remaining_service_distance=0.0,
                service_deadline_remaining=0.0,
                charging_time_left=0.0,
                relocation_target=None,
                remaining_relocation_time=0.0,
                stationary_duration_left=0.0,
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
    success_structured_score: float | None = None
    rejection_structured_score: float = 0.0
    rejection_probability: float = 0.0
    human_response_mask: bool = False
    expected_response_anchor: bool = False
    response_model_hash: str | None = None
    metadata: tuple[tuple[str, float | int | str | bool | None], ...] = ()

    def __post_init__(self):
        from src.rejection_anchor import expected_structured_score
        success = self.structured_score if self.success_structured_score is None else self.success_structured_score
        expected = expected_structured_score(success, self.rejection_structured_score,
                                            self.rejection_probability, self.human_response_mask)
        if self.human_response_mask and (self.vehicle_type != 1 or self.action_type != ActionType.SERVICE
                                         or dict(self.metadata).get('continuing', False)):
            raise ValueError('Only unanswered EV service edges may have a response mask')
        if self.expected_response_anchor and abs(expected - self.structured_score) > 1e-5 * max(1., abs(expected)):
            raise ValueError('Snapshot structured score differs from its frozen expected anchor')

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
            "rejection_probability": self.rejection_probability,
            "human_response_mask": self.human_response_mask,
            "response_model_hash": self.response_model_hash,
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
    objective_cost_scale: int = 10_000
    objective_precision_mode: str = "integer_q_grid"

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
    oracle_rejection_probability: float
    acceptance_uniform: float
    accepted: bool
    rejected: bool
    rejection_reason: str | None
    request_snapshot: RequestSnapshot
    vehicle_snapshot: VehicleSnapshot
    predicted_rejection_probability: float | None = None
    response_model_hash: str | None = None


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
    rejection_event_id: str = ""
    transition_id: str = ""
    ultimately_served: bool = False
    repair_architecture: str = "ev_first"


def is_true_same_epoch_recourse(event: RecourseEvent) -> bool:
    return bool(event.residual_category == "rejected" and event.eligible
                and event.same_epoch_recourse_link and event.assigned
                and event.assigned_vehicle_type == 2
                and event.first_rejected_epoch is not None
                and event.assignment_epoch_id == event.first_rejected_epoch)


@dataclass(frozen=True)
class RewardLedger:
    ev_accepted_service: float = 0.0
    ev_rejection_penalty: float = 0.0
    ev_other: float = 0.0
    aev_rejected_repair_service: float = 0.0
    aev_unoffered_service: float = 0.0
    aev_other_service: float = 0.0
    aev_charging: float = 0.0
    aev_relocation: float = 0.0
    aev_waiting: float = 0.0
    aev_other: float = 0.0
    request_expiry_penalty: float = 0.0
    other_system_penalty: float = 0.0

    @property
    def stage1(self):
        return self.ev_accepted_service + self.ev_rejection_penalty + self.ev_other

    @property
    def stage2(self):
        return sum((self.aev_rejected_repair_service, self.aev_unoffered_service,
                    self.aev_other_service, self.aev_charging, self.aev_relocation,
                    self.aev_waiting, self.aev_other, self.request_expiry_penalty,
                    self.other_system_penalty))

    @property
    def system(self):
        return self.stage1 + self.stage2


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
    recourse_target_family: str = "auto"
    reward_ledger: RewardLedger | None = None
    committed_aev_edge_ids: tuple[str, ...] = ()
    repair_hold_aev_ids: tuple[int, ...] = ()
    repair_candidate_request_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay schema {self.schema_version}; expected {REPLAY_SCHEMA_VERSION}"
            )
        if self.mode not in {"integrated", "integrated_repair", "ev_first", "aev_first"}:
            raise ValueError(f"invalid transition mode: {self.mode}")
        from .config import canonical_variant, target_family
        object.__setattr__(self, "recourse_variant", canonical_variant(self.recourse_variant))
        family = target_family(self.recourse_variant, self.mode)
        if self.recourse_target_family not in {"auto", family}:
            raise ValueError("recourse target family does not match the physical/credit configuration")
        object.__setattr__(self, "recourse_target_family", family)
        if abs(self.reward_system - self.reward_ev - self.reward_aev) > 1e-6 * max(1., abs(self.reward_system)):
            raise ValueError("joint system reward does not reconcile with fleet rewards")
        if self.reward_ledger is not None:
            if abs(self.reward_ledger.system - self.reward_system) > 1e-6 * max(1., abs(self.reward_system)):
                raise ValueError("reward ledger does not reconcile with realized system reward")
            if (abs(self.reward_ledger.stage1 - self.reward_ev) > 1e-6 * max(1., abs(self.reward_ev))
                    or abs(self.reward_ledger.stage2 - self.reward_aev) > 1e-6 * max(1., abs(self.reward_aev))):
                raise ValueError("reward ledger does not reconcile with realized fleet rewards")
        for graph, action in ((self.ev_stage_graph, self.ev_joint_action), (self.aev_stage_graph, self.aev_joint_action)):
            if graph is None or action is None or not any(edge.response_model_hash for edge in graph.edges):
                continue
            expected = JointActionSnapshot.from_graph(graph, action.selected_edge_ids)
            if abs(action.structured_value - expected.structured_value) > 1e-5 * max(1., abs(expected.structured_value)):
                raise ValueError('Joint structured value is not the sum of selected frozen anchors')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def stage1_graph(self):
        return self.ev_stage_graph

    @property
    def stage2_graph(self):
        return self.aev_stage_graph

    @property
    def stage1_joint_action(self):
        return self.ev_joint_action

    @property
    def stage2_joint_action(self):
        return self.aev_joint_action


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


def _environment_metric(
    env: Any | None,
    name: str,
    origin: int,
    destination: int | None,
) -> float:
    if env is None or destination is None:
        return 0.0
    function = getattr(env, name, None)
    if callable(function):
        try:
            return float(function(int(origin), int(destination)))
        except (TypeError, ValueError, RuntimeError):
            pass
    return _environment_distance(env, origin, destination)


def _environment_distance(
    env: Any | None,
    origin: int,
    destination: int | None,
) -> float:
    if env is None or destination is None:
        return 0.0
    for name in ("get_distance_km", "_manhattan_distance_loc"):
        function = getattr(env, name, None)
        if callable(function):
            try:
                return float(function(int(origin), int(destination)))
            except (TypeError, ValueError, RuntimeError):
                pass
    return float(abs(int(destination) - int(origin)))
