"""Request-level lifecycle accounting shared by both environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import (
    OfferAttempt,
    OutcomeSummary,
    RecourseEvent,
    RejectionOutcomeSnapshot,
    ResidualObservation,
    RequestSnapshot,
    VehicleSnapshot,
)


@dataclass
class _MutableLifecycle:
    request_id: int
    repair_architecture: str = "ev_first"
    rejection_event_id: str = ""
    transition_id: str = ""
    rejected_epoch_id: int | None = None
    residual_category: str = "other"
    residual_epoch_id: int | None = None
    eligible: bool = False
    assigned: bool = False
    picked_up: bool = False
    completed: bool = False
    assigned_vehicle_id: int | None = None
    assigned_vehicle_type: int | None = None
    assignment_epoch_id: int | None = None
    same_epoch_recourse_link: bool = False
    pickup_vehicle_id: int | None = None
    pickup_vehicle_type: int | None = None
    pickup_epoch_id: int | None = None
    completion_vehicle_id: int | None = None
    completion_vehicle_type: int | None = None
    completion_epoch_id: int | None = None
    expired: bool = False
    cancelled: bool = False
    ultimately_served: bool = False
    residual_observations: list[ResidualObservation] = field(default_factory=list)


class RequestLifecycleTracker:
    """Own offer events and non-overloaded recovery outcomes.

    Matching is exclusively based on integer ``epoch_id``.  Simulator time is
    retained in state snapshots for features, but never determines whether an
    assignment belongs to the same decision epoch.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._requests: dict[int, _MutableLifecycle] = {}
        self._rejection_events: dict[
            tuple[int, int], _MutableLifecycle
        ] = {}
        self._offers: list[OfferAttempt] = []
        self._offer_counts: dict[tuple[int, int, int], int] = {}
        # Separate from EV-rejection events: Samitha can also repair unoffered
        # requests. Initial integrated commitments never enter this ledger.
        self._integrated_repairs: dict[int, _MutableLifecycle] = {}

    def next_attempt_index(self, epoch_id: int, ev_id: int, request_id: int) -> int:
        return int(self._offer_counts.get((int(epoch_id), int(ev_id), int(request_id)), 0))

    def record_offer(
        self,
        *,
        transition_id: str,
        epoch_id: int,
        request: Any,
        ev_id: int,
        vehicle: dict[str, Any],
        acceptance_probability: float,
        acceptance_uniform: float,
        accepted: bool,
        rejection_reason: str | None = None,
        selected_by_stage1: bool = True,
        predicted_rejection_probability: float | None = None,
        response_model_hash: str | None = None,
    ) -> OfferAttempt:
        request_snapshot = (
            request
            if isinstance(request, RequestSnapshot)
            else RequestSnapshot.from_request(request)
        )
        vehicle_snapshot = (
            vehicle
            if isinstance(vehicle, VehicleSnapshot)
            else VehicleSnapshot.from_vehicle(int(ev_id), vehicle)
        )
        key = (int(epoch_id), int(ev_id), request_snapshot.request_id)
        attempt_index = self._offer_counts.get(key, 0)
        self._offer_counts[key] = attempt_index + 1
        offer = OfferAttempt(
            offer_id=(
                f"{transition_id}:offer:{int(epoch_id)}:{int(ev_id)}:"
                f"{request_snapshot.request_id}:{int(attempt_index)}"
            ),
            transition_id=str(transition_id),
            epoch_id=int(epoch_id),
            attempt_index=int(attempt_index),
            request_id=request_snapshot.request_id,
            ev_id=int(ev_id),
            selected_by_stage1=bool(selected_by_stage1),
            oracle_rejection_probability=1.0 - float(acceptance_probability),
            acceptance_uniform=float(acceptance_uniform),
            accepted=bool(accepted),
            rejected=not bool(accepted),
            rejection_reason=None if accepted else (rejection_reason or "driver_reject"),
            request_snapshot=request_snapshot,
            vehicle_snapshot=vehicle_snapshot,
            predicted_rejection_probability=predicted_rejection_probability,
            response_model_hash=response_model_hash,
        )
        self._offers.append(offer)
        state = self._requests.setdefault(
            request_snapshot.request_id,
            _MutableLifecycle(request_id=request_snapshot.request_id),
        )
        if offer.rejected:
            if state.rejected_epoch_id is None:
                state.rejected_epoch_id = int(epoch_id)
            state.residual_category = "rejected"
            event_key = (request_snapshot.request_id, int(epoch_id))
            rejection_event = self._rejection_events.setdefault(
                event_key,
                _MutableLifecycle(
                    request_id=request_snapshot.request_id,
                    rejection_event_id=(
                        f"{transition_id}:rejection:{request_snapshot.request_id}:"
                        f"{int(epoch_id)}"
                    ),
                    transition_id=str(transition_id),
                    rejected_epoch_id=int(epoch_id),
                    residual_category="rejected",
                    residual_epoch_id=int(epoch_id),
                ),
            )
            rejection_event.residual_category = "rejected"
        return offer

    def mark_residual(
        self,
        request_id: int,
        *,
        epoch_id: int,
        category: str,
        eligible: bool,
        repair_architecture: str = "ev_first",
    ) -> None:
        if category not in {"rejected", "unoffered", "other"}:
            raise ValueError(f"invalid residual category: {category}")
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        if category == "rejected" and state.rejected_epoch_id is None:
            state.rejected_epoch_id = int(epoch_id)
        observation = ResidualObservation(
            request_id=int(request_id),
            epoch_id=int(epoch_id),
            category=str(category),
            eligible=bool(eligible),
        )
        if observation not in state.residual_observations:
            state.residual_observations.append(observation)
        state.residual_category = category
        state.repair_architecture = repair_architecture
        state.residual_epoch_id = int(epoch_id)
        state.eligible = bool(eligible)
        if category != "rejected":
            latest_rejection = max(
                (
                    event
                    for (candidate_request_id, candidate_epoch), event
                    in self._rejection_events.items()
                    if candidate_request_id == int(request_id)
                    and candidate_epoch <= int(epoch_id)
                ),
                key=lambda event: _epoch_or_default(event.rejected_epoch_id),
                default=None,
            )
            if (
                latest_rejection is not None
                and observation not in latest_rejection.residual_observations
            ):
                latest_rejection.residual_observations.append(observation)
        if category == "rejected":
            rejection_event = self._rejection_events.setdefault(
                (int(request_id), int(epoch_id)),
                _MutableLifecycle(
                    request_id=int(request_id),
                    rejection_event_id=(
                        f"rejection:{int(request_id)}:{int(epoch_id)}"
                    ),
                    rejected_epoch_id=int(epoch_id),
                ),
            )
            if observation not in rejection_event.residual_observations:
                rejection_event.residual_observations.append(observation)
            rejection_event.residual_category = "rejected"
            rejection_event.repair_architecture = repair_architecture
            rejection_event.residual_epoch_id = int(epoch_id)
            rejection_event.eligible = bool(eligible)

    def record_aev_assignment(
        self,
        request_id: int,
        *,
        vehicle_id: int,
        epoch_id: int,
        vehicle_type: int = 2,
    ) -> bool:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        if int(vehicle_type) != 2:
            return False
        rejection_event = self._rejection_events.get(
            (int(request_id), int(epoch_id))
        )
        if rejection_event is None:
            rejection_event = max(
                (
                    event
                    for (candidate_request_id, _), event
                    in self._rejection_events.items()
                    if candidate_request_id == int(request_id)
                    and not event.assigned
                    and _epoch_or_default(event.rejected_epoch_id) < int(epoch_id)
                ),
                key=lambda event: _epoch_or_default(event.rejected_epoch_id),
                default=None,
            )
        new_assignment = bool(
            rejection_event is not None and not rejection_event.assigned
        )
        if new_assignment:
            rejection_event.assigned = True
            rejection_event.assigned_vehicle_id = int(vehicle_id)
            rejection_event.assigned_vehicle_type = int(vehicle_type)
            rejection_event.assignment_epoch_id = int(epoch_id)
            rejection_event.same_epoch_recourse_link = (
                rejection_event.rejected_epoch_id == int(epoch_id)
            )
        same_epoch_recourse = (
            new_assignment
            and rejection_event is not None
            and rejection_event.rejected_epoch_id == int(epoch_id)
            and any(
                observation.category == "rejected"
                for observation in rejection_event.residual_observations
            )
        )
        if new_assignment:
            state.assigned = True
            state.assigned_vehicle_id = int(vehicle_id)
            state.assigned_vehicle_type = int(vehicle_type)
            state.assignment_epoch_id = int(epoch_id)
            state.same_epoch_recourse_link = bool(same_epoch_recourse)
            return bool(same_epoch_recourse)
        return False

    def record_integrated_repair_assignment(self, request_id: int, *, vehicle_id: int, epoch_id: int) -> None:
        self._integrated_repairs[int(request_id)] = _MutableLifecycle(
            request_id=int(request_id), repair_architecture="integrated_repair",
            assigned=True, assigned_vehicle_id=int(vehicle_id), assigned_vehicle_type=2,
            assignment_epoch_id=int(epoch_id),
        )

    def record_pickup(
        self,
        request_id: int,
        *,
        vehicle_id: int,
        epoch_id: int,
        vehicle_type: int = 2,
    ) -> bool:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        repair = self._integrated_repairs.get(int(request_id))
        if repair is not None and repair.assigned_vehicle_id == int(vehicle_id) and int(vehicle_type) == 2:
            repair.picked_up = True
            repair.pickup_epoch_id = int(epoch_id)
        candidates = [
            event
            for (candidate_request_id, _), event in self._rejection_events.items()
            if candidate_request_id == int(request_id)
            and event.pickup_epoch_id is None
            and event.assigned_vehicle_id == int(vehicle_id)
            and event.assigned_vehicle_type == int(vehicle_type)
        ]
        rejection_event = max(
            candidates,
            key=lambda event: _epoch_or_default(event.rejected_epoch_id),
            default=None,
        )
        if rejection_event is None:
            return False
        state.pickup_vehicle_id = int(vehicle_id)
        state.pickup_vehicle_type = int(vehicle_type)
        state.pickup_epoch_id = int(epoch_id)
        is_linked_aev = (
            int(vehicle_type) == 2
            and state.assigned_vehicle_type == 2
            and state.assigned_vehicle_id == int(vehicle_id)
        )
        state.picked_up = bool(is_linked_aev)
        rejection_event.pickup_vehicle_id = int(vehicle_id)
        rejection_event.pickup_vehicle_type = int(vehicle_type)
        rejection_event.pickup_epoch_id = int(epoch_id)
        rejection_event.picked_up = bool(is_linked_aev)
        return bool(is_linked_aev)

    def record_completion(
        self,
        request_id: int,
        *,
        epoch_id: int,
        vehicle_id: int | None = None,
        vehicle_type: int | None = None,
    ) -> bool:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        if state.completion_epoch_id is not None:
            return False
        state.completion_vehicle_id = (
            state.assigned_vehicle_id
            if vehicle_id is None
            else int(vehicle_id)
        )
        state.completion_vehicle_type = (
            state.assigned_vehicle_type
            if vehicle_type is None
            else int(vehicle_type)
        )
        repair = self._integrated_repairs.get(int(request_id))
        if (repair is not None and repair.assigned_vehicle_id == state.completion_vehicle_id
                and state.completion_vehicle_type == 2):
            repair.completed = True
            repair.completion_epoch_id = int(epoch_id)
        state.completed = bool(
            state.rejected_epoch_id is not None
            and state.assigned_vehicle_type == 2
            and state.assigned_vehicle_id == state.completion_vehicle_id
            and state.completion_vehicle_type == 2
        )
        state.completion_epoch_id = int(epoch_id)
        state.ultimately_served = True
        linked_completion = False
        for (candidate_request_id, _), event in self._rejection_events.items():
            if candidate_request_id != int(request_id):
                continue
            event.completion_vehicle_id = state.completion_vehicle_id
            event.completion_vehicle_type = state.completion_vehicle_type
            event.completion_epoch_id = int(epoch_id)
            event.ultimately_served = True
            event.completed = bool(
                event.assigned_vehicle_type == 2
                and event.assigned_vehicle_id == event.completion_vehicle_id
                and event.completion_vehicle_type == 2
            )
            linked_completion = linked_completion or event.completed
        return bool(linked_completion)

    def record_expiry(self, request_id: int) -> None:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        state.expired = True
        for (candidate_request_id, _), event in self._rejection_events.items():
            if candidate_request_id == int(request_id):
                event.expired = True

    def record_cancellation(self, request_id: int) -> None:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        state.cancelled = True
        for (candidate_request_id, _), event in self._rejection_events.items():
            if candidate_request_id == int(request_id):
                event.cancelled = True

    def rejection_outcome(
        self,
        *,
        transition_id: str | None = None,
        epoch_id: int | None = None,
    ) -> RejectionOutcomeSnapshot:
        offers = self._offers
        if transition_id is not None:
            offers = [offer for offer in offers if offer.transition_id == transition_id]
        if epoch_id is not None:
            offers = [offer for offer in offers if offer.epoch_id == int(epoch_id)]
        return RejectionOutcomeSnapshot(tuple(offers))

    def outcome_summary(self, *, epoch_id: int | None = None) -> OutcomeSummary:
        rows = list(self._rejection_events.values())
        rows.extend(
            state
            for state in self._requests.values()
            if state.rejected_epoch_id is None
            and state.residual_epoch_id is not None
            and state.residual_category in {"unoffered", "other"}
        )
        first_rejected_epoch = {
            request_id: min(
                event_epoch
                for candidate_request_id, event_epoch in self._rejection_events
                if candidate_request_id == request_id
            )
            for request_id, _ in self._rejection_events
        }
        events = []
        for state in sorted(
            rows,
            key=lambda item: (
                item.request_id,
                _epoch_or_default(
                    item.rejected_epoch_id
                    if item.rejected_epoch_id is not None
                    else item.residual_epoch_id
                ),
            ),
        ):
            event_epoch = int(
                state.rejected_epoch_id
                if state.rejected_epoch_id is not None
                else _epoch_or_default(state.residual_epoch_id)
            )
            if epoch_id is not None and event_epoch != int(epoch_id):
                continue
            events.append(
                RecourseEvent(
                    request_id=state.request_id,
                    epoch_id=event_epoch,
                    residual_category=(
                        "rejected"
                        if state.rejected_epoch_id is not None
                        else state.residual_category
                    ),
                    eligible=(
                        state.eligible
                        or any(
                            observation.category == "rejected"
                            and observation.eligible
                            for observation in state.residual_observations
                        )
                    ),
                    assigned=state.assigned,
                    picked_up=state.picked_up,
                    completed=state.completed,
                    assigned_vehicle_id=state.assigned_vehicle_id,
                    assignment_epoch_id=state.assignment_epoch_id,
                    pickup_epoch_id=state.pickup_epoch_id,
                    completion_epoch_id=state.completion_epoch_id,
                    expired=state.expired,
                    cancelled=state.cancelled,
                    residual_observations=tuple(state.residual_observations),
                    first_rejected_epoch=first_rejected_epoch.get(
                        state.request_id
                    ),
                    assigned_vehicle_type=state.assigned_vehicle_type,
                    pickup_vehicle_id=state.pickup_vehicle_id,
                    pickup_vehicle_type=state.pickup_vehicle_type,
                    completion_vehicle_id=state.completion_vehicle_id,
                    completion_vehicle_type=state.completion_vehicle_type,
                    same_epoch_recourse_link=state.same_epoch_recourse_link,
                    rejection_event_id=state.rejection_event_id,
                    transition_id=state.transition_id,
                    ultimately_served=state.ultimately_served,
                    repair_architecture=state.repair_architecture,
                )
            )
        return OutcomeSummary(tuple(events))

    def metrics(self) -> dict[str, float | int]:
        events = self.outcome_summary().events
        rejected = [event for event in events if event.residual_category == "rejected"]
        assigned = sum(event.same_epoch_recourse_link for event in rejected)
        picked_up = sum(event.picked_up for event in rejected)
        completed = sum(event.completed for event in rejected)
        denominator = max(1, len(rejected))
        eligible_rejected = [event for event in rejected if event.eligible]
        eligible_denominator = max(1, len(eligible_rejected))
        eligible_assigned = sum(
            event.same_epoch_recourse_link for event in eligible_rejected
        )
        eligible_pickup = sum(event.picked_up for event in eligible_rejected)
        eligible_completion = sum(event.completed for event in eligible_rejected)
        later_aev_rescue = sum(
            event.assigned
            and event.assigned_vehicle_type == 2
            and not event.same_epoch_recourse_link
            for event in rejected
        )
        later_ev_completion = sum(
            event.completion_vehicle_type == 1 for event in rejected
        )
        assignment_delays = [
            int(event.assignment_epoch_id) - int(event.epoch_id)
            for event in rejected
            if event.same_epoch_recourse_link
            and event.assignment_epoch_id is not None
        ]
        pickup_delays = [
            int(event.pickup_epoch_id) - int(event.epoch_id)
            for event in rejected
            if event.picked_up and event.pickup_epoch_id is not None
        ]
        completion_delays = [
            int(event.completion_epoch_id) - int(event.epoch_id)
            for event in rejected
            if event.completed and event.completion_epoch_id is not None
        ]
        later_aev_rescue_delays = [
            int(event.assignment_epoch_id) - int(event.epoch_id)
            for event in rejected
            if event.assigned
            and event.assigned_vehicle_type == 2
            and not event.same_epoch_recourse_link
            and event.assignment_epoch_id is not None
        ]
        later_ev_completion_delays = [
            int(event.completion_epoch_id) - int(event.epoch_id)
            for event in rejected
            if event.completion_vehicle_type == 1
            and event.completion_epoch_id is not None
        ]
        ultimate_service_delays = [
            int(event.completion_epoch_id) - int(event.epoch_id)
            for event in rejected
            if event.ultimately_served and event.completion_epoch_id is not None
        ]
        accepted_offers = sum(offer.accepted for offer in self._offers)
        rejected_offers = sum(offer.rejected for offer in self._offers)
        return {
            "samitha_repair_pickup_count": sum(e.picked_up for e in self._integrated_repairs.values()),
            "samitha_repair_completion_count": sum(e.completed for e in self._integrated_repairs.values()),
            "ev_offer_count": len(self._offers),
            "ev_accepted_offer_count": accepted_offers,
            "ev_rejected_offer_count": rejected_offers,
            "ev_acceptance_rate": (
                accepted_offers / len(self._offers) if self._offers else 0.0
            ),
            "ev_rejection_rate": (
                rejected_offers / len(self._offers) if self._offers else 0.0
            ),
            "rejected_residual_count": len(rejected),
            "eligible_rejected_residual_count": len(eligible_rejected),
            "unoffered_residual_count": sum(
                event.residual_category == "unoffered" for event in events
            ),
            "other_residual_count": sum(
                event.residual_category == "other" for event in events
            ),
            "same_epoch_aev_assignment_count": assigned,
            "later_aev_rescue_count": later_aev_rescue,
            "later_ev_completion_count": later_ev_completion,
            "not_same_epoch_aev_assignment_count": later_aev_rescue,
            "aev_pickup_after_rejection_count": picked_up,
            "completion_after_rejection_count": completed,
            "unrecovered_rejected_count": sum(
                not event.completed for event in rejected
            ),
            "not_recovered_by_recourse_count": sum(
                not event.completed for event in rejected
            ),
            "ultimately_unserved_count": sum(
                not event.ultimately_served for event in rejected
            ),
            "recovery_rate_assignment": assigned / denominator if rejected else 0.0,
            "recovery_rate_pickup": picked_up / denominator if rejected else 0.0,
            "recovery_rate_completion": completed / denominator if rejected else 0.0,
            "conditional_recovery_rate_assignment": (
                eligible_assigned / eligible_denominator
                if eligible_rejected
                else 0.0
            ),
            "conditional_recovery_rate_pickup": (
                eligible_pickup / eligible_denominator
                if eligible_rejected
                else 0.0
            ),
            "conditional_recovery_rate_completion": (
                eligible_completion / eligible_denominator
                if eligible_rejected
                else 0.0
            ),
            "mean_assignment_recovery_delay": _mean(assignment_delays),
            "median_assignment_recovery_delay": _quantile(assignment_delays, 0.5),
            "p90_assignment_recovery_delay": _quantile(assignment_delays, 0.9),
            "mean_pickup_recovery_delay": _mean(pickup_delays),
            "median_pickup_recovery_delay": _quantile(pickup_delays, 0.5),
            "p90_pickup_recovery_delay": _quantile(pickup_delays, 0.9),
            "mean_completion_recovery_delay": _mean(completion_delays),
            "median_completion_recovery_delay": _quantile(completion_delays, 0.5),
            "p90_completion_recovery_delay": _quantile(completion_delays, 0.9),
            "mean_later_aev_rescue_delay": _mean(later_aev_rescue_delays),
            "mean_later_ev_completion_delay": _mean(
                later_ev_completion_delays
            ),
            "mean_ultimate_service_delay": _mean(ultimate_service_delays),
        }

    def assert_reconciled(self) -> None:
        for event in self.outcome_summary().events:
            if (
                event.same_epoch_recourse_link
                and event.assignment_epoch_id != event.epoch_id
            ):
                raise AssertionError(
                    f"request {event.request_id} marked same-epoch recovery in epoch "
                    f"{event.epoch_id}, but assignment epoch is {event.assignment_epoch_id}"
                )
            if event.completed and event.completion_epoch_id is None:
                raise AssertionError(f"request {event.request_id} has no completion epoch")
            if event.same_epoch_recourse_link and event.assigned_vehicle_type != 2:
                raise AssertionError(
                    f"request {event.request_id} recovery is not assigned to an AEV"
                )
        if len(self._offers) != sum(offer.accepted for offer in self._offers) + sum(
            offer.rejected for offer in self._offers
        ):
            raise AssertionError("EV offer outcomes do not reconcile")


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _epoch_or_default(epoch_id: int | None) -> int:
    """Preserve epoch zero; only missing epochs sort before real events."""

    return -1 if epoch_id is None else int(epoch_id)


def _quantile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(probability)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
