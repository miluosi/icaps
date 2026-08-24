"""Request-level lifecycle accounting shared by both environments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import uuid

from .types import (
    OfferAttempt,
    OutcomeSummary,
    RecourseEvent,
    RejectionOutcomeSnapshot,
    RequestSnapshot,
    VehicleSnapshot,
)


@dataclass
class _MutableLifecycle:
    request_id: int
    rejected_epoch_id: int | None = None
    residual_category: str = "other"
    residual_epoch_id: int | None = None
    eligible: bool = False
    assigned: bool = False
    picked_up: bool = False
    completed: bool = False
    assigned_vehicle_id: int | None = None
    assignment_epoch_id: int | None = None
    pickup_epoch_id: int | None = None
    completion_epoch_id: int | None = None
    expired: bool = False
    cancelled: bool = False


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
        self._offers: list[OfferAttempt] = []
        self._offer_counts: dict[tuple[int, int, int], int] = {}

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
            offer_id=str(uuid.uuid4()),
            transition_id=str(transition_id),
            epoch_id=int(epoch_id),
            attempt_index=int(attempt_index),
            request_id=request_snapshot.request_id,
            ev_id=int(ev_id),
            selected_by_stage1=bool(selected_by_stage1),
            acceptance_probability=float(acceptance_probability),
            acceptance_uniform=float(acceptance_uniform),
            accepted=bool(accepted),
            rejected=not bool(accepted),
            rejection_reason=None if accepted else (rejection_reason or "driver_reject"),
            request_snapshot=request_snapshot,
            vehicle_snapshot=vehicle_snapshot,
        )
        self._offers.append(offer)
        state = self._requests.setdefault(
            request_snapshot.request_id,
            _MutableLifecycle(request_id=request_snapshot.request_id),
        )
        if offer.rejected:
            state.rejected_epoch_id = int(epoch_id)
            state.residual_category = "rejected"
        return offer

    def mark_residual(
        self,
        request_id: int,
        *,
        epoch_id: int,
        category: str,
        eligible: bool,
    ) -> None:
        if category not in {"rejected", "unoffered", "other"}:
            raise ValueError(f"invalid residual category: {category}")
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        if category == "rejected" and state.rejected_epoch_id is None:
            state.rejected_epoch_id = int(epoch_id)
        state.residual_category = category
        state.residual_epoch_id = int(epoch_id)
        state.eligible = bool(eligible)

    def record_aev_assignment(
        self,
        request_id: int,
        *,
        vehicle_id: int,
        epoch_id: int,
    ) -> bool:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        same_epoch_recourse = (
            state.rejected_epoch_id is not None
            and state.rejected_epoch_id == int(epoch_id)
            and state.residual_category == "rejected"
        )
        if same_epoch_recourse and not state.assigned:
            state.assigned = True
            state.assigned_vehicle_id = int(vehicle_id)
            state.assignment_epoch_id = int(epoch_id)
            return True
        return False

    def record_pickup(self, request_id: int, *, vehicle_id: int, epoch_id: int) -> bool:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        if state.rejected_epoch_id is None or state.picked_up:
            return False
        state.picked_up = True
        state.assigned_vehicle_id = int(vehicle_id)
        state.pickup_epoch_id = int(epoch_id)
        return True

    def record_completion(self, request_id: int, *, epoch_id: int) -> bool:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        if state.completed:
            return False
        state.completed = True
        state.completion_epoch_id = int(epoch_id)
        return True

    def record_expiry(self, request_id: int) -> None:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        state.expired = True

    def record_cancellation(self, request_id: int) -> None:
        state = self._requests.setdefault(
            int(request_id), _MutableLifecycle(request_id=int(request_id))
        )
        state.cancelled = True

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
        return OutcomeSummary(
            tuple(
                RecourseEvent(
                    request_id=state.request_id,
                    epoch_id=(
                        state.rejected_epoch_id
                        if state.rejected_epoch_id is not None
                        else (
                            state.residual_epoch_id
                            if state.residual_epoch_id is not None
                            else -1
                        )
                    ),
                    residual_category=state.residual_category,
                    eligible=state.eligible,
                    assigned=state.assigned,
                    picked_up=state.picked_up,
                    completed=state.completed,
                    assigned_vehicle_id=state.assigned_vehicle_id,
                    assignment_epoch_id=state.assignment_epoch_id,
                    pickup_epoch_id=state.pickup_epoch_id,
                    completion_epoch_id=state.completion_epoch_id,
                    expired=state.expired,
                    cancelled=state.cancelled,
                )
                for state in sorted(self._requests.values(), key=lambda item: item.request_id)
                if (
                    state.rejected_epoch_id is not None
                    or (
                        state.residual_epoch_id is not None
                        and state.residual_category in {"unoffered", "other"}
                    )
                )
                and (
                    epoch_id is None
                    or state.rejected_epoch_id == int(epoch_id)
                    or state.assignment_epoch_id == int(epoch_id)
                    or state.residual_epoch_id == int(epoch_id)
                )
            )
        )

    def metrics(self) -> dict[str, float | int]:
        events = self.outcome_summary().events
        rejected = [event for event in events if event.residual_category == "rejected"]
        assigned = sum(event.assigned for event in rejected)
        picked_up = sum(event.picked_up for event in rejected)
        completed = sum(event.completed for event in rejected)
        denominator = max(1, len(rejected))
        assignment_delays = [
            int(event.assignment_epoch_id) - int(event.epoch_id)
            for event in rejected
            if event.assignment_epoch_id is not None
        ]
        pickup_delays = [
            int(event.pickup_epoch_id) - int(event.epoch_id)
            for event in rejected
            if event.pickup_epoch_id is not None
        ]
        completion_delays = [
            int(event.completion_epoch_id) - int(event.epoch_id)
            for event in rejected
            if event.completion_epoch_id is not None
        ]
        accepted_offers = sum(offer.accepted for offer in self._offers)
        rejected_offers = sum(offer.rejected for offer in self._offers)
        return {
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
            "unoffered_residual_count": sum(
                event.residual_category == "unoffered" for event in events
            ),
            "other_residual_count": sum(
                event.residual_category == "other" for event in events
            ),
            "same_epoch_aev_assignment_count": assigned,
            "not_same_epoch_aev_assignment_count": len(rejected) - assigned,
            "aev_pickup_after_rejection_count": picked_up,
            "completion_after_rejection_count": completed,
            "unrecovered_rejected_count": sum(
                not event.completed for event in rejected
            ),
            "recovery_rate_assignment": assigned / denominator if rejected else 0.0,
            "recovery_rate_pickup": picked_up / denominator if rejected else 0.0,
            "recovery_rate_completion": completed / denominator if rejected else 0.0,
            "mean_assignment_recovery_delay": _mean(assignment_delays),
            "median_assignment_recovery_delay": _quantile(assignment_delays, 0.5),
            "p90_assignment_recovery_delay": _quantile(assignment_delays, 0.9),
            "mean_pickup_recovery_delay": _mean(pickup_delays),
            "median_pickup_recovery_delay": _quantile(pickup_delays, 0.5),
            "p90_pickup_recovery_delay": _quantile(pickup_delays, 0.9),
            "mean_completion_recovery_delay": _mean(completion_delays),
            "median_completion_recovery_delay": _quantile(completion_delays, 0.5),
            "p90_completion_recovery_delay": _quantile(completion_delays, 0.9),
        }

    def assert_reconciled(self) -> None:
        for event in self.outcome_summary().events:
            if event.assigned and event.assignment_epoch_id != event.epoch_id:
                raise AssertionError(
                    f"request {event.request_id} marked same-epoch recovery in epoch "
                    f"{event.epoch_id}, but assignment epoch is {event.assignment_epoch_id}"
                )
            if event.completed and event.completion_epoch_id is None:
                raise AssertionError(f"request {event.request_id} has no completion epoch")
        if len(self._offers) != sum(offer.accepted for offer in self._offers) + sum(
            offer.rejected for offer in self._offers
        ):
            raise AssertionError("EV offer outcomes do not reconcile")


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


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
