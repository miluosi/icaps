"""Method-specific mechanism contracts for production experiment runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import canonical_method


@dataclass(frozen=True)
class EventContractResult:
    method: str
    passed: bool
    checks: tuple[tuple[str, bool, float | int | str | None], ...]

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, passed, _value in self.checks if not passed)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "passed": self.passed,
            "checks": [
                {"name": name, "passed": passed, "value": value}
                for name, passed, value in self.checks
            ],
            "failures": list(self.failures),
        }


def _number(stats: Mapping, key: str) -> float:
    value = stats.get(key, 0)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def evaluate_method_event_contract(
    method: str, stats: Mapping
) -> EventContractResult:
    method = canonical_method(method)
    checks: list[tuple[str, bool, float | int | str | None]] = []

    def positive(name: str, key: str) -> None:
        value = _number(stats, key)
        checks.append((name, value > 0, value))

    def zero(name: str, key: str) -> None:
        value = _number(stats, key)
        checks.append((name, value == 0, value))

    if method == "no_repair":
        positive("integrated_stage0_graph_built", "integrated_stage0_graph_count")
        positive("service_candidate_present", "integrated_service_candidate_count")
        positive("exact_assignment_called", "exact_assignment_call_count")
        zero("hold_disabled", "hold_selected_count")
    elif method == "evfirst_no_rejection":
        positive("ev_offer_present", "ev_offer_count")
        zero("rejection_disabled", "ev_rejected_offer_count")
        zero("no_same_epoch_repair", "same_epoch_aev_assignment_count")
    elif method in {"evfirst_no_repair", "evfirst_no_repair_structured"}:
        positive("ev_offer_present", "ev_offer_count")
        positive("realized_rejection_present", "ev_rejected_offer_count")
        zero("no_same_epoch_repair", "same_epoch_aev_assignment_count")
        positive("aev_stage_built", "aev_stage_graph_count")
        if method == "evfirst_no_repair_structured":
            zero("aev_optimizer_frozen", "aev_follower_optimizer_steps")
            zero("aev_score_delta_zero", "aev_learned_score_difference_count")
        else:
            positive("aev_optimizer_updated", "aev_follower_optimizer_steps")
            positive("learned_score_changed", "aev_learned_score_difference_count")
    elif method == "repair_only":
        positive("eligible_rejection_present", "eligible_rejected_residual_count")
        positive("same_epoch_repair_present", "same_epoch_aev_assignment_count")
        zero("aev_optimizer_frozen", "aev_follower_optimizer_steps")
        zero("aev_score_delta_zero", "aev_learned_score_difference_count")
    elif method == "repair_learning":
        positive("eligible_rejection_present", "eligible_rejected_residual_count")
        positive("same_epoch_repair_present", "same_epoch_aev_assignment_count")
        positive("aev_optimizer_updated", "aev_follower_optimizer_steps")
        positive("learned_score_changed", "aev_learned_score_difference_count")
    elif method == "recourse_macro":
        positive("eligible_rejection_present", "eligible_rejected_residual_count")
        positive("same_epoch_repair_present", "same_epoch_aev_assignment_count")
        positive("macro_target_built", "macro_leader_target_count")
        positive("aev_optimizer_updated", "aev_follower_optimizer_steps")
    elif method == "recourse_nested_q2":
        positive("eligible_rejection_present", "eligible_rejected_residual_count")
        positive("same_epoch_repair_present", "same_epoch_aev_assignment_count")
        positive("nested_target_built", "nested_leader_target_count")
        positive("follower_provider_queried", "follower_target_query_count")
    elif method == "samitha":
        positive("hold_candidate_present", "hold_candidate_count")
        positive("hold_selected", "hold_selected_count")
        positive("repair_candidate_present", "repair_candidate_rejected_count")
        positive("repair_assignment_present", "samitha_repair_assignment_count")
        zero("committed_aev_not_reassigned", "committed_aev_reassignment_count")
    return EventContractResult(
        method=method,
        passed=all(passed for _name, passed, _value in checks),
        checks=tuple(checks),
    )


def assert_method_event_contract(method: str, stats: Mapping) -> EventContractResult:
    result = evaluate_method_event_contract(method, stats)
    if not result.passed:
        raise AssertionError(
            f"{result.method} event contract failed: {', '.join(result.failures)}"
        )
    return result
