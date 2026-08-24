"""Lifecycle-derived metrics and reconciliation helpers."""

from __future__ import annotations

from typing import Any
import math

from .lifecycle import RequestLifecycleTracker


def recourse_metrics(tracker: RequestLifecycleTracker) -> dict[str, float | int]:
    tracker.assert_reconciled()
    return tracker.metrics()


def reconcile_episode(
    *,
    generated: int,
    active_end: int,
    completed: int,
    expired: int,
    cancelled: int = 0,
    other_terminal: int = 0,
) -> None:
    right = active_end + completed + expired + cancelled + other_terminal
    if int(generated) != int(right):
        raise AssertionError(
            "request lifecycle does not reconcile: "
            f"generated={generated}, accounted={right}"
        )


def summarize_metric_with_uncertainty(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Summarize independent seed/day results with a Student-t interval."""
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    if not values:
        raise ValueError(f"no observations for metric {metric!r}")
    count = len(values)
    mean = sum(values) / count
    variance = (
        sum((value - mean) ** 2 for value in values) / (count - 1)
        if count > 1
        else 0.0
    )
    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(count)
    try:
        from scipy.stats import t

        critical = (
            float(t.ppf((1.0 + float(confidence)) / 2.0, df=count - 1))
            if count > 1
            else 0.0
        )
    except ImportError:
        critical = 1.96 if count > 1 else 0.0
    half_width = critical * standard_error
    return {
        "count": count,
        "seed_count": len({row.get("seed") for row in rows if "seed" in row}),
        "day_count": len({row.get("day_id") for row in rows if "day_id" in row}),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence": float(confidence),
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
    }
