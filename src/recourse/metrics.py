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
) -> dict[str, float | int | str]:
    """Compute a seed-clustered interval after averaging days within seed."""

    observations = [row for row in rows if row.get(metric) is not None]
    seed_groups: dict[Any, list[float]] = {}
    for row in observations:
        seed_groups.setdefault(row.get("seed", 0), []).append(float(row[metric]))
    values = [sum(group) / len(group) for group in seed_groups.values()]
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
        "observation_count": len(observations),
        "seed_count": len({row.get("seed") for row in rows if "seed" in row}),
        "day_count": len({row.get("day_id") for row in rows if "day_id" in row}),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence": float(confidence),
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "interval_unit": "seed_mean",
    }


def summarize_paired_crn_difference(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    baseline_variant: str,
    treatment_variant: str,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Return a seed-level paired CRN interval for treatment minus baseline."""

    by_key: dict[tuple[Any, Any], dict[str, float]] = {}
    for row in rows:
        if row.get(metric) is None:
            continue
        variant = str(row.get("recourse_variant", ""))
        if variant not in {baseline_variant, treatment_variant}:
            continue
        key = (row.get("seed", 0), row.get("day_id", ""))
        by_key.setdefault(key, {})[variant] = float(row[metric])
    seed_differences: dict[Any, list[float]] = {}
    for (seed, _day), values in by_key.items():
        if baseline_variant in values and treatment_variant in values:
            seed_differences.setdefault(seed, []).append(
                values[treatment_variant] - values[baseline_variant]
            )
    paired_rows = [
        {"seed": seed, "day_id": "paired", "difference": sum(values) / len(values)}
        for seed, values in seed_differences.items()
    ]
    if not paired_rows:
        raise ValueError("no complete paired CRN rows")
    result = summarize_metric_with_uncertainty(
        paired_rows, "difference", confidence=confidence
    )
    result.update(
        {
            "metric": metric,
            "baseline_variant": baseline_variant,
            "treatment_variant": treatment_variant,
        }
    )
    return result
