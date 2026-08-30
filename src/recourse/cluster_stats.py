"""Cluster-aware summaries for repeated held-out days per trained policy."""
from __future__ import annotations

import math
from typing import Any


def _critical_value(count: int, confidence: float) -> float:
    if count <= 1:
        return 0.0
    try:
        from scipy.stats import t
        return float(t.ppf((1.0 + confidence) / 2.0, df=count - 1))
    except ImportError:
        return 1.96


def summarize_cluster_metric(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    cluster_fields: tuple[str, ...] = ("seed", "train_day"),
    confidence: float = 0.95,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        if row.get(metric) is None:
            continue
        key = tuple(row.get(field) for field in cluster_fields)
        groups.setdefault(key, []).append(float(row[metric]))
    values = [sum(group) / len(group) for group in groups.values()]
    if not values:
        raise ValueError(f"no observations for metric {metric!r}")
    count = len(values)
    mean = sum(values) / count
    variance = (
        sum((value - mean) ** 2 for value in values) / (count - 1)
        if count > 1 else 0.0
    )
    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(count)
    half_width = _critical_value(count, float(confidence)) * standard_error
    return {
        "count": count,
        "observation_count": sum(len(group) for group in groups.values()),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence": float(confidence),
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "cluster_fields": list(cluster_fields),
        "cluster_keys": [list(key) for key in groups],
    }


def summarize_paired_cluster_difference(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    baseline: str,
    treatment: str,
    pair_fields: tuple[str, ...] = ("seed", "train_day", "day_id"),
    cluster_fields: tuple[str, ...] = ("seed", "train_day"),
    confidence: float = 0.95,
) -> dict[str, Any]:
    pairs: dict[tuple[Any, ...], dict[str, float]] = {}
    metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method not in {baseline, treatment} or row.get(metric) is None:
            continue
        key = tuple(row.get(field) for field in pair_fields)
        if method in pairs.setdefault(key, {}):
            raise ValueError(f"duplicate {method} observation for pair {key}")
        pairs[key][method] = float(row[metric])
        metadata[key] = {
            field: row.get(field)
            for field in set(pair_fields) | set(cluster_fields)
        }
    differences = []
    for key, values in pairs.items():
        if baseline not in values or treatment not in values:
            continue
        row = dict(metadata[key])
        row["difference"] = values[treatment] - values[baseline]
        differences.append(row)
    if not differences:
        raise ValueError(f"no complete {baseline}->{treatment} pairs for {metric}")
    result = summarize_cluster_metric(
        differences,
        "difference",
        cluster_fields=cluster_fields,
        confidence=confidence,
    )
    result.update({
        "metric": metric,
        "baseline": baseline,
        "treatment": treatment,
        "pair_fields": list(pair_fields),
        "paired_day_count": len(differences),
    })
    return result
