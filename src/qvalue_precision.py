"""Shared precision policy for assignment Q-value matrices.

The assignment solvers use an integer scale as the canonical value grid.  At
the default scale of 10,000 every Q value has four decimal places.  Keeping
the rounding in one module prevents NYC, synthetic, training, and inference
paths from handing subtly different float32/float64 objectives to a solver.
"""

from __future__ import annotations

import numpy as np


def validate_qvalue_scale(scale: int) -> int:
    """Return a validated positive integer Q-value scale."""

    if isinstance(scale, bool):
        raise ValueError("Q-value scale must be a positive integer")
    value = int(scale)
    if value <= 0 or value != scale:
        raise ValueError("Q-value scale must be a positive integer")
    return value


def decimal_places_for_scale(scale: int) -> int | None:
    """Return the decimal count for a power-of-ten scale, otherwise None."""

    value = validate_qvalue_scale(scale)
    places = 0
    while value > 1 and value % 10 == 0:
        value //= 10
        places += 1
    return places if value == 1 else None


def quantize_qvalues(
    values: np.ndarray,
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(canonical_float64, scaled_int64)`` on the shared Q grid."""

    scale = validate_qvalue_scale(scale)
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Q-value matrix contains NaN or infinity")
    limit = np.iinfo(np.int64).max / scale
    if np.max(np.abs(array), initial=0.0) > limit:
        raise OverflowError("Q-value * scale exceeds int64 range")

    scaled = np.rint(array * scale).astype(np.int64)
    canonical = scaled.astype(np.float64) / float(scale)
    # Avoid signed-zero differences in diagnostics and serialization.
    canonical[canonical == 0.0] = 0.0
    return canonical, scaled


def round_qvalue_matrix(values: np.ndarray, scale: int) -> np.ndarray:
    """Round an assignment Q matrix to the configured grid as float64."""

    canonical, _ = quantize_qvalues(values, scale)
    return canonical


def qvalue_rounding_diagnostics(
    original: np.ndarray,
    canonical: np.ndarray,
    scale: int,
) -> dict[str, int | float | None]:
    """Summarize how much canonicalization changed a Q matrix."""

    scale = validate_qvalue_scale(scale)
    before = np.asarray(original, dtype=np.float64)
    after = np.asarray(canonical, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError("original and canonical Q matrices must have equal shapes")
    delta = np.abs(before - after)
    return {
        "qvalue_scale": scale,
        "qvalue_decimal_places": decimal_places_for_scale(scale),
        "qvalue_entries": int(before.size),
        "qvalue_rounded_entries": int(np.count_nonzero(delta)),
        "qvalue_rounding_max_abs": float(np.max(delta, initial=0.0)),
        "qvalue_rounding_mean_abs": float(np.mean(delta)) if delta.size else 0.0,
        "qvalue_rounding_per_action_bound": 0.5 / float(scale),
    }
