"""Offer/vehicle-keyed exogenous draws for paired experiments only."""
from hashlib import sha256
import random
import numpy as np


def vehicle_uniform(env, vehicle_id, stream, *, numpy_fallback=False):
    if not getattr(env, 'common_random_numbers', False):
        return float(np.random.random() if numpy_fallback else random.random())
    seed = getattr(env, 'initial_random_seed', getattr(env, '_recourse_experiment_seed', 0))
    key = (seed, getattr(env, 'cumulative_episode_index', 0),
           getattr(env, 'episode_day_index', 0), int(round(env.current_time)), int(vehicle_id), str(stream))
    value = int.from_bytes(sha256(repr(key).encode()).digest()[:8], 'big')
    return (value >> 11) / float(1 << 53)


def vehicle_normal(
    env,
    vehicle_id,
    stream,
    std,
    *,
    request_id=None,
    attempt_index=0,
):
    """Policy-order-independent normal draw for reward-event CRN.

    The key includes the physical event rather than the order in which a
    method happens to execute vehicles. Non-CRN runs retain the historical
    global NumPy stream.
    """
    std = float(std or 0.0)
    if std == 0.0:
        return 0.0
    if not getattr(env, "common_random_numbers", False):
        return float(np.random.normal(0.0, std))
    seed = getattr(
        env, "initial_random_seed", getattr(env, "_recourse_experiment_seed", 0)
    )
    key = (
        int(seed or 0),
        str(getattr(env, "recourse_run_id", "")),
        int(getattr(env, "cumulative_episode_index", 0) or 0),
        int(getattr(env, "episode_day_index", 0) or 0),
        int(round(float(getattr(env, "current_time", 0.0)))),
        int(vehicle_id),
        str(stream),
        None if request_id is None else int(request_id),
        int(attempt_index),
    )
    digest = sha256(repr(key).encode()).digest()
    # Box-Muller from two independent 53-bit uniforms in one keyed digest.
    u1 = max((int.from_bytes(digest[:8], "big") >> 11) / float(1 << 53), 1e-15)
    u2 = (int.from_bytes(digest[8:16], "big") >> 11) / float(1 << 53)
    return float(std * np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2))
