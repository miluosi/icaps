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
