"""Release recomputable caches at episode/run boundaries, never replay samples."""
import gc
import os
import sys


class JsonArrayDigest:
    """Hash json.dumps(list_of_values) without retaining the history or JSON."""
    def __init__(self):
        import hashlib
        self.digest = hashlib.sha256(b'[')
        self.count = 0

    def extend(self, values):
        import json
        for value in values:
            if self.count:
                self.digest.update(b', ')
            self.digest.update(json.dumps(value).encode())
            self.count += 1

    def hexdigest(self):
        final = self.digest.copy()
        final.update(b']')
        return final.hexdigest()


def release_training_caches(*value_functions):
    seen = set()
    for value in value_functions:
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        for name in ('_graph_cache', '_target_graph_cache',
                     '_graph_cache_key', '_target_graph_cache_key'):
            if hasattr(value, name):
                setattr(value, name, None)
        cache = getattr(value, '_target_component_cache', None)
        if cache is not None:
            cache.clear()
        for name in ('_last_actor_loss_tensor', '_last_alpha_term_tensor'):
            tensor = getattr(value, name, None)
            if tensor is not None:
                setattr(value, name, tensor.detach())
        replay = getattr(value, 'joint_replay_buffer', None)
        clear = getattr(replay, 'clear_hot_cache', None)
        if clear is not None:
            clear()
    # Full cyclic GC is deliberately outside the per-step/per-edge hot path.
    collected = gc.collect()
    if sys.platform.startswith('linux'):
        import ctypes
        trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
    import torch
    # Do not initialize a GPU on CPU runs or change the active GPU.
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return collected


def ram_usage_gib():
    """Current host RSS, not CUDA memory or the process lifetime peak."""
    if sys.platform.startswith('linux'):
        with open('/proc/self/statm') as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE') / 1024**3
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**3
    except ImportError:
        return None


def check_training_memory(step, *value_functions):
    interval = max(1, int(os.environ.get('ICAPS_RAM_LOG_EVERY', '100')))
    if int(step) % interval:
        return None
    rss = ram_usage_gib()
    limit = float(os.environ.get('ICAPS_RAM_LIMIT_GB', '0'))
    if rss is not None and limit > 0 and rss > limit:
        release_training_caches(*value_functions)
        rss = ram_usage_gib()
        if rss is not None and rss > limit:
            raise MemoryError(f'Host RAM guard: RSS={rss:.2f} GiB exceeds '
                              f'ICAPS_RAM_LIMIT_GB={limit:g}; replay was not discarded')
    replays = {id(r): r for v in value_functions
               if (r := getattr(v, 'joint_replay_buffer', None)) is not None}
    detail = ', '.join(f'{getattr(r, "storage", "memory")}:{len(r)} rows' for r in replays.values())
    if rss is not None:
        print(f'[RAM] step={step} RSS={rss:.2f} GiB; joint replay {detail or "none"}', flush=True)
    return rss
