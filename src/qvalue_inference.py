"""Bounded, sequential edge inference without silently replacing failed scores."""
import os

import numpy as np
import torch


def inference_batch_size(value=None):
    size = int(os.environ.get("ICAPS_QVALUE_BATCH_SIZE", "4096") if value is None else value)
    if size < 1:
        raise ValueError("Q-value inference batch size must be positive")
    return size


def bounded_edge_inference(size, batch_size, evaluate, *, device):
    """Evaluate slices in order; release failed frames before retrying CUDA OOM."""
    output = np.empty(size, dtype=np.float32)
    batch_size = min(size, inference_batch_size(batch_size)) if size else 1
    start = batches = retries = largest = 0
    while start < size:
        stop = min(size, start + batch_size)
        failed = False
        try:
            values = evaluate(slice(start, stop))
        except torch.cuda.OutOfMemoryError:
            if torch.device(device).type != "cuda" or batch_size <= 1:
                raise
            failed = True
        if failed:
            # The exception and its frame (including intermediate tensors) are
            # gone here.  Do not empty the allocator cache on successful batches.
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            retries += 1
            continue
        output[start:stop] = values
        largest = max(largest, stop - start)
        batches += 1
        start = stop
    return output, {"edges": size, "batches": batches,
                    "max_batch_edges": largest, "oom_retries": retries,
                    "effective_batch_size": batch_size}
