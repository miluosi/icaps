"""Isolated host-RAM replay microbenchmark; does not simulate fleet performance."""
import argparse
from dataclasses import fields, make_dataclass, replace
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))


def worker(mode, rows, edges):
    from src.memory_lifecycle import ram_usage_gib
    from src.recourse.disk_replay import DiskPrioritizedJointReplayBuffer
    from src.recourse.replay import PrioritizedJointReplayBuffer
    from src.recourse.types import FeasibleEdgeSnapshot
    from test_recourse_must_fix import _graph, _transition
    # Historical objects had an instance dictionary, with the same fields.
    legacy = make_dataclass('LegacyEdge', [(f.name, f.type, f) for f in fields(FeasibleEdgeSnapshot)], frozen=True)
    gc.collect()
    baseline = ram_usage_gib()
    rss = []
    store_seconds = 0.
    with tempfile.TemporaryDirectory(prefix='icaps-replay-bench-') as directory:
        replay = (DiskPrioritizedJointReplayBuffer(capacity=rows, seed=7, directory=directory)
                  if mode == 'disk' else PrioritizedJointReplayBuffer(capacity=rows, seed=7))
        for index in range(rows):
            graph = _graph(str(index), stage=1, vehicle_id=0, vehicle_type=1)
            prototype = graph.edges[0]
            if mode == 'memory_legacy':
                prototype = legacy(**{f.name: getattr(prototype, f.name) for f in fields(prototype)})
            graph = replace(graph, edges=tuple(replace(prototype,
                edge_id=f'{index}:{i}', collection_score=float(i), structured_score=float(i),
                target_distance=float(i % 37), post_action_distance=float(i % 71)) for i in range(edges)))
            transition = _transition(str(index), sequence=index, ev_graph=graph)
            started = time.perf_counter()
            replay.add(transition)
            store_seconds += time.perf_counter() - started
            del transition, graph, prototype
            if index in {0, rows // 2, rows - 1}:
                rss.append([index + 1, ram_usage_gib()])
        before = ram_usage_gib()
        started = time.perf_counter()
        sample = replay.sample(8)
        checksum = sum(sum(e.structured_score for e in r.ev_stage_graph.edges)
                       for r in sample.transitions)
        sample_seconds = time.perf_counter() - started
        disk_bytes = sum(p.stat().st_size for p in Path(directory).rglob('*.pkl.gz'))
        print(json.dumps(dict(mode=mode, rows=rows, edges_per_row=edges,
            baseline_rss_gib=baseline, retained_rss_gib=before,
            retained_delta_mib=(before-baseline)*1024 if baseline is not None else None,
            store_seconds=store_seconds, mean_store_ms=store_seconds/rows*1000,
            sample_8_seconds=sample_seconds, checksum=checksum,
            disk_bytes=disk_bytes, progress_rss_gib=rss)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=40)
    parser.add_argument('--edges', type=int, default=4000)
    parser.add_argument('--worker', choices=['memory_legacy', 'memory', 'disk'])
    parser.add_argument('--output', default='results/memory_audit_20261004/replay_ram.json')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.rows, args.edges)
        return
    results = []
    for mode in ('memory_legacy', 'memory', 'disk'):
        process = subprocess.run([sys.executable, __file__, '--worker', mode,
            '--rows', str(args.rows), '--edges', str(args.edges)],
            cwd=ROOT, capture_output=True, text=True, check=True)
        result = json.loads(process.stdout.splitlines()[-1])
        results.append(result)
        print(json.dumps(result), flush=True)
    assert len({row['checksum'] for row in results}) == 1
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
