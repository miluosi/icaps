"""Synthetic Q-inference benchmark; excludes simulation, matrix generation and solve.

Compare original per-edge feature assembly in one batch with bounded bulk assembly.
CUDA peak numbers are real allocator measurements; CPU runs leave them null.
"""
import argparse
from datetime import datetime
import importlib
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch


def inputs_for_edges(count, vehicles, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(count)
    vids = idx % vehicles
    locations = vids % 81
    return dict(
        vehicle_ids=vids, vehicle_locations=locations,
        target_locations=rng.integers(0, 81, count),
        current_times=np.full(count, 120, dtype=np.float32),
        other_vehicles=np.full(count, vehicles), num_requests=np.full(count, 1000),
        battery_levels=rng.uniform(.2, .9, count).astype(np.float32),
        request_values=rng.uniform(0, 50, count).astype(np.float32),
        target_distances=rng.uniform(.1, 10, count).astype(np.float32),
        target_zoneids=np.zeros(count, dtype=int), vehicle_idle_times=np.zeros(count),
        action_type_ids=idx % 3 + 1,
        post_action_distances=rng.uniform(1, 20, count).astype(np.float32),
        post_action_durations=rng.integers(1, 20, count),
        post_action_locations=rng.integers(0, 81, count),
        post_action_zoneids=np.zeros(count, dtype=int),
        target_station_ids=np.full(count, -1),
        vehicle_neighbour_candidates={vid: [dict(node_type='zone', node_id=(vid+j)%81,
            target_location=(vid+j)%81, distance=float(j)) for j in range(1, 6)]
            for vid in range(vehicles)},
    )


def scalar_assembly(self, rows, vehicle_types, graph, sources, vids, locations):
    edges = [torch.cat((torch.tensor(row, dtype=torch.float32, device=self.device),
                       sources[int(vid)], self._graph_embedding_for_location(graph, loc)))
             for row, vid, loc in zip(rows, vids, locations)]
    weights = [graph['w_ev'] if typ == 1 else graph['w_aev'] for typ in vehicle_types]
    return torch.stack(edges), torch.stack(weights).unsqueeze(1), graph['baseline']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--edges', nargs='+', type=int, default=[1000, 10000, 50000])
    parser.add_argument('--vehicles', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min([args.vehicles, args.batch_size, args.repeats, *args.edges]) < 1:
        parser.error('sizes and repeats must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cls = importlib.import_module('src.ValueFunction_optimization_anchored_residual').PyTorchChargingValueFunction
    net = cls(grid_size=9, num_vehicles=args.vehicles, neighbour_number=5,
              device=args.device, qvalue_inference_batch_size=args.batch_size)
    net.training_step = 1000
    net.post_demand_predictor_trained = True
    # Queue predictor is untrained here: no real station state in this fixture.
    output = args.output or Path('results/qvalue_memory') / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(scope='synthetic Q inference only; single CPU thread/single GPU',
                  device=str(device), torch_version=torch.__version__,
                  cuda_available=torch.cuda.is_available(), arguments={**vars(args), 'output':str(output)}, rows=[])
    report['device_name'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    report['source_sha256'] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ('benchmark_qvalue_memory.py', 'src/qvalue_inference.py',
                     'src/ValueFunction_st_masac_gat.py',
                     'src/ValueFunction_st_masac_gat_post_demand.py')
    }
    print(f'Output: {output}', flush=True)
    warm = inputs_for_edges(min(100, min(args.edges)), args.vehicles, args.seed)
    net.batch_get_mixed_q_values(**warm)
    for count in args.edges:
        inputs = inputs_for_edges(count, args.vehicles, args.seed)
        reference = None
        for repeat in range(args.repeats):
            for method in (['legacy_style_full', 'bounded_bulk'] if repeat % 2 == 0
                           else ['bounded_bulk', 'legacy_style_full']):
                net._graph_cache_key = None
                net._graph_cache = None
                if hasattr(net, '_qvalue_safe_batch_size'):
                    del net._qvalue_safe_batch_size
                if method == 'legacy_style_full':
                    net.qvalue_inference_batch_size = count
                    net._assemble_edge_rows = MethodType(scalar_assembly, net)
                else:
                    net.qvalue_inference_batch_size = args.batch_size
                    if '_assemble_edge_rows' in net.__dict__:
                        del net._assemble_edge_rows
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                failure = None
                try:
                    values = np.asarray(net.batch_get_mixed_q_values(**inputs), dtype=np.float32)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                except torch.cuda.OutOfMemoryError:
                    failure = 'CUDA OOM even after batch reduction'
                    values = None
                elapsed = time.perf_counter() - start
                stats = dict(getattr(net, '_last_qvalue_inference_stats', {})) if values is not None else {}
                delta = None
                if values is not None:
                    if reference is None:
                        reference = values.copy()
                    delta = float(np.max(np.abs(values-reference)))
                    np.testing.assert_allclose(values, reference, atol=2e-5, rtol=2e-6)
                row = dict(method=method, edges=count, repeat=repeat, seconds=elapsed,
                           error=failure, max_abs_q_difference=delta, inference=stats,
                           baseline_retried=bool(method=='legacy_style_full' and stats.get('oom_retries')),
                           peak_allocated_bytes=(torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None),
                           peak_reserved_bytes=(torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None),
                           full_edge_feature_bytes=count*net.edge_dim*4,
                           configured_batch_feature_bytes=min(count,args.batch_size)*net.edge_dim*4)
                report['rows'].append(row)
                output.write_text(json.dumps(report, indent=2))
                print(json.dumps(row), flush=True)
    print(f'Saved: {output}', flush=True)


if __name__ == '__main__':
    main()
