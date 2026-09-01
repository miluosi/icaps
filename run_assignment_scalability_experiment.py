"""Run paper-scale fleet/graph/latency sweeps on canonical Macro recourse."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from run_recourse_multiday_panel import FORMAL_METRICS
from src.recourse.cluster_stats import summarize_cluster_metric


ROOT = Path(__file__).resolve().parent
RUNTIME_METRICS = (
    'candidate_generation_runtime_seconds', 'neural_edge_scoring_runtime_seconds',
    'graph_serialization_runtime_seconds', 'graph_reduction_runtime_seconds',
    'exact_solve_runtime_seconds', 'total_decision_runtime_seconds',
    'decision_latency_p50_seconds', 'decision_latency_p90_seconds',
    'decision_latency_p95_seconds', 'decision_latency_p99_seconds',
    'feasible_edge_count', 'exact_original_edge_count',
    'exact_reduced_edge_count', 'graph_reduction_ratio',
    'process_peak_memory_mb', 'solver_fallback_count',
    'max_solver_objective_gap', 'reward', 'completed_orders',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet-sizes', nargs='+', type=int,
                        default=[100, 250, 500, 1000, 2000, 3000])
    parser.add_argument('--aev-share', type=float, default=0.5)
    parser.add_argument('--backends', nargs='+',
                        choices=['primal_dual', 'ortools', 'gurobi_network'],
                        default=['primal_dual'])
    parser.add_argument('--reductions', nargs='+', choices=['on', 'off'],
                        default=['on', 'off'])
    parser.add_argument('--learner-variant',
                        choices=['structured_myopic', 'integrated_directq', 'optimization_anchored_residual'],
                        default='optimization_anchored_residual')
    parser.add_argument('--train-days', nargs='+', required=True)
    parser.add_argument('--test-days', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--parquet-path', type=Path, required=True)
    parser.add_argument('--epoch-length', type=float, default=30.0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--train-every', type=int, default=10)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--energy-model', choices=['general_charging', 'fixed_swap'], required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.fleet_sizes)) != len(args.fleet_sizes) or min(args.fleet_sizes) <= 1:
        parser.error('fleet sizes must be unique and greater than one')
    if not 0 < args.aev_share < 1:
        parser.error('aev-share must be in (0,1)')
    if len(set(args.backends)) != len(args.backends) or len(set(args.reductions)) != len(args.reductions):
        parser.error('backends and reductions must be unique')
    if args.energy_model == 'fixed_swap':
        parser.error('fixed_swap is not implemented')
    args.output_dir = (args.output_dir or ROOT / 'results/assignment_scalability' /
                       datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    args.parquet_path = args.parquet_path.resolve()
    return args


def jobs(args):
    result = []
    for fleet_size in args.fleet_sizes:
        num_ev = fleet_size - max(1, min(fleet_size - 1, round(fleet_size * args.aev_share)))
        for backend in args.backends:
            for reduction in args.reductions:
                tag = f'n-{fleet_size}-backend-{backend}-reduction-{reduction}'
                output = args.output_dir / tag
                command = [
                    sys.executable, str(ROOT / 'run_recourse_multiday_panel.py'),
                    '--methods', 'recourse_macro', '--learner-variant', args.learner_variant,
                    '--state-variant', 'joint_state_separate_critics',
                    '--train-days', *args.train_days, '--test-days', *args.test_days,
                    '--seeds', *map(str, args.seeds), '--parquet-path', str(args.parquet_path),
                    '--num-vehicles', str(fleet_size), '--num-ev', str(num_ev),
                    '--mcmf-backend', backend,
                    '--graph-reduction' if reduction == 'on' else '--no-graph-reduction',
                    '--epoch-length', str(args.epoch_length), '--batch-size', str(args.batch_size),
                    '--train-every', str(args.train_every), '--workers', str(args.workers),
                    '--event-contract-mode', 'record', '--energy-model', args.energy_model,
                    '--output-dir', str(output),
                ]
                if args.smoke_steps is not None:
                    command += ['--smoke-steps', str(args.smoke_steps)]
                result.append((fleet_size, num_ev, backend, reduction, command, output))
    return result


def aggregate(args, planned):
    rows = []
    summaries = {}
    for fleet_size, num_ev, backend, reduction, _command, output in planned:
        summary = json.loads((output / 'panel_summary.json').read_text())
        tag = f'n-{fleet_size}-backend-{backend}-reduction-{reduction}'
        summaries[tag] = summary
        for source in summary['rows']:
            row = dict(source)
            row.update(
                fleet_size=fleet_size, num_ev=num_ev,
                num_aev=fleet_size - num_ev, backend=backend,
                graph_reduction=reduction == 'on', configuration=tag,
            )
            rows.append(row)
    metrics = {}
    for configuration in sorted({row['configuration'] for row in rows}):
        selected = [row for row in rows if row['configuration'] == configuration]
        metrics[configuration] = {
            metric: summarize_cluster_metric(
                selected, metric, cluster_fields=('seed', 'train_window_id')
            )
            for metric in RUNTIME_METRICS
            if any(row.get(metric) is not None for row in selected)
        }
    payload = dict(
        method='recourse_macro', learner_variant=args.learner_variant,
        fleet_sizes=args.fleet_sizes, aev_share=args.aev_share,
        runtime_components=list(RUNTIME_METRICS),
        peak_memory_scope='worker_process_lifetime',
        rows=rows, metrics=metrics, child_summaries=summaries,
    )
    path = args.output_dir / 'scalability_summary.json'
    path.write_text(json.dumps(payload, indent=2))
    return path


def main(argv=None):
    args = parse_args(argv)
    planned = jobs(args)
    if args.dry_run:
        print(json.dumps({'commands': [job[4] for job in planned]}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for *_fields, command, _output in planned:
        subprocess.run(command, cwd=ROOT, check=True)
    print(aggregate(args, planned))


if __name__ == '__main__':
    main()
