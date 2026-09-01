"""Train and evaluate Macro under each state-information definition.

This is the state performance experiment. ``run_assignment_state_audit.py``
remains the fixed-trace information-leak audit.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import subprocess
import sys

from src.recourse.cluster_stats import summarize_cluster_metric, summarize_paired_cluster_difference
from src.recourse.types import STATE_VARIANTS


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-variants', nargs='+', choices=STATE_VARIANTS,
                        default=list(STATE_VARIANTS))
    parser.add_argument('--train-days', nargs='+', required=True)
    parser.add_argument('--test-days', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--parquet-path', type=Path, required=True)
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--energy-model', choices=['general_charging', 'fixed_swap'],
                        default='general_charging')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.state_variants)) != len(args.state_variants):
        parser.error('state variants must be unique')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be unique')
    for day_value in (*args.train_days, *args.test_days):
        date.fromisoformat(day_value)
    if set(args.train_days) & set(args.test_days):
        parser.error('training and held-out days must be disjoint')
    if args.energy_model == 'fixed_swap':
        parser.error('fixed_swap is not implemented')
    args.output_dir = (
        args.output_dir or ROOT / 'results/assignment_state_experiment'
        / datetime.now().strftime('%Y%m%d-%H%M%S')
    ).resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    jobs = []
    for variant in args.state_variants:
        output = args.output_dir / variant
        command = [
            sys.executable, str(ROOT / 'run_recourse_multiday_panel.py'),
            '--methods', 'recourse_macro', '--state-variant', variant,
            '--train-days', *args.train_days,
            '--test-days', *args.test_days,
            '--seeds', *map(str, args.seeds),
            '--parquet-path', str(args.parquet_path),
            '--num-vehicles', str(args.num_vehicles),
            '--num-ev', str(args.num_ev),
            '--event-contract-mode', 'record',
            '--energy-model', args.energy_model,
            '--output-dir', str(output),
        ]
        if args.smoke_steps is not None:
            command += ['--smoke-steps', str(args.smoke_steps)]
        jobs.append((variant, command, output))
    if args.dry_run:
        print(json.dumps({'commands': [job[1] for job in jobs]}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = {}
    rows = []
    for variant, command, output in jobs:
        subprocess.run(command, cwd=ROOT, check=True)
        summaries[variant] = json.loads((output / 'panel_summary.json').read_text())
        for source in summaries[variant]['rows']:
            row = dict(source)
            row['method'] = variant
            row['state_variant'] = variant
            rows.append(row)
    metric_names = (
        'reward', 'completed_orders', 'conditional_recovery_rate_completion',
        'model_parameter_count', 'observation_node_count_mean',
        'training_time_seconds', 'decision_latency_p95_seconds',
    )
    metrics = {
        variant: {
            metric: summarize_cluster_metric(
                [row for row in rows if row['state_variant'] == variant],
                metric, cluster_fields=('seed', 'train_window_id'),
            )
            for metric in metric_names
            if any(row.get(metric) is not None for row in rows
                   if row['state_variant'] == variant)
        }
        for variant in args.state_variants
    }
    baseline = 'joint_state_separate_critics'
    contrasts = (
        (baseline, 'fleet_local_separate_critics'),
        (baseline, 'strict_fleet_local_separate_critics'),
        (baseline, 'joint_state_shared_critic'),
    )
    paired = []
    for control, treatment in contrasts:
        if control not in args.state_variants or treatment not in args.state_variants:
            continue
        for metric in metric_names:
            try:
                paired.append(summarize_paired_cluster_difference(
                    rows, metric, baseline=control, treatment=treatment,
                    pair_fields=('seed', 'train_window_id', 'day_id'),
                    cluster_fields=('seed', 'train_window_id'),
                ))
            except ValueError:
                pass
    result = args.output_dir / 'state_experiment_summary.json'
    result.write_text(json.dumps({
        'state_variants': args.state_variants,
        'cluster_fields': ['seed', 'train_window_id'],
        'pair_fields': ['seed', 'train_window_id', 'day_id'],
        'rows': rows,
        'metrics': metrics,
        'paired_differences': paired,
        'summaries': summaries,
    }, indent=2))
    print(result)


if __name__ == '__main__':
    main()
