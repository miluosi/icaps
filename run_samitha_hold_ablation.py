"""Compare zero, fixed-fraction, learned, and EV-first AEV reserve policies."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from run_recourse_multiday_panel import FORMAL_METRICS
from src.recourse.cluster_stats import summarize_cluster_metric, summarize_paired_cluster_difference
from src.recourse.types import LEARNER_VARIANTS


ROOT = Path(__file__).resolve().parent
HOLD_METRICS = tuple(dict.fromkeys((*FORMAL_METRICS,
    'initial_integrated_aev_commit_count', 'hold_candidate_count',
    'hold_selected_count', 'repair_candidate_rejected_count',
    'repair_candidate_unassigned_count', 'samitha_repair_assignment_count',
    'repair_usage_per_hold', 'unused_hold_count',
    'ordinary_aev_service_displacement_fixed_graph',
)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-hold-fractions', nargs='+', type=float, default=[0.1, 0.25, 0.5])
    parser.add_argument('--learner-variant', choices=LEARNER_VARIANTS,
                        default='optimization_anchored_residual')
    parser.add_argument('--train-days', nargs='+', required=True)
    parser.add_argument('--test-days', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--parquet-path', type=Path, required=True)
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--epoch-length', type=float, default=30.0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--train-every', type=int, default=10)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--energy-model', choices=['general_charging', 'fixed_swap'], required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.fixed_hold_fractions)) != len(args.fixed_hold_fractions):
        parser.error('fixed hold fractions must be unique')
    if any(not 0 <= value <= 1 for value in args.fixed_hold_fractions):
        parser.error('fixed hold fractions must be in [0,1]')
    if args.energy_model == 'fixed_swap':
        parser.error('fixed_swap is not implemented')
    args.output_dir = (args.output_dir or ROOT / 'results/samitha_hold_ablation' /
                       datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    args.parquet_path = args.parquet_path.resolve()
    return args


def arms(args):
    result = [dict(arm='integrated_0', method='no_repair', rule='learned', fraction=0.0)]
    result.extend(dict(arm=f'fixed_hold_{int(round(value * 100))}', method='samitha',
                       rule='fixed', fraction=value) for value in args.fixed_hold_fractions)
    result.extend((
        dict(arm='samitha_learned', method='samitha', rule='learned', fraction=0.0),
        dict(arm='evfirst_macro_limit', method='recourse_macro', rule='learned', fraction=0.0),
    ))
    return result


def jobs(args):
    result = []
    for arm in arms(args):
        output = args.output_dir / arm['arm']
        command = [
            sys.executable, str(ROOT / 'run_recourse_multiday_panel.py'),
            '--methods', arm['method'], '--learner-variant', args.learner_variant,
            '--state-variant', 'joint_state_separate_critics',
            '--samitha-hold-rule', arm['rule'],
            '--samitha-fixed-hold-fraction', str(arm['fraction']),
            '--train-days', *args.train_days, '--test-days', *args.test_days,
            '--seeds', *map(str, args.seeds), '--parquet-path', str(args.parquet_path),
            '--num-vehicles', str(args.num_vehicles), '--num-ev', str(args.num_ev),
            '--epoch-length', str(args.epoch_length), '--batch-size', str(args.batch_size),
            '--train-every', str(args.train_every), '--workers', str(args.workers),
            '--event-contract-mode', 'record', '--energy-model', args.energy_model,
            '--output-dir', str(output),
        ]
        if args.smoke_steps is not None:
            command += ['--smoke-steps', str(args.smoke_steps)]
        result.append((arm, command, output))
    return result


def aggregate(args, planned):
    rows, summaries = [], {}
    for arm, _command, output in planned:
        summary = json.loads((output / 'panel_summary.json').read_text())
        summaries[arm['arm']] = summary
        for source in summary['rows']:
            row = dict(source)
            row.update(method=arm['arm'], hold_arm=arm['arm'],
                       physical_method=arm['method'], hold_rule=arm['rule'],
                       hold_fraction=arm['fraction'])
            rows.append(row)
    metrics = {}
    for arm in arms(args):
        selected = [row for row in rows if row['hold_arm'] == arm['arm']]
        metrics[arm['arm']] = {
            metric: summarize_cluster_metric(
                selected, metric, cluster_fields=('seed', 'train_window_id')
            )
            for metric in HOLD_METRICS
            if any(row.get(metric) is not None for row in selected)
        }
    paired = []
    comparisons = [('integrated_0', arm['arm']) for arm in arms(args) if arm['arm'] != 'integrated_0']
    comparisons.extend((f"fixed_hold_{int(round(value * 100))}", 'samitha_learned')
                       for value in args.fixed_hold_fractions)
    for baseline, treatment in comparisons:
        for metric in HOLD_METRICS:
            try:
                paired.append(summarize_paired_cluster_difference(
                    rows, metric, baseline=baseline, treatment=treatment,
                    pair_fields=('seed', 'train_window_id', 'day_id'),
                    cluster_fields=('seed', 'train_window_id'),
                ))
            except ValueError:
                pass
    payload = dict(
        arms=arms(args), cluster_fields=['seed', 'train_window_id'],
        pair_fields=['seed', 'train_window_id', 'day_id'],
        rows=rows, metrics=metrics, paired_differences=paired,
        child_summaries=summaries,
    )
    path = args.output_dir / 'samitha_hold_summary.json'
    path.write_text(json.dumps(payload, indent=2))
    return path


def main(argv=None):
    args = parse_args(argv)
    planned = jobs(args)
    if args.dry_run:
        print(json.dumps({'arms': arms(args), 'commands': [job[1] for job in planned]}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for _arm, command, _output in planned:
        subprocess.run(command, cwd=ROOT, check=True)
    print(aggregate(args, planned))


if __name__ == '__main__':
    main()
