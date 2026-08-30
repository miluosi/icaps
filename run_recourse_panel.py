"""Multi-seed, multi-day NYC recourse panel orchestrator.

The statistical unit is one ``(seed, held-out test day)`` cluster. Training
days must be either a single shared day or paired one-to-one with test days.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import subprocess
import sys

from run_recourse_audit import MAIN_METHODS
from src.recourse.metrics import summarize_metric_with_uncertainty


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-days', nargs='+', required=True)
    parser.add_argument('--test-days', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--test-seed-offset', type=int, default=90_000)
    parser.add_argument('--methods', nargs='+', choices=MAIN_METHODS, default=MAIN_METHODS)
    parser.add_argument('--parquet-path', type=Path, required=True)
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--epoch-length', type=float, default=30.0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--train-every', type=int, default=10)
    parser.add_argument('--joint-replay-capacity', type=int, default=256)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--event-contract-mode', choices=['required', 'record', 'off'], default='required')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if len(args.train_days) not in {1, len(args.test_days)}:
        parser.error('--train-days must contain one day or match --test-days length')
    for day_value in [*args.train_days, *args.test_days]:
        date.fromisoformat(day_value)
    if set(args.train_days) & set(args.test_days):
        parser.error('training and held-out test days must be disjoint')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be unique')
    if args.test_seed_offset == 0:
        parser.error('test-seed-offset must keep train/test streams disjoint')
    args.output_dir = (args.output_dir or ROOT / 'results/recourse_panel' /
                       datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    args.parquet_path = args.parquet_path.resolve()
    return args


def planned_clusters(args):
    train_days = (
        args.train_days * len(args.test_days)
        if len(args.train_days) == 1 else args.train_days
    )
    return [
        dict(seed=seed, test_seed=seed + args.test_seed_offset,
             train_day=train_day, test_day=test_day)
        for seed in args.seeds
        for train_day, test_day in zip(train_days, args.test_days)
    ]


def cluster_command(args, cluster, output):
    command = [
        sys.executable, str(ROOT / 'run_recourse_day.py'),
        '--methods', *args.methods,
        '--train-date', cluster['train_day'], '--test-date', cluster['test_day'],
        '--seed', str(cluster['seed']), '--test-seed', str(cluster['test_seed']),
        '--parquet-path', str(args.parquet_path),
        '--num-vehicles', str(args.num_vehicles), '--num-ev', str(args.num_ev),
        '--epoch-length', str(args.epoch_length), '--batch-size', str(args.batch_size),
        '--train-every', str(args.train_every),
        '--joint-replay-capacity', str(args.joint_replay_capacity),
        '--workers', str(args.workers), '--output-dir', str(output),
        '--event-contract-mode', args.event_contract_mode,
    ]
    if args.smoke_steps is not None:
        command += ['--smoke-steps', str(args.smoke_steps)]
    return command


def aggregate(args, clusters):
    rows = []
    for cluster in clusters:
        folder = args.output_dir / (
            f"seed-{cluster['seed']}-train-{cluster['train_day']}-test-{cluster['test_day']}"
        )
        summary = json.loads((folder / 'summary.json').read_text())
        for result in summary['runs']:
            testing = result['testing']
            rows.append(dict(
                method=result['method'], seed=cluster['seed'],
                day_id=cluster['test_day'], **testing,
            ))
    metrics = {}
    for method in args.methods:
        method_rows = [row for row in rows if row['method'] == method]
        metrics[method] = {
            metric: summarize_metric_with_uncertainty(method_rows, metric)
            for metric in ('reward', 'completed_orders', 'ev_rejected_offer_count',
                           'same_epoch_aev_assignment_count')
        }
    payload = dict(
        cluster_unit='seed_day', cluster_count=len(clusters),
        methods=args.methods, clusters=clusters, rows=rows, metrics=metrics,
    )
    (args.output_dir / 'panel_summary.json').write_text(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    args = parse_args(argv)
    clusters = planned_clusters(args)
    plan = []
    for cluster in clusters:
        output = args.output_dir / (
            f"seed-{cluster['seed']}-train-{cluster['train_day']}-test-{cluster['test_day']}"
        )
        plan.append(cluster_command(args, cluster, output))
    if args.dry_run:
        print(json.dumps(dict(clusters=clusters, commands=plan), indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for command in plan:
        subprocess.run(command, cwd=ROOT, check=True)
    aggregate(args, clusters)
    print(args.output_dir / 'panel_summary.json')


if __name__ == '__main__':
    main()
