"""Multi-seed, multi-day NYC recourse panel orchestrator.

The independent fitted-policy cluster is ``(seed, train_day)``. Held-out test
days within that cluster are repeated evaluations, not independent models.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import subprocess
import sys

from run_recourse_audit import MAIN_METHODS
from src.recourse.cluster_stats import (
    summarize_cluster_metric,
    summarize_paired_cluster_difference,
)
from src.recourse.config import (
    ARCHITECTURE_CONTRASTS,
    CAUSAL_CONTRASTS,
    DIAGNOSTIC_CONTRASTS,
)
from src.recourse.types import STATE_VARIANTS


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
    parser.add_argument('--checkpoint-replay', choices=['none', 'recent', 'full'], default='recent')
    parser.add_argument('--checkpoint-replay-recent', type=int, default=5000)
    parser.add_argument('--state-variant', choices=STATE_VARIANTS, default='joint_state_separate_critics')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--event-contract-mode', choices=['required', 'record', 'off'], default='record')
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
    if len(set(args.methods)) != len(args.methods):
        parser.error('methods must be unique')
    if args.test_seed_offset == 0:
        parser.error('test-seed-offset must keep train/test streams disjoint')
    test_seeds = {seed + args.test_seed_offset for seed in args.seeds}
    if len(test_seeds) != len(args.seeds) or test_seeds & set(args.seeds):
        parser.error('derived test seeds must be unique and disjoint from train seeds')
    if args.checkpoint_replay_recent <= 0:
        parser.error('checkpoint-replay-recent must be positive')
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
        '--checkpoint-replay', args.checkpoint_replay,
        '--checkpoint-replay-recent', str(args.checkpoint_replay_recent),
        '--state-variant', args.state_variant,
        '--workers', str(args.workers), '--output-dir', str(output),
        '--event-contract-mode', args.event_contract_mode,
    ]
    if args.smoke_steps is not None:
        command += ['--smoke-steps', str(args.smoke_steps)]
    return command


def aggregate(args, clusters):
    rows = []
    trained_hashes = {}
    for cluster in clusters:
        folder = args.output_dir / (
            f"seed-{cluster['seed']}-train-{cluster['train_day']}-test-{cluster['test_day']}"
        )
        summary = json.loads((folder / 'summary.json').read_text())
        for result in summary['runs']:
            testing = dict(result['testing'])
            contract = dict(result.get('event_contract', {}))
            rows.append({
                **testing,
                'method': result['method'],
                'seed': cluster['seed'],
                'day_id': cluster['test_day'],
                'train_day': cluster['train_day'],
                'contract_passed': float(bool(contract.get('passed', False))),
                'eligible_repair_exposure': float(
                    testing.get('eligible_rejected_residual_count', 0) > 0
                ),
                'repair_assignment_exposure': float(
                    testing.get('same_epoch_aev_assignment_count', 0) > 0
                    or testing.get('samitha_repair_assignment_count', 0) > 0
                ),
            })
            model_key = (
                cluster['seed'], cluster['train_day'], result['method']
            )
            trained_hash = result.get('trained_weight_hash')
            previous_hash = trained_hashes.setdefault(model_key, trained_hash)
            if previous_hash != trained_hash:
                raise AssertionError(
                    f'same fitted-policy key changed weights: {model_key}'
                )
    metric_names = (
        'reward', 'completed_orders', 'expired_request_count',
        'ev_rejected_offer_count', 'eligible_rejected_residual_count',
        'same_epoch_aev_assignment_count', 'completion_after_rejection_count',
        'conditional_recovery_rate_assignment',
        'conditional_recovery_rate_completion', 'contract_passed',
        'eligible_repair_exposure', 'repair_assignment_exposure',
    )
    metrics = {}
    for method in args.methods:
        method_rows = [row for row in rows if row['method'] == method]
        metrics[method] = {
            metric: summarize_cluster_metric(
                method_rows, metric, cluster_fields=('seed', 'train_day')
            )
            for metric in metric_names
            if any(row.get(metric) is not None for row in method_rows)
        }
    paired_differences = []
    for baseline, treatment in (
        *CAUSAL_CONTRASTS, *ARCHITECTURE_CONTRASTS, *DIAGNOSTIC_CONTRASTS
    ):
        if baseline not in args.methods or treatment not in args.methods:
            continue
        for metric in metric_names:
            try:
                paired_differences.append(summarize_paired_cluster_difference(
                    rows, metric, baseline=baseline, treatment=treatment,
                    pair_fields=('seed', 'train_day', 'day_id'),
                    cluster_fields=('seed', 'train_day'),
                ))
            except ValueError:
                continue
    independent_models = sorted({
        (cluster['seed'], cluster['train_day']) for cluster in clusters
    })
    payload = dict(
        cluster_unit='trained_policy_seed_train_day',
        cluster_fields=['seed', 'train_day'],
        pair_fields=['seed', 'train_day', 'day_id'],
        independent_model_count=len(independent_models),
        heldout_seed_day_count=len(clusters),
        cluster_count=len(clusters),
        methods=args.methods, clusters=clusters, rows=rows, metrics=metrics,
        paired_differences=paired_differences,
        repeated_training_hashes={
            str(key): value for key, value in trained_hashes.items()
        },
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
