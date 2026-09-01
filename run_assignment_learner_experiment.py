"""Compare Myopic, DirectQ, and residual learning under Macro recourse.

Only the learner changes.  Physical architecture, state, exact solver,
predictor controls, date windows, CRN, and update schedule are shared.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from run_recourse_multiday_panel import FORMAL_METRICS
from src.recourse.cluster_stats import summarize_cluster_metric, summarize_paired_cluster_difference


ROOT = Path(__file__).resolve().parent
LEARNERS = ('structured_myopic', 'integrated_directq', 'optimization_anchored_residual')
CONTRASTS = (
    ('structured_myopic', 'integrated_directq'),
    ('integrated_directq', 'optimization_anchored_residual'),
    ('structured_myopic', 'optimization_anchored_residual'),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learners', nargs='+', choices=LEARNERS, default=list(LEARNERS))
    parser.add_argument('--train-days', nargs='+', required=True)
    parser.add_argument('--test-days', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--parquet-path', type=Path, required=True)
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--epoch-length', type=float, default=30.0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--train-every', type=int, default=10)
    parser.add_argument('--joint-replay-capacity', type=int, default=256)
    parser.add_argument('--checkpoint-replay', choices=['none', 'recent', 'full'], default='recent')
    parser.add_argument('--checkpoint-replay-recent', type=int, default=5000)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--energy-model', choices=['general_charging', 'fixed_swap'], required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.learners)) != len(args.learners):
        parser.error('learners must be unique')
    if args.energy_model == 'fixed_swap':
        parser.error('fixed_swap is not implemented')
    args.output_dir = (args.output_dir or ROOT / 'results/assignment_learner_experiment' /
                       datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    args.parquet_path = args.parquet_path.resolve()
    return args


def commands(args):
    result = []
    for learner in args.learners:
        output = args.output_dir / learner
        command = [
            sys.executable, str(ROOT / 'run_recourse_multiday_panel.py'),
            '--methods', 'recourse_macro', '--learner-variant', learner,
            '--state-variant', 'joint_state_separate_critics',
            '--train-days', *args.train_days, '--test-days', *args.test_days,
            '--seeds', *map(str, args.seeds), '--parquet-path', str(args.parquet_path),
            '--num-vehicles', str(args.num_vehicles), '--num-ev', str(args.num_ev),
            '--epoch-length', str(args.epoch_length), '--batch-size', str(args.batch_size),
            '--train-every', str(args.train_every),
            '--joint-replay-capacity', str(args.joint_replay_capacity),
            '--checkpoint-replay', args.checkpoint_replay,
            '--checkpoint-replay-recent', str(args.checkpoint_replay_recent),
            '--workers', str(args.workers), '--event-contract-mode', 'record',
            '--energy-model', args.energy_model, '--output-dir', str(output),
        ]
        if args.smoke_steps is not None:
            command += ['--smoke-steps', str(args.smoke_steps)]
        result.append((learner, command, output))
    return result


def aggregate(args, jobs):
    rows = []
    summaries = {}
    for learner, _command, output in jobs:
        summary = json.loads((output / 'panel_summary.json').read_text())
        summaries[learner] = summary
        for source in summary['rows']:
            row = dict(source)
            row['method'] = learner
            row['learner_variant'] = learner
            rows.append(row)
    metrics = {}
    for learner in args.learners:
        learner_rows = [row for row in rows if row['learner_variant'] == learner]
        metrics[learner] = {
            metric: summarize_cluster_metric(
                learner_rows, metric, cluster_fields=('seed', 'train_window_id')
            )
            for metric in FORMAL_METRICS
            if any(row.get(metric) is not None for row in learner_rows)
        }
    paired = []
    for baseline, treatment in CONTRASTS:
        if baseline not in args.learners or treatment not in args.learners:
            continue
        for metric in FORMAL_METRICS:
            try:
                paired.append(summarize_paired_cluster_difference(
                    rows, metric, baseline=baseline, treatment=treatment,
                    pair_fields=('seed', 'train_window_id', 'day_id'),
                    cluster_fields=('seed', 'train_window_id'),
                ))
            except ValueError:
                pass
    payload = dict(
        physical_method='recourse_macro',
        fixed_axes=dict(
            state_variant='joint_state_separate_critics',
            solver='exact_reduced_primal_dual', predictor='p0_frozen',
            acceptance_probability_critic_input='off', energy_model=args.energy_model,
        ),
        learners=args.learners, cluster_fields=['seed', 'train_window_id'],
        pair_fields=['seed', 'train_window_id', 'day_id'],
        rows=rows, metrics=metrics, paired_differences=paired,
        child_summaries=summaries,
    )
    path = args.output_dir / 'learner_experiment_summary.json'
    path.write_text(json.dumps(payload, indent=2))
    return path


def main(argv=None):
    args = parse_args(argv)
    jobs = commands(args)
    if args.dry_run:
        print(json.dumps({'commands': [command for _learner, command, _output in jobs]}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for _learner, command, _output in jobs:
        subprocess.run(command, cwd=ROOT, check=True)
    print(aggregate(args, jobs))


if __name__ == '__main__':
    main()
