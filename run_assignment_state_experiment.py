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
            sys.executable, str(ROOT / 'run_recourse_panel.py'),
            '--methods', 'recourse_macro', '--state-variant', variant,
            '--train-days', *args.train_days,
            '--test-days', *args.test_days,
            '--seeds', *map(str, args.seeds),
            '--parquet-path', str(args.parquet_path),
            '--num-vehicles', str(args.num_vehicles),
            '--num-ev', str(args.num_ev),
            '--event-contract-mode', 'record',
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
    for variant, command, output in jobs:
        subprocess.run(command, cwd=ROOT, check=True)
        summaries[variant] = json.loads((output / 'panel_summary.json').read_text())
    result = args.output_dir / 'state_experiment_summary.json'
    result.write_text(json.dumps({
        'state_variants': args.state_variants,
        'summaries': summaries,
    }, indent=2))
    print(result)


if __name__ == '__main__':
    main()
