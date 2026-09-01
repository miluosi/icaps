"""Unified recourse sensitivity with adaptation and nominal robustness modes."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import torch

from run_acceptance_ablation import weight_hash
from run_recourse_audit import build_env, rollout
from run_recourse_day import validate_checkpoint_payload
from run_recourse_multiday_panel import (
    FORMAL_METRICS, _load_pair, parse_args as multiday_args,
    policy_folder, train_window_id,
)
from src.recourse.cluster_stats import summarize_cluster_metric
from src.recourse.types import LEARNER_VARIANTS


ROOT = Path(__file__).resolve().parent
DEFAULT_METHODS = (
    'repair_only', 'repair_learning', 'recourse_macro', 'samitha', 'no_repair',
)
SENSITIVITY_METRICS = tuple(dict.fromkeys((*FORMAL_METRICS,
    'ev_conditional_rejection_rate', 'ev_offer_count',
    'charging_station_count', 'charging_total_capacity',
    'battery_consumption_ratio', 'configured_initial_battery_mean',
    'charge_duration_scale', 'vehicles_unable_to_reach_charging',
)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocols', nargs='+',
                        choices=['adaptation', 'nominal_robustness'],
                        default=['adaptation', 'nominal_robustness'])
    parser.add_argument('--design', choices=['one_at_a_time', 'factorial'], default='one_at_a_time')
    parser.add_argument('--rejection-logit-shifts', nargs='+', type=float,
                        default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    parser.add_argument('--aev-shares', nargs='+', type=float,
                        default=[0.2, 0.4, 0.5, 0.6, 0.8])
    parser.add_argument('--demand-scales', nargs='+', type=float,
                        default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument('--station-capacity-scales', nargs='+', type=float, default=[1.0])
    parser.add_argument('--battery-consumption-ratios', nargs='+', type=float, default=[1.0])
    parser.add_argument('--initial-battery-means', nargs='+', type=float, default=[0.875])
    parser.add_argument('--charge-duration-scales', nargs='+', type=float, default=[1.0])
    parser.add_argument('--methods', nargs='+', default=list(DEFAULT_METHODS))
    parser.add_argument('--learner-variant', choices=LEARNER_VARIANTS,
                        default='optimization_anchored_residual')
    parser.add_argument('--train-days', nargs='+', required=True)
    parser.add_argument('--test-days', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--parquet-path', type=Path, required=True)
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--epoch-length', type=float, default=30.0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--train-every', type=int, default=10)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--energy-model', choices=['general_charging', 'fixed_swap'], required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.energy_model == 'fixed_swap':
        parser.error('fixed_swap is not implemented')
    for values, label in (
        (args.aev_shares, 'AEV shares'), (args.demand_scales, 'demand scales'),
        (args.station_capacity_scales, 'station capacity scales'),
        (args.battery_consumption_ratios, 'battery consumption ratios'),
        (args.initial_battery_means, 'initial SOC means'),
        (args.charge_duration_scales, 'charge duration scales'),
    ):
        if len(values) != len(set(values)) or any(value <= 0 for value in values):
            parser.error(f'{label} must be unique and positive')
    if any(not 0 < share < 1 for share in args.aev_shares):
        parser.error('AEV shares must be in (0,1)')
    if any(value > 1 for value in args.initial_battery_means):
        parser.error('initial SOC means must not exceed 1')
    args.output_dir = (args.output_dir or ROOT / 'results/recourse_sensitivity' /
                       datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    args.parquet_path = args.parquet_path.resolve()
    return args


def nominal_config(args):
    return dict(rejection_logit_shift=0.0, aev_share=0.5, demand_scale=1.0,
                station_capacity_scale=1.0, battery_consumption_ratio=1.0,
                initial_battery_mean=0.875, charge_duration_scale=1.0)


def configurations(args):
    nominal = nominal_config(args)
    if args.design == 'factorial':
        import itertools
        rows = [dict(zip(nominal, values)) for values in itertools.product(
            args.rejection_logit_shifts, args.aev_shares, args.demand_scales,
            args.station_capacity_scales, args.battery_consumption_ratios,
            args.initial_battery_means,
            args.charge_duration_scales,
        )]
    else:
        axes = (
            ('rejection_logit_shift', args.rejection_logit_shifts),
            ('aev_share', args.aev_shares),
            ('demand_scale', args.demand_scales),
            ('station_capacity_scale', args.station_capacity_scales),
            ('battery_consumption_ratio', args.battery_consumption_ratios),
            ('initial_battery_mean', args.initial_battery_means),
            ('charge_duration_scale', args.charge_duration_scales),
        )
        rows = [dict(nominal)]
        for name, values in axes:
            for value in values:
                row = dict(nominal)
                row[name] = float(value)
                rows.append(row)
    unique = {json.dumps(row, sort_keys=True): row for row in rows}
    result = []
    for index, row in enumerate(unique.values()):
        changed = [name for name, value in row.items() if value != nominal[name]]
        if not changed:
            tag = 'nominal'
        else:
            tag = '--'.join(f'{name}-{row[name]:g}' for name in changed)
        result.append(dict(configuration_id=tag, **row))
    return result


def multiday_command(args, config, output):
    num_aev = max(1, min(args.num_vehicles - 1, round(args.num_vehicles * config['aev_share'])))
    num_ev = args.num_vehicles - num_aev
    command = [
        sys.executable, str(ROOT / 'run_recourse_multiday_panel.py'),
        '--methods', *args.methods, '--train-days', *args.train_days,
        '--test-days', *args.test_days, '--seeds', *map(str, args.seeds),
        '--parquet-path', str(args.parquet_path), '--num-vehicles', str(args.num_vehicles),
        '--num-ev', str(num_ev), '--epoch-length', str(args.epoch_length),
        '--batch-size', str(args.batch_size), '--train-every', str(args.train_every),
        '--rejection-logit-shift', str(config['rejection_logit_shift']),
        '--nyc-demand-scale', str(config['demand_scale']),
        '--station-capacity-scale', str(config['station_capacity_scale']),
        '--battery-consumption-ratio', str(config['battery_consumption_ratio']),
        '--initial-battery-mean', str(config['initial_battery_mean']),
        '--charge-duration-scale', str(config['charge_duration_scale']),
        '--energy-model', args.energy_model, '--workers', str(args.workers),
        '--learner-variant', args.learner_variant,
        '--event-contract-mode', 'record', '--output-dir', str(output),
    ]
    if args.smoke_steps is not None:
        command += ['--smoke-steps', str(args.smoke_steps)]
    return command


def _base_multiday_namespace(args, output):
    command = multiday_command(args, nominal_config(args), output)
    return multiday_args(command[2:])


def evaluate_nominal_checkpoints(args, nominal_output, configs):
    base = _base_multiday_namespace(args, nominal_output)
    rows = []
    for config in configs:
        num_aev = max(1, min(args.num_vehicles - 1, round(args.num_vehicles * config['aev_share'])))
        base.num_ev = args.num_vehicles - num_aev
        base.rejection_logit_shift = config['rejection_logit_shift']
        base.nyc_demand_scale = config['demand_scale']
        base.station_capacity_scale = config['station_capacity_scale']
        base.battery_consumption_ratio = config['battery_consumption_ratio']
        base.initial_battery_mean = config['initial_battery_mean']
        base.charge_duration_scale = config['charge_duration_scale']
        for seed in args.seeds:
            for method in args.methods:
                folder = policy_folder(base, method, seed)
                checkpoint = folder / 'checkpoint.pt'
                payload = torch.load(checkpoint, weights_only=False, map_location='cpu')
                for day_index, test_day in enumerate(args.test_days):
                    base.test_date = test_day
                    test_seed = seed + base.test_seed_offset + day_index * 10_000_000
                    env = build_env(base, test_seed, method, training=False)
                    env.recourse_run_id = f'sensitivity-robustness-{seed}-{test_day}'
                    validate_checkpoint_payload(payload, method, env)
                    pair = _load_pair(env, payload, base)
                    before = weight_hash(pair)
                    tested = rollout(base, env, pair, method, training=False)
                    if before != weight_hash(pair):
                        raise AssertionError('robustness evaluation mutated checkpoint')
                    row = dict(tested)
                    row.update(
                        protocol='nominal_robustness', method=method,
                        seed=seed, day_id=test_day,
                        train_window_id=train_window_id(args.train_days),
                        checkpoint_weight_hash=before,
                    )
                    row.update(config)
                    rows.append(row)
    return rows


def aggregate_rows(rows):
    metrics = {}
    keys = sorted({(row['protocol'], row['configuration_id'], row['method']) for row in rows})
    for protocol, configuration, method in keys:
        selected = [row for row in rows if row['protocol'] == protocol
                    and row['configuration_id'] == configuration and row['method'] == method]
        metrics[f'{protocol}|{configuration}|{method}'] = {
            metric: summarize_cluster_metric(
                selected, metric, cluster_fields=('seed', 'train_window_id')
            )
            for metric in SENSITIVITY_METRICS
            if any(row.get(metric) is not None for row in selected)
        }
    return metrics


def main(argv=None):
    args = parse_args(argv)
    configs = configurations(args)
    adaptation_jobs = []
    if 'adaptation' in args.protocols:
        for config in configs:
            output = args.output_dir / 'adaptation' / config['configuration_id']
            adaptation_jobs.append((config, multiday_command(args, config, output), output))
    nominal_output = args.output_dir / 'nominal_training'
    nominal_command = multiday_command(args, nominal_config(args), nominal_output)
    if args.dry_run:
        print(json.dumps(dict(
            configurations=configs,
            adaptation_commands=[job[1] for job in adaptation_jobs],
            nominal_training_command=(nominal_command if 'nominal_robustness' in args.protocols else None),
            nominal_robustness_evaluations=(len(configs) * len(args.methods) * len(args.seeds) * len(args.test_days)
                                            if 'nominal_robustness' in args.protocols else 0),
        ), indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for config, command, output in adaptation_jobs:
        subprocess.run(command, cwd=ROOT, check=True)
        summary = json.loads((output / 'panel_summary.json').read_text())
        rows.extend(dict(source, protocol='adaptation', **config) for source in summary['rows'])
    if 'nominal_robustness' in args.protocols:
        if not nominal_output.exists():
            subprocess.run(nominal_command, cwd=ROOT, check=True)
        rows.extend(evaluate_nominal_checkpoints(args, nominal_output, configs))
    payload = dict(
        protocols=args.protocols, design=args.design, methods=args.methods,
        configurations=configs, cluster_fields=['seed', 'train_window_id'],
        rows=rows, metrics=aggregate_rows(rows),
    )
    path = args.output_dir / 'sensitivity_summary.json'
    path.write_text(json.dumps(payload, indent=2))
    print(path)


if __name__ == '__main__':
    main()
