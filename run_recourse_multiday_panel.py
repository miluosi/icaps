"""Train each fitted recourse policy once on a date window and evaluate many days.

The independent unit is ``(seed, train_window)``.  One immutable checkpoint is
created per method and fitted-policy seed, then reloaded independently for each
held-out day.  This runner reuses the canonical ADP environment, value
functions, rollout, checkpoint format, CRN, and event contracts.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import torch

from run_acceptance_ablation import attach_pair, json_default, save_pair, seed_everything, weight_hash
from run_recourse_audit import MAIN_METHODS, build_env, build_pair, rollout
from run_recourse_day import digest_file, save_json, validate_checkpoint_payload
from src.recourse.cluster_stats import summarize_cluster_metric, summarize_paired_cluster_difference
from src.recourse.config import ARCHITECTURE_CONTRASTS, CAUSAL_CONTRASTS, DIAGNOSTIC_CONTRASTS, METHODS, method_metadata
from src.recourse.contracts import assert_method_event_contract, evaluate_method_event_contract
from src.recourse.manifest import METRIC_DEFINITIONS, environment_metadata, git_commit
from src.recourse.types import LEARNER_VARIANTS, REPLAY_SCHEMA_VERSION, STATE_VARIANTS


ROOT = Path(__file__).resolve().parent
FORMAL_METRICS = (
    'reward', 'reward_ev', 'reward_aev', 'completed_orders', 'service_ratio',
    'lost_requests', 'unresolved_requests', 'ev_rejected_offer_count',
    'eligible_rejected_residual_count', 'same_epoch_aev_assignment_count',
    'aev_pickup_after_rejection_count', 'completion_after_rejection_count',
    'conditional_recovery_rate_assignment', 'conditional_recovery_rate_pickup',
    'conditional_recovery_rate_completion', 'unrecovered_rejected_count',
    'ordinary_aev_service_displacement_fixed_graph',
    'initial_integrated_aev_commit_count', 'hold_candidate_count',
    'hold_selected_count', 'repair_usage_per_hold', 'unused_hold_count',
    'samitha_repair_assignment_count', 'samitha_repair_pickup_count',
    'samitha_repair_completion_count', 'committed_aev_reassignment_count',
    'human_ev_charging_sessions', 'aev_charging_sessions',
    'avg_charging_wait_time', 'waiting_vehicle_count',
    'avg_station_utilization', 'vehicles_unable_to_reach_charging',
    'realized_initial_battery_mean_human_ev',
    'realized_initial_battery_mean_aev',
    'final_battery_mean_human_ev', 'final_battery_mean_aev',
    'battery_consumption_ratio', 'charge_duration_scale',
    'candidate_generation_runtime_seconds', 'neural_edge_scoring_runtime_seconds',
    'graph_serialization_runtime_seconds', 'graph_reduction_runtime_seconds',
    'exact_solve_runtime_seconds', 'total_decision_runtime_seconds',
    'decision_latency_p50_seconds', 'decision_latency_p90_seconds',
    'decision_latency_p95_seconds', 'decision_latency_p99_seconds',
    'training_time_seconds', 'process_peak_memory_mb',
    'observation_node_count_mean',
    'model_parameter_count', 'effective_trainable_parameter_count',
    'gradient_update_count', 'leader_td_target_mean', 'leader_td_target_std',
    'follower_td_target_mean', 'follower_td_target_std',
    'joint_prediction_abs_mean', 'gradient_clipping_rate',
    'replay_true_recourse_fraction', 'contract_passed',
)


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
    parser.add_argument('--learner-variant', choices=LEARNER_VARIANTS, default='optimization_anchored_residual')
    parser.add_argument('--rejection-logit-shift', type=float, default=0.0)
    parser.add_argument('--nyc-demand-scale', type=float, default=1.0)
    parser.add_argument('--station-capacity-scale', type=float, default=1.0)
    parser.add_argument('--battery-consumption-ratio', type=float, default=1.0)
    parser.add_argument('--initial-battery-mean', type=float, default=0.875)
    parser.add_argument('--charge-duration-scale', type=float, default=1.0)
    parser.add_argument('--energy-model', choices=['general_charging', 'fixed_swap'], required=True)
    parser.add_argument('--samitha-hold-rule', choices=['learned', 'fixed'], default='learned')
    parser.add_argument('--samitha-fixed-hold-fraction', type=float, default=0.0)
    parser.add_argument('--graph-reduction', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--mcmf-backend', choices=['primal_dual', 'ortools', 'gurobi_network'],
                        default='primal_dual')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--event-contract-mode', choices=['required', 'record', 'off'], default='record')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker-method', choices=MAIN_METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--worker-seed', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for value in (*args.train_days, *args.test_days):
        date.fromisoformat(value)
    if len(set(args.train_days)) != len(args.train_days) or len(set(args.test_days)) != len(args.test_days):
        parser.error('train/test days must each be unique')
    if set(args.train_days) & set(args.test_days):
        parser.error('training and held-out days must be disjoint')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error('seeds and methods must be unique')
    if args.test_seed_offset == 0 or ({seed + args.test_seed_offset for seed in args.seeds} & set(args.seeds)):
        parser.error('test seed streams must be disjoint from training seeds')
    if not 0 < args.num_ev < args.num_vehicles:
        parser.error('require both EV and AEV fleets')
    if min(args.epoch_length, args.batch_size, args.train_every, args.workers,
           args.nyc_demand_scale, args.station_capacity_scale,
           args.battery_consumption_ratio, args.initial_battery_mean,
           args.charge_duration_scale) <= 0:
        parser.error('counts, demand, and energy values must be positive')
    if args.initial_battery_mean > 1.0 or not 0 <= args.samitha_fixed_hold_fraction <= 1:
        parser.error('SOC and hold fraction must be in (0,1] and [0,1]')
    if args.energy_model == 'fixed_swap':
        parser.error('fixed_swap is a different physical model and is not implemented')
    if args.joint_replay_capacity < max(2, args.batch_size + 1):
        parser.error('joint replay is too small for linked transitions')
    if args.smoke_steps is not None and args.smoke_steps < args.train_every:
        parser.error('smoke steps must reach a training update')
    args.output_dir = (args.output_dir or ROOT / 'results/recourse_multiday_panel' /
                       datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    args.parquet_path = args.parquet_path.resolve()
    args.environment = 'nyc'
    args.start_hour, args.stop_hour, args.max_steps = 0.0, 24.0, args.smoke_steps
    return args


def train_window_id(days):
    text = '|'.join(days)
    return f"{days[0]}--{days[-1]}--{hashlib.sha256(text.encode()).hexdigest()[:10]}"


def policy_folder(args, method, seed):
    return args.output_dir / f"seed-{seed}-train-{train_window_id(args.train_days)}" / method


def _aggregate_training_stats(runs):
    """Return cumulative learner counters from the final training day.

    The same learner pair is reused across days, so optimizer and diagnostic
    counters in each rollout are cumulative. Summing daily snapshots would
    double-count all earlier dates.
    """
    keys = {
        'aev_follower_optimizer_steps', 'aev_stage_graph_count',
        'aev_learned_score_difference_count', 'macro_leader_target_count',
        'nested_leader_target_count', 'follower_target_query_count',
    }
    final = runs[-1] if runs else {}
    return {key: float(final.get(key, 0) or 0) for key in keys}


def _load_pair(env, payload, args):
    pair = build_pair(
        env,
        replay_buffer_size=5 * args.joint_replay_capacity,
        checkpoint_replay=args.checkpoint_replay,
        checkpoint_replay_recent=args.checkpoint_replay_recent,
    )
    for value, saved in zip(pair, payload['learners']):
        value.network.load_state_dict(saved['network'])
        value.target_network.load_state_dict(saved['target'])
        if saved.get('optimizer'):
            value.optimizer.load_state_dict(saved['optimizer'])
        value.load_extra_checkpoint_state(saved['extra'])
    return attach_pair(env, pair)


def run_policy(args):
    method, seed = args.worker_method, args.worker_seed
    if method is None or seed is None:
        raise ValueError('worker requires method and seed')
    folder = policy_folder(args, method, seed)
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'policy_results.json'
    if args.resume and result_path.exists():
        return
    torch.set_num_threads(1)
    with (folder / 'run.log').open('a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        pair = None
        training_runs = []
        initial_hash = None
        for day_index, day in enumerate(args.train_days):
            args.date = day
            day_seed = int(seed + day_index * 1_000_000)
            env = build_env(args, day_seed, method, training=True)
            env.recourse_run_id = f'multiday-train-{seed}-{day}'
            if pair is None:
                seed_everything(seed + 100_000)
                pair = build_pair(
                    env,
                    replay_buffer_size=5 * args.joint_replay_capacity,
                    checkpoint_replay=args.checkpoint_replay,
                    checkpoint_replay_recent=args.checkpoint_replay_recent,
                )
                initial_hash = weight_hash(pair)
            else:
                pair = attach_pair(env, pair)
            training_runs.append(rollout(
                args, env, pair, method, training=True,
                progress_path=folder / f'train-{day}.progress.json',
            ))
            del env
            gc.collect()
        trained_hash = weight_hash(pair)
        if trained_hash == initial_hash:
            raise AssertionError('multiday training did not change model tensors')
        spec = METHODS[method]
        checkpoint = folder / 'checkpoint.pt'
        metadata = dict(
            method=method, seed=seed, train_days=list(args.train_days),
            train_window_id=train_window_id(args.train_days),
            initial_weight_hash=initial_hash, trained_weight_hash=trained_hash,
            **method_metadata(spec.operating_mode, spec.variant),
            state_variant=args.state_variant, learner_variant=args.learner_variant,
            energy_model=args.energy_model,
            samitha_hold_rule=args.samitha_hold_rule,
            samitha_fixed_hold_fraction=args.samitha_fixed_hold_fraction,
            solver_config=dict(
                rollout_solver='exact', backend=args.mcmf_backend, graph_reduction=args.graph_reduction,
                verify=True, strict=True, cost_scale=10_000,
                target_policy='same_as_rollout_exact',
            ),
        )
        save_pair(pair, checkpoint, metadata)
        del pair
        gc.collect()
        payload = torch.load(checkpoint, weights_only=False, map_location='cpu')
        evaluations = []
        training_contract = _aggregate_training_stats(training_runs)
        for day_index, test_day in enumerate(args.test_days):
            test_seed = int(seed + args.test_seed_offset + day_index * 10_000_000)
            args.test_date = test_day
            env = build_env(args, test_seed, method, training=False)
            env.recourse_run_id = f'multiday-test-{seed}-{test_day}'
            validate_checkpoint_payload(payload, method, env)
            pair = _load_pair(env, payload, args)
            before = weight_hash(pair)
            if before != trained_hash:
                raise AssertionError('checkpoint did not reproduce fitted policy')
            tested = rollout(
                args, env, pair, method, training=False,
                progress_path=folder / f'test-{test_day}.progress.json',
            )
            if weight_hash(pair) != before:
                raise AssertionError('held-out evaluation mutated policy tensors')
            contract_stats = {**tested, **training_contract}
            contract = evaluate_method_event_contract(method, contract_stats)
            if args.event_contract_mode == 'required':
                assert_method_event_contract(method, contract_stats)
            evaluations.append(dict(
                day_id=test_day, test_seed=test_seed, testing=tested,
                event_contract=contract.as_dict(), checkpoint_loaded=True,
                test_weights_unchanged=True,
            ))
            del pair, env
            gc.collect()
        save_json(result_path, dict(
            **metadata,
            training=training_runs, evaluations=evaluations,
            checkpoint=str(checkpoint), checkpoint_loaded_for_all_tests=True,
        ))


def planned_jobs(args):
    jobs = []
    passthrough = [
        '--train-days', *args.train_days, '--test-days', *args.test_days,
        '--seeds', *map(str, args.seeds), '--methods', *args.methods,
        '--parquet-path', str(args.parquet_path), '--num-vehicles', str(args.num_vehicles),
        '--num-ev', str(args.num_ev), '--epoch-length', str(args.epoch_length),
        '--batch-size', str(args.batch_size), '--train-every', str(args.train_every),
        '--joint-replay-capacity', str(args.joint_replay_capacity),
        '--checkpoint-replay', args.checkpoint_replay,
        '--checkpoint-replay-recent', str(args.checkpoint_replay_recent),
        '--state-variant', args.state_variant, '--learner-variant', args.learner_variant,
        '--rejection-logit-shift', str(args.rejection_logit_shift),
        '--nyc-demand-scale', str(args.nyc_demand_scale),
        '--station-capacity-scale', str(args.station_capacity_scale),
        '--battery-consumption-ratio', str(args.battery_consumption_ratio),
        '--initial-battery-mean', str(args.initial_battery_mean),
        '--charge-duration-scale', str(args.charge_duration_scale),
        '--energy-model', args.energy_model, '--samitha-hold-rule', args.samitha_hold_rule,
        '--samitha-fixed-hold-fraction', str(args.samitha_fixed_hold_fraction),
        '--graph-reduction' if args.graph_reduction else '--no-graph-reduction',
        '--mcmf-backend', args.mcmf_backend,
        '--test-seed-offset', str(args.test_seed_offset), '--workers', '1',
        '--event-contract-mode', args.event_contract_mode,
        '--output-dir', str(args.output_dir),
    ]
    if args.smoke_steps is not None:
        passthrough += ['--smoke-steps', str(args.smoke_steps)]
    if args.resume:
        passthrough += ['--resume']
    for seed in args.seeds:
        for method in args.methods:
            jobs.append((method, seed, [
                sys.executable, str(Path(__file__)), *passthrough,
                '--worker-method', method, '--worker-seed', str(seed),
            ]))
    return jobs


def audit_dates(args):
    result = {}
    for day in (*args.train_days, *args.test_days):
        start = pd.Timestamp(day)
        frame = pd.read_parquet(
            args.parquet_path, columns=['tpep_pickup_datetime'],
            filters=[('tpep_pickup_datetime', '>=', start),
                     ('tpep_pickup_datetime', '<', start + timedelta(days=1))],
        )
        hours = sorted(map(int, frame.tpep_pickup_datetime.dt.hour.unique()))
        if hours != list(range(24)):
            raise ValueError(f'{day} does not contain all 24 hours: {hours}')
        result[day] = dict(raw_trip_count=len(frame), hours_present=hours)
    return result


def aggregate(args):
    rows = []
    policy_results = []
    for seed in args.seeds:
        for method in args.methods:
            path = policy_folder(args, method, seed) / 'policy_results.json'
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            policy_results.append(result)
            training_summary = result['training'][-1] if result['training'] else {}
            for evaluation in result['evaluations']:
                row = dict(evaluation['testing'])
                for key in FORMAL_METRICS:
                    if key not in row and key in training_summary:
                        row[key] = training_summary[key]
                row.update(
                    method=method, seed=seed, day_id=evaluation['day_id'],
                    train_window_id=result['train_window_id'],
                    learner_variant=args.learner_variant,
                    state_variant=args.state_variant,
                    contract_passed=float(evaluation['event_contract']['passed']),
                    training_time_seconds=float(sum(
                        item.get('elapsed_seconds', 0.0) or 0.0
                        for item in result['training']
                    )),
                )
                row['gradient_update_count'] = int(
                    training_summary.get('gradient_update_count', 0) or 0
                )
                rows.append(row)
    metrics = {}
    for method in args.methods:
        method_rows = [row for row in rows if row['method'] == method]
        metrics[method] = {}
        for metric in FORMAL_METRICS:
            if any(row.get(metric) is not None for row in method_rows):
                metrics[method][metric] = summarize_cluster_metric(
                    method_rows, metric,
                    cluster_fields=('seed', 'train_window_id'),
                )
    paired = []
    for baseline, treatment in (*CAUSAL_CONTRASTS, *ARCHITECTURE_CONTRASTS, *DIAGNOSTIC_CONTRASTS):
        if baseline not in args.methods or treatment not in args.methods:
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
        protocol='train_once_evaluate_many',
        independent_unit='seed_train_window',
        cluster_fields=['seed', 'train_window_id'],
        pair_fields=['seed', 'train_window_id', 'day_id'],
        independent_model_count=len(args.seeds),
        fitted_policy_count=len(policy_results),
        heldout_evaluation_count=len(rows),
        train_days=args.train_days, test_days=args.test_days,
        methods=args.methods, rows=rows, metrics=metrics,
        paired_differences=paired,
        experiment_axes=dict(
            learner_variant=args.learner_variant,
            state_variant=args.state_variant,
            rejection_logit_shift=args.rejection_logit_shift,
            demand_scale=args.nyc_demand_scale,
            aev_share=args.num_ev / args.num_vehicles,
            station_capacity_scale=args.station_capacity_scale,
            battery_consumption_ratio=args.battery_consumption_ratio,
            initial_battery_mean=args.initial_battery_mean,
            charge_duration_scale=args.charge_duration_scale,
            energy_model=args.energy_model,
            samitha_hold_rule=args.samitha_hold_rule,
            samitha_fixed_hold_fraction=args.samitha_fixed_hold_fraction,
            graph_reduction=args.graph_reduction,
            mcmf_backend=args.mcmf_backend,
        ),
    )
    save_json(args.output_dir / 'panel_summary.json', payload)
    return payload


def main(argv=None):
    args = parse_args(argv)
    jobs = planned_jobs(args)
    if args.dry_run:
        print(json.dumps({
            'protocol': 'train_once_evaluate_many',
            'train_window_id': train_window_id(args.train_days),
            'fitted_policy_jobs': len(jobs),
            'heldout_evaluations': len(jobs) * len(args.test_days),
            'commands': [command for _method, _seed, command in jobs],
        }, indent=2))
        return
    if args.worker_method is not None:
        run_policy(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = args.output_dir / 'manifest.json'
    arguments = json.loads(json.dumps(vars(args), default=json_default))
    for key in ('worker_method', 'worker_seed', 'dry_run', 'workers', 'resume'):
        arguments.pop(key, None)
    manifest = dict(
        manifest_version=3,
        git_commit=git_commit(ROOT),
        replay_schema_version=REPLAY_SCHEMA_VERSION,
        arguments=arguments, data_sha256=digest_file(args.parquet_path),
        data_audit=audit_dates(args),
        environment=environment_metadata(),
        metric_definitions=METRIC_DEFINITIONS,
        source_sha256={
            str(path.relative_to(ROOT)): digest_file(path)
            for path in [Path(__file__), ROOT / 'run_recourse_audit.py',
                         ROOT / 'run_recourse_day.py',
                         *sorted((ROOT / 'src').rglob('*.py'))]
        },
    )
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('resume manifest differs from the original experiment')
    else:
        save_json(manifest_path, manifest)
    pending = [job for job in jobs if not (
        args.resume and (policy_folder(args, job[0], job[1]) / 'policy_results.json').exists()
    )]
    active = {}
    failures = []
    while pending or active:
        while pending and len(active) < args.workers:
            method, seed, command = pending.pop(0)
            key = (method, seed)
            log_path = policy_folder(args, method, seed).parent / f'{method}.worker.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open('a', buffering=1)
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            active[key] = (process, log)
            print(f'Started {method}, seed={seed}: pid={process.pid}', flush=True)
        for key, (process, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            log.close()
            del active[key]
            print(f'Finished {key}: exit={status}', flush=True)
            if status:
                failures.append(dict(method=key[0], seed=key[1], exit_code=status))
        if active:
            time.sleep(.5)
    payload = aggregate(args)
    payload['failures'] = failures
    payload['status'] = 'failed' if failures else 'completed'
    save_json(args.output_dir / 'panel_summary.json', payload)
    if failures:
        raise RuntimeError(f'multiday fitted policies failed: {failures}')
    print(args.output_dir / 'panel_summary.json')


if __name__ == '__main__':
    main()
