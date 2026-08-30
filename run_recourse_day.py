"""Canonical recourse methods: one NYC training day and one held-out day.

Run workers in separate processes so complete replay graphs are freed between
methods. Checkpoints are loaded before testing; evaluation never trains.
"""
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
from run_recourse_audit import MAIN_METHODS, ROOT, build_env, build_pair, rollout
from src.recourse.contracts import assert_method_event_contract, evaluate_method_event_contract
from src.recourse.config import METHODS, method_metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=MAIN_METHODS, default=MAIN_METHODS)
    parser.add_argument('--train-date', default='2025-12-18')
    parser.add_argument('--test-date', default='2025-12-19')
    parser.add_argument('--parquet-path', type=Path, default=ROOT / 'nyedata/nye_simulation/parquet/yellow_tripdata_2025-12.parquet')
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--test-seed', type=int, default=90071)
    parser.add_argument('--epoch-length', type=float, default=30.)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--train-every', type=int, default=10)
    parser.add_argument('--joint-replay-capacity', type=int, default=256)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true', help='Resume completed train/test phase boundaries, not partial epochs')
    parser.add_argument('--smoke-steps', type=int, default=None, help='Explicit preflight only; omit for full 24h days')
    parser.add_argument('--event-contract-mode', choices=['required', 'record', 'off'], default='required')
    parser.add_argument('--worker-method', choices=MAIN_METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if date.fromisoformat(args.train_date) == date.fromisoformat(args.test_date):
        parser.error('Training and test dates must be disjoint')
    if args.seed == args.test_seed:
        parser.error('Training and test seeds must be disjoint')
    if not 0 < args.num_ev < args.num_vehicles:
        parser.error('Require both EV and AEV fleets')
    if min(args.epoch_length, args.batch_size, args.train_every, args.workers) <= 0:
        parser.error('Counts and epoch length must be positive')
    if args.joint_replay_capacity < max(2, args.batch_size + 1):
        parser.error('Replay must retain at least one successor in addition to the batch')
    if args.smoke_steps is not None and args.smoke_steps < args.train_every:
        parser.error('Preflight must reach at least one training update')
    if len(set(args.methods)) != len(args.methods):
        parser.error('Methods must be unique')
    args.environment, args.date = 'nyc', args.train_date
    args.start_hour, args.stop_hour, args.max_steps = 0., 24., args.smoke_steps
    args.parquet_path = args.parquet_path.resolve()
    args.output_dir = (args.output_dir or ROOT / 'results/recourse_day' / datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    return args


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dates(args):
    audits = {}
    for day in (args.train_date, args.test_date):
        start = pd.Timestamp(day)
        frame = pd.read_parquet(args.parquet_path, columns=['tpep_pickup_datetime'],
            filters=[('tpep_pickup_datetime', '>=', start), ('tpep_pickup_datetime', '<', start + timedelta(days=1))])
        times = frame['tpep_pickup_datetime']
        hours = sorted(map(int, times.dt.hour.unique()))
        if hours != list(range(24)):
            raise ValueError(f'{day} does not contain all 24 hours: {hours}')
        audits[day] = dict(raw_trip_count=len(frame), first_pickup=str(times.min()),
                          last_pickup=str(times.max()), hours_present=hours)
    return audits


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, default=json_default, ensure_ascii=False, indent=2))
    temporary.replace(path)


def run_worker(args):
    method = args.worker_method
    folder = args.output_dir / method
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    with (folder / 'run.log').open('a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        trained_path, checkpoint = folder / 'training.json', folder / 'checkpoint.pt'
        if not (args.resume and trained_path.exists() and checkpoint.exists()):
            env = build_env(args, args.seed, method, training=True)
            seed_everything(args.seed + 100000)
            pair = build_pair(env, replay_buffer_size=5 * args.joint_replay_capacity)
            initial_hash = weight_hash(pair)
            trained = rollout(args, env, pair, method, training=True, progress_path=folder / 'progress.json')
            trained_hash = weight_hash(pair)
            if initial_hash == trained_hash:
                raise AssertionError('Training did not change any model weights')
            spec = METHODS[method]
            save_pair(pair, checkpoint, dict(method=method, initial_weight_hash=initial_hash,
                trained_weight_hash=trained_hash, train_date=args.train_date, test_date=args.test_date,
                seed=args.seed, test_seed=args.test_seed,
                **method_metadata(spec.operating_mode, spec.variant),
                state_variant=env.state_variant, learner_variant=env.learner_variant,
                solver_config=dict(
                    rollout_solver=getattr(env, 'mcmf_solver', 'exact'),
                    backend=getattr(env, 'mcmf_backend', 'primal_dual'),
                    graph_reduction=getattr(env, 'mcmf_graph_reduction', True),
                    verify=getattr(env, 'mcmf_verify', True),
                    cost_scale=getattr(env, 'mcmf_cost_scale', 10_000),
                    target_policy=getattr(env, 'target_solver_policy', 'same_as_rollout_exact'),
                )))
            save_json(trained_path, trained)
            del pair, env
            gc.collect()
        trained = json.loads(trained_path.read_text())
        payload = torch.load(checkpoint, weights_only=False, map_location='cpu')
        env = build_env(args, args.test_seed, method, training=False)
        pair = build_pair(env, replay_buffer_size=5 * args.joint_replay_capacity)
        for value, saved in zip(pair, payload['learners']):
            value.network.load_state_dict(saved['network'])
            value.target_network.load_state_dict(saved['target'])
            if saved.get('optimizer'):
                value.optimizer.load_state_dict(saved['optimizer'])
            value.load_extra_checkpoint_state(saved['extra'])
        pair = attach_pair(env, pair)
        before = weight_hash(pair)
        if before != payload['metadata']['trained_weight_hash']:
            raise AssertionError('Checkpoint did not reproduce the trained weights')
        tested = rollout(args, env, pair, method, training=False, progress_path=folder / 'progress.json')
        if before != weight_hash(pair):
            raise AssertionError('Evaluation mutated model weights')
        expected_steps = args.max_steps or round(86400 / args.epoch_length)
        if trained['steps'] != expected_steps or tested['steps'] != expected_steps:
            raise AssertionError('A requested full training/test day ended early')
        contract_stats = dict(tested)
        for key in (
            'aev_follower_optimizer_steps', 'aev_stage_graph_count',
            'aev_learned_score_difference_count', 'macro_leader_target_count',
            'nested_leader_target_count', 'follower_target_query_count',
        ):
            contract_stats[key] = trained.get(key, 0)
        contract = evaluate_method_event_contract(method, contract_stats)
        if args.event_contract_mode == 'required':
            assert_method_event_contract(method, contract_stats)
        result = dict(training=trained, testing=tested,
            event_contract=contract.as_dict(),
            checkpoint_loaded=True, test_weights_unchanged=True, **payload['metadata'])
        save_json(folder / 'results.json', result)


def collect_results(args):
    runs = []
    for method in args.methods:
        path = args.output_dir / method / 'results.json'
        if path.exists():
            runs.append(json.loads(path.read_text()))
    if runs:
        for label, values in (
            ('initial weights', {r['initial_weight_hash'] for r in runs}),
            ('training demand', {r['training']['demand_hash'] for r in runs}),
            ('testing demand', {r['testing']['demand_hash'] for r in runs}),
        ):
            if len(values) != 1:
                raise AssertionError(f'Methods do not share {label}')
    summary = dict(scope='preflight' if args.smoke_steps else '24h_train_24h_heldout_test',
        train_date=args.train_date, test_date=args.test_date, num_vehicles=args.num_vehicles,
        num_ev=args.num_ev, completed_methods=len(runs), required_methods=len(args.methods),
        runs=runs, definitions=dict(recourse_number='same_epoch_aev_assignment_count',
            recourse_completed_number='completion_after_rejection_count',
            reward='sum of actual env.step vehicle rewards', completed_number='completed_orders'))
    save_json(args.output_dir / 'summary.json', summary)
    return summary


def main():
    args = parse_args()
    if args.worker_method:
        run_worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest = dict(arguments=vars(args), data_audit=audit_dates(args),
        data_sha256=digest_file(args.parquet_path),
        source_sha256={str(p.relative_to(ROOT)): digest_file(p) for p in
            [Path(__file__), ROOT / 'run_recourse_audit.py', ROOT / 'run_acceptance_ablation.py',
             ROOT / 'train_acceptance_model.py', *sorted((ROOT / 'src').rglob('*.py'))]})
    path = args.output_dir / 'manifest.json'
    if args.resume and path.exists():
        prior = json.loads(path.read_text())
        if prior['source_sha256'] != manifest['source_sha256'] or prior['data_sha256'] != manifest['data_sha256']:
            raise ValueError('Cannot mix source/data versions on resume')
        ignored = {'resume', 'workers', 'output_dir', 'worker_method'}
        current_args = json.loads(json.dumps(vars(args), default=json_default))
        if {k: v for k, v in prior['arguments'].items() if k not in ignored} != {
                k: v for k, v in current_args.items() if k not in ignored}:
            raise ValueError('Cannot change experiment settings on resume')
    else:
        save_json(path, manifest)
    pending = [m for m in args.methods if not (args.resume and (args.output_dir / m / 'results.json').exists())]
    active, failures = {}, []
    while pending or active:
        while pending and len(active) < args.workers:
            method = pending.pop(0)
            folder = args.output_dir / method
            folder.mkdir(exist_ok=True)
            log = (folder / 'worker.log').open('a', buffering=1)
            command = [sys.executable, str(Path(__file__)), *sys.argv[1:], '--output-dir', str(args.output_dir),
                       '--worker-method', method]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            active[method] = (process, log)
            print(f'Started {method}: pid={process.pid}', flush=True)
        for method, (process, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            log.close()
            del active[method]
            if status:
                failures.append(dict(method=method, exit_code=status))
            print(f'Finished {method}: exit={status}', flush=True)
            collect_results(args)
        if active:
            time.sleep(.5)
    summary = collect_results(args)
    summary['failures'] = failures
    save_json(args.output_dir / 'summary.json', summary)
    if failures:
        raise RuntimeError(f'Methods failed; inspect worker.log: {failures}')
    print(args.output_dir / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
