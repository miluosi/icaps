"""Paired CRN recourse/architecture training and checkpoint-inference checks.

Integrated and R1 are both retained. Comparisons against Integrated change the
operating architecture; R1->R2->R3->macro is the EV-first causal ladder.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from run_acceptance_ablation import seed_everything, weight_hash, save_pair, json_default, attach_pair
from train_acceptance_model import make_environment, parse_args as environment_args
from src.acceptance_features import configure_acceptance_feature
from src.recourse.config import METHODS, method_metadata
from src.recourse.metrics import summarize_paired_crn_difference, summarize_joint_targets, ordinary_service_displacement
from src.recourse.types import is_true_same_epoch_recourse
from src.value_function_registry import get_value_function_class

ROOT = Path(__file__).resolve().parent
MAIN_METHODS = ['no_repair', 'evfirst_no_repair', 'repair_only', 'repair_learning',
                'recourse_macro', 'recourse_nested_q2', 'samitha']
INFERENCE_CHECK_KEYS = ('reward', 'reward_ev', 'reward_aev', 'completed_orders',
    'ev_offer_count', 'ev_rejected_offer_count', 'samitha_repair_pickup_count',
    'samitha_repair_completion_count', 'human_ev_charging_sessions', 'aev_charging_sessions',
    'all_vehicle_charging_sessions', 'completed_charging_sessions_with_duration_all')


def verify_checkpoint_stats(expected, restored):
    for key in INFERENCE_CHECK_KEYS:
        if key not in expected or key not in restored:
            raise AssertionError(f'missing checkpoint inference metric: {key}')
        if expected[key] != restored[key]:
            raise AssertionError(f'checkpoint inference mismatch: {key}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['nyc', 'synthetic'], default='nyc')
    parser.add_argument('--methods', nargs='+', choices=tuple(METHODS), default=MAIN_METHODS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[71, 72, 73])
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--train-every', type=int, default=10)
    parser.add_argument('--date', default='2025-12-18')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.num_ev < args.num_vehicles or min(args.batch_size, args.train_every) <= 0:
        parser.error('Require both fleets and positive training counts')
    if args.max_steps is not None and args.max_steps <= 1:
        parser.error('max-steps must allow linked joint transitions (>1)')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error('Duplicate seeds or methods')
    args.output_dir = args.output_dir or ROOT / 'results/recourse_credit' / datetime.now().strftime('%Y%m%d-%H%M%S')
    return args


def build_env(args, seed, method, *, training):
    day = args.date if training else getattr(args, 'test_date', args.date)
    settings = environment_args(['--environment', args.environment, '--num-vehicles', str(args.num_vehicles),
                                 '--num-ev', str(args.num_ev), '--date', day])
    for name in ('parquet_path', 'start_hour', 'stop_hour', 'epoch_length'):
        if getattr(args, name, None) is not None:
            setattr(settings, name, getattr(args, name))
    env = make_environment(settings, seed)
    spec = METHODS[method]
    env.configure_recourse_experiment(spec.variant, common_random_numbers=True)
    configure_acceptance_feature(env, 'off')
    env.state_variant = 'joint_state_separate_critics'
    env.learner_variant = 'optimization_anchored_residual'
    env.adp_value, env.evaluatemode = 1., not training
    # Offer-keyed CRN includes run_id: it must be identical across methods.
    env.recourse_run_id = f'recourse-audit-{seed}'
    if hasattr(env, 'set_request_generation_seed'):
        env.set_request_generation_seed(seed)
    return env


def build_pair(env, *, replay_buffer_size=5000):
    cls = get_value_function_class(env.learner_variant)
    pair = [cls(env=env, num_vehicles=len(env.vehicles), grid_size=env.grid_size,
                episode_length=env.episode_length, max_requests=10000, neighbour_number=0,
                replay_buffer_size=replay_buffer_size, checkpoint_replay='none') for _ in range(2)]
    for value in pair:
        value.state_variant = env.state_variant
        value.recourse_variant = env.recourse_variant
    return attach_pair(env, pair)


def rollout(args, env, pair, method, *, training, progress_path=None):
    spec = METHODS[method]
    motion = {'integrated': env.simulate_motion, 'evfirst': env.simulate_motion_evfirst,
              'integrated_repair': env.simulate_motion_integrated_repair}[spec.operating_mode]
    reward = 0.
    fleet_rewards = {1: 0., 2: 0.}
    solver_seconds = 0.
    demand = {}
    for value in pair:
        value.deployment_edges_scored = value.deployment_edges_clipped = 0
    started = time.perf_counter()
    for step in range(min(env.episode_length, args.max_steps or env.episode_length)):
        for req in env.active_requests.values():
            demand[req.request_id] = (req.request_id, req.pickup, req.dropoff, req.created_time)
        actions, stored, stored_ev = motion(current_requests=list(env.active_requests.values()))
        solver_seconds += getattr(env, '_last_rebalancing_profile', {}).get('solver_time_sec', 0.)
        _, rewards, _, done, _ = env.step(actions, stored, stored_ev)
        reward += sum(map(float, rewards.values()))
        for vid, returned_reward in rewards.items():
            fleet_rewards[int(env.vehicles[vid]['type'])] += float(returned_reward)
        if training and (step + 1) % args.train_every == 0:
            for value, is_ev in ((pair[0], False), (pair[1], True)):
                loss = value.train_step(batch_size=args.batch_size, ifEV=is_ev)
                if not np.isfinite(loss):
                    raise AssertionError('nonfinite training loss')
        if progress_path is not None and (step == 0 or (step + 1) % 60 == 0 or done):
            lifecycle = env.request_lifecycle.metrics()
            progress = dict(method=method, phase='training' if training else 'testing',
                step=step + 1, total_steps=env.episode_length,
                reward=reward, completed_orders=len(env.completed_requests),
                recourse_number=lifecycle['same_epoch_aev_assignment_count'],
                ev_rejected_offer_count=lifecycle['ev_rejected_offer_count'],
                elapsed_seconds=time.perf_counter()-started,
                optimizer_steps_joint=[v.optimizer_steps_joint for v in pair])
            temporary = progress_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(progress, default=json_default, indent=2))
            temporary.replace(progress_path)
        if done:
            break
    stats = env.get_episode_stats()
    stats.update(env.request_lifecycle.metrics())
    stats.update(reward=float(reward), steps=step + 1, elapsed_seconds=time.perf_counter()-started)
    stats.update(reward_ev=fleet_rewards[1], reward_aev=fleet_rewards[2],
                 solver_runtime_seconds=solver_seconds,
                 deployment_clipping_rate=sum(v.deployment_edges_clipped for v in pair)
                    / max(1, sum(v.deployment_edges_scored for v in pair)))
    if not np.isclose(reward, sum(fleet_rewards.values())):
        raise AssertionError('Fleet rewards do not reconcile with the system return')
    stats['demand_hash'] = hashlib.sha256(json.dumps(sorted(demand.values())).encode()).hexdigest()
    if training:
        stats['optimizer_steps_joint'] = [v.optimizer_steps_joint for v in pair]
        stats['optimizer_steps_edge'] = [v.optimizer_steps_edge for v in pair]
        stats['optimizer_steps_queue'] = [v.optimizer_steps_queue for v in pair]
        stats['aev_follower_optimizer_steps'] = pair[0].optimizer_steps_joint if spec.operating_mode == 'evfirst' else 0
        stats['training_diagnostics'] = [v.joint_training_diagnostics for v in pair]
        stats.update(summarize_joint_targets([row for v in pair for row in v.joint_training_diagnostics]))
        stats['gradient_clipping_rate'] = sum(getattr(v, 'joint_gradient_clip_count', 0) for v in pair) / max(1, sum(v.optimizer_steps_joint for v in pair))
        rows = list(pair[0].joint_replay_buffer)
        stats['reward_ledgers'] = [dict(transition_id=r.transition_id, reward_ev=r.reward_ev,
            reward_aev=r.reward_aev, reward_system=r.reward_system,
            ledger=asdict(r.reward_ledger) if r.reward_ledger else None) for r in rows]
        stats['reward_ledger_scope'] = 'retained_replay_window'
        stats['replay_true_recourse_fraction'] = sum(any(is_true_same_epoch_recourse(e) for e in r.outcome_summary.events) for r in rows) / max(1, len(rows))
        stats['ordinary_aev_service_displacement_fixed_graph'] = sum(ordinary_service_displacement(r.stage2_graph) for r in rows)
        stats['charging_wait_reward_ledger'] = {key: sum(getattr(r.reward_ledger, key) for r in rows if r.reward_ledger)
            for key in ('aev_charging', 'aev_waiting', 'ev_rejection_penalty', 'request_expiry_penalty')}
        if sum(stats['optimizer_steps_joint']) == 0:
            raise AssertionError('No joint critic update; extend rollout or reduce train-every')
        if any(stats['optimizer_steps_edge']):
            raise AssertionError('Main recourse must not execute edge Bellman TD')
        if method == 'repair_only' and (pair[0].optimizer_steps_joint or pair[0].optimizer_steps_queue):
            raise AssertionError('Repair Only updated its follower')
    stats['method'] = method
    stats['configuration'] = method_metadata(spec.operating_mode, spec.variant)
    stats['ev_response_feature'] = 'off'
    return stats


def main():
    args = parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(arguments=vars(args), runs=[], paired_differences=[],
                   scope='smoke_not_convergence' if args.max_steps else 'full_episode')
    source_paths = [Path(__file__), *sorted((ROOT / 'src').rglob('*.py'))]
    summary['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    summary['inference_check_keys'] = INFERENCE_CHECK_KEYS
    summary['uncertainty_estimable'] = len(args.seeds) > 1
    for seed in args.seeds:
        hashes = set()
        demand_hashes = set()
        for method in args.methods:
            folder = args.output_dir / f'{method}-{seed}'
            folder.mkdir()
            print(f'Training and checkpoint inference: {method}, seed={seed}', flush=True)
            with (folder / 'run.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                env = build_env(args, seed, method, training=True)
                seed_everything(seed + 100000)
                pair = build_pair(env)
                initial_hash = weight_hash(pair)
                hashes.add(initial_hash)
                trained = rollout(args, env, pair, method, training=True)
                demand_hashes.add(trained['demand_hash'])
                save_pair(pair, folder / 'checkpoint.pt', dict(method=method, seed=seed))
                evaluation = build_env(args, seed + 90000, method, training=False)
                pair = attach_pair(evaluation, pair)
                expected = rollout(args, evaluation, pair, method, training=False)
                restored_env = build_env(args, seed + 90000, method, training=False)
                restored = build_pair(restored_env)
                payload = torch.load(folder / 'checkpoint.pt', weights_only=False, map_location='cpu')
                for value, saved in zip(restored, payload['learners']):
                    value.network.load_state_dict(saved['network'])
                    value.target_network.load_state_dict(saved['target'])
                    value.load_extra_checkpoint_state(saved['extra'])
                attach_pair(restored_env, restored)
                # Reproduce the simulator random stream; constructors consume
                # torch RNG only, while env initialization restores numpy/random.
                repeated = rollout(args, restored_env, restored, method, training=False)
                verify_checkpoint_stats(expected, repeated)
                result = dict(seed=seed, day_id=args.date, recourse_variant=method,
                    training=trained, evaluation=repeated, checkpoint_inference_verified=True,
                    initial_weight_hash=initial_hash, reward=repeated['reward'],
                    completed_orders=repeated.get('completed_orders', 0))
                (folder / 'results.json').write_text(json.dumps(result, default=json_default, indent=2))
                summary['runs'].append(result)
            (args.output_dir / 'summary.json').write_text(json.dumps(summary, default=json_default, indent=2))
        if len(hashes) != 1:
            raise AssertionError('Paired modes do not share initial neural weights')
        if len(demand_hashes) != 1:
            raise AssertionError('Paired modes do not share the generated demand stream')
    for baseline, treatment in [('evfirst_no_repair', 'repair_only'), ('repair_only', 'repair_learning'),
                                 ('repair_learning', 'recourse_macro'), ('recourse_macro', 'recourse_nested_q2'),
                                 ('no_repair', 'samitha')]:
        if baseline in args.methods and treatment in args.methods:
            for metric in ('reward', 'completed_orders'):
                difference = summarize_paired_crn_difference(summary['runs'], metric,
                    baseline_variant=baseline, treatment_variant=treatment)
                if len(args.seeds) < 2:
                    difference.update(ci_lower=None, ci_upper=None, uncertainty_estimable=False)
                summary['paired_differences'].append(difference)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, default=json_default, indent=2))
    print(args.output_dir / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
