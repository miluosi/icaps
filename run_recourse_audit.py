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
import resource
import sys
import time

import numpy as np
import torch

from run_acceptance_ablation import seed_everything, weight_hash, save_pair, json_default, attach_pair
from train_acceptance_model import make_environment, parse_args as environment_args
from src.acceptance_features import configure_acceptance_feature
from src.recourse.config import (
    ARCHITECTURE_CONTRASTS,
    CAUSAL_CONTRASTS,
    CAUSAL_PREDICTOR_VARIANTS,
    DIAGNOSTIC_CONTRASTS,
    METHOD_ALIASES,
    METHODS,
    PAPER_METHODS,
    canonical_method,
    method_metadata,
)
from src.recourse.contracts import (
    assert_method_event_contract,
    evaluate_method_event_contract,
)
from src.recourse.metrics import summarize_paired_crn_difference, summarize_joint_targets, ordinary_service_displacement
from src.recourse.types import LEARNER_VARIANTS, STATE_VARIANTS, is_true_same_epoch_recourse
from src.value_function_registry import get_value_function_class

ROOT = Path(__file__).resolve().parent
MAIN_METHODS = list(PAPER_METHODS)
INFERENCE_CHECK_KEYS = ('reward', 'reward_ev', 'reward_aev', 'completed_orders',
    'ev_offer_count', 'ev_rejected_offer_count', 'samitha_repair_pickup_count',
    'samitha_repair_completion_count', 'human_ev_charging_sessions', 'aev_charging_sessions',
    'all_vehicle_charging_sessions', 'completed_charging_sessions_with_duration_all')
PAIRED_METRICS = (
    'reward', 'completed_orders', 'ev_offer_count', 'ev_rejected_offer_count',
    'eligible_rejected_residual_count', 'same_epoch_aev_assignment_count',
    'aev_pickup_after_rejection_count', 'completion_after_rejection_count',
    'unrecovered_rejected_count', 'ultimately_unserved_count',
    'samitha_repair_pickup_count', 'samitha_repair_completion_count',
)


def verify_checkpoint_stats(expected, restored):
    for key in INFERENCE_CHECK_KEYS:
        if key not in expected or key not in restored:
            raise AssertionError(f'missing checkpoint inference metric: {key}')
        if expected[key] != restored[key]:
            raise AssertionError(f'checkpoint inference mismatch: {key}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['nyc', 'synthetic'], default='nyc')
    parser.add_argument('--methods', nargs='+', default=MAIN_METHODS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[71, 72, 73])
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--train-every', type=int, default=10)
    parser.add_argument('--state-variant', choices=STATE_VARIANTS, default='joint_state_separate_critics')
    parser.add_argument('--learner-variant', choices=LEARNER_VARIANTS,
                        default='optimization_anchored_residual')
    parser.add_argument('--rejection-logit-shift', type=float, default=0.0)
    parser.add_argument('--nyc-demand-scale', type=float, default=1.0)
    parser.add_argument('--station-capacity-scale', type=float, default=1.0)
    parser.add_argument('--battery-consumption-ratio', type=float, default=1.0)
    parser.add_argument('--initial-battery-mean', type=float, default=0.875)
    parser.add_argument('--charge-duration-scale', type=float, default=1.0)
    parser.add_argument('--energy-model', choices=['general_charging', 'fixed_swap'],
                        default='general_charging')
    parser.add_argument('--samitha-hold-rule', choices=['learned', 'fixed'], default='learned')
    parser.add_argument('--samitha-fixed-hold-fraction', type=float, default=0.0)
    parser.add_argument('--graph-reduction', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--mcmf-backend', choices=['primal_dual', 'ortools', 'gurobi_network'],
                        default='primal_dual')
    parser.add_argument('--checkpoint-replay', choices=['none', 'recent', 'full'], default='none')
    parser.add_argument('--checkpoint-replay-recent', type=int, default=5000)
    parser.add_argument('--date', default='2025-12-18')
    parser.add_argument('--dates', nargs='+', default=None)
    parser.add_argument(
        '--event-contract-mode', choices=['required', 'record', 'off'],
        default='required',
    )
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.num_ev < args.num_vehicles or min(args.batch_size, args.train_every) <= 0:
        parser.error('Require both fleets and positive training counts')
    if args.max_steps is not None and args.max_steps <= 1:
        parser.error('max-steps must allow linked joint transitions (>1)')
    if args.checkpoint_replay_recent <= 0:
        parser.error('checkpoint-replay-recent must be positive')
    if min(args.nyc_demand_scale, args.station_capacity_scale,
           args.battery_consumption_ratio, args.initial_battery_mean,
           args.charge_duration_scale) <= 0:
        parser.error('demand and energy sensitivity values must be positive')
    if args.initial_battery_mean > 1.0:
        parser.error('initial-battery-mean must not exceed 1')
    if not 0.0 <= args.samitha_fixed_hold_fraction <= 1.0:
        parser.error('samitha-fixed-hold-fraction must be in [0, 1]')
    if args.energy_model == 'fixed_swap':
        parser.error(
            'fixed_swap is not a parameter alias for the current charging system; '
            'select general_charging or implement/retrain a dedicated swap environment'
        )
    try:
        args.methods = [canonical_method(method) for method in args.methods]
    except ValueError as exc:
        parser.error(str(exc))
    args.dates = list(args.dates or [args.date])
    if ('recourse_aware' in args.methods and 'recourse_macro' in args.methods):
        parser.error('recourse_aware and recourse_macro are the same method')
    if (len(set(args.seeds)) != len(args.seeds)
            or len(set(args.methods)) != len(args.methods)
            or len(set(args.dates)) != len(args.dates)):
        parser.error('Duplicate seeds or methods')
    args.output_dir = args.output_dir or ROOT / 'results/recourse_credit' / datetime.now().strftime('%Y%m%d-%H%M%S')
    return args


def build_env(args, seed, method, *, training):
    day = args.date if training else getattr(args, 'test_date', args.date)
    settings = environment_args(['--environment', args.environment, '--num-vehicles', str(args.num_vehicles),
                                 '--num-ev', str(args.num_ev), '--date', day])
    for name in ('parquet_path', 'start_hour', 'stop_hour', 'epoch_length',
                 'nyc_demand_scale', 'station_capacity_scale',
                 'battery_consumption_ratio', 'initial_battery_mean',
                 'charge_duration_scale'):
        if getattr(args, name, None) is not None:
            setattr(settings, name, getattr(args, name))
    env = make_environment(settings, seed)
    spec = METHODS[method]
    env.configure_recourse_experiment(
        spec.variant,
        rejection_logit_shift=float(getattr(args, 'rejection_logit_shift', 0.0)),
        common_random_numbers=True,
    )
    configure_acceptance_feature(env, 'off')
    env.state_variant = args.state_variant
    env.learner_variant = str(getattr(
        args, 'learner_variant', 'optimization_anchored_residual'
    ))
    env.energy_model = str(getattr(args, 'energy_model', 'general_charging'))
    env.samitha_hold_rule = str(getattr(args, 'samitha_hold_rule', 'learned'))
    env.samitha_fixed_hold_fraction = float(
        getattr(args, 'samitha_fixed_hold_fraction', 0.0)
    )
    env.adp_value, env.evaluatemode = 1., not training
    # Offer-keyed CRN includes run_id: it must be identical across methods.
    env.recourse_run_id = f'recourse-audit-{seed}'
    env.recourse_method = method
    # Publication preset: every recourse architecture sees the same exact
    # assignment oracle. Solver audits vary these axes in a separate runner.
    env.mcmf_solver = 'exact'
    env.useauction = False
    env.mcmf_backend = str(getattr(args, 'mcmf_backend', 'primal_dual'))
    env.mcmf_graph_reduction = bool(getattr(args, 'graph_reduction', True))
    env.mcmf_verify = True
    env.mcmf_cost_scale = 10_000
    env.mcmf_strict = True
    env.target_solver_policy = 'same_as_rollout_exact'
    if hasattr(env, 'set_request_generation_seed'):
        env.set_request_generation_seed(seed)
    return env


def build_pair(
    env,
    *,
    replay_buffer_size=5000,
    checkpoint_replay='none',
    checkpoint_replay_recent=5000,
):
    cls = get_value_function_class(env.learner_variant)
    pair = [cls(env=env, num_vehicles=len(env.vehicles), grid_size=env.grid_size,
                episode_length=env.episode_length, max_requests=10000, neighbour_number=0,
                replay_buffer_size=replay_buffer_size,
                checkpoint_replay=checkpoint_replay,
                checkpoint_replay_recent=checkpoint_replay_recent) for _ in range(2)]
    for value in pair:
        value.state_variant = env.state_variant
        value.recourse_variant = env.recourse_variant
        if env.recourse_variant in CAUSAL_PREDICTOR_VARIANTS:
            value.predictor_variant = 'p0'
            value.freeze_causal_predictors = True
            value.queue_predictor_trained = False
            if hasattr(value, 'post_demand_predictor_trained'):
                value.post_demand_predictor_trained = False
    return attach_pair(env, pair)


def rollout(
    args,
    env,
    pair,
    method,
    *,
    training,
    progress_path=None,
    capture_random_events=False,
):
    spec = METHODS[method]
    motion = {'integrated': env.simulate_motion_integrated_control, 'evfirst': env.simulate_motion_evfirst,
              'integrated_repair': env.simulate_motion_integrated_repair}[spec.operating_mode]
    reward = 0.
    fleet_rewards = {1: 0., 2: 0.}
    solver_seconds = 0.
    runtime_profiles = []
    decision_latencies = []
    observation_node_counts = []
    episode_ordinary_displacement = 0
    demand = {}
    action_trace = []
    for value in pair:
        value.deployment_edges_scored = value.deployment_edges_clipped = 0
    started = time.perf_counter()
    for step in range(min(env.episode_length, args.max_steps or env.episode_length)):
        for req in env.active_requests.values():
            demand[req.request_id] = (req.request_id, req.pickup, req.dropoff, req.created_time)
        actions, stored, stored_ev = motion(current_requests=list(env.active_requests.values()))
        profile = dict(getattr(env, '_last_rebalancing_profile', {}) or {})
        simulation_profile = dict(getattr(env, '_last_simulation_profile', {}) or {})
        runtime_profiles.append(profile)
        decision_latencies.append(float(
            simulation_profile.get('total_time_sec', profile.get('solve_total_time_sec', 0.0)) or 0.0
        ))
        integrated_graphs = tuple(
            graph for graph in getattr(env, '_last_integrated_repair_graphs', ())
            if graph is not None
        )
        for graph in integrated_graphs:
            action_trace.extend(graph.selected_edge_ids)
            observation_node_counts.append(len(graph.state.vehicles))
        graph = getattr(env, '_last_feasible_graph_snapshot', None)
        if graph is not None and not integrated_graphs:
            action_trace.extend(graph.selected_edge_ids)
            observation_node_counts.append(len(graph.state.vehicles))
        displacement_graphs = integrated_graphs[-1:] or ((graph,) if graph is not None else ())
        episode_ordinary_displacement += sum(
            ordinary_service_displacement(item) for item in displacement_graphs
        )
        solver_seconds += getattr(env, '_last_rebalancing_profile', {}).get('solver_time_sec', 0.)
        _, rewards, _, done, _ = env.step(actions, stored, stored_ev)
        reward += sum(map(float, rewards.values()))
        for vid, returned_reward in rewards.items():
            fleet_rewards[int(env.vehicles[vid]['type'])] += float(returned_reward)
        if (training and env.learner_variant != 'structured_myopic'
                and (step + 1) % args.train_every == 0):
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
    stats['ordinary_aev_service_displacement_fixed_graph'] = episode_ordinary_displacement
    def profile_sum(key):
        return float(sum(float(row.get(key, 0.0) or 0.0) for row in runtime_profiles))

    positive_latencies = np.asarray(decision_latencies, dtype=float)
    stats.update(
        candidate_generation_runtime_seconds=profile_sum('qmatrix_time_sec'),
        neural_edge_scoring_runtime_seconds=profile_sum('qvalue_time_sec'),
        graph_serialization_runtime_seconds=profile_sum('graph_serialization_time_sec'),
        graph_reduction_runtime_seconds=profile_sum('exact_build_time_sec'),
        exact_solve_runtime_seconds=profile_sum('exact_solve_time_sec'),
        total_decision_runtime_seconds=float(sum(decision_latencies)),
        decision_latency_p50_seconds=(float(np.quantile(positive_latencies, .50)) if positive_latencies.size else 0.0),
        decision_latency_p90_seconds=(float(np.quantile(positive_latencies, .90)) if positive_latencies.size else 0.0),
        decision_latency_p95_seconds=(float(np.quantile(positive_latencies, .95)) if positive_latencies.size else 0.0),
        decision_latency_p99_seconds=(float(np.quantile(positive_latencies, .99)) if positive_latencies.size else 0.0),
        feasible_edge_count=int(sum(
            float(row.get('feasible_request_edges', 0.0) or 0.0)
            + float(row.get('feasible_charging_edges', 0.0) or 0.0)
            + float(row.get('feasible_zone_edges', 0.0) or 0.0)
            for row in runtime_profiles
        )),
        exact_original_edge_count=int(profile_sum('exact_original_edges')),
        exact_reduced_edge_count=int(profile_sum('exact_reduced_edges')),
        solver_fallback_count=int(profile_sum('solver_fallback_used')),
        max_solver_objective_gap=max(
            [float(row.get('solver_objective_gap', 0.0) or 0.0) for row in runtime_profiles]
            or [0.0]
        ),
        observation_node_count_mean=(float(np.mean(observation_node_counts))
                                     if observation_node_counts else 0.0),
    )
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    stats['process_peak_memory_mb'] = rss / (1024.0 * 1024.0 if sys.platform == 'darwin' else 1024.0)
    stats['peak_memory_scope'] = 'worker_process_lifetime'
    original_edges = stats['exact_original_edge_count']
    stats['graph_reduction_ratio'] = (
        1.0 - stats['exact_reduced_edge_count'] / original_edges
        if original_edges else 0.0
    )
    stats.update(reward_ev=fleet_rewards[1], reward_aev=fleet_rewards[2],
                 solver_runtime_seconds=solver_seconds,
                 deployment_clipping_rate=sum(v.deployment_edges_clipped for v in pair)
                    / max(1, sum(v.deployment_edges_scored for v in pair)))
    if not np.isclose(reward, sum(fleet_rewards.values())):
        raise AssertionError('Fleet rewards do not reconcile with the system return')
    stats['demand_hash'] = hashlib.sha256(json.dumps(sorted(demand.values())).encode()).hexdigest()
    stats['selected_action_trace_hash'] = hashlib.sha256(
        json.dumps(action_trace).encode()
    ).hexdigest()
    random_rows = sorted(
        (tuple(key), tuple(sorted(value.items())))
        for key, value in getattr(env, '_last_offer_realizations', {}).items()
    )
    event_rows = sorted(
        (tuple(key), float(value))
        for key, value in getattr(env, '_recourse_random_events', {}).items()
    )
    combined_random_rows = {'offers': random_rows, 'events': event_rows}
    stats['random_stream_hash'] = hashlib.sha256(
        json.dumps(combined_random_rows, default=json_default).encode()
    ).hexdigest()
    stats['random_event_count'] = len(event_rows)
    def random_stream_name(key):
        return str(key[7] if key and key[0] == 'normal' else key[-1])

    streams = sorted({random_stream_name(key) for key, _value in event_rows})
    stats['random_event_stream_hashes'] = {
        stream: hashlib.sha256(json.dumps([
            (key, value) for key, value in event_rows
            if random_stream_name(key) == stream
        ], default=json_default).encode()).hexdigest()
        for stream in streams
    }
    if capture_random_events:
        # Kept in memory only by the CRN audit caller and removed before JSON
        # serialization.  This permits an exact common-event comparison
        # without bloating production result files with every random draw.
        stats['_random_event_values'] = dict(event_rows)
    if training:
        stats['optimizer_steps_joint'] = [v.optimizer_steps_joint for v in pair]
        stats['optimizer_steps_edge'] = [v.optimizer_steps_edge for v in pair]
        stats['optimizer_steps_queue'] = [v.optimizer_steps_queue for v in pair]
        stats['aev_follower_optimizer_steps'] = pair[0].optimizer_steps_joint if spec.operating_mode == 'evfirst' else 0
        stats['training_diagnostics'] = [v.joint_training_diagnostics for v in pair]
        diagnostics = [row for value in pair for row in value.joint_training_diagnostics]
        stats['macro_leader_target_count'] = sum(
            row.get('phase') == 'ev_leader'
            and row.get('recourse_target_family') == 'macro_realized'
            for row in diagnostics
        )
        stats['nested_leader_target_count'] = sum(
            row.get('phase') == 'ev_leader'
            and row.get('recourse_target_family') == 'nested_follower'
            for row in diagnostics
        )
        stats['follower_target_query_count'] = stats['nested_leader_target_count']
        rows = list(pair[0].joint_replay_buffer)
        stats['aev_stage_graph_count'] = sum(row.aev_stage_graph is not None for row in rows)
        stats['aev_learned_score_difference_count'] = sum(
            abs(edge.collection_score - edge.structured_score) > 1e-8
            for row in rows if row.aev_stage_graph is not None
            for edge in row.aev_stage_graph.edges
        )
        stats.update(summarize_joint_targets([row for v in pair for row in v.joint_training_diagnostics]))
        stats['gradient_clipping_rate'] = sum(getattr(v, 'joint_gradient_clip_count', 0) for v in pair) / max(1, sum(v.optimizer_steps_joint for v in pair))
        stats['reward_ledgers'] = [dict(transition_id=r.transition_id, reward_ev=r.reward_ev,
            reward_aev=r.reward_aev, reward_system=r.reward_system,
            ledger=asdict(r.reward_ledger) if r.reward_ledger else None) for r in rows]
        stats['reward_ledger_scope'] = 'retained_replay_window'
        stats['replay_true_recourse_fraction'] = sum(any(is_true_same_epoch_recourse(e) for e in r.outcome_summary.events) for r in rows) / max(1, len(rows))
        stats['retained_replay_ordinary_aev_service_displacement_fixed_graph'] = sum(
            ordinary_service_displacement(r.stage2_graph) for r in rows
        )
        stats['charging_wait_reward_ledger'] = {key: sum(getattr(r.reward_ledger, key) for r in rows if r.reward_ledger)
            for key in ('aev_charging', 'aev_waiting', 'ev_rejection_penalty', 'request_expiry_penalty')}
        modules = ('network', 'critic2', 'graph_encoder', 'mixer', 'actor')
        stats['model_parameter_count'] = int(sum(
            parameter.numel()
            for value in pair
            for name in modules
            for parameter in getattr(value, name).parameters()
        ))
        stats['effective_trainable_parameter_count'] = (
            0 if env.learner_variant == 'structured_myopic'
            else stats['model_parameter_count']
        )
        stats['gradient_update_count'] = int(sum(stats['optimizer_steps_joint']))
        if env.learner_variant != 'structured_myopic' and sum(stats['optimizer_steps_joint']) == 0:
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
                   crn_common_event_checks=[],
                   evaluation_scope='same_day_new_seed_checkpoint_check',
                   scope='smoke_not_convergence' if args.max_steps else 'full_episode')
    source_paths = [Path(__file__), *sorted((ROOT / 'src').rglob('*.py'))]
    summary['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    summary['inference_check_keys'] = INFERENCE_CHECK_KEYS
    summary['uncertainty_estimable'] = len(args.seeds) > 1
    for seed in args.seeds:
        for day in args.dates:
            args.date = day
            hashes = set()
            demand_hashes = set()
            reference_method = None
            reference_events = None
            for method in args.methods:
                day_tag = day.replace('-', '')
                folder = args.output_dir / f'{day_tag}-{method}-{seed}'
                folder.mkdir()
                print(f'Training and checkpoint inference: {method}, seed={seed}, day={day}', flush=True)
                with (folder / 'run.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                    env = build_env(args, seed, method, training=True)
                    seed_everything(seed + 100000)
                    pair = build_pair(
                        env,
                        checkpoint_replay=args.checkpoint_replay,
                        checkpoint_replay_recent=args.checkpoint_replay_recent,
                    )
                    initial_hash = weight_hash(pair)
                    hashes.add(initial_hash)
                    trained = rollout(args, env, pair, method, training=True)
                    demand_hashes.add(trained['demand_hash'])
                    spec = METHODS[method]
                    save_pair(pair, folder / 'checkpoint.pt', dict(
                        method=method, seed=seed, day_id=day,
                        **method_metadata(spec.operating_mode, spec.variant),
                        state_variant=env.state_variant,
                        learner_variant=env.learner_variant,
                        solver_config=dict(
                            rollout_solver=getattr(env, 'mcmf_solver', 'exact'),
                            backend=getattr(env, 'mcmf_backend', 'primal_dual'),
                            graph_reduction=getattr(env, 'mcmf_graph_reduction', True),
                            verify=getattr(env, 'mcmf_verify', True),
                            strict=getattr(env, 'mcmf_strict', True),
                            cost_scale=getattr(env, 'mcmf_cost_scale', 10_000),
                            target_policy=getattr(env, 'target_solver_policy', 'same_as_rollout_exact'),
                        ),
                    ))
                    evaluation = build_env(args, seed + 90000, method, training=False)
                    pair = attach_pair(evaluation, pair)
                    expected = rollout(
                        args, evaluation, pair, method, training=False,
                        capture_random_events=True,
                    )
                    restored_env = build_env(args, seed + 90000, method, training=False)
                    restored = build_pair(restored_env)
                    payload = torch.load(folder / 'checkpoint.pt', weights_only=False, map_location='cpu')
                    if payload.get('checkpoint_schema_version') != 2:
                        raise ValueError('unsupported paired checkpoint schema')
                    if len(payload.get('learners', ())) != 2:
                        raise ValueError('paired checkpoint must contain exactly two learners')
                    if payload.get('metadata', {}).get('method') != method:
                        raise ValueError('checkpoint method mismatch before tensor loading')
                    for value, saved in zip(restored, payload['learners']):
                        value.network.load_state_dict(saved['network'])
                        value.target_network.load_state_dict(saved['target'])
                        if saved.get('optimizer'):
                            value.optimizer.load_state_dict(saved['optimizer'])
                        value.load_extra_checkpoint_state(saved['extra'])
                    attach_pair(restored_env, restored)
                    repeated = rollout(
                        args, restored_env, restored, method, training=False,
                        capture_random_events=True,
                    )
                    expected_events = expected.pop('_random_event_values')
                    repeated_events = repeated.pop('_random_event_values')
                    if expected_events != repeated_events:
                        raise AssertionError(
                            'checkpoint inference changed keyed random events'
                        )
                    if reference_events is None:
                        reference_method = method
                        reference_events = repeated_events
                    else:
                        common_keys = set(reference_events) & set(repeated_events)
                        mismatches = [
                            key for key in common_keys
                            if reference_events[key] != repeated_events[key]
                        ]
                        if mismatches:
                            raise AssertionError(
                                'CRN common-event values differ across methods: '
                                f'{reference_method} vs {method}, first={mismatches[0]!r}'
                            )
                        summary['crn_common_event_checks'].append({
                            'seed': seed,
                            'day_id': day,
                            'baseline': reference_method,
                            'treatment': method,
                            'common_event_count': len(common_keys),
                            'mismatch_count': 0,
                        })
                    verify_checkpoint_stats(expected, repeated)
                    for key in ('selected_action_trace_hash', 'random_stream_hash'):
                        if expected[key] != repeated[key]:
                            raise AssertionError(f'checkpoint inference mismatch: {key}')
                    contract_stats = dict(repeated)
                    for key in (
                        'aev_follower_optimizer_steps', 'aev_stage_graph_count',
                        'aev_learned_score_difference_count',
                        'macro_leader_target_count', 'nested_leader_target_count',
                        'follower_target_query_count',
                    ):
                        contract_stats[key] = trained.get(key, 0)
                    contract = evaluate_method_event_contract(method, contract_stats)
                    if args.event_contract_mode == 'required':
                        assert_method_event_contract(method, contract_stats)
                    result = dict(seed=seed, day_id=day, recourse_variant=method,
                        training=trained, evaluation=repeated,
                        event_contract=contract.as_dict(),
                        checkpoint_inference_verified=True,
                        initial_weight_hash=initial_hash, reward=repeated['reward'],
                        completed_orders=repeated.get('completed_orders', 0),
                        **{key: repeated.get(key, 0) for key in PAIRED_METRICS
                           if key not in {'reward', 'completed_orders'}})
                    (folder / 'results.json').write_text(json.dumps(result, default=json_default, indent=2))
                    summary['runs'].append(result)
                (args.output_dir / 'summary.json').write_text(json.dumps(summary, default=json_default, indent=2))
            if len(hashes) != 1:
                raise AssertionError('Paired modes do not share initial neural weights')
            if len(demand_hashes) != 1:
                raise AssertionError('Paired modes do not share the generated demand stream')
    for baseline, treatment in (
        *CAUSAL_CONTRASTS, *ARCHITECTURE_CONTRASTS, *DIAGNOSTIC_CONTRASTS
    ):
        if baseline in args.methods and treatment in args.methods:
            for metric in PAIRED_METRICS:
                difference = summarize_paired_crn_difference(summary['runs'], metric,
                    baseline_variant=baseline, treatment_variant=treatment)
                if len(args.seeds) < 2:
                    difference.update(ci_lower=None, ci_upper=None, uncertainty_estimable=False)
                summary['paired_differences'].append(difference)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, default=json_default, indent=2))
    print(args.output_dir / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
