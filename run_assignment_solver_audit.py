"""Evaluate one fixed recourse checkpoint across assignment solver settings."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from run_acceptance_ablation import attach_pair, weight_hash
from run_recourse_audit import build_env, build_pair, rollout
from run_recourse_day import parse_args as day_args, validate_checkpoint_payload
from src.recourse.config import PAPER_METHODS
from src.recourse.types import LEARNER_VARIANTS


ROOT = Path(__file__).resolve().parent
EXACT_SOLVERS = {'exact'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recourse-method', choices=PAPER_METHODS, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--parquet-path', type=Path, required=True)
    parser.add_argument('--solvers', nargs='+', choices=['exact', 'auction'], default=['exact'])
    parser.add_argument('--backends', nargs='+', choices=['primal_dual', 'ortools', 'gurobi_network'], default=['primal_dual'])
    parser.add_argument('--target-oracle-backend', choices=['primal_dual', 'ortools', 'gurobi_network'], default='primal_dual')
    parser.add_argument('--graph-reduction', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--verify', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--dates', nargs='+', required=True)
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--epoch-length', type=float, default=30.0)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if ('auction' in args.solvers
            and args.recourse_method in {'no_repair', 'samitha'}):
        parser.error(
            'Integrated/Samitha use the shared exact stage-0 oracle; '
            'do not label that rollout as auction until an approximate stage '
            'adapter exists'
        )
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.dates)) != len(args.dates):
        parser.error('seeds and dates must be unique')
    return args


def _settings(args, seed, day, learner_variant):
    # Reuse the production validation and environment namespace. The dummy
    # training day is kept disjoint; this runner performs evaluation only.
    from datetime import date, timedelta
    test_day = date.fromisoformat(day)
    train_day = (test_day - timedelta(days=1)).isoformat()
    settings = day_args([
        '--methods', args.recourse_method,
        '--train-date', train_day, '--test-date', day,
        '--seed', str(seed - 1), '--test-seed', str(seed),
        '--parquet-path', str(args.parquet_path),
        '--num-vehicles', str(args.num_vehicles), '--num-ev', str(args.num_ev),
        '--epoch-length', str(args.epoch_length), '--event-contract-mode', 'off',
        '--learner-variant', learner_variant,
        *(['--smoke-steps', str(args.max_steps)] if args.max_steps else []),
    ])
    return settings


def main(argv=None):
    args = parse_args(argv)
    payload = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    if payload.get('metadata', {}).get('method') != args.recourse_method:
        raise ValueError('checkpoint method does not match --recourse-method')
    learner_variant = payload.get('metadata', {}).get('learner_variant')
    if learner_variant not in LEARNER_VARIANTS:
        raise ValueError(
            f'checkpoint learner {learner_variant!r} is not one of {LEARNER_VARIANTS}'
        )
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for seed in args.seeds:
        for day in args.dates:
            for solver in args.solvers:
                backend_values = args.backends if solver in EXACT_SOLVERS else ['auction']
                for backend in backend_values:
                    settings = _settings(args, seed, day, learner_variant)
                    env = build_env(settings, seed, args.recourse_method, training=False)
                    validate_checkpoint_payload(payload, args.recourse_method, env)
                    env.mcmf_solver = solver
                    env.useauction = solver == 'auction'
                    env.mcmf_backend = (
                        backend if solver in EXACT_SOLVERS
                        else args.target_oracle_backend
                    )
                    env.target_oracle_backend = args.target_oracle_backend
                    env.mcmf_graph_reduction = bool(args.graph_reduction)
                    env.mcmf_verify = bool(args.verify)
                    env.target_solver_policy = (
                        'same_as_rollout_exact' if solver in EXACT_SOLVERS
                        else 'exact_oracle_for_approximate_rollout'
                    )
                    pair = build_pair(env)
                    for value, saved in zip(pair, payload['learners']):
                        value.network.load_state_dict(saved['network'])
                        value.target_network.load_state_dict(saved['target'])
                        if saved.get('optimizer'):
                            value.optimizer.load_state_dict(saved['optimizer'])
                        value.load_extra_checkpoint_state(saved['extra'])
                    attach_pair(env, pair)
                    before = weight_hash(pair)
                    stats = rollout(settings, env, pair, args.recourse_method, training=False)
                    if weight_hash(pair) != before:
                        raise AssertionError('solver audit performed a training update')
                    rows.append(dict(
                        method=args.recourse_method, seed=seed, day_id=day,
                        solver=solver, backend=backend,
                        target_oracle_backend=(
                            env.mcmf_backend if solver in EXACT_SOLVERS
                            else args.target_oracle_backend
                        ),
                        exact=solver in EXACT_SOLVERS,
                        training_during_test=False,
                        graph_reduction=args.graph_reduction, verify=args.verify,
                        target_solver_policy=env.target_solver_policy,
                        checkpoint_weight_hash=before,
                        reward=stats['reward'],
                        completed_orders=stats.get('completed_orders', 0),
                        selected_action_trace_hash=stats['selected_action_trace_hash'],
                        solver_runtime_seconds=stats.get('solver_runtime_seconds', 0),
                    ))
    result = dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha256(args.checkpoint.read_bytes()).hexdigest(),
        fixed_checkpoint=True, rows=rows,
    )
    output = args.output_dir / 'solver_audit.json'
    output.write_text(json.dumps(result, indent=2))
    print(output)


if __name__ == '__main__':
    main()
