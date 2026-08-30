"""Paired Integrated learning ablation: frozen EV acceptance input on/off.

Both arms use the same initial weights, demand seeds, keyed acceptance draws,
training budget, exact MCMF projection and unmodified environment rewards.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from src.acceptance_features import configure_acceptance_feature
from src.acceptance_model import collect_offers
from src.rejection_anchor import response_graph_diagnostics
from src.recourse.critics import wire_recourse_critics
from src.value_function_registry import get_value_function_class
from train_acceptance_model import make_environment, parse_args as environment_args

ROOT = Path(__file__).resolve().parent
LEARNERS = ['integrated_directq', 'optimization_anchored_residual']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['synthetic', 'nyc'], default='synthetic')
    parser.add_argument('--learners', nargs='+', choices=LEARNERS, default=LEARNERS)
    parser.add_argument('--train-seeds', nargs='+', type=int, default=[41, 42, 43])
    parser.add_argument('--test-seeds', nargs='+', type=int, default=list(range(9001, 9006)))
    parser.add_argument('--episodes', type=int, default=6)
    parser.add_argument('--train-every', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2, help='Joint transitions per TD update, not individual edges')
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=None, help='Smoke tests only; formal runs use complete episodes')
    parser.add_argument('--acceptance-model', type=Path, required=True,
                        help='Explicit calibrated rejected=1 neural v3 checkpoint; older models require retraining')
    parser.add_argument('--ev-response-anchor', choices=['auto', 'off', 'expected_immediate'], default='auto')
    parser.add_argument('--ev-response-critic-input', choices=['q_mask', 'none'], default='q_mask')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if min(args.episodes, args.train_every, args.batch_size, args.torch_threads) <= 0:
        parser.error('Training counts must be positive')
    if len(set(args.train_seeds)) != len(args.train_seeds) or len(set(args.test_seeds)) != len(args.test_seeds):
        parser.error('Seeds must be unique')
    training_rollouts = {50000 + seed * 100 + ep for seed in args.train_seeds for ep in range(args.episodes)}
    if training_rollouts & set(args.test_seeds):
        parser.error('Training rollout seeds and test seeds overlap')
    if args.output_dir is None:
        args.output_dir = ROOT / 'results/acceptance_ablation' / datetime.now().strftime('%Y%m%d-%H%M%S')
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(args, seed, arm, learner, training):
    settings = environment_args(['--environment', args.environment, '--num-vehicles', str(args.num_vehicles),
                                 '--num-ev', str(args.num_ev)])
    env = make_environment(settings, seed)
    request_seeder = getattr(env, 'set_request_generation_seed', None)
    if callable(request_seeder):
        request_seeder(seed)
    configure_acceptance_feature(env, arm, args.acceptance_model if arm == 'predicted' else None,
        anchor=args.ev_response_anchor, critic_input=args.ev_response_critic_input)
    env.configure_recourse_experiment('legacy', common_random_numbers=True)
    env.state_variant = 'joint_state_separate_critics'
    env.learner_variant = learner
    env.evaluatemode = not training
    env.adp_value = 1.0
    env.recourse_run_id = f'acceptance-ablation-{seed}'
    return env


def build_pair(env, learner):
    cls = get_value_function_class(learner)
    pair = [cls(env=env, num_vehicles=len(env.vehicles), grid_size=env.grid_size, device='cpu',
                episode_length=env.episode_length, max_requests=10000, replay_buffer_size=5000,
                checkpoint_replay='none', zone_distribution_mode=learner, neighbour_number=0)
            for _ in range(2)]
    for vf in pair:
        vf.state_variant = env.state_variant
        vf.learner_variant = learner
        vf.recourse_variant = 'legacy'
    return attach_pair(env, pair)


def attach_pair(env, pair):
    for vf in pair:
        vf.env = env
        vf._graph_cache_key = vf._target_graph_cache_key = None
        vf._graph_cache = vf._target_graph_cache = None
        vf._target_component_cache.clear()
        vf.network.eval()
        vf.critic2.eval()
    aev, ev = wire_recourse_critics(*pair, state_variant=env.state_variant)
    env.set_value_function(aev)
    env.set_value_function_ev(ev)
    return [aev, ev]


def weight_hash(pair, *, remove_acceptance=False):
    """Hash every tensor used by rollout, target, actor, or auxiliary inference."""
    digest = hashlib.sha256()
    module_names = (
        'network', 'critic2', 'target_network', 'target_critic2',
        'graph_encoder', 'target_graph_encoder', 'mixer', 'target_mixer',
        'actor', 'queue_predictor', 'target_queue_predictor',
        'post_demand_predictor', 'target_post_demand_predictor',
    )
    acceptance_modules = {'network', 'critic2', 'target_network', 'target_critic2'}
    for learner_index, vf in enumerate(pair):
        digest.update(f'learner:{learner_index}'.encode())
        for module_name in module_names:
            module = getattr(vf, module_name, None)
            if module is None:
                continue
            for name, value in sorted(module.state_dict().items()):
                if not hasattr(value, 'detach'):
                    continue
                tensor = value.detach().cpu().contiguous()
                if (remove_acceptance and vf.acceptance_input_enabled
                        and module_name in acceptance_modules
                        and name.endswith('net.0.weight')):
                    i = vf.acceptance_input_index
                    tensor = torch.cat([tensor[:, :i], tensor[:, i+2:]], 1)
                digest.update(module_name.encode())
                digest.update(name.encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.numpy().tobytes())
        log_alpha = getattr(vf, 'log_alpha', None)
        if hasattr(log_alpha, 'detach'):
            tensor = log_alpha.detach().cpu().contiguous()
            digest.update(b'log_alpha')
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_pair(pair, path, metadata):
    payload = {
        'checkpoint_schema_version': 2,
        'metadata': metadata,
        'learners': [],
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        },
    }
    for vf in pair:
        payload['learners'].append(dict(network=vf.network.state_dict(), target=vf.target_network.state_dict(),
                                       optimizer=vf.optimizer.state_dict(), extra=vf.extra_checkpoint_state()))
    torch.save(payload, path)


def load_pair(env, learner, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    pair = build_pair(env, learner)
    if payload.get('checkpoint_schema_version') != 2:
        raise ValueError('unsupported paired checkpoint schema')
    if len(payload.get('learners', ())) != len(pair):
        raise ValueError('paired checkpoint learner count mismatch')
    for vf, saved in zip(pair, payload['learners']):
        vf.load_acceptance_checkpoint_state(saved['extra'])
        vf.network.load_state_dict(saved['network'])
        vf.target_network.load_state_dict(saved['target'])
        if saved.get('optimizer'):
            vf.optimizer.load_state_dict(saved['optimizer'])
        vf.load_extra_checkpoint_state(saved['extra'])
    rng_state = payload.get('rng_state', {})
    if rng_state:
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
    return attach_pair(env, pair)


def rollout(args, env, pair, *, training, seed, directory):
    started = time.perf_counter()
    reward, updates, solver_calls = 0.0, 0, 0
    losses = []
    demand_rows = {}
    acceptance_inputs = []
    response_diagnostics = []
    def check_input(module, inputs):
        if pair[1].acceptance_input_enabled:
            p = inputs[0][:, pair[1].acceptance_input_index].detach().cpu().numpy()
            mask = inputs[0][:, pair[1].acceptance_input_index + 1].detach().cpu().numpy()
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise AssertionError('Invalid EV probability feature')
            if np.any((mask != 0) & (mask != 1)) or np.any(p[mask == 0] != 0):
                raise AssertionError('Invalid EV response mask')
            acceptance_inputs.extend(p[p > 0].tolist())
    hook = pair[1].network.register_forward_pre_hook(check_input)
    try:
        with collect_offers(env, episode_id=str(seed), seed=seed) as offers:
            for step in range(min(env.episode_length, args.max_steps or env.episode_length)):
                for req in env.active_requests.values():
                    demand_rows[int(req.request_id)] = [int(req.request_id), int(req.pickup), int(req.dropoff), float(req.created_time)]
                previous_result = getattr(env, 'mcmf_last_result', None)
                actions, stored, stored_ev = env.simulate_motion(
                    agents=[], current_requests=list(env.active_requests.values()), rebalance=True)
                graph = getattr(env, '_last_feasible_graph_snapshot', None)
                if graph is not None:
                    response_diagnostics.append(response_graph_diagnostics(graph))
                result = getattr(env, 'mcmf_last_result', None)
                if result is not None and result is not previous_result:
                    if not result['optimal'] or result['solver_fallback_used']:
                        raise AssertionError(f'Non-exact assignment: {result}')
                    solver_calls += 1
                _, rewards, _, done, _ = env.step(actions, stored, stored_ev)
                reward += float(sum(rewards.values()))
                if training and (step + 1) % args.train_every == 0:
                    before = pair[0].optimizer_steps_joint
                    loss = pair[0].train_step(batch_size=args.batch_size, ifEV=False)
                    # Integrated joint TD already updates both critics. This
                    # second call trains the EV auxiliaries, not a second TD.
                    loss += pair[1].train_step(batch_size=args.batch_size, ifEV=True)
                    if not np.isfinite(loss):
                        raise AssertionError('Nonfinite TD loss')
                    updates += pair[0].optimizer_steps_joint - before
                    losses.append(float(loss))
                if done:
                    break
            for req in env.active_requests.values():
                demand_rows[int(req.request_id)] = [int(req.request_id), int(req.pickup), int(req.dropoff), float(req.created_time)]
    finally:
        hook.remove()
    stats = env.get_episode_stats()
    if stats.get('request_lifecycle_gap', 0) != 0:
        raise AssertionError('Generated/completed/expired/active requests do not reconcile')
    rejected = [row for row in offers if not row['accepted']]
    rejected_ids = {row['request_id'] for row in rejected}
    completed = len(env.completed_requests)
    generated = int(stats.get('total_generated_requests', env.whole_req_num))
    if training and not updates:
        raise AssertionError('No Integrated joint TD update occurred')
    if pair[1].acceptance_input_enabled and offers and not acceptance_inputs:
        raise AssertionError('EV acceptance predictor was not consumed by the critic')
    with (directory / f'offers-{seed}.jsonl').open('w') as stream:
        for row in offers:
            stream.write(json.dumps(row) + '\n')
    stats_path = directory / f'stats-{seed}.json'
    stats_path.write_text(json.dumps(stats, indent=2, default=json_default) + '\n')
    (directory / f'response-diagnostics-{seed}.json').write_text(json.dumps(response_diagnostics, indent=2) + '\n')
    row = dict(seed=seed, training=training, steps=step + 1, reward=reward,
               generated_requests=generated, ev_offers=len(offers), ev_accepted_offers=len(offers)-len(rejected),
               rejected_offers=len(rejected), unique_rejected_requests=len(rejected_ids),
               ev_rejection_rate=len(rejected)/max(1, len(offers)),
               completed_orders=completed, completed_ev_orders=int(stats['completed_ev_orders']),
               completed_aev_orders=completed-int(stats['completed_ev_orders']),
               completion_rate=completed/max(1, generated), expired_requests=int(stats['expired_request_count']),
               active_requests=len(env.active_requests), lifecycle_gap=int(stats.get('request_lifecycle_gap', 0)),
               exact_mcmf_calls=solver_calls, joint_updates=updates,
               aev_optimizer_steps=pair[0].optimizer_steps_joint, ev_optimizer_steps=pair[1].optimizer_steps_joint,
               mean_training_loss=float(np.mean(losses)) if losses else None,
               acceptance_input_count=len(acceptance_inputs),
               acceptance_input_mean=float(np.mean(acceptance_inputs)) if acceptance_inputs else None,
               acceptance_input_min=float(min(acceptance_inputs)) if acceptance_inputs else None,
               acceptance_input_max=float(max(acceptance_inputs)) if acceptance_inputs else None,
               demand_hash=hashlib.sha256(json.dumps(sorted(demand_rows.values())).encode()).hexdigest(),
               elapsed_seconds=time.perf_counter()-started)
    row['charging'] = env._charging_session_stats()
    return row


def json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def summarize(rows):
    metrics = ['rejected_offers', 'unique_rejected_requests', 'completed_orders', 'completed_ev_orders',
               'completed_aev_orders', 'ev_offers', 'ev_rejection_rate', 'completion_rate', 'reward']
    comparisons = []
    for learner in sorted({r['learner'] for r in rows}):
        selected = [r for r in rows if r['learner'] == learner and not r['training']]
        paired = {}
        for row in selected:
            paired.setdefault((row['train_seed'], row['seed']), {})[row['arm']] = row
        if not paired or any(set(pair) != {'off', 'predicted'} for pair in paired.values()):
            raise AssertionError('Incomplete evaluation pairs')
        demand_matched = all(p['off']['demand_hash'] == p['predicted']['demand_hash'] for p in paired.values())
        if not demand_matched:
            raise AssertionError('Paired evaluations did not share the same exogenous demand')
        comparison = dict(learner=learner, paired_evaluations=len(paired), demand_matched=demand_matched, metrics={})
        for metric in metrics:
            off = np.asarray([p['off'][metric] for p in paired.values()], dtype=float)
            on = np.asarray([p['predicted'][metric] for p in paired.values()], dtype=float)
            # Cluster bootstrap by independently trained model, not by orders
            # or repeated rollouts of the same fitted model.
            groups = sorted({key[0] for key in paired})
            clustered = np.asarray([np.mean([p['predicted'][metric]-p['off'][metric]
                for key, p in paired.items() if key[0] == seed]) for seed in groups])
            interval = None
            if len(groups) >= 2:
                rng = np.random.default_rng(1823)
                means = clustered[rng.integers(len(groups), size=(5000, len(groups)))].mean(1)
                interval = np.quantile(means, [.025, .975]).tolist()
            comparison['metrics'][metric] = dict(off_mean=float(off.mean()), predicted_mean=float(on.mean()),
                delta_predicted_minus_off=float((on-off).mean()), cluster_bootstrap_95=interval,
                per_train_seed_delta=dict(zip(map(str, groups), clustered.tolist())))
        comparisons.append(comparison)
    return comparisons


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    all_rows, training_checks = [], []
    sources = sorted([ROOT / 'run_acceptance_ablation.py', *ROOT.glob('src/*.py'), *ROOT.glob('src/recourse/*.py')])
    manifest = dict(arguments=vars(args), git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    predictor_sha256=hashlib.sha256(args.acceptance_model.read_bytes()).hexdigest(),
                    target='Integrated single-stage; calibrated rejection q+mask; residual expected anchor; realized reward unchanged')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=json_default) + '\n')
    for learner in args.learners:
        for train_seed in args.train_seeds:
            initial_hash = None
            for arm in ['off', 'predicted']:
                directory = args.output_dir / learner / f'seed-{train_seed}' / arm
                directory.mkdir(parents=True)
                with (directory / 'run.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                    env = build_env(args, 50000 + train_seed * 100, arm, learner, True)
                    seed_everything(train_seed)
                    pair = build_pair(env, learner)
                    before_hash = weight_hash(pair)
                    comparable_hash = weight_hash(pair, remove_acceptance=True)
                    if initial_hash is not None and initial_hash != comparable_hash:
                        raise AssertionError('On/off initial non-acceptance weights differ')
                    initial_hash = comparable_hash
                for ep in range(args.episodes):
                    seed = 50000 + train_seed * 100 + ep
                    with (directory / 'run.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
                        env = build_env(args, seed, arm, learner, True)
                        attach_pair(env, pair)
                        seed_everything(train_seed * 10000 + ep)
                        row = rollout(args, env, pair, training=True, seed=seed, directory=directory)
                    row.update(learner=learner, train_seed=train_seed, arm=arm, episode=ep)
                    all_rows.append(row)
                    with (args.output_dir / 'episodes.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, default=json_default)+'\n')
                    print(f'{learner} seed={train_seed} {arm} train={ep+1}/{args.episodes} '
                          f'complete={row["completed_orders"]} rejected={row["rejected_offers"]} '
                          f'updates={row["joint_updates"]} seconds={row["elapsed_seconds"]:.1f}', flush=True)
                after_hash = weight_hash(pair)
                if before_hash == after_hash or min(v.optimizer_steps_joint for v in pair) == 0:
                    raise AssertionError('Critics did not learn')
                check = dict(learner=learner, train_seed=train_seed, arm=arm,
                             initial_common_weights_hash=comparable_hash, trained_weights_hash=after_hash,
                             aev_joint_updates=pair[0].optimizer_steps_joint, ev_joint_updates=pair[1].optimizer_steps_joint)
                if arm == 'predicted' and pair[1].acceptance_input_enabled:
                    net = getattr(pair[1].network, 'base', pair[1].network)
                    check['ev_acceptance_weight_norm'] = float(net.net[0].weight[:, pair[1].acceptance_input_index].norm().item())
                    if check['ev_acceptance_weight_norm'] == 0:
                        raise AssertionError('Probability feature received no learning update')
                    check['ev_response_mask_weight_norm'] = float(net.net[0].weight[:, pair[1].acceptance_input_index + 1].norm().item())
                    if check['ev_response_mask_weight_norm'] == 0:
                        raise AssertionError('Response mask received no learning update')
                    check['predictor_frozen'] = all(p.grad is None and not p.requires_grad for p in pair[1].response_model.network.parameters())
                    if not check['predictor_frozen']:
                        raise AssertionError('TD optimization modified the rejection predictor')
                training_checks.append(check)
                checkpoint = directory / 'checkpoint.pt'
                save_pair(pair, checkpoint, check)
                with (directory / 'run.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
                    pair = load_pair(env, learner, checkpoint)
                if weight_hash(pair) != after_hash:
                    raise AssertionError('Checkpoint weights failed roundtrip')
                for seed in args.test_seeds:
                    with (directory / 'run.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
                        env = build_env(args, seed, arm, learner, False)
                        attach_pair(env, pair)
                        seed_everything(seed)
                        row = rollout(args, env, pair, training=False, seed=seed, directory=directory)
                    if weight_hash(pair) != after_hash:
                        raise AssertionError('Frozen evaluation modified learned weights')
                    row.update(learner=learner, train_seed=train_seed, arm=arm)
                    all_rows.append(row)
                    with (args.output_dir / 'episodes.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, default=json_default)+'\n')
                    print(f'{learner} seed={train_seed} {arm} test={seed} '
                          f'complete={row["completed_orders"]} rejected={row["rejected_offers"]} '
                          f'seconds={row["elapsed_seconds"]:.1f}', flush=True)
                del pair, env
                gc.collect()
    comparisons = summarize(all_rows)
    summary = dict(manifest=manifest, comparisons=comparisons, training_checks=training_checks, episodes=all_rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=json_default)+'\n')
    report = ['# Integrated EV 拒单概率 v3 接口检查', '',
              f'车队：{args.num_vehicles} 辆，EV {args.num_ev}、AEV {args.num_vehicles-args.num_ev}。',
              f'环境：{args.environment}；每组 {len(args.train_seeds)} 个独立训练种子、每种子 {args.episodes} 轮训练，'
              f'随后用 {len(args.test_seeds)} 个独立环境种子冻结评估。', '',
              '使用独立校准的 q_reject＋human_response_mask 作为 critic 输入。residual 默认同时使用期望结构化基准；'
              '未读取 oracle，未更改实际执行奖励。'
              '没有 EV-first/AEV-first 二阶段 recourse。所有执行分配使用 exact primal_dual MCMF。', '',
              '| 学习器 | 指标（每评估轮均值） | 不加概率 | 加概率 | 加－不加 |',
              '|---|---|---:|---:|---:|']
    for comparison in comparisons:
        for metric, label in [('rejected_offers', '分配后拒单次数'), ('unique_rejected_requests', '被拒订单去重数'),
                              ('completed_orders', '平台完成订单'), ('completed_ev_orders', 'EV完成'),
                              ('completed_aev_orders', 'AEV完成'), ('ev_offers', 'EV订单邀约'),
                              ('ev_rejection_rate', 'EV拒单率'), ('completion_rate', '完成／生成')]:
            m = comparison['metrics'][metric]
            report.append(f'| {comparison["learner"]} | {label} | {m["off_mean"]:.4f} | {m["predicted_mean"]:.4f} | {m["delta_predicted_minus_off"]:+.4f} |')
    report += ['', '## 口径与限制', '',
               '- 拒单次数是实际 EV 邀约后的拒绝事件；同一订单可重复拒绝。另列去重被拒订单数。',
               '- 完成数是完整仿真时域内真正完成的 EV＋AEV 订单，未把已接受或载客中的订单计作完成。',
               '- generated = completed + expired + active 逐轮核对；保留 EV／AEV 真实充电次数和充电时长统计。',
               '- episodes.jsonl 保存所有训练／测试轮；各组目录含 checkpoint.pt、逐邀约记录、完整统计及运行日志。',
               '- summary.json 保存按训练种子聚类的配对差值与 bootstrap 区间；训练种子较少时，区间不应视为强显著性结论。',
               '- 这是给定训练预算下的本地仿真对照，不代表充分收敛、所有场景均改善或真实司机外部有效性。',
               '- 新增列初始权重为零，其他初始权重逐字节一致；评估从检查点重新加载且关闭所有学习更新。',
               f'- 是否截短时域：{args.max_steps is not None}。',
               f'- 各学习器配对需求序列一致：{all(c["demand_matched"] for c in comparisons)}。', '']
    (args.output_dir / 'report.md').write_text('\n'.join(report))
    print(f'Results saved: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
