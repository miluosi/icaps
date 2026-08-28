"""Audit continuous acceptance probabilities in pure NYC MCMF (no Q learning).

Default fleet: 100 human EV + 100 AEV. Train a neural predictor using fresh
train/validation/test seeds, log actual Adam epochs, and
compare identical test simulations with/without read-only predictions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np
from src.acceptance_model import BinaryAcceptanceModel, FEATURE_NAMES, collect_offers, probability_metrics
from train_acceptance_model import (
    ROOT, clustered_improvement_intervals, make_environment,
    parse_args as environment_args,
)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, datetime)):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=json_default, allow_nan=False) + '\n')


def save_rows(path, rows):
    path.write_text(''.join(json.dumps(row, default=json_default, allow_nan=False) + '\n'
                            for row in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--train-seeds', type=int, nargs='+', default=list(range(1100, 1120)))
    parser.add_argument('--validation-seeds', type=int, nargs='+', default=list(range(1200, 1206)))
    parser.add_argument('--test-seeds', type=int, nargs='+', default=list(range(1300, 1310)))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=None, help='Smoke tests only; omit for full NYC episodes')
    parser.add_argument('--require-random-rejection', action='store_true',
                        help='Verify reject_uniform=True; do not silently override the environment')
    parser.add_argument('--expected-assignment-range-km', type=float, default=None,
                        help='Verify the enabled pickup radius, not passenger trip length')
    parser.add_argument('--nn-epochs', type=int, default=120)
    parser.add_argument('--nn-patience', type=int, default=20)
    parser.add_argument('--nn-seed', type=int, default=42)
    parser.add_argument('--parquet-path', type=Path, default=ROOT / 'nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet')
    parser.add_argument('--station-csv', type=Path, default=ROOT / 'nyedata/nyc_all_charging_stations.csv')
    parser.add_argument('--date', default='2025-12-18')
    parser.add_argument('--start-hour', type=float, default=8.0)
    parser.add_argument('--stop-hour', type=float, default=10.0)
    parser.add_argument('--epoch-length', type=float, default=30.0)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args(argv)
    seeds = args.train_seeds + args.validation_seeds + args.test_seeds
    if len(seeds) != len(set(seeds)):
        parser.error('Train/validation/test seeds must be unique and disjoint')
    if not 0 < args.num_ev <= args.num_vehicles or args.workers < 1:
        parser.error('Require a positive worker count and 0 < num-ev <= num-vehicles')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('max-steps must be positive')
    if min(args.nn_epochs, args.nn_patience) <= 0:
        parser.error('Neural epochs and patience must be positive')
    if args.expected_assignment_range_km is not None and (
            not np.isfinite(args.expected_assignment_range_km) or args.expected_assignment_range_km <= 0):
        parser.error('expected-assignment-range-km must be finite and positive')
    if not 0 <= args.start_hour < args.stop_hour <= 24 or args.epoch_length <= 0:
        parser.error('Invalid NYC time window')
    if args.output_dir is None:
        args.output_dir = ROOT / 'results/acceptance_checks' / datetime.now().strftime('nyc-200-%Y%m%d-%H%M%S-%f')
    args.output_dir = args.output_dir.resolve()
    return args


def rejection_metrics(rows, probabilities, threshold=0.5):
    """Treat REJECTION as positive, unlike the model's accepted=1 labels."""
    accepted = np.asarray([row['accepted'] for row in rows])
    p = np.asarray(probabilities, dtype=float)
    if not len(rows) or p.shape != accepted.shape or not np.isfinite(p).all():
        raise ValueError('Aligned nonempty finite probabilities are required')
    if not set(np.unique(accepted)).issubset({0, 1}) or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid binary labels or probabilities')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Rejection threshold must be in [0,1]')
    actual = accepted == 0
    score = 1.0 - p
    predicted = score >= threshold
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    # Average precision groups equal scores; no arbitrary tie ordering gain.
    order = np.argsort(-score, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(score[order])), len(score) - 1]
    cumulative = np.cumsum(actual[order])[ends]
    ap = None
    if actual.any():
        recalls = cumulative / actual.sum()
        ap = float(np.sum(np.diff(np.r_[0.0, recalls]) * cumulative / (ends + 1)))
    return dict(threshold=float(threshold), count=len(rows), actual_rejections=int(actual.sum()),
                predicted_rejections=int(predicted.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=(tp + tn) / len(rows), error_rate=(fp + fn) / len(rows),
                rejection_precision=tp / (tp + fp) if tp + fp else None,
                rejection_recall=recall, rejection_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
                rejection_average_precision=ap, rejection_prevalence=float(actual.mean()))


def select_rejection_threshold(validation_rows, probabilities):
    """Maximize rejection F1 on VALIDATION only; prefer precision on ties."""
    score = 1.0 - np.asarray(probabilities, dtype=float)
    labels = np.asarray([row['accepted'] == 0 for row in validation_rows])
    # Validate before sorting; no test labels are accepted by this interface.
    rejection_metrics(validation_rows, probabilities)
    if not labels.any():
        raise ValueError('Validation must contain actual rejections')
    order = np.argsort(-score, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(score[order])), len(score) - 1]
    tp = np.cumsum(labels[order])[ends]
    count = ends + 1
    f1 = 2 * tp / (count + labels.sum())
    precision = tp / count
    best = max(range(len(ends)), key=lambda i: (float(f1[i]), float(precision[i]), float(score[order][ends[i]])))
    threshold = float(score[order][ends[best]])
    return threshold, rejection_metrics(validation_rows, probabilities, threshold)


def loss_trace(model, validation_rows):
    if any(row['validation_binary_cross_entropy'] is None for row in model.loss_history):
        raise ValueError('Neural validation losses must be recorded during fit(validation_rows=...)')
    return [dict(row) for row in model.loss_history]


def trace_diagnostics(trace):
    result = {}
    for key in ['objective', 'binary_cross_entropy', 'validation_binary_cross_entropy']:
        values = np.asarray([r[key] for r in trace])
        result[key] = dict(initial=float(values[0]), final=float(values[-1]),
                           net_decrease=float(values[0] - values[-1]),
                           increasing_steps=int(np.sum(np.diff(values) > 1e-10)))
    return result


def evaluate_model(model, train_rows, validation_rows, test_rows, *, validation_threshold=None):
    if validation_threshold is None:
        validation_threshold, _ = select_rejection_threshold(
            validation_rows, model.predict_proba(validation_rows))
    threshold = validation_threshold
    validation_threshold_metrics = rejection_metrics(
        validation_rows, model.predict_proba(validation_rows), threshold)
    p = model.predict_proba(test_rows)
    baseline = np.full(len(test_rows), np.mean([r['accepted'] for r in train_rows]))
    probability = probability_metrics(test_rows, p)
    probability['predicted_acceptance_range'] = [float(p.min()), float(p.max())]
    probability['predicted_rejection_range'] = [float(1 - p.max()), float(1 - p.min())]
    return dict(probability=probability,
                constant_baseline=probability_metrics(test_rows, baseline),
                default_threshold=rejection_metrics(test_rows, p, 0.5),
                validation_f1_threshold=rejection_metrics(test_rows, p, threshold),
                validation_threshold_metrics=validation_threshold_metrics,
                always_accept=rejection_metrics(test_rows, np.ones(len(test_rows)), 0.5),
                probability_gain_clustered_95=clustered_improvement_intervals(test_rows, p, float(baseline[0])))


def trajectory_state(env, rewards):
    fields = ['location', 'battery', 'assigned_request', 'passenger_onboard',
              'charging_station', 'charging_target', 'target_location', 'idle_timer', 'is_online']
    return dict(time=env.current_time, rewards=sorted(rewards.items()),
                vehicles=[(vid, {key: vehicle.get(key) for key in fields}) for vid, vehicle in sorted(env.vehicles.items())],
                active=sorted(env.active_requests),
                completed=sorted(int(getattr(r, 'request_id', r)) for r in env.completed_requests),
                expired=sorted(env.expired_request_ids),
                stations=[(sid, station.current_vehicles, station.charging_queue, station.charging_queue_notarrived)
                          for sid, station in sorted(env.charging_manager.stations.items())])


def environment_configuration(env, args):
    configuration = {name: getattr(env, name) for name in (
        'reject_uniform', 'ifreject', 'ride_acceptance_noise_std', 'rejection_logit_shift',
        'use_range_requests', 'assignmentrange', 'ride_acceptance_asc',
        'ride_acceptance_beta_idle_min', 'ride_acceptance_beta_pickup_min', 'ride_acceptance_beta_surge')}
    if getattr(args, 'require_random_rejection', False):
        assert configuration['reject_uniform'] is True and configuration['ifreject'], \
            'This experiment requires actual stochastic driver rejection'
        assert configuration['ride_acceptance_noise_std'] == 0.0, \
            'The diagnostic oracle assumes the default zero utility noise'
    expected = getattr(args, 'expected_assignment_range_km', None)
    if expected is not None:
        assert configuration['use_range_requests'], 'Pickup range filtering is disabled'
        assert abs(float(configuration['assignmentrange']) - expected) < 1e-9, \
            'The actual pickup radius does not match the requested experiment'
    return configuration


def verify_offer_response(realization, rejected, configuration, pickup_distance_km):
    """Check actual responses; diagnostic uniforms must NEVER become features."""
    p = float(realization['acceptance_probability'])
    uniform = float(realization['uniform'])
    assert np.isfinite(p) and 0 <= p <= 1, 'Invalid response probability'
    assert np.isfinite(uniform) and 0 <= uniform <= 1, 'Invalid response uniform'
    assert np.isfinite(pickup_distance_km) and pickup_distance_km >= 0
    assert bool(realization['rejected']) == bool(rejected) == (uniform < 1.0 - p), \
        'Recorded probability/uniform do not explain the actual driver answer'
    if configuration['ifreject'] and not configuration['reject_uniform']:
        assert uniform == 0.5, 'Deterministic rejection must use the fixed threshold'
    if configuration['use_range_requests']:
        assert pickup_distance_km <= configuration['assignmentrange'] + 1e-5, \
            'Observed an offer outside the enabled pickup radius'
    return dict(response_uniform=uniform, pickup_distance_km=float(pickup_distance_km),
                reject_uniform=bool(configuration['reject_uniform']))


def run_episode(args, seed, split, directory, model_states=None):
    import torch
    torch.set_num_threads(1)
    directory.mkdir(parents=True)
    started = time.perf_counter()
    models = {name: BinaryAcceptanceModel.from_dict(state) for name, state in (model_states or {}).items()}
    settings = environment_args(['--environment', 'nyc', '--num-vehicles', str(args.num_vehicles),
        '--num-ev', str(args.num_ev), '--parquet-path', str(args.parquet_path), '--station-csv', str(args.station_csv),
        '--date', args.date, '--start-hour', str(args.start_hour), '--stop-hour', str(args.stop_hour),
        '--epoch-length', str(args.epoch_length), '--mcmf-backend', 'primal_dual'])
    trace_hash = hashlib.sha256()
    calls, reward = 0, 0.0
    with (directory / 'run.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        env = make_environment(settings, seed)
        assert env.adp_value == 0 and not env.knownreject and env.usemcmf
        assert env.value_function is None and env.value_function_ev is None
        assert getattr(env, 'recourse_variant', 'legacy') == 'legacy'
        assert sum(v['type'] == 1 for v in env.vehicles.values()) == args.num_ev
        assert len(env.vehicles) == args.num_vehicles
        configuration = environment_configuration(env, args)
        with collect_offers(env, episode_id=f'nyc:{split}:seed-{seed}', seed=seed) as offers:
            original = env._should_reject_request
            def predict_then_answer(vid, request):
                human = env.vehicles[vid]['type'] == 1
                distance = env.get_distance_km(env.vehicles[vid]['location'], request.pickup) if human else None
                predictions = {name: model.predict_acceptance_probability(env, vid, request)
                               for name, model in models.items()} if human else {}
                response = original(vid, request)
                if human:
                    key = (env._epoch_id(), int(vid), int(request.request_id))
                    offers[-1].update(verify_offer_response(
                        env._last_offer_realizations[key], response, configuration, distance))
                    offers[-1].update({f'{name}_p_accept': p for name, p in predictions.items()})
                return response
            env._should_reject_request = predict_then_answer
            for step in range(min(env.episode_length, args.max_steps or env.episode_length)):
                previous = getattr(env, 'mcmf_last_result', None)
                actions, stored, stored_ev = env.simulate_motion(
                    agents=[], current_requests=list(env.active_requests.values()), rebalance=True)
                result = getattr(env, 'mcmf_last_result', None)
                if result is not None and result is not previous:
                    assert result['optimal'] and not result['solver_fallback_used'], 'Non-exact MCMF'
                    calls += 1
                _, rewards, _, done, _ = env.step(actions, stored, stored_ev)
                reward += float(sum(rewards.values()))
                trace_hash.update(json.dumps(trajectory_state(env, rewards), sort_keys=True, default=json_default).encode())
                if done:
                    break
        stats = env.get_episode_stats()
        assert stats['request_lifecycle_gap'] == 0
        charging = env._charging_session_stats()
    if not calls or not offers:
        raise AssertionError('No exact dispatches or actual EV offers observed')
    for name, model in models.items():
        np.testing.assert_allclose(model.predict_proba(offers), [r[f'{name}_p_accept'] for r in offers], rtol=1e-6, atol=1e-7)
    rejected = [r for r in offers if not r['accepted']]
    uniforms = [r['response_uniform'] for r in offers]
    summary = dict(seed=seed, split=split, shadow_prediction=bool(models), steps=step + 1,
                   ev_offers=len(offers), rejected_offers=len(rejected),
                   unique_rejected_requests=len({r['request_id'] for r in rejected}),
                   rejection_rate=len(rejected) / len(offers), completed_orders=len(env.completed_requests),
                   completed_ev_orders=int(stats['completed_ev_orders']), generated_requests=int(env.whole_req_num),
                   expired_requests=len(env.expired_request_ids), active_requests=len(env.active_requests),
                   reward=reward, exact_mcmf_calls=calls, trajectory_hash=trace_hash.hexdigest(),
                   environment=configuration, verified_ev_responses=len(offers),
                   observed_pickup_distance_max_km=max(r['pickup_distance_km'] for r in offers),
                   response_uniform_range=[min(uniforms), max(uniforms)],
                   response_uniform_unique_count=len(set(uniforms)),
                   charging=charging, elapsed_seconds=time.perf_counter() - started)
    assert summary['generated_requests'] == summary['completed_orders'] + summary['expired_requests'] + summary['active_requests']
    save_rows(directory / 'offers.jsonl', offers)
    save_json(directory / 'stats.json', stats)
    save_json(directory / 'episode.json', summary)
    return summary


def episode_job(args, seed, split, model_states):
    directory = args.output_dir / split / f'seed-{seed}'
    baseline = run_episode(args, seed, split, directory / 'baseline')
    if model_states is None:
        return baseline
    shadow = run_episode(args, seed, split, directory / 'shadow', model_states)
    for key in ['trajectory_hash', 'steps', 'ev_offers', 'rejected_offers', 'unique_rejected_requests',
                'completed_orders', 'completed_ev_orders', 'generated_requests', 'reward', 'charging',
                'environment', 'verified_ev_responses', 'observed_pickup_distance_max_km',
                'response_uniform_range', 'response_uniform_unique_count']:
        assert baseline[key] == shadow[key], f'Passive prediction changed {key} for seed {seed}'
    base_rows = read_rows(directory / 'baseline/offers.jsonl')
    shadow_rows = read_rows(directory / 'shadow/offers.jsonl')
    stripped = [{k: v for k, v in row.items() if not k.endswith('_p_accept')} for row in shadow_rows]
    assert base_rows == stripped, 'Actual offer sequence/outcomes changed'
    return dict(baseline=baseline, shadow=shadow, identical_trajectory=True)


def collect_group(args, split, seeds, model_states=None):
    episodes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(episode_job, args, seed, split, model_states): seed for seed in seeds}
        for job in as_completed(jobs):
            result = job.result()
            episodes.append(result)
            row = result['baseline'] if model_states else result
            print(f'{split} seed={row["seed"]}: offers={row["ev_offers"]}, rejected={row["rejected_offers"]}, '
                  f'completed={row["completed_orders"]}' + ('; shadow identical' if model_states else ''), flush=True)
    episodes.sort(key=lambda r: r['baseline']['seed'] if model_states else r['seed'])
    arm = 'shadow' if model_states else 'baseline'
    rows = [row for seed in seeds for row in read_rows(args.output_dir / split / f'seed-{seed}' / arm / 'offers.jsonl')]
    save_rows(args.output_dir / f'{split}_offers.jsonl', rows)
    return rows, episodes


def build_report(result):
    args, trace = result['configuration'], result['selected_loss_history']
    selected = trace[result['model']['selected_epoch']]
    environment = result['train_episodes'][0]['environment']
    report = ['# NYC 纯 MCMF 接单概率独立检查', '',
              f'车队 {args["num_vehicles"]}：EV {args["num_ev"]}，AEV {args["num_vehicles"] - args["num_ev"]}；'
              f'{args["date"]} {args["start_hour"]:g}:00–{args["stop_hour"]:g}:00，epoch={args["epoch_length"]:g} 秒。',
              f'实际环境 reject_uniform={environment["reject_uniform"]}；接客范围 '
              f'{environment["assignmentrange"]:g} km（不是乘客行程长度），范围过滤={environment["use_range_requests"]}。',
              '只有普通 exact MCMF：ADP=0、knownreject=False、无 sequential recourse。概率只旁路预测，不进入分配打分。', '',
              '## 函数与损失', '',
              '`src/acceptance_model.py::BinaryAcceptanceModel.predict_proba()` 返回连续接单概率；'
              '`predict_rejection_probability()` 返回其补数。',
              f'神经网络：{len(FEATURE_NAMES)}→64→32→1，两层 ReLU，输出 sigmoid；没有逻辑回归预测路径。',
              r'\(p_i=\sigma(f_\theta((x_i-\mu)/s)),\quad P(\mathrm{reject})=1-p_i.\)',
              f'完整的分配前输入：{", ".join(FEATURE_NAMES)}。accepted=1、rejected=0；不读取真实概率、随机数或回答后状态。',
              r'\(L=\mathrm{BCEWithLogitsLoss}(f_\theta(x),y)+\frac{\lambda}{2}\sum_\ell\lVert W_\ell\rVert_F^2.\)',
              '训练集标准化；权重 L2、偏置不罚；自然类别比例；Adam 优化，验证 BCE 早停并恢复最佳 epoch；不是 Q/residual TD loss。', '',
              '## 损失是否下降', '',
              '| 检查 | 初始 | 最终 |', '|---|---:|---:|']
    for label, start, end in [('神经网络选中 epoch：训练目标', trace[0]['objective'], selected['objective']),
                             ('神经网络选中 epoch：训练 BCE', trace[0]['binary_cross_entropy'], selected['binary_cross_entropy']),
                             ('神经网络选中 epoch：验证 BCE', trace[0]['validation_binary_cross_entropy'], selected['validation_binary_cross_entropy'])]:
        report.append(f'| {label} | {start:.8f} | {end:.8f} |')
    report += ['', f'选择的 L2={result["model"]["l2"]}，仅以验证集 log loss 选择。'
               f'记录 {len(trace) - 1} 个训练 epoch（另有初始值），恢复 epoch {result["model"]["selected_epoch"]}。',
               '这是重新采集完整特征后训练的神经网络；旧回归模型和三特征数据未加载。', '',
               'Adam 的训练目标和验证 BCE 都可能逐步波动；验证 BCE 不参与梯度更新。'
               f'本次验证 BCE 上升的迭代步数为 {result["loss_diagnostics"]["validation_binary_cross_entropy"]["increasing_steps"]}。', '',
               '## 新的留出测试：概率质量', '',
               '| 预测器 | Log loss↓ | Brier↓ | ROC-AUC↑ | 拒单 Average Precision↑ |', '|---|---:|---:|---:|---:|']
    for name in ['constant', 'fresh_fit', 'simulator_oracle']:
        m = result['fresh_test'][name]
        report.append(f'| {name} | {m["probability"]["log_loss"]:.6f} | {m["probability"]["brier_score"]:.6f} | '
                      f'{m["probability"]["roc_auc"]:.6f} | {m["default_threshold"]["rejection_average_precision"]:.6f} |')
    report += ['', '## 新的留出测试：是否拒单的分类', '',
               '正类为拒单，判定规则是 p_reject >= threshold；另外一组阈值只在验证集最大化拒单 F1，未用测试集调参。', '',
               'FP=实际接受却预测拒绝；FN=实际拒绝却预测接受。simulator_oracle 使用真实条件概率仅作诊断，'
               '不参与训练、模型选择或分配；它也无法知道单次未来随机响应。', '',
               '| 预测器/阈值 | 准确率 | 拒单 precision | 拒单 recall | 拒单 F1 | TP / FP / FN / TN |',
               '|---|---:|---:|---:|---:|---|']
    for name in ['constant', 'fresh_fit', 'simulator_oracle']:
        variants = ['default_threshold'] if name == 'constant' else ['default_threshold', 'validation_f1_threshold']
        for variant in variants:
            m = result['fresh_test'][name][variant]
            precision = '未定义（未报拒单）' if m['rejection_precision'] is None else f'{m["rejection_precision"]:.4%}'
            report.append(f'| {name}, {m["threshold"]:.6f} | {m["accuracy"]:.4%} | {precision} | '
                          f'{m["rejection_recall"]:.4%} | {m["rejection_f1"]:.4f} | {m["tp"]} / {m["fp"]} / {m["fn"]} / {m["tn"]} |')
    report += ['', '## 纯 MCMF 实际拒单是否下降', '',
               '| 测试种子 | 不预测：拒单 | 旁路预测：拒单 | 不预测：完单 | 旁路预测：完单 | 完整轨迹一致 |',
               '|---|---:|---:|---:|---:|---|']
    for pair in result['test_episodes']:
        a, b = pair['baseline'], pair['shadow']
        report.append(f'| {a["seed"]} | {a["rejected_offers"]} | {b["rejected_offers"]} | '
                      f'{a["completed_orders"]} | {b["completed_orders"]} | {pair["identical_trajectory"]} |')
    report += ['', '实际拒单和完单变化均为 0：只训练/调用预测器不改变纯 MCMF 策略，也不改变司机真实响应。',
               '概率预测误差降低不等于实际拒单降低；0.5 阈值下高准确率也不意味着识别出了拒单。', '',
               '## 范围与文件', '',
               f'- 样本数：{result["sample_counts"]}。按仿真种子分离训练、验证、测试。',
               f'- 是否截短时域：{args["max_steps"] is not None}。',
               '- NYC 真实订单、仿真人类司机标签；同一天需求在不同种子间复用，不是跨日期或真实司机外部验证。',
               '- summary.json：所有指标、混淆矩阵、概率误差改进区间、完整充电统计、来源 hash。',
               '- model.json：新独立训练的模型；没有覆盖原来用于 Q/residual 的 checkpoint。',
               '- loss_history.jsonl / candidate_loss_histories.json：选中模型及各候选的真实迭代损失。',
               '- test_predictions.jsonl：逐邀约的神经网络连续概率、两个阈值的分类及真实标签。',
               '- decision_thresholds.json：测试仿真开始前冻结的验证集阈值。',
               '- source_archive.tar.gz：运行开始时源文件快照；结束时核对相关源文件未变。',
               '- offers.jsonl 的 response_uniform 仅用于检查真实响应，不是预测模型输入。',
               '- train/、validation/、test/：完整仿真日志、逐邀约记录和保留充电统计的 stats.json。', '']
    return '\n'.join(report)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), ROOT / 'src/acceptance_model.py', ROOT / 'src/acceptance_inputs.py', ROOT / 'train_acceptance_model.py',
               ROOT / 'src/NYCEnvironment.py']
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with tarfile.open(args.output_dir / 'source_archive.tar.gz', 'w:gz') as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    save_json(args.output_dir / 'configuration.json', vars(args))
    print(f'Output: {args.output_dir}', flush=True)
    train, train_episodes = collect_group(args, 'train', args.train_seeds)
    validation, validation_episodes = collect_group(args, 'validation', args.validation_seeds)
    candidates, histories = [], {}
    for l2 in [0.0, 1e-5, 1e-4, 1e-3]:
        candidate = BinaryAcceptanceModel(l2=l2, max_epochs=args.nn_epochs,
            patience=args.nn_patience, seed=args.nn_seed).fit(train, validation_rows=validation)
        score = candidate.loss_history[candidate.selected_epoch]['validation_binary_cross_entropy']
        candidates.append((score, candidate))
        histories[str(l2)] = loss_trace(candidate, validation)
    score, model = min(candidates, key=lambda item: item[0])
    trace = histories[str(model.l2)]
    model.save(args.output_dir / 'model.json')
    restored = BinaryAcceptanceModel.load(args.output_dir / 'model.json')
    np.testing.assert_array_equal(model.predict_proba(validation), restored.predict_proba(validation))
    save_rows(args.output_dir / 'loss_history.jsonl', trace)
    save_json(args.output_dir / 'candidate_loss_histories.json', histories)
    # Freeze all thresholds before reading any test outcomes. The oracle is
    # strictly a diagnostic comparator, never a fitting target or a feature.
    thresholds = {}
    for name, p in [('fresh', model.predict_proba(validation)),
                    ('simulator_oracle', [r['oracle_acceptance_probability'] for r in validation])]:
        threshold, metrics = select_rejection_threshold(validation, p)
        thresholds[name] = dict(threshold=threshold, validation_metrics=metrics)
    save_json(args.output_dir / 'decision_thresholds.json', thresholds)
    print(f'Neural training: objective {trace[0]["objective"]:.6f} -> {model.fit_loss:.6f}; '
          f'validation log loss={score:.6f}; lambda={model.l2}', flush=True)
    test, test_episodes = collect_group(args, 'test', args.test_seeds,
                                      {'fresh': model.to_dict()})
    fresh_metrics = evaluate_model(model, train, validation, test,
                                   validation_threshold=thresholds['fresh']['threshold'])
    oracle_p = np.asarray([r['oracle_acceptance_probability'] for r in test])
    oracle_metrics = dict(probability=probability_metrics(test, oracle_p),
        default_threshold=rejection_metrics(test, oracle_p),
        validation_f1_threshold=rejection_metrics(test, oracle_p, thresholds['simulator_oracle']['threshold']))
    baseline = np.full(len(test), np.mean([r['accepted'] for r in train]))
    constant = dict(probability=probability_metrics(test, baseline), default_threshold=rejection_metrics(test, baseline))
    for row in test:
        for name, metrics in [('fresh', fresh_metrics)]:
            row[f'{name}_p_reject'] = 1 - row[f'{name}_p_accept']
            row[f'{name}_predict_reject_0_5'] = row[f'{name}_p_reject'] >= 0.5
            row[f'{name}_predict_reject_validation_threshold'] = row[f'{name}_p_reject'] >= metrics['validation_f1_threshold']['threshold']
    save_rows(args.output_dir / 'test_predictions.jsonl', test)
    assert source_hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, \
        'Source changed during the experiment; results must not be attributed to a single code version'
    result = dict(configuration=vars(args),
                  model=model.to_dict(), selected_loss_history=trace,
                  loss_diagnostics=trace_diagnostics(trace),
                  training_objective_nonincreasing=bool(np.all(np.diff([r['objective'] for r in trace]) <= 1e-10)),
                  sample_counts={k: len(v) for k, v in [('train', train), ('validation', validation), ('test', test)]},
                  validation_selection=[dict(l2=m.l2, validation_log_loss=loss) for loss, m in candidates],
                  fresh_test=dict(constant=constant, fresh_fit=fresh_metrics, simulator_oracle=oracle_metrics),
                  frozen_validation_thresholds=thresholds,
                  train_episodes=train_episodes, validation_episodes=validation_episodes, test_episodes=test_episodes,
                  unchanged_actual_rejections=all(r['baseline']['rejected_offers'] == r['shadow']['rejected_offers'] for r in test_episodes),
                  source_sha256=source_hashes, source_unchanged_during_run=True,
                  parquet_sha256=hashlib.sha256(args.parquet_path.read_bytes()).hexdigest(),
                  station_csv_sha256=hashlib.sha256(args.station_csv.read_bytes()).hexdigest())
    save_json(args.output_dir / 'summary.json', result)
    (args.output_dir / 'report.md').write_text(build_report(result))
    print(f'Finished: {args.output_dir / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
