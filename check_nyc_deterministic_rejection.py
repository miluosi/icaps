"""Audit/retrain NYC rejection prediction with reject_uniform=False.

Explicit deterministic rule: reject iff the latent acceptance score is < 0.5.
Keep the simulator, dispatch and behavior coefficients unchanged. Fit only if
actual new training offers contain both accepted and rejected labels; otherwise
report the missing class, rather than fabricate a successful binary model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
from pathlib import Path
import tarfile
from unittest.mock import patch

import numpy as np

import check_nyc_mcmf_acceptance as audit
from src.acceptance_model import BinaryAcceptanceModel, probability_metrics


ROOT = audit.ROOT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-vehicles', type=int, default=200)
    parser.add_argument('--num-ev', type=int, default=100)
    parser.add_argument('--train-seeds', type=int, nargs='+', default=list(range(5100, 5140)))
    parser.add_argument('--validation-seeds', type=int, nargs='+', default=list(range(5200, 5212)))
    parser.add_argument('--test-seeds', type=int, nargs='+', default=list(range(5300, 5320)))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=None, help='Smoke checks only; omit for full episodes')
    parser.add_argument('--reference-run', type=Path, default=None, help='Optional NEURAL v2 reference run; no regression loading')
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
        parser.error('Require 0 < num-ev <= num-vehicles and workers > 0')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('max-steps must be positive')
    if not 0 <= args.start_hour < args.stop_hour <= 24 or args.epoch_length <= 0:
        parser.error('Invalid NYC time window')
    args.output_dir = (args.output_dir or ROOT / 'results/acceptance_checks' /
                       datetime.now().strftime('nyc-deterministic-%Y%m%d-%H%M%S-%f')).resolve()
    return args


def deterministic_rows(rows):
    """Separate a latent logit score from the deterministic response probability.

    The older collector calls the latent score oracle_acceptance_probability.
    Under this rule the true conditional response probability is instead 0/1.
    Preserve the score, verify the actual label, and never feed either oracle
    quantity to the learned predictor.
    """
    if not rows:
        raise ValueError('No observed offers')
    result = []
    for row in rows:
        score = float(row['oracle_acceptance_probability'])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid latent acceptance score')
        if row['accepted'] not in (0, 1) or int(row['accepted']) != int(score >= 0.5):
            raise ValueError('Actual response does not match reject_uniform=False')
        result.append(dict(row, latent_acceptance_score=score,
                           oracle_acceptance_probability=float(score >= 0.5),
                           response_rule='deterministic_threshold', response_threshold=0.5))
    return result


def coverage(rows):
    accepted = sum(int(row['accepted']) for row in rows)
    scores = [row['latent_acceptance_score'] for row in rows]
    return dict(offers=len(rows), accepted=accepted, rejected=len(rows) - accepted,
                latent_acceptance_score_min=min(scores), latent_acceptance_score_max=max(scores),
                acceptance_score_below_half=sum(score < 0.5 for score in scores),
                both_classes_observed=0 < accepted < len(rows))


def deterministic_episode(args, seed, split, model_states=None):
    """Guard the actual call path without changing labels, policy or RNG."""
    original_factory = audit.make_environment
    verification = dict(reject_uniform=False, ride_acceptance_noise_std=0.0,
                        verified_ev_answers=0, acceptance_uniform_calls=0)

    def guarded_factory(settings, episode_seed):
        env = original_factory(settings, episode_seed)
        assert env.reject_uniform is False, 'This audit requires an explicitly configured deterministic branch'
        assert env.ride_acceptance_noise_std == 0.0, 'Default behavior noise must remain zero'
        assert env.rejection_logit_shift == 0.0 and env.ifreject
        verification['behavior_coefficients'] = {
            name: getattr(env, name) for name in ('ride_acceptance_asc',
                'ride_acceptance_beta_idle_min', 'ride_acceptance_beta_pickup_min', 'ride_acceptance_beta_surge')}

        def forbid_uniform(*args, **kwargs):
            verification['acceptance_uniform_calls'] += 1
            raise AssertionError('Deterministic rejection must not draw an acceptance uniform')

        env._acceptance_uniform = forbid_uniform
        original_answer = env._should_reject_request

        def checked_answer(vehicle_id, request):
            rejected = original_answer(vehicle_id, request)
            if env.vehicles[vehicle_id]['type'] == 1:
                realization = env._last_offer_realizations[(env._epoch_id(), int(vehicle_id), int(request.request_id))]
                assert realization['uniform'] == 0.5
                assert bool(rejected) == (realization['acceptance_probability'] < 0.5)
                verification['verified_ev_answers'] += 1
            return rejected

        env._should_reject_request = checked_answer
        return env

    directory = args.output_dir / split / f'seed-{seed}'
    with patch.object(audit, 'make_environment', guarded_factory):
        episode = audit.run_episode(args, seed, split, directory, model_states)
    assert verification['verified_ev_answers'] == episode['ev_offers']
    episode['deterministic_rule_verification'] = verification
    audit.save_json(directory / 'episode.json', episode)
    return episode


def collect(args, split, seeds, models=None):
    states = {name: model.to_dict() for name, model in (models or {}).items()} or None
    episodes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(deterministic_episode, args, seed, split, states) for seed in seeds]
        for job in as_completed(jobs):
            episode = job.result()
            episodes.append(episode)
            print(f'{split} seed={episode["seed"]}: offers={episode["ev_offers"]}, '
                  f'rejected={episode["rejected_offers"]}, completed={episode["completed_orders"]}; '
                  'fixed threshold verified, no rejection-uniform draw', flush=True)
    episodes.sort(key=lambda row: row['seed'])
    rows = [row for seed in seeds for row in deterministic_rows(audit.read_rows(
        args.output_dir / split / f'seed-{seed}' / 'offers.jsonl'))]
    audit.save_rows(args.output_dir / f'{split}_offers.jsonl', rows)
    return rows, episodes


def fit_if_identifiable(train, validation):
    """Never silently turn a single-class run into a trained reject detector."""
    if len({row['accepted'] for row in train}) != 2:
        return None, dict(status='not_trainable_single_class',
                          reason='Actual training offers do not contain both acceptance and rejection labels',
                          observed_training_labels=sorted({row['accepted'] for row in train})), {}
    candidates, histories = [], {}
    for l2 in [0.0, 1e-5, 1e-4, 1e-3]:
        model = BinaryAcceptanceModel(l2=l2).fit(train, validation_rows=validation)
        score = model.loss_history[model.selected_epoch]['validation_binary_cross_entropy']
        histories[str(l2)] = audit.loss_trace(model, validation)
        candidates.append((score, model))
    score, model = min(candidates, key=lambda item: item[0])
    return model, dict(status='trained', validation_bce=score, selected_l2=model.l2,
                       loss_diagnostics=audit.trace_diagnostics(histories[str(model.l2)]),
                       validation_selection=[dict(l2=m.l2, validation_bce=value) for value, m in candidates]), histories


def freeze_thresholds(validation, models):
    result = {}
    for name, model in models.items():
        selected = None
        if any(not row['accepted'] for row in validation):
            threshold, metrics = audit.select_rejection_threshold(validation, model.predict_proba(validation))
            selected = dict(threshold=threshold, validation_metrics=metrics)
        result[name] = dict(default=dict(threshold=0.5), validation_f1=selected,
                            validation_f1_unavailable_reason=None if selected else 'No validation rejections')
    return result


def evaluate(rows, p, choices):
    result = dict(probability=probability_metrics(rows, p), classifications={})
    for name in ('default', 'validation_f1'):
        selected = choices.get(name)
        result['classifications'][name] = None if selected is None else audit.rejection_metrics(rows, p, selected['threshold'])
    return result


def build_report(result):
    args = result['configuration']
    lines = ['# NYC 显式 reject_uniform=False：神经网络重训练与拒单识别检查', '',
             f'{args["num_vehicles"]} 车：{args["num_ev"]} EV + {args["num_vehicles"] - args["num_ev"]} AEV；'
             f'{args["date"]} {args["start_hour"]:g}–{args["stop_hour"]:g} 点。',
             '`reject_uniform=False`、`ride_acceptance_noise_std=0`，拒单判断不抽随机数。',
             '实际规则：latent_acceptance_score < 0.5 时拒单；等于 0.5 时接受。没有筛掉拒单样本。',
             'MCMF、行为系数、充电规则均保持默认；新数据只来自真实被分配的 EV 邀约。', '',
             '## 实际样本覆盖', '',
             '| 集合 | 轮数 | 邀约数 | 接单数 | 拒单数 | 最低接单分数 |', '|---|---:|---:|---:|---:|---:|']
    for split, counts in result['coverage'].items():
        lines.append(f'| {split} | {len(result["episodes"][split])} | {counts["offers"]} | '
                     f'{counts["accepted"]} | {counts["rejected"]} | {counts["latent_acceptance_score_min"]:.6f} |')
    training = result['training']
    lines += ['', '## 是否完成有效重训练', '', f'状态：`{training["status"]}`。']
    if training['status'] != 'trained':
        lines += ['训练样本只有一个类别，无法从这些数据学习接受/拒绝的边界。',
                  '因此没有生成新的 model.json，没有复用旧随机拒单标签或人为制造拒单来凑齐类别。',
                  '全预测接单可获得高准确率，但不能证明模型会识别拒单；没有真实拒单时召回率不可估计。']
    else:
        loss = training['loss_diagnostics']
        lines += [f'训练目标：{loss["objective"]["initial"]:.8f} → {loss["objective"]["final"]:.8f}；'
                  f'L2={training["selected_l2"]}，仅按验证 BCE 选择。']
    lines += ['', '## 留出测试', '',
              '若指定参考 checkpoint，只允许神经网络 v2，并作为已有模型对照；不加载旧回归权重。',
              'deterministic_oracle 直接执行真实固定阈值规则，是诊断对照，不是训练出来的模型。', '',
              '| 预测器 / 阈值 | 接受误判拒绝 FP | 拒绝误判接受 FN | 正确拒单 TP | 精确率 | 召回率 | 准确率 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, model_metrics in result['test_metrics'].items():
        for choice, metrics in model_metrics['classifications'].items():
            if metrics is None:
                continue
            precision = '不可估计' if metrics['rejection_precision'] is None else f'{metrics["rejection_precision"]:.2%}'
            recall = '不可估计（无拒单）' if metrics['rejection_recall'] is None else f'{metrics["rejection_recall"]:.2%}'
            lines.append(f'| {name} / {choice} | {metrics["fp"]} | {metrics["fn"]} | {metrics["tp"]} | '
                         f'{precision} | {recall} | {metrics["accuracy"]:.2%} |')
    lines += ['', '## 指标口径与验证', '',
              '- 旧收集器的 oracle_acceptance_probability 是底层 logit 分数；确定性分支的实际条件接单概率为 0 或 1。',
              '- 汇总数据将原分数保存为 latent_acceptance_score，将确定性响应概率另行核对；二者均不作为预测器输入。',
              '- 拒单 RNG 入口设置了报错哨兵，所有 EV 实际回答逐次检查固定阈值与标签一致。',
              '- 每轮检查 exact MCMF、订单生命周期守恒，并保留 EV/AEV 的充电次数与时长。',
              '- 测试只在回答前旁路调用冻结模型，不影响订单分配或实际拒单。',
              f'- 是否截短：{args["max_steps"] is not None}；同一天纽约订单、多随机种子，不是跨日期泛化验证。',
              '- 本结果只解释显式启用的确定性分支，不代表当前默认随机拒单分支；旧结果文件保持不动。',
              '- summary.json / test_predictions.jsonl / 各轮 stats.json 保存全部指标与证据。', '']
    return '\n'.join(lines)


def main(argv=None):
    args = parse_args(argv)
    current_path = args.reference_run / 'model.json' if args.reference_run else None
    original_hash = hashlib.sha256(current_path.read_bytes()).hexdigest() if current_path else None
    current = BinaryAcceptanceModel.load(current_path) if current_path else None
    if current is not None:
        if current.feature_schema != 'nyc_minutes':
            raise ValueError('Reference model must use NYC minute features')
        reference_seeds = {r['seed'] for split in ('train', 'validation', 'test')
                           for r in audit.read_rows(args.reference_run / f'{split}_offers.jsonl')}
        if reference_seeds.intersection(args.train_seeds + args.validation_seeds + args.test_seeds):
            raise ValueError('Use new seeds disjoint from the original model run')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), ROOT / 'check_nyc_mcmf_acceptance.py', ROOT / 'train_acceptance_model.py',
               ROOT / 'src/acceptance_model.py', ROOT / 'src/NYCEnvironment.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with tarfile.open(args.output_dir / 'source-at-run-start.tar.gz', 'w:gz') as archive:
        for path in [*sources[:3], ROOT / 'src']:
            archive.add(path, arcname=str(path.relative_to(ROOT)),
                        filter=lambda info: None if '__pycache__' in Path(info.name).parts else info)
    audit.save_json(args.output_dir / 'configuration.json', vars(args))
    print(f'Output: {args.output_dir}', flush=True)
    data, episodes = {}, {}
    for split, seeds in [('train', args.train_seeds), ('validation', args.validation_seeds)]:
        data[split], episodes[split] = collect(args, split, seeds)
    model, training, histories = fit_if_identifiable(data['train'], data['validation'])
    audit.save_json(args.output_dir / 'training_status.json', training)
    models = {'current': current} if current is not None else {}
    if model is not None:
        model.save(args.output_dir / 'model.json')
        np.testing.assert_array_equal(model.predict_proba(data['validation']),
            BinaryAcceptanceModel.load(args.output_dir / 'model.json').predict_proba(data['validation']))
        audit.save_json(args.output_dir / 'candidate_loss_histories.json', histories)
        audit.save_rows(args.output_dir / 'loss_history.jsonl', histories[str(model.l2)])
        models['fresh'] = model
    print(f'Training status: {training["status"]}; train rejections={coverage(data["train"])["rejected"]}', flush=True)
    thresholds = freeze_thresholds(data['validation'], models)
    audit.save_json(args.output_dir / 'decision_thresholds.json', thresholds)
    data['test'], episodes['test'] = collect(args, 'test', args.test_seeds, models)
    metrics = {}
    for name, fitted in models.items():
        p = fitted.predict_proba(data['test'])
        np.testing.assert_allclose(p, [r[f'{name}_p_accept'] for r in data['test']], rtol=1e-6, atol=1e-7)
        metrics[name] = evaluate(data['test'], p, thresholds[name])
        for row, acceptance_p in zip(data['test'], p):
            q = 1 - float(acceptance_p)
            row[f'{name}_p_reject'] = q
            for choice in ('default', 'validation_f1'):
                selected = thresholds[name][choice]
                if selected is None:
                    continue
                predicted = q >= selected['threshold']
                row[f'{name}_{choice}_predict_reject'] = predicted
                row[f'{name}_{choice}_outcome'] = ('FP' if row['accepted'] else 'TP') if predicted else ('TN' if row['accepted'] else 'FN')
    metrics['deterministic_oracle'] = evaluate(data['test'],
        [r['oracle_acceptance_probability'] for r in data['test']], {'default': {'threshold': 0.5}})
    audit.save_rows(args.output_dir / 'test_predictions.jsonl', data['test'])
    if current_path is not None:
        assert hashlib.sha256(current_path.read_bytes()).hexdigest() == original_hash
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    result = dict(configuration=vars(args), rule='reject_uniform=False; reject iff acceptance score < 0.5',
                  training=training, coverage={name: coverage(rows) for name, rows in data.items()},
                  episodes=episodes, decision_thresholds=thresholds, test_metrics=metrics,
                  reference_model_path=current_path, reference_model_sha256=original_hash,
                  source_sha256=hashes, parquet_sha256=hashlib.sha256(args.parquet_path.read_bytes()).hexdigest(),
                  station_csv_sha256=hashlib.sha256(args.station_csv.read_bytes()).hexdigest())
    audit.save_json(args.output_dir / 'summary.json', result)
    (args.output_dir / 'report.md').write_text(build_report(result), encoding='utf-8')
    print(f'Finished: {args.output_dir / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
