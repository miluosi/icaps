"""Independently verify saved v3 predictor + single-stage Q/residual runs.

No retraining and no threshold selection on test labels. Replays saved learner
checkpoints in fresh environments and reports probability/support diagnostics.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch

from src.acceptance_model import EVRejectionProbabilityModel, probability_metrics
from check_nyc_mcmf_acceptance import rejection_metrics, select_rejection_threshold
from run_acceptance_ablation import build_env, load_pair, rollout, seed_everything, weight_hash

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_predictor(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    model = EVRejectionProbabilityModel.load(directory / 'model.json')
    assert model.to_dict() == summary['model']
    assert model.VERSION == 3 and model.to_dict()['target_semantics'] == 'rejected=1'
    assert model.feature_variant == 'driver_offer_core' and len(model.feature_names) == 3
    assert model.calibration['fitted']
    assert model.calibration['validation_nll_after'] <= model.calibration['validation_nll_before'] + 1e-12
    assert not any(p.requires_grad for p in model.network.parameters())
    for name, digest in summary['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Source changed: {name}'
    rows = {split: read_rows(directory / f'{split}_offers.jsonl') for split in ('train', 'validation', 'test')}
    seeds = [set(r['seed'] for r in data) for data in rows.values()]
    assert sum(map(len, seeds)) == len(set.union(*seeds))
    for split, data in rows.items():
        assert {r['rejected'] for r in data} == {0, 1}
        assert all(r['accepted'] + r['rejected'] == 1 for r in data)
        assert all(r['feature_version'] == 3 and r['feature_variant'] == 'driver_offer_core' for r in data)
        assert probability_metrics(data, model.predict_proba(data)) == summary['metrics'][split]['model']
    policies = {r['behavior_policy_id'] for r in rows['train']}
    assert policies == {'mcmf', 'stratified', 'random'}
    assert {r['behavior_policy_id'] for r in rows['test']} == {'mcmf'}
    predictions = model.predict_proba(rows['test'])
    saved = read_rows(directory / 'test_predictions.jsonl')
    np.testing.assert_array_equal(predictions, [r['predicted_rejection_probability'] for r in saved])
    threshold, _ = select_rejection_threshold(rows['validation'], model.predict_proba(rows['validation']))
    trace = read_rows(directory / 'loss_history.jsonl')
    selected = trace[model.selected_epoch]
    assert selected['objective'] == model.fit_loss
    support = {}
    for split in rows:
        feasible = read_rows(directory / f'{split}_feasible_features.jsonl')
        support[split] = dict(selected=model.support_diagnostics(rows[split]), feasible=model.support_diagnostics(feasible))
    conditional_bins, monotonicity = {}, {}
    for name, expected_sign in (('pickup_time', 1), ('surge_bonus', -1)):
        feature = np.array([r[name] for r in rows['test']])
        boundaries = np.unique(np.quantile(feature, [0., .25, .5, .75, 1.]))
        bins = []
        for i, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            mask = (feature >= lower) & ((feature <= upper) if i == len(boundaries) - 2 else (feature < upper))
            if mask.any():
                indexes = np.flatnonzero(mask)
                bins.append(dict(lower=float(lower), upper=float(upper),
                    metrics=probability_metrics([rows['test'][j] for j in indexes], predictions[mask])))
        conditional_bins[name] = bins
        altered = [dict(row, **{name: min(row[name] + 1., model.training_support[name]['max'])}) for row in rows['test']]
        active = np.array([r[name] for r in altered]) > feature
        delta = model.predict_proba(altered)[active] - predictions[active]
        monotonicity[name] = dict(in_support_increment=1., count=int(active.sum()),
            mean_q_change=float(delta.mean()) if len(delta) else None,
            opposite_direction_fraction=float(np.mean(delta * expected_sign < -1e-7)) if len(delta) else None,
            diagnostic_only=True)
    episodes = [e for group in summary['episodes'].values() for e in group]
    charge_keys = ('human_ev_charging_sessions', 'aev_charging_sessions', 'all_vehicle_charging_sessions')
    for episode in episodes:
        c = episode['charging']
        assert c['all_vehicle_count'] == 200 and c['human_ev_vehicle_count'] == c['aev_vehicle_count'] == 100
        assert c[charge_keys[2]] == c[charge_keys[0]] + c[charge_keys[1]]
    return dict(verified=True, predictor_hash=model.predictor_hash,
        sample_counts={name: len(data) for name, data in rows.items()},
        train_bce_initial=trace[0]['binary_cross_entropy'], train_bce_selected=selected['binary_cross_entropy'],
        validation_bce_initial=trace[0]['validation_binary_cross_entropy'],
        validation_bce_selected=selected['validation_binary_cross_entropy'],
        selected_epoch=model.selected_epoch, epochs_run=model.epochs_run,
        calibration=model.calibration, test_metrics=summary['metrics']['test'],
        classification_at_half=rejection_metrics(rows['test'], predictions),
        classification_at_validation_f1=rejection_metrics(rows['test'], predictions, threshold),
        support=support, conditional_calibration_bins=conditional_bins, monotonicity_diagnostics=monotonicity,
        behavioral_direction_check_passed=all(
            value['mean_q_change'] is not None and value['mean_q_change'] * sign >= 0.
            for (name, sign) in (('pickup_time', 1), ('surge_bonus', -1))
            for value in [monotonicity[name]]),
        quality_warnings=[
            f"{name}: mean q change {monotonicity[name]['mean_q_change']:.8f} has the wrong behavioral direction"
            for name, sign in (('pickup_time', 1), ('surge_bonus', -1))
            if monotonicity[name]['mean_q_change'] is not None and monotonicity[name]['mean_q_change'] * sign < 0.],
        split=summary['split'], date_ids={name: sorted({r['day_id'] for r in data}) for name, data in rows.items()},
        charging_totals={key: sum(e['charging'][key] for e in episodes) for key in charge_keys})


def verify_integrated(directory, output):
    summary = json.loads((directory / 'summary.json').read_text())
    manifest = summary['manifest']
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Source changed: {name}'
    args = SimpleNamespace(**manifest['arguments'])
    args.acceptance_model = Path(args.acceptance_model)
    assert hashlib.sha256(args.acceptance_model.read_bytes()).hexdigest() == manifest['predictor_sha256']
    torch.set_num_threads(args.torch_threads)
    results = []
    keys = ('reward', 'generated_requests', 'ev_offers', 'rejected_offers', 'completed_orders',
            'completed_ev_orders', 'completed_aev_orders', 'expired_requests', 'active_requests',
            'lifecycle_gap', 'demand_hash', 'charging', 'exact_mcmf_calls')
    for saved in summary['episodes']:
        if saved['training']:
            assert saved['joint_updates'] > 0 and np.isfinite(saved['mean_training_loss'])
            continue
        learner, train_seed, arm, seed = (saved[k] for k in ('learner', 'train_seed', 'arm', 'seed'))
        checkpoint = directory / learner / f'seed-{train_seed}' / arm / 'checkpoint.pt'
        replay_dir = output / f'{learner}-{train_seed}-{arm}-{seed}'
        replay_dir.mkdir()
        with (replay_dir / 'run.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            env = build_env(args, seed, arm, learner, False)
            pair = load_pair(env, learner, checkpoint)
            before = weight_hash(pair)
            seed_everything(seed)
            actual = rollout(args, env, pair, training=False, seed=seed, directory=replay_dir)
            assert before == weight_hash(pair), 'Frozen inference modified critic parameters'
        for key in keys:
            assert saved[key] == actual[key], f'Reload/replay mismatch: {learner}/{arm}/{key}'
        results.append(dict(learner=learner, arm=arm, seed=seed, exact_match=True,
                            completed=actual['completed_orders'], rejected=actual['rejected_offers'], charging=actual['charging']))
        print(f'Replayed {learner}/{arm}: exact saved-result match', flush=True)
    assert len(results) == len(args.learners) * len(args.train_seeds) * len(args.test_seeds) * 2
    for check in summary['training_checks']:
        assert check['ev_joint_updates'] > 0 and check['aev_joint_updates'] > 0
        if check['arm'] == 'predicted':
            assert check['predictor_frozen'] and check['ev_response_mask_weight_norm'] > 0
            assert check['ev_acceptance_weight_norm'] > 0
    return dict(verified=True, checkpoint_replays=results, training_checks=summary['training_checks'],
                comparisons=summary['comparisons'], max_steps=args.max_steps,
                conclusion='Interface smoke only; four TD updates do not establish policy convergence or statistically reliable improvement')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor-run', type=Path, required=True)
    parser.add_argument('--integrated-run', type=Path, required=True)
    parser.add_argument('--pytest-xml', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(args.pytest_xml)
    suites = list(tree.iter('testsuite'))
    tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    assert tests['tests'] > 0 and tests['failures'] == tests['errors'] == 0
    result = dict(verified=True, tests=tests, predictor=verify_predictor(args.predictor_run),
                  integrated=verify_integrated(args.integrated_run, args.output_dir))
    (args.output_dir / 'verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    p = result['predictor']
    m, baseline = p['test_metrics']['model'], p['test_metrics']['constant_train_rate']
    lines = ['# EV 拒单 v3 本地验收', '',
        f"全量测试：{tests['tests']} 项，失败 {tests['failures']}，错误 {tests['errors']}，跳过 {tests['skipped']}。",
        '范围：当前拒单 trainer + 单阶段 Integrated Direct-Q / residual；未实现多阶段 ADP。', '',
        '## 拒单预测', '',
        f"样本数：{p['sample_counts']}；{p['split']}。",
        f"训练 BCE：{p['train_bce_initial']:.6f} → {p['train_bce_selected']:.6f}。",
        f"验证 BCE：{p['validation_bce_initial']:.6f} → {p['validation_bce_selected']:.6f}。",
        f"校准验证 NLL：{p['calibration']['validation_nll_before']:.6f} → {p['calibration']['validation_nll_after']:.6f}。", '',
        '| 留出测试指标 | 三输入拒单网络 | 常数基线 |', '|---|---:|---:|',
        *[f'| {key} | {m[key]:.6f} | {baseline[key]:.6f} |' for key in ('log_loss', 'brier_score', 'roc_auc', 'ece_10_bins')], '',
        f"0.5 阈值：TP={p['classification_at_half']['tp']}、FP={p['classification_at_half']['fp']}、FN={p['classification_at_half']['fn']}、TN={p['classification_at_half']['tn']}。概率区分能力不等于能预知每次随机回答。",
        f"采集仿真累计充电：{p['charging_totals']}。", '',
        '## 训练与推理', '',
        '四组检查点均在新环境独立加载并复跑；奖励、拒单、完单、需求、充电统计与保存结果逐项一致。',
        'q 与 mask 输入列均收到 TD 更新，预测器保持冻结；详细结果见 verification.json。', '',
        '## 概率质量告警（不与代码测试混为一谈）', '',
        f"行为方向诊断是否通过：{p['behavioral_direction_check_passed']}。",
        *[f'- {warning}' for warning in p['quality_warnings']],
        f"测试 ECE 为 {m['ece_10_bins']:.6f}，常数基线为 {baseline['ece_10_bins']:.6f}；验证集校准不保证有限测试集 ECE 改善。",
        '该短训模型还不能判定为概率质量完全合格。未用测试标签重新选模型或硬加单调性损失。', '',
        '## 不应从本测试得出的结论', '',
        '- 只有一个 NYC 需求日期，尚未验证跨日期泛化或真实司机外部有效性。',
        '- 混合可行采集扩大了覆盖，但仍需查看 support 的越界比例；没有证明所有部署状态均被覆盖。',
        '- 40 步/4 次 TD 更新是接口冒烟测试，不是已收敛的性能实验。',
        '- 尚未运行完整 learned-vs-oracle 策略差距、三种集成方式长期消融或多拒单率实验。',
        '- anchor 是现有成功服务 surrogate 与显式拒单 penalty 的期望；不是单时间步全部现金奖励的期望。', '']
    (args.output_dir / 'verification.md').write_text('\n'.join(lines))
    print(f'Verification saved: {args.output_dir / "verification.md"}', flush=True)


if __name__ == '__main__':
    main()
