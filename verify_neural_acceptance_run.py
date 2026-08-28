"""Verify a finished neural NYC experiment and save a compact acceptance audit.

This never fits a model or chooses a threshold on test labels. It checks the
saved checkpoint, source hashes, split isolation, paired trajectories and
charging counters, then describes calibration using predeclared risk bins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_nyc_mcmf_acceptance import read_rows, save_json, rejection_metrics, select_rejection_threshold
from src.acceptance_model import BinaryAcceptanceModel, probability_metrics
from src.acceptance_inputs import FEATURE_NAMES, FEATURE_VERSION
from train_acceptance_model import ROOT


def verify(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    config = summary['configuration']
    model = BinaryAcceptanceModel.load(directory / 'model.json')
    assert model.to_dict() == summary['model']
    assert model.VERSION == 3 and model.MODEL_TYPE == 'mlp_ev_rejection'
    assert len(FEATURE_NAMES) == 3 and not any(p.requires_grad for p in model.network.parameters())
    for name, digest in summary['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Source changed: {name}'
    data = {split: read_rows(directory / f'{split}_offers.jsonl') for split in ('train', 'validation', 'test')}
    seed_groups = [set(r['seed'] for r in data[split]) for split in data]
    assert sum(map(len, seed_groups)) == len(set.union(*seed_groups)), 'Overlapping split seeds'
    for split, rows in data.items():
        assert {r['seed'] for r in rows} == set(config[f'{split}_seeds'])
        assert len(rows) == summary['sample_counts'][split]
        assert all(r['feature_version'] == FEATURE_VERSION for r in rows)
        assert set(r['accepted'] for r in rows) == {0, 1}, f'Missing class in {split}'
        assert all(r['reject_uniform'] is True for r in rows)
        assert all(bool(not r['accepted']) == (r['response_uniform'] < r['oracle_rejection_probability']) for r in rows)
        assert all(r['pickup_distance_km'] <= 2 + 1e-5 for r in rows)
        model._features(rows)  # all required fields must be present and finite
    test_p = model.predict_proba(data['test'])
    np.testing.assert_allclose(test_p, [r['fresh_p_reject'] for r in data['test']], rtol=1e-6, atol=1e-7)
    threshold, _ = select_rejection_threshold(data['validation'], model.predict_proba(data['validation']))
    saved_threshold = summary['frozen_validation_thresholds']['fresh']['threshold']
    assert threshold == saved_threshold
    for name, value in [('default_threshold', .5), ('validation_f1_threshold', threshold)]:
        metrics = rejection_metrics(data['test'], test_p, value)
        assert metrics == summary['fresh_test']['fresh_fit'][name]
    computed = probability_metrics(data['test'], test_p)
    for key, value in computed.items():
        assert value == summary['fresh_test']['fresh_fit']['probability'][key], key
    episodes = [*summary['train_episodes'], *summary['validation_episodes']]
    for pair in summary['test_episodes']:
        a, b = pair['baseline'], pair['shadow']
        assert pair['identical_trajectory'] and a['trajectory_hash'] == b['trajectory_hash']
        for key in ('charging', 'rejected_offers', 'completed_orders', 'reward'):
            assert a[key] == b[key], key
        episodes.extend([a, b])
    expected_episodes = len(config['train_seeds']) + len(config['validation_seeds']) + 2 * len(config['test_seeds'])
    assert len(episodes) == expected_episodes
    for episode in episodes:
        env = episode['environment']
        assert env['reject_uniform'] is True and env['use_range_requests'] is True and env['assignmentrange'] == 2.0
        assert episode['verified_ev_responses'] == episode['ev_offers']
        assert episode['response_uniform_unique_count'] > 1
        if config['max_steps'] is None:
            expected_steps = int(np.ceil((config['stop_hour'] - config['start_hour']) * 3600 / config['epoch_length']))
            assert episode['steps'] == expected_steps
        charging = episode['charging']
        assert charging['all_vehicle_charging_sessions'] == charging['human_ev_charging_sessions'] + charging['aev_charging_sessions']
    q = test_p
    actual = np.asarray([r['accepted'] == 0 for r in data['test']])
    oracle = np.asarray([r['oracle_rejection_probability'] for r in data['test']])
    bins = []
    boundaries = [0, .01, .025, .05, .075, .1, .15, .2, .3, .5, 1.000001]
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (q >= lower) & (q < upper)
        if mask.any():
            bins.append(dict(lower=lower, upper=min(upper, 1.0), count=int(mask.sum()),
                predicted_rejection_probability=float(q[mask].mean()),
                actual_rejection_rate=float(actual[mask].mean()),
                oracle_rejection_probability=float(oracle[mask].mean())))
    return dict(verified=True, neural_model_type=model.MODEL_TYPE, features=list(FEATURE_NAMES),
        episodes=len(episodes), paired_test_seeds=len(summary['test_episodes']), sample_counts=summary['sample_counts'],
        actual_test_rejections=int(actual.sum()), validation_threshold=threshold,
        selected_epoch=model.selected_epoch, epochs_run=model.epochs_run,
        charging_sessions_all_episodes={key: sum(ep['charging'][key] for ep in episodes) for key in
            ('human_ev_charging_sessions', 'aev_charging_sessions', 'all_vehicle_charging_sessions')},
        test_calibration_bins=bins, test_metrics=summary['fresh_test'],
        checks=['neural checkpoint and feature schema', 'source hashes', 'disjoint complete seeds',
                'actual stochastic responses and 2 km radius', 'prediction roundtrip and saved metrics',
                'validation-only threshold', 'paired full trajectories', 'preserved charging counters'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    result = verify(args.run_directory)
    save_json(args.run_directory / 'neural_verification.json', result)
    report = ['# 神经网络实验复核', '',
              f'复核通过：{result["episodes"]} 轮仿真、{result["paired_test_seeds"]} 组配对测试，v3 三特征拒单神经网络。',
              f'训练/验证/测试样本：{result["sample_counts"]}；测试真实拒单 {result["actual_test_rejections"]} 次。', '',
              '已核对：' + '；'.join(result['checks']) + '。', '',
              f'所有仿真累计充电次数（包含测试的两个对照臂）：{result["charging_sessions_all_episodes"]}。', '',
              '## 固定区间概率校准', '',
              '| 预测拒单概率区间 | 样本数 | 平均预测概率 | 实际拒单率 | 平均仿真条件概率 |',
              '|---|---:|---:|---:|---:|']
    for row in result['test_calibration_bins']:
        report.append(f'| {row["lower"]:.1%}–{row["upper"]:.1%} | {row["count"]} | '
                      f'{row["predicted_rejection_probability"]:.3%} | {row["actual_rejection_rate"]:.3%} | '
                      f'{row["oracle_rejection_probability"]:.3%} |')
    report += ['', '这些是描述性校准分组，不用测试标签调阈值。单次随机回答不能由概率预测器确定。', '']
    (args.run_directory / 'neural_verification.md').write_text('\n'.join(report))
    print(json.dumps({k: result[k] for k in ('verified', 'episodes', 'sample_counts', 'actual_test_rejections')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
