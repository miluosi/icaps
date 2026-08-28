"""Explain completed deterministic NYC offer outputs using actual parameters.

Read-only with respect to the environment, dispatch rules, and saved models.
Write derived diagnostics beside the completed run; counterfactual calculations
hold the recorded offers fixed and are not policy reruns or training samples.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from check_nyc_mcmf_acceptance import ROOT, read_rows, save_json, save_rows
from train_acceptance_model import make_environment, parse_args as environment_args


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(min=float(values.min()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), max=float(values.max()), mean=float(values.mean()))


def inspect_outputs(rows, parameters):
    if not rows:
        raise ValueError('A nonempty completed run is required')
    x = np.asarray([[r['idle_time'], r['pickup_time'], r['surge_bonus']] for r in rows])
    coefficients = np.array([parameters[name] for name in
        ['ride_acceptance_beta_idle_min', 'ride_acceptance_beta_pickup_min', 'ride_acceptance_beta_surge']])
    asc = parameters['ride_acceptance_asc']
    contributions = x * coefficients
    u = asc + contributions.sum(axis=1)
    score = np.array([r['latent_acceptance_score'] for r in rows])
    reproduced = 1 - np.clip(1 - expit(np.clip(u, -60, 60)), 0, .99)
    accepted = np.asarray([r['accepted'] for r in rows])
    if not np.isfinite(x).all() or not np.isfinite(score).all():
        raise ValueError('Nonfinite recorded inputs')
    if np.max(np.abs(reproduced-score)) > 1e-10:
        raise AssertionError('Recorded probabilities do not match current behavior parameters')
    if not np.array_equal(accepted, score >= .5):
        raise AssertionError('Recorded labels do not follow the deterministic rule')
    speed = parameters['average_velocity_kmph']
    distance = x[:, 1] * speed / 60
    pickup_limit = parameters['assignmentrange'] / speed * 60
    terms = dict(intercept=np.full(len(rows), asc), idle=contributions[:, 0],
                 pickup=contributions[:, 1], surge=contributions[:, 2])
    summary = dict(offers=len(rows), accepted=int(accepted.sum()), rejected=int(len(rows)-accepted.sum()),
                   parameters=parameters, inputs={name: describe(x[:,i]) for i,name in
                   enumerate(['idle_minutes', 'pickup_minutes', 'surge_bonus_amount'])},
                   pickup_distance_km_from_recorded_minutes=describe(distance),
                   latent_acceptance_score=describe(score), latent_rejection_score=describe(1-score),
                   utility_terms={name: describe(value) for name, value in terms.items()},
                   logit_utility=describe(u), probability_reconstruction_max_error=float(np.abs(reproduced-score).max()),
                   range_limit_minutes=pickup_limit,
                   pickup_limit_violations=int(np.sum(distance > parameters['assignmentrange'] + 1e-5)),
                   zero_surge_offers=int(np.sum(x[:,2] == 0)),
                   same_offers_no_surge=dict(rejections=int(np.sum(u-contributions[:,2] < 0)),
                                            acceptance_score=describe(expit(u-contributions[:,2]))),
                   threshold_examples=dict(
                       pickup_minutes_at_idle_0_bonus_0=-asc/coefficients[1],
                       pickup_minutes_at_idle_half_minute_bonus_0=-(asc+coefficients[0]*.5)/coefficients[1],
                       idle_minutes_at_max_range_bonus_0=-(asc+coefficients[1]*pickup_limit)/coefficients[0]),
                   minimum_score_example=rows[int(np.argmin(score))])
    per_offer = [dict(seed=row['seed'], offer_index=row['offer_index'], vehicle_id=row['vehicle_id'],
                      request_id=row['request_id'], accepted=int(row['accepted']),
                      idle_minutes=float(x[i,0]), pickup_minutes=float(x[i,1]), pickup_distance_km=float(distance[i]),
                      surge_bonus_amount=float(x[i,2]), utility=float(u[i]),
                      latent_acceptance_score=float(score[i]), latent_rejection_score=float(1-score[i]),
                      deterministic_reject=bool(score[i] < .5),
                      actual_acceptance_probability=float(score[i] >= .5),
                      no_surge_score_same_offer=float(expit(u[i]-contributions[i,2])))
                 for i,row in enumerate(rows)]
    return summary, per_offer


def report(result):
    p = result['parameters']
    lines = ['# 当前 EV 实际邀约、接单/拒单输出与参数检查', '',
             f'检查 {result["offers"]:,} 次实际 EV 邀约：接受 {result["accepted"]:,}，拒绝 {result["rejected"]:,}。',
             '这里只统计真正分配给 EV 的邀约，不包含未分配给任何车的全部需求。', '',
             '## 公式和单位', '',
             f'U = {p["ride_acceptance_asc"]} {p["ride_acceptance_beta_idle_min"]:+g} × 空闲分钟 '
             f'{p["ride_acceptance_beta_pickup_min"]:+g} × 接客分钟 '
             f'{p["ride_acceptance_beta_surge"]:+g} × 加价金额。',
             '接单分数 = sigmoid(U)，拒单分数 = 1 - 接单分数。',
             '`reject_uniform=False` 且噪声为 0：只有拒单分数 > 0.5 才实际拒单；等于 0.5 时接受。',
             '此时分数不是实际随机响应概率，确定性回答给定输入后的概率是 0 或 1。', '',
             '| 参数 | 当前值 |', '|---|---:|']
    for name, value in p.items():
        lines.append(f'| {name} | {value} |')
    lines += ['', '## 实际输入和输出', '', '| 指标 | 最小 | 中位数 | P90 | 最大 | 平均 |', '|---|---:|---:|---:|---:|---:|']
    descriptions = dict(result['inputs'], pickup_distance_km=result['pickup_distance_km_from_recorded_minutes'],
                        acceptance_score=result['latent_acceptance_score'], rejection_score=result['latent_rejection_score'],
                        utility=result['logit_utility'])
    for name, value in descriptions.items():
        lines.append(f'| {name} | {value["min"]:.6f} | {value["median"]:.6f} | '
                     f'{value["p90"]:.6f} | {value["max"]:.6f} | {value["mean"]:.6f} |')
    threshold = result['threshold_examples']
    lines += ['', '## 为什么当前全接受／是否接客太近', '',
              f'候选接客范围上限 {p["assignmentrange"]:g} km，车速 {p["average_velocity_kmph"]:.6f} km/h，'
              f'对应最多 {result["range_limit_minutes"]:.3f} 分钟；观测越界次数 {result["pickup_limit_violations"]}。',
              f'无加价、空闲 0.5 分钟时，需要接客时间超过 {threshold["pickup_minutes_at_idle_half_minute_bonus_0"]:.3f} 分钟才拒单。',
              f'接客恰为范围上限、无加价时，需要空闲超过 {threshold["idle_minutes_at_max_range_bonus_0"]:.3f} 分钟才拒单。',
              f'实际加价金额中位数 {result["inputs"]["surge_bonus_amount"]["median"]:.3f}，'
              f'对应效用增量中位数 {result["utility_terms"]["surge"]["median"]:.3f}，进一步提高接单分数。',
              f'对同批已记录邀约仅在公式中去掉加价项，计算得到拒单 {result["same_offers_no_surge"]["rejections"]} 次。'
              '这不是改变加价后的重跑结果，不包含分配或状态反馈。',
              '纯 MCMF 请求打分使用 final_value；在可行域内它不是一个专门最小化接客距离的模型。',
              '拒单公式只使用空闲时间、接客时间、加价金额；不直接使用乘客上车后的行程距离、目的地或电量。',
              '空闲系数为负：当前实现里，空闲越久接单分数越低；这里记录代码事实，不替代论文参数有效性验证。', '',
              '## 校验', '',
              f'逐条重新计算分数的最大误差：{result["probability_reconstruction_max_error"]:.3g}。',
              '逐条规则输出与实际回答完全一致。未修改任何行为参数、MCMF 分配、模型或充电统计。',
              '完整逐条数值见 rejection_output_checks.jsonl；全部统计见 rejection_parameter_diagnostics.json。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    original = json.loads((run / 'summary.json').read_text())
    current_hash = hashlib.sha256((ROOT / 'src/NYCEnvironment.py').read_bytes()).hexdigest()
    if current_hash != original['source_sha256']['src/NYCEnvironment.py']:
        raise ValueError('Environment source changed since the completed run')
    cfg = original['configuration']
    settings = environment_args(['--environment', 'nyc', '--num-vehicles', str(cfg['num_vehicles']),
        '--num-ev', str(cfg['num_ev']), '--parquet-path', cfg['parquet_path'], '--station-csv', cfg['station_csv'],
        '--date', cfg['date'], '--start-hour', str(cfg['start_hour']), '--stop-hour', str(cfg['stop_hour']),
        '--epoch-length', str(cfg['epoch_length']), '--mcmf-backend', 'primal_dual'])
    with (run / 'parameter_environment.log').open('w') as stream, redirect_stdout(stream), redirect_stderr(stream):
        env = make_environment(settings, cfg['train_seeds'][0])
    names = ['ride_acceptance_asc', 'ride_acceptance_beta_idle_min', 'ride_acceptance_beta_pickup_min',
             'ride_acceptance_beta_surge', 'ride_acceptance_noise_std', 'rejection_logit_shift', 'reject_uniform',
             'use_range_requests', 'assignmentrange', 'request_top_k', 'average_velocity_kmph',
             'surge_min_multiplier', 'surge_max_multiplier']
    parameters = {name: getattr(env, name) for name in names}
    assert parameters['reject_uniform'] is False and parameters['ride_acceptance_noise_std'] == 0
    assert parameters['rejection_logit_shift'] == 0 and parameters['use_range_requests'] is True
    rows = [row for split in ['train', 'validation', 'test'] for row in read_rows(run / f'{split}_offers.jsonl')]
    result, per_offer = inspect_outputs(rows, parameters)
    result['environment_sha256'] = current_hash
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save_json(run / 'rejection_parameter_diagnostics.json', result)
    save_rows(run / 'rejection_output_checks.jsonl', per_offer)
    (run / 'rejection_parameters_report.md').write_text(report(result), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
