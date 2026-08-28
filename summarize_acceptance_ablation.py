"""Validate and combine independently run acceptance-ablation seeds."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tarfile

import numpy as np

from run_acceptance_ablation import summarize, json_default


def combine(root, seeds):
    rows, checks, manifests = [], [], []
    for seed in seeds:
        directory = root / f'seed-{seed}'
        saved = json.loads((directory / 'summary.json').read_text())
        manifests.append(saved['manifest'])
        rows.extend(saved['episodes'])
        checks.extend(saved['training_checks'])
    first = manifests[0]
    for manifest in manifests[1:]:
        assert manifest['source_sha256'] == first['source_sha256'], 'Run source differs'
        assert manifest['predictor_sha256'] == first['predictor_sha256'], 'Predictor differs'
        for field in ['environment', 'episodes', 'train_every', 'batch_size', 'num_vehicles',
                      'num_ev', 'test_seeds', 'learners', 'max_steps']:
            assert manifest['arguments'][field] == first['arguments'][field], field
    args = first['arguments']
    assert args['environment'] == 'synthetic', 'This report validates the complete synthetic experiment'
    expected = len(seeds) * len(args['learners']) * 2 * (args['episodes'] + len(args['test_seeds']))
    assert len(rows) == expected, 'Incomplete run set'
    assert len({(r['learner'], r['train_seed'], r['arm'], r['seed']) for r in rows}) == len(rows)
    assert args['max_steps'] is None, 'Formal report cannot use truncated episodes'
    identities = {}
    for row in rows:
        assert row['lifecycle_gap'] == 0
        assert row['generated_requests'] == row['completed_orders'] + row['expired_requests'] + row['active_requests']
        assert row['completed_orders'] == row['completed_ev_orders'] + row['completed_aev_orders']
        assert row['ev_offers'] == row['ev_accepted_offers'] + row['rejected_offers']
        assert row['exact_mcmf_calls'] == row['steps'], 'Every epoch must use exact assignment'
        charging = row['charging']
        assert charging['all_vehicle_count'] == args['num_vehicles']
        assert charging['human_ev_vehicle_count'] == args['num_ev']
        assert charging['human_ev_charging_sessions'] + charging['aev_charging_sessions'] == charging['all_vehicle_charging_sessions']
        key = (row['learner'], row['train_seed'], row['seed'])
        identities.setdefault(key, {})[row['arm']] = row
        directory = root / f'seed-{row["train_seed"]}' / row['learner'] / f'seed-{row["train_seed"]}' / row['arm']
        offers = [json.loads(x) for x in (directory / f'offers-{row["seed"]}.jsonl').read_text().splitlines()]
        rejected = [x for x in offers if not x['accepted']]
        assert len(offers) == row['ev_offers']
        assert len(rejected) == row['rejected_offers']
        assert len({x['request_id'] for x in rejected}) == row['unique_rejected_requests']
        assert np.isclose(row['ev_rejection_rate'], len(rejected) / max(1, len(offers)))
        assert np.isclose(row['completion_rate'], row['completed_orders'] / max(1, row['generated_requests']))
        if row['training']:
            assert row['joint_updates'] == row['steps'] // args['train_every']
        else:
            assert row['joint_updates'] == 0
        if row['arm'] == 'predicted' and row['ev_offers']:
            assert row['acceptance_input_count'] > 0
            assert 0 < row['acceptance_input_min'] <= row['acceptance_input_max'] <= 1
    for pair in identities.values():
        assert set(pair) == {'off', 'predicted'}
        assert pair['off']['demand_hash'] == pair['predicted']['demand_hash']
    for learner in args['learners']:
        for seed in seeds:
            pair = [c for c in checks if c['learner'] == learner and c['train_seed'] == seed]
            assert len(pair) == 2
            assert pair[0]['initial_common_weights_hash'] == pair[1]['initial_common_weights_hash']
            for check in pair:
                assert check['aev_joint_updates'] > 0 and check['ev_joint_updates'] > 0
                if check['arm'] == 'predicted':
                    assert check['ev_acceptance_weight_norm'] > 0
    source_archive = root / 'source-at-run-start.tar.gz'
    with tarfile.open(source_archive) as archive:
        for name, digest in first['source_sha256'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
    comparisons = summarize(rows)
    code_root = Path(__file__).resolve().parent
    current_source = {name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
                      for name in first['source_sha256']}
    return dict(arguments={**args, 'train_seeds': seeds, 'output_dir': str(root)}, comparisons=comparisons,
                episodes=rows, training_checks=checks, manifests=manifests,
                source_archive_sha256=hashlib.sha256(source_archive.read_bytes()).hexdigest(),
                current_source_sha256=current_source,
                files_changed_after_launch=[name for name, digest in current_source.items()
                                            if digest != first['source_sha256'][name]],
                runtime=dict(python=sys.version, platform=platform.platform(),
                             packages={name: importlib.metadata.version(name)
                                       for name in ['torch', 'numpy', 'scipy', 'pandas', 'pytest']}),
                validation='All counts, paired demand, frozen checkpoints, predictor learning and archived source verified')


def report_text(result):
    args = result['arguments']
    report = ['# Integrated 200 车 EV 接单概率消融结果', '',
              '配置：100 EV + 100 AEV；synthetic 24×24；每轮 200 步、两个虚拟日；无 sequential recourse。',
              f'每个学习器/特征组用 {len(args["train_seeds"])} 个训练种子，各训练 {args["episodes"]} 轮；'
              f'再分别用 {len(args["test_seeds"])} 个独立需求种子冻结测试。',
              '训练每 25 步进行一次 joint TD（batch=1 条完整联合转移），每个模型共 48 次更新。',
              'CPU 单线程；EV/AEV 分离 critic、共享联合回放；邻域 top-k 聚合关闭（neighbour_number=0）。',
              '所有执行分配使用 exact primal_dual MCMF；接单概率模型从启动起可用且固定，不改奖励或真实司机响应。', '',
              '## 主要结果', '',
              '以下均为每次完整冻结测试的均值；“变化”是加概率减去不加概率。', '',
              '| 学习模式 | 指标 | 不加概率 | 加概率 | 变化 | 95% 聚类 bootstrap 区间 |',
              '|---|---|---:|---:|---:|---|']
    labels = [('rejected_offers', '分配后拒单次数'), ('unique_rejected_requests', '去重被拒订单'),
              ('completed_orders', '平台完成订单'), ('completed_ev_orders', 'EV完成'),
              ('completed_aev_orders', 'AEV完成'), ('ev_offers', 'EV邀约次数'),
              ('ev_rejection_rate', 'EV拒单率'), ('completion_rate', '完单/生成')]
    for comparison in result['comparisons']:
        for metric, label in labels:
            row = comparison['metrics'][metric]
            interval = row['cluster_bootstrap_95']
            ci = '—' if interval is None else f'[{interval[0]:+.4f}, {interval[1]:+.4f}]'
            report.append(f'| {comparison["learner"]} | {label} | {row["off_mean"]:.4f} | '
                          f'{row["predicted_mean"]:.4f} | {row["delta_predicted_minus_off"]:+.4f} | {ci} |')
    report += ['', '## 结果解读', '']
    for comparison in result['comparisons']:
        metrics = comparison['metrics']
        report.append(
            f'- {comparison["learner"]}：平台完单平均变化 '
            f'{metrics["completed_orders"]["delta_predicted_minus_off"]:+.2f} 单，'
            f'实际拒单次数变化 {metrics["rejected_offers"]["delta_predicted_minus_off"]:+.2f}；'
            f'EV 邀约变化 {metrics["ev_offers"]["delta_predicted_minus_off"]:+.2f}，'
            f'拒单率变化 {100 * metrics["ev_rejection_rate"]["delta_predicted_minus_off"]:+.2f} 个百分点。'
        )
        intervals = [metrics[key]['cluster_bootstrap_95']
                     for key in ['completed_orders', 'rejected_offers']]
        if all(ci is not None and ci[0] <= 0 <= ci[1] for ci in intervals):
            report.append('  完单和拒单变化的训练种子聚类区间均跨零，本次实验未证明稳定改善。')
    report += ['', '## 不同训练种子的差值', '',
               '| 学习模式 | 训练种子 | 拒单次数变化 | 完单数变化 |', '|---|---:|---:|---:|']
    for comparison in result['comparisons']:
        rejects = comparison['metrics']['rejected_offers']['per_train_seed_delta']
        completes = comparison['metrics']['completed_orders']['per_train_seed_delta']
        for seed in args['train_seeds']:
            report.append(f'| {comparison["learner"]} | {seed} | {rejects[str(seed)]:+.2f} | {completes[str(seed)]:+.2f} |')
    report += ['', '## 保留的充电次数统计', '',
               '| 学习模式 | 概率特征 | EV充电次数 | AEV充电次数 | 全平台充电次数 |', '|---|---|---:|---:|---:|']
    for learner in args['learners']:
        for arm in ['off', 'predicted']:
            rows = [r for r in result['episodes'] if not r['training'] and r['learner'] == learner and r['arm'] == arm]
            means = [np.mean([r['charging'][key] for r in rows]) for key in
                     ['human_ev_charging_sessions', 'aev_charging_sessions', 'all_vehicle_charging_sessions']]
            report.append(f'| {learner} | {arm} | {means[0]:.2f} | {means[1]:.2f} | {means[2]:.2f} |')
    report += ['', '## 口径与验证', '',
               '- 拒单是实际 EV 邀约结果，拒单次数和去重订单数分开报告；拒单率分母为 EV 邀约次数。',
               '- 只统计时域内真正完成的订单，AEV+EV=平台完单；生成=完成+过期+仍活跃逐轮核对。',
               '- 训练与测试的每一对 on/off 均使用相同需求序列，已逐轮核对订单 ID/起终点/创建时刻 hash。',
               '- 概率输入列的学习权重非零；EV/AEV均完成 joint TD 更新。两个组的其他初始权重逐字节相同。',
               '- 每组重新载入 checkpoint 后才测试，载入前后和所有冻结测试后的网络权重 hash 相同。',
               '- 逐邀约日志与报告拒单次数重新核对；完整统计中保留所有充电次数、充电时长指标。',
               '- source-at-run-start.tar.gz 与各运行 manifest 的全部源文件 hash 完全一致。', '',
               '## 解读限制', '',
               '这是有限预算下的本地对照，不是收敛证明。Residual 默认 warmup 为500次更新，本次48次仍处在 warmup。',
               '仅3个独立训练种子，区间为按训练种子聚类的描述性 bootstrap，不能夸大显著性或外推到真实司机。',
               '区间针对固定测试种子集上的训练模型波动，不覆盖测试需求抽样的全部不确定性。',
               '拒单下降可能伴随 EV 分单减少，须同时看 EV 邀约、拒单率、EV/AEV 完单和平台总完单。',
               '指标来自 synthetic 正式实验；NYC 单独的短时接口测试与所有 pilot/冒烟样本均不混入。', '',
               '## 本地文件', '',
               '- summary.json：完整汇总数据、配对区间、逐训练种子差值及验证信息。',
               '- episodes.jsonl：全部训练和冻结测试的逐轮指标。',
               '- seed-*/学习模式/seed-*/off 或 predicted/：checkpoint.pt、run.log、offers-*.jsonl、stats-*.json。',
               '- source-at-run-start.tar.gz：正式实验运行时的精确源代码快照。', '']
    report += ['源码溯源：summary.json 同时记录当前代码哈希，以及运行启动后修改的文件。',
               '启动后的修改涉及 NYC 分配入口、NYC 统计适配、Bayes 便捷接口和检查点路径辅助函数；正式 synthetic '
               '学习代码以 source-at-run-start.tar.gz 为准。supporting-inputs.tar.gz 保存环境构造脚本及冻结概率模型。', '']
    return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])
    args = parser.parse_args()
    result = combine(args.root, args.seeds)
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2, default=json_default) + '\n')
    (args.root / 'episodes.jsonl').write_text(''.join(json.dumps(row, default=json_default) + '\n' for row in result['episodes']))
    (args.root / 'report.md').write_text(report_text(result))
    print(args.root / 'report.md')


if __name__ == '__main__':
    main()
