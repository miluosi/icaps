"""Render the actual completed full-day runs without treating progress as results."""
import argparse
import json
from pathlib import Path


LABELS = {
    'no_repair': 'Integrated (No repair)',
    'evfirst_no_rejection': 'R0 (No rejection)',
    'evfirst_no_repair': 'R1 (Learned no repair)',
    'evfirst_no_repair_structured': 'C0 (Structured no repair)',
    'repair_only': 'Repair Only (R2)', 'repair_learning': 'Repair Learning (R3)',
    'recourse_macro': 'Macro Recourse-aware', 'recourse_nested_q2': 'Nested R4',
    'samitha': 'Samitha',
}


def build_report(summary, manifest):
    rows = summary['runs']
    execution = summary.get('execution', {})
    train_only = execution.get('phase') == 'train-only'
    complete = summary['completed_methods'] == summary['required_methods'] and not summary.get('failures')
    smoke_steps = manifest['arguments'].get('smoke_steps')
    scope = (f'短程冒烟测试（每阶段 {smoke_steps} 步，不是完整一天）' if smoke_steps else
             '仅训练一天，未执行测试' if train_only else '一天训练、一天独立测试')
    status = '有方法失败' if summary.get('failures') else '全部完成' if complete else '尚未全部完成，以下仅列已完成方法'
    if summary.get('status') == 'stopped_by_user':
        status = '已按用户要求停止'
    date_line = (f"训练日期：{summary['train_date']}；仅执行训练。预留的测试日期 {summary['test_date']} 未运行。"
                 if train_only else f"训练日期：{summary['train_date']}，测试日期：{summary['test_date']}；环境时间窗各 00:00–24:00。")
    lines = [f"# NYC {summary['num_vehicles']} 车：{scope}", '',
        f"状态：{status}（{len(rows)}/{summary['required_methods']}）。", '',
        date_line,
        f"车辆：{summary['num_vehicles']}，其中 EV={summary['num_ev']}，AEV={summary['num_vehicles']-summary['num_ev']}。", '',
        '## 运行与数据口径', '',
        '- 数据来源：[NYC TLC 原始 Yellow Taxi 数据](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)。使用完整日期，不再使用旧的 08:00–10:00 sample。',
        f"- 沿用项目默认 Manhattan-only 清洗与需求范围、2 km 接驾限制、{manifest['arguments']['epoch_length']:g} 秒 epoch、真实随机拒单；拒单 predictor 特征仍关闭。",
        '- 学习器：`optimization_anchored_residual`；状态：`joint_state_separate_critics`；分配：exact MCMF。这里的方法是 recourse 配置与因果控制，不是不同神经网络。',
        ('- 各方法使用相同初始神经网络权重、训练需求和配对 CRN。仅保存训练 checkpoint，没有测试数据或测试权重检查。' if train_only else
         '- 各方法使用相同初始神经网络权重、训练需求、测试需求和配对 CRN。测试从磁盘 checkpoint 加载，且验证测试不改变网络权重。'),
        f"- 每 {manifest['arguments']['train_every']} 步训练，joint batch={manifest['arguments']['batch_size']}，joint replay 容量={manifest['arguments']['joint_replay_capacity']}；测试无梯度更新。", '',
        '## 指标定义', '',
        '- **Recourse number**：当轮 EV 拒绝后，当轮分配给 AEV 的补救次数（`same_epoch_aev_assignment_count`），不是已完成订单数。',
        '- **Rejected number**：EV 实际拒绝平台订单的次数（`ev_rejected_offer_count`）。',
        '- **Accomplished number**：本阶段真正完成的全部订单数（`completed_orders`）。',
        '- **Reward**：所有车辆实际 `env.step` reward 在本阶段执行步数内的累计，保留负 reward；未扣除环境本来没有执行的额外丢单罚项。',
        '- Accomplished number 不是 accepted、assigned 或 pickup 数；`completed_number` 作为旧结果读取别名保留。',
        '- “拒单后 AEV 完成”单独列出；它包括后来轮次的 AEV rescue，不冒充“同轮分配的补救全部完成”。', '',
    ]
    if execution.get('phase') == 'test-only':
        lines += ['本次只执行测试；下表训练统计与 checkpoint 复用自：', '',
                  f"`{execution['training_reused_from']}`。没有重新训练。", '']
    if smoke_steps:
        lines += ['本报告所有累计指标只覆盖上述短程步数，不能解释为全天结果。', '']
    phases = [('training', '训练日')]
    if not train_only:
        phases.append(('testing', '测试日（独立日期）'))
    else:
        lines += ['本次仅训练；没有执行或生成测试阶段结果。后续使用 `test-only --source-dir 本目录` 测试。', '']
    for phase, title in phases:
        lines += [f'## {title}', '',
            '| 方法 | 步数 | Recourse number | Rejected number | Accomplished number | Reward | 拒单后 AEV 完成 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in rows:
            s = row[phase]
            lines.append(f"| {LABELS[row['method']]} | {s['steps']} | {s['same_epoch_aev_assignment_count']} | {s['ev_rejected_offer_count']} | {s['completed_orders']} | {s['reward']:.6f} | {s['completion_after_rejection_count']} |")
        lines.append('')
    verification_header = ('| 方法 | Joint 更新 AEV/EV | Edge 更新 AEV/EV | 训练充电次数 EV/AEV | Checkpoint 已保存 |'
                           if train_only else
                           '| 方法 | Joint 更新 AEV/EV | Edge 更新 AEV/EV | 测试充电次数 EV/AEV | Checkpoint 加载 / 测试权重不变 |')
    lines += ['## 运行检查与保留统计', '', verification_header,
        '| --- | --- | --- | --- | --- |']
    for row in rows:
        t, e = row['training'], row['training'] if train_only else row['testing']
        verified = str(row['checkpoint_saved']) if train_only else f"{row['checkpoint_loaded']} / {row['test_weights_unchanged']}"
        lines.append(f"| {LABELS[row['method']]} | {t['optimizer_steps_joint']} | {t['optimizer_steps_edge']} | {e['human_ev_charging_sessions']} / {e['aev_charging_sessions']} | {verified} |")
    if summary.get('incomplete_progress'):
        lines += ['', '## 尚未完成的方法：进度快照（不是最终结果）', '',
                  '| 方法 | 阶段 | 已执行 / 阶段总步数 |', '| --- | --- | ---: |']
        for method, progress in summary['incomplete_progress'].items():
            lines.append(f"| {LABELS[method]} | {progress['phase']} | {progress['step']} / {progress['total_steps']} |")
    if summary.get('failures'):
        lines += ['', '失败记录：`' + str(summary['failures']) + '`；查看对应 `worker.log`。']
    lines += ['', '## 解释限制', '',
        '- 每个已执行阶段仅一个日期和一个随机种子；可检查运行与该次实际表现，不足以确认收敛或统计显著优势。',
        '- Learned R1 仅保留为诊断对照。主 EV-first 因果链为 Structured R1/C0→R2→R3→Macro→R4；Integrated→Samitha 是架构比较，不能把 Integrated 与 EV-first 的差异全部归因于 repair。',
        '- Samitha 的额外 repair 统计还包括初始未分配订单；上述 Recourse number 仅统计 EV 拒单补救。',
        '- 原始详细数据位于各方法的 `training.json`、`results.json`、`checkpoint.pt`、`run.log`；进度在 `progress.json`，不当作最终结果，也不单凭旧进度认定进程仍在运行。',
        '- `reward_ledgers` 等训练 replay 明细只覆盖保留的回放窗口；本表累计 reward / recourse / completed 来自本阶段完整环境统计，不受回放容量截断；完整 2880 步运行才是全天结果。', '',
        '复现参数与数据/源代码 SHA256：`manifest.json`。汇总机器可读结果：`summary.json`；核心指标表：`metrics.json` 和 `metrics.csv`。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory / 'summary.json').read_text())
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    output = args.directory / 'REPORT.md'
    output.write_text(build_report(summary, manifest))
    print(output.resolve())


if __name__ == '__main__':
    main()
