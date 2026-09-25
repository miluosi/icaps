# 代码与实验清单核查（2026-09-23）

核查依据：`必须核对的代码与实验清单.md`（2026-09-22）。本轮仅修复评分一致性的实质问题；没有重训、改写历史 NYC 实验数值、修改论文或更换优化算法。

## 结论与修改

| 清单项 | 核查结论及处理 |
|---|---|
| Huber、PER、target twin minimum、terminal | 保留当前实现：阈值 1 的加权 smooth-L1；target 逐边取 minimum 再求和；terminal 不 bootstrap。相关现有回归测试通过。 |
| 部署与 replay anchor | 已修复。学习方法统一调用 `assignment_anchor`，无论 response anchor 是否启用都保存学习器实际使用的未量化基准分数。原先 response off 分支可能把纯票价保存为 replay anchor，而部署扣除了行驶成本。 |
| 特征与 clipping | 已修复 NYC 服务行程时长、每车充电时长、WAIT 的网络输入编码、其他在线车辆数量和等待预测截断的不一致。按车队的候选边计算 clipping dispersion，固定 continuing edges 不再改变可选边的截断尺度。容量到达桶仍约束资源，不再覆盖另一口径的网络时间特征。 |
| R2 structured-only follower | 保留实际 matrix scorer，显式标记为 `matrix_anchor_only`，回放时不意外叠加网络残差。Myopic 基准分数定义不变。 |
| 学习 reward 与系统 reward | 不修改：Macro leader 使用 system reward 减低 SoC 等待惩罚；AEV follower 使用 AEV reward 减该惩罚；r1/r2/r3 leader 使用 EV reward。运营报表仍是未扣此学习惩罚的系统 reward。不能声称此 shaping 保持原运营目标的最优策略。 |
| Macro 历史 follower replay | 保留 behavior-mixture TD 近似；没有切换为 R4，也没有增加“当前 follower 无偏 Bellman 样本”的保证。 |
| 真正空图 | 当前训练入口默认 `ifdropoff=False`，且论文配置两类车均非空；忙碌车辆仍有 continuing edges。两阶段全忙的确定性测试共检查 2 张图，真正空图 0 张。没有历史训练图日志，不能给出历史训练空图次数或对应未来收益统计。开启 dropout、零车队等配置的表达缺口仍在，不在本轮添加新值网络。`daily_drop_off=False` 本身不代表禁止离线，禁止离线来自 `ifdropoff=False`。 |
| Conservative 条件保证 | 图中 Conservative 确为 `real_conservative`，Reactive 来自旧 `current`。实际预约执行、专用 AEV 站的检查见下文。保留初始可行日历、绑定 plug/interval、确定性执行、同刻先释放后到达等前提。 |
| SSG/求解时间 | 不改 SSG 或 solver。读取现有正向弧数生成删边表；修正 relocation-rich LaTeX 表“读取 end-to-end 却标作 solve”的错误，使该表读取 `solve_seconds`。原始时间数据保留。 |

主要代码位置：

- `src/ValueFunction_st_masac_gat.py`：`assignment_anchor`、`batch_get_mixed_q_values`、`_edge_experience`、`_queue_wait_features_for_edges`、`_correction_bounds_for_edges`、`_graph_edge_scores`。
- `src/recourse/state_snapshot.py`：`feasible_graph_from_matrix`。
- `src/NYCEnvironment.py`：`generate_vehicle_qvalue`。

这些修改统一的是给定网络下的评分与回放语义。SSG 对给定分数的缩图规则、OR-Tools 后端和原有整数精度策略未修改；不能由此推出学习策略具有全局最优保证。

## 确定性验证

新增 `tests/test_nyc_scoring_consistency.py`，9 项检查覆盖：

- response anchor off/on，零残差、随机非零残差和触发截断的大残差。
- 实际 NYC batch 输入与 replay 输入逐边比较；EV/AEV 和 service/charge/reloc/wait。
- 逐边 anchor 与部署完整分数对齐，OR-Tools 联合投影目标值一致（允许等价并列解）。
- 充电等待预测、实际服务时长及忙碌车辆的影响。
- R2 的 structured-only scorer 不被共享网络对象恢复后的模式改变。
- 修改 live environment 后，已存图的评分不被当前状态污染。
- 固定 continuing edges 不改变候选边 clipping；两阶段全忙仍非空。

最终检查：

```bash
MPLCONFIGDIR=/tmp/icaps-mpl .venv/bin/python -m pytest -q \
  tests/test_nyc_scoring_consistency.py \
  tests/test_rejection_v3_contract.py tests/test_recourse_must_fix.py \
  tests/test_recourse_reaudit.py tests/test_joint_batch_speedups.py \
  tests/test_nyc_arrival_snapshot.py tests/test_qvalue_inference.py \
  tests/test_nyc_soc_wait_learning.py
# 142 项通过，退出码 0

MPLCONFIGDIR=/tmp/icaps-mpl .venv/bin/python -m pytest -q \
  tests/test_real_conservative_charging.py tests/test_real_conservative_nyc_cli.py
# 14 项通过，退出码 0
```

测试重点是 NYC 论文配置及上述分支，不等同于穷尽所有可选 encoder、邻居数量、fleet-local 状态或 R4 配置。未运行 3000 车全流水线性能实验，不能据此承诺 30 秒。

## 现有预约记录复核

来源：`results/nyc_real_conservative_1000_20260919/*/reservation_audit.json`，仅 `real_conservative`。

- burst/staggered 各 10 个种子，共 20 次实验；1000 车，其中 500 HEV，3 个 AEV 站。
- 共 8,472 条 `arrival_start` 事件：正等待事件 0，最大等待 0；实际到达与预约到达不一致事件 0。
- 原记录累计 9,620 次 invariant checks。
- 这是实际执行了预约的 AEV 到站统计，不包含出发前等待，不是 HEV 公共站统计，也不代表未到达预约已完成验证。
- Reactive 与此模式同时存在运动终点/执行时序差别，现有比较不能仅归因于 admission mask。无需为当前条件式机制保证重跑这组实验。

## SSG 删边表

`plot_cplex_mcmf_ssg.ipynb` 已执行，只读现有原始数据，不重跑 solver：

```python
raw['ssg_edge_shrink_number'] = raw.original_graph_edges - raw.graph_edges
```

统计正向 flow arcs，包含 source/resource/fallback，不含残量反向弧；不是仅车辆—动作候选边。共 720 条方法运行记录、180 个实例、72 组，每组 10 个独立种子；两个 SSG 后端不混作额外种子。按场景、NV/NA/NC/NR、方法保存均值、样本标准差、最小/最大值和缩减率均值。校验了同实例两个 SSG 后端边数一致、未缩图方法删边为零，以及原有缩减率一致性。

输出在 `results/cplex_mcmf_ssg/tables/`：

- `ssg_edge_counts_raw.csv`：逐次记录。
- `ssg_edge_counts_summary.csv`：分组统计。
- `ssg_edge_counts.tex`：全场景均值 LaTeX 表。
- `reloc_rich_solve_time.tex`：原 relocation-rich 表加入 NE 删边均值，保留毫秒和大于 1000 的科学计数格式。

3000 车、MCMF+SSG 示例：

| 场景 | 原边数均值 | 保留边数均值 | 删除边数均值 | 缩减率均值 |
|---|---:|---:|---:|---:|
| adp_control | 567,484.90 | 492,367.90 | 75,117.00 | 13.24% |
| aev_joint | 1,699,591.80 | 625,924.70 | 1,073,667.10 | 63.17% |
| reloc_rich | 1,427,871.90 | 491,332.20 | 936,539.70 | 65.59% |

`solve_seconds` 的实际范围是 backend 模型创建、优化、解码与精确 assignment 检查，不是优化器内部 solve 时间。`end_to_end_seconds` 加上 graph build/SSG，仍不含候选生成和神经网络评分等完整 NYC 流程。CPLEX 配置为 continuous network LP；不能把速度差解释为 LP 对 MIP。

## 历史结果与后续适用范围

本地旧图 manifest 能关联输入 Excel hash、日期和评价种子，但没有完整关联到训练代码、checkpoint hash、resolved response-anchor/shaping 配置；本地 smoke checkpoints 不能代替服务器正式训练 checkpoint。无法断言所有历史运行都触发同一分支，也没有用当前默认值补写历史事实。

本次修改涉及学习输入和 target score。**使用修复后实现作为论文算法时，应重新训练并评价受影响的 learned arms；只加载旧权重重新测试不足以替代重训。** 旧 replay 中缺失的统一 anchor/WAIT 元数据也不能自动视为已修复，重训应重新收集 replay。旧 NYC 表保留为旧实现的描述性结果，本轮没有修改其数值。

按清单优先做 r3 与 Macro 的配对重训/评价并保存代码 hash、命令、配置和 checkpoint hash；多个测试日期不等于独立训练种子。本轮没有自动启动这类大规模实验，也没有修改学习 reward、新增网络或改动 Macro 近似定义。

本次检查日志、逐运行预约审计汇总和源码 hash 保存在 `results/code_audit_20260923/`。
