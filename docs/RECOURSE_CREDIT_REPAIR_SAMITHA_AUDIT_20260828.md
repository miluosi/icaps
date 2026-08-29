# Recourse credit、Repair 和 Integrated 修订与本地测试

日期：2026-08-28。依据用户提供的 `ICAPS_RECOURSE_CREDIT_REPAIR_AND_SAMITHA_IMPLEMENTATION_AUDIT.md` 核对和修改。

## 1. 保留内容与模式对应

按用户最后的补充要求：**保留 R1；确认既有 Integrated 存在，不删除、不用 R1 替换 Integrated。**

| CLI `--recourse-method` | 实际运行模式 / variant | 当轮拒单补救 | 第一阶段训练目标 |
| --- | --- | --- | --- |
| `no_repair` | 原有 `integrated / legacy` | 无第二次分配 | 原 Integrated 联合 system TD |
| `evfirst_no_repair` | 原有 `evfirst / r1` | 不补救当轮 EV 拒单；AEV 仍可接未派订单 | 原 R1 目标 |
| `repair_only` | `evfirst / r2` | structured/myopic AEV repair | EV 自身 reward + 下一轮 EV value；不接当轮 AEV credit |
| `repair_learning` | `evfirst / r3` | learned AEV repair | 同 R2 的 uncoupled EV target；AEV follower 单独学习 |
| `recourse_macro`，别名 `recourse_aware` | `evfirst / recourse_macro` | 与 R3 相同的 learned repair | 当轮 EV+AEV 全部实际 reward + 下一轮 EV value |
| `recourse_nested_q2` | 原 `evfirst / r4` | learned repair | 保留 EV reward + 当轮 residual-state follower target value |
| `samitha` | 新增 `integrated_repair / legacy` | 初始明确 hold 的 AEV 做有限补救 | 单次 Integrated macro system TD；不另训 repair Q2 |

新增模式配置在 `src/recourse/config.py`。`operating_mode`、`repair_policy`、`leader_credit` 分开记录，旧 r0–r4 仍可用。显式 named method 默认选择 `optimization_anchored_residual`，不改变未指定 named method 时的旧默认入口。

**实验解释不能混淆：** `R1 → R2 → R3 → recourse_macro` 是同一 EV-first 架构的逐项消融；`Integrated → Samitha` 是 Integrated 架构内的补救对照。将 Integrated 与 EV-first 比较时同时改变了运行架构，不能把差异全归因于 repair。

## 2. 按审计项修改的内容

| 审计章节 | 实现 / 验证 |
| --- | --- |
| §3–4 / P0：macro credit | 新增 `macro_realized` target family；实际 joint trainer 使用 `reward_system`。R3 与 macro 共用同一物理 repair policy，仅 EV Bellman credit 不同。 |
| §4.3 / P0：保留 nested | R4 不改成 macro；`recourse_nested_q2` 明确映射到旧 R4。 |
| §3.2：Repair Only | R2 follower 的 `train_step(ifEV=False)` 直接返回；包括 queue/demand 辅助优化器；synthetic trainer 不为 R2 AEV 预训练辅助预测器。 |
| §3.3：Repair Learning | R3 follower 使用 learned collection score 并有真实梯度；EV target 不读取当轮 Q2。 |
| §5 / P0：joint-only | R2/R3/R4/macro 的主训练入口仅执行 joint TD；旧 equal-share edge 路径保留兼容代码，但不能进入这些主方法。非 joint 的旧 learner 与这些方法组合会报错，不静默退回 edge TD。 |
| §6 / P0：PER | recourse bonus 只作用于 rejected、eligible、已分配给 AEV、有 same-epoch link 且 assignment epoch 等于 first rejected epoch 的事件。unoffered、later rescue、EV 再接受不享受此 bonus。 |
| §7 / P0：奖励账本 | 新增不可变 `RewardLedger`，从实际 `env.step` 返回的 reward 分类；检查 EV、AEV、system 三者分别对账，不用 option score 代替实际收入。见下面的奖励口径限制。 |
| §8–9 / P1：Samitha | NYC 和 synthetic 新增共享 `simulate_motion_integrated_repair` 实现；第一阶段是全车队 Integrated 图，显式添加 `hold_for_repair`。 |
| §8.2–8.4：hold / commitment | 仅健康且可调度的 AEV 可 hold；已服务、充电及等待充电等承诺不能重派；第二阶段只允许初始实际选中 hold 的 AEV。 |
| §8.3：候选请求 | 初始 EV 拒单 + 初始未分配订单；已承诺服务的请求不进入候选集。修复了被拒 EV offer 错占请求容量的问题。 |
| §9.3：残余容量 | 第一阶段新增 charge/resource commitment 从初始容量扣一次；已有 continuing charge 不重复扣；以当前剩余容量再取更严格的约束。 |
| §9.7–9.8：生命周期 / 统计 | 记录 `repair_architecture`、初始 EV offer、AEV commit、hold、拒单/未分配候选、repair assignment/pickup/completion、committed reassignment。专门区分未派单 repair 与真正 EV-rejection recourse，初始 AEV 服务不算 repair。 |
| §11：确定性测试 | 覆盖 C1/C2 行为与梯度、C3 数值目标、macro/nested 身份、无 edge TD、PER、两环境 limited hold、残余充电容量、无 residual 时与既有 Integrated 等价。 |
| §12 / P0：配对实验 | 新增 `run_recourse_audit.py`，一组 seed/day 跑七种方法；一致初始化权重与需求流，按 offer 和 vehicle/epoch 键生成 CRN。 |
| §13：指标 | 保存实际 EV/AEV/system reward、订单/拒单/修复生命周期、充电次数、reward ledger、solver/target projection 时间、joint/edge 更新次数、TD target mean/std、残差大小和裁剪比例。另提供固定图上的 ordinary-service displacement 诊断。 |
| P2：阶段命名 | `RecourseTransition` 新增 `stage1_graph/stage2_graph` 及 joint-action 只读别名；旧字段继续可读取。Integrated repair 的 stage 1 表示初始全车队计划，不是 EV-only。 |
| P2：兼容性 | replay v3 保留，新增 `recourse_credit_schema=1` 扩展标记；兼容旧 hash，迁移 bonus 和 target-family 元数据；不为旧数据伪造 reward ledger。旧 R4 仍解释为 nested。 |

### 核心目标与实际训练

设 `G1` 为当前已选择联合动作的 structured anchor，`V1_next` 由真实链接的下一轮 feasible graph 进行 online selection / lagged target evaluation 得到。

```text
macro full target     = reward_ev + reward_aev + (1-done) * gamma^elapsed * V1_next
macro residual target = macro full target - G1
nested R4 full target = reward_ev + within_epoch_gamma * V2_target(residual state)
R2/R3 EV full target  = reward_ev + (1-done) * gamma^elapsed * V1_next
follower full target  = reward_aev + (1-done) * gamma^elapsed * V1_next
```

Macro 没有把当轮 AEV reward 再乘一次跨轮折扣，也没有调用近似 Q2 代替已观测 reward。终止时只去掉跨轮 continuation，不去掉当轮 reward。非终止样本缺失下一条真实 transition 链接时不更新，不构造虚假的下一状态。

数值回归：`reward_ev=-2, reward_aev=7, gamma=.9, V1_next=10, G1=4` 时，full target 是 **14**，residual target 是 **10**；终止时分别是 **5** 和 **1**。Full-Q 分支不减 `G1`。精确 follower 给出 16 时 nested 与 macro 都是 14，近似 follower 偏离时二者不必相等。

Samitha 的 `integrated_repair` 按 system phase 训练一次 joint macro loss；它可以通过 fleet router 更新两套 critic，但不是两个独立 Bellman 更新，也没有新建多步 ADP lookahead 或 full provisional replan。

### 奖励与指标口径限制

- 这里的 system reward 是当前环境**实际返回**的奖励总和。两环境当前没有单独实际扣除 request-expiry/lost-order penalty；这些账本项保留为 0。没有把遗留但未执行的 `unserved_penalty` 参数擅自加到目标。不能把这个 return 描述成“已包含所有丢单成本的经济利润”。
- `RewardLedger.stage1/stage2` 沿用审计命名，分别汇总 EV/AEV 奖励；对 Integrated repair 而言它们不是两个实际执行阶段各自的现金流，system 总额才是其 macro reward。
- `ordinary_aev_service_displacement_fixed_graph` 是训练回放中固定状态、固定 scores/capacities，屏蔽当前拒单请求后多出的普通服务数；不是跨轨迹因果损失估计。未改变实际执行动作。
- `deployment_clipping_rate` 统计经过 graph scorer 的在线动作评分（包括训练中的 online projection）；`gradient_clipping_rate` 另计。TD target 本身不做部署裁剪。
- 默认拒单 predictor 和 `q_reject` 特征仍然关闭；真实 Bernoulli 拒单机制仍开启。充电次数、每车日均充电次数、充电时长统计保留。

## 3. 本地验证

### 自动化测试

```bash
cd /Users/seinzhou/Desktop/icaps
/opt/anaconda3/bin/python -m pytest -o addopts='' -q --junitxml=results/recourse_credit/regression-20260828.xml
/opt/anaconda3/bin/python -m compileall -q run_trainer.py run_nyctrainer.py run_recourse_audit.py src tests
git diff --check
```

本轮最终结果：**360 passed，0 failed**；一条 pyogrio/Shapely 弃用警告。compileall 和 diff whitespace 检查通过。JUnit：`results/recourse_credit/regression-20260828.xml`。

新增测试文件：

- `tests/test_recourse_credit.py`：目标数值、PER、旧回放、R2 冻结、配对 CRN、真实 checkpoint 指标、统计口径。
- `tests/test_repair_only_learning.py`：真实 NYC EV-first 收集，比较 R2/R3 可行域、scores、EV target 与 follower 参数梯度。
- `tests/test_integrated_repair.py`：两环境真实执行、hold/commit/resource 边界、无 residual 的 Integrated 等价性、NYC relocation 索引、未派单 repair 的 pickup/completion。

### 训练与重载推理 smoke

```bash
/opt/anaconda3/bin/python run_recourse_audit.py \
  --environment nyc --num-vehicles 200 --num-ev 100 \
  --seeds 71 --max-steps 12 --train-every 3 \
  --output-dir results/recourse_credit/nyc-200-100ev-100aev-20260828-final

/opt/anaconda3/bin/python run_recourse_audit.py \
  --environment synthetic --num-vehicles 8 --num-ev 4 \
  --seeds 71 --max-steps 8 --train-every 2 \
  --output-dir results/recourse_credit/synthetic-20260828-final-v3
```

NYC 使用本地 `yellow_tripdata_2025-12-18_sample.parquet`、30 秒 epoch、2 km 接驾范围、真实随机拒单；100 EV + 100 AEV。每模式先训练，再在不同 seed 下推理；重新构造模型、加载 checkpoint 后重复同一推理。检查 system/EV/AEV reward、完成订单、offer/rejection、Samitha repair、EV/AEV 充电次数完全一致，缺失检查字段会报错而不是把两个 `None` 当相等。

每个方法文件夹保存 `checkpoint.pt`、`run.log`、`results.json`；根目录 `summary.json` 保存模式配置、源代码 hash、配对差值和实际检查字段。**单 seed、12 epoch 是运行链路 smoke，不是收敛实验，也不足以判断哪种方法效果更好。** 单 seed 的 CI 明确保存为 null。

最终 NYC 推理统计如下（评估 seed=90071；充电次数是启动的 charging sessions，不是完成充电次数）：

| 模式 | 训练 joint 更新 AEV/EV | 推理 system reward | 完成单 | EV 拒单 | 同轮拒单补救 assignment | 充电次数 EV/AEV |
| --- | --- | --- | --- | --- | --- | --- |
| no_repair / Integrated | 4 / 4 | 64.296672 | 9 | 8 | 0 | 1 / 85 |
| evfirst_no_repair / R1 | 4 / 4 | 78.641033 | 9 | 6 | 0 | 1 / 85 |
| repair_only / R2 | 0 / 4 | 13.102187 | 8 | 11 | 11 | 1 / 0 |
| repair_learning / R3 | 4 / 4 | 80.672780 | 9 | 6 | 1 | 1 / 85 |
| recourse_macro | 4 / 4 | 78.510137 | 9 | 6 | 1 | 1 / 85 |
| recourse_nested_q2 / R4 | 4 / 4 | 29.160142 | 9 | 9 | 0 | 1 / 85 |
| samitha | 4 / 4 | -1.046509 | 3 | 13 | 13 | 1 / 0 |

七种模式的 edge optimizer steps 均为 `[0, 0]`，checkpoint 重载检查均通过。Integrated/Samitha 的两套参数各更新 4 次，来自同一 system joint loss；不能将其解释为各训了一个独立 follower。Samitha 初始承诺 AEV 的重派次数为 **0**；repair pickup 为 **9**，repair completion 为 **0**，没有把 assignment 或 pickup 冒充完成订单。

NYC 与 synthetic 各七种方法均完成检查；每组方法的初始权重 hash 相同，训练需求 hash 和评估需求 hash 各自相同。Synthetic 8-step 自然需求短测未出现 EV offer，不作为拒单补救有效性证据；该机制由上述 NYC 实际拒单及两环境强制接受/拒绝的确定性测试覆盖。

最终结果位置：

- `results/recourse_credit/nyc-200-100ev-100aev-20260828-final/summary.json`
- `results/recourse_credit/synthetic-20260828-final-v3/summary.json`

早期 `nyc-smoke-*`、`synthetic-smoke-*`、`*-verified`、`*-preflight` 和 synthetic `final` / `final-v2` 是调试历史，不作为这张最终表的来源：其中曾定位到 NYC relocation 索引、CRN run-id、两环境不同充电字段名，以及 synthetic 人类充电 CDF/cooldown 一致性的问题，均已修正并重新检查；未删除历史结果。

## 4. 常规训练和推理入口

例：启动 NYC macro recourse training。将 `--recourse-method` 改为表格中的其他方法即可选择对应配置。

```bash
cd /Users/seinzhou/Desktop/icaps
/opt/anaconda3/bin/python run_nyctrainer.py \
  --recourse-method recourse_macro \
  --num-vehicles 200 --num-ev 100 --episodes 1 \
  --learner-variant optimization_anchored_residual \
  --state-variant joint_state_separate_critics --ev-response-feature off \
  --parquet-path nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet \
  --station-csv nyedata/nyc_all_charging_stations.csv \
  --start-date 2025-12-18 --end-date 2025-12-18 \
  --start-hour 8 --stop-hour 9 --epoch-length 30 \
  --use-mcmf --mcmf-backend primal_dual \
  --batch-size 64 --training-frequency 10 --start-training-episode 0 \
  --checkpoint-suffix macro-recourse-example
```

该一小时命令是后续完整训练示例，不是上述已执行的 12-step smoke。推理使用相同 method/learner/state/suffix，增加 `--evaluate-only --load-checkpoint --checkpoint-selection latest`；跨日期时同时指定 `--load-checkpoint-start-date/--load-checkpoint-end-date` 和训练时间窗口参数。Synthetic 对应入口为 `run_trainer.py`。

这次已扩展、测试的是两个 training CLI 及独立 audit runner。旧的 `test_model.py` / `test_nyc_model.py` 多策略脚本没有完整的 named-recourse 参数接口，**不能直接把它们当成所有新增模式的评估入口**；使用上述 NYC evaluate-only 或 audit runner。

另外实跑了 NYC CLI 的 `samitha` 4 车、3-step 启动/保存/`--evaluate-only --load-checkpoint` 流程，退出码 0，manifest 正确记录 `integrated_repair / macro_realized / structured`。该极短 CLI 测试未达到训练 warmup，不作为真实梯度更新证据；真实参数更新由七模式 audit runner 和确定性梯度测试验证。

`integrated_directq` 保留为 Integrated / Integrated-repair 的 full-Q comparator；EV-first 主 CLI 使用 residual learner。底层 macro 的 full-Q 与 residual target 分支均有确定性测试，但这次七模式训练 smoke 统一使用 residual，不宣称执行了 full-Q 大规模消融。

## 5. 未扩大的范围

- 未实现审计 §10 的 `integrated_full_replan`：没有取消、改派或重走已承诺服务，也没有引入零成本自由 replan。
- 未实现可选的 learned Samitha second-stage Q2；当前明确为 structured limited-hold repair。
- 未修改上一轮文档指出的 demand/queue `p0` 训练总开关缺口；本次不宣称完成该独立消融。
- 未进行跨日、多 seed、长时间训练收敛和论文级显著性评估；不能从本次 smoke 的收益排序得出算法优劣结论。
- 所有改动和结果只保存在本地；没有创建 git commit 或发布到远端。
