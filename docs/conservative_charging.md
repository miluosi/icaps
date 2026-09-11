# 500 车保守充电约束：实现与本地验证

## 默认模式更新（2026-09-11）

NYC 的 `conservative_charging=False` 现只检查待分配车辆到达时刻的物理空位。
背景日历包含正在充电、已到站排队和已预约但尚未到达的确定车辆；按照预计开始/完成时间计算占用，完成时刻等于候选车到达时刻的车辆不再占位。
不会把其他尚未选择的候选车辆加入背景，也不会因候选车到达之后才开始的预约而拒绝该边。
Assignment 按站点、到达 epoch 限制本批次同时到达的选中车辆数；不同到达 epoch 的新车不预占彼此的后续充电区间。
因此这只是到达时刻的准入约束，不能保证后续不排队或不影响较晚预约。
Conservative 模式仍检查完整区间并加入所有潜在车辆的虚拟预约。

**下文两个已保存的 500 车数据集使用更新前的默认完整区间检查及冲突修复。数据保留不变，其零冲突结论不代表当前默认模式；比较新默认模式需要重新运行 benchmark。**

本次将“所有潜在到达车辆”纳入充电可行性计算，并完成 500 辆总车、3／5 个站、每站 50 个充电位、4 个场景、每场景 10 个随机种子的配对测试。求解使用单线程 OR-Tools。

**结论：按确定的预约日历执行时，保守规则可以避免本次新分配引起的容量冲突，但空置明显增加。完整 NYC 仿真还存在到达与充电释放的时序边界问题，不能据此宣称完整仿真完全无排队。** 当前方法在这些日历测试中也没有新增排队。

## 代码与模型定义

- `src/conservative_charging.py`：独立的保守可行性计算。
- `src/NYCEnvironment.py`：在 `generate_vehicle_chargerange` 中接入，可通过 `NYCEnvironment(..., conservative_charging=True)` 开启；原有训练默认值保持 `False`。
- `benchmark_conservative_charging.py`：生成配对案例、调用真实 NYC charge/wait 生成器与现有 OR-Tools assignment 接口、验证容量及 SSG、保存结果。
- `plot_conservative_charging.ipynb`：读取本地结果、绘制均值和标准差以及分站占用曲线；输出 PNG 与 PDF。

每个站分别处理三类车辆：正在充电、已经确定会到达、当前决策中可能去该站的车辆。前两类保留原有充电位预约区间；第三类按到达时间、车辆 ID 排序，在不挤占已有预约的条件下计算虚拟开始及完成时间。只有虚拟开始时间等于到达时间、且原充电边可行的候选边才能保留。已经被拒绝的潜在车辆仍占用虚拟日历，符合“假设所有潜在车辆都去这个站”的保守前提。这里不会写入真实站点队列。

区间采用 `[开始, 完成)`：完成时间等于下一辆车到达时间时可以复用充电位。保留下来的区间是一个容量可行虚拟日历的子集；实际 assignment 再从中选择，每辆车至多选择一个动作。因此，在预约时间与实际执行一致、背景预约本身可行的前提下，本次新选择不会增加排队或推迟已有预约。

这定义了一个**新的原问题**。SSG 保留的是该新问题的最优值，不能把新旧模型的奖励差、充电数量差称为 SSG 的最优性损失。原 NYC wait 规则还会在车辆失去所有可行 charge 边时允许 wait，因此使用原 wait 逻辑时，不应简单地把整个新动作集合都称为旧集合的子集；严格的子集关系是 charge 边。

## 实验范围与复现

这里使用可复现的合成车辆状态与行程时间，调用真实 NYC charge/wait 生成器、现有 OR-Tools adapter 和 `ChargingStation` 队列 API。它是单次决策的控制实验，**不是完整 NYC.step 多时段 rollout，也不是训练或真实 TLC 需求实验**。Assignment 分数是合成分数，不能解释为实际美元收益。

500 辆是总车数，包含背景车辆：

| 场景 | 站数 | 正在充电 | 确定到达 | 待决策车辆 |
|---|---:|---:|---:|---:|
| synchronized / staggered | 3 或 5 | 0 | 0 | 500 |
| committed | 3 | 90 | 60 | 350 |
| committed | 5 | 150 | 100 | 250 |
| mixed_demand | 3 | 30 | 30 | 440 |
| mixed_demand | 5 | 50 | 50 | 400 |

运行原 NYC wait 规则的比较（全部规模、场景、种子均有默认值）：

```bash
.venv/bin/python benchmark_conservative_charging.py --wait-policy nyc
```

另有配对控制组，两种方法均允许 wait，用于单独考察 charge 边变化：

```bash
.venv/bin/python benchmark_conservative_charging.py --wait-policy available
```

每次运行自动建立独立时间戳目录。Notebook 默认选择最新完整的 `available` 运行；将首个代码单元的 `WAIT_POLICY` 改为 `"nyc"` 可读取原 wait 规则的结果，也可显式指定 `RUN_DIR`。

两组最终运行分别为：

- 原 NYC wait：`results/conservative_charging/20260911_014609_440411/`
- 两边均可 wait：`results/conservative_charging/20260911_014507_643705/`

每组保存 160 行方法结果、80 对案例：`raw.csv`、逐条检查点 `raw.jsonl`、`summary.csv`、`paired.csv`、`station_metrics.csv`、`metadata.json` 和 `report.md`。逐案例保存输入车辆、行程、分数、两种 mask、SSG 开关两套 assignment、分站占用、虚拟预约及诊断。Metadata 记录 Python / NumPy / OR-Tools 版本、参数与运行代码 SHA-256；最终数据的代码校验值已复核一致。

## 原 NYC wait 规则的结果

以下均为 10 个种子的平均值。“空置率”是在两种方法共享的观察区间内，空闲充电位时间占所有可用充电位时间的比例。“增加”是保守方法减去当前方法，单位为百分点。观察区间包含正常行驶等待，所以不能把保守方法的全部空置都归因于保守规则。

| 场景 | 站数 | 当前安排充电 | 保守安排充电 | 当前空置率 | 保守空置率 | 空置率增加 |
|---|---:|---:|---:|---:|---:|---:|
| 同步到达 | 3 | 150.00 | 50.00 | 16.67% | 72.22% | 55.56 |
| 同步到达 | 5 | 250.00 | 50.00 | 16.67% | 83.33% | 66.67 |
| 错峰到达 | 3 | 500.00 | 143.00 | 51.86% | 86.17% | 34.31 |
| 错峰到达 | 5 | 500.00 | 210.90 | 70.91% | 87.69% | 16.79 |
| 已有充电＋确定到达 | 3 | 350.00 | 35.70 | 37.83% | 68.59% | 30.75 |
| 已有充电＋确定到达 | 5 | 250.00 | 53.50 | 57.70% | 69.43% | 11.73 |
| 混合充电意愿 | 3 | 141.50 | 35.70 | 74.42% | 85.73% | 11.31 |
| 混合充电意愿 | 5 | 132.80 | 48.90 | 81.07% | 86.46% | 5.39 |

同步到达时，保守方案在所有站保留同一批优先级最高的 50 辆车，而每辆车实际只能去一个站。因此，到达后的充电阶段，3 站合计 **100 个充电位空闲（66.67%）**，5 站合计 **200 个充电位空闲（80.00%）**；当前方法此时全部占满。该场景是相同到达时间按车辆 ID 排序造成的强保守情况，不代表所有实际状态。但错峰到达的实验也出现显著空置，说明问题并不只来自同时间的排序。

两种方法在这两组全部测试中：新增排队车辆数、超容量充电位时间、已有预约额外延误、SSG 开关的整数目标值差均为 **0**。保守方案无需现有 adapter 的时间区间冲突修复。当前比较方法包含原有的时间冲突删边修复，其最终图由 OR-Tools 精确求解；它不是原始时间区间问题的全局最优 oracle。

在本机原 NYC wait 组中，保守方案生成 charge 可行性的场景平均时间约 **9.41–17.26 ms**；生成 charge 加 assignment 的场景平均时间约 **10.59–19.12 ms**。这不包含案例生成、文件保存、额外验证或整个仿真 step。500 车、3／5 站、50 位的保守计算成本可控，主要代价是充电位利用率。

## 尚未满足的完整仿真无排队条件

当前 `NYCEnvironment.step` 先执行动作和车辆到达，再在 `_update_environment` 中递减充电剩余时间并释放充电位。如果旧车剩余充电时间为 1、候选车行程为 1，日历认为下一时刻可用，但到达执行可能早于旧车的释放，从而仍进入队列。该问题在当前方法与保守方法都存在，仅增加可行性 mask 不能修复执行时序。

验证命令：

```bash
.venv/bin/python -m pytest -q -rx \
  tests/test_conservative_charging.py \
  tests/test_conservative_charging_benchmark.py \
  tests/test_expected_charging_feasibility.py
```

结果为 **21 个通过，2 个已知时序边界问题的预期失败（strict xfail）**。两项预期失败分别覆盖保守开关关闭与开启；它们不是“无排队验证通过”。论文可陈述确定性预约模型中的容量安全性，现阶段不能陈述完整 NYC 仿真绝不堵塞。

## 训练与测试接口

`run_nyctrainer.py` 与 `test_nyc_model.py` 已接通该开关。训练保守模型会自动在 checkpoint 命名空间中加入 `charge-conservative`，与原模型区分。以下命令可叠加原有的日期、数据路径、训练次数等参数：

```bash
# 训练保守充电模型
.venv/bin/python run_nyctrainer.py --methods r1 r2 r3 macro --conservative-charging

# 选择保守训练的 checkpoint；测试默认继承保存的充电模型
.venv/bin/python test_nyc_model.py --methods r1 r2 r3 macro --checkpoint-conservative-charging

# 同一组保守模型权重，显式改用原充电模型测试
.venv/bin/python test_nyc_model.py --methods r1 r2 r3 macro --checkpoint-conservative-charging --no-conservative-charging

# 原模型权重，显式在保守充电模型中测试
.venv/bin/python test_nyc_model.py --methods r1 r2 r3 macro --conservative-charging
```

`--checkpoint-conservative-charging` 选择**训练模型的 checkpoint 目录**；`--conservative-charging` / `--no-conservative-charging` 选择**测试环境的充电规则**。旧 checkpoint 缺失该字段时按原模型解释，EV／AEV 配对 checkpoint 的设置不一致时会报错。实际测试模式、checkpoint 模式及设置来源均写入结果；显式对照实验的输出文件名也区分训练／测试设置。

另一套整日实验入口也支持独立设置：

```bash
.venv/bin/python test_all_nyc_models.py train-test --conservative-charging --no-test-conservative-charging
.venv/bin/python test_all_nyc_models.py test-only --source-dir EXISTING_RUN_DIR --conservative-charging
```

`train-test` 中省略 `--test-conservative-charging` 时，测试沿用训练模式；`test-only` 省略覆盖开关时，沿用源 checkpoint 的训练模式。该整日入口原有的源代码 SHA 一致性检查保留，因此不同代码修订生成的历史 manifest 仍可能被拒绝。

接口验证共 **108 项通过**：

```bash
.venv/bin/python -m pytest -q tests/test_conservative_charging_cli.py tests/test_nyc_method_cli.py tests/test_all_nyc_models_runner.py tests/test_recourse_day.py
```

## MDP 命题与论文修改

“只要计入三类车辆，原有 MDP 必然无排队”的无条件命题不成立。有条件命题已经通过数学审阅：完整预约状态与实际物理状态一致，确定性到达／服务区间在跨期转移中保持一致，所有相关到达受该准入规则约束，同一时刻先释放完成的充电再处理到达，且初始状态无队列，则无排队集合是 MDP 转移核的不变集合。该结论适用于过程持续定义、在可达状态选择合法动作的策略，不保证合法动作集合永远非空。

完整符号、逐充电位排程递推、容量界及转移核证明作为旧版推导保存在 `docs/paper_updates/conservative_20260911/conservative_proposition.tex`。该长版本已按用户提供的修改建议从论文中替换。

当前 Overleaf 项目 `6a834a5015344234d20053d9` 使用同目录的 `compact_main.tex` 和 `compact_appendix.tex`：正文放简短的 Proposition 2 和一个准入 mask 公式；附录 1.3 仅保留日历构造说明、子集证明及简短跨期归纳。Reactive 作为附录中的运行比较说明，不以“只看当前空位”的约束替代预约安全性。预约配额记号统一为 `\bar K`，旧公式交叉引用已恢复。上述编辑没有修改 NYC 执行代码，也没有修复原有的到达／释放时序。

简化版已分别编译正文与附录：0 errors、0 warnings。正文 PDF 仍为 7 页；正文第 3 页的 Proposition 2 与附录第 2 页的短证明已视觉检查，公式 (15)、Proposition 2 和 Appendix 1.3 的引用均正确。
