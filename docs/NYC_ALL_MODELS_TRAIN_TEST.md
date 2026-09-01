# NYC 九方法：本地训练、测试与结果保存

统一入口：`test_all_nyc_models.py`。不带参数时，默认运行下列九种方法，各训练一天、在另一日期测试一天，200 车（100 EV + 100 AEV）。R1 保留，No repair 就是原 Integrated。

现在也支持 `--action train` 仅训练、`--action test` 仅测试、`--action train-test` 连续训练测试，以及 `--interactive` 键盘输入选择。`--mode` 是 `--action` 的等价参数，这里的 mode 指执行阶段，不是 integrated/evfirst 分配架构。

## “所有模型”的范围

这里按本轮约定指九种 canonical assignment/recourse 方法，不是旧脚本中所有 solver × 所有历史神经网络的笛卡尔积：

| CLI 名称 | 方法 | 运行模式 | Recourse variant |
| --- | --- | --- | --- |
| `no_repair` | Integrated / No repair | `integrated` | `legacy` |
| `evfirst_no_rejection` | R0，EV-first、禁止拒单 | `evfirst` | `r0` |
| `evfirst_no_repair` | R1，EV-first 无补救 | `evfirst` | `r1` |
| `evfirst_no_repair_structured` | C0，structured no-repair 因果基线 | `evfirst` | `r1_structured` |
| `repair_only` | R2，仅物理补救，AEV follower 不学习 | `evfirst` | `r2` |
| `repair_learning` | R3，补救学习，leader 无补救 credit | `evfirst` | `r3` |
| `recourse_macro` | Macro recourse-aware | `evfirst` | `recourse_macro` |
| `recourse_nested_q2` | Nested Q2 / R4 | `evfirst` | `r4` |
| `samitha` | Integrated + limited-hold repair | `integrated_repair` | `legacy` |

`--list-models` 同时输出实际训练函数和 motion 函数。当前映射为：

| 别名 | 规范方法 | 训练函数 | Motion 函数 |
| --- | --- | --- | --- |
| Integrated | `no_repair` | `train_integrated` | `NYCEnvironment.simulate_motion_integrated_control` |
| R0 | `evfirst_no_rejection` | `train_r0` | `NYCEnvironment.simulate_motion_evfirst` |
| R1 | `evfirst_no_repair` | `train_r1` | `NYCEnvironment.simulate_motion_evfirst` |
| C0 | `evfirst_no_repair_structured` | `train_structured_r1` | `NYCEnvironment.simulate_motion_evfirst` |
| R2 | `repair_only` | `train_r2` | `NYCEnvironment.simulate_motion_evfirst` |
| R3 | `repair_learning` | `train_r3` | `NYCEnvironment.simulate_motion_evfirst` |
| Macro | `recourse_macro` | `train_macro` | `NYCEnvironment.simulate_motion_evfirst` |
| R4 | `recourse_nested_q2` | `train_r4` | `NYCEnvironment.simulate_motion_evfirst` |
| Samitha | `samitha` | `train_samitha` | `NYCEnvironment.simulate_motion_integrated_repair` |

这些训练函数是显式、可测试的命名入口，最终都调用同一个经过验证的 `run_training_worker`，再由所选配置决定物理补救、credit/target 和 motion；不是复制九份训练算法。

九种方法默认使用 `optimization_anchored_residual` 和 `joint_state_separate_critics`，便于比较 repair/credit；也可统一指定 `--learner-variant integrated_directq` 运行 full-Q。公开入口只保留这两种 learner，历史 learner 不再出现在注册表或命令行选择中。

## 与 ADP/NYC train/test 的关系

参考 `src/ADPtrainer.py`、`src/NYCtrainer.py`、`run_nyctrainer.py`、`test_nyc_model.py` 的“环境 → 分配 → env.step → 按频率训练 → 保存/加载 → 独立评估”流程。

实际复用 `run_recourse_day.py` 和 `run_recourse_audit.py` 的实验引擎：同一个 `NYCEnvironment`、生产 value-function 类、joint replay/target/train_step 和同一套九方法配置。不是重写 reward、target 或 recourse 算法，也不是直接调用旧脚本的完整训练主循环。生产 NYCTrainer 的 motion 选择映射另有回归测试核对。

不同于旧 `test_nyc_model.py` 的 `ADP-MCMF-FT`，此入口没有 test-time fine-tuning：测试设置 `evaluatemode=True`，不调用 `train_step`，加载后的模型权重 hash 必须等于训练保存值，且测试结束不得改变。

使用独立的 paired-critic `checkpoint.pt` 格式；不自动扫描或猜测旧 `checkpoints/q_networks_*` 的 checkpoint，不会在找不到模型时临时训练。

## 默认设置

- NYC TLC Yellow Taxi：`nyedata/nye_simulation/parquet/yellow_tripdata_2025-12.parquet`。
- 训练 2025-12-18，测试 2025-12-19，各 00:00–24:00；30 秒/epoch，各 2880 步。训练和测试日期/seed 必须不同。
- 原始文件两日分别 166137 / 171960 行，均覆盖全部 24 小时；之后使用环境原有 Manhattan-only 清洗，不对保留需求下采样。旧 `*_sample.parquet` 只覆盖约两小时，不能用于全天测试。
- exact MCMF，`primal_dual` backend；2 km 接驾限制；`reject_uniform=True`；`knownreject=False`。
- rejection predictor 和预测拒单概率输入都关闭。不额外修改原有 post-demand / queue predictor 设置。
- `batch-size=2`，每 10 步训练，joint replay 容量 256，默认最多同时运行 2 个方法进程，每个进程 PyTorch 1 线程。
- 相同模型初始权重、相同训练需求流、相同测试需求流、offer-keyed common random numbers；测试加载训练 checkpoint。

数据来源：[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)。脚本不自动下载未知文件；数据缺失或任一日期缺少 24 小时覆盖会报错。

## 使用方法

从仓库根目录运行。下面 `python` 在本机可替换为 `/opt/anaconda3/bin/python`。

### 命令行选择训练、测试及方法列表

```bash
# 查看当前训练/测试支持的方法列表，不执行模型
python test_all_nyc_models.py --list-models

# 仅训练指定方法，保存 checkpoint 后结束
python test_all_nyc_models.py --action train \
  --train-models no_repair evfirst_no_rejection evfirst_no_repair evfirst_no_repair_structured repair_only repair_learning recourse_macro recourse_nested_q2 samitha \
  --learner-variant optimization_anchored_residual \
  --output-dir results/nyc_all_models/train-selected

# 只测试已有训练结果中的指定子集，不重新训练
python test_all_nyc_models.py --action test \
  --test-models no_repair recourse_macro samitha \
  --source-dir results/nyc_all_models/train-selected \
  --output-dir results/nyc_all_models/test-selected

# 同一列表连续训练与测试
python test_all_nyc_models.py --action train-test --train-models all \
  --learner-variant integrated_directq \
  --output-dir results/nyc_all_models/train-test-all

# 直接输入 R1--R4；内部结果仍保存规范方法名
python test_all_nyc_models.py --action train-test --r R1 R2 R3 R4 \
  --output-dir results/nyc_all_models/train-test-r1-r4

# 只检查选择与参数，不创建结果，不启动模型
python test_all_nyc_models.py --action train --train-models no_repair samitha --dry-run
```

等价子命令：`train` / `train-only`、`test` / `test-only`、`train-test`。原来的命令仍可使用。

代码中的 `TRAIN_MODELS` 和 `TEST_MODELS` 从当前九方法注册列表取得，并分别用于 `train.add_argument(... choices=...)` 和 `test.add_argument(... choices=...)`：

- 训练或训练后测试：`--train-models`，等价于 `--models` / `--methods`；默认 `all`。
- 仅测试：`--test-models`，等价于 `--models` / `--methods`；默认来源目录中全部方法，不要求目录必须包含九个模型。
- `--r` 是相同列表的短参数；支持 `R1`、`R2`、`R3`、`R4`（不区分大小写），分别映射到 `evfirst_no_repair`、`repair_only`、`repair_learning`、`recourse_nested_q2`。另支持 `integrated`、`macro`、`recourse-aware`、`samitha`。
- `all` 只能单独使用；方法名、编号冲突、重复方法和与执行阶段不匹配的参数会报错，不默默回退到默认训练。
- `train-test` 使用同一训练/测试列表。需要训练全部、测试其中一部分时，分两次使用上面的 `train` 和 `test` 命令。

仅训练仍保存训练日、预留测试日/seed 与数据版本，便于随后 `test-only` 复用。训练目录仍检查数据对这两个日期的覆盖，但只创建训练环境、执行训练 rollout，没有测试 rollout。仅训练的 checkpoint 与现有双 critic 格式兼容；原先保存的 checkpoint 也仍可用，不因增加命令行选项而更改学习算法源码。

### 键盘交互输入

```bash
python test_all_nyc_models.py --interactive

# 交互预览，不执行任何模型
python test_all_nyc_models.py --interactive --dry-run
```

依次选择：操作（1=仅训练、2=仅测试、3=训练后测试）、方法（例如 `1,5,7`、`R1 R2 R3 R4` 或规范方法名，回车=all）、仅测试所需的来源目录、结果目录。目录可包含空格。命令行已经提供的选项不重复询问。

真正执行前需要输入 `yes` 确认；直接回车取消，不创建或复制实验文件。只有显式 `--interactive` 才会询问，批处理和后台 worker 不会等待键盘输入。输入 `q` 可在操作选择时退出。

### 1. 九方法完整训练一天、测试一天

```bash
python test_all_nyc_models.py train-test \
  --num-vehicles 200 --num-ev 100 \
  --train-date 2025-12-18 --test-date 2025-12-19 \
  --workers 2 \
  --output-dir results/nyc_all_models/nyc200-full-day
```

输出目录必须是新目录，防止覆盖既有实验。不指定目录时自动生成带微秒时间戳的目录。

`--models all` 是默认；也可以选择子集，例如 `--models no_repair repair_only recourse_macro samitha`。`--methods` 是等价参数。用 `python test_all_nyc_models.py list` 查看全部九个预设。

### 2. 先检查参数，或执行短程运行检查

```bash
python test_all_nyc_models.py train-test --dry-run

python test_all_nyc_models.py train-test \
  --num-vehicles 200 --num-ev 100 \
  --smoke-steps 12 --train-every 2 --joint-replay-capacity 16 \
  --workers 1 --output-dir results/nyc_all_models/my-smoke
```

`--dry-run` 不创建结果、不训练。`--smoke-steps` 明确标记为短程测试，不能当作全天表现或收敛结论。

### 3. 只加载已有 checkpoint 测试

```bash
python test_all_nyc_models.py test-only \
  --source-dir results/nyc_all_models/nyc200-full-day \
  --output-dir results/nyc_all_models/nyc200-test-repeat \
  --workers 1
```

也支持 `results/recourse_day/nyc-200-100ev-100aev-24h-20260828` 格式；该旧全天批次已按用户要求停止，并未完成。使用前须确认所选方法有完整训练 checkpoint，不能把进度文件当作模型。

- 继承来源 manifest 的日期、车辆、seed、步长、训练设置与模型列表；可以用 `--models` 选择其子集。
- 开始前检查**所有所选方法**的 `checkpoint.pt` 和 `training.json`、checkpoint 身份/双 critic 格式、数据 SHA256、训练源代码 SHA256。不完整或不匹配直接报错，不回退到训练。
- 将训练阶段文件复制到新目录，再使用共同引擎的阶段恢复机制，仅执行测试。训练统计在报告中明确标为复用，不冒充本次训练。
- 不覆盖来源目录；不会复制旧 `results.json` 跳过新评估。只加载自己信任的本地 checkpoint。
- 这是相同 held-out 日期/seed 的重测，不是额外的独立随机重复。

### 4. 汇总既有结果，不启动任何训练/推理

```bash
python test_all_nyc_models.py report \
  --output-dir results/recourse_day/nyc-200-100ev-100aev-24h-20260828
```

可选 `--wait-seconds 600`：最多观察 10 分钟，每 30 秒刷新，全部完成则提前结束。不重启任何方法。未完成方法只列进度快照，不把部分 reward 当作最终结果。旧进度文件也不能单独证明进程仍存活。

### 5. 中断后恢复

对于**已经停止且没有同目录进程仍运行**的实验，用原命令原参数加 `--resume`。源码、数据和核心参数必须相同。

只支持阶段边界恢复：完成训练则从 checkpoint 开始测试；已完成测试则跳过；训练中途退出会重跑该方法训练日，不是 mid-epoch 恢复。不要对仍运行的目录再执行 train-test/resume；只用 `report` 查看。

## 本地输出

```text
output-dir/
  manifest.json            参数、原始数据日覆盖、数据/训练源码 SHA256
  summary.json             详细训练/测试结果、完成状态、指标定义
  metrics.json             每方法 × training/testing 的扁平核心指标
  metrics.csv              同一核心指标的 CSV 表，可直接用表格软件读取
  REPORT.md                可读表格、充电次数、验证结果与限制
  execution.json           train-only/test-only 执行阶段、训练来源/入口版本
  no_repair/               其余八种方法各有同样目录
    checkpoint.pt          两个 critic 及关联网络状态
    training.json          已完成训练阶段的统计和诊断
    training_result.json   仅 train-only：训练完成记录，明确不含 testing
    results.json           只有独立测试成功结束才写出
    progress.json          当前/最近阶段进度，不是最终结果
    run.log                仿真、训练日志
    worker.log             进程输出和异常
```

核心指标：

- `recourse_number` = `same_epoch_aev_assignment_count`：EV 本轮拒单后，当轮交给 AEV 的次数。
- `rejected_number` = `ev_rejected_offer_count`：EV 实际拒绝平台订单的次数。
- `accomplished_number` = `completed_orders`：真实完成订单数，不是分配/接受/pickup 数。
- `reward`：所有车辆实际 `env.step` reward 的累计（包含负值）；不是最后一个 batch 的 TD target 或 reward ledger 小窗口。
- `completed_number` 和 `ev_rejected_offers` 作为旧结果读取别名保留，分别等于 `accomplished_number` 和 `rejected_number`。
- `aev_completions_after_rejection`：拒单后由 AEV 完成，可能包括后续 epoch 的 rescue，不能与同轮补救数混淆。
- `ev_charging_sessions` / `aev_charging_sessions`：保留 EV/AEV 充电次数。

一日训练、一日测试、单 seed 只能检查运行和本次表现，不能证明收敛或统计显著优势。主要因果路径是 C0（structured no-repair）→R2→R3→Macro→R4；learned R1 是诊断对照，Integrated→Samitha 是另一条架构比较路径。

## 正式多日实验入口

`run_recourse_multiday_panel.py` 对每个 `(seed, train_window, method)` 只训练一次 checkpoint，再为每个 held-out 日期重新加载该不可变 checkpoint。正式主实验应显式指定 `--energy-model general_charging`；当前 `fixed_swap` 会直接报错，因为固定换电中心不是 general charging 的参数别名。

配套入口：

- `run_assignment_learner_experiment.py`：同一 Macro 架构下比较 DirectQ 与 optimization-anchored residual；
- `run_assignment_state_experiment.py` 与 `run_assignment_state_audit.py`：state performance paired summary 和 pre/residual/stage-graph 泄漏审计；
- `run_assignment_scalability_experiment.py`：100–3000 车、reduction/backend、图规模、延迟和内存；
- `run_recourse_sensitivity.py`：拒单 logit、AEV share、需求、站点容量、耗电、初始 SOC 和充电时长；
- `run_samitha_hold_ablation.py`：0%、固定比例、learned hold 与 EV-first limit；
- `run_recourse_spatiotemporal_analysis.py`：把 panel 中的逐小时、TLC zone、hold、车辆与充电统计整理为 CSV/JSON。

这些 runner 生成实验计划和基础设施；只有实际完成预先冻结的多 seed/date 运行后，才能声称论文数值结果已经完成。

## 本地验证

```bash
python -m pytest -q tests/test_all_nyc_models_runner.py tests/test_recourse_day.py
python -m pytest -q
```

新增测试覆盖：九方法映射、日期/seed 泄漏、参数校验、dry-run 无执行、缺失 checkpoint 不训练、错误 checkpoint/数据拒绝、仅测试阶段复制、来源不覆盖、进度不冒充最终结果、失败状态保留、负 reward / completed / recourse / 充电统计语义。

命令接口补充覆盖：train/test/action/mode 等价写法、训练/测试列表别名、交互编号/名称/取消、仅训练 worker 不创建测试环境、仅训练报告不声称测试通过、中断时清理本次子进程。命令接口修改只做参数和执行分支的本地测试，没有重新启动全天实验。
