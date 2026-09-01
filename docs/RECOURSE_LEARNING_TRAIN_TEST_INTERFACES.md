# Recourse learning 训练、测试接口与命令行列表

本文记录当前代码的实际入口。公开训练/测试接口只保留两个 learner，但保留九个 assignment/recourse 方法以及 Samitha。

## 1. 两个维度不要混淆

`--methods`/`--models` 选择物理分配、拒单后补救和 credit 结构；`--learner-variant` 选择同一结构使用哪一种 Q 学习器。

公开 learner 只有：

| CLI 值 | 含义 | 实现类 |
| --- | --- | --- |
| `optimization_anchored_residual` | MASAC residual；在结构化分配值上学习 correction | `src/ValueFunction_optimization_anchored_residual.py` |
| `integrated_directq` | full-Q；直接学习联合分配的完整 Q 值 | `src/ValueFunction_integrated_directq.py` |

部署分数分别为：

$$
\Psi_{\mathrm{residual}}(e,S)=G(e,S)+\Delta_\theta(e,S),
\qquad
\Psi_{\mathrm{fullQ}}(e,S)=Q_\theta(e,S).
$$

两个类复用 `src/ValueFunction_st_masac_gat.py` 中的图编码、双 critic、joint replay、exact projection 和 `_train_joint_step()`。这些基础文件是保留 learner 的实现依赖，不再作为独立 CLI 学习方法。

## 2. 九个 recourse/assignment 方法

注册表在 `src/recourse/config.py` 的 `PAPER_METHODS` 和 `METHODS`。统一训练函数在 `test_all_nyc_models.py` 的 `TRAINING_FUNCTIONS`。

| 方法名 | 简写 | 训练函数 | 物理 motion | 说明 |
| --- | --- | --- | --- | --- |
| `no_repair` | Integrated | `train_integrated()` | `simulate_motion_integrated_control` | Integrated/no repair |
| `evfirst_no_rejection` | R0 | `train_r0()` | `simulate_motion_evfirst` | EV-first，禁止拒单 |
| `evfirst_no_repair` | R1 | `train_r1()` | `simulate_motion_evfirst` | EV-first，可拒单但无同轮 repair |
| `evfirst_no_repair_structured` | C0 | `train_structured_r1()` | `simulate_motion_evfirst` | structured no-repair control |
| `repair_only` | R2 | `train_r2()` | `simulate_motion_evfirst` | 有物理 repair，follower 不更新 |
| `repair_learning` | R3 | `train_r3()` | `simulate_motion_evfirst` | repair learner，leader credit 仍 uncoupled |
| `recourse_macro` | Macro | `train_macro()` | `simulate_motion_evfirst` | recourse-aware，leader 使用 realized macro system return |
| `recourse_nested_q2` | R4 | `train_r4()` | `simulate_motion_evfirst` | leader 使用 nested follower target |
| `samitha` | Samitha | `train_samitha()` | `simulate_motion_integrated_repair` | Integrated stage-0 加 limited-hold repair |

常用别名：`integrated -> no_repair`、`recourse-aware`/`recourse_aware`/`macro -> recourse_macro`、`R1 -> evfirst_no_repair`、`R2 -> repair_only`、`R3 -> repair_learning`、`R4 -> recourse_nested_q2`。

## 3. 核心训练和测试函数

### 3.1 统一九方法入口

`test_all_nyc_models.py` 是当前“列表选择方法”的统一入口；`run_all_nyc_assignment_methods.py` 是同一入口的生产名称包装。

- `parse_args()`：解析 `train-only`、`train-test`、`test-only`、`report`、`list`。
- `method_selection()`：规范化 `all`、R1--R4、Integrated、Macro 和 Samitha 列表。
- `train_integrated()`、`train_r0()`、`train_r1()`、`train_structured_r1()`、`train_r2()`、`train_r3()`、`train_macro()`、`train_r4()`、`train_samitha()`：九个显式训练函数。
- `run_training_worker()`：构造环境和双 critic，训练 rollout，验证权重发生变化，保存 `checkpoint.pt` 与 `training.json`。
- `prepare_test_only()`：验证数据 hash、source hash、method/state/learner/solver metadata 和 paired-critic schema；任何 checkpoint 缺失都会终止，不回退到训练。
- `launch()`：调用 `run_recourse_day.py` 完成 train-test 或 test-only。
- `write_report()`：生成 `summary.json`、`metrics.json`、`metrics.csv` 和 `REPORT.md`。

测试阶段固定 `evaluatemode=True`，加载 checkpoint 后不调用训练，并检查测试前后完整权重 hash 相同。

### 3.2 公共 recourse 引擎

`run_recourse_audit.py` 提供被其他 runner 复用的核心函数：

- `build_env()`：创建 NYC 或 synthetic 环境，应用方法的 operating mode、recourse variant、CRN、state variant 和 learner variant。
- `build_pair()`：按 learner 注册表构造 AEV/EV 两个 critic，绑定共享的 joint replay/critic 关系。
- `rollout()`：调用对应 motion，执行 `env.step()`，每 `train_every` 步调用两个 learner 的 `train_step()`，并输出 reward、拒单、recourse、完成订单、充电、runtime 和 target diagnostics。

`src/ValueFunction_st_masac_gat.py` 的 `train_step()` 将保留的两个 learner 路由到 `_train_joint_step()`。对 joint transition 的 full Bellman target 为：

$$
y_t^{\mathrm{full}}=r_t+\gamma^{\Delta t}V_{\bar\theta}(S_{t+1}).
$$

full-Q 直接拟合 $y_t^{\mathrm{full}}$；residual 拟合扣除当前 structured value 后的 correction target：

$$
y_t^{\mathrm{residual}}=y_t^{\mathrm{full}}-G(S_t,A_t).
$$

Macro leader 使用当轮系统实际收益：

$$
y_t^{\mathrm{Macro}}=r_t^{\mathrm{EV}}+r_t^{\mathrm{AEV}}+gamma^{\Delta t}V_{\bar\theta}(S_{t+1}).
$$

R4 使用 follower 的同轮 nested value；EV 拒单是已观察结果，不是 terminal mask。

### 3.3 NYC 低层接口

- `run_nyctrainer.py::main()`：单一 NYC 配置的命令行训练入口，可用 `--recourse-method` 选择一个方法。
- `run_nyctrainer.py::run_nyc_training()`：Python 调用包装。
- `src/NYCtrainer.py::NYCTrainer.run_nyc_training()`：实际 NYC episode、训练频率、checkpoint、loss 和结果输出循环。
- `test_nyc_model.py::main()`：旧 solver/checkpoint inference 对比入口，只比较 ILP/MCMF/Auction/heuristic 等策略；它不是九方法 recourse runner。其 checkpoint learner 也只允许 residual/full-Q。
- `test_model.py::main()`：对应 synthetic solver/checkpoint evaluator，同样只允许两个 learner。

## 4. `run_recourse_*` 文件的作用

| 文件 | 是否训练 | 是否测试 | 作用 |
| --- | --- | --- | --- |
| `run_recourse_audit.py` | 是 | 是 | 最小公共引擎及 paired CRN/checkpoint inference audit；可跑 NYC 或 synthetic |
| `run_recourse_day.py` | 训练一天 | held-out 测试一天 | canonical production worker；每个方法独立进程，训练后从磁盘加载 checkpoint 测试 |
| `run_recourse_panel.py` | 是 | 是 | 多 seed、多 train/test day cluster 编排；每个 cluster 调用一次 `run_recourse_day.py` |
| `run_recourse_multiday_panel.py` | 多个训练日连续拟合一次 | 多个 held-out 日分别重载 | 正式 train-once/evaluate-many 主入口，统计单位是 `(seed, train_window)` |
| `run_recourse_sensitivity.py` | 视 protocol | 是 | rejection logit、AEV share、需求、充电容量、SOC、耗电和充电时长敏感性；复用 multiday runner |
| `run_recourse_spatiotemporal_analysis.py` | 否 | 否 | 读取已有 panel summary，汇总 hour/TLC zone/repair/hold/车辆/充电 CSV 与 JSON，不重新仿真 |

相关实验入口：

- `run_assignment_learner_experiment.py`：在相同 Macro 物理方法下比较 full-Q 与 residual。
- `run_assignment_state_experiment.py`：固定 learner，比较 state-information variants。
- `run_assignment_state_audit.py`：读取 checkpoint/replay，检查 pre/residual/stage graph 信息泄漏；不训练。
- `run_assignment_solver_audit.py`：读取一个固定 checkpoint，比较 solver/backend；从 checkpoint metadata 自动继承 learner，不训练。
- `run_assignment_scalability_experiment.py`：固定方法，比较 fleet size、backend、graph reduction、延迟和内存。
- `run_samitha_hold_ablation.py`：Integrated 0 hold、固定 hold、learned hold、Macro limit 的物理消融，可选两个 learner。
- `run_acceptance_ablation.py`：EV acceptance probability feature 的 paired on/off 学习消融，learner 列表也只有两个。

## 5. 列表式命令行训练和测试

所有命令从仓库根目录执行。

### 5.1 查看全部方法和两个 learner

```bash
python test_all_nyc_models.py list
```

也可使用：

```bash
python run_all_nyc_assignment_methods.py list
```

### 5.2 residual：九方法训练一天并测试一天

```bash
python test_all_nyc_models.py train-test \
  --models all \
  --learner-variant optimization_anchored_residual \
  --train-date 2025-12-18 \
  --test-date 2025-12-19 \
  --num-vehicles 200 --num-ev 100 \
  --workers 2 \
  --output-dir results/nyc_all_models/residual-all
```

### 5.3 full-Q：用列表选择 recourse-aware、R3、R4 和 Samitha

```bash
python test_all_nyc_models.py train-test \
  --models recourse-aware repair_learning R4 samitha \
  --learner-variant integrated_directq \
  --train-date 2025-12-18 \
  --test-date 2025-12-19 \
  --num-vehicles 200 --num-ev 100 \
  --output-dir results/nyc_all_models/fullq-selected
```

内部保存的规范方法名分别是 `recourse_macro`、`repair_learning`、`recourse_nested_q2`、`samitha`。

### 5.4 只训练，再从同一 checkpoint 只测试

```bash
python test_all_nyc_models.py train-only \
  --models no_repair R1 R2 R3 recourse_macro R4 samitha \
  --learner-variant optimization_anchored_residual \
  --output-dir results/nyc_all_models/train-residual

python test_all_nyc_models.py test-only \
  --source-dir results/nyc_all_models/train-residual \
  --models recourse_macro samitha \
  --output-dir results/nyc_all_models/test-residual
```

`test-only` 不接受另一个 learner 覆盖值；它从 source manifest 读取并验证训练时的 learner。

### 5.5 单方法低层 NYC 训练

recourse-aware residual：

```bash
python run_nyctrainer.py \
  --recourse-method recourse-aware \
  --learner-variant optimization_anchored_residual \
  --episodes 1 --num-vehicles 200 --num-ev 100 \
  --start-date 2025-12-18 --end-date 2025-12-18
```

Samitha full-Q：

```bash
python run_nyctrainer.py \
  --recourse-method samitha \
  --learner-variant integrated_directq \
  --episodes 1 --num-vehicles 200 --num-ev 100 \
  --start-date 2025-12-18 --end-date 2025-12-18
```

`run_nyctrainer.py` 的 `--distribution-mode` 仅作为兼容别名，选择范围仍是相同两个 learner。新命令统一使用 `--learner-variant`。

### 5.6 正式多日 recourse-aware 训练/测试

```bash
python run_recourse_multiday_panel.py \
  --methods recourse_macro repair_learning recourse_nested_q2 samitha \
  --learner-variant optimization_anchored_residual \
  --train-days 2025-12-15 2025-12-16 \
  --test-days 2025-12-18 2025-12-19 \
  --seeds 71 72 73 \
  --parquet-path nyedata/nye_simulation/parquet/yellow_tripdata_2025-12.parquet \
  --num-vehicles 200 --num-ev 100 \
  --energy-model general_charging \
  --output-dir results/recourse_multiday/residual
```

将 `--learner-variant` 改为 `integrated_directq` 即运行同一方法列表的 full-Q 版本。两个 learner 的 Macro 对照也可直接使用：

```bash
python run_assignment_learner_experiment.py \
  --learners integrated_directq optimization_anchored_residual \
  --train-days 2025-12-15 2025-12-16 \
  --test-days 2025-12-18 2025-12-19 \
  --seeds 71 72 73 \
  --parquet-path nyedata/nye_simulation/parquet/yellow_tripdata_2025-12.parquet \
  --energy-model general_charging
```

### 5.7 只检查列表和命令，不开始训练

```bash
python test_all_nyc_models.py train-test \
  --models recourse-aware repair_only repair_learning R4 samitha \
  --learner-variant integrated_directq \
  --dry-run
```

## 6. 结果与核对字段

统一入口输出以下核心字段：

- `recourse_number`：`same_epoch_aev_assignment_count`；
- `rejected_number`：`ev_rejected_offer_count`；
- `accomplished_number`：`completed_orders`；
- `reward`：所有车辆 `env.step()` 实际 reward 的累计；
- `aev_completions_after_rejection`：拒单后最终由 AEV 完成的订单；
- `ev_charging_sessions`、`aev_charging_sessions`：两类车辆充电次数。

短 `--smoke-steps` 或 `--dry-run` 只证明接口和计算路径可运行，不代表训练收敛或方法优于对照。

## 7. 本次本地核对结果

- `python -m compileall`：训练、测试、recourse runner、`src` 和 `tests` 全部通过；
- residual 的九方法 synthetic joint-training/checkpoint-inference smoke：9/9 通过；
- full-Q 的九方法 synthetic joint-training/checkpoint-inference smoke：9/9 通过；
- 每个保留 learner 的所有方法都有 joint optimizer update；C0/R2 按设计只冻结 AEV follower，EV critic 仍更新；
- 所有 18 个 smoke run 均通过 checkpoint inference identity 检查；
- 完整测试：453/453 通过；
- `test_all_nyc_models.py` 的 residual-all 与 full-Q selected-method dry-run 均通过。

上述 8-step synthetic smoke 用于验证训练/保存/重载/推理路径，不是 NYC 200 车正式数值实验。
