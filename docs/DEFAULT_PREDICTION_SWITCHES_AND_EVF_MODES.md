# 默认预测开关与 EV-first 模式核查

后续更新：本文件保留上一轮预测开关核查记录。新增 macro recourse、R2/R3 命名、保留 R1/Integrated 及 Samitha 的实现与实测，请见 [Recourse 修订报告](RECOURSE_CREDIT_REPAIR_SAMITHA_AUDIT_20260828.md)；下文原 r3/r4 表不是当前全部模式清单。

核查日期：2026-08-28。范围：默认关闭拒单预测、检查 demand/queue 开关、说明现有 EVF 训练路径。本次没有新增多阶段 ADP，没有修改 demand/queue 的学习机制，也没有删除充电、接单或拒单统计。

## 1. 拒单预测现在默认关闭

共享 v3 接口原本已经默认 `--ev-response-feature off`、`--ev-response-model None`，见 `src/acceptance_features.py`。

默认情况下：

- 不读取模型文件，不创建冻结的 v3 概率模型。
- Q/residual 不增加 `q_reject` 和 human-response mask 输入列。
- residual 不启用拒单概率混合的 expected anchor。
- 默认 `--ev-response-critic-input q_mask` 只有在主开关为 `predicted` 时才生效，并不表示默认输入概率。
- 仿真的真实司机拒单机制不变；`ifreject`、`reject_uniform` 与是否使用预测器是不同开关。

核查发现的遗漏：NYC 入口不指定 learner/distribution 时解析到 `none`，但 `none` 在注册表中映射到 Bayes 值函数；只要 ADP 学习开启，该类以前仍会创建并在线训练旧的 `RejectionPredictor`。它与冻结的 v3 模型不是同一个模型。

本次修正：

| 文件 | 修改 |
| --- | --- |
| `src/ValueFunction_pytorch_bayes.py` | 新增构造参数 `enable_legacy_rejection_predictor=False`；默认不创建旧模型或其优化器；训练入口直接返回，不执行预测或旧风险罚项。保留结果缓冲区和统计。 |
| `src/Environment.py` | 旧模型不存在时不调用其训练函数；仍保留接单/拒单数据导出。 |
| `src/ADPtrainer.py` | 加载含旧预测器权重的检查点时，不重新启用已关闭的旧模型，也不对 `None` 加载权重。 |
| `tests/test_default_prediction_switches.py` | 新增 16 项回归测试，覆盖两个训练入口、全部注册 learner、禁用后的训练调用、真实拒单开关不变、结果导出和旧检查点。 |

旧预测器只保留显式 Python 构造参数的兼容入口，训练 CLI 不会自动开启它。显式使用 `--ev-response-feature predicted --ev-response-model <v3模型路径>` 仍可开启共享 v3 特征，不会连带开启旧模型。单独的 `train_acceptance_model.py` 和显式预测消融实验不属于默认关闭的 Q 训练入口。

## 2. demand / wait-queue：没有完整的“特征与学习一起关闭”开关

| 现有参数或模式 | 实际效果 | 能否作为完整关闭开关 |
| --- | --- | --- |
| `ADPTrainer` 的 `disable_queue_predictor=True` | 加载后把 `queue_predictor_trained` 置为 False；对应预测返回零 | 不能。继续训练会重新置为 True；没有禁止创建模型或更新参数。 |
| `ADPTrainer` 的 `disable_post_demand_predictor=True` | 同理，只清除 demand 的 trained 标记 | 不能。后续训练会重新启用。 |
| `test_model.py` 的 `ADP-ILP-QOFF` / `DOFF` / `QDOFF`，以及 `ADP-MCMF-QOFF` | 使用上述标记做不继续学习的评估消融 | 是评估侧消融，不是训练全程关闭。NYC trainer 没有接入这两个参数。 |
| NYC `--predictor-variant p0/p1/p2/p3`，默认 p3 | 只向 residual 类传入并保存字符串；目前没有按该值分支控制特征或训练 | 不能。`p0` 的帮助文本写着 no forecast，但实现没有兑现；不能据此声称完成 p0 消融。 |
| `--post-demand-q-weight 0` | 将 direct-demand critic 的动作系数初始化为零 | 不能。系数仍是可训练参数，demand 网络仍学习。 |
| `--post-demand-head-lr-multiplier` | 调整 demand 动作头学习率，代码下限为 1 | 不能通过设零冻结。 |
| `--no-charge-wait-bool` | 改变 NYC 充电/等待动作的可行性约束 | 与关闭预测器无关。 |
| synthetic `env.use_queue_forecast_action_filter=False` | 不使用 queue 预测过滤可行动作 | 不关闭 Q 特征和预测器学习。 |
| `--evaluate-only` / `--adp 0` | 停止整个值函数训练，或禁用 ADP | 不是仅关闭辅助预测器的对照。 |

证据位置：`src/ADPtrainer.py` 的加载后消融设置及 predictor pretraining；`src/ValueFunction_st_masac_gat.py` 的 `train_queue_predictor`、`train_step`；`src/ValueFunction_st_masac_gat_post_demand.py` 的 `train_post_demand_predictor`、`train_step`；`src/ValueFunction_optimization_anchored_residual.py` 的构造函数。

动态复核：使用相同随机种子、同样的 8 条辅助训练样本，分别构造 p0、p3 residual learner，执行 `train_step(batch_size=4, ifEV=False)`。两者的 queue 和 demand 参数均改变，trained 标记均变为 True；queue MSE 均为 3.7890286446，normalized demand MSE 均为 0.0359413140。再次把两种 trained 标记置 False 后训练，标记又变为 True。初始 demand 动作系数为零时，测试特征仍得到非零梯度 `[0.5, 0, 0]`。

这些数值只用于验证开关是否生效，不是预测质量评估。

NYC 可选择基础 `--distribution-mode st_masac_gat`，该类不创建 post-demand 网络，但仍有 queue predictor；这会改变 learner 配置，不等价于在同一个 residual 模型里关闭两个预测器。`none` 也不是只关闭 demand/queue 的开关，而是另一个值函数实现。

## 3. EVF 已有 r3 / r4，但“独立”需要区分含义

EVF 对应 `--transportation-mode evfirst`，不是另一个独立训练脚本。

| 配置 | 同轮 AEV 补救拒单 | EV 训练目标 | 含义 |
| --- | --- | --- | --- |
| `--recourse-variant r3` | 有，AEV 使用学习值函数 | `r_EV + gamma^elapsed * V_next_EV` | EV 目标不接入当轮 AEV follower 值；可作为不带当轮补救价值耦合的对照。 |
| `--recourse-variant r4` | 有，AEV 使用学习值函数 | `r_EV + within_epoch_gamma * V_AEV_residual` | Recourse-aware EVF；EV 目标显式计入当轮拒单结果之后的 AEV 补救值。 |
| `--recourse-variant r1` | 当轮不允许 AEV 接回被拒订单；未派订单仍可接 | 普通跨轮目标 | 如果“独立”指不做同轮拒单补救，应区分此模式与 r3。 |

默认 `recourse_variant=legacy`，不会自动选择 r4。r0—r4 只接受 evfirst。

重要限制：r3 **不是严格的两车队完全独立 Q-learning**。当前 r3/r4 都复用 joint transition；AEV follower 的跨轮目标在 `src/ValueFunction_st_masac_gat.py::_temporal_successor_graph` 中仍接到下一轮 EV leader graph。r3 只是没有 r4 的 EV→当轮 AEV 目标连接。不能把它描述为“EV 和 AEV 各自只 bootstrap 自己的下一状态”。

网络共享与目标耦合是不同维度：

- `--state-variant joint_state_separate_critics`：默认；联合状态，EV/AEV 两套 critic。
- `--state-variant fleet_local_separate_critics`：车队局部状态视图，两套 critic；不会自动改变上述 Bellman 连接。
- `--state-variant joint_state_shared_critic`：联合状态，共用 critic；不等于开启 recourse-aware。

代码位置：

- `run_nyctrainer.py`：NYC 训练 CLI，recourse/state/learner 参数。
- `run_trainer.py`：synthetic 的对应训练 CLI。
- `src/NYCtrainer.py`：选择 `simulate_motion_evfirst`；按训练频率分别执行 AEV `train_step(ifEV=False)` 和 EV `train_step(ifEV=True)`。
- `src/NYCEnvironment.py::simulate_motion_evfirst`：先派 EV，观察接单/拒单，再构造剩余订单状态并派 AEV。
- `src/recourse/target_builder.py`：r0—r4 策略与目标定义。
- `src/recourse/critics.py::wire_recourse_critics`：连接两车队 critic 和 R4 的 AEV target provider。
- `src/ValueFunction_st_masac_gat.py::_train_joint_step`：实际 R4 分支使用同轮 AEV 的 lagged target 值；r3 使用下一轮图。residual 目标再减去当前 structured anchor。
- `scripts/smoke_recourse_nyc.sh`、`scripts/smoke_recourse_toy.sh`：已有小车队 R4 冒烟训练脚本。

在 r4 中关闭拒单概率模型不会关闭 recourse-aware：它仍从真实发生的拒单结果和 AEV follower transition 学习。`integrated_directq` 专用于 integrated，不能直接作为 evfirst 的 learner 参数。

## 4. 本地 NYC 200 车训练命令

以下是本地已有 sample 数据的一小时训练示例，100 EV + 100 AEV。用于启动实验，不代表训练收敛；正式训练应扩大数据日期和训练量。本次只验证了参数解析和输入文件存在，**没有执行这两组 200 车训练**。

```bash
cd /Users/seinzhou/Desktop/icaps
for variant in r3 r4; do
  /opt/anaconda3/bin/python run_nyctrainer.py \
    --transportation-mode evfirst \
    --recourse-variant "$variant" \
    --learner-variant optimization_anchored_residual \
    --state-variant joint_state_separate_critics \
    --ev-response-feature off \
    --num-vehicles 200 --num-ev 100 --episodes 1 --adp 1 \
    --parquet-path /Users/seinzhou/Desktop/icaps/nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet \
    --station-csv /Users/seinzhou/Desktop/icaps/nyedata/nyc_all_charging_stations.csv \
    --start-date 2025-12-18 --end-date 2025-12-18 \
    --start-hour 8 --stop-hour 9 --epoch-length 30 \
    --assignment-gurobi --use-mcmf --mcmf-backend primal_dual \
    --batch-size 64 --training-frequency 10 --start-training-episode 0 \
    --checkpoint-suffix "evf-check-$variant"
done
```

`primal_dual` 是内置 exact MCMF 后端，此处不依赖 Gurobi 许可证。保留了现有 demand/queue 学习，因为目前没有可靠的训练禁用开关；不要加上 `p0` 后就将实验标为“无预测”。CLI 会把 recourse/state/learner 加入 checkpoint 后缀，区分 r3/r4。

## 5. 验证结果

- `python -m pytest tests/test_default_prediction_switches.py -q`：16 项通过。
- `python -m pytest tests -q`：324 项通过，包括原有拒单特征、充电统计和 recourse 回归测试。
- 修改文件的 `compileall` 和 `git diff --check` 通过。
- p0/p3 和清除 trained 标记的动态诊断证明现有预测开关不完整；本次没有修复这两个开关，也没有实现完全独立的 EV/AEV Bellman 模式。
