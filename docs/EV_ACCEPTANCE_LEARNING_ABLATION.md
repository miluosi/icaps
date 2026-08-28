# EV 接单概率输入与 Integrated 消融

> 概率预测器现已替换为 [30 输入神经网络](NEURAL_ACCEPTANCE_MODEL.md)。历史消融结果和 checkpoint
> 使用旧回归模型，仅供追溯，不能直接加载为新网络实验。需提供新神经网络 checkpoint 并重新训练 Q/residual。

## 接口与信息时序

所有当前 registry 学习模式均支持以下参数：

```text
--ev-acceptance-feature off|predicted
--ev-acceptance-model PATH_TO_MODEL_JSON
```

默认 `off` 保持旧输入维度；`predicted` 必须提供预训练神经网络 v2 JSON，启动时即加载并冻结。
EV 订单边的新增输入是连续的接单概率，不是 0/1 标签：

\[
x^{\rm new}_{v,r,t}=\bigl[x^{\rm old}_{v,r,t},\widehat p_{v,r,t}^{\rm accept}\bigr].
\]

当前输入包括原模型所有特征，以及 EV 的 `p_accept` 一个新增标量；AEV、充电、等待、再平衡边的该项为零。
预训练模型使用全部 30 个分配前输入，覆盖司机、订单、行程、价格、时段、地理和供需；
订单/车辆 ID 只用于查找，不进入预测器。回放的时刻和供需也来自历史快照。
Synthetic 与 NYC 的时间单位不同，接口检查模型 schema，不允许混用。

ST-GAT 全系列、`integrated_directq`、`optimization_anchored_residual` 通过共享局部边特征接入。
Bayes、time-only 和 none-network 模式通过路径网络最终状态层接入。
新增列权重初始为零，既有初始输出在浮点误差内一致，且不改变随机数状态，随后由正常 TD 损失学习。
这不是奖励乘以概率、手工拒单罚项或 oracle MCMF-K；`knownreject` 不需要开启。

## 训练、目标与推理一致性

- 在线分配：用当前、尚未回答的订单计算概率。
- 回放：从 `FeasibleEdgeSnapshot.acceptance_probability` 或不可变状态/订单快照读入，不读取当前活跃车辆/订单。
- 下一时刻目标：使用下一状态/候选动作自身的概率，不能复用当前订单概率。
- `RequestSnapshot` 同时保存显式 `surge_bonus`，避免被错误重算为 `final_value-value`。
- 检查点嵌入概率模型参数及 feature 模式；模式或模型不一致会拒绝加载。
- 冻结评估不写入 critic 经验；重新载入检查点后验证权重完全相同，评估后再次验证权重没有变化。

现有 Integrated Bellman 目标没有改变：DirectQ 使用 `R_t + gamma^Delta * Q_target(s_next,a_next)`；
residual 使用 `R_t + gamma^Delta * (g_next + residual_target) - g_t`。
因此当前 reward 仍只计算一次，新增概率是条件特征，不会将当前 reward 推迟到 t+1。

## 运行

通用训练入口（默认 synthetic 场景、Integrated、200=100 EV+100 AEV）：

```bash
python run_trainer.py --transportation-mode integrated --num-vehicles 200 --num-ev 100 \
  --learner-variant optimization_anchored_residual --episodes 6 \
  --mcmf-backend primal_dual --ev-acceptance-feature predicted \
  --ev-acceptance-model PATH_TO_SYNTHETIC_NEURAL_MODEL_JSON
```

`run_nyctrainer.py`、`test_model.py`、`test_nyc_model.py` 也接受同名参数。
两个测试 CLI 额外支持 `--checkpoint-suffix`，传入训练 CLI 输出的实验命名空间
（例如 `rec-legacy_state-joint_state_separate_critics_learner-optimization_anchored_residual_shift-0`）；
预测器哈希由接口自动补充，不要在这个参数中重复添加。独立消融检查点则使用
`run_acceptance_ablation.load_pair` 加载，它已在每组冻结评估之前实际验证。
NYC 要使用新训练的 NYC 神经网络 v2 checkpoint，不能使用 synthetic 模型或旧回归模型。
`integrated_directq` 与 `optimization_anchored_residual` 现在按各自名称路由目标与检查点；
仅选择 distribution mode 也不会再被 trainer 的 `legacy` 默认值覆盖为其他目标。

独立消融入口复用生产环境的 `simulate_motion`、`step`、critic router 和 `train_step`：

```bash
python run_acceptance_ablation.py --environment synthetic --num-vehicles 200 --num-ev 100 \
  --episodes 6 --train-seeds 41 42 43 --test-seeds 9001 9002 9003 9004 9005 \
  --train-every 25 --batch-size 1 \
  --acceptance-model PATH_TO_SYNTHETIC_NEURAL_MODEL_JSON \
  --output-dir results/acceptance_ablation/NEW_UNIQUE_DIRECTORY
```

两种 Integrated learner × 概率开/关 × 3 个独立训练种子。每轮完整 200 步（两个虚拟日），
每 25 步抽取 1 条完整联合转移更新，所以每个模型训练 48 次 joint TD，EV/AEV 均更新。
不要把 joint batch size 1 当成单条车辆边：一次联合转移包含整个车队和全部可行边。
本地对照使用 CPU 单线程，EV/AEV 分离 critic、共享联合回放，`neighbour_number=0`。
保持默认 residual warmup（500 updates）；48 次更新仍处于 warmup，不能宣称已收敛。

训练环境种子是 `50000 + training_seed*100 + episode_index`；测试种子与训练种子隔离。
开启专用需求随机流、offer-keyed 接单随机数；相同 seed 的位置/电量初始化一致。
逐轮核对订单生成序列 hash；不匹配则实验失败。加价由仿真当前供需决定，允许随策略变化。
独立训练模型可分进程运行，各自指定独立输出目录。

## 指标与保存

- `rejected_offers`：实际派给 EV 后被拒绝的邀约次数，可重复涉及同一订单。
- `unique_rejected_requests`：至少被拒一次的去重订单数。
- `ev_rejection_rate`：拒绝邀约次数 / EV 总邀约次数。
- `completed_orders`：时域内真正完单数，另列 EV/AEV 完单数；已接单、载客中不算完单。
- `completion_rate`：完单 / 生成订单，分母不包含重复拒单。
- 验证 `generated = completed + expired + active`，保留全部真实充电次数、完成充电时长等原统计。

输出包含 `manifest.json`、`episodes.jsonl`、`summary.json`、`report.md`，以及每个实验组的
`checkpoint.pt`、运行日志、逐邀约 JSONL、完整 episode stats。置信区间按独立训练种子聚类，
不把同一个模型的多次测试或单个订单当成独立训练重复。3 个训练种子的区间仅作描述性参考。
该区间反映固定测试种子集上的训练模型波动，不覆盖测试需求抽样的全部不确定性。

这项实验回答的是给定有限训练预算下增加输入是否影响拒单/完单，不能证明真实司机泛化，
也不能仅凭拒单数减少推断有效：还须同时查看 EV 总邀约、拒单率和平台完单数。

## 本次运行中一并修复

完整仿真曾因充电队列清理后，回放图仍使用清理前容量而中止。Integrated 现在在实际分配执行前
冻结经过清理的决策状态；未删除容量检查，也未修改充电次数定义。
旧 Bayes 接口补充接收规范化 replay 字段、保留下一服务动作 request ID，并避免在概率模式下
重复创建缺少分配前快照的拒单 TD 行；拒单监督标签与规范 ServiceAction 回放保留。
Bayes 的便捷批量接口同时保留调用方显式提供的接客距离、空闲时间及动作后状态特征。
NYC 分配入口现在同时识别 `joint_training_step` 与旧的 `training_step`；两个 Integrated
learner 及启用概率的模型从启动时就走网络，避免联合训练后冻结测试仍绕过已训练网络。

## 本地验证范围

单元/回归测试覆盖全部 11 个 registry 条目的概率输入和检查点配置一致性；覆盖分配前快照、
下一状态独立概率、AEV/非服务动作掩码、时间单位校验和概率列梯度。
此外实际运行 Bayes、time-only、queue-demand 的短程训练，及 NYC 200 车的
DirectQ/residual 开关概率短程训练和检查点重载推理。NYC 40 步测试仅验证接口，
不作为正式性能结论；正式对照使用完整 synthetic 时域。

保存的独立消融 checkpoint 可以直接重载复核，例如：

```bash
python verify_acceptance_checkpoints.py RESULTS_ROOT/seed-41 \
  --test-seed 9001 --output-dir NEW_VERIFICATION_DIRECTORY
```

该命令不再训练，逐项核对原测试的订单流、拒单、完单、奖励和充电统计，并断言权重不变。
正式多进程输出合并使用 `python summarize_acceptance_ablation.py RESULTS_ROOT`；
缺少任一组的最终结果、配对需求不同或计数不一致时不会生成正式汇总。

本次完整运行：
[`results/acceptance_ablation/integrated-200-100ev-100aev-20260828/report.md`](../results/acceptance_ablation/integrated-200-100ev-100aev-20260828/report.md)。
共 72 轮训练、60 轮冻结评估；全部原始日志、12 个双 critic 检查点及充电统计保存在同一目录树。
