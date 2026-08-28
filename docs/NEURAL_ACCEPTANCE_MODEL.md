# EV 概率模型：当前 v3 与历史 v2

当前代码已升级为 `EVRejectionProbabilityModel`：默认三输入 `3→16→8→1`，
`rejected=1`，直接输出校准拒单概率，并以 `q_reject + human_response_mask` 接入 critic。
主 residual 同时使用拒单期望结构化基准；没有新增多阶段 ADP。

当前接口、逐条修改和测试说明见 [EV_REJECTION_V3_CHANGELOG_AND_TESTS.md](EV_REJECTION_V3_CHANGELOG_AND_TESTS.md)。
旧 v1/v2 模型和 replay 不能用于当前 v3。下文保留用于追溯的旧说明及原有本地结果，**不描述当前接口**。

## 历史 v2：神经网络 EV 接单概率模型

`BinaryAcceptanceModel` 已替换为 PyTorch MLP，不再有逻辑回归训练、推理或回退路径。
`Binary` 指二元标签，而不是只能输出 0/1。输出是连续的 `p_accept`，`p_reject = 1 - p_accept`。

## 架构与损失

30 个标准化输入 → Linear(30,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1)。
训练使用 logits 和 `BCEWithLogitsLoss`，推理才使用 sigmoid：

\(p=\sigma(f_\theta((x-\mu_{train})/s_{train})),\quad L=\mathrm{BCEWithLogitsLoss}(f_\theta(x),y)+\frac{\lambda}{2}\sum_l\|W_l\|_F^2.\)

- `y=1` 为实际接受、`y=0` 为实际拒绝；自然类别比例，不平衡抽样、不改标签。
- Adam，默认学习率 0.001、batch size 512、最多 120 epochs、验证早停 patience 20。
- 均值/标准差仅在训练集拟合；只用验证 BCE 选择 epoch 和 L2（0、1e-5、1e-4、1e-3）。
- 恢复验证最佳权重，而不是最后一次更新。记录每个 epoch 的训练 BCE、L2、总目标和验证 BCE。
- 初始化、训练打乱、checkpoint 加载和预测均不消耗环境的随机数流。冻结预测不更新网络参数。

## 分配前输入

统一定义在 `src/acceptance_inputs.py`。不使用司机/订单 ID 做数值输入，也不使用真实概率、
实际随机数、当前回答结果、事后电量或未来订单。

| 类别 | 输入 |
|---|---|
| 当前拒单公式的全部可观测直接因素 | 空闲时间、接客时间、加价金额 |
| 行程与价格 | 接客距离、乘客行程距离、行程时间、基础车费、报价、加价倍数 |
| 司机状态与时效 | 电量 SOC、订单已等待时间、剩余接客期限、剩余送达期限 |
| 时间和地理 | 时刻 sin/cos；司机位置、接客点、下客点各两个坐标 |
| 平台供需 | 可用 EV 数、可用 AEV 数、未分配订单数 |
| 接客/下客区域供需 | 各自的订单数、可用车辆数、需求/供给比 |

NYC 的时间特征为分钟、距离为 km、坐标为经纬度；synthetic 为模拟步数/网格距离/网格坐标。
两个 schema 不能混用。输入必须完整且有限，旧三特征数据不能用补零方式冒充新采集数据。

目前仿真响应公式直接依赖三个因素；其他输入是分配前上下文，并非已经证明存在额外因果效应。
增大网络和特征集不能消除 Bernoulli 响应的随机性，也不能保证准确率一定上升。

## 训练与推理一致性

独立采样、所有学习器的在线 EV 边特征和 Q/residual 回放均调用同一特征构造函数。
回放从不可变 `SystemSnapshot` 的车辆、订单、时刻构造供需，或直接读取当时保存的预测概率；
不使用当前 live 环境的供需。车辆快照补充保存 `penalty_timer`，以保持可用车辆统计一致。
下一状态的概率使用下一状态快照；Bellman reward、TD 目标和充电统计不变。

`--ev-acceptance-feature predicted --ev-acceptance-model PATH_TO_NEURAL_MODEL_JSON`
让所有已注册学习模式在启动时加载并冻结网络。未启用该特征时保持原来的学习输入。
Bayes 类的内部 `RejectionPredictor` 是另外一个已有 MLP；共享概率特征现在使用这里的 30 输入网络。

## Checkpoint 迁移

模型 JSON 使用 `version=2`、`model_type=mlp_binary_acceptance`、完整特征顺序、标准化参数和网络 state_dict。
旧 version=1 回归模型明确报错，必须重新采样和训练；旧 Q/residual checkpoint 中嵌入的回归模型
也不能作为新神经网络 checkpoint 加载。历史实验文件仍保留用于追溯，但不代表新网络结果。
独立消融现在要求显式传 `--acceptance-model`，不再默认寻找旧回归权重。

## NYC 本地检查

当前 NYC 默认 `reject_uniform=True`、`assignmentrange=2.0` km；2 km 指司机到接客点的范围，
不是乘客行程长度。其他拒单参数和充电规则不变。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python check_nyc_mcmf_acceptance.py \
  --num-vehicles 200 --num-ev 100 --workers 2 \
  --require-random-rejection --expected-assignment-range-km 2 \
  --train-seeds {8100..8139} --validation-seeds {8200..8211} --test-seeds {8300..8319} \
  --output-dir results/acceptance_checks/NEW_NEURAL_RUN
```

逐次验证实际响应是否为 `uniform < p_reject`，且接客距离符合 2 km。真实 uniform 仅存为诊断列。
测试种子仿真开始前冻结 L2、网络权重和验证 F1 阈值。测试对比神经网络、常数基线和仿真真实条件概率；
后者仅用于诊断。报告同时包含概率误差和拒单 TP/FP/FN/TN，不只看多数类准确率。
每个测试种子对比纯 MCMF 与旁路调用神经网络的完整轨迹，保留并核对 EV/AEV/平台充电次数。

完成后可独立复核（不重训，也不按测试标签调阈值）：

```bash
python verify_neural_acceptance_run.py results/acceptance_checks/NEW_NEURAL_RUN
```

输出 `neural_verification.json/md`，包含全部复核项和固定拒单概率区间的实测校准表。

这是概率预测验证，不是把模型加入分配打分后的性能消融，也不是跨日期或真实司机外部验证。

## 已完成的本地结果

2026-08-28：288 项测试通过；NYC 完成 92 轮完整仿真，训练/验证/测试分别 23,148 / 6,979 / 11,612 次邀约。
训练 BCE 从 0.174220 降至 0.121102，验证 BCE 从 0.176885 降至 0.129517。
留出 Log loss 为 0.118744（常数基线 0.166249），ROC-AUC 为 0.887955。
验证阈值 0.108071 下拒单召回 76.86%、精确率 16.36%；0.5 阈值下仍无报拒单，不能以总体准确率代替拒单识别。

- [模型与实验报告](../results/acceptance_checks/nyc-neural-2km-200-100ev-100aev-20260828/report.md)
- [验收记录、误报/漏报和充电统计](../results/acceptance_checks/nyc-neural-2km-200-100ev-100aev-20260828/verification.md)
- [新神经网络 checkpoint](../results/acceptance_checks/nyc-neural-2km-200-100ev-100aev-20260828/model.json)

DirectQ/residual 的 200 车短程训练、概率列梯度、重载冻结推理及四组独立轨迹重放均通过。
短程接口检查使用单独的冒烟模型，不是上述完整训练模型的平台性能消融。
