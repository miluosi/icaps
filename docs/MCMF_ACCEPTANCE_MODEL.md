# 普通 MCMF 下的人类司机接单概率学习

> 当前实现已替换成 30→64→32→1 神经网络，详见 [神经网络说明](NEURAL_ACCEPTANCE_MODEL.md)。
> 下方 2026-08-27 的数值、系数和旧 checkpoint 链接仅保留为回归模型历史记录，不能当作新网络结果。

## 目的与范围

`train_acceptance_model.py` 在现有 integrated 仿真环境中执行普通 exact MCMF，收集人类司机实际收到的订单及其 binary 响应，训练独立的 `BinaryAcceptanceModel`。

这不是 ADP Q 网络训练，也不使用 MCMF-K 的已知接单概率给订单打分。环境的拒单规律、车辆运动、充电决策和充电次数统计均保持不变；预测模型暂不参与分配。

## 2026-08-27 本地验证结果

两套实验均为 200 辆车，其中 100 辆为人类司机；分别训练模型，不混合两种环境的时间单位。

| 项目 | 合成环境 | NYC 样本环境 |
|---|---:|---:|
| 训练 / 验证 / 测试报价数 | 16,202 / 4,001 / 8,005 | 9,665 / 2,900 / 4,800 |
| 训练 / 验证 / 测试种子数 | 60 / 15 / 30 | 20 / 6 / 10 |
| 测试实际接单率 | 83.0606% | 96.1458% |
| 预测接单率均值 | 82.5600% | 96.0005% |
| Log loss：模型 / 常数基线 | 0.39955 / 0.45500 | 0.09605 / 0.16333 |
| Brier score：模型 / 常数基线 | 0.12673 / 0.14072 | 0.03019 / 0.03706 |
| ROC-AUC | 0.73956 | 0.93774 |
| 相对仿真真实概率的 MAE | 0.2535 个百分点 | 0.0806 个百分点 |
| 相对仿真真实概率的 RMSE | 0.2969 个百分点 | 0.2644 个百分点 |
| Exact MCMF 调用次数 | 21,000 | 8,233 |

按整条测试种子轨迹进行 2,000 次 paired bootstrap：

- 合成：log loss 改进 95% 区间 `[0.04781, 0.06279]`，Brier 改进 `[0.01184, 0.01612]`。
- NYC：log loss 改进 95% 区间 `[0.05705, 0.07507]`，Brier 改进 `[0.00481, 0.00853]`。

两个区间均大于 0，支持模型优于仅预测训练集平均接单率的常数基线。这里的“概率 MAE”比较的是预测概率与仿真概率，不是与单次 0/1 响应比较。

分类准确率没有优于“始终预测接单”：合成模型 83.0356%，常数基线 83.0606%；NYC 两者均为 96.1458%。这正是不能用准确率单独判断概率预测器的原因。常数模型的总体 ECE 也可能很小，但它不区分不同报价条件。

合成模型恢复出截距 `1.79307`、空闲系数 `-0.01681`、接客系数 `-0.05120`、surge 系数 `0.10345`，接近仿真系数。NYC 的空闲时间均值仅 `0.680` 分钟、标准差 `0.762` 分钟，其学习到的空闲系数为 `+0.04291`，没有恢复出环境的负号；**当前分布上的良好概率预测不代表每个行为系数都已被可靠识别，也不能外推到长空闲等低覆盖场景**。

验证还包括：212 项项目测试通过；保存/加载模型概率逐项相同；真实仿真报价前的单条预测与保存快照的批量预测一致；两套环境各 40 步的对照实验中，安装采样器前后的车辆状态和奖励轨迹完全一致。

本地输出：

- [合成模型](../results/acceptance_model/synthetic-mcmf-20260827/model.json)、[合成完整报告](../results/acceptance_model/synthetic-mcmf-20260827/report.md)。
- [NYC 模型](../results/acceptance_model/nyc-mcmf-20260827/model.json)、[NYC 完整报告](../results/acceptance_model/nyc-mcmf-20260827/report.md)。

## 标签、特征与模型

- 标签：实际接单为 `accepted=1`，实际拒单为 `accepted=0`，不强制平衡类别。
- 特征在调用司机响应函数**之前**复制：全部 30 个司机/订单/行程/价格/时段/地理/供需输入，见神经网络说明。
- 合成环境遵循当前行为函数的原生单位：空闲步数、Manhattan 接客距离；NYC 使用空闲分钟和接客分钟。模型保存单位模式，并拒绝混用。
- 车辆/订单 ID、接单结果、随机抽样值、事后状态与仿真真实概率均不进入输入特征。
- 普通 MCMF 只决定报价对象；采样器调用原司机响应函数一次，不增加随机抽样或改变响应。

模型为

$$\widehat p(\mathrm{accept}\mid x)=\sigma\left(f_\theta\left(\frac{x-\mu_{\mathrm{train}}}{s_{\mathrm{train}}}\right)\right).$$

`f_theta` 为两层 ReLU MLP。Adam 最小化 BCEWithLogitsLoss + 权重 L2，参数从实际 binary 标签学习，
不复制环境行为系数。均值和尺度只从训练集计算。L2 候选 `0, 1e-5, 1e-4, 1e-3` 和早停 epoch
仅用验证 BCE 选择，测试集不参与拟合和选择。旧回归 checkpoint 不再支持加载。

## 复现

合成环境：200 辆车，其中 100 辆为人类司机；使用当前 predictive/充电默认配置。以下种子范围语法适用于 zsh/bash。

```bash
python train_acceptance_model.py --environment synthetic \
  --num-vehicles 200 --num-ev 100 \
  --train-seeds {100..159} --validation-seeds {200..214} --test-seeds {300..329}
```

NYC 环境：使用仓库自带的 2025-12-18 08:00–10:00 订单样本、30 秒 epoch，以及当前人类司机充电决策间隔。

```bash
python train_acceptance_model.py --environment nyc \
  --num-vehicles 200 --num-ev 100 \
  --train-seeds {100..119} --validation-seeds {200..205} --test-seeds {300..309}
```

默认使用不需要 Gurobi 许可证的 exact `primal_dual` 后端，并检查每次分配的最优状态和 solver fallback。可用 `--mcmf-backend` 显式切换精确后端。每次运行创建独立结果目录，不覆盖已有实验；`--output-dir` 可指定一个尚不存在的目录。

## 推理

```python
from src.acceptance_model import BinaryAcceptanceModel

model = BinaryAcceptanceModel.load("results/acceptance_model/<run>/model.json")
# 在司机回答前、同一环境单位模式下调用：
p_accept = model.predict_acceptance_probability(env, vehicle_id, request)
p_reject = model.predict_rejection_probability(env, vehicle_id, request)
```

AEV 返回接单概率 1。对人类司机，单条环境推理与批量快照推理使用同一个特征函数。模型文件为 JSON，不依赖 pickle；实验会验证保存前后概率逐项一致。

## 结果与判断标准

每个输出目录包括：

- `model.json`：学习到的参数、训练尺度、单位模式与标签定义。
- `train_offers.jsonl` / `validation_offers.jsonl` / `test_offers.jsonl`：实际报价和响应；`oracle_acceptance_probability` 明确仅用于评估。
- `test_predictions.jsonl`：留出报价的预测概率。
- `summary.json` / `report.md`：log loss、Brier score、ROC-AUC、校准分箱、概率 MAE/RMSE、训练集接单率常数基线，以及按整条种子轨迹重采样的 95% 改进区间。
- 每个种子的运行日志、精确 MCMF 调用计数、充电次数与相关指标。

重点看模型是否降低留出样本的 log loss/Brier score，以及预测概率是否接近仿真概率；高接单率下的分类准确率不能单独证明模型有用。

结论仅适用于当前 MCMF 实际报价的状态分布。NYC 的订单来自真实数据，但司机接单标签仍由仿真产生；留出的是司机随机响应和车辆初始化种子，同一天订单被复用，**不是跨日期或真实司机泛化验证**。没有得到报价的候选订单不在验证样本范围内。

## 与历史 rejection predictor 的关系

Bayes 类内部已有一个使用 `1=拒单` 的 RejectionPredictor MLP，独立于这里的共享接单模型。
所有已注册学习器现在可通过 `src/acceptance_features.py` 使用新神经网络输出的冻结概率。
本次独立训练不改变 MCMF 打分，也不把历史 Bayes 网络的结果当作新模型验证。
