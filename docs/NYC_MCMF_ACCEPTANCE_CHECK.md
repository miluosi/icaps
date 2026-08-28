# NYC 200 车：独立概率模型与纯 MCMF 检查

> 当前版本：`reject_uniform=True`、2 km 接客范围、30 输入神经网络。
> 完整架构与输入见 [神经网络说明](NEURAL_ACCEPTANCE_MODEL.md)。文末旧结果来自 5 km 回归模型，
> 仅作历史记录，不是新网络的验证结果。

## 函数在哪里

`src/acceptance_model.py` 的 `BinaryAcceptanceModel.predict_proba(rows)` 返回连续接单概率数组。
`predict_acceptance_probability(env, vehicle_id, request)` 是单条分配前预测接口；
`predict_rejection_probability(...)` 返回 `1 - p_accept`。AEV 的接单概率为 1。
Q/residual 的 EV 输入适配在 `src/acceptance_features.py::predicted_acceptance`。

输入扩展为 30 个分配前的司机/订单/行程/价格/时段/地理/供需特征，包含
`idle_time`、`pickup_time`、`surge_bonus`；NYC 时间单位均为分钟。
均值、标准差只从训练集估计，模型为

\[
\widehat p_i=\sigma\left(f_\theta\left(\frac{x_i-\mu_{\mathrm{train}}}{s_{\mathrm{train}}}\right)\right),\qquad \widehat p_i^{\mathrm{reject}}=1-\widehat p_i.
\]

连续概率、按阈值生成的分类结果、仿真司机实际的随机回答是三件不同的事。
`_should_reject_request` 的布尔返回值是实际标签，不是预测器的概率输出。

## 单独训练的损失

`BinaryAcceptanceModel.fit(rows)` 最小化平均二元交叉熵加 L2 正则：

\[
L(\theta)=\mathrm{BCEWithLogitsLoss}(f_\theta(x),y)+\frac{\lambda}{2}\sum_l\lVert W_l\rVert_F^2.
\]

标签 `accepted=1`、`rejected=0`，保持自然类别比例，偏置不正则化。
网络为 30→64→32→1、两层 ReLU；使用 Adam 和验证早停，恢复最佳 epoch。
当前四个 L2 候选为 `0、1e-5、1e-4、1e-3`，仅按验证集 BCE 选择。

`fit` 提供 `loss_history`，记录初始状态及每个 epoch 的总目标、训练 BCE、L2 和验证 BCE。
Adam 的逐步损失不保证单调下降。模型使用新的 v2 checkpoint；旧回归权重/三特征数据必须重新训练/采集。

## 独立运行

```bash
python check_nyc_mcmf_acceptance.py \
  --num-vehicles 200 --num-ev 100 --workers 2 \
  --require-random-rejection --expected-assignment-range-km 2 \
  --output-dir results/acceptance_checks/NEW_UNIQUE_DIRECTORY
```

默认使用仓库 NYC 2025-12-18 08:00–10:00 样本，30 秒 epoch、240 步完整时域。
训练种子 `1100..1119`、验证种子 `1200..1205`、测试种子 `1300..1309` 互不重叠，
也与原 NYC 概率模型的训练/验证种子隔离。

脚本执行：

1. 归档当前源文件，核对随机拒单和 2 km 范围，不读取历史回归模型。
2. 当前 NYC 环境重新采集 20 轮训练、6 轮验证的完整 30 特征数据；独立训练神经网络并保存全部候选损失。
3. 冻结模型及验证阈值后，每个测试种子各运行两遍：普通 MCMF、不调用预测器；以及普通 MCMF、只在回答前计算神经网络概率。
   两遍都保持 `ADP=0`、`knownreject=False`、无 Q 网络、无 sequential recourse。
4. 核对两遍的逐步车辆/订单/充电状态、奖励、实际邀约与回答完全一致。
   预测值不参与分配，因此实际拒单、完单数应相同，而不是因预测器训练而自动下降。
5. 在新留出数据上比较神经网络、训练接单率常数基线和仅诊断的仿真真实条件概率。

默认正式检查包含 46 个完整环境 episode：20 训练 + 6 验证 + 10×2 测试。
`--max-steps` 仅供短程接口检查，不应当作完整仿真性能结果。
输出目录必须不存在；原来的概率模型及 Q/residual checkpoint 不会被覆盖。

## 怎么看结果

- 概率质量：留出 log loss、Brier score 越低越好，ROC-AUC 和拒单 Average Precision 越高越好。
- 分类：以拒单为正类，报告 TP/FP/FN/TN、accuracy、precision、recall、F1、balanced accuracy。
- 两个阈值：固定 `p_reject >= 0.5`；以及只在验证集最大化拒单 F1 的阈值。
  后者通常提高拒单召回，但可能增加误报、降低总体准确率；不改变原概率输出或 MCMF 策略。
- 实际拒单：区分重复邀约拒绝次数和去重被拒订单数，不能把“预测拒单”当作“实际拒单”。
- 充电：所有 episode 保留 EV/AEV/平台的次数、日均次数与时长，旁路预测前后逐项核对。

输出包括 `report.md`、`summary.json`、`loss_history.jsonl`、`candidate_loss_histories.json`、
`model.json`、`test_predictions.jsonl`、`decision_thresholds.json`、源代码归档，以及逐环境的完整日志和统计。
概率误差改进区间按整条测试种子轨迹重采样；仍是同一天订单上的仿真响应，
不能当成跨日期或真实人类司机泛化证明。

## 历史回归模型结果（非当前神经网络）

2026-08-28 已完成 NYC 200 车（100 EV、100 AEV）的上述完整检查，结果保存在
`results/acceptance_checks/nyc-200-100ev-100aev-20260828/`。
`report.md` 是实验结论，`verification.md` 是 252 项测试和完整仿真的验收记录。
新模型训练目标从 `0.172773` 降到 `0.103039`，验证 BCE 从 `0.156461` 降到 `0.095299`。
当前 checkpoint 在新的留出测试上，0.5 拒单阈值的召回率为 0；
验证集选出的 `0.199321` 阈值使拒单召回率为 69.42%、精确率为 24.24%，同时总体准确率降低。
10 组配对测试的实际拒绝邀约次数均为 206、完单均为 7,313；旁路预测不改变 MCMF 策略。
