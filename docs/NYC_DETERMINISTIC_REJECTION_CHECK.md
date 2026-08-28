# NYC 历史确定性拒单：独立重训练检查

> 当前默认已改为 `reject_uniform=True`、2 km 接客范围，概率模型也已替换为神经网络。
> 本文数值对应此前 False / 5 km 版本，保留作历史记录；下文“当前”仅指该历史运行。
> `check_nyc_deterministic_rejection.py` 只适用于显式设为 False 的环境，默认 True 时会拒绝继续；
> 新实验请看 [神经网络检查](NEURAL_ACCEPTANCE_MODEL.md)。

## 当前规则

`src/NYCEnvironment.py` 默认 `reject_uniform=False`，同时 `ride_acceptance_noise_std=0`。
`_should_reject_request` 使用固定比较值 0.5，不调用拒单的 uniform 随机数入口：

```python
rejected = 0.5 < rejection_probability
# 等价于底层 acceptance_score < 0.5；等于 0.5 时接受。
```

因此“拒单概率小于 0.5”不是当前的拒单区域；“接单分数小于 0.5”才是拒单区域。
完整仿真仍有车辆初始化等其他随机过程，但不能把这些随机性误认为本次拒单判断在抽样。

## 为什么另设检查脚本

旧的概率训练与审计曾使用随机拒单版本。其历史结果不能直接解释当前固定阈值版本，
旧模型也不会因为环境规则变化而自动学到新的确定性标签。

`check_nyc_deterministic_rejection.py` 使用当前默认环境重新采集真实分配的邀约，
不修改 MCMF、拒单系数、需求、范围限制或充电规则：

1. 默认 200 车（100 EV、100 AEV），40 轮训练、12 轮验证、20 轮测试。
2. 使用新种子，训练、验证、测试互斥；不混用原随机响应的训练标签。
3. 每轮验证 `reject_uniform=False`、噪声为零、logit shift 为零；对拒单随机数入口设置报错哨兵。
4. 每个实际 EV 回答都核对固定比较值、底层分数和真实标签；测试时概率在司机回答前计算。
5. 只有实际训练数据同时含接受、拒绝两类时才拟合。沿用生产模型的 BCE + L2；仅验证集选择 L2 和分类阈值。
6. 单一类别时保存 `not_trainable_single_class`，不伪造新模型。验证集无拒单时不选择 F1 阈值。
7. 保留所有 EV/AEV/平台充电统计、完整订单守恒检查、原始邀约与逐步日志。

## 运行

```bash
python check_nyc_deterministic_rejection.py \
  --num-vehicles 200 --num-ev 100 --workers 2 \
  --output-dir results/acceptance_checks/NEW_UNIQUE_DIRECTORY
```

默认 NYC 数据为 2025-12-18 08:00–10:00，30 秒一步，每轮 240 步。
新输出目录必须不存在，原模型不会覆盖。`--max-steps` 仅用于接口冒烟检查。

## 分数、真实响应概率与样本覆盖

底层 logit 的接单分数可能是 0.8。在随机版本中它表示 80% 接单概率；
在当前确定性版本中，0.8 大于阈值，因此给定这些输入时实际接单概率为 1。

旧数据收集器把底层分数命名为 `oracle_acceptance_probability`。此独立脚本在汇总数据中
将它保留为 `latent_acceptance_score`，逐条验证后将真实确定性条件概率另存为 0/1。
这些诊断字段都不进入学习模型；输入仍只有分配前空闲时间、接客时间和加价额。
各轮子目录的原始邀约保留旧收集器字段，汇总层明确新口径，避免误改历史数据。

如果实际 MCMF 分配的邀约没有落到接单分数小于 0.5 的区域：

- 没有拒单训练样本，无法由这批数据学习分类边界。
- 没有拒单测试样本，拒单召回率不可估计，而不是已经达到 100%。
- 即使旧模型全预测接受、准确率达到 100%，也不能证明它能识别应该拒绝的邀约。
- 若需要验证边界，必须另外明确采集范围，例如未被最终选中的候选配对；不能暗中改规则或制造标签。

## 输出

`report.md`、`summary.json`、`training_status.json`、`decision_thresholds.json`、
`test_predictions.jsonl`，以及所有完整仿真日志和充电统计。
仅有效训练时才产生 `model.json` 和损失轨迹。
源代码归档与 SHA-256 用于区分旧随机分支与当前确定性分支。

## 进一步检查“为什么全部接受”

完整运行结束后可执行：

```bash
python inspect_nyc_rejection_outputs.py \
  --run-dir results/acceptance_checks/nyc-deterministic-200-100ev-100aev-20260828
```

该检查读取当前环境实例的真实参数，并逐邀约复算接单/拒单分数；
输出接客距离、时间、空闲时间、加价金额及效用项的分布。
同时计算默认 5 km 接客范围和固定阈值的关系，以及同一批邀约去掉加价项的静态结果。
后者不是更改加价后的策略重跑，不包含订单分配反馈。
报告保存为 `rejection_parameters_report.md`，逐邀约数值为 `rejection_output_checks.jsonl`。

## 已完成的正式检查

`results/acceptance_checks/nyc-deterministic-200-100ev-100aev-20260828/` 已完成 72 轮完整检查：
训练/验证/测试分别有 18,618 / 5,574 / 9,324 次实际 EV 邀约，全部接受，拒单为 0。
最低接单分数为 0.711696，未进入小于 0.5 的拒单区域。
训练状态为 `not_trainable_single_class`，没有生成新的分类模型，不能报告有效的拒单召回率。
274 项代码测试通过；参数诊断、逐邀约输出和原有充电统计均已保留。
