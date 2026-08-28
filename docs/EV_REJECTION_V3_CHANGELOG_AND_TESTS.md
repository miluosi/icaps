# EV 拒单预测与单阶段 Q/residual：逐条修改与验收

日期：2026-08-28。基础提交：`2ff74b8`。本次修改保留在工作区，未提交 git。

## 1. 范围与结论口径

依据用户提供的 `MUST_CHANGE_EV_REJECTION_RESIDUAL.md`（下称文档 A）和
`ICAPS_RESIDUAL_REJECTION_MUST_CHANGE.md`（下称文档 B），落实当前拒单 trainer、
预测输入、结构化基准、回放和检查点协议。**没有实现多阶段 ADP**；已有 recourse 分支没有新增决策阶段。

本次验收针对代码协议和本地可运行性，不把短程测试说成已收敛的论文实验。
原有充电次数、充电时长统计保留；历史 v1/v2 模型、训练结果没有删除或覆盖。

## 2. 文档要求逐条对照

| 原文位置 | 修改与检查 | 本轮状态 |
|---|---|---|
| A §3；B P0-1 | 主类改为 `EVRejectionProbabilityModel`；标签 `rejected=1`；`predict_proba()` 直接返回拒单概率。旧类名只保留导入别名，不能载入旧模型。 | 已实现 |
| A §4；B P0-2 | 默认 `driver_offer_core`，仅 idle time、pickup time、surge bonus。30 维 `platform_context` 单独作为消融，禁止 schema 混用。 | 已实现 |
| B P0-3 | 神经网络定位为仿真响应规律的辅助估计器，不声称发现新的真实司机行为。保持现有响应公式，不新增异质性/非线性真实响应机制，也不恢复回归预测器。 | 按本轮范围保留 |
| A §4.4；B §4.1 | 默认 `3→16→8→1` ReLU MLP，209 参数；context 模型独立使用 `30→64→32→1`。 | 已实现 |
| A §5；B P0-5 | 自然类别比例，真实二元标签，BCEWithLogitsLoss + 权重 L2、Adam；训练集标准化；验证 NLL 选 epoch/L2。 | 已实现 |
| A §5.3；B P0-5 | 独立验证集 temperature/Platt 校准；保存参数、校准前后 NLL。优化未改善则保留恒等校准。Q 使用校准概率。 | 已实现 |
| A §8.3；B P0-4 | 独立监督采集：MCMF/分层可行提案/随机可行提案。只执行真正选中的报价并记录实际响应，不对未选边调用响应函数造标签。 | 已实现、NYC 已运行 |
| A §5.5；B P0-4 | 完整日期列表接口与重叠检查；单日期运行明确标注 same-demand-day stochastic-response holdout。保存 day/episode/seed。 | 接口已实现；跨日期实验未运行 |
| B P0-4、§6.3 | 保存全部可行边的未标注输入、行为策略、候选数、分层及可计算的条件提案概率；比较训练支持范围与测试/可行边范围。 | 已实现；不声称覆盖所有部署状态 |
| A §6.1–6.2；B P0-6 | 默认 residual 的未回答 EV service 边使用 `(1-q)g_success + q g_reject`。AEV、充电、重定位、等待、已接受的继续服务不做拒单混合。 | 已实现 |
| A §6.3；B P0-7 | 所有已注册学习器的概率输入升级为 `q_reject + human_response_mask`；两列零初始化，critic/target 同维；actor 不接这两列。 | 已实现 |
| A §6.4；B P0-6 | 不用 0.5 阈值删除边，不把整个 Q/residual 乘 `(1-q)`，不另外重复扣期望拒单罚。 | 已检查 |
| A §6.5；B P0-9 | 保留真实执行 reward、单阶段折扣与残差 Bellman 形式；current/next 分别使用自身冻结概率与 anchor。Direct-Q 不减 current anchor。 | 已实现/回归检查 |
| A §8.7；B P0-8 | 快照保存 success/rejection/expected score、q、mask、predictor hash；joint value 校验选中边之和；continuing 边 q=mask=0。 | 已实现 |
| A §8.4、§10.3；B §7.5–7.6 | 预测器冻结并从 TD optimizer 隔离；checkpoint/replay 升级 v3；拒绝旧版本、错误 predictor hash、不同 anchor/input 配置。 | 已实现 |
| B §6.10 | 训练/推理入口共享 v3 检查点命名；提供 response flags，旧 acceptance flags 是显式别名。 | 已实现 |
| A §10.4；B §7 | 新增边界、快照、当前/下一状态、梯度、schema、策略采集测试；独立输出 pickup-time/surge 条件校准与方向诊断。 | 已实现 |
| A §11；B §8、P1 | learned-vs-oracle 策略差距、多拒单率、多日期、足够更新预算的完整论文消融。 | 本轮未运行，不算已验收 |

### 两份文档的可选项如何处理

- 采集默认比例为 `0.8,0.1,0.1`（MCMF、分层、随机），对应 A 的 20% 探索建议。
  B 的 `0.5,0.25,0.25` 可通过参数设置；未把两个不同建议混写为同一个默认值。
- 默认 temperature；Platt 可选。`none` 仅供显式未校准消融，不能描述为已校准结果。
- schema 使用统一的 `feature_version=3` 加 `feature_variant`，而不是分别给核心输入和 context 重用含糊的 v1/v2 名称。
- 2 km 是司机到接客点的半径；保持 `reject_uniform=True` 及原实际拒单系数。未自动调整 logit shift 来制造更多拒单。
- 模型单独训练的目标不是拒单 F1；若诊断中使用阈值，只从验证集选择，不参与概率模型的 BCE/L2 选择或 Q 打分。

## 3. 核心公式与重要边界

令 `m` 表示“尚未得到回答的人类 EV 服务报价”。核心网络输出的校准概率为：

`q = sigmoid(logit / T)`，或 Platt 的 `q = sigmoid(a * logit + b)`。

结构化基准：

`g_expected = (1-q) * g_success + q * g_reject`（m=1）；m=0 时 q=0、基准保持原有 g。

部署：

`Q_total = g_expected + beta * clipped_residual`。

单阶段 residual TD：

`target_residual = reward_realized + gamma**elapsed * (next_g_expected + next_raw_target_residual) - current_g_expected`。

终止时 bootstrap 为零。Direct-Q 的目标是普通 full-Q target，不减 current anchor。
部署的 clip/beta 不能代替 Bellman target 中的 raw target residual。

**关于文档中 “expected immediate reward” 的字面表述：**现有成功分数包含整单价值和行驶成本，
而仿真当步奖励可能只有移动成本、上客事件或下客收入。为保持 `q=0` 恢复现有成功分数，
本次 anchor 是“服务选项结构化分数的拒单混合”，不是当步全部现金奖励的条件期望。
CLI 保留 `expected_immediate` 这一请求中的命名，但不能据此宣称两者数值相等。

NYC 的显式拒单 penalty 与执行路径共用 `rejection_score()`：默认按整单价值比例，或已有的 base+距离分支。
拒单后的实际移动奖励仍保留在 TD 的真实 reward 中。synthetic 默认没有新增显式拒单罚；
同时修复了旧存储分支把实际 `0` 奖励硬改成 `-1` 的问题。

因此 Monte Carlo 测试验证的是**同一对结构化分支**的均值，并另外测试 penalty helper 与环境一致；
没有伪造“整单 surrogate 等于单 epoch 总奖励”的测试。如果要求后者，需要另行统一动作时域和奖励定义，超出本轮修改。

## 4. 文件与接口

| 文件 | 职责 |
|---|---|
| `src/acceptance_inputs.py` | 核心/context 输入及 NYC 分钟、synthetic 步数单位 |
| `src/acceptance_model.py` | rejected=1 MLP、BCE/L2、校准、概率指标、支持范围、v3 JSON |
| `src/rejection_collection.py` | 独立混合可行采集；独立 RNG；保持精确求解器容量约束 |
| `train_acceptance_model.py` | 监督训练、日期/种子划分、测试、完整源代码快照和报告 |
| `src/acceptance_features.py` | 冻结模型，q/mask 构造、检查点身份、在线/回放接口 |
| `src/rejection_anchor.py` | 纯函数 penalty/expected anchor、选中与可行边诊断 |
| `src/ValueFunction_st_masac_gat*.py`、`src/ValueFunction_pytorch_bayes.py` | critic/target 两列输入、actor 排除、经验存储 |
| `src/ValueFunction_optimization_anchored_residual.py` | residual 默认启用响应期望 anchor |
| `src/recourse/types.py`、`state_snapshot.py`、`replay.py`、`lifecycle.py` | v3 不可变快照、joint value、schema/身份验证、oracle 与预测分离 |
| `src/NYCEnvironment.py`、`src/Environment.py` | 记录回答前的预测；共享实际 penalty；不改变随机响应规则和充电统计 |
| `src/ADPtrainer.py`、`src/NYCtrainer.py`、四个 train/test 入口 | 同一参数与检查点命名 |
| `run_acceptance_ablation.py` | 单阶段 Direct-Q/residual 短训练、q/mask 更新、冻结加载、响应分数诊断 |
| `verify_rejection_v3_run.py` | 独立复算概率/支持范围/条件校准，复跑四组检查点并对照结果 |

新代码应导入 `EVRejectionProbabilityModel`。`BinaryAcceptanceModel` 只是导入别名，
其 `predict_proba` **也已经是拒单概率**，旧调用者不能继续把它当作接单概率。
显式的 `predict_acceptance_probability()` 仅返回拒单概率的补数，供兼容诊断使用。

独立训练器选项：

```text
--ev-response-target rejection
--ev-response-feature-variant driver_offer_core|platform_context
--ev-response-calibration temperature|platt|none
--ev-response-behavior-policy-mixture 0.8,0.1,0.1
--train-dates ... --validation-dates ... --test-dates ...
```

Q/residual 入口选项（feature variant、calibration 从指定模型的 v3 checkpoint 读取）：

```text
--ev-response-feature predicted --ev-response-model PATH/model.json
--ev-response-anchor auto|off|expected_immediate
--ev-response-critic-input q_mask|none
```

不加概率仍可用 `--ev-response-feature off`，默认不会私自寻找旧模型。
`auto + q_mask`：Direct-Q 只增加输入；主 residual 增加输入和 anchor，其他历史 learner 不改打分基准。

| residual 机制消融 | anchor | critic-input |
|---|---|---|
| feature-only | off | q_mask |
| anchor-only | auto | none |
| anchor + residual inputs | auto | q_mask |

不同设置的检查点文件名及内容身份均不同。旧 v1/v2 模型或 joint replay 必须重训/重采，不能补零或静默反转标签继续用。

## 5. 测试需求由哪些代码验证

主要新增测试：[tests/test_rejection_v3_contract.py](../tests/test_rejection_v3_contract.py)。
原 `test_acceptance_model.py`、`test_acceptance_learning.py` 等同步升级到 v3 语义，保留原有回归覆盖。

| 验收需求 | 自动检查 |
|---|---|
| 核心输入无越权/未来/标签泄漏 | 缺少 dropoff、fare、market 对象仍能构造核心输入；污染 oracle、ID、回答不改变模型 |
| 校准/单位/模型 roundtrip | temperature/Platt 验证 NLL 不增；NYC live/snapshot 分钟一致；重载概率相等 |
| q=0、q=1、单调 anchor | 恢复 success/reject 分支；固定分支 Monte Carlo；错误 q/mask 报错 |
| AEV、非 service、continuing | q=0、mask=0；继续服务不发生第二次拒单混合 |
| 在线/快照/回放一致性 | 同一新报价的 g 一致；修改 live 时间、车辆、订单后旧回放不变 |
| 当前/下一状态 target | 不同 current/next q；真实负奖励；终止 bootstrap；Direct-Q 不减 anchor |
| 梯度与 actor | q/mask 两列有梯度；预测器不在优化器中且无 TD 梯度；actor 不受这两列影响 |
| 策略采集 | 不给未选边生成回答；可行性/不重复请求提案；独立 RNG；策略与 propensity 含义记录 |
| checkpoint/replay | 旧 schema、错模型 hash、不同集成模式拒绝；训练/测试入口同一命名 |
| 三种集成方式 | beta=0 时 feature-only 不改变 g，anchor-only 与两者一起的期望 g 相同，q 增大时 g 下降 |
| 真实运行 | NYC 200=100+100、精确 MCMF、真实监督训练；两种 learner 各 off/predicted 的 TD 更新和独立检查点复跑 |

主测试命令：

```bash
/opt/anaconda3/bin/python -m pytest -o addopts='' -q \
  --junitxml=results/rejection_v3/pytest-final-20260828.xml
```

本次监督训练：

```bash
/opt/anaconda3/bin/python train_acceptance_model.py \
  --environment nyc --num-vehicles 200 --num-ev 100 \
  --train-seeds 411 412 413 414 --validation-seeds 511 512 --test-seeds 611 612 \
  --start-hour 8 --stop-hour 9 --nn-epochs 60 --nn-patience 15 \
  --ev-response-behavior-policy-mixture 0.8,0.1,0.1 \
  --output-dir results/rejection_v3/NEW_PREDICTOR_RUN
```

单阶段接口短训练：

```bash
/opt/anaconda3/bin/python run_acceptance_ablation.py \
  --environment nyc --num-vehicles 200 --num-ev 100 \
  --train-seeds 71 --test-seeds 9001 --episodes 1 --max-steps 40 \
  --train-every 10 --batch-size 2 \
  --acceptance-model results/rejection_v3/NEW_PREDICTOR_RUN/model.json \
  --output-dir results/rejection_v3/NEW_INTEGRATED_RUN
```

完整输出目录不能重复使用，防止覆盖旧实验。独立复核：

```bash
/opt/anaconda3/bin/python verify_rejection_v3_run.py \
  --predictor-run results/rejection_v3/nyc-200-100ev-100aev-verified-20260828 \
  --integrated-run results/rejection_v3/nyc-integrated-verified-20260828 \
  --pytest-xml results/rejection_v3/pytest-final-20260828.xml \
  --output-dir results/rejection_v3/NEW_VERIFICATION
```

## 6. 本地结果入口与尚未满足的研究要求

实际验收：**308 项代码测试通过**，0 失败/错误/跳过；四组保存的检查点独立重放均与原运行结果一致。
NYC 200 车、100 EV + 100 AEV；监督数据为训练 1,381、验证 695、测试 678 次实际报价。
训练 BCE `0.217173→0.181399`，验证 BCE `0.199937→0.168640`；校准后验证 NLL `0.164494`。
测试 NLL `0.214523`（常数 `0.261634`），AUC `0.859382`；独立采集 8 轮累计实际充电 19 次，统计保留。

**概率质量不能全部判通过：**接客时间增加 1 分钟（在训练范围内）的诊断中，平均 q 变化为
`-0.001049`，72.38% 样本方向与现有响应公式相反；surge 增加时 q 下降，方向诊断正确。
测试 ECE `0.030314` 也高于常数基线 `0.015791`。因此“代码通过”不等于该小样本模型行为方向和校准已经合格。
测试 19,964 条可行边中 4 条超出训练特征最小/最大范围；这只是一维范围诊断，不是联合分布覆盖证明。
0.5 阈值漏掉全部 49 次实际拒单；验证集阈值 `0.083803` 在测试得到 TP=35、FP=106、FN=14、TN=523，
这项阈值诊断不进入 Q 打分，也不改变概率模型选择。

四组 40 步冻结评估分别为 Direct-Q off/predicted 完单 `86/83`、拒单 `13/17`，
residual off/predicted 完单 `89/93`、拒单 `8/8`。仅作运行检查，不能宣称所有模式性能改善。
该短训练使用的 v3 模型与最终监督训练模型文件 SHA-256 完全相同。

- [监督训练报告](../results/rejection_v3/nyc-200-100ev-100aev-verified-20260828/report.md)
- [单阶段短训练报告](../results/rejection_v3/nyc-integrated-verified-20260828/report.md)
- [独立验收报告](../results/rejection_v3/verification-20260828/verification.md)
- [机器可读验收及概率诊断](../results/rejection_v3/verification-20260828/verification.json)
- [全量 pytest XML](../results/rejection_v3/pytest-final-20260828.xml)

测试能检查上述实现协议、奖励口径和本地运行，但**不能代替**以下研究验收：

1. 跨日期泛化：本轮只运行指定 2025-12-18 订单文件；种子留出不是日期留出。
2. 充分覆盖：保存并报告越界比例，但没有证明训练样本覆盖所有 residual 部署状态。
3. 充分收敛与统计显著性：本轮每臂 40 步、4 次 TD 更新，不足以证明学习收敛或性能稳定提升。
4. 完整论文矩阵：learned/oracle 策略差距、三种集成方式长期消融、多拒单率仍需另行运行。
5. 条件校准和单调性属于诊断，不硬编码成损失，也不以测试集调参抹掉失败区间。

因此这里的“通过”是本轮**代码及接口验收通过**，不是声称两份文档的所有论文 Definition of Done 均已完成。
