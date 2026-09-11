# ICAPS Q-value 显存检查（2026-09-11）

## 核查范围和发现

对照本地 `/Users/seinzhou/Desktop/adp_trainer` 的同名实现。

- NYC 的 `generate_vehicle_qvalue` 返回的是 NumPy Q matrix，矩阵本体在 CPU 内存；主要显存压力是全部可行边的神经网络特征、中间激活及逐边临时张量。
- adp_trainer 的旧 Bayesian 接口中有小批处理/失败重试思路，但其 `ValueFunction_st_masac_gat.batch_get_mixed_q_values` 同样一次性处理整批边，不能直接照搬来解决当前默认 learner 的显存问题。此次未采用其多 GPU 路径。
- ICAPS 默认 `optimization_anchored_residual` 继承 post-demand/direct/GAT 基类。普通 Q matrix 路径一次性生成所有边的特征，原实现每条边单独把 local features 送到 device 并 concat，随后整批调用 twin critics。
- recourse 的 `_graph_edge_scores` 是另一条独立路径，用于在线图打分和 Bellman target；也曾一次性堆叠某 fleet provider 的全部边。
- 每次推理后调用 `empty_cache` 不能解决仍然活跃的全部边张量，因此没有增加常规 cache 清理。

## 实际修改

- 默认最多每批 **4096 条边**执行 Q 推理，只分批构造特征、运行网络和回传结果；完整 NumPy 结果保持原边顺序。
- CUDA OOM 时释放失败调用的局部张量后，把批量减半并重试相同边。Q 推理与图打分会记住降低后的安全批量；批量为 1 仍失败则抛出异常，不使用零 Q 值代替失败结果。
- local features 统一传输，用索引批量获取车辆和目标位置的 embedding，保留训练所需梯度。
- 排队预测器的推理也有批量上限；post-demand 预测随 edge chunk 执行，其完整预测向量在 CPU 合并，避免 direct-demand 子类看到最后一个分块的数据。
- 保留全批次 `std(g)`、原来的 online/target correction 语义及 post-demand relocation/request 比较。没有在 NYC 外层直接分割完整公共评分调用。
- 图打分显式 `no_grad`，因为该接口最终返回 Python 浮点分数；可微训练接口保持原来的梯度。

此开关控制 Q 推理，不限制整个进程的总显存。模型/优化器状态、GAT 节点注意力、训练反向传播另有内存成本；CPU 的 dense Q matrix、动作矩阵和边属性也仍随规模增长。其他非 GAT 历史 learner 未统一改写。

## 本地结果

文件：`results/qvalue_memory/20260911_104211_734486.json`。

单线程 CPU，合成 500 车、81 区域的可行边输入，默认 residual learner；每个规模 3 次重复。比较原逐边特征组装方式的一次性推理与新版分批批量组装。包含特征、图上下文和预测，**不包含 NYC 仿真、可行边生成或 assignment**。

| 可行边数 | 原方式平均耗时 | 新方式平均耗时 | 耗时下降 |
|---:|---:|---:|---:|
| 1,000 | 32.33 ms | 33.97 ms | -5.08% |
| 10,000 | 252.06 ms | 249.01 ms | 1.21% |
| 50,000 | 1,283.52 ms | 1,181.43 ms | 7.95% |

这些输入的最大 Q 差异为 0。小规模没有稳定加速；当前优化首先保证边推理的显存占用有界，不能根据 CPU 结果宣称服务器 GPU 会加速同样比例。

当前 edge_dim=211。仅 float32 edge feature 矩阵的体积，50,000 条边为 40.25 MiB，4096 条边为 3.30 MiB。对于 1,000,000 条边，整批特征约 804.90 MiB；新方式同一个分块仍为 3.30 MiB。这是张量大小计算，**不是 GPU 峰值实测**，不包括其他张量。本机无 CUDA，JSON 显存测量字段为 null。

## 服务器运行

自动使用新默认值，无需改模型权重或 checkpoint 格式。也可显式降低每批边数，环境变量在新建 value function 时读取：

```bash
ICAPS_QVALUE_BATCH_SIZE=2048 python run_nyctrainer.py --methods r1 r2 r3 macro
```

沿用你自己的其余训练/测试参数。Python 构造接口也支持 `qvalue_inference_batch_size=2048`。

服务器显存和速度诊断：

```bash
python benchmark_qvalue_memory.py --device cuda --edges 1000 10000 50000 200000 --batch-size 4096 --repeats 3
```

每次在 `results/qvalue_memory/` 保存独立 JSON，包括源文件 hash、实际设备、耗时、Q 差异、CUDA peak allocated/reserved、OOM 重试和实际批量。原方式若 OOM 后缩批，会标记 `baseline_retried=true`，不能把该行当作成功的全量推理基线。只使用一个 GPU，不改变进程级并行设置。

## NYC 充电容量问题是否还存在

**仍存在两个不同层面的排队来源。**

1. 当前默认非保守模式按照此前要求，只检查确定车辆在候选车到达时刻的占用；本批次不同到达时刻的新分配不会互相预占完整充电区间。因此它不是整个未来区间的无排队保证。Conservative 模式仍加入全部潜在车辆的虚拟区间。
2. `NYCEnvironment.step` 先执行动作/到达，再调用 `_update_environment` 释放旧车。仅剩 1 epoch 的旧车与行程 1 epoch 的新车构成边界反例：预计日历认为可复用，但新车可能在实际释放前入队。两种模式均可复现，现有两个 strict xfail 仍成立。本次没有改充电策略、MDP 时间顺序或旧实验数据。

## 验证

验证覆盖五种 value function 的分块/完整评分一致性、残差裁剪启用状态、actor 项、post-demand 子类输出、重复索引的 embedding/mixer 梯度、排队预测分批、OOM 缩批/释放失败帧和真实异常传播，以及 online/target 图分数一致性。

```bash
.venv/bin/python -m pytest -q tests/test_qvalue_inference.py tests/test_recourse_must_fix.py tests/test_repair_only_learning.py tests/test_rejection_v3_contract.py tests/test_recourse_credit.py tests/test_recourse_reaudit.py
```

上述 **84 项通过**。充电相关回归此前同时运行通过，另有 **2 项已知仿真时序问题的预期失败**；预期失败不属于无排队验证通过。
