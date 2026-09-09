# MCMF 单线程性能核查与 ADP 计时输入对齐

> 本页记录替换 MCMF 后端后的第一轮结果。随后已完成 [精确 SSG 预处理加速](SSG_PREPROCESSING_ACCELERATION_2026-09-09.md)，当前构图代码及新的时间请以该报告为准。

## 结论

旧计时脚本的 MCMF 慢，主要因为它调用 Python SPFA 逐次最短路增广：每辆车发送一单位流，重新扫描残量网络。性能瓶颈不是 SSG 缩图的正确性。现在计时脚本默认使用 **OR-Tools 9.14.6206 的 `SimpleMinCostFlow`**，保留原有精确 SSG/EAGR、容量约束及 `cost_scale=10000`。

最终默认工作负载已对齐本地 `adp_trainer/test_alg_time.py` 的矩阵生成规则。3000 辆车 / 15000 个订单、5 个相同种子的单线程实验中，SSG+MCMF 的平均建图及求解时间为 **1.005 秒**，SSG+CPLEX 为 **4.318 秒**，快 **4.30 倍**。四种方法在全部 25 个实例上得到完全相同的整数目标值，共 100 条求解记录。

## 最终默认接口

计时脚本：`benchmark_cplex_mcmf_ssg.py`。

```text
--case-distribution adp
--mcmf-backend ortools
--repeats 5
--seed 2101
--cplex-threads 1
--warmup
--cost-scale 10000
--plot-metric end_to_end_seconds
```

调用链为 `solve_method → src.exact_mcmf.solve_ortools → ortools.graph.python.min_cost_flow.SimpleMinCostFlow`。完整图和 SSG 图交给同一个 MCMF 后端；SSG 的缩图代码没有改动。OR-Tools 批量读回流量，随后仍校验每车一次分配、可行性、容量和整数目标值。没有启用 GPU、并行求解、近似 top-K、降低精度或求解器自动回退。

**训练入口与计时入口需要区分。** `run_trainer.py` 的默认仍是 `mcmf_solver=exact, mcmf_backend=docplex_network`；`solve_exact(backend="auto")` 也优先尝试 CPLEX。因此训练时要明确传 `--mcmf-solver exact --mcmf-backend ortools --mcmf-use-cpu`，不要把 `auto` 当成 OR-Tools。实际后端会写入 `env.mcmf_last_result["backend"]`。

## 与 adp_trainer 对齐到什么程度

参考文件：`/Users/seinzhou/Desktop/adp_trainer/test_alg_time.py`，核查时 SHA256 为 `8697266f23a392647c1f785d8c1cdee89d90ab6eb6dd3a42550588b5578a86dd`。

原 ICAPS 脚本只对齐了五档规模，其他部分有明显差别。现在默认 `adp` 模式对齐了：

| 项目 | 最终 ICAPS 默认 |
|---|---|
| 车辆数 | 100、500、1000、2000、3000 |
| 订单数 | 车辆数 × 5 |
| 充电 / relocation 动作数 | 50/10、100/50、200/100、400/200、600/300 |
| 随机种子 | 每档均为 2101–2105 |
| 候选边 | 与 ADP 相同的 Bernoulli 密度、保底候选和随机数调用顺序 |
| 订单默认密度 | `min(0.25, max(8/R, 0.01))` |
| 充电默认密度 | `min(0.4, max(4/C, 0.05))` |
| relocation 默认密度 | `min(0.5, max(4/Z, 0.1))` |
| 充电容量 | 同一随机容量生成规则，可显式固定容量 |
| 等待动作 | 同一个真实、零收益、容量 N 的最后一列 |
| Q 值 | 同一 float32 原始奖励、无效值 -1e6、四位小数 canonical float64 |
| 统计口径 | 5 个种子的平均值、标准差；主指标包含建图与求解，排除输入生成 |

已直接调用参考文件的 `build_matrix`，在 5 个配置（含 100、500 车辆、纯 EV、纯 AEV、自定义密度）逐项检查可行性矩阵、Q 矩阵和容量数组，全部精确相等。两组参考数组的 SHA256 还作为回归测试固定下来，无需测试时依赖另一个仓库。

**不能把两个项目的旧图直接当成同一个实验结果：**

- ADP 原脚本的 `mcmf` 是旧 Python SSP/SPFA；`mcmf-exact` 是 Python `primal_dual`；`mcmf-exact-gurobi` 是 Gurobi 网络后端。
- ADP 原脚本默认 `gurobi` 使用二元变量，除非显式 `--gurobi-lp`。ICAPS 当前 CPLEX 对照是连续网络 LP，且 `lpmethod=auto`，没有显式强制 network simplex。
- ADP 创建优化器时沿用其默认线程配置；ICAPS 按本次要求固定 CPLEX 为 1 线程。
- ICAPS 默认预热，并计入求解后完整的分配验证；ADP 计时业务分配调用，返回后的 `assignment_reward` 求和不计时。两者主口径接近，细小包装边界并不完全相同。

原 ICAPS 固定每车 8 个候选订单的实验仍可通过 `--case-distribution sparse` 复现。这个工作负载在大规模下远稀于 ADP，不能混用两者结果。

## 最终对齐后的性能

均为单线程、预热后、5 个种子的算术平均值，单位秒。包括共同建图、后端模型构建、求解、解码与校验；不包括矩阵生成、学习网络打分、完整仿真运行。

| 车辆 / 订单 | CPLEX 完整图 | MCMF 完整图 | SSG+CPLEX | SSG+MCMF | SSG 上的加速比 |
|---|---:|---:|---:|---:|---:|
| 100 / 500 | 0.023266 | 0.002242 | 0.009382 | 0.001947 | 4.82× |
| 500 / 2500 | 0.166728 | 0.026029 | 0.131854 | 0.028085 | 4.69× |
| 1000 / 5000 | 0.559385 | 0.099846 | 0.508012 | 0.111780 | 4.54× |
| 2000 / 10000 | 2.114087 | 0.399733 | 1.976406 | 0.463800 | 4.26× |
| 3000 / 15000 | 4.698770 | 0.977411 | 4.317773 | 1.004996 | 4.30× |

3000 车辆时，SSG 后的后端调用均值是 **CPLEX 3.410370 秒、MCMF 0.097592 秒**，约 34.95×。但这包含 DOcplex 的 Python 建模等开销，不能宣称“底层优化算法单独快 35 倍”。诊断中 CPLEX 明确报告模型为 LP，其原生优化时间小于整个 DOcplex 调用时间。

同规模的 SSG 建图均值为 **0.907403 秒**，占 SSG+MCMF 总时间约 90.3%。所以库后端替换后，进一步优化的重点已经转到预处理。当前 SSG+MCMF 总时间略高于完整图 MCMF，是缩图成本超过本例省下的原生求解成本；按要求保留精确 SSG，不通过去掉缩图制造加速。

## 旧实现为什么慢

令 N 为车辆数、A 为动作数、V 为流图节点数、E 为正向弧数、F 为实际发送流量；本分配问题源到车辆容量为 1，因此完整分配 F=N。

| 实现 | 算法与复杂度 | 本地表现 |
|---|---|---|
| `GurobiOptimizer._MCMFSolver` | SPFA + SSP；最坏 `O(FVE)`，残量图空间 `O(V+E)` | 每车一次增广，每次搜索到队列为空；用 NumPy 数组做大量 Python 标量访问 |
| `PrimalDualMinCostFlow` | Johnson 势 + 堆 Dijkstra；初始化 `O(VE)`，随后约 `O(F(E log V+V))` | 初始分层图通常只需很少 Bellman-Ford 扫描；仍有 Python 搜索与堆开销 |
| `solve_primal_dual` 的基线分支 | 只发送严格正收益的改进流，增广数 K≤N | 已有优化；当每行都有真实基线时省掉大量基线增广，无基线的行仍需完整流求解 |
| OR-Tools `SimpleMinCostFlow` | C++ cost-scaling push-relabel；官方实现给出的最坏界 `O(V²E log(VC))`，C 为最大绝对整数费用 | 实测速度优势来自原生实现、批量操作和算法行为，不能仅用最坏界断言必胜 |

OR-Tools 算法及界见 [9.14 官方源代码说明](https://github.com/google/or-tools/blob/v9.14/ortools/graph/min_cost_flow.h)。CPLEX 可以解网络流 LP，也提供专用网络优化器；本仓库调用只设置线程、数值容差，未显式选择专用算法，见 [IBM 网络 LP 说明](https://www.ibm.com/docs/en/icos/22.1.2?topic=problems-solving-network-flow-as-lp)。

在原稀疏分布的 500 车辆 profile 中，旧版执行 **501 次 SPFA**，SPFA 累计时间占该包装调用约 **99.2%**，约 204.6 万次出队。缩图后虽然 E 显著降低，旧版仍发送 N 单位基线/共享流，保留的源边和基线边会造成大量重复搜索。

预热后原稀疏分布的旧版完整图求解中位数：500 车 1.457 秒、1000 车 5.897 秒，翻倍规模约四倍耗时。SSG 旧版对应 0.142 秒、0.566 秒。这里是观察到的近似二次增长，不能把最坏 `O(FVE)` 改写成一般情况下的二次保证。

仓库源代码都由 Python 调用，但求解内核不同：本机 CPLEX 包实际含 `py313_cplex2220.so` 和 `libcplex2220.dylib`；OR-Tools 同样由 Python 接口调用原生库。没有要求用户编写或维护 C++ 源代码。

## 下一步单线程优化方向

1. **保持精确 SSG，改造稀疏输入预处理。** 当前 `build_reduced_problem` 对整张 N×A 矩阵做量化、差值统计和存储，时间与空间均有 `O(NA)` 项；即使原始可行边很少也支付该成本。应在保证相同 Q 网格、无效值验证和统计语义的前提下，让后续图构建仅处理可行边，减少稠密副本。
2. **优化 EAGR 实现常数。** 现有单调指针使每条支配边至多删除一次，但行排序、Python incident 列表与 ArcMeta 对象仍有成本。当前预处理可概括为 `O(NA + Σ d_i log d_i + E + A log A)`，存储 `O(NA+E+N+A)`。可改数组结构和批量构建，不改变固定点、严格支配和容量证书。
3. **只在图中移除已被精确证书解决的行。** 仍需正确保留无 outside action 的强制行，不能为其虚构零收益等待。采用原生求解后，这通常低于稠密预处理的优先级。

本次已实现库接口选择、预热、输入对齐和批量流量提取；上述稀疏预处理重构没有冒充已完成。实验覆盖合成分配矩阵，尚未宣称完整 NYC 训练或全部仿真耗时也按同一倍数改善。

## 环境、复现和输出

实际测试：macOS ARM，Python 3.13.9，NumPy 2.3.5，CPLEX 22.2.0.1，DOcplex 2.32.264，OR-Tools 9.14.6206。求解串行运行；环境变量 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`VECLIB_MAXIMUM_THREADS`、`NUMEXPR_NUM_THREADS` 均为 1。

本地已创建 `.venv` 并安装可选依赖 `requirements-mcmf.txt`。初次继承 Anaconda 的 PyArrow 二进制时，与 OR-Tools 同进程导入发生原生崩溃；已在此虚拟环境中安装相同版本 `pyarrow==21.0.0` 的 PyPI wheel，验证两种导入和求解可共存。原 Anaconda 环境没有改动。因此应使用这里的 `.venv/bin/python`，而不是仍缺 OR-Tools 的系统 `python`。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python benchmark_cplex_mcmf_ssg.py
```

上述命令直接采用最终默认值并生成图。输出目录自动带时间戳；本次完整实验位于：

- [ADP 对齐结果：summary.csv](../results/cplex_mcmf_ssg/adp_aligned_ortools_default_20260909/summary.csv)
- [ADP 对齐结果：100 条原始记录](../results/cplex_mcmf_ssg/adp_aligned_ortools_default_20260909/raw_results.csv)
- [与参考项目的输入数组校验](../results/cplex_mcmf_ssg/adp_input_alignment_20260909.json)
- [旧 SPFA 的预热对照](../results/cplex_mcmf_ssg/legacy_warm_single_thread_20260909/summary.csv)
- [原稀疏分布 OR-Tools，扩展至 5000 车](../results/cplex_mcmf_ssg/ortools_single_thread_20260909/summary.csv)
- [强竞争稀疏分布 OR-Tools](../results/cplex_mcmf_ssg/ortools_contested_single_thread_20260909/summary.csv)
- [诊断 profile 与复现程序](../results/cplex_mcmf_ssg/runtime_profile_20260909/profile_solvers.py)

原稀疏分布的 5000 车/25000 订单实验：SSG+MCMF / SSG+CPLEX 总时间中位数为 1.195 / 1.310 秒。强竞争稀疏分布的 5000 车/5000 订单、32 候选订单实验：0.509 / 1.957 秒。这两组是补充实验，**不是最终 ADP 默认输入**。
