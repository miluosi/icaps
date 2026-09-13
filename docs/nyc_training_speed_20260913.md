# 3000 车训练耗时核查与等价加速（2026-09-13）

## 结论

当前服务器日志中的主要问题已经不是 Q matrix 或 OR-Tools assignment，而是随历史增长的生命周期统计、联合 critic 的逐边张量构建/训练，以及少数保存 checkpoint 的大停顿。本次在本地修改这些实现开销，保留每轮给定 Q 分数下的精确 MCMF 问题。

不能承诺任何服务器、任何订单规模、包含训练与保存的每一步都小于 30 秒。已记录的在线策略决策样本全部低于 30 秒；但日志并非每一步采样，也没有覆盖任意 3000 车全量候选订单的最坏情形。网络训练、回放收集、checkpoint 不应混为在线 assignment 的求解时间。

## 服务器日志证据

输入：Downloads 下 `nyc_r1_r2.log`、`nyc_r3.log`、`nyc_macro.log`（哈希已保存）。配置为 3000 辆总车，1500 HEV + 1500 AEV，3 个 AEV 站、150 slot，428 个全部站点、1850 总容量，OR-Tools exact、strict=True、整数 cost scale=10000。本次日志显示 device=cuda，与先前 CUDA 初始化失败回落 CPU 的旧日志不同。

`nyc_r1_r2.log` 只包含 method=r1，不能把它作为 r2 的实测结果。

以下是日志所采样时刻的统计，不是所有 2880 步的严格平均：

| 项目（秒） | r1 | r3 | macro |
|---|---:|---:|---:|
| Q matrix + Q value + assignment，采样均值 | 2.54 | 2.19 | 2.43 |
| 完整 simulate_motion，采样均值 | 3.97 | 3.64 | 4.05 |
| 完整 simulate_motion，采样最大 | 24.23 | 10.65 | 10.26 |
| env.step 未列入子阶段的时间，采样均值 | 29.67 | 24.79 | 22.49 |
| env.step 未列入子阶段的时间，最后一个细分样本 | 78.48 | 85.18 | 76.77 |

例如 r3 第 2850 步：simulate_motion 约 1.98 秒，env.step 86.305 秒，而动作执行、环境更新、经验记录等已列项合计仅 1.13 秒；随后 AEV + EV 网络更新 53.01 秒，总步耗时 141.48 秒。

另外 r3 第 410 步 learning_phase=1401.379 秒，但 AEV/EV 更新仅 4.487/33.965 秒；macro 第 850 步 learning_phase=2782.870 秒，网络更新仅 5.391/27.954 秒。两处紧邻多次 best checkpoint 保存。日志没有单独计时每次写盘，因此不能把差额全称为纯磁盘 I/O；其中还包括 replay 展开、哈希、序列化及内存压力等。

## 与 adp_trainer 的区别

对照 `/Users/seinzhou/Desktop/adp_trainer/src/NYCEnvironment.py`：它的 step 在环境更新后直接处理经验，不包含 ICAPS 新增的 `_finalize_joint_collection`。因此沿用旧子阶段日志会漏掉 ICAPS 的联合 transition 构建时间。

adp_trainer 的批量数组接口是可以沿用的计算组织方式；不能直接用它的旧 learner 替换 ICAPS 的 recourse learner，否则 Bellman 目标、跨期链接、r1/r3/macro 定义会变。本次将 ICAPS 的 replay 图特征接到已有批量数组接口，保留联合 TD 定义。

## 修改内容

1. `src/recourse/lifecycle.py`：旧 `outcome_summary` 为每个历史拒单事件再遍历全部历史，计算该订单第一次拒单时刻，复杂度 O(H²)。改为单次遍历求每个订单的最早拒单时刻；有 epoch 过滤时先过滤再排序。该部分变为 O(H + R + K log K)，H 为历史拒单事件数，R 为跟踪订单数，K 为返回的当期候选事件数。事件字段、排序、重复拒单与 epoch=0 的含义保留。
2. `src/recourse/coordinator.py`、`reward_ledger.py`：车辆归属与选中边查询使用集合，避免 O(V²) 和 O(E×V) 的 tuple 成员查询；输出奖励与排序仍按原逻辑计算。
3. `src/ValueFunction_st_masac_gat.py`：在线联合 critic 和目标图估值按 provider 分组、按现有 batch 上限构建边张量并执行网络。每批建立状态视图、车辆索引和在线车辆数；消除逐边扫描车队、逐边设备小张量传输和逐边 online critic forward。fleet-local/strict-local 视图分别处理，原始边序恢复后再求和，在线梯度保留。仍保留目标推理的 OOM 分块重试。
4. `src/recourse/replay.py`：checkpoint 哈希按 transition 顺序处理，直接序列化冻结 dataclass 字段，不将整个 replay 递归 deepcopy 成大字典列表再生成一个巨大 JSON。哈希输入字节与旧版 canonical JSON 相同，兼容旧 checkpoint；内容、条数、保存频率保持。
5. `src/NYCEnvironment.py`、`src/NYCtrainer.py`：新增 `joint_collection` 时间。`JointTrainTiming` 分开报告抽样、online prediction、target evaluation、backward/optimizer，保存到 step timing 数据。CUDA 为异步执行，这些细分是宿主侧墙钟范围，未逐阶段强制同步，不能当作纯 GPU kernel 时间。

没有减少训练频率、batch 中样本预算、候选动作或真实订单；没有改 wait、charge、奖励、SSG 规则、MCMF 量化精度、求解器或最优性要求。没有通过时间截止强制返回近似 assignment。

这里保留的是固定 assignment 问题的全局最优性，不是宣称非凸神经网络训练具有全局最优保证。批量浮点运算允许数值容差内差异，不能承诺以后每次 SGD 或近似平局动作与旧版逐位相同。

## 本地验证结果

原始数据、冻结旧源码和可复现脚本保存在 `results/nyc_speed_audit_20260913/`。

### 长历史统计

同一历史数据调用冻结旧方法与新方法，返回的完整事件对象一致：

| 历史事件数 | 旧方法（秒） | 新方法（秒） |
|---|---:|---:|
| 1,000 | 0.01028 | 0.000089 |
| 10,000 | 0.93599 | 0.000530 |
| 40,000 | 14.99745 | 0.002562 |

这验证了历史统计的平方瓶颈已经移除；不能直接拿本机倍率乘服务器整个 env.step。

### 3000 车带订单的配对运行

相同 seed=901、初始权重、100 个初始合成订单、3 个收集步、单 CPU 线程、OR-Tools。使用实际 NYC 执行与 recourse learner，P0 辅助预测设置；并非服务器已训练 P3 网络的整日回放。订单可在第一步被接受，因此后续并非持续 100 单负载。

| 更新 | 旧方法（秒） | 新方法（秒） | 倍率 |
|---|---:|---:|---:|
| AEV 联合训练 | 1.114 | 0.256 | 4.34× |
| EV 联合训练 | 0.935 | 0.124 | 7.55× |

新版本三次 simulate_motion 分别为 5.713、1.582、1.482 秒；env.step 为 0.364、0.454、0.419 秒。此次修改主要针对回放/学习，因此短历史的 action generation 没有预期的系统性加速；不应比较几百毫秒波动来宣称 assignment 提速。

六个阶段的全部可行图边、Q 分数、structured score 和选中 assignment 完全相同。AEV loss 1829.670532 → 1829.670654；EV loss 676.433121 → 676.433121，均在 float32 容差内。

### checkpoint 哈希

4 条大型 transition、3000 辆车、每图 6000 边的受控比较：旧哈希用时 0.353 秒，新哈希 0.282 秒；独立 tracemalloc 测量 Python 分配峰值由约 83.44 MiB 降至 17.59 MiB，摘要完全相同。这只测哈希阶段，不包含完整 checkpoint 磁盘写入，不保证服务器分钟级停顿全部消失。

## 服务器更新与 30 秒验收

测试：综合 recourse/NYC 回归运行通过 97 项；随后扩展的批量测试文件通过 50 项（包含前次已测项目）。覆盖 6 种状态视图、online/target、基础 critic 与 post-demand critic、缺省及混合 demand 特征；比较特征、预测和 encoder/mixer/twin-critic 全部参数梯度。另验证重复拒单的首次 epoch、按 epoch 过滤顺序，以及 checkpoint 的旧哈希兼容和加载回环。日志保存为 `tests.log` 和 `extended_batch_tests.log`；7 个修改源码编译检查通过。

同步下列文件后重启训练，原命令参数可以继续使用：

- `src/NYCEnvironment.py`
- `src/NYCtrainer.py`
- `src/ValueFunction_st_masac_gat.py`
- `src/recourse/coordinator.py`
- `src/recourse/lifecycle.py`
- `src/recourse/reward_ledger.py`
- `src/recourse/replay.py`

无需升级 NumPy、PyTorch、OR-Tools 或修改 checkpoint replay 策略。

重新观察 `SimTiming`（在线策略到动作）、`joint_collection`（联合经验收集）、`JointTrainTiming`（网络更新）和 `TrainTiming.full_step`（含学习阶段的完整一步）。30 秒要求若指在线 assignment，现有采样证据已经满足；若指最坏完整训练步，当前不能保证，尤其包含 checkpoint 保存时。完整服务器逐步统计、目标订单规模与设备资源竞争仍需实测。
