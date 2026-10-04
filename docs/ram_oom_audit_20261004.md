# Host RAM OOM 核查与无损存储修改（2026-10-04）

## 主要发现

此前服务器日志已经确认 host RAM OOM。本次没有连接服务器复现该进程，但源码中存在足以解释随训练增长的内存结构。`torch.cuda.empty_cache()` 不释放这些 CPU 活对象。

| 路径 | 原有 RAM 风险 | 本次处理 |
|---|---|---|
| `ValueFunction_st_masac_gat.py` / `recourse/replay.py` | 默认最多 100,000 个联合 transition，每个包含完整候选图与状态。按条数有界不代表按字节足够小。 | 大车队使用无损磁盘 replay，完整样本总体、容量、优先级、PER 概率、beta 和 RNG 不变。 |
| `store_experience` | 逐车辅助 experience 还引用整图，即使联合 replay 落盘，旧图仍被 RAM 引用。 | 对实际只用 joint TD 的 learner 去除辅助行中未使用的整图引用；保留数值特征和原有行数，queue/post-demand 专用样本不变；legacy edge TD 分支不删必需图。 |
| `sample_ready` / `has_trainable_joint_rows` | 抽样前扫描完整图；抽中 64 条可能一次加载 64 个巨图，尽管训练稍后会按原有 edge budget 提前停止。 | 索引头判断 readiness；抽样 payload 惰性读回。没有缩小 batch、样本总体或 target graph。 |
| `FeasibleEdgeSnapshot` | 每条候选边有 Python 实例字典；重复 deepcopy 开销大。 | 使用 slots，完整字段、浮点精度、枚举和 tuple 不变；不可变边共享 deepcopy；兼容读取旧字典形式的边 pickle。 |
| `target_components_for_graph` | 保留旧权重版本的 target 结果，之后不会命中。 | 模型版本签名变化时清除失效结果；同版本的正常缓存仍保留。 |
| joint training diagnostics | 10,000 条诊断各保留整车队 selected-edge ID tuple，可另外累积数千万字符串。 | 磁盘 replay 模式诊断保存数量与精确 JSON 哈希；完整 IDs 仍在对应 transition 中。 |
| `NYCEnvironment._load_demand_data` | 读取指定日期范围的完整 DataFrame；多文件 frames 和 concat 结果同时存在；类级缓存长期持有不同日期/区域组合。 | concat 后及时释放输入 frames；跨环境复用缓存最多保留两个组合，活跃环境自己的数据不受影响。保留日期过滤、需求数量和全局站点选择口径。 |
| `generate_vehicle_qvalue` | 全部边的多组 NumPy parts 与 concatenate 结果同时被引用。 | 拼接完成后释放 parts 列表引用；没有改变 dtype、完整候选集合或评分。单次完整矩阵本身仍需 RAM。 |
| `run_recourse_audit.rollout` | `list(joint_replay_buffer)` 会重新把所有历史图读进 RAM；action trace 列表持续累积。 | 单遍流式聚合；action trace 使用与旧 JSON 字节一致的增量哈希，不再保留整串 ID 历史。 |
| 显式 `checkpoint_replay=recent/full` | 组装整个 replay 列表、哈希/序列化可能造成保存峰值。 | 磁盘模式将已压缩记录流式复制到持久 `.tar` sidecar，模型 checkpoint 保存索引与校验和。读取时逐条恢复，不一次展开整段历史。 |
| episode/方法切换 | 有环对象、过期张量缓存、上一方法的 env/model 可能拖到下一模型创建后才释放。 | episode 边界清理可重算缓存并 GC；Linux 可用时调用 malloc_trim；切换方法前解除旧 env/results 局部引用。CUDA 清理仅在已初始化时执行。 |

当前注册的 residual/full-Q learner 共用 GAT 基类，因此两者均采用上述 replay 改动。默认 NYC 训练没有 DataLoader 多 worker 预取：旧 Bayesian rejection learner 的 DataLoader 未设置 num_workers（默认 0）。两个独立检查脚本存在 ProcessPool，不能因此归因于当前训练 DataLoader。未改变这些独立脚本的并发设置。

主路径损失历史存的是浮点数，没有发现将所有 epoch 的 loss 计算图追加到历史列表的模式。joint TD 更新内部保留当前 batch 的反向图是必要计算，不作为历史缓存删除。

## 如何生效

同步全部本次代码后，新进程默认 `ICAPS_REPLAY_STORAGE=auto`：环境车辆数 **>=500** 使用磁盘 replay，小实例使用原内存 replay。服务器可在原启动命令前设置：

```bash
export ICAPS_REPLAY_STORAGE=disk
export ICAPS_REPLAY_DIR=/home/syejiang/icaps/replay_cache
export ICAPS_REPLAY_CACHE_ROWS=2
export ICAPS_RAM_LOG_EVERY=100
# 然后执行原来的 python/nohup 训练命令。
```

目录应在本地 NVMe/SSD，不放 `/dev/shm`、tmpfs 或网络盘。默认目录为工作目录下的 `results/.replay_cache`。目录按进程/缓冲区隔离，写入失败会明确抛错，不用空样本或零分替代。正常退出会移除临时训练 spool；SIGKILL 后的残留只能在确认对应进程已结束后清理。持久 checkpoint sidecar 不自动删除。

热缓存默认 2 条，清缓存不删除磁盘样本。`ICAPS_REPLAY_STORAGE=memory` 可显式恢复旧内存模式。正式多方法运行的 manifest 会记录 storage/cache_rows。运行中的旧进程不会自动切换到新代码。

每 100 步打印 `[RAM] ... RSS=... GiB; joint replay disk:... rows`。可自行按每个进程的服务器预算设置 `ICAPS_RAM_LIMIT_GB`；默认 0 表示不设置任意硬上限。超过指定值先清可重算缓存，再抛出清楚的 MemoryError，避免悄悄删样本改变训练。它只在监测点检查当前 RSS，不能保证拦住两次检查之间的瞬时峰值，也不是整机/cgroup 总内存上限。

## 检查点与精度

- 默认 NYC episode-end/best **inference checkpoint 不保存 replay**，原 Test 路径与格式不变，不依赖临时 replay 目录。
- 仅显式保存 `recent/full` replay 时新增持久 sidecar。`joint_replay_state_dict.disk_replay_archive` 记录绝对路径，`disk_replay_sha256` 验证内容。**跨机器恢复训练必须同时复制 sidecar 并更新该路径；不能只复制主 `.pth`。** 在磁盘模式下恢复。新代码仍支持旧内存 replay 状态的读取，但旧大 checkpoint 的 Python 对象已在 torch.load 时展开，这一历史格式不能被 mmap 自动变成按需 Python 对象。
- 训练数据不是重新生成、近似重建或缩短窗口；磁盘序列化保留完整原始字段。没有 float64→float32/float16 的额外转换。
- loss、梯度计算、优化器步数、目标网络更新、抽样规则、候选边、SSG 及 OR-Tools cost scaling 均未修改。
- 固定输入分数下 assignment 的最优性不因存储位置变化；这不等于对学习所得策略的全局最优保证。

## 本地验证与速度代价

相关回归 **191 项通过**；RAM 专项 **27 项通过**。最后补充保存/加载与环形缓冲区覆盖位置修复后，重跑相关 **53 项，全部通过**（不是额外的独立 53 项）。覆盖：

- 相同 PER indices、概率、importance weights、优先级、beta/RNG；容量覆盖淘汰与 next-transition 链接。
- integrated / EV-first / AEV-first、r1/r2/r3/r4/Macro 的 readiness 与原判断一致，扫描不触发磁盘 payload 读取。
- 非 terminal successor 投影和训练：RAM 与磁盘版本连续更新后 CPU 参数逐位相同，TD loss 和优先级相同。
- 惰性采样、热缓存清理不丢样本，磁盘写失败不破坏原记录。
- 旧边 pickle、新边 pickle、recent/full sidecar、torch checkpoint 和 replay.save/load；损坏归档拒绝读取。
- 完整 replay checkpoint 保留环形覆盖位置，恢复后继续写入不会错误地从第零个位置开始淘汰。
- 部署/replay 评分、精确求解、现有训练与检查点回归。

`scripts/benchmark_replay_ram.py` 在独立子进程比较 **40 个快照 × 4000 条边**。这是合成存储微基准，不是 3000 车完整训练基准：

| 方式 | 保留数据新增 RSS | 总写入时间 | 平均每快照写入 | 读回并遍历 8 个样本 |
|---|---:|---:|---:|---:|
| 原字典式边对象、RAM replay | 77.05 MiB | 0.00014 s | 0.0034 ms | 4.17 ms |
| slots 边对象、RAM replay | 69.84 MiB | 0.00010 s | 0.0024 ms | 1.04 ms |
| slots 边对象、磁盘 replay | 5.70 MiB | 0.32387 s | 8.10 ms | 58.73 ms |

磁盘版在第 21/40 个快照的 RSS 都约 0.2747 GiB；这一微基准的 replay 增量内存较旧版下降约 **92.6%**。40 个压缩记录合计约 2.53 MiB。实际候选边数量、字段内容与磁盘会影响压缩率和性能，不能把这些数字直接外推为正式训练的峰值。

**Inference 不查询 replay，评分与求解热路径不新增磁盘读回。训练有额外序列化/I/O 成本，不能承诺零降速。** 大图的求解时间与图构造时间可能远大于 I/O，但服务器完整训练降速比例尚未实测。推荐本地 SSD 和小热缓存，不为提速重新一次性装载全部历史。

结果与日志：`results/memory_audit_20261004/`。复现存储基准：

```bash
MPLCONFIGDIR=/tmp/icaps-mpl .venv/bin/python scripts/benchmark_replay_ram.py
```

## 仍需容量预算的部分

本次消除了主训练路径“完整历史图常驻 RAM”的增长源，**没有证明任意规模或任意并发任务都不会 OOM**。仍需 RAM 的有当前完整候选/Q 矩阵、当前图、训练计算图、模型/优化器、当日活动请求和统计。超大单图、无日期限制的需求读入、旧巨型 checkpoint 的一次性反序列化、旧未注册 learner、自行 list(replay) 或多进程同时占满内存仍可能超预算。

未清空学习 replay、裁剪请求或降低数值精度来掩盖这些峰值。应结合服务器 `[RAM]` 日志确认多进程合计占用，并给进程留出当前图与序列化的余量。没有在服务器修改 cgroup、swap 或停止任何训练进程。
