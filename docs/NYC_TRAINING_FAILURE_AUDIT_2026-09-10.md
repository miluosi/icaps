# NYC 训练日志核查与 OR-Tools 默认配置（2026-09-10）

本次检查的服务器日志为 `nyc_r1_r2.log`、`nyc_r3.log`、`nyc_macro.log`。
日志已越过 NumPy、SciPy、PyTorch 导入并执行环境步；停止原因是容量模型不一致和训练显存耗尽。
本次代码修改统一 assignment 默认后端；下述两处训练故障尚未修复，不能把修改后端当作故障修复。

## 日志结论

| 方法 | 直接证据 | 结论 |
| --- | --- | --- |
| R1 | `nyc_r1_r2.log:157–176`，`station 229902`，count=4，capacity=0 | AEV 分配完成后，回放图的容量校验失败 |
| R2 | 文件仅出现 `ICAPS method=r1`，没有 R2 启动标记 | 同一进程按顺序运行，R1 异常导致 R2 尚未开始；不能据此判断 R2 有独立故障 |
| R3 | `nyc_r3.log:146–215`，`_train_joint_step → _selected_raw_tensors → _graph_context` | EV 联合价值网络训练 CUDA OOM |
| macro | `nyc_macro.log:146–215`，相同调用链 | 与 R3 相同的训练内存问题 |

R3/macro 日志各报告进程占用 23.50 GiB，PyTorch 已分配 21.32 GiB、预留未用 1.73 GiB，GPU 总容量 23.52 GiB。
这是已有张量/计算图占用导致的显存不足，不能把 allocator 碎片化提示当作唯一原因。
日志中的 `GPU 0` 是进程可见设备编号；仅凭日志不能判断几个进程是否使用同一张物理显卡。

## 容量错误的可复现原因

- `NYCEnvironment.generate_vehicle_chargerange` 生成到达时刻和充电持续时间，保存 `_last_expected_charge_expansion`。
- `GurobiOptimizer._build_exact_mcmf_inputs` 在存在该时间窗元数据时，将站点的聚合动作容量设为车辆数，再用 `_expected_charge_edges_to_disable` 检查未来各 epoch 的占用。
- `StateSnapshotBuilder.feasible_graph_from_matrix` 却给同一充电动作填写当前 `remaining_admission_capacity`。
- `RecourseTargetBuilder.verify_feasible` 按这个当前容量计数，因此“现在已满，但车辆到达前会释放”的站点能通过求解、随后在校验时失败。

本地构造了四车、四槽位、当前满站但第 3 个 epoch 已空出的最小案例。分别调用 `primal_dual` 和 `ortools`，两者均分配四个未来可行的充电动作，随后都复现：

```text
joint action exceeds ('station', 229902) capacity: count=4, capacity=0,
vehicles=[1586, 1796, 2071, 2147]
```

这是与日志一致的故障机制复现，不是从服务器恢复出的精确崩溃快照。
后续应让 rollout、图序列化、回放 target 和可行性校验共享同一容量语义，保留到达/充电时间窗。
不能直接跳过断言或把回放容量设为无限大；这会丢失容量约束。
另外，当前未来容量冲突处理按机会成本删除动作边，`exact` 只保证剩余流图上的最优性，不能据此证明完整时间窗分配问题的全局最优性。

## 显存错误的代码原因

`ValueFunction_st_masac_gat._selected_raw_tensors` 对每条选中动作都把 provider 的 `_graph_cache_key` / `_graph_cache` 清空。
随后 `_edge_raw_tensors` 再调用完整 `_GraphAttentionEncoder`，每条动作都保留独立的反向传播图，直到联合 TD loss 最后统一 `backward()`。
`_selected_correction_tensors` 也存在同样模式。

本地以 69 个 relocation 区域、425 个站点、1 个全局节点构造 495 节点图，并对真实 post-demand-direct 价值网络加 forward hook：

| 选中动作数 | 同一张图的 encoder forward 次数 |
| --- | --- |
| 1 | 1 |
| 8 | 8 |

对应日志是 3000 车、1500 EV、425 公共站；`AEV charging centers=0` 表示使用公共站，不是零站点。
`gat_neighbour_number=0` 只关闭车辆邻居模块，`iftransformer=False` 关闭路径 transformer；两者都没有关闭这里的全局图注意力。

后续修复应在同一图、同一 provider、同一梯度模式的一次联合计算内复用图编码，并验证输出及梯度等价；必要时对联合样本分批累计梯度。
仅减小普通 batch size 不保证解决：一个联合样本本身就包含上千车辆动作。
`--mcmf-use-cpu` 控制 assignment，不会把价值网络训练移到 CPU。
PyTorch 内存管理说明：<https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management>。

## 已完成的默认后端修改

- NYC/合成训练、模型评估、recourse day/panel/audit、assignment scalability/solver audit、敏感性实验、接受率模型和相应辅助入口：默认 `ortools`。
- 环境和 trainer 的 Python API 默认 `exact + ortools`；合成环境不再因未传 `mcmf_solver` 而走 legacy。
- `run_nyctrainer.apply_paper_parameter_preset` 不再覆盖 `--mcmf-backend`，因此论文预设也默认 OR-Tools，显式指定其他后端仍有效。
- 新训练的 target 默认沿用 rollout backend。历史明确固定 `primal_dual` 的回放策略保留原语义。
- `solve_exact()` 默认直接调用 OR-Tools；显式 `auto` 优先尝试 OR-Tools。
- Q scale 10000 和 exact SSG 保持启用；默认严格模式不会因 OR-Tools 缺失而静默换为 legacy。
- OR-Tools 仍固定 9.14.6206，兼容此前修复的服务器 NumPy 1.26.4。数值库线程控制要求 `threadpoolctl>=3.6`。
- 显式 CPLEX/Gurobi/启发式基线和四算法比较的各方法仍可运行；默认 MILP 线程数改为 1。

检查了 16 处 assignment 后端 CLI 默认值；独立的固定图多后端审计继续默认比较其三种指定后端。
回归测试覆盖默认实际 OR-Tools 求解与 primal-dual 最优目标一致、论文预设下七方法的参数传递、显式后端保留、target projection、recourse 和 assignment 约束。

## 服务器使用

修改位于本地工作区，尚未同步到服务器，未重启服务器训练。
已修复的服务器环境本身已具备 OR-Tools 9.14.6206、NumPy 1.26.4、Pandas 2.1.4、threadpoolctl 3.6.0，无需再次执行不带版本的 `pip install ortools`。
同步补丁后，下面的默认配置即可选择 OR-Tools：

```bash
python -u run_nyctrainer.py --paper-parameter-preset --methods r1 r2
```

但应先修复上述容量与显存缺陷，再恢复完整训练；本次小规模求解与配置检查不代表三日训练已通过。
