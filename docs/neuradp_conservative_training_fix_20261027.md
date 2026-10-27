# NeurADP NYC conservative 训练错误检查（2026-10-27）

## 日志结论

输入日志：`nyc_neuradp_conservative_train.log`。配置为 3000 车、1500 HEV、
3 个 AEV 专用站（每站 50 slots）、NeurADP、`real_conservative`、OR-Tools exact MCMF。

第一天 2025-12-08 已完成并成功保存 latest/best 推理检查点；第二天 2025-12-09
在约 4:20 后的 integrated dispatch 执行转换中停止：

```text
src/recourse/integrated_repair.py, _action_target
list(env.hotspot_locations).index(edge.target_location)
ValueError: 119 is not in list
```

这次直接停止的原因是动作目标转换错误。日志没有 CUDA OOM 或 AEV 预约不变量报错。
日志中的 394 条 `added to queue` 事件，车辆 ID 全部小于 1500；此配置下它们均为 HEV，
不是受 real_conservative 预约约束的 AEV。

此外日志存在明显 RAM 增长：step 0 为 1.05 GiB，step 500 为 11.90 GiB，
step 2000 为 49.03 GiB，step 3400 为 73.07 GiB；后者 CUDA allocated 仅 0.03 GiB。
joint replay 已采用 disk，但旧版逐车 replay 仍积累数据和历史图引用，因此 disk replay
本身不足以消除这条 RAM 增长路径。

## 修改

1. HEV 默认 outside action 的目的地可能是其当前非 Manhattan 热点区域，例如车辆从
   公共充电站出发，或者低电量时留在该区域。它不要求属于 AEV 的 hotspot 列表。
   integrated executor 现在通过 `reloc_to_<zone_id>` 执行图中已采样的具体目标；
   不重新采样，不把合法动作换成其他动作。AEV 原有热点索引接口保留。
2. 这些 HEV 动作写入 `storeactions_ev`，避免写入 AEV 动作缓存而留下额外历史引用。
3. NeurADP 声明 `collects_legacy_experiences = False`。NYC 每步跳过其不使用的逐车
   experience 构造/过滤，底层 storage 接口也拒绝保存这些未使用的 legacy rows，
   并解除临时 action context。未使用的 queue-predictor replay 同样不再积累。
   NeurADP 的 TD 更新只使用 joint replay，原有容量、磁盘存储、优先级、采样及
   训练目标不变。其他 learner 默认继续原有收集行为。

未修改 conservative 准入/执行规则、可行图、目标分数或 OR-Tools/SSG 求解逻辑。
修复适用于复用该 NYC integrated executor 的方法，不仅限于 NeurADP。

## 验证

- 新增用例先复现原来的 `119 is not in list`，以及 NeurADP 无用旧版 replay 的积累。
- current / real_conservative × SoC 0.2 / 0.8：构造非热点 119 的 HEV outside
  action，连续执行实际派单和运动，检查图中目标与实际目标一致、动作写入正确 fleet
  缓存、joint replay 正常收集、TD 更新有效。
- real_conservative、独立双 critic、OR-Tools + SSG：6 车 12 步训练成功，joint replay
  保留 12 条；旧版逐车/queue replay 均为空；保存两组 best 检查点并重新读取评估，
  参数逐项一致，冻结评估无优化器更新。
- 回归范围：NeurADP、integrated repair、HEV 充电执行、磁盘 replay、
  conservative checkpoint 接口、评估内存、NYC 分数一致性和严格预约执行。
  共 103 项测试通过；`git diff --check` 通过。

小规模验证不等同于完整 3000 车三天运行，不据此承诺 RAM 峰值或运行时上界。

## 服务器同步

需同步以下运行代码后，使用原来的训练命令重新启动：

```text
src/recourse/integrated_repair.py
src/NYCEnvironment.py
src/ValueFunction_neuradp.py
src/ValueFunction_st_masac_gat.py
```

现有第一天的推理检查点仍可用于 Test。它们明确不保存 replay/optimizer，不能据此
声称可以精确恢复中断前的完整训练状态。该修复仅在本地完成，不表示已部署到服务器。
