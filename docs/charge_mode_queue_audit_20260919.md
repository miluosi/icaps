# charge_mode.png 队列口径核查（2026-09-19）

核查图：Downloads/charge_mode.png 与本地论文 ADP_ICAPS_9_19/AnonymousSubmission/charge_mode.png 的 SHA256 相同。图的布局与曲线对应 plot_nyc_charging_rollout.ipynb 的 AEV 时间序列图；原始数据目录为 results/nyc_charging_rollout_1000。

## 统计对象

- 1000 辆总车数包含 500 HEV、500 AEV。图取 `FLEET = "AEV"`，统计 3 个 AEV 专用站、共 150 个 slot。
- `benchmark_nyc_charging_rollout.py` 的 update 包装函数对专用站的 `len(station.charging_queue)` 求和。不是 `charging_queue_notarrived`，也不是 `pending_dispatch`。
- 因此没有包括未获准派出的车辆、还在路上的车辆或公共站 HEV。直接复核 queue_enter 事件，在三个 AEV 站未发现 HEV 入队。
- 采样点在动作执行之后、环境充电进度更新和释放之前。曲线为每时刻 10 个种子的均值；阴影为 95% Student-t 均值置信区间。

## 从 sessions / admissions / events 独立复核

以下是每个场景 10 次 conservative 运行合计的车辆会话数，不是同一时刻的队长，也不是全部场景中的唯一车 ID 数：

| 场景 | 到站排队的 AEV | 新派出且有 admission 记录 | 初始已承诺在途 | 未获准派出 | 有正的原始 arrival-to-start 时间差 |
|---|---:|---:|---:|---:|---:|
| burst | 2742 | 2740 | 2 | 0 | 2742 |
| staggered | 2197 | 2197 | 0 | 0 | 2197 |

所有 4937 个新派出排队会话均有 conservative 派出时的预测记录；实际到站比预测提前 2–6 个 epoch，中位数 4 个 epoch。每个 epoch 为 30 秒，即提前 1–3 分钟，中位数 2 分钟。不能以“同一个 epoch 内瞬时入队后立即释放”解释这些正等待。

例：burst / seed 0 / AEV 545 / 站 9000002。在第 172 步派出，预测第 183 步到站，实际第 179 步到站，第 183 步开始充电，实际排队 2 分钟。

## 是否执行完整预约规则

准入端确实开启 conservative：运行设置 env.conservative_charging，generate_vehicle_chargerange 调用 conservative_charge_mask，将当前充电、站内排队及已承诺在途作为背景，再对潜在候选构建虚拟日历。

但执行端没有实现论文命题要求的完整预约语义：

1. _register_aev_notarrived_reservation 仅保存车辆 ID；没有把被选虚拟 plug 和绝对服务区间锁定为后续决策不可改变的承诺。
2. get_expected_station_occupancy 从当前车辆状态重新计算在途时间并重建背景日历。
3. calculate_expected_entercharge_station_time_battery 按区域间行驶时间预测；_execute_movement_towards_charging_station 在车辆进入站点所属 zone 时就尝试充电，不要求行驶到预测目标坐标。因此实际到达可早于预约计算时间。
4. 站点实际调用 start_charging 按 available_slots 接纳车辆，满位则加入物理队列，不执行虚拟 per-plug 预约表。动作执行先于环境更新，也须与命题的同刻先释放后到达约定对齐。

本次复核的 benchmark、conservative filter、GurobiOptimizer 文件与原实验 manifest 哈希一致；NYCEnvironment 已有后续改动，哈希不同。因此原实验事实以保存的事件和 admission 为依据；不能将旧图当成当前所有代码修改后重新运行的验证。

## 论文结论

图支持“当前仿真实现下，保守准入显著减少 AEV 到站排队，充电占用与完成数接近”。不能支持“该实验已验证到站即充”。命题以完整且被遵守的预约、确定的实际到达与服务时长、同刻先释放后到达为条件；该实验没有满足完整条件。不能通过过滤掉新派出排队车辆来获得零队列曲线。

建议补充的英文说明：

> Queue length counts physically arrived AEVs waiting for a plug across the three dedicated stations; it excludes HEVs, vehicles awaiting dispatch, and vehicles still traveling. Curves show means over ten paired seeds, with 95% Student-t confidence intervals. Nonzero conservative queues occur under the simulator's existing execution semantics: arrivals can precede the times used by admission, and virtual plug reservations are not retained as binding service commitments. The figure therefore evaluates queue mitigation, not the queue-free guarantee under the proposition's reservation assumptions.

若要验证命题，应先统一准入与运动的到达/充电完成时间、持久保留和执行所选预约，并规定同刻释放先于到达，再对所有新获准 AEV 检查实际到站到开始充电的时间差。原实验和原图保留，新增实验应另存结果。
