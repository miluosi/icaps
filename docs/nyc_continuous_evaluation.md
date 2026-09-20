# NYC 连续测试与统计口径

`test_nyc_model.py` 对多日 date range 默认只初始化一次，每个方法/策略/种子仍是独立实验。首日随机生成车辆状态；后续日期继承车辆位置、电量、在途订单、充电进度、到站队列和在途预约。日工资/司机每日退出等机制仍按原有日边界规则执行。训练默认仍按 episode reset。

时间分两种：

- 仿真内部的 `current_time` 连续，用于行程、订单截止时间和充电预约，防止午夜把已有预约变成过去时间。
- 网络输入的日内时间每天循环；旧 Q 网络的标量/批量时间特征先转换为当日时间，再应用 checkpoint 的训练窗口偏移，且归一化分母保留单日模型长度。GAT 的 hour-of-day 同样每日循环。

24 小时窗口的三日测试共 8640 个 30 秒步骤，第二天 00:00 的模型时间为当天 0，不是 2880，也不是训练结束的常量。部分日内窗口（如 12:00–17:00）只在该窗口生成需求；跨夜期间仿真仍推进车辆与充电，不重新随机状态。测试窗口应与模型训练窗口一致。

纯 Myopic (`adp_value=0`，含 Myopic r1/r2) 在 `battery < min_battery_level` 时 wait feasibility 为 0，等于阈值时仍可等待。Learning 各方法保留 wait fallback（包括 AEV follower 无学习的 Learning r2）。硬禁 wait 可能在没有其他可行动作/充电竞争时使 assignment 不可解；没有偷偷恢复 wait。

## Excel / NPY

连续测试输出名增加 `_continuous`，不会覆盖此前独立逐日的原始 NPY。Excel 名仍包含时间戳。

- `episodes`：覆盖日数；`rollouts`：连续轨迹数（每个种子 1）。
- `avg_reward` / summary `mean_reward`：每日日均 reward；`total_reward`：全日期区间之和。
- `complete`、`total_orders`、`recourse_requests`、`lost_requests`：整个日期区间计数；summary 的 `mean_*` 为跨种子的这些区间计数均值。
- `mean_ev_completed_orders`、`mean_aev_completed_orders`、EV/AEV reward、`finished_charge`：按日均值。
- 服务率：整个轨迹完成数/生成数。价值均值、平均等待时长不再除以天数。
- `daily_evaluation_json`：逐日 reward 台账；最后电量为该日边界的车辆平均电量。

队列每个决策步采样，先对全网络求和，再对时间取均值（含 EV 与 AEV，各站车辆去重，剔除充电中车辆）：

- `avg_queue_length_waiting`：已到站但未开始充电的车辆。
- `avg_queue_length_reservations`：已确定赴站、尚未到站的预约车辆，不包含潜在候选车辆。
- `avg_queue_length_including_reservations`：上述两项之和，不是平均每站长度。

小时/区域 sheet 中 `mean_queue_vehicle_count` 同样包含预约，并同时输出 `mean_waiting_vehicle_count` 和 `mean_reservation_vehicle_count`。这些指标不能当作“到站后实际排队”的证据。

Recourse 按唯一订单追踪，跨午夜不清空：

- `recourse_assigned_requests`：EV 拒绝后，同轮指派给 AEV 的订单数。
- `recourse_completed_requests`：上述订单中，截止测试结束由该指派 AEV 完成的数量。
- `recourse_uncompleted_requests`：两者差，包含测试结束时仍未完成者，不等同于最终失败。
- `recourse_success_rate = recourse_completed_requests / recourse_assigned_requests`；`_pct` 为百分数。分母为 0 时留空。summary 将各种子的分子分母分别相加后取比值，同时输出这两个汇总计数。
- `recourse_recovery_share = recourse_requests / (recourse_requests + lost_requests)`：保留此前图中提出的比例；它与实际完成成功率不同，分母不必等于所有被拒绝订单。

历史 Excel 无法凭已有同轮指派数恢复最终完成成功率，需要重新运行测试。
