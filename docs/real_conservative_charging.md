# real_conservative：可执行的 AEV 保守充电预约

原来的 `current` 和 `conservative` 均保留。新增模式使用独立的
`src.real_conservative_charging.RealConservativeNYCEnvironment`，由非学习实验的
`--policies real_conservative` 显式选择。没有修改 run_nyctrainer/test_nyc_model
的默认充电模式，也没有把旧的 conservative 静默替换成新实现。

## 执行语义

- 适用对象：受控的 AEV 专用站。HEV 继续使用公共站和旧仿真逻辑。
- 准入预测和实际充电行驶共享 `charging_motion_step`；只有到达站点坐标才算到站，不再以进入站点所属 zone 代替到达。
- 沿用 NYC 每轮先执行动作再推进时间的约定：在 t 派出即执行第一个移动步，n 个移动步的到站事件是 t+n-1（同地车辆在 t 到站）。预测和执行都按该约定记录，没有人为延迟已到站车辆。
- 服务时长同时考虑计时器和实际达到目标 SoC 的终止条件，和 NYC 的物理充电更新一致。不同 duration scale、最短/最长充电时间不会被当成与实际充电无关的区间。
- 准入仍调用原 `conservative_charge_mask`：为所有潜在候选建立虚拟 per-plug 日历，只有虚拟开始等于到达的候选可被选择。没有因为候选未被选择而重新放宽同一次求解的掩码。
- 所选充电动作将具体 plug、绝对 arrival/start/end 保存为不可变 `Reservation`。下一轮只把绝对区间换算为相对时间，不重新安排、移动已有预约。
- 已承诺车辆不能再次进入待派充电候选、改派其他站或执行其他动作。没有当前准入证书的充电动作报错。初始已有队列或不相容的在途承诺会被拒绝，不能以它们证明无队列。
- 基础环境更新到 t+1 时完成充电并释放 slot；下一轮 t+1 的到站动作随后执行。额外 plug 占用登记将实际 slot 与所选预约逐一对应，检查同刻释放后复用。
- 每轮检查专用站真实队列为空、物理占用与 plug 登记一致、在途车辆与持久预约一致。每次到站、开始和完成均检查绝对时刻。错误直接使运行失败，不以入队回退或过滤记录掩盖。

## 复现实验

```bash
.venv/bin/python -u benchmark_nyc_charging_rollout.py \
  --policies conservative real_conservative \
  --vehicles 1000 --hev 500 --centers 3 --hours 4 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --scenarios burst staggered \
  --output-dir results/nyc_real_conservative_1000_20260919
```

输出目录必须不存在；重新运行请指定新目录。`--resume` 只允许配置和源码哈希完全相同。
可以加入 `--policies current conservative real_conservative` 运行三个模式。
默认仍使用旧的两个模式。原实验和原图仍在 `results/nyc_charging_rollout_1000`，不读写覆盖。

这是充电需求隔离实验：1000 车中 500 AEV、500 HEV；3 个 AEV 站共 150 slot；
初始 30 AEV 正在充电、15 AEV 在途，包含在上述 500 AEV 中；每车最多充电一次。
固定 OR-Tools + SSG 目标（优先接纳、其次行驶时间），不训练、不使用价值网络、不产生载客订单。
两个场景 × 十个配对种子 × 两个模式，共 40 次 4 小时运行。

`plot_real_conservative_charging.ipynb` 读取新实验、校验全部运行后绘图。
除旧的 sessions/events/admissions/timeseries/stations/summary 文件，严格模式还保存
`reservation_audit.json`：所有不可变预约、真实执行事件、期末未完成预约和状态检查次数。
Notebook 独立复核预约不重叠、实际 arrival=start、actual completion=reserved end、
无物理 queue_enter、无 slot 超容量以及 HEV 对照完全相同，保存 `validation.json`。

充电完成记录修正了旧 benchmark 多加一次 epoch 的记录偏差：
stop_charging 回调时 NYC 已经推进了 current_time，新实验直接记录此值。
该改动只修正事件时间标签，不改变旧 conservative 的动作、队列、占用或完成数量。
原有数据文件保持原样。

## 解释边界

这里不通过抬高 penalty 或减小需求获得零队列，也不把原地等待计入站内队列。
出发前等待单独报告；尚未派出和期末在途/充电车辆分别保存。

新模式修正了充电运动终点和执行预约语义，所以吞吐差异不应全部归因为保守筛选。
有限场景零队列是实现校验，不能推成含随机行驶时间、充电故障或未受控外来车辆的无条件保证。
服务收益、全日服务订单和训练策略性能不在本实验范围内。

## 2026-09-19 实测结果

40 次运行全部完成；两个模式各 20 次。旧 conservative 的 20 次结果逐种子复现原实验的初始状态哈希、占用、排队数、等待均值和完成数量。

| 场景 | 模式 | 占用率 | 到站平均等待（分钟） | 平均队长 | 平均完成数 | 出发前等待（分钟） |
|---|---|---:|---:|---:|---:|---:|
| burst | conservative | 90.60% | 1.38 | 2.44 | 412.00 | 81.30 |
| burst | real_conservative | 90.67% | 0.00 | 0.00 | 408.40 | 82.02 |
| staggered | conservative | 86.85% | 0.94 | 1.65 | 361.00 | 43.82 |
| staggered | real_conservative | 86.61% | 0.00 | 0.00 | 357.10 | 44.58 |

严格模式 8472 条实际到站记录全部 arrival=start，其中新获准派出 8172 条、初始确定在途 300 条；初始已在充电的 600 条不计入到站等待分母。物理 queue_enter 为零；所有运行期末在途为零，故没有未到站的已获准车辆逃过到站检查。
7655 条物理完成事件全部与保留的绝对 end 相同，1417 条已开始但尚未完成的会话在期末删失；总共检查 9072 条保留区间的 per-plug 无重叠，执行 9620 次站点状态检查。
未获准派出的 AEV 平均每次 46.4 辆，两个模式相同。这些车不在站内队列中；出发前等待单列。
新模式完成数平均略降（burst −3.6，staggered −3.9），不能声称消除队列必然提高完成量。两个场景的平均占用与原模式接近。
相关测试共 36 项通过；notebook 所有代码单元及独立数据断言执行成功，并检查了生成图像。
结果目录：`results/nyc_real_conservative_1000_20260919/`；主图：`figures/charge_mode_real_conservative.png`。
