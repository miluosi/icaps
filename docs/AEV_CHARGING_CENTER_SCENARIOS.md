# Manhattan AEV 专属充电中心场景

数据配置来自 `manhattan_aev_charging_centers_3_4_5.xlsx`，仓库运行时读取
`nyedata/manhattan_aev_charging_centers.csv`。三种场景是嵌套的：

| 场景 | AEV 专属中心 |
| --- | --- |
| 3 | D1、M1、U1 |
| 4 | D1、M1、M2、U1 |
| 5 | D1、M1、M2、U1、U2 |

每个专属中心的同时充电容量固定为 50 辆 AEV，不乘以
`--station-capacity-scale`。公共 Manhattan 充电站完整保留，Human EV 仍只使用公共站；
选择 3/4/5 场景后，AEV 的充电动作、未来到站占用约束和充电排队只使用该场景的专属中心。

`0` 是兼容模式：不加载专属中心，AEV 保持原来使用公共站的行为。

训练示例：

```bash
python run_nyctrainer.py \
  --methods r0 r1 r2 r3 r4 macro samitha \
  --aev-charging-center-count 3 \
  [其他 NYC 训练参数]
```

测试示例：

```bash
python test_nyc_model.py \
  --methods r0 r1 r2 r3 r4 macro samitha \
  --aev-charging-center-count 3 \
  [其他 NYC 测试参数]
```

分别把参数改为 `3`、`4`、`5` 即可运行三种 AEV 基础设施情景。若需使用另一份同格式配置，
训练和测试命令均可加 `--aev-charging-center-csv /absolute/path/to/file.csv`。
启用场景后，checkpoint 命名会自动加入 `aev-centers-3`、`aev-centers-4` 或
`aev-centers-5`，防止三个场景相互覆盖或在测试时加载错模型。
