# 数据说明

项目自带 `nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet`，它从 NYC TLC 2025-12 Yellow Taxi 月度文件中按上车时间截取 2025-12-18 08:00–10:00，共 14,772 条记录。样本保留原始 20 列 schema，用于快速验证真实分区、订单生成、充电站映射和训练循环，不用于正式统计结论。

`nyedata/nye_simulation/taxi_zones.geojson` 是 TLC taxi-zone geometry；`nyedata/nyc_all_charging_stations.csv` 是 NYC 充电站输入。两者都是 NYC 环境运行所需的最小空间数据。

正式实验可通过 `--parquet-path` 指向一个文件，或用逗号分隔多个 Yellow Taxi parquet。若研究 Yellow Taxi 与非拼车 HVFHV 的合并需求，再增加 `--full-demand --hvfhv-parquet-path ...`。大数据、下载缓存和预处理产物不提交到本项目。

示例样本窗口对应：

$$
\mathcal{D}_{\mathrm{sample}}
=
\left\{r_i:\ 2025\text{-}12\text{-}18\ 08{:}00
\le t_i^{\mathrm{pickup}}
<2025\text{-}12\text{-}18\ 10{:}00\right\}.
$$

