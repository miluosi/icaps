# NYC real_conservative 训练与评估

正式入口：`run_nyctrainer.py`、`test_nyc_model.py`。新的 `--charging-model` 可取 `current`、`conservative`、`real_conservative`。保留旧 `--conservative-charging` 接口，新字符串参数优先，建议只使用其中一种写法。

`real_conservative` 实例化 `RealConservativeNYCEnvironment`，对 AEV 使用同一运动/充电完成预测、持久的逐 slot 预约、先释放再到达，以及实际到站立即开始充电的严格检查。HEV 保持原行为。要求配置专属 AEV 中心（当前为 3、4、5 个），初始预约必须可行且无 AEV 到站队列。wait 在 Myopic 和 Learning 中均保持可行。

## 训练

```bash
python -u run_nyctrainer.py \
  --methods macro \
  --charging-model real_conservative \
  --aev-charging-center-count 3 \
  --start-date 2025-12-08 --end-date 2025-12-10 --episodes 3 \
  --mcmf-backend ortools
```

正常训练仍每个 episode 独立 reset；latest/best 成对保存。`--methods` 可替换为 `r1`、`r2`、`r3` 等已有方法。

新 checkpoint 目录为：

```text
checkpoints/charge-real-conservative/q_networks…_method-macro_…_ev/
checkpoints/charge-real-conservative/q_networks…_method-macro_…_aev/
```

模型元数据同时保存 `charging_model="real_conservative"` 和兼容布尔字段 `conservative_charging=true`。单独使用布尔字段不能区分两种 conservative，因此加载时核对精确名称，拒绝混合的 EV/AEV 模型及命名空间不一致。旧模型缺少名称字段时，通过原布尔字段恢复 current/conservative。

独立父目录避免 Macro 原有长文件名追加新标记后超过系统单个文件名长度限制。旧 current、conservative checkpoint 的路径不改变。训练统计表位于 `results/nyc_tests/charge-real-conservative/`（非 assignmentgurobi 时为对应 `nyc_tests_h` 子目录）。

## 读取 real 模型并测试

```bash
python -u test_nyc_model.py \
  --methods macro --strategies ADP-MCMF \
  --checkpoint-charging-model real_conservative \
  --load-model-start-date 2025-12-08 --load-model-end-date 2025-12-10 \
  --start-date 2025-12-15 --end-date 2025-12-17 \
  --aev-charging-center-count 3 --mcmf-backend ortools \
  --output-dir results/test_macro_real_conservative
```

测试省略 `--charging-model` 时从 checkpoint 元数据继承。可加 `--checkpoints-only` 先核查读取路径。日期范围多日时自动连续评估，车辆/充电站状态及预约跨午夜继承，网络仍使用训练对齐的日内时间。

训练模型的读取空间与测试环境是两项独立设置：

```bash
# 读取原 current 模型，但用 real_conservative 环境做冻结评估
python -u test_nyc_model.py \
  --methods macro --strategies ADP-MCMF \
  --checkpoint-charging-model current --charging-model real_conservative \
  --output-dir results/test_macro_current_to_real
```

这不是重新训练；结果应标为充电环境改变后的模型迁移评估。继续训练要求环境和 checkpoint 模式一致，避免写回不同语义的模型空间。

测试结果文件名包含 `traincharge-real_conservative_testcharge-inherit`（若显式指定环境则显示具体名称），连续多日再带 `_continuous`。Excel detail/summary 和 NPY 记录精确 `charging_model`、`checkpoint_charging_model`。不同方法并行测试应使用不同 `--output-dir`，避免相同 NPY 文件名覆盖。

## 本地验证

`tests/test_real_conservative_nyc_cli.py` 覆盖新旧模式解析、独立路径、混合 checkpoint 拒绝、6 辆车的实际训练与保存/读取、权重一致性、冻结评估不更新网络；另外用受控充电偏好在两阶段 OR-Tools 派单中触发真实预约，验证有限 slot 下的到站/完成事件与跨午夜继承。

该用例包含乘客需求的短训练；充电预约执行检查采用独立的受控偏好，用于确保事件确实发生，不能视为真实需求下的策略性能实验。没有运行 3000 车完整训练。

结果：`results/real_conservative_interface_check_20260921/summary.json`、`smoke.log`、`regression.log`。
