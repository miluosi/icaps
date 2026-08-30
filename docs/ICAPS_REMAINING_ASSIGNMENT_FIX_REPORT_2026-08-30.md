# ICAPS remaining assignment fixes：逐条修改与本地验证

日期：2026-08-30  
审计基线：`6f6848e5ea39eb40d33d2d2080ea138cdd29cf79`  
输入审计：`ICAPS_REMAINING_ASSIGNMENT_ISSUES_AND_PATCH_PLAN_2026-08-30.md`  
参考脚本：`apply_remaining_icaps_assignment_fixes.py`（只读参考，未直接执行 `--apply`）

## 1. 逐条对照

| 审计项 | 状态 | 修改/证据 |
| --- | --- | --- |
| P0-1 实际 graph scoring 绕过 masked state | 已修复 | `src/ValueFunction_st_masac_gat.py` 的 online/target 路径均传递 experience 中已遮罩的 `state_snapshot`；严格 fleet-local 回归测试验证每次只看本 fleet。 |
| P0-2 主因果对照使用 learned R1 | 已修复 | 主路径改为 C0 structured no-repair → R2 → R3 → Macro → R4；learned R1 只保留为 diagnostic contrast。 |
| P0-3 R3/Macro/R4 predictor control 不一致 | 已修复 | `CAUSAL_PREDICTOR_VARIANTS` 成为唯一共享集合，value function、trainer、ADP trainer 与 post-demand trainer 均使用相同冻结 P0 predictor 规则。 |
| P0-4 panel duplicate keyword | 已修复 | panel row 先展开 testing，再以显式 method/seed/day identity 覆盖；专项测试包含 testing 自带 `method` 的情况。 |
| P0-5 panel 错误独立性单位 | 已修复 | 独立模型簇为 `(seed, train_day)`；held-out day 是簇内重复；treatment-control 先按 `(seed, train_day, day_id)` 配对，再按 fitted-policy cluster 计算标准误和区间。 |
| P0-6 Integrated/Samitha 被错误标为 auction | 已修复 | solver audit 对这两个 exact-only stage-0 架构的 auction 请求 fail closed；不伪造尚不存在的 approximate adapter。 |
| P0-7 当前 HEAD 没有 CI 结果 | 本地 gate 已通过；远端待提交 | `.github/workflows/ci.yml` 已覆盖 Python 3.11/3.12 和 assignment matrix；本次未替用户提交/push，因此不能声称 GitHub Actions 已绿色。 |
| P1-1 event contract 不要求物理机制 | 已修复 | learned R1 要求 follower optimizer 与 learned score 均发生变化；R3/Macro/R4 要求 eligible rejection 和真实 same-epoch repair；Samitha 要求真实 repair assignment。 |
| P1-2 formal panel 默认 required contract | 已修复 | panel 默认 `event_contract_mode=record`，保留零 exposure clusters，不按随机机制是否出现来筛选样本。 |
| P1-3 day checkpoint 不保留 replay graph | 已修复 | day/audit/all-model runner 支持 `--checkpoint-replay none/recent/full`；正式默认 `recent`，并保存 replay RNG、priority、beta 和 transition graph。 |
| P1-4 state audit 只是静态 leakage audit | 已修复 | 保留 `run_assignment_state_audit.py`；新增 `run_assignment_state_experiment.py`，用同一 panel 对每个 state variant 重新训练/测试 Macro。 |
| P1-5 model mutation hash 不完整 | 已修复 | 哈希包含 learner identity、shape、online/target critics、graph encoders、mixers、actor、queue/post-demand predictors 和 `log_alpha`。 |
| P1-6 CRN 未审计完整事件流 | 已修复 | offer 之外记录 charge decision/station/reward、charge utility shocks、pickup/dropoff/movement reward、relocation destination 等事件键随机值；audit 对不同方法的共同事件逐键比较，差异立即失败。 |
| P1-7 缺少 fixed-graph solver audit | 已修复 | 新增 `run_fixed_graph_exact_solver_audit.py`，在不可变 replay graph 上比较 backend/reduction 的可行性、量化目标、选择 trace 与运行时间。 |
| P1-8 raw recourse variant 混入 method name | 已修复 | raw choices 只含真实 variant 及明确的 EV-first alias，不再接受 `no_repair`/`samitha` 作为 raw variant。 |
| P1-9 无 resume equivalence | 已修复 | 新测试比较 4 次连续更新与 2 次更新→保存→恢复→2 次更新；sampled transition/replay index、selected edges、PER、optimizer、online/target/auxiliary tensors 和诊断完全一致。 |
| P1-10 panel 重复训练同一 fitted policy | 近端检查已修复 | 对重复 `(seed, train_day, method)` 强制 trained-weight hash 一致。审计文档列为“preferred”的 train-once/evaluate-many 重构仍是后续基础设施优化，本次未改变实验目录协议。 |
| P1-11 tensor load 前未验证 identity | 已修复 | schema、双 learner、method、operating mode、repair policy、leader credit、state/learner variant 和完整 solver configuration 均在 load tensors 前 fail closed。 |
| P1-12 audit scope 标签不准确 | 已修复 | `run_recourse_audit.py` 输出 `evaluation_scope=same_day_new_seed_checkpoint_check`，不再作为 held-out-day 泛化证据。 |
| P2-1 objective wording | 已整理 | objective policy 固定为 `execution_reward_with_separate_service_loss_metrics`；丢单/过期单单独报告，不声称 reward 已包含所有 lost-order economic costs。 |
| P2-2 canonical aliases | 已修复 | `test_all_nyc_models.py` 直接复用 `METHOD_ALIASES` 和 `canonical_method`，不再维护另一套相互漂移的 alias 语义。 |
| P2-3 legacy evaluators | 已标记 | `test_nyc_model.py`、`test_model.py` 明确标为 legacy solver transfer / online adaptation，不作为固定策略 recourse 主表入口。 |

审计文档明确列为需要额外建模或基础设施决定的事项未擅自实现：Integrated/Samitha approximate stage adapter、lost-order economic penalty、panel train-once/evaluate-many 大重构、最终 seed-day cluster 数量、论文正文改写。

## 2. 新增/更新的验证入口

- `tests/test_remaining_assignment_fixes.py`：11 个针对本轮剩余问题的回归测试。
- `src/recourse/cluster_stats.py`：按 fitted-policy cluster 的均值、标准误、置信区间和 paired difference。
- `run_assignment_state_experiment.py`：state-information 性能实验。
- `run_fixed_graph_exact_solver_audit.py`：固定 replay graph 的 exact solver 差分审计。
- `Makefile`：`assignment-audit` 纳入本轮专项测试。
- `docs/NYC_ALL_MODELS_TRAIN_TEST.md`：从旧“七方法”更新为当前九方法及正确 causal/architecture 路径。

附件 companion audit 的只读结果：`remaining=0`，`already_resolved=11`。

## 3. 本地测试

以下命令均成功：

```text
python -m compileall -q src run_*.py test_all_nyc_models.py tests
git diff --check
python -m pytest -q tests/test_remaining_assignment_fixes.py
python -m pytest -q tests/test_assignment_audit_repairs.py ...
python -m pytest -q
make assignment-matrix-smoke
```

- 专项组合：142 tests passed。
- 完整套件：467 tests collected，全部通过；只有 `pyogrio` 关于 `shapely.geos` 的第三方 deprecation warning。
- canonical production matrix smoke：9/9 方法通过。
- synthetic C0/R2 CRN audit：115 个共同随机事件逐值相同，mismatch=0；两个 checkpoint inference 均复现。

## 4. NYC 200 车统一 train/test 接口

真实 NYC 数据，200 车（100 EV + 100 AEV），训练日 2025-12-18、测试日 2025-12-19。为代码 smoke 使用 12 steps、每 2 steps 更新；这不是全天性能或收敛实验。

结果目录：`results/icaps-remaining-fixes-nyc200-smoke-20260830`

| 方法 | test reward | completed | rejected | same-epoch recourse | contract | checkpoint/test weights |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Integrated | -59.615 | 1 | 11 | 0 | pass | loaded / unchanged |
| R0 | 51.804 | 4 | 0 | 0 | pass | loaded / unchanged |
| learned R1 | -46.223 | 3 | 8 | 0 | pass | loaded / unchanged |
| C0 structured R1 | -12.861 | 5 | 14 | 0 | pass | loaded / unchanged |
| R2 | -10.784 | 3 | 9 | 9 | pass | loaded / unchanged |
| R3 | -46.118 | 3 | 8 | 2 | pass | loaded / unchanged |
| Macro | -26.926 | 3 | 5 | 2 | pass | loaded / unchanged |
| R4 | -80.293 | 3 | 9 | 2 | pass | loaded / unchanged |
| Samitha | -9.635 | 2 | 12 | 12 | pass | loaded / unchanged |

独立 `test-only` 从上述 checkpoint 目录再次执行，9/9 完成，没有回退到训练：`results/icaps-remaining-fixes-nyc200-smoke-20260830-test-only`。

## 5. Replay 审计结果

- Macro checkpoint retained 12 joint transitions。
- 静态状态审计成功生成全部 6 个 state variants 的观测；同一 `trajectory_hash` 下 joint、fleet-local、strict fleet-local 分别得到不同 observation hash，strict-local 标记正确。
- fixed-graph solver audit：20 graphs、80 个成功配置行，所有配置可行，最大量化 objective gap 为 0。
- 本机可用 `primal_dual` 与 `gurobi_network`，两种 graph reduction 均完成；`ortools` 未安装，因此 40 个 OR-Tools 配置被明确记录为 unavailable。
- 9/20 graphs 在相同最优目标下选择 trace 不同，属于多最优解/tie 的可观测差异；结果未把“目标等价”错误写成“选边必然相同”。
