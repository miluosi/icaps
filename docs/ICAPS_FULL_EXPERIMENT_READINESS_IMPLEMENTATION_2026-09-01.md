# ICAPS full-experiment readiness implementation

对照基线：`ICAPS_FULL_EXPERIMENT_READINESS_AUDIT_F2E2096_2026-08-31 (1).md`，审计 HEAD `f2e2096ece81a2157269195d2401a47bf02e04de`。

本文只记录代码/实验基础设施是否实现，不把短 smoke test 表述为正式论文结果。

## 逐项实现状态

| 审计项 | 状态 | 实现位置 |
|---|---|---|
| train-once/evaluate-many | 已实现 | `run_recourse_multiday_panel.py`；每个 `(seed, train_window, method)` 只生成一个 checkpoint，每个 test day 独立重载 |
| Macro 下 Myopic/DirectQ/Residual | 已实现 | `run_assignment_learner_experiment.py`、`src/ValueFunction_structured_myopic.py` |
| 规模化实验 | 已实现 runner | `run_assignment_scalability_experiment.py`；100–3000 车、backend、reduction、边数、延迟、内存、fallback、gap |
| rejection/AEV/demand sensitivity | 已实现 | `run_recourse_sensitivity.py`；adaptation 与 nominal-checkpoint robustness 分开 |
| energy sensitivity | 已实现 general-charging 轴 | station capacity、battery consumption、initial SOC、charge-duration scale、unreachable charging vehicles |
| state performance paired summary | 已实现 | `run_assignment_state_experiment.py` |
| deeper state leakage audit | 已实现 | `run_assignment_state_audit.py` 同时检查 pre、residual、EV graph、AEV graph state |
| Samitha hold ablation | 已实现 | `run_samitha_hold_ablation.py`；0%、fixed 10/25/50、learned、EV-first limit |
| hourly/spatial mechanism tables | 已实现 | 环境输出 event/zone/hour/hold 数据，`run_recourse_spatiotemporal_analysis.py` 输出 CSV/JSON |
| formal metrics/runtime | 已补充 | recourse/service/hold/energy/learning、candidate/scoring/serialization/reduction/solve/total、p50–p99、RSS |
| reproducibility manifest | 已补充 | source/data hash、Git revision、replay schema、Python/NumPy/Pandas/SciPy/PyTorch/solver/hardware metadata |
| CI gate split | workflow 已拆分 | unit-fast、recourse-contracts、assignment-production-smoke、synthetic-smoke、nyc-sample-smoke，并上传 artifacts |
| 文档口径 | 已同步 | README、数学模型、NYC 实验说明、example config、day summary causal ladder |
| two-center fixed swap | 未实现，主动 fail closed | 这会改变物理环境，不能作为 charging 参数别名；需论文/建模选择后另行实现 |
| 远端 Actions/branch protection | 需仓库管理员验证 | 本地代码不能分配 GitHub runner、修改 billing/quota 或启用 branch protection |
| 正式多 seed 数值结果 | 未运行 | runner 已就绪；需先冻结日期、seed、预算与最终 energy model |

## Canonical learning targets

Macro 主方法：

$$
y_t^{\mathrm{Macro}}=R_t^{\mathrm{EV}}+R_t^{\mathrm{AEV}}+\gamma^{\Delta t_t}V_{1,t+1}^{-}(S_{t+1}).
$$

Nested $Q_2$ comparator：

$$
y_t^{R4}=R_t^{\mathrm{EV}}+V_{2,t}^{-}(\bar S_t^2),\qquad y_t^{(2)}=R_t^{\mathrm{AEV}}+\gamma^{\Delta t_t}V_{1,t+1}^{-}(S_{t+1}).
$$

同一 Macro 物理架构的 learner family：

$$
\Psi^{\mathrm{myopic}}(e,S)=G(e,S),\qquad \Psi^{\mathrm{DirectQ}}_\theta(e,S)=Q_\theta(e,S),\qquad \Psi^{\mathrm{residual}}_\theta(e,S)=G(e,S)+\Delta_\theta(e,S).
$$

`structured_myopic` 永久使用 structured-only projection，不调用神经网络生成部署分数，且 `train_step` 不执行 optimizer update。

## 实验统计单位

独立 fitted-policy 单位：

$$
(\mathrm{seed},\mathrm{train\_window}).
$$

held-out 配对单位：

$$
(\mathrm{seed},\mathrm{train\_window},\mathrm{test\_day}).
$$

每个 test day 从磁盘重新加载同一个不可变 checkpoint；测试前后完整 tensor hash 必须一致。汇总输出保留 raw rows、cluster summary、paired mean difference、95% CI 与 standardized effect size。

## 本地验证

- 新增 readiness 单元测试：9/9 通过；
- 受影响的 recourse/day/assignment/hold/charging/acceptance 测试（含上述 readiness 测试）：90/90 通过；
- learner、state、scale、sensitivity、hold 五类正式入口 dry-run 通过；
- NYC 真实接口 smoke：8 车、4 EV、8 train steps、8 held-out steps、一个 seed；Myopic 更新 0 次，DirectQ 更新 8 次，Residual 更新 8 次；checkpoint 全部独立重载且测试不改权重；
- smoke artifacts：`results/full_experiment_readiness_smoke_2026-09-01/`（该目录受 `.gitignore` 管理）。

- 完整测试套件：476/476 通过；
- `make smoke-toy`、`make smoke-nyc`、`make smoke-recourse` 和 assignment runner dry-run 均通过。

## 正式运行前仍需冻结

1. 选择并在论文中声明 `general_charging`；若必须使用 two-center fixed swap，则先实现新物理环境并让所有方法重训。
2. 预注册 train/validation/test 日期、seed、update budget、primary outcomes 与 Holm correction 范围。
3. 运行 pilot 估计 paired variance，再确定独立 fitted-policy 数量。
4. 获得绿色远端 CI，启用 required checks，并为正式 source/data/checkpoint/results 打 immutable tag。
5. response delay 当前仍采用同 epoch negligible-delay 假设，即 $\gamma_{\mathrm{within}}=1$；如论文需要延迟效应，另做 0/15/30/60 秒物理敏感性实验。
