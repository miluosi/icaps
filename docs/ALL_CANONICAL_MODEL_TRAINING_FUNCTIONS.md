# ICAPS 当前所有 canonical 模型：训练函数与核心公式

本文按当前代码仓库的真实执行路径整理。这里的“模型”首先指论文主实验中的九个 assignment/recourse 方法；它们不是九套重复的神经网络，而是共享同一个训练 worker 和同一类 joint critic，通过以下四个轴产生差异：

1. 物理决策架构：Integrated、EV-first 或 Integrated + limited-hold repair；
2. EV 拒单后是否允许同一 epoch 的 AEV repair；
3. AEV follower 使用 structured score 还是 learned residual；
4. EV leader 的 recourse credit 是 uncoupled、macro realized 还是 nested follower。

论文主 runner 默认使用 `optimization_anchored_residual` 和 `joint_state_separate_critics`。`integrated_directq` 是 Integrated/Integrated-repair 的 full-Q comparator，不是 EV-first R0--R4 的默认 learner。

## 1. 九个训练入口

入口位于 `test_all_nyc_models.py`。每个函数只验证 method identity，然后进入同一个 `run_training_worker(settings)`：

```python
def train_integrated(settings):
    return _train_named_method(settings, 'no_repair')

def train_r0(settings):
    return _train_named_method(settings, 'evfirst_no_rejection')

def train_r1(settings):
    return _train_named_method(settings, 'evfirst_no_repair')

def train_structured_r1(settings):
    return _train_named_method(settings, 'evfirst_no_repair_structured')

def train_r2(settings):
    return _train_named_method(settings, 'repair_only')

def train_r3(settings):
    return _train_named_method(settings, 'repair_learning')

def train_macro(settings):
    return _train_named_method(settings, 'recourse_macro')

def train_r4(settings):
    return _train_named_method(settings, 'recourse_nested_q2')

def train_samitha(settings):
    return _train_named_method(settings, 'samitha')
```

完整映射如下。


| 显示名             | canonical method               | 入口函数              | motion                               | variant          | repair policy           | leader credit    |
| ------------------ | ------------------------------ | --------------------- | ------------------------------------ | ---------------- | ----------------------- | ---------------- |
| Integrated         | `no_repair`                    | `train_integrated`    | `simulate_motion_integrated_control` | `legacy`         | none                    | uncoupled/system |
| R0                 | `evfirst_no_rejection`         | `train_r0`            | `simulate_motion_evfirst`            | `r0`             | none                    | uncoupled        |
| learned R1         | `evfirst_no_repair`            | `train_r1`            | `simulate_motion_evfirst`            | `r1`             | none                    | uncoupled        |
| C0 / structured R1 | `evfirst_no_repair_structured` | `train_structured_r1` | `simulate_motion_evfirst`            | `r1_structured`  | structured              | uncoupled        |
| R2                 | `repair_only`                  | `train_r2`            | `simulate_motion_evfirst`            | `r2`             | structured              | uncoupled        |
| R3                 | `repair_learning`              | `train_r3`            | `simulate_motion_evfirst`            | `r3`             | learned                 | uncoupled        |
| Macro              | `recourse_macro`               | `train_macro`         | `simulate_motion_evfirst`            | `recourse_macro` | learned                 | macro realized   |
| R4                 | `recourse_nested_q2`           | `train_r4`            | `simulate_motion_evfirst`            | `r4`             | learned                 | nested follower  |
| Samitha            | `samitha`                      | `train_samitha`       | `simulate_motion_integrated_repair`  | `legacy`         | structured limited hold | macro realized   |

注册表的唯一来源是 `src/recourse/config.py` 中的 `PAPER_METHODS` 和 `METHODS`。

## 2. 公共训练主循环

`run_training_worker` 与 `run_recourse_day.run_worker` 的训练阶段执行相同流程：

```python
env = build_env(settings, settings.seed, method, training=True)
pair = build_pair(env, ...)

for step in range(number_of_steps):
    actions, stored, stored_ev = selected_motion(...)
    _, rewards, _, done, _ = env.step(actions, stored, stored_ev)

    if (step + 1) % train_every == 0:
        pair[0].train_step(batch_size=batch_size, ifEV=False)  # AEV
        pair[1].train_step(batch_size=batch_size, ifEV=True)   # EV

save_pair(pair, checkpoint, metadata)
```

`pair[0]` 是 AEV value function，`pair[1]` 是 EV value function。两者共享一份 immutable joint replay；在 shared-critic state mode 下二者可以是同一对象，在 separate-critic mode 下保持不同对象。

每个 replay transition 保存：pre-state、EV/stage-0 feasible graph、实际 EV joint action、拒单结果、residual state、AEV/stage-2 graph、实际 AEV joint action、当前 epoch 的 EV/AEV/system reward、下一状态和直接 successor identity。

系统 reward 只由环境实际返回的车辆 reward 相加得到：

$$
r_t^{\mathrm{sys}}=r_t^{\mathrm{EV}}+r_t^{\mathrm{AEV}}.
$$

过期单、丢单和 recovery 作为独立 service metrics 报告；当前没有额外虚构 lost-order economic penalty。

## 3. 共同的 exact assignment 层

所有论文方法都先构造 feasible edge graph，再在相同资源约束下进行 exact additive projection：

$$
A_t^\star=\arg\max_{A\in\mathcal F(G_t)}\sum_{e\in A}s_t(e).
$$

其中可行集要求每个图中车辆恰好选择一条真实边，并满足 request、charging station 等共享资源容量：

$$
\sum_{e\in A:\,v(e)=v}1=1,
\qquad
\sum_{e\in A:\,r(e)=r}1\le c_r.
$$

论文 runner 默认的 rollout 和 target oracle 都是 exact `primal_dual`，使用相同 cost quantization、graph reduction、feasibility verification 和 strict fail-closed 配置。

## 4. 默认 residual learner

### 4.1 网络结构

`optimization_anchored_residual` 继承 ST-MASAC-GAT 主体：

- graph node input 16 维，graph hidden 96 维；
- edge local input 18 维；启用拒单预测输入时额外加入概率和 mask 两维；
- twin critics：edge input → 128 → 128 → 1；
- actor：edge input → 128 → 128 → 1，但当前默认 `eta_pi=0`，因此 actor 不改变部署的 MCMF score；
- queue predictor：9 → 64 → 64 → 1；
- target critics、target graph encoder、target mixer 和 target queue predictor 使用 Polyak 更新。

### 4.2 部署分数

令 structured/myopic edge score 为 `g`。默认 residual learner 使用两个 online critics 的均值做动作选择：

$$
\delta_{\theta}^{\mathrm{sel}}(e,H_t)
=w_{\mathrm{type}(e)}
\frac{Q_{\theta_1}(e,H_t)+Q_{\theta_2}(e,H_t)}{2}.
$$

部署给 exact assignment oracle 的分数为：

$$
s_t(e)=g_t(e)+\beta_k\,
\operatorname{clip}\!\left(
\delta_{\theta}^{\mathrm{sel}}(e,H_t),-b_t(e),b_t(e)
\right).
$$

普通边的 correction bound 为：

$$
b_t(e)=\rho\max\!\left(\operatorname{Std}\{g_t(e')\}_{e'\in G_t},1\right),
\qquad \rho=0.30.
$$

充电边的 bound 还必须覆盖预计充电成本和预计排队成本。

Residual 权重在训练初期线性 warm-up：

$$
\beta_k=\min\!\left(\beta_{\max},
\beta_{\max}\frac{k}{K_{\mathrm{warm}}}\right),
\qquad \beta_{\max}=0.30,
\qquad K_{\mathrm{warm}}=500.
$$

### 4.3 Solver-consistent double-Q target

下一 feasible graph 先由 online full scores 选择 joint action：

$$
A_{t+1}^{\mathrm{online}}
=\arg\max_{A\in\mathcal F(G_{t+1})}
\sum_{e\in A}s_{\theta}(e).
$$

再由 lagged twin target critics 保守评估同一 selected joint action：

$$
\bar V(G_{t+1})
=\sum_{e\in A_{t+1}^{\mathrm{online}}}
\left[g_{t+1}(e)
+w_{\mathrm{type}(e)}
\min\!\left(
Q_{\bar\theta_1}(e),Q_{\bar\theta_2}(e)
\right)\right].
$$

target correction 刻意不乘部署时的 beta，也不做部署 clip；critic 拟合的是完整、无界的 Bellman residual。

对当前 selected joint action，完整 Bellman target 与 residual label 为：

$$
Y_t=r_t+d_t\bar V(G_{t+1}),
$$

$$
y_t^{\mathrm{res}}=Y_t-g_t(A_t),
\qquad
g_t(A_t)=\sum_{e\in A_t}g_t(e),
$$

其中：

$$
d_t=
\begin{cases}
0, & \text{terminal},\\
\gamma^{\Delta_t}, & \text{temporal successor},\\
\gamma_{\mathrm{within}}, & \text{same-epoch nested follower},
\end{cases}
\qquad
\gamma=0.95,
\qquad
\gamma_{\mathrm{within}}=1.
$$

Twin residual predictions是当前 joint action 中所有 selected edges 的 raw correction 之和：

$$
\widehat y_{t,j}^{\mathrm{res}}
=\sum_{e\in A_t}w_{\mathrm{type}(e)}Q_{\theta_j}(e,H_t),
\qquad j\in\{1,2\}.
$$

Joint loss 为带 prioritized-replay importance weight 的 twin Huber loss：

$$
\mathcal L_{\mathrm{joint}}
=\frac{1}{B}\sum_{i=1}^{B}\omega_i
\left[
\operatorname{Huber}_{\kappa}
\left(\widehat y_{i,1}^{\mathrm{res}}-y_i^{\mathrm{res}}\right)
+
\operatorname{Huber}_{\kappa}
\left(\widehat y_{i,2}^{\mathrm{res}}-y_i^{\mathrm{res}}\right)
\right],
\qquad \kappa=1.
$$

Huber 定义为：

$$
\operatorname{Huber}_{\kappa}(x)=
\begin{cases}
\frac{1}{2}x^2, & |x|\le\kappa,\\
\kappa\left(|x|-\frac{1}{2}\kappa\right), & |x|>\kappa.
\end{cases}
$$

成功更新后执行 gradient clipping、PER priority 更新和 target soft update：

$$
\bar\theta\leftarrow(1-\tau)\bar\theta+\tau\theta,
\qquad \tau=0.005.
$$

Replay priority 同时保留 TD error、拒单事件和真实 recourse 事件：

$$
p_i=|\delta_i|+\epsilon
+\lambda_{\mathrm{rej}}\mathbf 1\{i\text{ contains rejection}\}
+\lambda_{\mathrm{rec}}\mathbf 1\{i\text{ contains true recourse}\}.
$$

## 5. 九个方法分别训练什么

### 5.1 Integrated / `train_integrated`

物理执行只有一个联合 stage-0 graph，EV 与 AEV 同时参加 exact assignment；不生成 hold edge，不运行 stage 2。

$$
G_t^0=G_t^{\mathrm{joint}}
\left(\mathcal V_t^{\mathrm{EV}}\cup
\mathcal V_t^{\mathrm{AEV}},\mathcal R_t\right).
$$

Joint transition 使用 system reward：

$$
Y_t^{\mathrm{Integrated}}
=r_t^{\mathrm{sys}}+d_t\bar V(G_{t+1}^{0}).
$$

当前 selected graph 中属于 EV/AEV 的边分别路由到 EV/AEV critic；一个 system joint loss 对所有实际参与的 provider 反向传播。没有 recourse stage，也没有 leader/follower credit 拆分。

### 5.2 R0 / `train_r0`

R0 是 EV-first、禁止拒单控制：

$$
\mathbf 1\{\text{EV rejection at }t\}=0.
$$

Stage 1 训练 EV leader，stage 2 训练 AEV follower；二者都使用 learned residual，但 target uncoupled：

$$
Y_t^{\mathrm{EV}}
=r_t^{\mathrm{EV}}+d_t\bar V^{\mathrm{EV}}(G_{t+1}),
$$

$$
Y_t^{\mathrm{AEV}}
=r_t^{\mathrm{AEV}}+d_t\bar V^{\mathrm{AEV}}(G_{t+1}).
$$

R0 用于检查“完全没有拒单”时的上界机制，不是 no-repair rejection baseline。

### 5.3 learned R1 / `train_r1`

R1 允许 EV 真实拒单，但拒绝的订单不能在同一 epoch 进入 AEV repair graph：

$$
\mathcal R_{t,\mathrm{AEV}}^{\mathrm{R1}}
=\mathcal R_t^{\mathrm{residual}}
\setminus\mathcal R_t^{\mathrm{rejected}}.
$$

EV leader 与 AEV follower 都学习 residual；target 与 R0 一样 uncoupled。拒单订单仍保留在环境中，可以在后续 epoch 被服务，但本轮 `same_epoch_repair=0`。

learned R1 会改变 AEV score，因此不是 R2 的纯物理 no-repair 因果基线。

### 5.4 C0 / structured R1 / `train_structured_r1`

C0 的物理执行与 learned R1 相同：允许 EV 拒单，但禁止同轮 repair。区别是 AEV stage 只使用 structured score：

$$
s_t^{\mathrm{AEV}}(e)=g_t(e).
$$

AEV follower 的 `train_step(ifEV=False)` 直接返回零，queue/post-demand 等 causal auxiliary predictors 也冻结。EV leader 仍训练 uncoupled residual。

C0 是主因果路径中的 no-repair baseline：

$$
\mathrm{C0}\rightarrow\mathrm{R2}
$$

只打开同轮 structured physical repair。

### 5.5 R2 / `train_r2`

R2 与 C0 使用相同 frozen P0 predictor 输入和 structured AEV score，但把拒单订单重新放回同轮 AEV feasible support：

$$
\mathcal R_{t,\mathrm{AEV}}^{\mathrm{R2}}
\supseteq\mathcal R_t^{\mathrm{rejected}}.
$$

$$
s_t^{\mathrm{AEV}}(e)=g_t(e),
\qquad
\nabla_{\theta^{\mathrm{AEV}}}\mathcal L=0.
$$

EV target 仍然 uncoupled，不接收 AEV repair reward：

$$
Y_t^{\mathrm{EV,R2}}
=r_t^{\mathrm{EV}}+d_t\bar V^{\mathrm{EV}}(G_{t+1}).
$$

因此 C0→R2 测量的是“只加入同轮物理 repair”的变化。

### 5.6 R3 / `train_r3`

R3 保留 R2 的同轮 physical repair，但把 AEV follower 从 structured-only 改为 learned residual：

$$
s_t^{\mathrm{AEV,R3}}(e)
=g_t(e)+\beta_k\operatorname{clip}
\left(\delta_{\theta}^{\mathrm{AEV}}(e),-b_t(e),b_t(e)\right).
$$

AEV follower target 为：

$$
Y_t^{\mathrm{AEV,R3}}
=r_t^{\mathrm{AEV}}+d_t\bar V^{\mathrm{AEV}}(G_{t+1}).
$$

EV leader 仍然 uncoupled：

$$
Y_t^{\mathrm{EV,R3}}
=r_t^{\mathrm{EV}}+d_t\bar V^{\mathrm{EV}}(G_{t+1}).
$$

因此 R2→R3 只引入 repair-stage learning，不引入 leader recourse credit。

### 5.7 Macro / `train_macro`

Macro 的物理 repair policy 与 R3 相同，AEV follower 也继续学习。唯一核心变化是 EV leader 使用本轮真实 realized system reward：

$$
Y_t^{\mathrm{EV,Macro}}
=r_t^{\mathrm{EV}}+r_t^{\mathrm{AEV}}
+d_t\bar V^{\mathrm{EV}}(G_{t+1}).
$$

等价写成：

$$
Y_t^{\mathrm{EV,Macro}}
=r_t^{\mathrm{sys}}+d_t\bar V^{\mathrm{EV}}(G_{t+1}).
$$

AEV follower 仍使用自身 realized reward 的普通 temporal target：

$$
Y_t^{\mathrm{AEV,Macro}}
=r_t^{\mathrm{AEV}}+d_t\bar V^{\mathrm{AEV}}(G_{t+1}).
$$

因此 R3→Macro 测量的是“给 EV leader 加入 realized recourse credit”。Macro 是当前主 recourse-credit 方法。

### 5.8 R4 / `train_r4`

R4 的物理 repair、AEV learned follower 和 frozen auxiliary controls 与 Macro 相同。区别在于 EV leader 不使用 realized AEV reward，而是 bootstrap 当前同一 epoch 的 AEV follower target value：

$$
Y_t^{\mathrm{EV,R4}}
=r_t^{\mathrm{EV}}
+\gamma_{\mathrm{within}}
\bar V^{\mathrm{AEV}}(G_t^{\mathrm{AEV}}).
$$

当前配置中：

$$
\gamma_{\mathrm{within}}=1.
$$

AEV follower 完成本轮第二阶段后，再连接到下一 epoch 的 EV leader graph：

$$
Y_t^{\mathrm{AEV,R4}}
=r_t^{\mathrm{AEV}}
+d_t\bar V^{\mathrm{EV}}(G_{t+1}^{\mathrm{EV}}).
$$

因此 Macro→R4 比较的是 realized macro target 与 nested target estimator，而不是两种不同的物理 repair policy。

### 5.9 Samitha / `train_samitha`

Samitha 与 Integrated 共享完全相同的 stage-0 graph builder、score grid 和 exact oracle，但额外允许 AEV 选择 `hold_for_repair`：

$$
H_t=\left\{v\in\mathcal V_t^{\mathrm{AEV}}:
(v,\mathrm{hold\_for\_repair})\in A_t^0\right\}.
$$

未被 hold 的 AEV stage-0 action 立即 commit，之后不允许重新分配。只有 held AEV 能进入 stage 2：

$$
G_t^2
=G^{\mathrm{residual}}
\left(H_t,\mathcal R_t^{\mathrm{repair}}mid A_t^0\right).
$$

Stage 2 使用 structured/myopic score，且显式扣除 stage 0 已占用的资源容量：

$$
s_t^2(e)=g_t^2(e).
$$

Stage-0 joint critic 以整个 epoch 的 realized system reward 学习 hold 的长期价值：

$$
Y_t^{\mathrm{Samitha}}
=r_t^{\mathrm{sys}}+d_t\bar V(G_{t+1}^{0}).
$$

Samitha 没有独立的 learned stage-2 follower TD；stage 2 的真实结果通过 system reward 回到 stage-0 hold/commit decision。Integrated→Samitha 是 operating architecture comparison，不属于 C0→R2→R3→Macro→R4 的 EV-first 因果链。

## 6. Full-Q comparator：`integrated_directq`

Direct-Q 与 residual learner 使用同一个 graph encoder、twin critics、feasible graph 和 exact projection，但部署 score 直接等于网络输出：

$$
s_t^{\mathrm{DirectQ}}(e)=Q_{\theta}^{\mathrm{full}}(e,H_t).
$$

它不把 structured score `g` 加回，也不进行 residual beta warm-up 和 residual clip。

Bellman label 直接拟合完整 Q：

$$
y_t^{\mathrm{DirectQ}}
=r_t+d_t\bar V^{\mathrm{DirectQ}}(G_{t+1}).
$$

$$
\mathcal L_{\mathrm{DirectQ}}
=\operatorname{Huber}
\left(\widehat Q_{\theta_1}(A_t)-y_t^{\mathrm{DirectQ}}\right)
+\operatorname{Huber}
\left(\widehat Q_{\theta_2}(A_t)-y_t^{\mathrm{DirectQ}}\right).
$$

CLI 明确限制 `integrated_directq` 只能用于 `integrated` 或 `integrated_repair`；九方法统一 runner 默认不把它用于 EV-first R0--R4。

## 7. EV 拒单概率神经网络

拒单预测器独立于 ADP critic，类名为 `EVRejectionProbabilityModel`，兼容别名为 `BinaryAcceptanceModel`。它是两层 PyTorch MLP，不是逻辑回归：

```text
driver_offer_core: 3 -> 16 -> 8 -> 1
platform_context:  30 -> 64 -> 32 -> 1
activation: ReLU
optimizer: Adam
label: rejected=1, accepted=0
```

标准化输入和未校准 logit 为：

$$
\tilde x_i=\frac{x_i-\mu_i}{\sigma_i},
\qquad
z=f_{\phi}(\tilde x).
$$

训练目标为自然类别比例下的 BCE-with-logits 加权重 L2，不做 class balancing：

$$
\mathcal L_{\mathrm{reject}}(\phi)
=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\right]
+\frac{\lambda}{2}\sum_{W\in\phi}\|W\|_2^2.
$$

校准后输出的连续拒单概率为：

$$
\widehat p_{\mathrm{reject}}(x)
=\sigma(az+b).
$$

temperature calibration 是 `b=0` 的特殊情况。训练按 train/validation/test seeds 或完整日期分离，验证集用于正则强度选择、early stopping 和概率校准。

论文九方法 runner 当前默认关闭 rejection predictor，也不把预测拒单概率输入 Q-value；因此上述网络不会暗中改变主因果表。显式启用时，critic 输入同时包含概率和有效性 mask，且 predictor checkpoint hash 会写入 replay edge，防止训练/推理模型错配。

## 8. Queue 与 post-demand 辅助预测器

Queue predictor 使用 9 维充电站/车辆/时间特征预测 observed wait，目标为：

$$
\mathcal L_{\mathrm{queue}}
=\frac{1}{N}\sum_{i=1}^{N}
\left(\widehat w_i-w_i^{\mathrm{observed}}\right)^2.
$$

Post-demand predictor 使用 8 维 post-action location/time/duration/current-demand 特征，经过 softplus 保证非负：

$$
\widehat d_i^{\mathrm{post}}
=\operatorname{softplus}\left(f_{\psi}(x_i)+b_0\right).
$$

$$
\mathcal L_{\mathrm{post{-}demand}}
=\frac{1}{N}\sum_{i=1}^{N}
\left(\widehat d_i^{\mathrm{post}}-d_i^{\mathrm{observed}}\right)^2.
$$

在 `optimization_anchored_residual` 中，post-demand 不直接进入 actor，也不让 base critic MLP 任意吸收；它只通过可解释的 action-specific 线性 head 修正 twin critic：

$$
Q_{\theta_j}(e,H_t,\widehat d_t^{\mathrm{post}})
=Q_{\theta_j}^{\mathrm{base}}(e,H_t)
+\widehat d_t^{\mathrm{post}}\,
\omega_{j,a(e)},
\qquad j\in\{1,2\}.
$$

为保证主因果链只改变指定机制，C0、R2、R3、Macro、R4 都使用同一个 `CAUSAL_PREDICTOR_VARIANTS` 控制集合，默认冻结这些 auxiliary predictors。Integrated、R0、learned R1 和 Samitha 不属于该因果链，是否训练辅助 predictor 由其自身配置决定。

## 9. 九方法更新矩阵


| 方法       | EV 拒单  | 同轮 AEV repair            | EV critic                 | AEV critic/follower                                         | auxiliary predictor  | EV leader target       |
| ---------- | -------- | -------------------------- | ------------------------- | ----------------------------------------------------------- | -------------------- | ---------------------- |
| Integrated | realized | 无独立 repair stage        | system joint 更新         | 同一 system joint loss 中更新                               | 非 causal frozen set | system temporal        |
| R0         | 禁止     | 无拒单可修复               | learned                   | learned                                                     | 非 causal frozen set | uncoupled temporal     |
| learned R1 | 允许     | 禁止                       | learned                   | learned residual，但不含本轮 rejected requests              | 非 causal frozen set | uncoupled temporal     |
| C0         | 允许     | 禁止                       | learned                   | structured-only、冻结                                       | frozen P0            | uncoupled temporal     |
| R2         | 允许     | structured repair          | learned                   | structured-only、冻结                                       | frozen P0            | uncoupled temporal     |
| R3         | 允许     | learned repair             | learned                   | learned                                                     | frozen P0            | uncoupled temporal     |
| Macro      | 允许     | learned repair             | learned                   | learned                                                     | frozen P0            | realized system reward |
| R4         | 允许     | learned repair             | learned                   | learned                                                     | frozen P0            | nested follower target |
| Samitha    | 允许     | held AEV structured repair | stage-0 system joint 更新 | stage-0 selected AEV edge参与 joint loss；无单独 stage-2 TD | 非 causal frozen set | system temporal        |

## 10. 正确的比较路径

EV-first 主因果链为：

$$
\underbrace{\mathrm{C0}}_{\text{structured no repair}}
\rightarrow
\underbrace{\mathrm{R2}}_{\text{physical repair}}
\rightarrow
\underbrace{\mathrm{R3}}_{\text{repair learning}}
\rightarrow
\underbrace{\mathrm{Macro}}_{\text{realized leader credit}}
\rightarrow
\underbrace{\mathrm{R4}}_{\text{nested target estimator}}.
$$

诊断对照为：

$$
\mathrm{C0}\rightarrow\mathrm{learned\ R1}.
$$

架构对照为：

$$
\mathrm{Integrated}\rightarrow\mathrm{Samitha}.
$$

R0 是 no-rejection control，不应插入 physical repair 的主因果链。

## 11. 训练与测试命令

查看九个入口：

```bash
python test_all_nyc_models.py list
```

训练全部方法：

```bash
python test_all_nyc_models.py train-only \
  --models all \
  --num-vehicles 200 --num-ev 100 \
  --output-dir results/nyc_all_models/train-all
```

训练后测试：

```bash
python test_all_nyc_models.py train-test \
  --models all \
  --num-vehicles 200 --num-ev 100 \
  --output-dir results/nyc_all_models/train-test-all
```

只加载已有 checkpoint 测试：

```bash
python test_all_nyc_models.py test-only \
  --source-dir results/nyc_all_models/train-all \
  --output-dir results/nyc_all_models/test-all
```

测试阶段先验证 checkpoint schema、两个 learner、method axes、state/learner variant 和 solver configuration，再加载 tensors；测试不调用训练更新，并要求完整模型 weight hash 保持不变。

## 12. 其余 value-function registry 条目

`src/value_function_registry.py` 还保留以下历史或专项 value-function 路由。它们是网络/特征消融，不等于新增的 recourse method，也不进入九方法 runner 的默认主表。


| distribution mode                  | 实际类/用途                                                 |
| ---------------------------------- | ----------------------------------------------------------- |
| `bayes`、`time-only`               | `ValueFunction_pytorch_bayes`，历史 Bayes/time feature 路径 |
| `st_masac_gat`                     | 基础 ST-MASAC-GAT twin critic                               |
| `st_masac_gat_frozen`              | 冻结 graph encoder 的基础 ST-MASAC-GAT                      |
| `st_masac_gat_neighbour_frozen`    | 冻结 neighbour context 的 ST-MASAC-GAT                      |
| `st_masac_gat_post_demand`         | 将 post-demand prediction 加入 edge feature 的版本          |
| `st_masac_gat_post_demand_direct`  | action-specific direct demand-value head                    |
| `st_masac_gat_queue_demand_gurobi` | 指向同一 post-demand direct 实现的兼容名称                  |
| `optimization_anchored_residual`   | 九方法默认 solver-consistent residual learner               |
| `integrated_directq`               | Integrated/Integrated-repair full-Q comparator              |
| `none`                             | ADP 关闭时的兼容 control 映射；trainer 不构造学习模型       |

顶层 `learner_variant` 只有三种正式语义：`legacy`、`integrated_directq`、`optimization_anchored_residual`。当选择后两者时，trainer 直接按 learner name 路由类；`legacy` 则继续由 `distribution_mode` 选择历史 value function。

## 13. 关键源码位置


| 内容                                                    | 文件                                                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 九方法注册与 causal contrasts                           | `src/recourse/config.py`                                                             |
| 九个命名训练函数                                        | `test_all_nyc_models.py`                                                             |
| 训练日/测试日 worker                                    | `run_recourse_day.py`                                                                |
| 公共 rollout 与定期`train_step`                         | `run_recourse_audit.py`                                                              |
| Residual/Direct-Q 的 twin critic 与 joint loss          | `src/ValueFunction_st_masac_gat.py`                                                  |
| Optimization-anchored residual override                 | `src/ValueFunction_optimization_anchored_residual.py`                                |
| Integrated Direct-Q comparator                          | `src/ValueFunction_integrated_directq.py`                                            |
| R0--R4 policy 与 target builder                         | `src/recourse/target_builder.py`                                                     |
| Integrated/Samitha 共享 stage-0 与 limited-hold stage 2 | `src/recourse/integrated_repair.py`                                                  |
| EV-first NYC 执行                                       | `src/NYCEnvironment.py`                                                              |
| EV-first synthetic 执行                                 | `src/Environment.py`                                                                 |
| Critic wiring 与共享 replay identity                    | `src/recourse/critics.py`                                                            |
| Replay/PER                                              | `src/recourse/replay.py`                                                             |
| 拒单概率神经网络                                        | `src/acceptance_model.py`、`train_acceptance_model.py`                               |
| Queue/post-demand predictors                            | `src/ValueFunction_st_masac_gat.py`、`src/ValueFunction_st_masac_gat_post_demand.py` |
