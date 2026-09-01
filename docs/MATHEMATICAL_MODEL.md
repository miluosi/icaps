# Mixed-Fleet ADP 数学模型

## 1. 决策过程

离散决策时刻为：

$$
t\in\{0,1,\ldots,T-1\}.
$$

车辆集合由 human-driven EV 和 autonomous EV 两部分构成：

$$
\mathcal{V}=\mathcal{V}^{\mathrm{EV}}\cup\mathcal{V}^{\mathrm{AEV}},
\qquad
\mathcal{V}^{\mathrm{EV}}\cap\mathcal{V}^{\mathrm{AEV}}=\varnothing.
$$

时刻的系统状态包含车辆位置、电量、载客/空闲状态，活跃订单，充电站占用与队列，以及时间和需求特征：

$$
s_t=
\left(
\{\ell_{v,t},b_{v,t},m_{v,t}\}_{v\in\mathcal V},
\mathcal R_t,
\{c_{j,t},q_{j,t}\}_{j\in\mathcal C},
\tau_t,
d_t
\right).
$$

每辆车的候选动作包括服务订单、前往充电站、重定位和保持空闲：

$$
\mathcal A_v(s_t)=
\mathcal A_v^{\mathrm{serve}}
\cup\mathcal A_v^{\mathrm{charge}}
\cup\mathcal A_v^{\mathrm{relocate}}
\cup\{a_v^{\mathrm{idle}}\}.
$$

## 2. 联合可行 assignment

令二元变量表示车辆是否选择候选动作：

$$
x_{v,a,t}\in\{0,1\}.
$$

每辆车至多执行一个动作：

$$
\sum_{a\in\mathcal A_v(s_t)}x_{v,a,t}\le 1,
\qquad \forall v\in\mathcal V.
$$

每个订单至多由一辆车服务：

$$
\sum_{v\in\mathcal V}
\sum_{a\in\mathcal A_v^{\mathrm{serve}}(r)}
x_{v,a,t}
\le 1,
\qquad \forall r\in\mathcal R_t.
$$

考虑可用充电位和允许进入的队列容量，站点约束为：

$$
\sum_{v\in\mathcal V}
\sum_{a\in\mathcal A_v^{\mathrm{charge}}(j)}
x_{v,a,t}
\le
\bar c_j-c_{j,t}+\bar q_j-q_{j,t},
\qquad \forall j\in\mathcal C.
$$

候选集合已过滤续航不可行的动作。若动作的总行驶能耗为所需电量，则：

$$
b_{v,t}-e_{v,a,t}\ge b_v^{\min}.
$$

## 3. 即时收益与 ADP 分数

即时收益统一计入订单收益、移动成本、拒单损失、充电成本和排队惩罚：

$$
g_{v,a,t}
=p_{v,a,t}
-c^{\mathrm{move}}_{v,a,t}
-c^{\mathrm{reject}}_{v,a,t}
-c^{\mathrm{charge}}_{v,a,t}
-\lambda^{\mathrm{wait}}w_{v,a,t}.
$$

价值网络估计动作后状态的 continuation value：

$$
\widehat V_{\theta}(s^{x}_{v,a,t}).
$$

因此交给 ILP、MCMF 或 heuristic 的统一边分数为：

$$
Q_{v,a,t}
=g_{v,a,t}
+\gamma\widehat V_{\theta}(s^{x}_{v,a,t}).
$$

精确联合分配求解：

$$
x_t^{\star}
\in
\arg\max_{x\in\mathcal X(s_t)}
\sum_{v\in\mathcal V}
\sum_{a\in\mathcal A_v(s_t)}
Q_{v,a,t}x_{v,a,t}.
$$

其中可行域集中表达订单互斥、每车一动作、充电容量和电量约束。MCMF 在固定可分解边分数下实现相同的 capacitated projection；heuristic 则按优先级和分数排序逐项接受仍可行的动作。

## 4. Heuristic 排列

对每个可行车辆—动作边构造排序键：

$$
k(v,a,t)=
\left(
\pi^{\mathrm{critical}}_{v,a,t},
\pi^{\mathrm{service}}_{v,a,t},
Q_{v,a,t},
-d_{v,a,t}
\right).
$$

按字典序从高到低扫描候选边，并维护剩余订单与站点容量。已经接受的边集合记为：

$$
\mathcal H_n.
$$

heuristic 更新为：

$$
\mathcal H_{n+1}=
\begin{cases}
\mathcal H_n\cup\{(v,a)\}, &
\text{if }\mathcal H_n\cup\{(v,a)\}\in\mathcal X(s_t),\\
\mathcal H_n, & \text{otherwise}.
\end{cases}
$$

## 5. 价值学习

一条训练 transition 记为：

$$
\left(s_t,x_t,r_t,s_{t+1}\right).
$$

基本 TD target 为：

$$
y_t=r_t+\gamma\widehat V_{\bar\theta}(s_{t+1}),
$$

网络以均方 TD 误差更新：

$$
\mathcal L_{\mathrm{TD}}(\theta)
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(
\widehat V_{\theta}(s_i^{x})-y_i
\right)^2.
$$

post-demand direct 模型把基础 continuation、队列特征和未来需求 head 分开：

$$
\widehat V_{\theta}
=
\widehat V_{\theta}^{\mathrm{base}}
+\alpha_q\widehat q_{\theta}^{\mathrm{queue}}
+\alpha_d\widehat d_{\theta}^{\mathrm{post}}.
$$

## 6. 三种 heuristic 实验的严格区别

精确联合分配策略与 heuristic 分配策略分别记为：

$$
\pi^{\mathrm{exact}},
\qquad
\pi^{\mathrm{heu}}.
$$

两个训练算子为：

$$
\theta^{\mathrm{exact}}
=\operatorname{Train}(\pi^{\mathrm{exact}}),
\qquad
\theta^{\mathrm{heu}}
=\operatorname{Train}(\pi^{\mathrm{heu}}).
$$

三种评估策略分别是：

$$
\mathrm{HEU}=\pi^{\mathrm{heu}}(g),
$$

$$
\mathrm{ADP\mbox{-}HEU}
=
\pi^{\mathrm{heu}}
\left(g+\gamma\widehat V_{\theta^{\mathrm{exact}}}\right),
$$

$$
\mathrm{ADP\mbox{-}HEU\mbox{-}HEU}
=
\pi^{\mathrm{heu}}
\left(g+\gamma\widehat V_{\theta^{\mathrm{heu}}}\right).
$$

这一定义保证旧 `ADP-HEU` 没有被重解释：它始终使用 `gurobi` 标签的 exact-trained checkpoint；只有 `ADP-HEU-HEU` 使用 `heu` 标签的 heuristic-trained checkpoint。

## 7. 评估指标

服务率为：

$$
\mathrm{ServiceRate}
=
\frac{N_{\mathrm{completed}}}{N_{\mathrm{generated}}}.
$$

仅在实际等待过的车辆上计算平均充电等待：

$$
\overline W_{+}
=
\frac{
\sum_v W_v\mathbf 1\{W_v>0\}
}{
\sum_v\mathbf 1\{W_v>0\}
}.
$$

论文比较至少同时报告累计收益、完成订单、服务率、EV/AEV 分组结果、等待时间、拒单/掉线结果以及每个 assignment backend 的运行时间。

## 8. 同一决策时刻内的 recourse

`evfirst` 在一个整数 epoch 内包含两个阶段。EV leader 先从联合可行域
$\mathcal F_t^{(1)}$ 选择 $A_t^{(1)}$，随后实现每次 offer 的接受结果
$z_{v,r,t}\in\{0,1\}$。拒单、未出价以及其他剩余请求分别标记为
`rejected`、`unoffered` 和 `other`，由此构造不可变 residual state
$s_t^{\mathrm{res}}$。AEV follower 再求解：

$$
A_t^{(2)}\in\arg\max_{A\in\mathcal F_t^{(2)}(s_t^{\mathrm{res}})}
\sum_{e\in A}\Psi_e^{(2)}.
$$

同一 epoch 的 recourse 只由整数 `epoch_id` 判断。AEV assignment、实际 pickup 和最终 completion 是三个独立的恢复结果；pickup/completion 不能反向改变“同轮 assignment recovery”的定义。

每轮训练保存一个不可变联合 transition：

$$
\mathcal T_t=
(s_t,G_t^{(1)},A_t^{(1)},z_t,s_t^{\mathrm{res}},
G_t^{(2)},A_t^{(2)},r_t^{\mathrm{EV}},r_t^{\mathrm{AEV}},s_{t+1}).
$$

其中 $G_t^{(1)}$ 与 $G_t^{(2)}$ 保存当时的全部可行边、结构化分数、采集分数和资源容量。历史 replay 的编码和 target projection 只能读取这些快照，不能读取当前模拟器的可变图。

## 9. Canonical 因果链、Macro 与 Nested $Q_2$

主文的同架构因果链固定为：

$$
\text{Structured no-repair (C0)}\rightarrow\text{Repair Only (R2)}\rightarrow\text{Repair Learning (R3)}\rightarrow\text{Macro}\rightarrow\text{Nested }Q_2\text{ (R4)}.
$$

Learned R1 只作为诊断对照；Integrated 与 Samitha 属于 commitment architecture 对照，不混入这条学习因果链。Macro 主方法使用实现后的同轮两阶段系统收益：

$$
y_t^{\mathrm{Macro}}=R_t^{\mathrm{EV}}+R_t^{\mathrm{AEV}}+\gamma^{\Delta t_t}V_{1,t+1}^{-}(S_{t+1}).
$$

Nested R4 保持相同物理执行，只把 leader credit estimator 改为：

$$
y_t^{R4}=R_t^{\mathrm{EV}}+V_{2,t}^{-}(\bar S_t^2),\qquad y_t^{(2)}=R_t^{\mathrm{AEV}}+\gamma^{\Delta t_t}V_{1,t+1}^{-}(S_{t+1}).
$$

拒单是已观察到的阶段结果，而不是 terminal mask，因此不会把同轮 follower continuation 置零。R3、Macro 和 R4 使用相同的 predictor control、可行图和 exact execution solver。

target 动作采用 double-Q 语义：online critic 在序列化可行图上通过与执行相同的联合资源约束选择动作，target critic 只评估已选联合动作：

$$
\widehat A=\Pi_{\mathcal F(G)}(Q_\theta),
\qquad
V_{\bar\theta}(G)=\sum_{e\in\widehat A}Q_{\bar\theta}(e).
$$

这里的投影同时执行每车至多一项、请求互斥和充电站容量约束。

## 10. Learner、Samitha 与状态消融

系统收益严格满足 $r_t^{\mathrm{sys}}=r_t^{\mathrm{EV}}+r_t^{\mathrm{AEV}}$。在同一 Macro 物理架构、同一 exact feasible graph 下，只保留两种 learner：

$$
\Psi^{\mathrm{DirectQ}}_\theta(e,S)=Q_\theta(e,S),\qquad \Psi^{\mathrm{residual}}_\theta(e,S)=G(e,S)+\Delta_\theta(e,S).
$$

Samitha 与 Integrated 共用 stage-0 graph builder；Samitha 只允许明确 hold 的 AEV 进入 outcome 后 repair，已 commitment 的 AEV 不可重派。固定 10%/25%/50% hold 是显式物理控制臂，learned hold 则由相同 exact projection 选择。

状态消融由同一个原始 `SystemSnapshot` 的确定性 mask 得到：joint-state 变体保留两类车状态，fleet-local 变体只把另一类车的字段清零。`shared_critic` 与 `separate_critics` 只控制参数共享；可行动作图、transition ID 和采集到的原始状态保持不变。
