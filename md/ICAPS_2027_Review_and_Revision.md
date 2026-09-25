# ICAPS 2027 审稿意见与修改方案

**论文：** *Recourse-Aware Approximate Dynamic Programming for Mixed-Fleet Ride-Hailing*  
**审查日期：** 2026-09-22  
**正文 M：** `ADP_ICAPS_2027 (7).pdf`，8 页。  
**补充材料 S：** `ADP_ICAPS_2027 (8).pdf`，9 页。

> 两个文件分别是正文与补充材料，并非两个需要择一审查的正文版本。本文中的页码均为 PDF 实际页码；M/S 后的章节、公式及图表编号用于定位原文。外部来源列在文末，会议要求与审稿人的改进建议明确区分。
>
> 本次审查覆盖两份文档的叙事、模型、证明、学习目标、表格和图片。没有获得作者的源代码、原始训练日志或逐实例结果，因此不把“文稿缺少说明”写成“代码已经出错”，也不声称复现了 NYC 或求解器实验。另对 SSG 的逻辑规则进行了独立的小规模枚举检查，范围见第 5 节。

## 1. 总体意见

**按当前文稿作接收判断，我会倾向弱拒，建议实质修改后再投。主要问题不是 SSG 定理已经被推翻，而是：模型与默认仿真的适用边界没有闭合，学习目标的策略语义存在缺口，实验尚不足以支撑主要贡献。**

这篇论文有明确的 ICAPS 切入点：人类司机的执行不确定性产生可观测的失败结果，平台再调用可控 AEV 进行同一决策周期内的适应性修复；两阶段行动都受请求互斥、能量和资源容量约束。这个问题比“给网约车加一个神经网络”更具体。正文也已经明确区分固定分数的精确行动优化与动态策略的近似性，这是应该保留的优点。[M, 摘要、§3、§4、§5.1]

理论部分中，Theorem 1 的流等价、Theorem 2 的 witness/forced-row 处理以及量化误差界，按目前写出的假设可以成立。真正需要警惕的是把这些条件性结果延伸为默认 reactive charging 的物理预约保证，或者延伸为当前非线性、off-policy 学习过程的理论保证。[M, §5; S, §2、§4]

实验部分的诚实表述是优点，但不能代替缺少的证据。主文承认只有一个 seed，Table 2 只计 backend solve time，Conclusion 把多 seed 和端到端时间留作 future work。对于以学习效果和可扩展精确行动优化为主要卖点的投稿，这两项应当在投稿前完成，而不是仅在结论中降低措辞。[M, §6、§7]

### 1.1 当前最值得保留的内容

| 内容 | 审查判断 | 修订方向 |
|---|---|---|
| HEV offer → observe execution → AEV recourse | 问题结构清楚，有 planning under execution uncertainty 的解释空间 | 放到摘要和引言的贡献中心 |
| 两阶段 reward ledger 与 closed recursion | 固定策略下数学上成立，避免了第二阶段收益重复记账 | 压缩标准推导，补清其与训练策略的关系 |
| SSG 的真实 witness、严格改进 tie rule、forced rows | 证明中较扎实的部分 | 保留，补已有 reduction 的准确定位及实现验证 |
| 固定图、固定分数上的 exactness 边界 | 表述较谨慎 | 统一到 charging、filtering、学习和实验的各处 |
| Recourse 并不支配所有指标 | 与 Table 1 一致 | 把 reward–service trade-off 解释建立在可复现的奖励定义上 |

### 1.2 修改优先级

P0 表示会影响模型成立、核心贡献解释或主要结果可信度；P1 表示投稿前应补齐；P2 为表达与排版修订。这是本报告的优先级，不是 ICAPS 官方分类。

| 编号 | 优先级 | 问题 | 主要位置 | 最小完成标准 |
|---|---|---|---|---|
| R1 | P0 | reactive 默认实验与 Assumption 2 的关系不清 | M §4；S §1.2–1.4 | 给出两种模式各自的 feasible set、容量含义和保证范围 |
| R2 | P0 | 旧 replay 的 follower 行动与当前第一阶段 Bellman 算子不一致 | M (26)–(28)；S A.2 | 说明行为策略版本，并采用一致的数据/目标处理或明确近似语义 |
| R3 | P0 | 空行动集合时 critic 被强制为零，第一阶段上下文也未说明 | M (23)；§5.3 | 处理空阶段，明确 AEV 资源如何进入 HEV 决策评分 |
| R4 | P0 | operating reward 和 structured score 无具体定义 | M §4、§5.3、§6 | 提供实际使用的奖励、费用、评分与入账时点 |
| R5 | P0 | 单 seed 无法支撑小幅学习收益 | M Table 1、Figure 1、§7 | 多次独立训练/评估，配对差异与离散程度 |
| R6 | P0 | Decoupled 的目标不只是少了“当期修复收益” | S (A.35)–(A.38) | 正确解释递归目标，并补一个使用完整系统收益的强对照 |
| R7 | P0/P1 | SSG 与已有 abundant-vertex / domination reduction 的区别不足 | M §5.2 | 给出具体规则对照，收紧原创性表述 |
| R8 | P1 | backend 时间不能支持端到端 scalable dispatch | M Table 2 | 真实图与压力图上的端到端时间、图规模、内存、尾延迟 |
| R9 | P1 | repeated rejection、completion、service rate 等指标口径不完全统一 | M §6；S §4.2、Figure 2 | 请求级事件账本和清晰分母，消除 first-rejection 歧义 |
| R10 | P1 | 仿真、学习、求解器配置缺失 | M §6.1；S 全文 | 统一参数表、数据处理说明、版本及资源限制 |
| R11 | P2 | 终止样本说明矛盾；引言动作权限概括不准 | M §1、§5.3；S §3.2 | 逐项文字修正，并确认实现与最终表述一致 |

## 2. ICAPS 2027 的实际要求及当前适配情况

已核对 ICAPS 2027 官方 CFP，而非沿用前届规则。长文为 **8 页正文，另加参考文献页**，采用 AAAI 格式、双盲；第 9 页不能出现普通正文。会议要求随机实验报告多次运行及离散程度，并交代硬件、软件版本、基准选择和资源限制。仅仅使用 RL 或处理序贯决策不足以构成 scope，需要说明 substantive planning/scheduling contribution。[E1]

**篇幅：当前正文未显示超页问题。** M 共 8 页，参考文献在第 7 页开始，第 8 页主要是参考文献。因此不能把“整个 PDF 只能 8 页”作为继续删掉实验与模型定义的理由。可以让实质正文使用到第 8 页，再将参考文献后移。S 是独立补充文件；当前 CFP 没有在所查页面明确列出独立补充文件页数上限或保证全部评阅，应在实际投稿系统中确认，不能把“9 页补充材料必然全部被读”当作前提。[M, pp.7–8; E1]

**匿名与格式：** 正文为 Anonymous submission，两个 PDF 的 author 元数据为空，未发现明显作者署名。PDF 外观是双栏布局，但仅凭 PDF 不能认证 LaTeX 源文件没有修改 AAAI 模板；最终仍须检查模板、字体、页边距与匿名资源链接。

**Scope：建议强化而不是重做问题。** 摘要和引言应直接说明：本文研究的是执行不确定性下的 contingent resource-constrained action selection；观测 HEV 执行结果后，修复仍须保持行动互斥、已接受承诺及资源容量。不要以“我们用了 ADP，所以属于 planning”作为唯一论证。也不需要为此强行引入 PDDL 或另做一个无关搜索问题。[M, §3–§5; E1]

**审稿标准：** 显著性、清晰性、与已有工作的关系、可复现性都需要同时成立。没有必要为了增加“理论含量”补一个与主线关系不大的定理，也没有必要加入收入公平约束、ADMM 或新网络模块；先把现有模型、算法和证据连起来。

## 3. 叙事与贡献定位

### 3.1 把主线写成一个问题，而不是三个模块

建议引言围绕以下逻辑展开：

> HEV 的执行结果在平台的第一阶段行动之后才揭示，因此可控 AEV 的资源价值取决于随机的 residual state。平台不仅需要在结果揭示后找到可执行的修复方案，还需要在形成 HEV offers 时考虑这些后续行动的价值。本文用阶段化价值学习处理前者对初始决策的影响，用受约束的精确行动优化保证每次输出可执行；图 reduction 只降低该行动优化的计算负担。

这里要准确区分“阶段化价值学习”和“精确行动优化”。GAT、twin critics、replay、量化及 MCMF 不应被逐一包装成独立创新。SSG 的贡献则取决于其相对于现有 reduction 的具体新增内容和实测收益，不能由名称自动成立。[M, §1、§5]

### 3.2 Proposition 1 正确，但不宜承担主要理论新颖性

(4) 由 (2)–(3) 和 tower property 得到。它能解释为何第一阶段应接收两个阶段的已实现收益，并说明不能对同一收益再通过 Q2 重复 bootstrap；这是必要的正确性说明。但“把两个阶段的递推代入”本身不构成很强的新理论。[M, §3; S, §2.1]

建议正文保留两个价值递推及简短命题，把详细 sigma-field 证明留在补充材料。省出的篇幅用于回答：当前第一阶段 scorer 能看到什么 AEV 信息？旧 replay 对应哪个 follower？何种实验能区分收益记账和适应性修复的作用？

### 3.3 Related Work 应当形成技术区别，而非背景清单

至少要在已有引用中区分三类最近工作：电动车队的 ADP 行动协调、学习价值与组合分配结合的方法、带匹配失败的动态随机匹配。外部核对显示，Al-Kanj 等的模型已经联合考虑服务、充电、重定位与停车；Shah 等已经将神经价值近似与组合分配结合；You–Vossen 已考虑可能失败的动态匹配。因此，不能把上述任一宽泛组合表述为本文独有。[E3–E5]

正文应突出更窄的差异：本文在已明确的同一 epoch 信息结构内，让 HEV offer 的结果先揭示，再进行请求级 AEV 行动修复，并将其价值反馈给初始决策。是否具有充分的新颖性，需要结合第 6 节的目标分析与第 8 节的实验回答，不能只依赖术语。

### 3.4 一处明确的动作权限表述需要修正

引言称“两阶段都在 service、battery replenishment、relocation、waiting 上使用精确 optimizer”。但 §4 的 Stage 1 只有 request offer / no offer；HEV 充电和 idle relocation 是 simulator outcomes，并非平台控制。[M, §1 末段、§4、(9)–(10)]

建议改为：

> The first stage optimizes HEV request offers and no-offer decisions, whereas the second stage jointly selects feasible AEV service, charging, relocation, and waiting actions.

不要为迁就引言这句话而给 Stage 1 新增当前模型没有的控制变量。

## 4. 最重要的模型问题：charging 模式与保证范围

### 4.1 R1：默认 reactive 实验并未自动满足 Assumption 2

Assumption 2 要求：任何满足 station–arrival quotas 的组合选择，都能对应完整服务窗口内的联合物理预约。可是紧接着的段落说，默认 reactive admission 只检查当前已有承诺，当前新选的候选可能互相竞争并形成队列。[M, p.3, Assumption 2、(14)、Charging admission rules]

补充材料其实已经正确给出反例：一个物理 slot，两个车辆分别在 1 和 2 到达，各服务 2 个时间单位；分别看到空位并不意味着可以同时预约。因此，问题不是缺少这个反例，而是正文没有明确说明默认 NYC 实验究竟使用哪个容量模型。[S, p.1, §1.2]

**必须选择清晰的陈述方式：**

| 模式 | 输入 optimizer 的约束 | 可以主张的结论 | 不能直接主张的结论 |
|---|---|---|---|
| 预分配 token/envelope quotas | 满足 Assumption 2 的独立容量 | 图内 assignment 精确；所选充电集合有可行物理预约 | 全部可能预约方案中的最优 |
| Conservative virtual-calendar admission | 先对全体候选分配不冲突虚拟窗口，再筛选 | 保留的候选任取合法子集不会制造预约冲突 | 未筛选物理模型的最优、没有充电延期损失 |
| Reactive admission | 需要明确其实际 admission quotas 和排队状态转移 | 固定独立资源图上的最优选择；按其排队规则执行 | 未经证明的 arrival-time 开始服务或 Assumption 2 保证 |

这不要求删掉 reactive。可以将其保留为主要运行环境，但必须明确：此时 flow exactness 是对编码的行动选择问题而言，物理排队与实际开始时间由另一个完整说明的状态转移处理。相反，如果希望把无排队物理预约作为主要模型承诺，就应使用相应的 conservative/quota 模型运行主实验。

**仅加一句“the theorem is conditional”不够。** 还需要给出 Table 1 的具体模式、(14) 在该模式下的 K 如何构造、队列中车辆和 inbound vehicles 如何进入状态，以及所用持续时间是否包含等待。

### 4.2 确定性速度不等于固定的无排队完成时间

在 reactive 模式下，服务时长本身可以确定，但等待时长仍可能依赖联合行动。用本报告的诊断记号表示：

$$
\text{finish}_{k,c}(Y_t)
=
\tau_t+\Delta_{k,c}
+w_{k,c}(Y_t;S_t)
+h_{k,c,t}.
$$

这里的等待量 w 不是作者已定义的变量，而是用于说明必须解释的机制。如果算法评分采用无等待到达/结束时刻，而仿真执行时会排队，那么这是 scoring approximation；如果状态转移也忽略了等待，则是另一种、更严重的建模问题。仅凭当前 PDF 不能判定属于哪一种。[M, Assumption 1、§5.3; S, §1.3–1.4]

应明确服务先后规则、同时到达 tie-breaking、充电完成与新到达的事件顺序，以及真实等待后的车辆 availability。不要把“速度、能耗和充电功率为常数”写成这些问题已被解决的证明。

### 4.3 A.1 和 A.2 是两种构造，不能默认为完全相同

A.1 使用 arrival-group envelope 和 token；A.2 使用候选车辆的实际窗口与 plug identity。两者都有充分可行性证明，但并非同一个算法，也不必产生相同可行集合。[S, §1.2–1.3]

特别是 A.2：全部获准候选的虚拟窗口已经共同可行，因此优化器只选其子集时，物理容量实际上已由上游候选生成处理。如果继续保留 station–arrival 节点，应说明节点容量与该候选集合的对应关系；不能再把任意 raw free capacity 当作同一保证。

建议只把实际使用的一种构造写成主算法，另一种放在“alternative sufficient construction”。对应实验必须写清调用哪一种。

### 4.4 无可行行动的运行恢复不能只交给“假设可行”

Theorem 1/2 正确允许实例不可行，且 SSG 不会伪造 wait。但是 Algorithm A.2 假设每次输入可行，正文没有说明低电量车辆既不能 wait 又未获充电 admission 时怎么办。[S, pp.5、7–8]

应报告真实仿真中此类状态是否出现；若出现，给出运行恢复动作及计价、计时规则。若采用应急停车、拖车或延迟入站，必须在物理模型中合法地定义，不能给 forced row 偷加零分 dummy。若通过前序安全策略排除，则说明该安全策略及其失败次数。

### 4.5 “站内队列为零”不等于没有等待

S Figure 1 的 conservative 零队列与 A.2 一致，不是异常。但 admission 之前被拒绝或推迟的车辆不计入站内队列，因此需要同时报告充电需求出现到开始充电的总延迟、延后车辆数、低 SoC 滞留和未完成补能需求。[S, §1.4、Figure 1]

当前相近的 completed sessions 曲线只支持所测场景内吞吐量相近，不能推出一般条件下无服务损失。原文已作部分限制说明，应保留，并补充上游延期指标。

## 5. 定理与证明的逐项审查

### 5.1 已成立的证明链，不需要推倒重写

| 结果 | 审查结论 | 需要保持的条件 |
|---|---|---|
| Proposition 1 | 按给定 Markov sufficiency 和可积性成立 | Y 必须对应声明的 policy；区分数学固定策略与训练中的行为策略 |
| Theorem 1 | assignment–flow 双向映射与积分性成立 | 每条边只消费一个编码资源，无额外跨资源耦合约束 |
| Lemma A.1 | witness ownership invariant 成立 | 只在严格改进时换 witness，batch 使用旧 baseline |
| Lemma A.2 | domination 与 forced-row retention 成立 | 不给 null witness 构造可执行零分动作 |
| Theorem 2 | 目标偏移与可行性双向论证成立 | 固定分数、容量、候选和真实 witness 身份 |
| Proposition A.1/A.2 | 各自的充分可行性证明成立 | token/plug calendar 的明确构造及后续承诺被遵守 |
| Proposition A.3 | n/L 的固定 assignment 量化误差界成立 | 每个 complete assignment 恰选 n 条边，统一最近整数舍入 |
| (A.41) 的联合梯度 | 与 (28) 的系数一致 | 对 target stop-gradient；共享 encoder 汇总两 critics 的梯度 |

依据：[M, §3、§5; S, §1.2–1.3、§2、§4.2]。

SSG 的关键不是“把共享动作直接当独立 wait”，而是先证明某个已删除资源最多被容量允许数量的车辆持有，再利用这些 witness 的任意子集仍可行。这个证明结构是正确的。原问题到 reduced problem 的方向允许把劣选项替换成较优 witness，因此保留的是最优值与可恢复最优解，而不是所有原可行解或全部 tied optima。[S, §2.3]

### 5.2 独立枚举检查的范围

本次另写了一个与作者实现无关的小型检查器，对 **2,000 个随机实例**逐个穷举原 assignment 和 reduced assignment。车辆数及资源数均为 0–5，随机有缺边、零容量、负分数、同分数和缺少 private fallback 的车辆。1,367 个实例可行、633 个不可行；未发现最优值、可行性判定或最优解展开不一致。另检查了 5 个定向边界例子，包括共享零分 tied action、无 fallback 但可行、无 fallback 且不可行、空图等。

这个检查只增加对**文中逻辑规则**的信心，不替代一般证明，不验证作者的 pointer/vectorized 实现，也不验证超过四轮时的实际 fallback 分支。本批随机例子的最大 folding 轮数为 4，不能据此声称深 cascade 实现已经通过测试。

### 5.3 SSG 的新颖性必须与已有规则正面对照

正文目前只写“drawing on the general idea of exact reductions”，比较不足。已核对的 Großmann 等公开版本包含 **Abundant Vertices** 和 **Weighted Domination** 等具体规则；2026 年期刊条目也已确认。它不是只有泛泛的“图压缩思想”。[M, §5.2; E2a–E2b]

一个需要在论文中正面解释的关系是：当所有车辆都有真实 private baseline 时，可先对每行减去 baseline、去掉非正增益边。此时 SSG 的条件

$$
|B_a(\ell)|\le u_a
$$

就是在当前 improving-edge graph 上识别容量不紧的资源。删除这种资源后，原本的 vehicle–resource 选项可成为仅依赖该车辆的局部备选；再用这个备选压掉较低分边，继续产生新的容量松弛。这与 abundant-vertex removal、局部 baseline 和 dominance 的组合非常接近。

这是本报告基于两者规则做出的结构分析，**不是已完成对所有 signed-score、forced-row、batch 实现的等价证明**。因此不能直接断言 SSG 完全没有新内容；同样，也不能把它未经比较地宣传成全新的普适 matching reduction。

最小修订要求是：明确本文是否贡献了一个 specialized composition、无 fallback 时的安全完整分配处理、可执行 witness 的恢复，或者更低开销的事件驱动实现；给出这些差异各自的证明或实验。可以增加简单的 static capacity-only reduction 作为对照，检验 score-driven cascades 的额外作用。无需为此引入一套庞大的 hypergraph 系统。

### 5.4 复杂度：逻辑规则与实际实现仍须分开

(A.29) 的界依赖 event-driven degree updates、每条边只跨过一次的单调 pointer、以及避免逐轮扫描所有剩余边。正文里的逻辑 Algorithm A.1 本身没有自动保证这些实现性质。[S, pp.5–7]

应在代码材料和实验中给出 preprocessing round 数、pointer fallback 触发率，以及是否重建 dense score matrix。四轮 fast path 后若还有 foldable batch，应继续至 fixed point 或按文稿说明重启 pointer routine，不能将“四轮”误实现成数学算法的终止条件。

### 5.5 不应新增或暗示的保证

SSG 不保证不同最优 tied action 的动态后果相同；保留一个 fixed-score optimum 不等于保留原 solver 的整条仿真轨迹。对不同 backend，可以要求逐实例目标一致及可行性一致，但不应要求动作 identity 必然相同。

量化界是每个固定 assignment 的分数损失界，不是多周期 policy regret。候选 top-K、conservative mask、additive value approximation 和行为模型误差也没有被 n/L 吸收。[M, §5.2; S, pp.5–6]

## 6. 学习目标：当前最需要补清的理论连接

### 6.1 R2：linked replay 解决事件错配，但不解决 follower 策略错配

Proposition 1 对固定策略成立。当前训练却从 replay 中读取实际发生的第二阶段行动 Y、R2 和下一状态，再用更新后的 first-stage target value 构造 (26)。[M, (26)–(28); S, Algorithm A.2、§4.2]

令 μ2 表示生成旧记录的 follower，π2 表示当前 follower。暂时将下一 epoch 的 continuation 记为 V。条件于第一阶段状态和行动，旧样本对应的算子为

$$
(\mathcal T_{\mu_2}V)(s,x)
=
\mathbb E_{\xi\mid s,x}
\left[
R^1+
\mathbb E_{Y\sim\mu_2(\cdot\mid\bar S^2)}
\left[R^2+\gamma^{\Delta t}V(S^+)\mid\bar S^2,Y\right]
\right].
$$

把 μ2 换成当前 π2，一般会改变第二阶段奖励及下一状态的条件分布。因此

$$
\mathcal T_{\mu_2}V\ne\mathcal T_{\pi_2}V
$$

在一般情况下成立。只把旧记录的 next state 链接正确、只更新 V 的网络参数，或者重新计算 reduced graph，都不会把已经发生的旧 Y 变成当前 Y。

这与“第一阶段行动 X 来自旧 policy”需要区分：Q-learning 可以条件于被记录的 X 更新该行动价值；这里的问题是 Q1 在自身一步转移内部还包含一个正在改变的 follower 决策。Q2 的 replay 则条件于其记录的 Y，具有另一层语义。

**不应据此说 Proposition 1 错误，也不应把它简化成一般的 nonlinear convergence 问题。** 文稿真正缺少的是：当前算法在估计哪个 follower 下的 Q1，以及如何处理 follower 非平稳性。

### 6.2 三种处理路径，任选与真实实现一致的一种

**路径 A：保留 sampled closed target，采用 follower-frozen 的数据区间。** 在某个学习区间固定 follower；leader 更新使用该 follower 生成的数据，并对旧 follower 版本的数据清理、分层或重新采集。这里仍可能有普通函数近似误差，但当前阶段 reward/transition 的策略语义是清楚的。仅给 transition 写一个版本号却照样混用所有版本，不构成修正。

**路径 B：使用阶段一致的 Q2 bootstrap。** 这是一个可选算法修改，不是声称作者当前已经采用。当前 residual state 上先选可行 follower action，再用 target Q2 评估：

$$
\widehat Y_t\in
\arg\max_{Y\in\mathcal F^2_t(\bar S_t^2)}
\sum_{(k,a)\in Y}\Psi^2_{k,a},
$$

$$
T^1_t
=
R^1_t+
\min_{r=1,2}Q_{2,\bar\theta_{2,r}}(\bar S_t^2,\widehat Y_t).
$$

第二阶段仍使用

$$
T^2_t
=
R^2_t+(1-d_t)\gamma^{\Delta t_t}\bar V_1(S^1_{t+1}).
$$

第一式内部不再额外加一次 R2，也没有 within-epoch 折扣。该路径会增加 target-side follower projection 的成本，而且 Q2 仍是近似值，需要在实验中检查，不能声称自动得到收敛保证。

**路径 C：保留现有算法，准确承认其行为混合近似。** 将 (26) 描述为 evolving-policy replay 下的 recourse-aware semi-gradient target，不把它声称为当前 greedy follower 的无偏 Bellman 样本。至少提供 replay age / follower refresh 的敏感性或控制实验。对于强调理论连接的投稿，仅加一句笼统 disclaimer 不如 A/B 有说服力。

### 6.3 R6：Decoupled 不只是“少记当期 AEV repair”

A.35 对 R1/R2/R3 的第一阶段写的是

$$
T^m_{1,t}=R^1_t+D_t\bar V^m_1(S^1_{t+1}).
$$

由于 continuation 仍由同一类第一阶段目标学习，在固定策略、理想精确评估的诊断条件下，它递归对应的是 HEV-side reward 的长期回报：

$$
V^{m,\pi}_{1,t}(s)
=
\mathbb E^{\pi}\left[
\sum_{u=t}^{T-1}
\gamma^{\sum_{v=t}^{u-1}\Delta t_v}R^1_u
\mid S^1_t=s
\right].
$$

它不是一个准确的“系统总价值”只在当前一步漏加 R2。因此，R3 follower 的 (A.36) 也变成当前 AEV reward 加未来 HEV-side continuation，而不包含一般意义上的全部未来 AEV rewards。[S, §3.2]

这不使 Decoupled 无法作为设计消融；它仍然是很有用的 **HEV-only leader credit ablation**。但是，Recourse 击败它既可能反映 adaptivity 的好处，也可能反映“学习目标终于与评估的系统总收益一致”。不能把这个对照单独解释成新的 recourse reasoning 优势。

建议至少补一个使用完整系统收益的强对照，例如：在 Learning r2 的 structured follower 不变时，把 leader target 改为完整 R1+R2 target。这样可以同时检查 learned follower 和 coupled credit，而不是只与目标本来就不同的 leader 对比。

### 6.4 R2 是全部 AEV-side reward，不只是 rejection repair reward

原文已经明确 R2 包括 never-offered requests 的 AEV service。因而将 R2 反馈给第一阶段，同时改变了对以下内容的 credit：拒单修复、从未给 HEV 的请求服务、以及 AEV 的其他计价动作。[M, §4; S, §4.2]

可在日志中分解为

$$
R^2_t
=
R^{2,\mathrm{rej}}_t+
R^{2,\mathrm{rem}}_t+
R^{2,\mathrm{other}}_t,
$$

其中分类必须与实际奖励定义一致；其他项可含服务以外成本，不能凭空新增环境奖励。

**一个重要的实验解释边界：即使所有 HEV 都接受 offer，加入完整 R2 credit 也不必与 uncoupled 方法等价。** AEV 仍可服务 never-offered requests。因此，“拒单率为零时两者差距必须为零”不是本框架的必然性质，不应作为硬验收标准。

### 6.5 R3：空阶段导致 additive critic 的表达问题

按 (23)，若某阶段没有可重新决策车辆，则 selected-edge sum 与 structured-edge sum 都为空，于是

$$
Q_{h,\theta}(s,\varnothing)=0.
$$

但按 (2)–(3)，这个状态的真实阶段价值通常不为零。例如没有 available HEV 时，AEV 仍可能在随后阶段服务；没有 available AEV 时，下一 epoch 仍可能有正的 continuation。正文没有排除这些状态。[M, §3、§4、(23)]

最小修正之一是增加 action-independent state-value readout：

$$
Q_{h,\theta_{h,r}}(s,A)
=
B_{h,\theta_{h,r}}(s)+G_h(s,A)
+
\sum_{(k,a)\in A}Z^h_{\theta_{h,r}}(f^h_{k,a}).
$$

B 不影响同一状态下的 assignment argmax，因此无需改变 flow/SSG，也不是给无 fallback 车辆增加一个不合法动作。它改变的是 critic 的价值表示与训练，必须同步修改 target evaluation 与 gradient。另一条可行路径是明确将空阶段跳过并正确累积奖励/折扣，但这也需要写入转移定义。

### 6.6 第一阶段如何感知 AEV 的修复能力

§5.3 写的是 full feasible decision graph，但 Stage 1 的行动图主要是 HEV–request edges。需要明确 AEV 的 available locations、SoC、忙碌时间、可服务 residual request 的能力和 station availability，究竟通过哪些 context nodes、global features 或 pooling 进入 HEV edge features。[M, §4、§5.3]

否则，对于 HEV 与请求图相同、但 AEV 数量/位置不同的两个状态，first-stage scorer 可能输出同一组分数，而真实的 recourse value 应当不同。这是需要检查的架构信息缺失，当前文件不足以认定代码确实没有这些输入。

有 global context 也不代表可加分解能够表示所有 request-competition interactions。保留原文“additive approximation”的限制；不要把 flow 的精确性当作价值分解精确性的证明。

### 6.7 终止样本说明必须统一

正文说只使用有有效 next-state link 的 nonterminal replay records；补充材料又明确 terminal samples 保留当前阶段 rewards、只将 continuation 置零。两种文字描述不一致。[M, §5.3 末段; S, §3.2]

建议改成：

> Terminal records retain the realized rewards and use zero continuation. Nonterminal records are used only when the actual next-state link is valid.

实现上应在调用 next-state optimizer 前屏蔽 terminal 项，而不是先对不存在的 next graph 求解，再指望乘以零消除错误。

### 6.8 一个较小的数学记号问题

固定 deterministic π 下，把 Qπ(s,x) 写作条件于 X=x 的期望是常用简写，但对于不由 π 在 s 选择的行动，条件事件可能概率为零。建议补一句：Qπ(s,x) 表示当前执行 x、随后遵循 π 的 action value，或直接用 transition kernel 定义。无需为了这个记号问题重写整个理论。[M, (2)–(4)]

## 7. 奖励、仿真和可复现信息

### 7.1 R4：必须给出真实 operating reward 和 structured scores

17 页材料中，对 R1/R2 有记账原则，对 Gh 有可加定义，但没有足够具体的奖励和 structured edge score 公式，使他人能够复现 Table 1 的美元结果及 Myopic baselines。[M, §4、(23)–(24)、§6; S, §3–§4]

至少要回答：HEV 部分是平台抽成、全额车费还是其他系统收益？AEV 车费减哪些成本？充电、换电、重定位和 wait 如何计价？g 是否使用接受概率、未来需求或启发式 bonus？奖励在 assignment、pickup、completion 的哪一刻记账？取消、跨日未完成和忙碌车辆的收益如何分到两个阶段？

写一个核对表即可，内容必须从实际实现提取，不能为了让结果看起来合理而补写从未运行的 reward：

| 动作/事件 | 环境奖励项 | structured score g | 入账时点 | 参数/单位 |
|---|---|---|---|---|
| HEV offer accepted/rejected | 按实际实现填写 | 按实际实现填写 | 按实际实现填写 | 费率、成本、概率 |
| AEV request service | 按实际实现填写 | 按实际实现填写 | 按实际实现填写 | 车费、里程/时间成本 |
| Charging / swapping | 按实际实现填写 | 按实际实现填写 | 按实际实现填写 | 电价、固定费、时长 |
| Relocation / wait | 按实际实现填写 | 按实际实现填写 | 按实际实现填写 | 距离成本、惩罚 |

若 g 含未来需求预测或补能启发式，“Myopic”应解释为不学习 continuation 的 structured heuristic，而不是严格只最大化当前环境 reward。当前论文已有“structured scores”的说明，但必须给出这些分数的具体定义。

### 7.2 当前 reward 差异值得解释，但不构成造假或 bug 证据

根据 Table 1 已舍入值计算，下面是**日均收益与日均完成数之比**，不是逐订单净利润估计：

| 方法 | 总收益/总完成数（约，美元） | HEV 收益/HEV 完成数 | AEV 收益/AEV 完成数 |
|---|---:|---:|---:|
| Learning r1 | 28.79 | 36.68 | 15.07 |
| Decoupled r3 | 27.80 | 35.08 | 21.71 |
| Recourse | 26.59 | 35.34 | 20.83 |
| Learning r2 | 19.78 | 23.78 | 17.74 |

这些差异可以由服务订单的价格/距离结构、平台计价方式、非服务成本及入账时点产生。不能单凭它们认定“HEV reward 算错”。但它们表明：只说 Recourse 的 reward 高而 service rate 低，尚不足以解释机制。建议报告两类车的已服务行程距离/费用分布，以及环境 reward 的分项。

Table 1 的总收益与分项之和存在小幅末位差异，在当前有效位数下可由舍入解释；不能作为确定的会计错误。应在未舍入日志中检查精确 reconciliation，再决定是否需要修表。[M, Table 1]

### 7.3 最少需要补充的配置

| 类别 | 目前必须补清的内容 |
|---|---|
| 数据 | 数据集具体名称、文件/版本、筛选标准、Manhattan 定义、时间区间、时区、请求 release/截止时刻构造、是否采样 |
| 行驶 | 固定速度具体值、距离来源、路线计算、时间取整规则、30 秒内事件顺序 |
| 车辆 | 初始位置与 SoC 分布、可用/忙碌/离线定义、Qmin、充电目标、HEV 自主充电及 relocation 规则 |
| 站点 | 数量、位置、各站 slot/plug 数；HEV/AEV 是否共享；charging 与 swapping 哪个用于哪组实验；固定 swapping duration |
| 接受机制 | 接受概率模型、参数、driver heterogeneity、结果是否有时间相关性、是否从数据校准 |
| 学习 | GAT 层数/维度、critic 架构、全局上下文、discount 的时间单位、优化器、batch、replay 大小、更新频率、训练轮数、探索策略、target update rate |
| 数据划分 | 验证集、checkpoint 选择、hyperparameter 搜索范围，预测器是否仅使用训练/过去信息 |
| 求解器 | OR-Tools/CPLEX 实际测量版本、CPU/GPU、线程、OS、presolve/算法、容差、时限、内存限、计时边界 |

正文目前给出了部分电池参数、日期和 seed，但这不能替代上述配置。[M, §6.1; S, §2.3、§4]

尤其不能把 Table 2 中“battery-swapping stations”的设置默认解释成 Table 1 中的 20 kW charging 模型。两个实验可以不同，但要分别命名并交代参数。

## 8. 实验：最少补哪些，才能支撑论文主线

### 8.1 R5：多 seed 必须针对学习效果，不可由充电实验代替

Table 1 使用三天、单一 seed 256。S 的充电实验虽然使用 10 paired seeds，但它明确不比较 learned dispatch policies，因此不能为 Table 1 的学习收益提供跨 seed 的不确定性证据。[M, §6.1–6.2; S, §1.4]

建议主要策略至少使用 5 次独立训练，预算允许时增至 10 次；这是本报告的实验建议，不是官方指定数量。每个训练 seed 在相同固定测试日和配对环境随机数下评估。先对每次独立训练的固定测试集表现聚合，再以训练重复为主要随机单位比较。

令 Jm(s,d) 为方法 m、训练重复 s、测试日 d 的日收益，重点报告

$$
\Delta_s
=\frac{1}{|\mathcal D|}
\sum_{d\in\mathcal D}
\left[J_{\mathrm{Recourse}}(s,d)-J_{\mathrm{baseline}}(s,d)\right].
$$

给出 Δ 的均值、离散程度和适当区间，说明独立单位。不能把每 30 秒的 epoch 或同一训练模型下的所有车辆当成独立重复；三个固定日期也不自动代表对任意未来日期的总体推断。

优先比较 Recourse–Decoupled r3、Recourse–Learning r1，以及新增的 full-reward structured-follower 对照。当前 Recourse 比 Learning r1 约高 1.1%，在没有 run-to-run variation 时，只能作为描述性结果。[M, Table 1]

### 8.2 用最小的因子设计补齐缺口

不用新增十几种 RL 算法。现有基线已经有一个差一格的 2×2 设计：

| follower scoring | uncoupled leader target | full-reward leader target |
|---|---|---|
| Structured | Learning r2 | **需要补充的 coupled structured-follower 对照** |
| Learned | Decoupled r3 | Recourse |

这四格共同保持 same-epoch rejection eligibility，分别回答 learned follower 与 full-system credit 的作用。现有 Learning r1 可以继续作为 restricted-eligibility 对照，但它与 Learning r2 同时改变两个因素，不能拿两者差值解释单一机制；原附录对此已经提醒，应保留。[S, §3]

进一步的 eligibility-only 学习对照应保持 scorer architecture、训练目标与预算一致，只改变 rejected-request eligibility。算法设计层面的隔离不等于训练后的模型数值、轨迹或优化路径完全相同，应避免因果措辞过强。

### 8.3 增加一个可枚举的小例子，比泛泛新增定理更有价值

以下是本报告构造的诊断实例，不是论文现有实验数据：一个 HEV、一个 AEV、两个请求，无后续需求；每辆车最多服务一个请求，AEV 在结果揭示后选择尚未被接受的请求。

| 请求 | HEV 接受概率 | 接受后的 HEV reward | AEV service reward |
|---|---:|---:|---:|
| j1 | 0.5 | 20 | 10 |
| j2 | 1.0 | 9 | 6 |

HEV-only greedy 偏好 offer j1，因为 0.5×20=10 大于 9。若 offer j1，接受时 AEV 服务 j2，拒绝时 AEV 服务 j1，总期望为

$$
0.5(20+6)+0.5(0+10)=18.
$$

若 offer j2，则 AEV 服务 never-offered j1，总收益为

$$
9+10=19.
$$

因此完整两阶段评价会改变第一阶段选择。固定 offer j1 后，允许 AEV 观察结果再行动，其第二阶段期望为 8；若事先承诺一个固定请求，且与 HEV 接受冲突时只能取消，则最佳固定 AEV 选择的期望为 6。这个局部对照还能单独说明观察后适应的价值。

用这样的小实例检查：exact enumeration、两阶段账本、learned score 的方向、以及算法在不同 AEV 可行边下的响应。它不能证明大规模学习收敛，但能确认论文所说的机制确实被实现。

### 8.4 只做与研究假设直接相关的敏感性

优先改变 HEV 接受水平、AEV 比例/可用性和 charging capacity 中的少量代表性条件。保持总需求和其他配置可比，报告 reward、service、真正的 same-epoch repair、总补能延期和运行时。

这不是要求覆盖一个巨大的参数网格，也不是把论文扩展成交通系统设计。目标是说明何时 recourse 资源紧张、何时满系统 credit 有用、何时 conservative admission 的代价明显。

### 8.5 不能只把主结果写成 future work

需要保留“描述性、单 seed”的诚实限制，直到重跑完成；不能先把 wording 改成 robust/generalizable。但论文投稿前，应当把多次独立运行和真实端到端计时从 future work 移入已完成的验证。若资源确实不足，就必须相应缩小论文的核心贡献，而不是只在结尾道歉。[M, §7]

## 9. Solver scalability 的具体审查与重跑方案

### 9.1 Table 2 实际说明了什么

依据表中已舍入时间，MCMF 分支的结果如下。这里只重新计算表内数值，不是新 benchmark：[M, Table 2]

| NV | Full MCMF solve（ms） | MCMF+SSG solve（ms） | Full/Reduced | solve 时间节省 |
|---:|---:|---:|---:|---:|
| 100 | 2.76 | 5.49 | 0.503× | −2.73 ms |
| 500 | 39.22 | 70.23 | 0.558× | −31.01 ms |
| 1,000 | 151.22 | 136.70 | 1.106× | 14.52 ms |
| 2,000 | 679.38 | 598.48 | 1.135× | 80.90 ms |
| 3,000 | 1,520 | 1,330 | 1.143× | 190 ms |
| 6,000 | 6,630 | 5,380 | 1.232× | 1,250 ms |

小规模 reduced solve 反而慢不必然是错误；图拓扑、容量与费用结构变化可改变 backend 的工作量。但**表内计时排除了 preprocessing，不能用“SSG preprocessing overhead”直接解释表内 solve-call 变慢**。应通过 backend profiling 或重复计时确认。

较大规模时，现有结果支持 backend 有一定改善，但还没有支持 SSG 端到端更快。反过来，也不能单凭 preprocessing 比上表节省时间大就判定端到端更慢，因为 reduced graph 可能节约建图、传输和解码成本。

### 9.2 正确的计时边界

分别报告 assignment pipeline 与整个 dispatch pipeline：

$$
T_{\mathrm{assign}}
=T_{\mathrm{quantize}}+T_{\mathrm{reduce}}
+T_{\mathrm{build}}+T_{\mathrm{backend}}
+T_{\mathrm{decode/check}},
$$

$$
T_{\mathrm{dispatch}}
=T_{\mathrm{candidates/admission}}
+T_{\mathrm{encode/score}}
+T_{\mathrm{assign}},
$$

并按需要汇总同一 epoch 两个阶段的实际墙钟时间。报告 median、p95/p99、最大值、超时/不可行次数和峰值内存；训练更新的时间应单独列出，除非部署确实在线训练且占用同一计算资源。

30 秒决策频率是系统约束，不是可以省略 tail latency 的理由。平均 solve time 低于 30 秒，也不能证明整个 pipeline 每次按时完成。[M, §6.1、Table 2]

### 9.3 relocation-rich 压力图不能替代真实 NYC 图

Table 2 中 NR 随 NV 取 2NV，最大 NC=1,200，且这些 NC 被称为 swapping stations。这可以作为抽象压力测试，但不能默认代表 NYC 的站点和 relocation-zone 配置。[M, Table 2]

建议保留压力测试，同时从真实 NYC rollout 固定保存两阶段图，覆盖高峰、低峰、不同 charging 紧张程度、不同 forced-row 比例。所有 solver 使用同一份 fixed graph 和同一组整数分数，避免用不同 learned trajectory 比 solver。

还要区分物理 station 数和 station–arrival resource nodes 数。§5 的 m 统计编码资源节点，若一个站有多个 arrival groups，NC 并不是 charging action nodes 的实际数量。[M, §4、§5.1]

### 9.4 图缩减效果不能只用时间间接推测

每个实例至少保存 n、m、e，以及 reduced m/e、folding rounds、KB/KF 数量、reduction 时间、backend 时间和 unfolded objective。报告不同类型资源与边的删除比例。

否则无法判断 SSG 在 relocation-rich 图中主要删除的是原本就无限容量的 relocation 节点，还是经过 score-driven cascades 才减少的真正竞争结构。无 private fallback 的理论扩展也需要对应测试实例，而不能只在人人有 wait 的 synthetic benchmark 上测量。[S, §2.3]

### 9.5 CPLEX 对照要明确到底解 LP 还是 MIP

Theorem 1 已给出 integral LP optimum。Table 2 的 CPLEX 一栏因此至少应包含同一 network formulation 的连续变量求解，交代算法/presolve/线程；若另跑 binary assignment MIP，可以作为补充，但不能只靠一个未说明的 MIP 设置来证明专用 flow solver 的普遍优势。[M, §5.1、Table 2]

CPLEX 与 OR-Tools 应使用相同量化分数、同一可行边集与容量。记录求解状态、容差和 objective gap；不能只用保留三位有效数字的平均 reward 相等作为 exactness 实证。

### 9.6 内存与 dense 输入是实际风险，而非已证实的错误

仅按最大实例所列 30,000 requests、1,200 stations 和 12,000 relocation zones 各对应一个资源估算，6,000×43,200 已是 2.592 亿个 dense entries。一份 8-byte 矩阵约 2.07 GB；raw scores 与整数矩阵并存时还要翻倍，尚未计 GAT、索引和图结构。

这是对表中规模的条件性算术估算。实际资源节点可能因 arrival groups 更多，实际存储也可能是 sparse，因此不能据此断言作者程序内存不够。但既然附录给出 O(nm) dense space，峰值内存、稀疏候选构造和 GPU batch 方法不能省略。[M, Table 2; S, (A.29)]

## 10. 图表与指标：哪些是真问题，哪些不能误判

### 10.1 first-rejection epoch 与 current-round rejection 存在歧义

S §1.1 明确按当前 offer round 划分 Jacc/Jrej/Jrem；正文的 same-epoch repair 也按当前 HEV-rejection epoch 说明。可是 S §4.2 写的是 assignment epoch 与 **first-rejection epoch** 相同。[M, §6.1; S, §1.1、§4.2]

若同一请求允许在 t 被拒、在 t+1 再被 offer 并被拒，然后由 AEV 在 t+1 服务，它属于当前轮的 repair，却不属于“首次在整个生命周期被拒的 epoch”的 repair。当前文本没有声明每个请求一生只能被 offer 一次，因此需要统一定义。

若 first-rejection 指当前轮的事件字段，应重新命名。若确实指首次 lifetime rejection，则需要给出相应统计解释，不能把它与模型的 current Jrej 直接等同。

### 10.2 建议明确的同周期修复统计

对每个 epoch，按模型定义当前拒绝集合，再统计当前轮由 AEV assignment 的其中请求：

$$
N^{\mathrm{repair}}_t
=
\sum_{j\in\mathcal J^{\mathrm{rej}}_t\cap\mathcal J^{\mathrm{live}}_t}
\mathbf 1\{j\text{ is assigned to an AEV in epoch }t\}.
$$

跨 epoch 的日汇总是否去重，必须单独说明：按 request–epoch 事件计数，和按 unique requests 计数是两种指标，不能 numerator 用事件数、denominator 用唯一请求数。还应将所有 rejected requests 的 coverage 与仍有可行 AEV edge 的 rejected requests coverage 区分，后者更接近可修复机会下的利用率。

### 10.3 service rate 的均值不等于均值之比

正文已经说 service rate 是 mean of daily rates。因此，不应拿表中日均 completed counts 除以表中 service rate，反推出略有差别的需求总量，再宣称各方法用了不同数据。[M, Table 1、§6.1]

正确做法是保存每一天的 released、assigned、picked-up、completed、expired、cancelled、end-of-day active 请求数。仅当这些集合互斥且同一 cohort 时，才能写守恒等式。尤其要明确跨日未完成订单的归属。

### 10.4 Figure 1 有信息重复，也有需要保留的机制指标

每日 reward 在 Table 1 和 Figure 1 重复出现。建议删除重复 reward bar，为 paired reward difference 的区间图或一个机制消融留空间。拒单未修复数和 recourse coverage 值得保留，但要按前述定义统一；平均日末 SoC 是状态描述，不能单独说明能效或更好的充电策略。[M, p.6, Figure 1]

EV/HEV 命名应在图、表和正文统一，而不是每个 caption 都解释一次。

### 10.5 补充 Figure 2 的纵轴没有充分定义

“Hourly request-count profiles for six policies”应明确画的是 released demand、assignments、pickups 还是 completions。不同政策不会改变固定的外生 released requests，因此若六条 policy 曲线不同，它们应对应某个服务事件，而不能统称需求。[S, p.9, Figure 2]

KDE demand curve 与原始每小时服务 counts 不是同一统计量。某些服务曲线高于当小时 arrival curve，可能来自等待请求、跨小时完成或平滑处理，不能直接断言服务量超过需求。为避免误读，优先用同一 hourly bins 的 raw demand，并交代延迟。

三天、单 seed 的 pointwise t band 已被原文诚实解释为日间变化，而非跨 seed 变化。图注应保留这一限制；新增多 seed 实验后，最好将主结论不确定性按训练重复报告。

## 11. 必须有的验证与最小补跑清单

不需要重跑所有可想到的组合。建议按以下四个实验包组织，先做正确性，再做性能。

| 实验包 | 目的 | 最低输出 | 优先级 |
|---|---|---|---|
| A. 固定图正确性 | 验证作者实现与 Theorem 1/2 一致 | 逐实例 full/reduced 整数 objective、unfold 后容量与合法动作检查、infeasibility 一致 | 最高 |
| B. 真实 pipeline 时间 | 判断 SSG 是否真正有运行价值 | NYC snapshots + 压力图；端到端分解、尾延迟、图缩减率、峰值内存 | 高 |
| C. 多次训练与关键消融 | 验证 reward improvement 与设计机制 | 主要方法的独立训练重复；2×2 缺失对照；配对结果区间 | 高 |
| D. Charging 与边界状态 | 检查理论和仿真是否同一模型 | 两种模式的排队与总延期、空阶段、must-charge、repeated rejection、terminal transition | 高 |

实验包 A 应包含：全 fallback、混合 forced rows、无 fallback 但可行、无 fallback 且不可行、负分数、exact ties、零容量、多轮 cascades，以及 **超过四轮**触发 pointer 路径的例子。也要检查 GAT 是否只在 reduction 前计算一次，scores 更新后是否重新 reduction。[S, §2.3、§4]

实验包 C 不应通过测试集选择“最好 seed”；训练、调参与 checkpoint 规则要预先固定。Myopic 方法无需伪造训练重复，但仍要在相同评估随机数下配对比较。

请求与资源账本应至少验证下列不变量：同一个已接受 HEV commitment 不被 AEV 再占用；单车单行动；station commitments 不重复入账；same-epoch repair 与对应拒单事件一致；收益只记一次；输出 objective 包含 witness offset；terminal 不丢弃当期 rewards。

## 12. 逐章节修改与篇幅安排

### 12.1 推荐的正文结构

| 部分 | 建议正文篇幅 | 内容取舍 |
|---|---:|---|
| Abstract + Introduction | 1.0 页 | 执行不确定性、观测后修复、核心贡献；不列网络模块清单 |
| Related Work | 0.6 页 | 三类近邻及 SSG 规则关系，不拉长交通背景 |
| Decision model | 1.3 页 | 状态/权限、奖励、两阶段可行集、两种 charging 的精确边界 |
| Recourse value and learning | 1.6 页 | 两阶段递推、目标语义、joint critic、replay、空阶段处理 |
| Exact action optimization | 1.2 页 | Flow 与 SSG 主结果、forced rows、简短证明思路 |
| Experiments | 2.0 页 | 多 seed 主结果、关键消融、端到端 scalability |
| Conclusion / limitations | 0.3 页 | 经验证的结论及真实剩余限制 |
| **合计** | **8.0 页** | 参考文献另外安排 |

这是编辑预算，而不是要求每段严格卡页。当前没有必要把补充材料所有证明塞回正文；但 reward 定义、主算法的关键语义和核心实验不能全部藏在附录。

### 12.2 优先移入附录或压缩的内容

SoC 的逐项代入式及其简单边检查、tower-property 的详细条件期望展开、四轮 vectorized fallback 的工程细节，可以留附录。Table 2 可把四个 solver 重复列 n/m 的长表改成横向比较，节约空间并增加误差/规模信息。Figure 1 重复的 reward bar 可以删除。

正文应补而不是删：奖励/structured-score 小表、第一阶段可见的 AEV context、与行为 follower 一致的 target 说明、端到端计时边界和多 seed 主要差异。

### 12.3 可以直接用于修订的英文表述

**Contribution scope：**

> We study contingent fleet-action selection under execution uncertainty: HEV offer outcomes are observed before AEV actions are selected. The learning component assigns downstream AEV value to the initial offer decision, while the optimization component returns an exact solution of each fixed-score, resource-constrained assignment instance.

这里的 downstream value 表述需与第 6 节选定的 replay/target 语义一致后才能定稿。

**Exactness scope：**

> Exactness is with respect to the supplied candidate graph, fixed rounded scores, and encoded independent capacities. Physical reservation feasibility additionally requires a reservation-compatible admission construction; it does not follow from reactive per-candidate availability checks alone.

**Recourse credit interpretation：**

> The first-stage target includes the entire realized AEV-side reward, covering both rejected-request repair and service of never-offered requests. Consequently, comparisons with an uncoupled leader measure the effect of full downstream credit, not solely the value of repairing rejected requests.

**Terminal replay：**

> Terminal transitions retain their realized rewards and use zero continuation. Nonterminal transitions are included only when the actual next-state link is available.

这些句子是修改建议，不是对当前实现的认证。未完成对应实现和实验前，不应加入更强的“robust improvement”“queue-free deployment”或“end-to-end acceleration”表述。

## 13. 审稿人最可能要求作者回答的关键问题

**Q1：** Table 1 的 reactive 模式具体采用什么 K？存在新候选排队时，Assumption 2 是否被放弃？实际 charging start/completion time 如何进入下一状态？

**Q2：** 式 (26) 的旧 R2 与旧下一状态由哪个 follower 产生？为什么它们可以代表当前 follower 的 recourse value，或者作者实际上只主张怎样的近似？

**Q3：** 第一阶段网络通过什么信息感知 AEV 的可用位置、SoC 和可修复能力？某阶段没有 available vehicles 时，critic 与转移如何处理？

**Q4：** operating reward、g 和 Table 1 的美元量是什么关系？为何 HEV/AEV 的收益与服务量结构显著不同？

**Q5：** 相对使用同样系统收益的强对照，Recourse 的收益是否仍然存在？跨独立训练重复的不确定性有多大？

**Q6：** SSG 相对于已存在的 capacity-slack/abundant-vertex 与 domination 思想，究竟新增了什么？在真实 NYC 图上，包含 preprocessing、建图和评分后是否仍然有收益？

这六个问题比继续增加一个一般性定理或再添加一种神经网络 baseline 更可能改变接收判断。

## 14. 最终执行顺序与验收

| 顺序 | 操作 | 完成标志 |
|---|---|---|
| 1 | 从实现提取模型配置、reward、g、charging 模式和 request lifecycle | 正文/附录不再依赖猜测才能复现 |
| 2 | 统一 reactive/conservative 的模型及保证；处理空阶段和 terminal | 两份文档与日志中的状态转移一致 |
| 3 | 明确 follower-policy/replay 语义；调整目标或采样流程 | 能明确写出 Q1 更新所对应的对象 |
| 4 | 跑作者代码的 fixed-graph correctness 和边界测试 | 逐实例目标、可行性、unfold 检查通过 |
| 5 | 跑真实图的端到端 profile，再确定是否主推 SSG acceleration | 结论由实际 full-pipeline 时间决定 |
| 6 | 跑多次训练与完整系统收益对照 | 主要差异有配对统计，非只展示最佳 seed |
| 7 | 重写引言、Related Work、结论；精简重复图表 | 贡献、假设、实验一一对应 |

**底线判断：这篇稿子已有可以继续发展的 ICAPS 问题结构，SSG 的数学骨架也不是当前最薄弱处。最短的有效修改路线，是把“谁在何时观察什么、谁按哪个策略行动、优化器到底保证哪一个模型、实验证实了哪一个效果”四件事统一起来。当前不建议仅做文字润色后直接投稿。**

## 外部核对来源

以下来源只用于会议要求和最相关的方法定位；论文内部事实均以 M/S 的具体位置为依据。访问日期均为 2026-09-22。链接以纯文本代码形式保留，便于离线核对。

**[E1] ICAPS 2027, Call for Papers.** 会议范围、篇幅、双盲、可复现与随机实验要求。  
`https://icaps27.icaps-conference.org/calls/cfp/`

**[E2a] Großmann, E., Joos, F., Reinstädtler, H., and Schulz, C. (2026). Engineering Hypergraph b-Matching Algorithms. JGAA, 30(1), 1–24.** 已核对期刊元数据及摘要。  
`https://jgaa.info/index.php/jgaa/article/view/3166`  
DOI: `10.7155/jgaa.v30i1.3166`

**[E2b] 同题公开预印本，arXiv:2408.06924v1，§3.2 Exact Reduction Rules.** 本次规则级比较依据可读的预印本正文，特别是 Abundant Vertices 与 Weighted Domination；期刊 PDF 下载未成功，未声称逐条核对期刊版本的全部修改。  
`https://arxiv.org/html/2408.06924v1`

**[E3] Shah, S., Lowalekar, M., and Varakantham, P. (2020). Neural Approximate Dynamic Programming for On-Demand Ride-Pooling. AAAI, 34(1), 507–515.** 价值学习与组合分配的近邻。  
`https://ojs.aaai.org/index.php/AAAI/article/view/5388`  
DOI: `10.1609/aaai.v34i01.5388`

**[E4] Al-Kanj, L., Nascimento, J., and Powell, W. B. (2020). Approximate dynamic programming for planning a ride-hailing system using autonomous fleets of electric vehicles. EJOR, 284(3), 1088–1106.** 电动车队服务、充电、重定位和停车的 ADP 近邻。  
`https://www.sciencedirect.com/science/article/abs/pii/S0377221720300540`  
DOI: `10.1016/j.ejor.2020.01.033`

**[E5] You, F., and Vossen, T. (2024). An Approximate Dynamic Programming Approach to Dynamic Stochastic Matching. IJOC, 36(4), 1006–1022.** 含可能匹配失败的动态随机匹配近邻。  
`https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2021.0203`  
DOI: `10.1287/ijoc.2021.0203`
