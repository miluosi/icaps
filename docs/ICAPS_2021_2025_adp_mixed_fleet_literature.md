# ICAPS 2021–2025 与 ADP Mixed-Fleet Assignment 强相关工作

> 调研日期：2026-08-19  
> 检索范围：ICAPS 正式 proceedings Vol. 31–35（2021–2025）的 archival papers；不含 workshop、demo、doctoral consortium 和 previously-published track。  
> 配套 BibTeX：`references/icaps_2021_2025_adp_mixed_fleet.bib`

## 1. 本项目问题画像与筛选口径

根据当前代码与已有研究叙事，本项目研究的是 NYC 动态需求下由 human-driven EV 与 centrally controlled AEV 构成的混合电动车队。系统在重复决策周期内联合处理：

- 在线订单—车辆匹配；
- 空车重定位；
- 电量可行性、充电站选择、容量与排队；
- 人驾 EV 的接单/拒单和充电选择；
- Integrated、EV-first 与 AEV-first 等信息和决策时序；
- ADP/神经价值修正与 ILP、MCMF 或 auction 的全局可行投影。

纳入文献至少满足下列五项中的两项，并且能对论文的模型、算法或实验设计产生直接作用：

1. 动态、可复用车队/agent 的在线任务分配；
2. 车辆异质性、驾驶员偏好、人类执行不确定性或多方公平；
3. MDP、ADP、价值函数、RL 或 receding-horizon 非近视决策；
4. ILP、network flow、matching、auction 等联合可行优化；
5. EV 路径、电量和充电联合规划。

“A 类”是正文 related work 应优先正面比较的工作；“B 类”是方法或实验设计锚点，不应被表述为同一应用问题的直接竞争者。

## 2. 最相关论文

### 2.1 A 类：正文应优先比较

| 年份 | 论文与 cite key | 核心内容 | 与本项目的连接 | 本项目仍可主张的差异 |
|---|---|---|---|---|
| 2023 | [Using Simple Incentives to Improve Two-Sided Fairness in Ridesharing Systems](https://doi.org/10.1609/icaps.v33i1.27199), `KumarEtAl2023TwoSidedFairnessRidesharing` | 批量 ridesharing dispatch；用车辆—乘客匹配的未来价值作为 ILP 边权；在不重训价值函数的情况下加入 passenger/driver fairness incentives。 | **算法骨架最接近**：learned/estimated future edge values + centralized ILP matching；也直接涉及司机与乘客两侧结果。 | 该文不处理 EV/AEV 混合控制、SoC/充电容量、随机拒单后的同周期 recourse，也不研究 MCMF/auction 的大规模 fixed-score projection。 |
| 2022 | [Joint Pricing and Matching for City-Scale Ride-Pooling](https://doi.org/10.1609/icaps.v32i1.19836), `ShahEtAl2022JointPricingMatching` | 把 pricing 作为不同 matching 决策之上的 meta-level optimization；发展 auction/posting-price 机制，并在 city-scale 真实数据上测试。 | 说明 ICAPS 接受真实城市尺度的需求—供给匹配、auction/MILP 与平台—用户交互联合决策。 | 重点是 passenger pricing/ride-pooling；没有电动车充电、mixed autonomy、长期 ADP edge correction 或 execution recourse。 |
| 2024 | [SKATE: Successive Rank-based Task Assignment for Proactive Online Planning](https://doi.org/10.1609/icaps.v34i1.31499), `NedelmannEtAl2024SKATE` | 在线 multi-agent multi-task assignment；用 receding horizon 预测近未来可用 agents；比较 ILP、GA、SKATE 在不同负载下的质量与时限。 | 与订单在线到达、车辆可用性预测、大规模实时指派和 ILP scalability 直接对应。 | SKATE 不学习长期价值，也不建模订单收益、电量、充电容量、人类拒单和 mixed-fleet control authority。 |
| 2021 | [DeepFreight](https://doi.org/10.1609/icaps.v31i1.15998), `ChenEtAl2021DeepFreight` | 用 QMIX 学习多步 truck dispatch，再做 package matching，并和 MILP 进一步结合；强调 fleet scale、delivery success、time 与 fuel。 | 与“多车协同策略 + request matching + exact optimizer”的混合 learning/planning 结构高度相似。 | 是 freight transfer，不是在线 ride-hailing；没有 EV/AEV、人类行为、充电共享资源与 outcome-conditioned repair。 |
| 2022 | [Reinforcement Learning Approach to Solve Dynamic Bi-objective Police Patrol Dispatching and Rescheduling Problem](https://doi.org/10.1609/icaps.v32i1.19831), `JoeEtAl2022DynamicPatrolDispatch` | 将动态派警建模为 route-based MDP；用 TD learning + experience replay 近似价值函数，并以 ejection-chain heuristic 联合学习 dispatch/rescheduling。 | 是 ADP 叙事的强方法先例：动态事件、近似未来价值、即时重调度和相互冲突的服务目标。 | 不含共享充电资源、两类控制权、request acceptance/rejection 和 exact capacitated matching。 |
| 2022 | [Talking Trucks](https://doi.org/10.1609/icaps.v32i1.19834), `PingenEtAl2022TalkingTrucks` | 将动态物流订单调度建模为 DCOP；比较 decentralized heuristic、DQN、MILP、人类与启发式基线；包含 driver preferences、truck characteristics 和三次真实 field trials。 | 为异质车辆/驾驶员偏好、动态扰动、RL—MILP 对照以及真实运行评估提供强先例。 | 研究多方自治物流而非同一平台内“部分人控、部分全控”的电动出行车队，也不含 charging/recourse。 |
| 2024 | [Multi-Objective Electric Vehicle Route and Charging Planning with Contraction Hierarchies](https://doi.org/10.1609/icaps.v34i1.31467), `CuchyEtAl2024EVRouteCharging` | 联合优化 EV 路径和充电 session 的时间/成本；multi-objective A* + contraction hierarchies，在含 12,000+ 充电站的真实路网上评估 182,000 个实例。 | 是近五年 ICAPS 中最直接的 EV charging planning 参照；提示电池、充电和多目标不能只作为 simulator 细节。 | 是单车 country-scale route planning，不研究动态订单、多车共享站容量、人类驾驶员或 non-myopic fleet assignment。 |

### 2.2 B 类：方法与实验设计锚点

| 年份 | 论文与 cite key | 可借鉴点 | 使用边界 |
|---|---|---|---|
| 2022 | [Distributed Fleet Management in Noisy Environments via Model-Predictive Control](https://doi.org/10.1609/icaps.v32i1.19843), `BoghEtAl2022DistributedFleetManagement` | Q-learning、两层 MPC、在线分布式重规划和 noisy execution；适合支撑“滚动规划 + 及时修复”。 | AMR factory fleet，不是出行市场。 |
| 2023 | [Imitation Improvement Learning for Large-Scale Capacitated Vehicle Routing Problems](https://doi.org/10.1609/icaps.v33i1.27236), `BuiMai2023ImitationImprovementCVRP` | learned improvement + classical heuristic expert + problem decomposition；真实规模达到 30,000 nodes；适合对照规模泛化和 learned/exact hybrid。 | 静态 CVRP，不能当作动态 mixed-fleet assignment 的直接 competitor。 |
| 2024 | [Progressive State Space Disaggregation for Infinite Horizon Dynamic Programming](https://doi.org/10.1609/icaps.v34i1.31479), `ForghieriEtAl2024StateSpaceDisaggregation` | 根据 value function 逐步细分状态聚合，给出 convergence，并接入 approximate value/Q-value/policy iteration；与 zone-level ADP 表征和状态压缩直接相关。 | 通用 MDP 方法，没有 fleet application。 |
| 2024 | [Neural Combinatorial Optimization on Heterogeneous Graphs](https://doi.org/10.1609/icaps.v34i1.31494), `LuttmannXie2024HeterogeneousGraphNCO` | heterogeneous graph encoder、hierarchical decoder、integrated selection-routing，并与 exact/heuristic solvers 全面比较；可用于定位 GAT 与组合动作空间贡献。 | 应用是 warehouse picker routing；只证明“用了 GNN”远不足以构成本文新颖性。 |
| 2021 | [Integrating Knowledge Compilation with Reinforcement Learning for Routes](https://doi.org/10.1609/icaps.v31i1.16002), `LingEtAl2021KnowledgeCompilationRoutes` | 把图连通性和 domain constraints 编译进 decision diagrams，改善 MARL sample efficiency 和可行探索；可支撑“结构化约束不应全交给网络学习”。 | cooperative MAPF，不含 request market 或 charging。 |
| 2022 | [Solving Simultaneous Target Assignment and Path Planning Efficiently with Time-Independent Execution](https://doi.org/10.1609/icaps.v32i1.19810), `OkumuraDefago2022TargetAssignmentPathPlanning` | 联合 target assignment 与 path planning；强调在线 timing uncertainty、可扩展性和 assignment/execution coupling。 | unlabeled MAPF 的目标、约束和本项目显著不同。 |

## 3. 五年覆盖结论

| Proceedings | 筛选结果 |
|---|---|
| ICAPS 2021, Vol. 31 | 2 篇：DeepFreight 为 A 类；knowledge compilation + RL routes 为 B 类。 |
| ICAPS 2022, Vol. 32 | 5 篇：该年最集中，覆盖 ride-pooling、动态 dispatch/rescheduling、异质物流、fleet MPC 和 assignment/path coupling。 |
| ICAPS 2023, Vol. 33 | 2 篇：ridesharing fairness/ILP future values 为最接近论文；另有 large-scale learned CVRP。 |
| ICAPS 2024, Vol. 34 | 4 篇：在线任务指派、EV route/charging、ADP state abstraction 与 heterogeneous-graph NCO。 |
| ICAPS 2025, Vol. 35 | [正式录用列表](https://icaps25.icaps-conference.org/program/accepted/)和 proceedings 中没有达到上述纳入阈值的 fleet/ride-hailing/EV-charging 工作，因此 BibTeX 中没有为凑年份而加入弱相关论文。 |

结论不是“ICAPS 不关心该方向”，而是近五年的直接工作稀疏且分散在 online assignment、ride-pooling、fleet planning、charging 和 learning-for-planning 几条线上。这给本文留下空间，但也意味着论文必须把交通系统抽象为一个清楚的 planning problem，而不能只呈现工程 simulator 和神经网络结果。

## 4. ICAPS 对文章的正式要求（以 ICAPS 2025 为历史依据）

以下是 [ICAPS 2025 Main Track CFP](https://icaps25.icaps-conference.org/calls/main_track/)、[Conditions for Acceptance](https://icaps25.icaps-conference.org/organisation/conditions_for_acceptance/) 和 [Paper Classification System](https://icaps25.icaps-conference.org/organisation/paper_classification_system/) 的概括。具体投稿年份必须重新核对当年 CFP，不能把 2025 的页数、政策和日期直接视为未来届次规则。

### 4.1 形式硬要求

- 论文类型只能选一个最主要贡献：Theoretical、Algorithmic、Modelling、Position 或 Tools。
- 2025 main track 长文为 8 页正文 + 1 页参考文献，短文为 4 页正文 + 1 页参考文献，使用指定 AAAI 风格；超页或改字体、页边距会 desk reject。
- 双盲：去除作者、机构和可识别致谢，自引按第三人称处理。
- 投稿必须自包含。可交匿名补充材料，但除 tools paper 外，审稿人没有义务阅读补充材料。
- 审稿期间不得同时投其他会议/期刊；同作者的多篇投稿必须有显著且独立的内容。
- 至少选择一个 Planning and Scheduling 类 subject tag；标签明显错配会触发 desk rejection。
- 2025 的 GenAI 政策禁止把 LLM 生成文本作为普通论文正文，但允许对作者自己写的文本做编辑/润色；作者对真实性、抄袭和全部内容负责。未来投稿须重新核对最新政策。
- 涉及伦理或社会影响时应给出 impact statement；录用后通常要求至少一名作者注册并现场报告。

### 4.2 实质审稿标准

官方 review questionnaire 反复检查：

- technical soundness；
- 与所选 topic/tag 的真实契合；
- 在既有工作中的准确定位；
- novelty 与 significance；
- 完整的定义、证明和/或实验数据；
- 令人信服的动机、影响和结论；
- 摘要是否准确、英语和呈现是否清楚、图表与例子是否真正帮助理解。

对本文最关键的一点是：ICAPS 的 Machine Learning in Planning and Scheduling topic 明确要求把 ML 假设连接到严格的 planning/scheduling 数学表述；若采用新表述，需要说明现有方法缺少什么能力。因此“把 GAT/SAC 用到 NYC taxi”本身不够，必须说清 planning model、联合可行集合、执行不确定性、repair/recourse 时序和 learning—optimizer 接口。

## 5. 从近五年录用论文归纳出的写作要求

下列是对上述录用论文的归纳，不是 CFP 的逐字规定：

1. **先有 rigorous problem，再有模型。** 用 MDP/两阶段随机规划/在线任务分配/网络流明确状态、动作、信息时序、容量和目标。
2. **Learning 必须解决 planning bottleneck。** 例如估计未来 edge value、预测近未来资源、学习改进算子或加速结构化搜索，而不是只替换一个 predictor。
3. **说明可行性由谁保证。** 网络输出 score 还是完整 joint plan？订单互斥、每车一动作、充电容量、SoC 和 repair 约束由 ILP/MCMF 保证，还是只靠惩罚项？
4. **给出结构性新颖性。** 对本文而言，最有力的候选不是“混合车队 + RL”，而是 partially controllable human actions、stochastic compliance outcome、same-epoch exact repair 和 residual ADP 的耦合。
5. **质量、时间和规模同时评估。** 录用工作常同时报告 objective/service、runtime、规模扩展和高负载退化；只报平均 reward 不够。
6. **与 exact、heuristic、learning baselines 同台。** 至少要有 myopic exact、ADP-exact、无 recourse、不同信息时序和可扩展近似方法；若宣称 MCMF 与 ILP 等价，应在固定 edge scores 下验证解质量/可行性。
7. **真实数据之外还要可重复。** 多日期、多随机种子、需求强度与 fleet composition 扫描、acceptance-model misspecification、charging capacity 和 ablation 应形成系统实验矩阵。
8. **不要把 domain detail 冒充 generality。** NYC TLC、真实充电站和驾驶员模型证明应用价值；通用贡献则应抽象成 mixed-agent planning under stochastic action compliance and capacitated recourse。

## 6. 对本项目最合适的 ICAPS 定位

### 推荐分类

- **Type**：Algorithmic；若新颖性主要在两阶段 partially controllable planning formulation，则可考虑 Modelling，但不要同时宣称多个主类型。
- **Topics**：Machine Learning in Planning and Scheduling + Human-aware Planning and Scheduling；若当届保留 Applications special topic，可作为应用侧定位。
- **Subject tags**：PS: Applications、PS: Learning for planning and scheduling、PS: Optimisation of spatio-temporal systems、PS: Planning with MDP models、PS: Planning under uncertainty、PS: Re-planning and plan repair、PS: Routing、PS: Real-time planning、PS: Mathematical programming、UAI: Sequential decision making、ML: Reinforcement learning。

### 推荐的一句话问题定义

> We study online capacitated planning for a mixed population of fully controllable agents and human-operated agents with stochastic action compliance, where a learned residual value model scores first-stage actions and an exact flow planner constructs both the initial assignment and the outcome-conditioned repair plan.

### 正文贡献建议收紧为三点

1. 一个包含 human offer、stochastic compliance 和 controlled-fleet repair 的两阶段 mixed-agent planning model；
2. 一个把 structured myopic score、learned residual continuation value 与 exact capacitated flow projection 结合的 planner；
3. 在 NYC 多日、多负载和多 fleet-mix 条件下，对 recourse value、服务率/公平、充电拥堵、求解时间和泛化的系统评估。

## 7. BibTeX 使用

全文引用：

```latex
\bibliography{references/icaps_2021_2025_adp_mixed_fleet}
```

正文最小核心引用集可从以下 7 篇开始：

```latex
\cite{KumarEtAl2023TwoSidedFairnessRidesharing,
      ShahEtAl2022JointPricingMatching,
      NedelmannEtAl2024SKATE,
      ChenEtAl2021DeepFreight,
      JoeEtAl2022DynamicPatrolDispatch,
      PingenEtAl2022TalkingTrucks,
      CuchyEtAl2024EVRouteCharging}
```

其余 6 篇应按具体方法论论证使用，不建议为了显示覆盖面而全部堆入同一段 related work。
