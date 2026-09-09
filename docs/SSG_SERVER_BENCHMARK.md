# 四种精确算法的服务器规模实验

入口：[benchmark_cplex_mcmf_ssg.py](../benchmark_cplex_mcmf_ssg.py)。绘图：[plot_cplex_mcmf_ssg.ipynb](../plot_cplex_mcmf_ssg.ipynb)。

默认比较 CPLEX、CPLEX+SSG、MCMF(OR-Tools)、MCMF(OR-Tools)+SSG。两种 CPLEX 方法采用相同的连续网络 LP；两种 MCMF 方法采用相同的 OR-Tools 最小费用流后端。SSG 使用原有精确容量证书与支配删除，Q scale 保持 10000。

## 默认场景与规模

每个场景从小到大运行车辆数 `100 500 1000 2000 3000 6000`，订单数为车辆数的五倍，每个规模使用相同的 10 个随机种子 `2101–2110`。共 180 个输入、720 次求解。

| 场景 | AEV 比例 | 站点数 C | relocation 地区数 Z | 作用 |
|---|---:|---:|---:|---|
| `adp_control` | 50% | V/5 | V/10 | 原大规模动作比例的对照 |
| `reloc_rich` | 50% | V/5 | 2V | 增加容量不紧的可行地区动作 |
| `aev_joint` | 90% | V | V | 同时增加 AEV、站点与地区动作 |
| `fixed_candidates`（可选） | 50% | V/5 | 2V | 地区增多但每车仍约五个 relocation 候选 |

预设每站容量统一为 4，保证对照口径一致。请求容量为 1，relocation 容量为总车辆数，等待容量也不紧。这是计时矩阵中的站点动作数，不是 `run_trainer.py` 的默认物理站点数。

前三个场景保留 ADP 生成器的可行率规则：请求 `min(0.25, max(8/R, 0.01))`、充电 `min(0.4, max(4/C, 0.05))`、relocation `min(0.5, max(4/Z, 0.1))`，并保证相应车辆至少有一个候选。只有 AEV 有充电和 relocation 候选。合成 Q 范围仍为请求 [8,35]、充电 [1,12]、relocation [0,8]、等待 0。没有修改收益来强制某种方法胜出。

可选 `fixed_candidates` 只把 relocation 概率设为 `min(1,4/Z)`，另有一个保证候选。它检验优势来自增加可删除的可行边，还是仅仅增加地区列。预设及覆盖后的有效参数全部保存到 `metadata.json` 的 `experiment_settings.case_specs`。

有利场景依据之前的本地敏感性测试确定，种子不按速度筛选。保留小规模和对照场景，重点观察大规模时缩弧比例、后端调用耗时以及端到端净加速。规模增长并不保证加速倍数单调上升。

## 服务器运行

建议使用独立 Python 环境。默认四种方法不需要加载 Torch、Gym、GeoPandas 或训练数据；CPLEX 需要可用的本地安装与许可。

```bash
python -m venv .venv-benchmark
source .venv-benchmark/bin/activate
python -m pip install -r requirements-benchmark.txt

python benchmark_cplex_mcmf_ssg.py \
  --output-dir results/cplex_mcmf_ssg/server_run
```

脚本在导入 NumPy 前设置单线程环境变量，并通过 threadpoolctl 限制已加载的 BLAS 线程池。CPLEX 固定为一个线程；各方法串行运行。不要同时启动多个计时进程。

只运行两种有利结构，仍保留从小到大的规模：

```bash
python benchmark_cplex_mcmf_ssg.py \
  --scenarios reloc_rich aev_joint \
  --vehicle-counts 100 500 1000 2000 3000 6000 \
  --output-dir results/cplex_mcmf_ssg/favorable_run
```

更大的服务器可将 `--vehicle-counts` 扩展至 `100 500 1000 2000 3000 6000 10000`。当前输入仍有稠密 `V × (R+C+Z+1)` 矩阵，CPLEX 另有模型存储；需要按服务器内存选择最大规模。结果保存采用稀疏压缩格式，不能据此推断运行时也没有稠密内存开销。

每档默认十个种子。需要其它数量或固定种子列表时使用 `--repeats 20` 或 `--seeds 2101 2102 2103`；两者互斥。`--seed` 指定连续种子的起点。

自定义精确尺寸可使用旧的显式接口，例如：

```bash
python benchmark_cplex_mcmf_ssg.py \
  --scales 100:500:100:100 500:2500:500:500 1000:5000:1000:1000 \
           3000:15000:3000:3000 6000:30000:6000:6000 \
  --aev-ratio 0.9 --fixed-charge-capacity 4 \
  --output-dir results/cplex_mcmf_ssg/custom_run
```

格式为 `车辆:订单:站点:地区`，也可写 `车辆:订单` 自动推导 C=V/5、Z=V/10。`--scales` 与预设 `--scenarios/--vehicle-counts` 互斥。显式场景未指定 AEV 比例时为 50%，未固定充电容量时使用原 ADP 随机容量规则。`--aev-ratio` 与 `--ev-ratio` 分别表示自主和人工驾驶车辆比例，不能同时设置。

## 保存与续跑

每个完整输入比较完后立即更新 CSV/JSON；方法级事件在每次求解返回后写入。输入压缩、哈希和写盘都在计时范围之外。

| 文件 | 内容 |
|---|---|
| `metadata.json` | 实验状态、计划/完成数量、参数、十个种子、有效场景配置、线程与库版本、源代码 SHA256、续跑会话 |
| `raw_results.csv` / `raw_results.json` | 每种方法每个输入的时间、原图/缩图大小、删边分类、最优目标、实际执行顺序及输入文件哈希 |
| `summary.csv` | 每场景/规模/方法的算术平均、样本标准差、中位数、种子列表与缩弧比例 |
| `paired_speedups.csv` | 同输入、同种子的完整图/缩图时间比及胜出标志 |
| `method_events.jsonl` | 逐次返回的原始事件；可能包含未完成比较或续跑重算的事件，不直接用于论文作图 |
| `inputs/<场景>/<规模>/seed<种子>.npz` | 可无损恢复的矩阵输入：可行边坐标、原始可行 Q 值及 dtype、不可行位置常量、容量、fallback、shape |

`load_case_input(path)` 可还原完整 `AssignmentCase`；默认保存全部输入。可显式用 `--no-save-inputs` 节省存储，但此时只有配置、种子与结果，没有矩阵档案。

中断后使用同一命令附加 `--resume`。它核对实验设置、求解相关源代码、运行环境与已完成输入的哈希，跳过完成的输入，对不完整输入整组重跑。设置或代码不同应使用新输出目录。未完成结果不会进入正式汇总；失败原因保存在 metadata，默认遇到最优目标不一致就停止并保留诊断。

若将来启用 `--allow-objective-mismatch`，不一致的诊断仍会保存，notebook 默认拒绝将其作成有效比较图。

## 本地 notebook 作图

复制结果目录到本地，打开 `plot_cplex_mcmf_ssg.ipynb`。只画图时本地需要 Jupyter、NumPy、pandas、Matplotlib，不需要 CPLEX 或 OR-Tools。

将首个配置单元的 `RESULT_DIR` 指向该目录，然后 Run All。留为 `None` 时选择本项目最近完成的新版实验。默认核对完整性、种子配对、重复行、每种方法的目标与计时加和。

每场景输出两套参考图风格的双面板：分组柱状图 + 对数规模折线图。四种方法分别使用红、深蓝、青绿、灰蓝；全部种子的均值作图，黑色误差棒默认为样本标准差。主图包含 SSG 预处理在内的总时间；另一套图显示后端调用时间。PNG 默认 3180×1080，同时导出 PDF、SVG 和实际作图统计 CSV 到 `figures/`。对数图中跨越零的误差棒仅截断显示下端，并给出提示，原始标准差不变。

notebook 还输出 MCMF/SSG 配对对照表：总时间比、求解时间比、缩弧率、SSG 更快的种子数、逐对加速比范围和几何平均。所有方法使用同一输入；加速比大于 1 才说明缩图在该口径下更快。

## 计时含义

- `graph_build_seconds`：量化与验证、精确 SSG（如启用）、流图构建。
- `solve_seconds`：后端建模、优化、解码、分配与目标校验；并非优化器内核独立计时。
- `end_to_end_seconds`：上述两项相加，作为默认主结论。不包含输入随机生成、神经网络评分、完整训练或仿真。

每个输入先后构建完整图与缩图，每张图供两个后端使用；每种方法的总时间都完整计入对应构图时间。完整图/缩图先后顺序与后端先后顺序交替，以减少顺序偏差。原生 MCMF 更换后端的加速与 SSG 本身的加速分别通过四种方法对照，不能混为同一贡献。

## 本地验证记录

本次用默认三个场景、车辆数 `100 500 1000`、默认十个种子运行了 90 个输入 / 360 次求解，四种方法的最优目标全部一致，全部 90 份输入已归档。相关 267 项测试通过，包含场景分组、矩阵无损保存、失败后续跑与精确后端检查。

notebook 的全部单元已实际执行，通过本地数据导出六组 PNG/PDF/SVG；PNG 均为 3180×1080，并已目视检查。验证数据位于 `results/cplex_mcmf_ssg/scenario_suite_local_validation_10seeds/`，其中 `plot_preview.ipynb` 保留执行输出。服务器默认的完整六档、180 个输入 / 720 次求解留待正式运行。
