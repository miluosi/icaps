# 精简清单

本目录是从 `adp_trainer` 当前主链路提炼出的独立 ICAPS 复现工程，不是源仓库镜像。

保留内容：

- `run_trainer.py`、`test_model.py`：合成 mixed-fleet 训练与评估；
- `run_nyctrainer.py`、`test_nyc_model.py`：NYC 训练与评估；
- `src/ADPtrainer.py`、`src/NYCtrainer.py`、两个环境、动作/请求/充电站与求解器核心；
- 当前主模型 `st_masac_gat`、post-demand 与 direct demand head；
- ADP-HEU、ADP-HEU-HEU、HEU 语义测试以及 MCMF/场景基础测试；
- 小型真实 NYC 数据、近五年 ICAPS 调研和 BibTeX。

明确不包含：

- `results/`、`checkpoints/`、`logs/` 和任何训练产物；
- notebook、图片预览、缓存、临时文件；
- 旧 A2C/PPO/DDPG、旧版 `ValueFunction_pytorch2/3/o/s`；
- cuOpt/CUDA 原型、旧数据处理脚本和一次性诊断/绘图脚本；
- 与 ICAPS 主实验无关的消融网络。它们只在 `adp_journal` 中按需保留。

