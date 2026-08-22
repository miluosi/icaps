#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 70 steps cross the trainer's 64-step replay warmup and exercise a real
# value-network update with heuristic assignment used to collect transitions.
python run_trainer.py \
  --adp 1 \
  --episodes 1 \
  --num-vehicles 8 \
  --num-ev 4 \
  --transportation-mode integrated \
  --assignment-heuristic \
  --no-mcmf \
  --no-intense-requests \
  --start-training-episode 0 \
  --batch-size 2 \
  --simulation-period 70 \
  --episode-days 1 \
  --synthetic-demand-scale 0.2 \
  --grid-size 3 \
  --num-stations 1 \
  --station-capacity 1 \
  --station-queue-capacity 2 \
  --distribution-mode st_masac_gat_queue_demand_gurobi
