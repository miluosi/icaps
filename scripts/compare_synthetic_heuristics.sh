#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES="${EPISODES:-5}"
VEHICLES="${VEHICLES:-20}"
EVS="${EVS:-10}"
COMMON=(
  --episodes "$EPISODES"
  --num-vehicles "$VEHICLES"
  --num-ev "$EVS"
  --transportation-mode integrated
  --no-mcmf
  --distribution-mode st_masac_gat_queue_demand_gurobi
  --simulation-period 12
  --episode-days 1
  --grid-size 4
  --num-stations 2
  --station-capacity 1
  --station-queue-capacity 3
)

# Exact assignment during training: checkpoint consumed by legacy ADP-HEU.
python run_trainer.py --adp 1 --assignment-gurobi "${COMMON[@]}"

# Heuristic assignment during training: checkpoint consumed by ADP-HEU-HEU.
python run_trainer.py --adp 1 --assignment-heuristic "${COMMON[@]}"

python test_model.py \
  --episodes 1 \
  --num-vehicles "$VEHICLES" \
  --num-ev "$EVS" \
  --transportation-modes integrated \
  --demand-patterns intense \
  --strategies ADP-HEU ADP-HEU-HEU HEU \
  --distribution-mode st_masac_gat_queue_demand_gurobi \
  --simulation-period 12 \
  --episode-days 1 \
  --grid-size 4 \
  --num-stations 2 \
  --station-capacity 1 \
  --station-queue-capacity 3

