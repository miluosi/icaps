#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python run_trainer.py \
  --adp 0 \
  --episodes 1 \
  --num-vehicles 8 \
  --num-ev 4 \
  --transportation-mode integrated \
  --assignment-heuristic \
  --no-mcmf \
  --no-intense-requests \
  --simulation-period 2 \
  --episode-days 1 \
  --grid-size 3 \
  --num-stations 1 \
  --station-capacity 1 \
  --station-queue-capacity 2

