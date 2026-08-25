#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Exercise the real NYCTrainer wiring, two-stage graph serialization, joint
# scheduling and exact target projection on the checked-in demand sample.
python run_nyctrainer.py \
  --adp 1 \
  --episodes 1 \
  --num-vehicles 8 \
  --num-ev 4 \
  --transportation-mode evfirst \
  --recourse-variant r4 \
  --state-variant joint_state_separate_critics \
  --learner-variant optimization_anchored_residual \
  --assignment-heuristic \
  --no-mcmf \
  --distribution-mode optimization_anchored_residual \
  --batch-size 2 \
  --training-frequency 1 \
  --start-training-episode 0 \
  --checkpoint-replay none \
  --parquet-path nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet \
  --station-csv nyedata/nyc_all_charging_stations.csv \
  --start-year-month 2025-12 \
  --start-date 2025-12-18 \
  --end-date 2025-12-18 \
  --start-hour 8 \
  --stop-hour 8.05 \
  --epoch-length 30 \
  --checkpoint-suffix ci-joint-smoke
