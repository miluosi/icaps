#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python run_nyctrainer.py \
  --adp 0 \
  --episodes 1 \
  --num-vehicles 8 \
  --num-ev 4 \
  --transportation-mode integrated \
  --assignment-heuristic \
  --no-mcmf \
  --distribution-mode none \
  --parquet-path nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet \
  --station-csv nyedata/nyc_all_charging_stations.csv \
  --start-year-month 2025-12 \
  --start-date 2025-12-18 \
  --end-date 2025-12-18 \
  --start-hour 8 \
  --stop-hour 8.05 \
  --epoch-length 30

