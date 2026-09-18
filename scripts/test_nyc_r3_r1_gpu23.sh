#!/usr/bin/env bash
# Run from the server's icaps checkout after activating the training environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
mkdir -p results
run_dir="$(mktemp -d results/test_gpu23_XXXXXXXX)"
common=(
  --strategies ADP-MCMF --mcmf-backend ortools
  --aev-charging-center-count 3
  --load-model-start-date 2025-12-08 --load-model-end-date 2025-12-10
  --start-date 2025-12-15 --end-date 2025-12-17
  --checkpoint-selection best-reward
)
# Both model checks must pass before either long evaluation starts.
CUDA_VISIBLE_DEVICES=2 "$python_bin" -u test_nyc_model.py \
  "${common[@]}" --methods r3 r1 --checkpoints-only
for entry in 2:r3 3:r1; do
  gpu="${entry%%:*}"
  method="${entry#*:}"
  mkdir -p "$run_dir/$method"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$python_bin" -u test_nyc_model.py \
    "${common[@]}" --methods "$method" --output-dir "$run_dir/$method" \
    > "$run_dir/$method/test.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$run_dir/$method/test.pid"
  printf 'Started %s on GPU %s: PID=%s, log=%s/%s/test.log\n' \
    "$method" "$gpu" "$pid" "$run_dir" "$method"
done
