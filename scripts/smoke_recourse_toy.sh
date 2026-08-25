#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Exercise the real ADPTrainer EV-first/R4 production path long enough to
# create linked successors and update both raw joint critics.
python run_trainer.py \
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
  --distribution-mode optimization_anchored_residual \
  --checkpoint-replay none
