# NYC 3000-vehicle speed audit (2026-09-11)

The supplied `nyc_r3.log` and `nyc_r1_r2.log` both use 3000 vehicles
(1500 HEVs + 1500 AEVs), 425 public stations + 3 AEV centers, OR-Tools,
30-second epochs, and CPU neural inference after CUDA initialization error
804. The second file has only reached r1; it contains no r2 timing samples.

## What the server logs actually measure

Means over logged steps 1 and 2 (excluding the expensive initial HEV charging
choice; this is **not** a whole-episode average):

| Recorded component | r1 seconds | r3 seconds |
| --- | ---: | ---: |
| Feasibility matrix preparation | 5.899 | 5.662 |
| Q-value phase, including another matrix build | 7.368 | 7.242 |
| Assignment phase, including graph serialization/verification | 1.191 | 1.194 |
| Other simulation preparation/action processing | 5.096 | 5.423 |
| Environment step, including experience collection | 1.425 | 1.233 |
| Rollout total | 20.980 | 20.755 |

Combined proportions: matrix 27.7%, Q phase 35.0%, assignment phase 5.7%,
other simulation 25.2%, environment step 6.4% (rounding and negligible charging
phase account for the remainder). `solve` is not pure OR-Tools runtime;
`qvalue` is not pure neural forward time.

Step zero additionally spends about 12.84 seconds on the HEV charging choice.
The logs explicitly show minimum battery 0.800 at that step: the new low-SoC
wait penalty is zero then. These files provide no controlled evidence that
making wait universally feasible caused the slowdown.

The former `step_wall` timer ended **before** the trainer's `train_step` calls
and checkpoint writes. `qlearn_aev/qlearn_ev` inside `env.step` mostly measure
experience construction/storage, not critic forward/backward/optimization.
Thus the logged 21 seconds does not capture the entire training-loop cost.

## Direct optimizations implemented

1. `_charging_station_ids_for_vehicle` and its record variant now evaluate
   the fallback station sort only if the public list is absent. Previously
   `getattr(..., sorted(stations))` sorted eagerly even when the attribute
   existed. Lists are still copied, ordering is retained, and there is no
   stale cross-epoch station cache.
2. A decision matrix containing only HEVs skips charge/relocation blocks that
   the original code computed and then unconditionally cleared. HEV charging
   remains in the preceding `_ev_charging_phase`. Request and wait columns
   are retained; AEV and mixed-fleet feasibility logic is unchanged.
3. Q inference reuses the matrix and column layout built immediately before
   it in `_solve_rebalancing`. This is call-local reuse, not a cache across
   EV rejection decisions, reservations, or simulation epochs. Standalone Q
   callers still build their own matrix.
4. Selected joint critic predictions build one differentiable graph context
   per provider/state instead of clearing it for each selected edge. Encoder
   and mixer have no dropout. A snapshot identity fast path also avoids
   repeatedly hashing the same frozen fleet state. The cache is cleared at
   operation boundaries; sum gradients are preserved within floating-point
   tolerance. No detach/no-grad shortcut is applied to online training.
5. `NYCTrainer` now saves `train_aev_time_sec`, `train_ev_time_sec`,
   `learning_phase_time_sec`, and `full_step_time_sec`. `TrainTiming` logs
   actual network updates and the complete step through the learning phase.
   Learning-phase time includes readiness checks, updates, and checkpoint/
   loss persistence. Existing rollout-only timing fields retain their meaning.

These changes do not remove real assignment edges, change Q formulas,
capacity quotas, the SSG reduction rule, OR-Tools precision, or its optimality
requirement. Shared graph gradients are numerically equivalent, not promised
bitwise identical over an entire stochastic training run. This preserves the
existing assignment model's optimum; it does not strengthen any pre-existing
claim about interval scheduling or a queue-free policy.

## Verification and saved measurements

`tests/test_nyc_exact_speedups.py` checks masks and Q values, both charging
policies, absence of duplicate matrix builds, station ordering, and all
encoder/mixer/twin-critic gradients against separate per-edge graph encoding.
It also checks that a second backward uses a fresh graph.

The focused charging/recourse suite passes: 90 passed, 2 pre-existing expected
failures for the physical admission-before-release epoch boundary.

Local CPU profiling and paired comparison artifacts are saved under
`results/nyc_speed_audit_20260911/`. The profile uses 3000 vehicles, 3 AEV
centers, an empty initial demand state, seed 901, and the audit runner's P0
auxiliary setting. It is not an exact replay of the server's P3 December 15
training. cProfile timings include instrumentation overhead; use the separate
unprofiled paired run for before/after time comparisons.

The first-step profile observed roughly 1.53 million visible-station lookups,
1.53 million sorting calls, four full feasibility builds (two per fleet), and
about 685,000 expected charging-window queries. Profiling also exposed
per-selected-vehicle graph reconstruction during learning. Server-specific
speedup and whole-day training time still require rerunning the updated code
on the server.

## Paired local result (single seed, two collection steps)

The unprofiled paired run uses frozen pre-edit Python sources and exactly the
same seed, starting state, model weights and thread settings.

| Operation | Before (s) | After (s) | Speedup |
| --- | ---: | ---: | ---: |
| Initial action generation | 20.70 | 3.89 | 5.32x |
| Next action generation | 7.19 | 1.19 | 6.05x |
| One AEV network update | 20.80 | 1.14 | 18.17x |
| One EV network update | 20.04 | 0.96 | 20.96x |

All four stage assignments, feasible edge keys, structured scores, Q values
and selected objective values matched exactly. Both reported training losses
also matched exactly. The gradient unit test additionally compares every
encoder/mixer/critic parameter gradient within float32 tolerance.

This is a short empty-demand case, not ten seeds or a full day of server
training. Startup/import effects can affect the first-step measurement.
The next-step result and gradient checks provide additional evidence, but
these ratios should not be quoted as universal or paper-level speedups.

To deploy these changes together, synchronize `src/NYCEnvironment.py`,
`src/ValueFunction_st_masac_gat.py`, and `src/NYCtrainer.py`, then restart
the affected training processes. No package upgrades or CLI changes are
needed. This task changed local source only.
