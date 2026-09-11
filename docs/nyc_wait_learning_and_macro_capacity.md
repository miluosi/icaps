# NYC wait shaping and macro capacity verification

NYC keeps every decision vehicle's real wait edge, in both current and
conservative charging modes. `charge_wait_bool` remains accepted for old
command lines but no longer removes wait. Charging masks and physical slot
counts are unchanged. This prevents infeasibility caused solely by removing
low-SoC wait while charging slots are shared; it does not fix unrelated graph
or input errors.

## Learning reward

For online AEVs executing stationary wait, the default extra learning cost is

```
P(q, d) = 8 * d * EPOCH_LENGTH / 3600 * (max(0, 0.30-q) / 0.10)**2
```

`q` is fractional SoC; `d` is duration in simulation epochs, not seconds.
For 30-second epochs, the per-epoch costs at SoC 30%, 20%, 10%, and 0% are
0, 0.0666667, 0.2666667, and 0.6 respectively. The environment fields
`learning_soc_wait_threshold`, `learning_soc_wait_reference`,
`learning_soc_wait_penalty_per_hour`, and `learning_soc_wait_exponent`
define this policy. The threshold/reference initialize to
`min_battery_level + 0.1` and `min_battery_level` (defaults 0.30/0.20).

This replaces the old flat extra **ordinary wait learning penalty**. It does
not alter the physical idle cost (4/hour), physical charger-queue waiting
cost, relocation shaping, or HEV rewards. Charger-queue wait is a charging
action and does not receive this extra ordinary-wait shaping.

The legacy per-action experience path subtracts the cost over the action's
duration. The joint collector freezes the epoch cost in
`RecourseTransition.aev_soc_wait_learning_penalty`. Its original
`reward_aev`, `reward_system`, and reconciled `reward_ledger` remain economic
rewards. AEV targets use `learning_reward_aev`; macro EV-leader targets use
`learning_reward_system`. Integrated system targets also use the latter.
Old/unshaped replay defaults to zero extra cost.

This is a soft learning penalty, not a lexicographic objective or a big-M
change to solver scores. It cannot guarantee immediate charging, especially
before training. The r2 follower remains frozen by design. Existing learned
checkpoints need further training to reflect the changed learning reward.

## Server macro traceback

The supplied 2026-09-11 log stops in `RecourseTargetBuilder.verify_feasible`:

```
joint action exceeds ('station', 114986) capacity: count=8, capacity=4
```

This is graph validation before physical arrival, not an exception from
inserting a vehicle into the station queue. Counting all assignments against
one static capacity conflates different arrival epochs. Current local code
records `station_arrival:<epoch>` capacity when a matching arrival expansion
is available. It also freezes `allow_charging_queue=True` for the default NYC
model, so station admission quotas do not become a physical queue prohibition.
Exactly-one, valid-edge, and request-capacity checks remain active.

The supplied traceback is consistent with an older/incompletely synchronized
server source tree. Server files have **not** been inspected or changed in
this local check. To carry the existing capacity fix across, synchronize all
three files together:

- `src/recourse/state_snapshot.py`
- `src/recourse/types.py`
- `src/recourse/target_builder.py`

For the new wait/learning policy, also synchronize:

- `src/NYCEnvironment.py`
- `src/NYCtrainer.py`
- `src/ValueFunction_st_masac_gat.py`
- `src/recourse/coordinator.py`
- `run_nyctrainer.py`

Restart the affected Python training process after synchronization; existing
processes keep their imported modules. These changes require no new solver
or NumPy installation. The new trainer startup prints `NYC AEV wait: always
feasible; learning-only SoC penalty ...`.

## Local verification

Run from the repository root using its Python environment:

```sh
python -m pytest -q tests/test_nyc_soc_wait_learning.py tests/test_nyc_arrival_snapshot.py tests/test_expected_charging_feasibility.py tests/test_conservative_charging_benchmark.py tests/test_charging_metrics_sensitivity.py tests/test_recourse_must_fix.py tests/test_recourse_reaudit.py tests/test_repair_only_learning.py
```

The targeted suite passes (85 passed, 2 pre-existing expected failures).
It covers actual NYC r3/macro execution and replay collection, economic reward
invariance, duration units, low-battery slot competition, and eight macro
charging assignments across two arrival epochs at one four-slot station.
The two expected failures concern the known physical epoch-order boundary
(movement admission happens before charging release), not this wait change.
This is not a completed 3000-vehicle server training run.
