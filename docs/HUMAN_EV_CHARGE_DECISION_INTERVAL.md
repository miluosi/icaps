# Human EV charge-decision interval calibration

## Selected setting

The NYC Human EV charge-decision interval is **120 minutes** by default in
`icaps`, `adp_trainer`, and `adp_journal`. This setting was calibrated in ICAPS
without learning. It is an interval between actual stochastic charging
decisions, not a mandatory charging schedule or a hard daily charging cap.

CLI option for NYC training and evaluation:

```text
--human-ev-charge-decision-interval-minutes 120
```

Python API argument:

```python
human_ev_charge_decision_interval_minutes=120.0
```

The interval is converted to epochs using the ceiling of minutes times 60
divided by the epoch length in seconds. Thus 120 minutes equals 60 epochs at
120 seconds per epoch, or 240 epochs at 30 seconds per epoch.

## Decision-clock semantics

- Only eligible idle Human EVs make the stochastic charging decision.
- After an actual decision, both a charge outcome and a no-charge outcome
  set the next eligible decision time.
- Skipping an ineligible decision does **not** extend the deadline. This
  fixes the previous rolling-reset behavior of the hard-coded five-epoch
  cooldown.
- A driver busy at the deadline decides when it next becomes eligible.
- The existing SOC <= 0.20 safety bypass remains active, as does the existing
  must-charge threshold. The interval does not prevent safety charging.
- AEV assignment, charging actions, ADP/HEU logic, battery consumption, and
  the charging probability and station-choice formulas are unchanged.

The complete `compute_ev_charge_probability` function has the same SHA-256
before and after the change in all three projects:

```text
cec44e685128296a80528d2e561856d3b7312f1a2fd27ff59e410066d8c90ba9
```

## Experiment configuration

| Item | Setting |
|---|---|
| Demand | NYC yellow taxi data, 2025-12-18 |
| Spatial scope | Manhattan only; 128,393 filtered trips |
| Stations | 425 real NYC charging-station coordinates mapped to Manhattan TLC zones |
| Station capacity | 4 per station, 1,700 total |
| Fleet | 200 vehicles: 100 Human EVs and 100 AEVs |
| Assignment | Exact MCMF, `primal_dual`, no learning/checkpoint |
| Initial SOC | Mean 0.875, uniform on [0.80, 0.95] |
| Consumption ratio | 1.0 |
| Epoch length | 120 seconds |
| Full-day observation | 00:00--24:00 |

Demand source:
`/Users/seinzhou/Desktop/adp_trainer/nyedata/nye_simulation/parquet/yellow_tripdata_2025-12.parquet`

Station source:
`/Users/seinzhou/Desktop/icaps/nyedata/nyc_all_charging_stations.csv`

## Full-day results

| Decision interval | Seeds | Human EV sessions/vehicle-day | AEV sessions/vehicle-day | Human EV mean completed session duration (min) | Positive charger-queue wait (min) |
|---|---:|---:|---:|---:|---:|
| 60 min | 256 | 3.42 | 0.00 | 32.76 | 0.00 |
| **120 min** | **256, 512, 1024** | **2.3133** | **0.00** | **39.43** | **0.00** |
| 180 min | 256 | 1.92 | 0.00 | 44.57 | 0.00 |

The three 120-minute runs give Human EV rates of 2.41, 2.39, and 2.14;
their population standard deviation is 0.1228. The corresponding all-fleet
average is 1.1567 sessions/vehicle-day. The duration column averages the
per-run completed-session means. The 60- and 180-minute comparisons use one
seed each, so their zero reported standard deviations are not uncertainty
estimates.

Charging frequency counts actual station charging-session starts, not repeated
`ChargingAction` objects. Queue waiting is a separate metric. AEV charging is
zero in these baseline runs; its policy was not changed by this calibration.

These results establish the 2--3 target for the tested high-initial-SOC baseline.
They do not guarantee a hard maximum for every vehicle, seed, lower initial SOC,
or energy-consumption sensitivity setting.

## Short-window diagnostic

The initial 08:00--11:00 three-hour tests gave day-normalized Human EV rates
of 0.88, 0.80, and 0.1333 for 60, 120, and 180 minutes respectively (three
seeds each). They were retained for diagnosis but **not** used to select the
default: high initial SOC and early-hour observations do not represent the
full-day charging process.

## Saved outputs and reproduction

Full-day data and plot:

- `results/human_ev_charge_interval_full_day/detail.csv`
- `results/human_ev_charge_interval_full_day/summary.csv`
- `results/human_ev_charge_interval_full_day/metadata.json`
- `results/human_ev_charge_interval_full_day/human_ev_charge_interval_calibration.png`

Short-window outputs: `results/human_ev_charge_interval_calibration/`.
Smoke-run outputs: `results/human_ev_charge_interval_smoke/`.

Run the selected setting from the ICAPS project root:

```bash
python calibrate_human_ev_charge_interval.py \
  --interval-minutes 120 --seeds 256 512 1024 \
  --start-hour 0 --stop-hour 24 \
  --output-dir results/human_ev_charge_interval_full_day
```

Completed interval/seed rows are skipped. Use a new output directory for a
different initial SOC or observation window. The initial-SOC/consumption
sensitivity runner rejects resuming legacy results that have a different or
unrecorded decision interval, preventing old and new behavior from being
silently pooled. Existing experiment results were preserved.

## Synchronization and verification

All three projects have the same default and NYC behavior in:

- `src/NYCEnvironment.py`
- `src/NYCtrainer.py`
- `run_nyctrainer.py`
- `test_nyc_model.py`

The ICAPS sensitivity runners also expose/record the interval.
The synthetic environments were not changed by this NYC-specific calibration.

Verification completed:

- Four decision-clock regression tests passed independently in each project.
- API and training/evaluation CLI defaults were checked as 120 minutes.
- All modified NYC source files passed syntax checks.
- Each project's NYC training entry point completed a no-learning exact-MCMF
  smoke run with real demand and all 425 stations.
- Probability-function hashes match before and after the changes.
