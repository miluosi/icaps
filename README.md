# ICAPS Mixed-Fleet ADP Reproducibility Project

This repository is a compact reproducibility package for an ICAPS study of mixed electric ride-hailing fleets. It trains and evaluates human-driven electric vehicles (HEVs) and centrally controlled autonomous electric vehicles (AEVs) on either a synthetic grid or NYC Taxi Zones.

The decision problem jointly covers online request assignment, empty-vehicle relocation, state-of-charge feasibility, charging-station capacity and queues, and alternative decision orders for partially controllable and fully controllable vehicles.

This is a standalone project. It does not depend on a sibling checkout, and it intentionally excludes historical results, checkpoints, logs, notebooks, obsolete Python entry points, and unrelated model variants.

## Method overview

The mixed fleet is

$$
\mathcal V
=
\mathcal V^{\mathrm{HEV}}
\cup
\mathcal V^{\mathrm{AEV}},
\qquad
\mathcal V^{\mathrm{HEV}}
\cap
\mathcal V^{\mathrm{AEV}}
=
\varnothing.
$$

At decision epoch `t`, each vehicle `k` has a feasible action set containing request service, relocation, charging, and idle actions:

$$
\mathcal A_k(s_t)
=
\mathcal A_k^{\mathrm{serve}}(s_t)
\cup
\mathcal A_k^{\mathrm{relocate}}(s_t)
\cup
\mathcal A_k^{\mathrm{charge}}(s_t)
\cup
\{a_k^{\mathrm{idle}}\}.
$$

The learned method preserves the structured myopic edge score and adds a learned future-value correction:

$$
\Psi_{k,a,t}
=
g_{k,a,t}
+
\widehat\Delta_{\theta}^{\tau(k)}
\left(\boldsymbol f_{k,a,t}\right),
\qquad
\tau(k)\in\{\mathrm{HEV},\mathrm{AEV}\}.
$$

The assignment layer then solves a joint feasible projection:

$$
A_t^{\star}
\in
\arg\max_{A\in\mathcal F_t}
\sum_{(k,a)\in A}
\Psi_{k,a,t}.
$$

The feasible set enforces exactly one real action (including WAIT/continuation) per represented vehicle, at most one vehicle per request, charging capacity, queue admission limits, and battery feasibility. Exact ILP, exact MCMF, auction, and heuristic backends operate on the same edge-score interface.

## Strategy semantics


| Strategy      | Value-function training assignment | Inference assignment | Checkpoint tag |
| ------------- | ---------------------------------- | -------------------- | -------------- |
| `HEU`         | No value function                  | Heuristic            | None           |
| `ADP-HEU`     | Exact/ILP                          | Heuristic            | `gurobi`       |
| `ADP-HEU-HEU` | Heuristic                          | Heuristic            | `heu`          |

The legacy meaning of `ADP-HEU` is unchanged: it loads a value function trained with exact assignment and applies heuristic assignment only at inference. `ADP-HEU-HEU` is the separate heuristic-training plus heuristic-inference method.

Let the two learned parameter sets be

$$
\theta^{\mathrm{exact}}
=
\operatorname{Train}
\left(\pi^{\mathrm{exact}}\right),
\qquad
\theta^{\mathrm{heu}}
=
\operatorname{Train}
\left(\pi^{\mathrm{heu}}\right).
$$

The three heuristic evaluations are therefore

$$
\mathrm{HEU}
=
\pi^{\mathrm{heu}}(g),
$$

$$
\mathrm{ADP-HEU}
=
\pi^{\mathrm{heu}}
\left(g+\widehat\Delta_{\theta^{\mathrm{exact}}}\right),
$$

$$
\mathrm{ADP-HEU-HEU}
=
\pi^{\mathrm{heu}}
\left(g+\widehat\Delta_{\theta^{\mathrm{heu}}}\right).
$$

Exact-trained and heuristic-trained checkpoints are stored in different directories and never overwrite each other.

## Repository structure

```text
icaps/
├── run_trainer.py          # Synthetic-data training
├── test_model.py           # Synthetic policy comparison
├── run_nyctrainer.py       # NYC training
├── test_nyc_model.py       # NYC policy comparison
├── src/                    # ADP, environments, solvers, and core value models
├── config/                 # Training and sampling configuration
├── tests/                  # Compact regression suite
├── scripts/                # Toy, NYC, and heuristic-comparison workflows
├── nyedata/                # Small real NYC sample and spatial data
├── docs/                   # Mathematical, data, and literature documentation
└── references/             # ICAPS 2021--2025 BibTeX
```

See `docs/PROJECT_MANIFEST.md` for the inclusion policy and `docs/MATHEMATICAL_MODEL.md` for the detailed formulation.

## Installation

Python 3.11 or 3.12 is recommended.

```bash
git clone <repository-url> icaps
cd icaps
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current modules import `gurobipy` even when heuristic assignment is selected. A valid Gurobi licence is required only for ILP or the Gurobi network-flow backend. Without a licence, use heuristic assignment with `--no-mcmf`, or select the `primal_dual` MCMF backend.

## Quick verification

```bash
make test
make smoke-toy
make smoke-nyc
make smoke-adp-heu
```

- `smoke-toy` runs eight vehicles for two synthetic decision steps.
- `smoke-nyc` uses the bundled 2025-12-18 Yellow Taxi sample and simulates approximately three minutes.
- `smoke-adp-heu` crosses the 64-step replay warm-up and executes real value-network updates from heuristic-assignment transitions.

These commands validate the complete software path; they are not paper-scale experiments.

## Recourse experiments

EV-first runs expose the audit-defined `R0`--`R4` variants. Recourse variants are rejected for integrated and AEV-first execution because those modes do not have the EV-leader/AEV-follower semantics.

```bash
python run_trainer.py \
  --episodes 5 \
  --num-vehicles 20 \
  --num-ev 10 \
  --transportation-mode evfirst \
  --recourse-variant r4 \
  --learner-variant optimization_anchored_residual \
  --state-variant joint_state_shared_critic \
  --common-random-numbers \
  --assignment-heuristic \
  --no-mcmf
```

The corresponding NYC flags are identical. `R0` disables behavioral rejection; `R1` blocks same-epoch repair; `R2` uses only structured follower scores; `R3` learns the follower without coupling its value into the EV target; and `R4` uses the full structured-plus-correction AEV target value in the EV leader target. Checkpoint namespaces include the recourse, state, learner, and rejection-stress settings. `--checkpoint-replay {none,recent,full}` controls replay persistence (`recent` stores the newest 5,000 transitions by default). Replay/checkpoint files use Python pickle through `torch.save`/`pickle`; load only artifacts created by a trusted local run. Each completed CLI run writes a `*.manifest.json` beside its statistics workbook with the resolved arguments, commit, data/checkpoint hashes, effective model and replay hyperparameters, seed-clustered uncertainty, and runtime metadata. `config/recourse_experiment.example.json` is a documentation template, not a runner input; the CLI arguments and generated manifest are the executable and resolved configuration sources.

## Synthetic training

Heuristic training, which creates `heu` checkpoints:

```bash
python run_trainer.py \
  --adp 1 \
  --episodes 50 \
  --num-vehicles 200 \
  --num-ev 100 \
  --transportation-mode integrated \
  --assignment-heuristic \
  --no-mcmf \
  --distribution-mode st_masac_gat_queue_demand_gurobi
```

Exact training, which creates `gurobi` checkpoints for legacy `ADP-HEU`:

```bash
python run_trainer.py \
  --adp 1 \
  --episodes 50 \
  --num-vehicles 200 \
  --num-ev 100 \
  --transportation-mode integrated \
  --assignment-gurobi \
  --no-mcmf \
  --distribution-mode st_masac_gat_queue_demand_gurobi
```

Train both checkpoint types under one synthetic scenario and compare the three heuristic strategies:

```bash
EPISODES=50 VEHICLES=200 EVS=100 bash scripts/compare_synthetic_heuristics.sh
```

## NYC training

The following command uses the bundled real-data sample:

```bash
python run_nyctrainer.py \
  --adp 1 \
  --episodes 1 \
  --num-vehicles 20 \
  --num-ev 10 \
  --transportation-mode integrated \
  --assignment-heuristic \
  --no-mcmf \
  --distribution-mode st_masac_gat_post_demand_direct \
  --parquet-path nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet \
  --station-csv nyedata/nyc_all_charging_stations.csv \
  --start-year-month 2025-12 \
  --start-date 2025-12-18 \
  --end-date 2025-12-18 \
  --start-hour 8 \
  --stop-hour 10 \
  --epoch-length 30
```

For full-month or multi-day experiments, replace `--parquet-path` and the date range. See `docs/DATA.md` for sample provenance and full-data integration.

## Outputs

For supervised binary driver-acceptance training on ordinary MCMF offers,
see [MCMF acceptance model](docs/MCMF_ACCEPTANCE_MODEL.md). The standalone
`train_acceptance_model.py` runner supports both synthetic and NYC environments,
keeps train/validation/test simulation seeds separate, and reports probability
calibration against a constant-rate baseline without changing the dispatcher.
For a separate NYC 200-vehicle loss-history, rejection-classification and
pure-MCMF noninterference audit, run `check_nyc_mcmf_acceptance.py`; see
[NYC probability check](docs/NYC_MCMF_ACCEPTANCE_CHECK.md).
The current predictor is a 30-input PyTorch MLP (not logistic regression); see
[neural acceptance model](docs/NEURAL_ACCEPTANCE_MODEL.md). NYC now defaults to
`reject_uniform=True` and a **2 km pickup radius**. To verify these settings, add
`--require-random-rejection --expected-assignment-range-km 2` to the audit command.
Legacy regression checkpoints require new full-feature data and neural retraining.
For the explicitly configured, historical `reject_uniform=False` branch, use
`check_nyc_deterministic_rejection.py`; see
[deterministic rejection check](docs/NYC_DETERMINISTIC_REJECTION_CHECK.md).
It verifies the fixed threshold and reports missing rejection labels without
claiming a successful binary fit on single-class data.

For using that frozen probability as an optional EV Q/residual input and running
paired 200-vehicle Integrated learning experiments, see
[EV acceptance learning ablation](docs/EV_ACCEPTANCE_LEARNING_ABLATION.md).
The feature flag is shared by all registered learners and both simulators.

Training creates `checkpoints/`, `results/`, and statistical output files as needed. These paths are ignored by Git. The clean project contains no historical experimental results or model weights.

## Paper resources

- `docs/ICAPS_2021_2025_adp_mixed_fleet_literature.md`: related ICAPS work from 2021--2025 and historical ICAPS 2025 submission requirements.
- `references/icaps_2021_2025_adp_mixed_fleet.bib`: local BibTeX database.
- `docs/MATHEMATICAL_MODEL.md`: state, feasibility, value learning, and heuristic training/inference definitions.
