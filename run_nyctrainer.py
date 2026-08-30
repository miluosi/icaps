"""
Entrypoint to run NYC zone-based ADP training.
Mirrors run_trainer.py but creates NYCEnvironment instead of
ChargingIntegratedEnvironment, and adds parquet / date parameters.
"""
import argparse
import calendar
import os
import sys
import time
import re
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.NYCEnvironment import DEFAULT_INITIAL_BATTERY_MEAN, NYCEnvironment
from src.ADPtrainer import ADPTrainer
from src.NYCtrainer import NYCTrainer
from src.charging_wait_metrics import aggregate_wait_metrics
from src.value_function_registry import (
    VALUE_FUNCTION_CHOICES,
    get_value_function_class,
    validate_value_function_registry,
)
from src.recourse.types import LEARNER_VARIANTS, STATE_VARIANTS
from src.recourse.config import VARIANT_CHOICES, add_method_arguments, resolve_method_arguments
from src.recourse.manifest import write_experiment_manifest


def _get_value_function_class(distribution_mode: str):
    return get_value_function_class(distribution_mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Run NYC zone-based ADP training")
    from src.acceptance_features import add_acceptance_arguments
    add_acceptance_arguments(parser)
    add_method_arguments(parser)
    parser.add_argument("--paper-parameter-preset", action="store_true",
                        help="Apply the paper-aligned EV preset: 3000 EVs, 24h window, 30s epoch; battery/speed/charge parameters are already defined in NYCEnvironment")
    # --- NYC-specific ---
    parser.add_argument("--parquet-path", type=str, default=None,
                        help="Yellow Taxi parquet path or comma-separated list (overrides automatic Yellow paths)")
    parser.add_argument("--full-demand", action="store_true",
                        help="Combine Yellow Taxi with non-pooled HVFHV trips; default is Yellow-only")
    parser.add_argument("--hvfhv-parquet-path", type=str, default=None,
                        help="HVFHV parquet path or comma-separated list for --full-demand; otherwise downloaded by month")
    parser.set_defaults(full_demand=False)
    parser.add_argument("--start-year-month", type=str, default="2025-12",
                        help="Start year-month for training data (YYYY-MM)")
    parser.add_argument("--end-year-month", type=str, default=None,
                        help="Deprecated compatibility flag. When parquet-path is not provided, the effective end month is inferred from --start-year-month and --episodes")
    parser.add_argument("--coord-csv", type=str, default=None,
                        help="Deprecated compatibility option; official TLC taxi-zone geometry is used")
    parser.add_argument("--station-csv", type=str, default=None,
                        help="Path to nyc_charging_stations.csv")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date for training data (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date for training data (YYYY-MM-DD); defaults to --start-date + episodes - 1 day")
    parser.add_argument("--station-capacity-scale", type=float, default=None,
                        help="Multiply NYC charging station capacity by this factor")
    parser.add_argument("--only-manhattan-zones", action="store_true",
                        help="Restrict NYC demand, relocation zones, and charging stations to Manhattan zones")
    parser.add_argument("--full-nyc-zones", dest="only_manhattan_zones", action="store_false",
                        help="Use all official TLC taxi zones for demand, relocation, and charging stations")
    parser.set_defaults(only_manhattan_zones=True)
    parser.add_argument("--start-hour", type=float, default=7.0,
                        help="Start hour of simulation window (default: 7)")
    parser.add_argument("--stop-hour", type=float, default=22.0,
                        help="Stop hour of simulation window (default: 22)")
    parser.add_argument("--epoch-length", type=float, default=30.0,
                        help="Epoch length in seconds (default: 30)")

    # --- common training args (same as run_trainer.py) ---
    parser.add_argument("--adp", type=float, default=1.0,
                        help="ADP value (0 disables NN training)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes")
    parser.add_argument("--num-vehicles", type=int, default=50,
                        help="Total vehicles")
    parser.add_argument("--num-ev", type=int, default=25,
                        help="EV vehicles")
    parser.add_argument(
        "--initial-battery-mean",
        type=float,
        default=DEFAULT_INITIAL_BATTERY_MEAN,
        help=(
            "Requested mean initial vehicle SOC. The legacy default 0.875 "
            "corresponds to Uniform(0.80, 0.95)."
        ),
    )
    parser.add_argument(
        "--charge-wait-bool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, reserve reachable current charger capacity for "
            "the lowest-SOC AEV rows and make their wait actions infeasible. "
            "Use --no-charge-wait-bool to restore an all-one wait column."
        ),
    )
    parser.add_argument(
        "--human-ev-charge-decision-interval-minutes",
        type=float,
        default=120.0,
        help=(
            "Minimum real-time interval between stochastic Human EV charge "
            "decisions. SOC <= 0.20 bypasses the interval for safety."
        ),
    )
    parser.add_argument("--transportation-mode", type=str, nargs="+", default=["integrated"],
                        choices=["integrated", "integrated_repair", "evfirst", "aevfirst"],
                        help="One or more transportation modes")
    parser.add_argument("--use-intense-requests", action="store_true",
                        help="Compatibility flag only; NYC always uses real parquet demand")
    parser.add_argument("--no-intense-requests", dest="use_intense_requests",
                        action="store_false")
    parser.set_defaults(use_intense_requests=True)
    parser.add_argument("--assignment-gurobi", action="store_true",
                        help="Use Gurobi assignment")
    parser.add_argument("--assignment-heuristic", dest="assignment_gurobi",
                        action="store_false")
    parser.set_defaults(assignment_gurobi=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--checkpoint-replay",
        choices=["none", "recent", "full"],
        default="recent",
        help="Joint replay payload stored in checkpoints",
    )
    parser.add_argument(
        "--checkpoint-replay-recent",
        type=int,
        default=5000,
        help="Number of newest joint transitions saved by --checkpoint-replay recent",
    )
    parser.add_argument("--start-training-episode", type=int, default=0)
    parser.add_argument("--prestep", type=int, default=0,
                        help="Use heuristic-only rollouts for the first N global training steps to accumulate experience before enabling gradient updates")
    parser.add_argument("--training-frequency", type=int, default=10,
                        help="Train value networks every N environment steps; default 10 steps = 5 minutes when epoch=30s")
    parser.add_argument("--use-mcmf", action="store_true", dest="usemcmf")
    parser.add_argument("--no-mcmf", action="store_false", dest="usemcmf")
    parser.set_defaults(usemcmf=True)
    parser.add_argument("--mcmf-use-gpu", action="store_true",
                        help="Use the existing GPU MCMF solver interface when MCMF is enabled")
    parser.add_argument("--mcmf-use-cpu", dest="mcmf_use_gpu", action="store_false",
                        help="Force the CPU MCMF solver even if GPU kernels are available")
    parser.set_defaults(mcmf_use_gpu=False)
    parser.add_argument("--mcmf-solver", choices=["exact", "legacy", "auction"], default="exact")
    parser.add_argument("--mcmf-backend", choices=["auto", "ortools", "gurobi_network", "primal_dual"], default="gurobi_network")
    parser.add_argument(
        "--mcmf-cost-scale",
        type=int,
        default=10_000,
        help="Shared Q-value precision grid; 10000 means four decimal places",
    )
    parser.add_argument("--mcmf-online", dest="mcmf_strict", action="store_false")
    parser.set_defaults(mcmf_strict=True)
    parser.add_argument("--no-mcmf-graph-reduction", dest="mcmf_graph_reduction", action="store_false")
    parser.set_defaults(mcmf_graph_reduction=True)
    parser.add_argument("--mcmf-verify", action="store_true")
    parser.add_argument("--use-auction", action="store_true", dest="useauction",
                        help="Use auction solver as the MCMF backend")
    parser.add_argument("--no-auction", action="store_false", dest="useauction",
                        help="Disable auction solver backend")
    parser.set_defaults(useauction=False)
    parser.add_argument("--auction-use-gpu", action="store_true",
                        help="Use GPU auction solver when auction is enabled")
    parser.add_argument("--auction-use-cpu", dest="auction_use_gpu", action="store_false",
                        help="Force CPU auction solver")
    parser.set_defaults(auction_use_gpu=False)
    parser.add_argument("--auction-epsilon", type=float, default=1e-3,
                        help="Auction bidding epsilon")
    parser.add_argument("--auction-max-rounds", type=int, default=None,
                        help="Maximum auction iterations/rounds before falling back to MCMF")
    parser.add_argument("--auction-top-k", type=int, default=None,
                        help="Keep only each vehicle's top-K feasible auction actions before solving")
    parser.add_argument("--known-reject", action="store_true")
    parser.set_defaults(known_reject=False)
    parser.add_argument("--ifreject", action="store_true", help="Enable EV request rejection model")
    parser.add_argument("--no-ifreject", dest="ifreject", action="store_false", help="Disable EV request rejection model")
    parser.set_defaults(ifreject=True)
    parser.add_argument("--rejection-penalty-base", type=float, default=4.0,
                        help="Fixed reward penalty applied when an EV rejects a request")
    parser.add_argument("--rejection-penalty-per-km", type=float, default=0.35,
                        help="Distance-proportional EV rejection penalty per km")
    parser.add_argument("--rejection-penalty-final-value-ratio", type=float, default=0.25,
                        help="If non-negative, EV rejection reward is -ratio * request final_value when the request is known")
    parser.add_argument("--ifdropoff", action="store_true", help="Enable EV dropout and rejoin model")
    parser.add_argument("--no-ifdropoff", dest="ifdropoff", action="store_false", help="Disable EV dropout and rejoin model")
    parser.set_defaults(ifdropoff=False)
    parser.add_argument("--daily-drop-off", action="store_true", help="Only evaluate EV dropout at daily refresh instead of after each completed trip")
    parser.set_defaults(daily_drop_off=False)
    parser.add_argument("--heuristic-battery-threshold", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=64)
    parser.add_argument("--load-checkpoint", action="store_true",
                        help="Load existing checkpoint before training")
    parser.set_defaults(load_checkpoint=False)
    parser.add_argument(
        "--load-checkpoint-start-date",
        type=str,
        default=None,
        help="Start date used in the checkpoint directory to load, e.g. train on 2025-12-15 but evaluate on 2025-12-18",
    )
    parser.add_argument(
        "--load-checkpoint-end-date",
        type=str,
        default=None,
        help="End date used in the checkpoint directory to load",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=["auto", "latest", "best_reward", "best_loss"],
        default="auto",
        help="Which checkpoint to load when --load-checkpoint is set",
    )
    parser.add_argument(
        "--checkpoint-trained-start-hour",
        type=float,
        default=None,
        help="Simulation start hour used when the checkpoint was trained; used to align time features at inference",
    )
    parser.add_argument(
        "--checkpoint-trained-stop-hour",
        type=float,
        default=None,
        help="Simulation stop hour used when the checkpoint was trained; used to align time features at inference",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to checkpoint directories so new experiments do not overwrite earlier models",
    )
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate mode (no training)")
    parser.set_defaults(evaluate_only=False)
    parser.add_argument(
        "--distribution-mode",
        type=str,
        default=None,
        choices=VALUE_FUNCTION_CHOICES,
        help="ICAPS core NYC value-function mode",
    )
    parser.add_argument(
        "--gat-neighbour-number",
        type=int,
        default=0,
        help="Number of nearest feasible action-graph nodes; 0 is the stable default and K>0 is experimental",
    )
    parser.add_argument(
        "--post-demand-q-weight",
        type=float,
        default=0.0,
        help="Initial TD-learned demand-head coefficient; zero exactly reproduces the base critic initially",
    )
    parser.add_argument(
        "--post-demand-head-lr-multiplier",
        type=float,
        default=10.0,
        help="Learning-rate multiplier for the three TD-learned demand-head coefficients",
    )
    parser.add_argument(
        "--masac-target-entropy-ratio",
        type=float,
        default=0.9,
        help="Target entropy as a fraction of log(candidate count) for masac_baseline/standard_masac_gat",
    )
    parser.add_argument(
        "--residual-target-policy",
        choices=["joint_projection"],
        default="joint_projection",
        help=(
            "Target-action policy for optimization_anchored_residual: "
            "joint_projection selects online target actions through the "
            "serialized feasible graph and evaluates them with target critics"
        ),
    )
    parser.add_argument(
        "--predictor-variant",
        choices=["p0", "p1", "p2", "p3"],
        default="p3",
        help=(
            "Completion-context ablation: p0=no forecast, p1=current "
            "observation, p2=arrival horizon, p3=completion demand plus "
            "arrival queue"
        ),
    )
    parser.add_argument(
        "--recourse-variant",
        choices=VARIANT_CHOICES,
        default="legacy",
        help=(
            "EV-first rejection/recourse experiment: r0=no rejection; "
            "r1=no same-epoch recovery; r2=myopic recovery; r3=learned "
            "recovery with uncoupled EV target; r4=stage-coupled target"
        ),
    )
    parser.add_argument(
        "--rejection-logit-shift",
        type=float,
        default=0.0,
        help="Additive shift to EV acceptance utility for rejection stress tests",
    )
    parser.add_argument(
        "--common-random-numbers",
        action="store_true",
        help="Use deterministic offer-keyed acceptance uniforms for paired experiments",
    )
    parser.add_argument(
        "--state-variant",
        choices=STATE_VARIANTS,
        default="joint_state_separate_critics",
        help="State visibility and critic-sharing ablation",
    )
    parser.add_argument(
        "--learner-variant",
        choices=LEARNER_VARIANTS,
        default="legacy",
        help="Integrated learner target family",
    )
    parser.add_argument(
        "--pretrained-zone-dir",
        type=str,
        default="checkpoints/zone_pretrain",
        help="Root directory produced by pretrain_zonepredictor/pretrain_zone.py and consumed by bayes_simple_pretrain",
    )
    parser.add_argument("--iftransformer", action="store_true",
                        help="Enable path self-attention before the LSTM path encoder. Default off for old checkpoint compatibility")
    parser.set_defaults(iftransformer=False)
    parser.add_argument("--zone-pretrain-output-dir", type=str, default=None,
                        help="Output root for --distribution-mode pretrain_zonepredictor; defaults to --pretrained-zone-dir")
    parser.add_argument("--zone-pretrain-max-steps", type=int, default=None,
                        help="Optional per-episode rollout cap for zone predictor pretraining")
    parser.add_argument("--zone-pretrain-epochs", type=int, default=100,
                        help="Gradient epochs for zone predictor pretraining")
    parser.add_argument("--zone-pretrain-batch-size", type=int, default=64,
                        help="Batch size for zone predictor pretraining")
    parser.add_argument("--zone-pretrain-learning-rate", type=float, default=1e-3,
                        help="Learning rate for zone predictor pretraining")
    parser.add_argument("--zone-pretrain-validation-fraction", type=float, default=0.2,
                        help="Validation fraction for zone predictor pretraining")
    parser.add_argument("--zone-pretrain-label-smoothing", type=float, default=0.02,
                        help="Target label smoothing for zone predictor KL training")
    parser.add_argument("--zone-pretrain-top-k", type=int, default=8,
                        help="Top-K zones to save/print in zone predictor diagnostics")
    parser.add_argument("--all-modes", action="store_true", help="Run all transportation modes instead of only the specified mode")
    parser.add_argument("--all-demand-patterns", action="store_true", help="Compatibility flag only; NYC real-demand runs ignore synthetic demand-pattern sweeps")
    parser.add_argument("--benchmark-solvers-only", action="store_true",
                        help="Skip training and compare Manhattan solver wall time for MCMF vs Gurobi on identical real-demand snapshots")
    parser.add_argument("--benchmark-warmup-steps", type=int, default=40,
                        help="Advance this many epochs before taking benchmark snapshots")
    parser.add_argument("--benchmark-steps", type=int, default=1,
                        help="How many consecutive benchmark snapshots to run")
    parser.add_argument("--benchmark-log-dir", type=str, default="logs",
                        help="Directory for benchmark log files")
    return parser.parse_args()


def apply_paper_parameter_preset(args):
    if not args.paper_parameter_preset:
        return args

    args.num_vehicles = 3000
    args.num_ev = args.num_vehicles//2
    if args.station_csv is None:
        real_station_csv = Path(__file__).resolve().parent / "nyedata" / "nyc_all_charging_stations.csv"
        if real_station_csv.exists():
            args.station_csv = str(real_station_csv)
    if args.station_capacity_scale is None:
        args.station_capacity_scale = 1.0 if args.station_csv else 5.0
    args.start_hour = 0.0
    args.stop_hour = 24.0
    args.epoch_length = 30.0
    args.mcmf_solver = "exact"
    args.mcmf_backend = "primal_dual"
    args.mcmf_strict = True
    args.mcmf_graph_reduction = True
    args.mcmf_verify = True
    args.target_solver_policy = "same_as_rollout_exact"
    if args.start_date is None:
        args.start_date = f"{args.start_year_month}-15"
    if args.end_date is None:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        args.end_date = (start_dt + timedelta(days=max(0, int(args.episodes) - 1))).strftime("%Y-%m-%d")
    return args


def _set_random_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False














def reset_random_seed(seed: int):
    print(f"Changing random seed to {seed}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)










def _parse_year_month(year_month: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{4})-(\d{1,2})", year_month.strip())
    if match is None:
        raise ValueError(f"Invalid year-month '{year_month}', expected YYYY-MM")
    year = int(match.group(1))
    month = int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month in '{year_month}', expected 01-12")
    return year, month


def _iter_year_months(start_year_month: str, end_year_month: str) -> Iterable[str]:
    start_year, start_month = _parse_year_month(start_year_month)
    end_year, end_month = _parse_year_month(end_year_month)
    start_key = start_year * 12 + start_month
    end_key = end_year * 12 + end_month
    if start_key > end_key:
        raise ValueError(
            f"start-year-month {start_year_month} must be earlier than or equal to end-year-month {end_year_month}"
        )

    year, month = start_year, start_month
    while year * 12 + month <= end_key:
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            year += 1
            month = 1


def _add_months(year: int, month: int, delta_months: int) -> tuple[int, int]:
    month_index = (year * 12 + (month - 1)) + delta_months
    return month_index // 12, month_index % 12 + 1


def infer_end_year_month(start_year_month: str, num_episodes: int) -> str:
    if num_episodes <= 0:
        raise ValueError(f"episodes must be positive, got {num_episodes}")

    year, month = _parse_year_month(start_year_month)
    remaining_days = int(num_episodes)
    while True:
        days_in_month = calendar.monthrange(year, month)[1]
        if remaining_days <= days_in_month:
            return f"{year:04d}-{month:02d}"
        remaining_days -= days_in_month
        year, month = _add_months(year, month, 1)


def _resolve_and_validate_date_range(
    start_date: str | None,
    end_date: str | None,
    start_year_month: str,
    num_episodes: int,
) -> tuple[str, str, str]:
    if num_episodes <= 0:
        raise ValueError(f"episodes must be positive, got {num_episodes}")

    if start_date is None:
        start_date = f"{start_year_month}-01"

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid start-date '{start_date}', expected YYYY-MM-DD") from exc

    if end_date is None:
        end_dt = start_dt + timedelta(days=int(num_episodes) - 1)
        end_date = end_dt.strftime("%Y-%m-%d")
    else:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid end-date '{end_date}', expected YYYY-MM-DD") from exc

    if end_dt < start_dt:
        raise ValueError(f"end-date {end_date} must be on or after start-date {start_date}")

    actual_episodes = (end_dt - start_dt).days + 1
    if actual_episodes != int(num_episodes):
        raise ValueError(
            "Date range does not match episodes: "
            f"start-date={start_date}, end-date={end_date} covers {actual_episodes} days, "
            f"but episodes={num_episodes}."
        )

    start_year_month_from_date = start_dt.strftime("%Y-%m")
    if start_year_month_from_date != start_year_month:
        raise ValueError(
            f"start-date {start_date} does not match start-year-month {start_year_month}"
        )

    inferred_end_year_month = infer_end_year_month(start_year_month, num_episodes)
    actual_end_year_month = end_dt.strftime("%Y-%m")
    if actual_end_year_month != inferred_end_year_month:
        raise ValueError(
            "end-date month does not match episodes-derived end-year-month: "
            f"end-date={end_date} -> {actual_end_year_month}, expected {inferred_end_year_month}."
        )

    return start_date, end_date, inferred_end_year_month


def _download_nyc_parquet(
    target_file: Path,
    year_month: str,
    demand_kind: str = "yellow",
) -> None:
    import requests

    if demand_kind not in {"yellow", "fhvhv"}:
        raise ValueError(f"Unsupported NYC demand kind: {demand_kind}")
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{demand_kind}_tripdata_{year_month}.parquet"
    print(f"📥 Downloading missing NYC {demand_kind} parquet for {year_month}: {url}")
    target_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = target_file.with_suffix(target_file.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary_file.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output_file.write(chunk)
        temporary_file.replace(target_file)
    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise
    print(f"   ✓ Saved {target_file}")


def resolve_nyc_parquet_paths(
    parquet_path: str | None,
    start_year_month: str,
    end_year_month: str | None,
    num_episodes: int,
    *,
    full_demand: bool = False,
    hvfhv_parquet_path: str | None = None,
) -> list[str]:
    def resolve_explicit_paths(raw_value: str, label: str) -> list[str]:
        raw_paths = [part.strip() for part in raw_value.split(",") if part.strip()]
        resolved = [str(Path(part).expanduser().resolve()) for part in raw_paths]
        missing = [path for path in resolved if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"{label} parquet file(s) not found: {missing}")
        return resolved

    if hvfhv_parquet_path and not full_demand:
        raise ValueError("--hvfhv-parquet-path requires --full-demand")

    if parquet_path:
        yellow_paths = resolve_explicit_paths(parquet_path, "Yellow Taxi")
        if not full_demand:
            return yellow_paths
        if hvfhv_parquet_path:
            return yellow_paths + resolve_explicit_paths(
                hvfhv_parquet_path, "HVFHV"
            )
    else:
        yellow_paths = []

    inferred_end_year_month = infer_end_year_month(start_year_month, num_episodes)
    if end_year_month and end_year_month != inferred_end_year_month:
        print(
            f"ℹ Ignoring --end-year-month={end_year_month}; using "
            f"{start_year_month}..{inferred_end_year_month} derived from start month + episodes={num_episodes}"
        )
    end_year_month = inferred_end_year_month

    parquet_dir = Path(__file__).resolve().parent / "nyedata" / "nye_simulation" / "parquet"
    year_months = list(_iter_year_months(start_year_month, end_year_month))
    if not yellow_paths:
        for year_month in year_months:
            parquet_file = parquet_dir / f"yellow_tripdata_{year_month}.parquet"
            if not parquet_file.exists():
                _download_nyc_parquet(parquet_file, year_month, "yellow")
            yellow_paths.append(str(parquet_file.resolve()))

    if not full_demand:
        return yellow_paths

    if hvfhv_parquet_path:
        hvfhv_paths = resolve_explicit_paths(hvfhv_parquet_path, "HVFHV")
    else:
        hvfhv_paths = []
        for year_month in year_months:
            parquet_file = parquet_dir / f"fhvhv_tripdata_{year_month}.parquet"
            if not parquet_file.exists():
                _download_nyc_parquet(parquet_file, year_month, "fhvhv")
            hvfhv_paths.append(str(parquet_file.resolve()))
    return yellow_paths + hvfhv_paths


def _create_nyc_environment(
    *,
    num_vehicles: int,
    num_ev: int,
    parquet_paths: list[str],
    full_demand: bool = False,
    coord_csv: str | None,
    station_csv: str | None,
    start_date: str | None,
    end_date: str | None,
    station_capacity_scale: float,
    epoch_length: float,
    start_hour: float,
    stop_hour: float,
    heuristic_battery_threshold: float,
    use_intense_requests: bool,
    assignmentgurobi: bool,
    usemcmf: bool,
    mcmf_solver: str = "exact",
    mcmf_backend: str = "gurobi_network",
    mcmf_strict: bool = True,
    mcmf_cost_scale: int = 10_000,
    mcmf_graph_reduction: bool = True,
    mcmf_verify: bool = False,
    useauction: bool,
    auction_use_gpu: bool,
    auction_epsilon: float,
    auction_max_rounds: int | None,
    auction_top_k: int | None,
    ifsolveauctioncuda: bool = False,
    knownreject: bool,
    random_seed: int,
    daily_drop_off: bool,
    ifreject: bool,
    ifdropoff: bool,
    rejection_penalty_base: float = 4.0,
    rejection_penalty_per_km: float = 0.35,
    rejection_penalty_final_value_ratio: float = 0.25,
    recourse_variant: str = "legacy",
    rejection_logit_shift: float = 0.0,
    common_random_numbers: bool = False,
    integrated_repair_hold_enabled: bool = True,
    target_solver_policy: str = "same_as_rollout_exact",
    only_manhattan_zones: bool = False,
    battery_consumption_ratio: float = 1.0,
    initial_battery_mean: float = DEFAULT_INITIAL_BATTERY_MEAN,
    charge_wait_bool: bool = True,
    human_ev_charge_decision_interval_minutes: float = 120.0,
):
    print(f"NYCEnvironment zone filter flag: ifonlymanhatten={only_manhattan_zones}")
    env = NYCEnvironment(
        num_vehicles=num_vehicles,
        num_stations=5,
        ev_num_vehicles=num_ev,
        parquet_path=parquet_paths,
        full_demand=full_demand,
        coord_csv=coord_csv,
        station_csv=station_csv,
        start_date=start_date,
        end_date=end_date,
        station_capacity_scale=station_capacity_scale,
        epoch_length_sec=epoch_length,
        start_hour=start_hour,
        stop_hour=stop_hour,
        episode_length=max(1, int(((stop_hour - start_hour) * 3600) / epoch_length)),
        heuristic_battery_threshold=heuristic_battery_threshold,
        use_intense_requests=use_intense_requests,
        assignmentgurobi=assignmentgurobi,
        usemcmf=usemcmf,
        mcmf_solver=mcmf_solver,
        mcmf_backend=mcmf_backend,
        mcmf_strict=mcmf_strict,
        mcmf_cost_scale=mcmf_cost_scale,
        mcmf_graph_reduction=mcmf_graph_reduction,
        mcmf_verify=mcmf_verify,
        useauction=useauction,
        auction_use_gpu=auction_use_gpu,
        auction_epsilon=auction_epsilon,
        auction_max_rounds=auction_max_rounds,
        auction_top_k=auction_top_k,
        ifsolveauctioncuda=ifsolveauctioncuda,
        knownreject=knownreject,
        gurobi_network=True,
        gurobi_network_lp=True,
        daily_drop_off=daily_drop_off,
        ifreject=ifreject,
        ifdropoff=ifdropoff,
        rejection_penalty_base=rejection_penalty_base,
        rejection_penalty_per_km=rejection_penalty_per_km,
        rejection_penalty_final_value_ratio=rejection_penalty_final_value_ratio,
        ifonlymanhatten=only_manhattan_zones,
        random_seed=random_seed,
        battery_consumption_ratio=battery_consumption_ratio,
        initial_battery_mean=initial_battery_mean,
        charge_wait_bool=charge_wait_bool,
        human_ev_charge_decision_interval_minutes=human_ev_charge_decision_interval_minutes,
    )
    env.configure_recourse_experiment(
        recourse_variant,
        rejection_logit_shift=rejection_logit_shift,
        common_random_numbers=common_random_numbers,
        integrated_repair_hold_enabled=integrated_repair_hold_enabled,
        target_solver_policy=target_solver_policy,
    )
    return env


def run_nyc_solver_benchmark(
    *,
    num_episodes: int,
    num_vehicles: int,
    num_ev: int,
    use_intense_requests: bool,
    heuristic_battery_threshold: float,
    random_seed: int,
    parquet_path: str | None,
    full_demand: bool = False,
    hvfhv_parquet_path: str | None = None,
    start_year_month: str,
    end_year_month: str,
    start_date: str | None,
    end_date: str | None,
    coord_csv: str | None,
    station_csv: str | None,
    station_capacity_scale: float,
    start_hour: float,
    stop_hour: float,
    epoch_length: float,
    knownreject: bool,
    mcmf_use_gpu: bool,
    mcmf_solver: str = "exact",
    mcmf_backend: str = "gurobi_network",
    mcmf_strict: bool = True,
    mcmf_cost_scale: int = 10_000,
    mcmf_graph_reduction: bool = True,
    mcmf_verify: bool = False,
    useauction: bool,
    auction_use_gpu: bool,
    auction_epsilon: float,
    auction_max_rounds: int | None,
    auction_top_k: int | None,
    daily_drop_off: bool,
    ifreject: bool,
    ifdropoff: bool,
    warmup_steps: int,
    benchmark_steps: int,
    log_dir: str,
    only_manhattan_zones: bool = False,
):
    trainer = NYCTrainer(
        create_environment=_create_nyc_environment,
        resolve_parquet_paths=resolve_nyc_parquet_paths,
        get_value_function_class=_get_value_function_class,
        set_random_seeds=_set_random_seeds,
    )
    return trainer.run_nyc_solver_benchmark(
        num_episodes=num_episodes,
        num_vehicles=num_vehicles,
        num_ev=num_ev,
        use_intense_requests=use_intense_requests,
        heuristic_battery_threshold=heuristic_battery_threshold,
        random_seed=random_seed,
        parquet_path=parquet_path,
        full_demand=full_demand,
        hvfhv_parquet_path=hvfhv_parquet_path,
        start_year_month=start_year_month,
        end_year_month=end_year_month,
        start_date=start_date,
        end_date=end_date,
        coord_csv=coord_csv,
        station_csv=station_csv,
        station_capacity_scale=station_capacity_scale,
        start_hour=start_hour,
        stop_hour=stop_hour,
        epoch_length=epoch_length,
        knownreject=knownreject,
        mcmf_use_gpu=mcmf_use_gpu,
        mcmf_solver=mcmf_solver,
        mcmf_backend=mcmf_backend,
        mcmf_strict=mcmf_strict,
        mcmf_cost_scale=mcmf_cost_scale,
        mcmf_graph_reduction=mcmf_graph_reduction,
        mcmf_verify=mcmf_verify,
        useauction=useauction,
        auction_use_gpu=auction_use_gpu,
        auction_epsilon=auction_epsilon,
        auction_max_rounds=auction_max_rounds,
        auction_top_k=auction_top_k,
        daily_drop_off=daily_drop_off,
        ifreject=ifreject,
        ifdropoff=ifdropoff,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        log_dir=log_dir,
        only_manhattan_zones=only_manhattan_zones,
    )


def run_nyc_training(
    *,
    adpvalue: float,
    num_episodes: int,
    use_intense_requests: bool,
    assignmentgurobi: bool,
    batch_size: int,
    num_vehicles: int,
    num_ev: int,
    heuristic_battery_threshold: float,
    transportation_mode: str,
    start_training_episode: int,
    usemcmf: bool,
    knownreject: bool,
    mcmf_use_gpu: bool,
    mcmf_solver: str = "exact",
    mcmf_backend: str = "gurobi_network",
    mcmf_strict: bool = True,
    mcmf_cost_scale: int = 10_000,
    mcmf_graph_reduction: bool = True,
    mcmf_verify: bool = False,
    useauction: bool,
    auction_use_gpu: bool,
    auction_epsilon: float,
    auction_max_rounds: int | None,
    auction_top_k: int | None,
    ifsolveauctioncuda: bool = False,
    ifloadcheckpoint: bool,
    trainnetwork: bool,
    random_seed: int,
    parquet_path: str | None,
    full_demand: bool = False,
    hvfhv_parquet_path: str | None = None,
    start_year_month: str,
    end_year_month: str,
    start_date: str | None = None,
    end_date: str | None = None,
    coord_csv: str | None,
    station_csv: str | None,
    load_checkpoint_assign_tag: str | None = None,
    station_capacity_scale: float = 1.0,
    start_hour: float = 7.0,
    stop_hour: float = 22.0,
    epoch_length: float = 30.0,
    zone_distribution_mode: str = "none",
    daily_drop_off: bool = False,
    ifreject: bool = True,
    ifdropoff: bool = False,
    rejection_penalty_base: float = 4.0,
    rejection_penalty_per_km: float = 0.35,
    rejection_penalty_final_value_ratio: float = 0.25,
    prestep: int = 0,
    training_frequency: int = 10,
    load_best_loss: bool = False,
    checkpoint_selection: str = "auto",
    only_manhattan_zones: bool = False,
    load_checkpoint_start_date: str | None = None,
    load_checkpoint_end_date: str | None = None,
    checkpoint_trained_start_hour: float | None = None,
    checkpoint_trained_stop_hour: float | None = None,
    checkpoint_suffix: str | None = None,
    pretrained_zone_dir: str = "checkpoints/zone_pretrain",
    iftransformer: bool = False,
    gat_neighbour_number: int = 0,
    post_demand_q_weight: float = 0.0,
    checkpoint_replay: str = "recent",
    checkpoint_replay_recent: int = 5_000,
    post_demand_head_lr_multiplier: float = 10.0,
    masac_target_entropy_ratio: float = 0.9,
    residual_target_policy: str = "joint_projection",
    predictor_variant: str = "p3",
    recourse_variant: str = "legacy",
    rejection_logit_shift: float = 0.0,
    common_random_numbers: bool = False,
    integrated_repair_hold_enabled: bool = True,
    target_solver_policy: str = "same_as_rollout_exact",
    state_variant: str = "joint_state_separate_critics",
    learner_variant: str = "legacy",
    ev_acceptance_feature: str = "off",
    ev_acceptance_model: str | None = None,
        ev_response_anchor: str = 'auto',
        ev_response_critic_input: str = 'q_mask',
    battery_consumption_ratio: float = 1.0,
    initial_battery_mean: float = DEFAULT_INITIAL_BATTERY_MEAN,
    charge_wait_bool: bool = True,
    human_ev_charge_decision_interval_minutes: float = 120.0,
):
    """Compatibility wrapper that delegates NYC training to src.NYCtrainer.NYCTrainer."""

    trainer = NYCTrainer(
        create_environment=_create_nyc_environment,
        resolve_parquet_paths=resolve_nyc_parquet_paths,
        get_value_function_class=_get_value_function_class,
        set_random_seeds=_set_random_seeds,
    )
    return trainer.run_nyc_training(
        adpvalue=adpvalue,
        num_episodes=num_episodes,
        use_intense_requests=use_intense_requests,
        assignmentgurobi=assignmentgurobi,
        batch_size=batch_size,
        num_vehicles=num_vehicles,
        num_ev=num_ev,
        heuristic_battery_threshold=heuristic_battery_threshold,
        transportation_mode=transportation_mode,
        start_training_episode=start_training_episode,
        usemcmf=usemcmf,
        knownreject=knownreject,
        mcmf_use_gpu=mcmf_use_gpu,
        mcmf_solver=mcmf_solver,
        mcmf_backend=mcmf_backend,
        mcmf_strict=mcmf_strict,
        mcmf_cost_scale=mcmf_cost_scale,
        mcmf_graph_reduction=mcmf_graph_reduction,
        mcmf_verify=mcmf_verify,
        useauction=useauction,
        auction_use_gpu=auction_use_gpu,
        auction_epsilon=auction_epsilon,
        auction_max_rounds=auction_max_rounds,
        auction_top_k=auction_top_k,
        ifsolveauctioncuda=ifsolveauctioncuda,
        ifloadcheckpoint=ifloadcheckpoint,
        trainnetwork=trainnetwork,
        random_seed=random_seed,
        parquet_path=parquet_path,
        full_demand=full_demand,
        hvfhv_parquet_path=hvfhv_parquet_path,
        start_year_month=start_year_month,
        end_year_month=end_year_month,
        start_date=start_date,
        end_date=end_date,
        coord_csv=coord_csv,
        station_csv=station_csv,
        load_checkpoint_assign_tag=load_checkpoint_assign_tag,
        station_capacity_scale=station_capacity_scale,
        start_hour=start_hour,
        stop_hour=stop_hour,
        epoch_length=epoch_length,
        zone_distribution_mode=zone_distribution_mode,
        daily_drop_off=daily_drop_off,
        ifreject=ifreject,
        ifdropoff=ifdropoff,
        rejection_penalty_base=rejection_penalty_base,
        rejection_penalty_per_km=rejection_penalty_per_km,
        rejection_penalty_final_value_ratio=rejection_penalty_final_value_ratio,
        prestep=prestep,
        training_frequency=training_frequency,
        load_best_loss=load_best_loss,
        checkpoint_selection=checkpoint_selection,
        only_manhattan_zones=only_manhattan_zones,
        load_checkpoint_start_date=load_checkpoint_start_date,
        load_checkpoint_end_date=load_checkpoint_end_date,
        checkpoint_trained_start_hour=checkpoint_trained_start_hour,
        checkpoint_trained_stop_hour=checkpoint_trained_stop_hour,
        checkpoint_suffix=checkpoint_suffix,
        pretrained_zone_dir=pretrained_zone_dir,
        iftransformer=iftransformer,
        gat_neighbour_number=gat_neighbour_number,
        post_demand_q_weight=post_demand_q_weight,
        checkpoint_replay=checkpoint_replay,
        checkpoint_replay_recent=checkpoint_replay_recent,
        post_demand_head_lr_multiplier=post_demand_head_lr_multiplier,
        masac_target_entropy_ratio=masac_target_entropy_ratio,
        residual_target_policy=residual_target_policy,
        predictor_variant=predictor_variant,
        recourse_variant=recourse_variant,
        rejection_logit_shift=rejection_logit_shift,
        common_random_numbers=common_random_numbers,
        integrated_repair_hold_enabled=integrated_repair_hold_enabled,
        target_solver_policy=target_solver_policy,
        state_variant=state_variant,
        learner_variant=learner_variant,
        ev_acceptance_feature=ev_acceptance_feature,
        ev_acceptance_model=ev_acceptance_model,
        ev_response_anchor=ev_response_anchor,
        ev_response_critic_input=ev_response_critic_input,
        battery_consumption_ratio=battery_consumption_ratio,
        initial_battery_mean=initial_battery_mean,
        charge_wait_bool=charge_wait_bool,
        human_ev_charge_decision_interval_minutes=human_ev_charge_decision_interval_minutes,
    )


def main():
    args = resolve_method_arguments(apply_paper_parameter_preset(parse_args()))
    validate_value_function_registry()
    if args.recourse_variant != "legacy":
        invalid_modes = [
            mode for mode in args.transportation_mode if mode != "evfirst"
        ]
        if invalid_modes or args.all_modes:
            raise ValueError(
                f"recourse variant {args.recourse_variant} is defined only for "
                "--transportation-mode evfirst"
            )
    if args.useauction:
        args.usemcmf = True
    if args.station_capacity_scale is None:
        args.station_capacity_scale = 1.0
    args.start_date, args.end_date, inferred_end_year_month = _resolve_and_validate_date_range(
        args.start_date,
        args.end_date,
        args.start_year_month,
        args.episodes,
    )
    if args.end_year_month and args.end_year_month != inferred_end_year_month:
        print(
            f"ℹ Ignoring --end-year-month={args.end_year_month}; using "
            f"{inferred_end_year_month} derived from date range and episodes={args.episodes}"
        )
    args.end_year_month = inferred_end_year_month
    zone_distribution_mode = (
        args.learner_variant
        if args.learner_variant != "legacy"
        else (args.distribution_mode or "none")
    )
    if args.learner_variant == "integrated_directq" and (
        args.all_modes
        or any(mode not in {"integrated", "integrated_repair"} for mode in args.transportation_mode)
    ):
        raise ValueError(
            "integrated_directq requires --transportation-mode integrated"
        )
    experiment_namespace = (
        f"rec-{args.recourse_variant}_state-{args.state_variant}_"
        f"learner-{args.learner_variant}_shift-{args.rejection_logit_shift:g}"
    )
    args.checkpoint_suffix = "_".join(
        part for part in (args.checkpoint_suffix, experiment_namespace) if part
    )
    parquet_desc = args.parquet_path or f"{args.start_year_month}..{args.end_year_month}"
    demand_desc = "yellow+hvfhv_nonshared" if args.full_demand else "yellow_only"

    print("NYC ADP Training")
    print(f"  ADP={args.adp}, episodes={args.episodes}, vehicles={args.num_vehicles}, ev={args.num_ev}")
    print(f"  mode={args.transportation_mode}, gurobi={args.assignment_gurobi}, mcmf={args.usemcmf}, auction={args.useauction}")
    if args.useauction:
        print(f"  auction_solver={'GPU' if args.auction_use_gpu else 'CPU'}, epsilon={args.auction_epsilon}, max_rounds={args.auction_max_rounds}, top_k={args.auction_top_k}")
    print(f"  parquet={parquet_desc}, hours={args.start_hour}-{args.stop_hour}")
    print(f"  demand={demand_desc}, full_demand={args.full_demand}")
    if args.full_demand and args.hvfhv_parquet_path:
        print(f"  hvfhv_parquet={args.hvfhv_parquet_path}")
    print(f"  dates={args.start_date}..{args.end_date}")
    print(f"  distribution_mode={zone_distribution_mode}")
    if zone_distribution_mode in {
        "st_masac_gat",
        "st_masac_gat_post_demand",
        "st_masac_gat_post_demand_direct",
        "standard_masac_gat",
        "standard_masac_gat_total_q",
        "optimization_anchored_residual",
        "standard_masac_gat_greedy_alpha",
        "standard_masac_gat_fixed_alpha",
        "st_masac_gat_frozen",
        "st_masac_gat_neighbour_frozen",
    }:
        print(f"  gat_neighbour_number={args.gat_neighbour_number}")
    if zone_distribution_mode in {
        "st_masac_gat_post_demand_direct",
        "standard_masac_gat",
        "standard_masac_gat_total_q",
        "optimization_anchored_residual",
        "standard_masac_gat_greedy_alpha",
        "standard_masac_gat_fixed_alpha",
    }:
        print(f"  post_demand_q_weight={args.post_demand_q_weight:g}")
        print(
            "  post_demand_head_lr_multiplier="
            f"{args.post_demand_head_lr_multiplier:g}"
        )
    if zone_distribution_mode in {
        "masac_baseline",
        "standard_masac_gat",
        "standard_masac_gat_total_q",
        "optimization_anchored_residual",
        "standard_masac_gat_greedy_alpha",
        "standard_masac_gat_fixed_alpha",
    }:
        print(
            "  masac_target_entropy_ratio="
            f"{args.masac_target_entropy_ratio:g}"
        )
    if zone_distribution_mode == "optimization_anchored_residual":
        print(
            f"  residual_target_policy={args.residual_target_policy}, "
            f"predictor_variant={args.predictor_variant}"
        )
    print(
        f"  recourse_variant={args.recourse_variant}, "
        f"rejection_logit_shift={args.rejection_logit_shift:g}, "
        f"common_random_numbers={args.common_random_numbers}"
    )
    print(f"  iftransformer={args.iftransformer}")
    print(
        f"  daily_drop_off={args.daily_drop_off}, ifreject={args.ifreject}, "
        f"known_reject={args.known_reject}, ifdropoff={args.ifdropoff}"
    )
    if args.rejection_penalty_final_value_ratio >= 0.0:
        print(f"  rejection_penalty={args.rejection_penalty_final_value_ratio:.4g} * final_value (fallback base {args.rejection_penalty_base:.4g} + {args.rejection_penalty_per_km:.4g}/km)")
    else:
        print(f"  rejection_penalty=base {args.rejection_penalty_base:.4g} + {args.rejection_penalty_per_km:.4g}/km")
    print(f"  real_demand_source={demand_desc} (compat_flag_use_intense_requests={args.use_intense_requests})")
    if args.paper_parameter_preset:
        print("  paper_preset=enabled")
        print("  paper_params: ev_model=Tesla Model 3 standard range, battery=51.25kWh, consumption=230Wh/mi, charge=20kW, avg_speed=11.21mph, fleet=3000 EVs")

    transportation_mode_list = ["evfirst", "integrated", "aevfirst"] if args.all_modes else list(dict.fromkeys(args.transportation_mode))

    if args.all_demand_patterns:
        print("  note: --all-demand-patterns ignored for NYC real-demand runs")

    if zone_distribution_mode == "pretrain_zonepredictor":
        requested_modes = list(dict.fromkeys(args.transportation_mode))
        if args.all_modes:
            pretrain_modes = ["evfirst", "aevfirst"]
        else:
            pretrain_modes = [mode for mode in requested_modes if mode in {"evfirst", "aevfirst"}]
            if not pretrain_modes:
                pretrain_modes = ["evfirst", "aevfirst"]
        skipped_modes = [mode for mode in requested_modes if mode not in {"evfirst", "aevfirst"}]
        if skipped_modes:
            print(f"  pretrain_zonepredictor skips unsupported mode(s): {skipped_modes}")

        auction_mode = "torch CUDA auction" if (
            torch.cuda.is_available() if args.auction_use_gpu is None else bool(args.auction_use_gpu and torch.cuda.is_available())
        ) else "CPU auction"
        print("\nNYC Zone Predictor Pretraining")
        print(f"  modes={pretrain_modes}")
        print(f"  rollout_solver={auction_mode}")
        print(f"  output_dir={args.zone_pretrain_output_dir or args.pretrained_zone_dir}")
        print("  objective=minimize KL(target zone distribution || predictor zone distribution)")

        from pretrain_zone import run_zone_predictor_pretraining

        pretrain_result = run_zone_predictor_pretraining(
            argparse.Namespace(
                transportation_modes=pretrain_modes,
                episodes=args.episodes,
                num_vehicles=args.num_vehicles,
                num_ev=args.num_ev,
                start_year_month=args.start_year_month,
                end_year_month=args.end_year_month,
                start_date=args.start_date,
                end_date=args.end_date,
                parquet_path=args.parquet_path,
                full_demand=args.full_demand,
                hvfhv_parquet_path=args.hvfhv_parquet_path,
                coord_csv=args.coord_csv,
                station_csv=args.station_csv,
                station_capacity_scale=args.station_capacity_scale,
                start_hour=args.start_hour,
                stop_hour=args.stop_hour,
                epoch_length=args.epoch_length,
                max_steps=args.zone_pretrain_max_steps,
                epochs=args.zone_pretrain_epochs,
                batch_size=args.zone_pretrain_batch_size,
                learning_rate=args.zone_pretrain_learning_rate,
                validation_fraction=args.zone_pretrain_validation_fraction,
                label_smoothing=args.zone_pretrain_label_smoothing,
                top_k=args.zone_pretrain_top_k,
                auction_use_gpu=args.auction_use_gpu,
                auction_epsilon=args.auction_epsilon,
                auction_max_rounds=args.auction_max_rounds,
                auction_top_k=args.auction_top_k,
                random_seed=args.random_seed,
                known_reject=args.known_reject,
                only_manhattan_zones=args.only_manhattan_zones,
                output_dir=args.zone_pretrain_output_dir or args.pretrained_zone_dir,
            )
        )
        print(f"Zone predictor pretraining complete: {pretrain_result['manifest_path']}")
        print("Then train Q network with: --distribution-mode bayes_simple_pretrain --pretrained-zone-dir <same output dir>")
        return

    if args.benchmark_solvers_only:
        benchmark_result = run_nyc_solver_benchmark(
            num_episodes=args.episodes,
            num_vehicles=args.num_vehicles,
            num_ev=args.num_ev,
            use_intense_requests=args.use_intense_requests,
            heuristic_battery_threshold=args.heuristic_battery_threshold,
            random_seed=args.random_seed,
            parquet_path=args.parquet_path,
            full_demand=args.full_demand,
            hvfhv_parquet_path=args.hvfhv_parquet_path,
            start_year_month=args.start_year_month,
            end_year_month=args.end_year_month,
            start_date=args.start_date,
            end_date=args.end_date,
            coord_csv=args.coord_csv,
            station_csv=args.station_csv,
            station_capacity_scale=args.station_capacity_scale,
            start_hour=args.start_hour,
            stop_hour=args.stop_hour,
            epoch_length=args.epoch_length,
            knownreject=args.known_reject,
            mcmf_use_gpu=args.mcmf_use_gpu,
            mcmf_solver=args.mcmf_solver,
            mcmf_backend=args.mcmf_backend,
            mcmf_strict=args.mcmf_strict,
            mcmf_cost_scale=args.mcmf_cost_scale,
            mcmf_graph_reduction=args.mcmf_graph_reduction,
            mcmf_verify=args.mcmf_verify,
            useauction=args.useauction,
            auction_use_gpu=args.auction_use_gpu,
            auction_epsilon=args.auction_epsilon,
            auction_max_rounds=args.auction_max_rounds,
            auction_top_k=args.auction_top_k,
            daily_drop_off=args.daily_drop_off,
            ifreject=args.ifreject,
            ifdropoff=args.ifdropoff,
            warmup_steps=args.benchmark_warmup_steps,
            benchmark_steps=args.benchmark_steps,
            log_dir=args.benchmark_log_dir,
            only_manhattan_zones=args.only_manhattan_zones,
        )
        print(f"Benchmark complete. Log: {benchmark_result['log_path']}")
        return


    for mode in transportation_mode_list:
        print(f"\n--- real_demand_run, mode={mode} ---")
        print(f"Zone scope: {'Manhattan only' if args.only_manhattan_zones else 'full NYC CSV zones'}")
        results, env = run_nyc_training(
            adpvalue=args.adp,
            num_episodes=args.episodes,
            use_intense_requests=args.use_intense_requests,
            assignmentgurobi=args.assignment_gurobi,
            batch_size=args.batch_size,
            num_vehicles=args.num_vehicles,
            num_ev=args.num_ev,
            heuristic_battery_threshold=args.heuristic_battery_threshold,
            transportation_mode=mode,
            start_training_episode=args.start_training_episode,
            usemcmf=args.usemcmf,
            mcmf_use_gpu=args.mcmf_use_gpu,
            mcmf_solver=args.mcmf_solver,
            mcmf_backend=args.mcmf_backend,
            mcmf_strict=args.mcmf_strict,
            mcmf_cost_scale=args.mcmf_cost_scale,
            mcmf_graph_reduction=args.mcmf_graph_reduction,
            mcmf_verify=args.mcmf_verify,
            useauction=args.useauction,
            auction_use_gpu=args.auction_use_gpu,
            auction_epsilon=args.auction_epsilon,
            auction_max_rounds=args.auction_max_rounds,
            auction_top_k=args.auction_top_k,
            knownreject=args.known_reject,
            ifloadcheckpoint=args.load_checkpoint,
            trainnetwork=not args.evaluate_only,
            random_seed=args.random_seed,
            parquet_path=args.parquet_path,
            full_demand=args.full_demand,
            hvfhv_parquet_path=args.hvfhv_parquet_path,
            start_year_month=args.start_year_month,
            end_year_month=args.end_year_month,
            start_date=args.start_date,
            end_date=args.end_date,
            coord_csv=args.coord_csv,
            station_csv=args.station_csv,
            station_capacity_scale=args.station_capacity_scale,
            start_hour=args.start_hour,
            stop_hour=args.stop_hour,
            epoch_length=args.epoch_length,
            zone_distribution_mode=zone_distribution_mode,
            daily_drop_off=args.daily_drop_off,
            ifreject=args.ifreject,
            ifdropoff=args.ifdropoff,
            rejection_penalty_base=args.rejection_penalty_base,
            rejection_penalty_per_km=args.rejection_penalty_per_km,
            rejection_penalty_final_value_ratio=args.rejection_penalty_final_value_ratio,
            prestep=args.prestep,
            training_frequency=args.training_frequency,
            checkpoint_selection=args.checkpoint_selection,
            only_manhattan_zones=args.only_manhattan_zones,
            load_checkpoint_start_date=args.load_checkpoint_start_date,
            load_checkpoint_end_date=args.load_checkpoint_end_date,
            checkpoint_trained_start_hour=args.checkpoint_trained_start_hour,
            checkpoint_trained_stop_hour=args.checkpoint_trained_stop_hour,
            checkpoint_suffix=args.checkpoint_suffix,
            pretrained_zone_dir=args.pretrained_zone_dir,
            iftransformer=args.iftransformer,
            gat_neighbour_number=args.gat_neighbour_number,
            post_demand_q_weight=args.post_demand_q_weight,
            checkpoint_replay=args.checkpoint_replay,
            checkpoint_replay_recent=args.checkpoint_replay_recent,
            post_demand_head_lr_multiplier=args.post_demand_head_lr_multiplier,
            masac_target_entropy_ratio=args.masac_target_entropy_ratio,
            residual_target_policy=args.residual_target_policy,
            predictor_variant=args.predictor_variant,
            recourse_variant=args.recourse_variant,
            rejection_logit_shift=args.rejection_logit_shift,
            common_random_numbers=args.common_random_numbers,
            integrated_repair_hold_enabled=args.integrated_repair_hold_enabled,
            target_solver_policy=args.target_solver_policy,
            state_variant=args.state_variant,
            learner_variant=args.learner_variant,
            ev_acceptance_feature=args.ev_acceptance_feature,
            ev_acceptance_model=args.ev_acceptance_model,
                ev_response_anchor=args.ev_response_anchor,
                ev_response_critic_input=args.ev_response_critic_input,
            initial_battery_mean=args.initial_battery_mean,
            charge_wait_bool=args.charge_wait_bool,
            human_ev_charge_decision_interval_minutes=(
                args.human_ev_charge_decision_interval_minutes
            ),
        )

        print(f"\nFinished: {len(results.get('episode_rewards', []))} episodes")
        if results.get('episode_rewards'):
            print(f"Avg reward: {sum(results['episode_rewards'])/len(results['episode_rewards']):.2f}")
        if results.get('drop_off_rates'):
            print(f"Avg drop-off rate: {sum(results['drop_off_rates'])/len(results['drop_off_rates']):.4f}")
        detailed = results.get('episode_detailed_stats', [])
        if detailed:
            wait_summary = aggregate_wait_metrics(detailed)
            print(
                "Avg wait among waiting vehicles: "
                f"{wait_summary['avg_wait']:.2f} steps "
                f"(mean waiting vehicles/episode: "
                f"{wait_summary['mean_waiting_vehicle_count']:.2f})"
            )
            print(
                "Avg request outcomes: "
                f"rejected={sum(row.get('rejected_requests', 0) for row in detailed) / len(detailed):.2f}, "
                f"recourse={sum(row.get('recourse_requests', 0) for row in detailed) / len(detailed):.2f}, "
                f"lost={sum(row.get('lost_requests', 0) for row in detailed) / len(detailed):.2f}"
            )
        if results.get('excel_path'):
            print(f"Stats: {results['excel_path']}")
            manifest_arguments = dict(vars(args))
            manifest_arguments["resolved_distribution_mode"] = zone_distribution_mode
            manifest_path = Path(results["excel_path"]).with_suffix(
                ".manifest.json"
            )
            parquet_paths = getattr(env, "parquet_path", ())
            if isinstance(parquet_paths, (str, Path)):
                parquet_paths = [parquet_paths]
            write_experiment_manifest(
                manifest_path,
                arguments=manifest_arguments,
                results=results,
                data_paths=parquet_paths or (),
                value_functions=(
                    getattr(env, "value_function", None),
                    getattr(env, "value_function_ev", None),
                ),
                checkpoint_paths=tuple(dict.fromkeys(
                    path
                    for value_function in (
                        getattr(env, "value_function", None),
                        getattr(env, "value_function_ev", None),
                    )
                    if value_function is not None
                    for path in getattr(
                        value_function, "checkpoint_artifact_paths", ()
                    )
                )),
            )
            print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
