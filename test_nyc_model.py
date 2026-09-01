"""Legacy solver/inference evaluator (not the ICAPS recourse paper runner).

Test trained models on unseen seeds.
Compare assignment backends explicitly so MCMF results are not mislabeled as
heuristic.  ADP-HEU-HEU additionally uses a checkpoint collected and trained
with heuristic assignment:
    1. ADP-ILP   : load checkpoint + gurobi assignment   (NN + ILP)
    2. ILP       : no checkpoint   + gurobi assignment   (pure ILP)
    3. MCMF-K    : no checkpoint   + MCMF assignment     (pure MCMF with known reject)
    4. ADP-MCMF  : load checkpoint + MCMF assignment     (NN + MCMF)
    5. ADP-MCMF-K: load checkpoint + MCMF assignment     (NN + MCMF + known reject)
    6. MCMF      : no checkpoint   + MCMF assignment     (pure MCMF)
    7. ADP-HEU   : exact-trained checkpoint + heuristic assignment (NN + heuristic)
    8. ADP-HEU-K : exact-trained checkpoint + heuristic + known reject
    9. ADP-HEU-HEU: load heuristic-trained checkpoint + heuristic assignment
   10. ADP-HEU-HEU-K: heuristic-trained checkpoint + heuristic + known reject
   11. HEU       : no checkpoint   + heuristic assignment (pure heuristic)
   12. HEU-K     : no checkpoint   + heuristic assignment (request value * known acceptance probability)
Loops over transportation_mode × demand_pattern × seed × strategy.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime

from src.ADPtrainer import ADPTrainer
from src.NYCtrainer import NYCTrainer
from src.charging_wait_metrics import aggregate_wait_metrics
from src.recourse.types import LEARNER_VARIANTS
from run_nyctrainer import run_nyc_training


# ── Test strategies ───────────────────────────────────────────────
STRATEGIES = [
    {"name": "ADP-ILP",  "adp": 1.0, "gurobi": True,  "usemcmf": False, "load_ckpt": True,  "known_reject": False},
    {"name": "ILP",      "adp": 0.0, "gurobi": True,  "usemcmf": False, "load_ckpt": False, "known_reject": False},
    {"name": "MCMF-K",   "adp": 0.0, "gurobi": True,  "usemcmf": True, "load_ckpt": False, "known_reject": True},
    {"name": "ADP-MCMF", "adp": 1.0, "gurobi": True,  "usemcmf": True,  "load_ckpt": True,  "known_reject": False},
    {"name": "ADP-MCMF-K", "adp": 1.0, "gurobi": True,  "usemcmf": True,  "load_ckpt": True,  "known_reject": True},
    {"name": "ADP-MCMF-FT", "adp": 1.0, "gurobi": True,  "usemcmf": True,  "load_ckpt": True,  "known_reject": False, "train_during_test": True},
    {"name": "MCMF",     "adp": 0.0, "gurobi": True,  "usemcmf": True,  "load_ckpt": False, "known_reject": False},
    {"name": "ADP-AUCTION", "adp": 1.0, "gurobi": True, "usemcmf": False, "useauction": True, "load_ckpt": True, "known_reject": False},
    {"name": "ADP-AUCTION-K", "adp": 1.0, "gurobi": True, "usemcmf": False, "useauction": True, "load_ckpt": True, "known_reject": True},
    {"name": "AUCTION", "adp": 0.0, "gurobi": True, "usemcmf": False, "useauction": True, "load_ckpt": False, "known_reject": False},
    {"name": "ADP-HEU",  "adp": 1.0, "gurobi": False, "usemcmf": False, "load_ckpt": True, "checkpoint_assign_tag": "gurobi", "known_reject": False},
    {"name": "ADP-HEU-K", "adp": 1.0, "gurobi": False, "usemcmf": False, "load_ckpt": True, "checkpoint_assign_tag": "gurobi", "known_reject": True},
    {"name": "ADP-HEU-HEU", "adp": 1.0, "gurobi": False, "usemcmf": False, "load_ckpt": True, "checkpoint_assign_tag": "heu", "known_reject": False},
    {"name": "ADP-HEU-HEU-K", "adp": 1.0, "gurobi": False, "usemcmf": False, "load_ckpt": True, "checkpoint_assign_tag": "heu", "known_reject": True},
    {"name": "HEU",      "adp": 0.0, "gurobi": False, "usemcmf": False, "load_ckpt": False, "known_reject": False},
    {"name": "HEU-K",    "adp": 0.0, "gurobi": False, "usemcmf": False, "load_ckpt": False, "known_reject": True},
]

PRESERVED_OUTPUT_COLUMNS = {
    "avg_wait",
    "waiting_vehicle_count",
    "mean_waiting_vehicle_count",
    "rejected_requests",
    "recourse_requests",
    "lost_requests",
    "mean_rejected_requests",
    "mean_recourse_requests",
    "mean_lost_requests",
}


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Q-network checkpoints across ILP, MCMF, and heuristic backends")
    from src.acceptance_features import add_acceptance_arguments
    add_acceptance_arguments(parser)
    parser.add_argument('--checkpoint-suffix', default='', help='Experiment namespace printed by the training CLI, excluding the auto-added EV predictor hash')
    parser.add_argument("--paper-parameter-preset", action="store_true",
                        help="Apply the paper-aligned EV preset: 3000 EVs, 24h window, 30s epoch; battery/speed/charge parameters are already defined in NYCEnvironment")
    # --- NYC-specific ---
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes per seed")
    parser.add_argument("--num-vehicles", type=int, default=50, help="Total vehicles")
    parser.add_argument("--num-ev", type=int, default=25, help="EV vehicles")
    parser.add_argument("--seeds", type=int, nargs="+", default=[256],
                        help="Random seeds for evaluation (different from training seed)")
    parser.add_argument("--transportation-modes", type=str, nargs="+",
                        default=["integrated", "evfirst", "aevfirst"],
                        choices=["integrated", "evfirst", "aevfirst"],
                        help="Transportation modes to test")
    parser.add_argument("--start-date", type=str, default="2025-12-18", help="Start date for NYC evaluation dataset (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date for NYC evaluation dataset (YYYY-MM-DD); defaults to --start-date")
    parser.add_argument("--parquet-path", type=str, default=None,
                        help="Yellow Taxi parquet path or comma-separated list")
    parser.add_argument("--full-demand", action="store_true",
                        help="Evaluate on Yellow Taxi plus non-pooled HVFHV; default is Yellow-only")
    parser.add_argument("--hvfhv-parquet-path", type=str, default=None,
                        help="HVFHV parquet path or comma-separated list for --full-demand")
    parser.set_defaults(full_demand=False)
    parser.add_argument("--coord-csv", type=str, default=None,
                        help="Deprecated compatibility option; official TLC taxi-zone geometry is used")
    parser.add_argument("--station-csv", type=str, default=None,
                        help="Path to nyc_charging_stations.csv")
    parser.add_argument("--station-capacity-scale", type=float, default=None,
                        help="Multiply NYC charging station capacity by this factor")
    parser.add_argument("--only-manhattan-zones", action="store_true",
                        help="Restrict NYC demand, relocation zones, and charging stations to Manhattan zones")
    parser.set_defaults(only_manhattan_zones=True)
    parser.add_argument("--start-hour", type=float, default=8.0,
                        help="Start hour of simulation window")
    parser.add_argument("--stop-hour", type=float, default=24.0,
                        help="Stop hour of simulation window")
    parser.add_argument("--epoch-length", type=float, default=30.0,
                        help="Epoch length in seconds")
    parser.add_argument(
        "--human-ev-charge-decision-interval-minutes",
        type=float,
        default=120.0,
        help="Minimum Human EV charge-decision interval; SOC <= 0.20 bypasses it.",
    )
    parser.add_argument("--checkpoint-trained-start-hour", type=float, default=0.0,
                        help="Start hour used when the loaded checkpoint was trained")
    parser.add_argument("--checkpoint-trained-stop-hour", type=float, default=24.0,
                        help="Stop hour used when the loaded checkpoint was trained")
    parser.add_argument("--load-model-start-date", type=str, default=None,
                        help="Start date encoded in the checkpoint directory; defaults to evaluation start date")
    parser.add_argument("--load-model-end-date", type=str, default=None,
                        help="End date encoded in the checkpoint directory; defaults to evaluation end date")
    
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["ADP-MCMF","ADP-MCMF-K", "MCMF","MCMF-K", "ADP-HEU", "ADP-HEU-K", "HEU",],
                        choices=["ADP-ILP", "ILP", "MCMF-K", "ILP-K", "ADP-MCMF", "ADP-MCMF-K", "ADP-MCMF-FT", "MCMF", "ADP-AUCTION", "ADP-AUCTION-K", "AUCTION", "ADP-HEU", "ADP-HEU-K", "ADP-HEU-HEU", "ADP-HEU-HEU-K", "HEU", "HEU-K"],
                        help="Strategies to compare")
    parser.add_argument("--auction-use-gpu", action="store_true", help="Use GPU auction solver for AUCTION strategies")
    parser.add_argument("--auction-use-cpu", dest="auction_use_gpu", action="store_false", help="Force CPU auction solver")
    parser.set_defaults(auction_use_gpu=False)
    parser.add_argument("--auction-epsilon", type=float, default=1e-3, help="Auction bidding epsilon")
    parser.add_argument("--auction-max-rounds", type=int, default=None, help="Maximum auction iterations/rounds before falling back to MCMF")
    parser.add_argument("--auction-top-k", type=int, default=None, help="Keep only each vehicle's top-K feasible auction actions before solving")
    parser.add_argument("--mcmf-solver", choices=["exact", "legacy"], default="exact")
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
    parser.add_argument("--heuristic-battery-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--known-reject", action="store_true")
    parser.set_defaults(known_reject=False)
    parser.add_argument("--num-stations", type=int, default=5, help="Number of charging stations")
    parser.add_argument("--grid-size", type=int, default=20, help="Grid size for the environment (NxN)")
    parser.add_argument("--test_episodenumber", type=int, default=1, help="Starting episode number for evaluation (used in checkpoint naming convention)")
    parser.add_argument("--test-steps-per-episode", type=int, default=1920,
                        help="Expected steps for alignment check; actual episode length is derived from start/stop-hour and epoch-length")
    parser.add_argument(
        "--checkpoint-selection",
        choices=["latest", "best-reward", "best-loss"],
        default="best-reward",
        help="Checkpoint tag to load for ADP evaluation",
    )
    parser.add_argument("--load-best-loss", action="store_true", help="Legacy alias for --checkpoint-selection best-loss")
    parser.set_defaults(load_best_loss=False)
    parser.add_argument(
        "--distribution-mode",
        type=str,
        default="optimization_anchored_residual",
        choices=LEARNER_VARIANTS,
        help="Checkpoint learner: MASAC residual or full-Q",
    )
    parser.add_argument("--iftransformer", action="store_true",
                        help="Enable path self-attention before the LSTM path encoder. Default off for old checkpoint compatibility")
    parser.add_argument(
        "--gat-neighbour-number",
        type=int,
        default=0,
        help="Number of nearest feasible action-graph nodes; 0 is the stable baseline",
    )
    parser.add_argument(
        "--post-demand-q-weight",
        type=float,
        default=0.0,
        help="Initial TD-learned demand-head coefficient used when no checkpoint value is available",
    )
    parser.add_argument(
        "--post-demand-head-lr-multiplier",
        type=float,
        default=10.0,
        help="Learning-rate multiplier used when fine-tuning demand-head coefficients",
    )
    parser.add_argument(
        "--masac-target-entropy-ratio",
        type=float,
        default=0.9,
        help="Target entropy as a fraction of log(candidate count) for masac_baseline/standard_masac_gat fine-tuning",
    )
    parser.set_defaults(iftransformer=False)
    parser.add_argument('--allow-online-adaptation', action='store_true',
                        help='Allow ADP-MCMF-FT as a separately labeled online-adaptation run')
    args = parser.parse_args()
    if 'ADP-MCMF-FT' in args.strategies and not args.allow_online_adaptation:
        parser.error('ADP-MCMF-FT trains during test; use --allow-online-adaptation and exclude it from fixed-policy tables')
    return args


def get_distribution_suffix(distribution_mode: str) -> str:
    return NYCTrainer._distribution_suffix(normalize_distribution_mode(distribution_mode))


def normalize_distribution_mode(distribution_mode: str | None) -> str:
    mode = distribution_mode or "optimization_anchored_residual"
    if mode not in LEARNER_VARIANTS:
        raise ValueError(f"unsupported learner {mode!r}; choose one of {LEARNER_VARIANTS}")
    return mode


def _first_present_value(stats: dict, *keys: str):
    for key in keys:
        value = stats.get(key)
        if value is not None:
            return value
    return None


def _derive_completed_aev_orders(stats: dict) -> float:
    completed_orders = float(stats.get("completed_orders", 0) or 0)
    completed_ev_orders = float(stats.get("completed_ev_orders", 0) or 0)
    return max(0.0, completed_orders - completed_ev_orders)


def _derive_assignment_success_rate(stats: dict) -> float:
    total_orders = float(stats.get("total_orders", 0) or 0)
    if total_orders <= 0:
        return 0.0
    accepted_orders = _first_present_value(stats, "accepted_orders")
    if accepted_orders is None:
        accepted_orders = float(stats.get("active_orders", 0) or 0) + float(stats.get("completed_orders", 0) or 0)
    return float(accepted_orders) / total_orders


def _mean_detail_metric(detailed: list[dict], *keys: str, default: float = 0.0, fallback=None) -> float:
    if not detailed:
        return float(default)

    values = []
    for stats in detailed:
        value = _first_present_value(stats, *keys)
        if value is None and fallback is not None:
            value = fallback(stats)
        if value is None:
            value = default
        values.append(value)
    return float(np.mean(values)) if values else float(default)


def _drop_all_zero_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    drop_cols = []
    for column in df.columns:
        if column in PRESERVED_OUTPUT_COLUMNS:
            continue
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        if numeric_values.notna().any() and (numeric_values.fillna(0) == 0).all():
            drop_cols.append(column)
    return df.drop(columns=drop_cols)









def _aggregate_hourly_zone_request_completed_orders(detailed: list[dict]) -> tuple[list[dict], list[dict]]:
    hourly_zone_map = {}
    hourly_zone_rows = []

    for episode_idx, episode_detail in enumerate(detailed, start=1):
        for zone_row in episode_detail.get("hourly_zone_request_completed_orders", []) or []:
            date_value = zone_row.get("request_date")
            hour_value = int(zone_row.get("request_hour", 0))
            zone_id = int(zone_row.get("zone_id", 0))
            generated_requests = int(zone_row.get("generated_requests", 0) or 0)
            completed_requests = int(zone_row.get("completed_requests", 0) or 0)
            hourly_zone_rows.append({
                "episode_number": episode_idx,
                "request_date": date_value,
                "request_hour": hour_value,
                "zone_id": zone_id,
                "generated_requests": generated_requests,
                "completed_requests": completed_requests,
                "completion_ratio": completed_requests / generated_requests if generated_requests > 0 else 0.0,
            })

            bucket = hourly_zone_map.setdefault(
                (date_value, hour_value, zone_id),
                {
                    "request_date": date_value,
                    "request_hour": hour_value,
                    "zone_id": zone_id,
                    "generated_requests": 0,
                    "completed_requests": 0,
                },
            )
            bucket["generated_requests"] += generated_requests
            bucket["completed_requests"] += completed_requests

    aggregated_rows = []
    for key in sorted(hourly_zone_map.keys(), key=lambda item: (item[0] or "", item[1], item[2])):
        bucket = hourly_zone_map[key]
        generated_requests = int(bucket["generated_requests"])
        completed_requests = int(bucket["completed_requests"])
        aggregated_rows.append({
            "request_date": bucket["request_date"],
            "request_hour": bucket["request_hour"],
            "zone_id": bucket["zone_id"],
            "generated_requests": generated_requests,
            "completed_requests": completed_requests,
            "completion_ratio": completed_requests / generated_requests if generated_requests > 0 else 0.0,
        })

    return aggregated_rows, hourly_zone_rows


def _aggregate_daily_zone_request_completion_shares(detailed: list[dict]) -> tuple[list[dict], list[dict]]:
    daily_zone_map = {}
    daily_zone_rows = []

    for episode_idx, episode_detail in enumerate(detailed, start=1):
        for zone_row in episode_detail.get("daily_zone_request_completion_shares", []) or []:
            date_value = zone_row.get("request_date")
            zone_id = int(zone_row.get("zone_id", 0))
            generated_requests = int(zone_row.get("generated_requests", 0) or 0)
            completed_requests = int(zone_row.get("completed_requests", 0) or 0)
            daily_zone_rows.append({
                "episode_number": episode_idx,
                "request_date": date_value,
                "zone_id": zone_id,
                "generated_requests": generated_requests,
                "completed_requests": completed_requests,
                "completion_ratio": completed_requests / generated_requests if generated_requests > 0 else 0.0,
            })

            bucket = daily_zone_map.setdefault(
                (date_value, zone_id),
                {
                    "request_date": date_value,
                    "zone_id": zone_id,
                    "generated_requests": 0,
                    "completed_requests": 0,
                },
            )
            bucket["generated_requests"] += generated_requests
            bucket["completed_requests"] += completed_requests

    aggregated_rows = []
    for key in sorted(daily_zone_map.keys(), key=lambda item: (item[0] or "", item[1])):
        bucket = daily_zone_map[key]
        generated_requests = int(bucket["generated_requests"])
        completed_requests = int(bucket["completed_requests"])
        aggregated_rows.append({
            "request_date": bucket["request_date"],
            "zone_id": bucket["zone_id"],
            "generated_requests": generated_requests,
            "completed_requests": completed_requests,
            "completion_ratio": completed_requests / generated_requests if generated_requests > 0 else 0.0,
        })

    return aggregated_rows, daily_zone_rows


def _aggregate_hourly_zone_charge_station_counts(detailed: list[dict]) -> tuple[list[dict], list[dict]]:
    hourly_zone_map = {}
    hourly_zone_rows = []

    for episode_idx, episode_detail in enumerate(detailed, start=1):
        for zone_row in episode_detail.get("hourly_zone_charge_station_counts", []) or []:
            date_value = zone_row.get("date")
            hour_value = int(zone_row.get("hour", 0))
            zone_id = int(zone_row.get("zone_id", 0))
            snapshot_count = int(zone_row.get("snapshot_count", 0) or 0)
            mean_station_count = float(zone_row.get("mean_station_count", 0.0) or 0.0)
            mean_total_capacity = float(zone_row.get("mean_total_capacity", 0.0) or 0.0)
            mean_queue_vehicle_count = float(zone_row.get("mean_queue_vehicle_count", 0.0) or 0.0)
            mean_queue_to_capacity_ratio = float(zone_row.get("mean_queue_to_capacity_ratio", 0.0) or 0.0)
            hourly_zone_rows.append({
                "episode_number": episode_idx,
                "date": date_value,
                "hour": hour_value,
                "zone_id": zone_id,
                "snapshot_count": snapshot_count,
                "mean_station_count": mean_station_count,
                "mean_total_capacity": mean_total_capacity,
                "mean_queue_vehicle_count": mean_queue_vehicle_count,
                "mean_queue_to_capacity_ratio": mean_queue_to_capacity_ratio,
            })

            weight = float(snapshot_count) if snapshot_count > 0 else 1.0
            bucket = hourly_zone_map.setdefault(
                (date_value, hour_value, zone_id),
                {
                    "date": date_value,
                    "hour": hour_value,
                    "zone_id": zone_id,
                    "snapshot_count": 0,
                    "weighted_station_count": 0.0,
                    "weighted_total_capacity": 0.0,
                    "weighted_queue_vehicle_count": 0.0,
                },
            )
            bucket["snapshot_count"] += snapshot_count
            bucket["weighted_station_count"] += mean_station_count * weight
            bucket["weighted_total_capacity"] += mean_total_capacity * weight
            bucket["weighted_queue_vehicle_count"] += mean_queue_vehicle_count * weight

    aggregated_rows = []
    for key in sorted(hourly_zone_map.keys(), key=lambda item: (item[0] or "", item[1], item[2])):
        bucket = hourly_zone_map[key]
        divisor = float(bucket["snapshot_count"]) if bucket["snapshot_count"] > 0 else 1.0
        mean_station_count = bucket["weighted_station_count"] / divisor
        mean_total_capacity = bucket["weighted_total_capacity"] / divisor
        mean_queue_vehicle_count = bucket["weighted_queue_vehicle_count"] / divisor
        aggregated_rows.append({
            "date": bucket["date"],
            "hour": bucket["hour"],
            "zone_id": bucket["zone_id"],
            "snapshot_count": bucket["snapshot_count"],
            "mean_station_count": mean_station_count,
            "mean_total_capacity": mean_total_capacity,
            "mean_queue_vehicle_count": mean_queue_vehicle_count,
            "mean_queue_to_capacity_ratio": mean_queue_vehicle_count / mean_total_capacity if mean_total_capacity > 0 else 0.0,
        })

    return aggregated_rows, hourly_zone_rows



def _aggregate_hourly_zone_vehicle_counts(detailed: list[dict]) -> tuple[list[dict], list[dict]]:
    hourly_zone_map = {}
    hourly_zone_rows = []

    for episode_idx, episode_detail in enumerate(detailed, start=1):
        for zone_row in episode_detail.get("hourly_zone_vehicle_counts", []) or []:
            date_value = zone_row.get("date")
            hour_value = int(zone_row.get("hour", 0))
            zone_id = int(zone_row.get("zone_id", 0))
            snapshot_count = int(zone_row.get("snapshot_count", 0) or 0)
            mean_total = float(zone_row.get("mean_total_vehicles", 0.0) or 0.0)
            mean_ev = float(zone_row.get("mean_ev_vehicles", 0.0) or 0.0)
            mean_aev = float(zone_row.get("mean_aev_vehicles", 0.0) or 0.0)

            hourly_zone_rows.append({
                "episode_number": episode_idx,
                "date": date_value,
                "hour": hour_value,
                "zone_id": zone_id,
                "snapshot_count": snapshot_count,
                "mean_total_vehicles": mean_total,
                "mean_ev_vehicles": mean_ev,
                "mean_aev_vehicles": mean_aev,
            })

            weight = float(snapshot_count) if snapshot_count > 0 else 1.0
            bucket = hourly_zone_map.setdefault(
                (date_value, hour_value, zone_id),
                {
                    "date": date_value,
                    "hour": hour_value,
                    "zone_id": zone_id,
                    "snapshot_count": 0,
                    "weighted_total_vehicles": 0.0,
                    "weighted_ev_vehicles": 0.0,
                    "weighted_aev_vehicles": 0.0,
                },
            )
            bucket["snapshot_count"] += snapshot_count
            bucket["weighted_total_vehicles"] += mean_total * weight
            bucket["weighted_ev_vehicles"] += mean_ev * weight
            bucket["weighted_aev_vehicles"] += mean_aev * weight

    aggregated_rows = []
    for key in sorted(hourly_zone_map.keys(), key=lambda item: (item[0] or "", item[1], item[2])):
        bucket = hourly_zone_map[key]
        divisor = float(bucket["snapshot_count"]) if bucket["snapshot_count"] > 0 else 1.0
        aggregated_rows.append({
            "date": bucket["date"],
            "hour": bucket["hour"],
            "zone_id": bucket["zone_id"],
            "snapshot_count": bucket["snapshot_count"],
            "mean_total_vehicles": bucket["weighted_total_vehicles"] / divisor,
            "mean_ev_vehicles": bucket["weighted_ev_vehicles"] / divisor,
            "mean_aev_vehicles": bucket["weighted_aev_vehicles"] / divisor,
        })

    return aggregated_rows, hourly_zone_rows




def apply_paper_parameter_preset(args):
    if not args.paper_parameter_preset:
        return args

    default_paper_date = "2025-12-18"
    load_model_start_dataset_date = "2025-12-15"
    load_model_end_dataset_date = "2025-12-17"
    args.num_vehicles = 3000
    args.num_ev = args.num_vehicles//2
    if args.station_csv is None:
        real_station_csv = Path(__file__).resolve().parent / "nyedata" / "nyc_all_charging_stations.csv"
        if real_station_csv.exists():
            args.station_csv = str(real_station_csv)
    if args.station_capacity_scale is None:
        args.station_capacity_scale = 1.0 if args.station_csv else 5.0
    args.start_hour = 0
    args.stop_hour = 24
    args.epoch_length = 30.0
    start_dt = datetime.strptime(args.start_date or default_paper_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date or args.start_date or default_paper_date, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError(f"end-date {end_dt.date()} must be on or after start-date {start_dt.date()}")
    args.start_date = start_dt.strftime("%Y-%m-%d")
    args.end_date = end_dt.strftime("%Y-%m-%d")
    if args.load_model_start_date is None:
        args.load_model_start_date = load_model_start_dataset_date
    if args.load_model_end_date is None:
        args.load_model_end_date = (
            load_model_end_dataset_date
            if args.load_model_start_date == load_model_start_dataset_date
            else args.load_model_start_date
        )
    args.episodes = (end_dt - start_dt).days + 1
    return args


def build_checkpoint_dir(
    assign_tag: str,
    mode: str,
    num_ev: int,
    intense: bool,
    vtype: str,
    start_date: str | None,
    end_date: str | None,
    distribution_mode: str,
    only_manhattan_zones: bool = False,
    full_demand: bool = False,
    checkpoint_suffix: str = '',
) -> str:
    """Checkpoint dir matching NYCTrainer save/load convention."""
    if vtype not in {"ev", "aev"}:
        raise ValueError(f"Unknown vehicle checkpoint type: {vtype}")
    ev_dir, aev_dir = NYCTrainer._checkpoint_dirs(
        transportation_mode=mode,
        assignmentgurobi=(assign_tag == "gurobi"),
        num_ev=num_ev,
        use_intense_requests=intense,
        start_date=start_date,
        end_date=end_date,
        zone_distribution_mode=normalize_distribution_mode(distribution_mode),
        only_manhattan_zones=only_manhattan_zones,
        full_demand=full_demand,
        checkpoint_suffix=checkpoint_suffix,
    )
    return ev_dir if vtype == "ev" else aev_dir


def resolve_dataset_dates(start_date: str, end_date: str | None) -> tuple[str, str, str, str]:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date or start_date, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError(f"end-date {end_dt.date()} must be on or after start-date {start_dt.date()}")
    return (
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
        start_dt.strftime("%Y-%m"),
        end_dt.strftime("%Y-%m"),
    )


def main():
    args = apply_paper_parameter_preset(parse_args())
    zone_distribution_mode = normalize_distribution_mode(args.distribution_mode)
    from src.acceptance_features import acceptance_checkpoint_suffix
    acceptance_lookup_suffix = acceptance_checkpoint_suffix(args.ev_acceptance_feature, args.ev_acceptance_model,
        anchor=args.ev_response_anchor, critic_input=args.ev_response_critic_input)
    intense = True
    expected_steps = max(1, int(((args.stop_hour - args.start_hour) * 3600) / args.epoch_length))
    if args.test_steps_per_episode != expected_steps:
        print(
            f"⚠ test-steps-per-episode={args.test_steps_per_episode} does not match the configured window "
            f"({args.start_hour:.1f}-{args.stop_hour:.1f}, epoch={args.epoch_length:.1f}s => {expected_steps} steps). "
            f"Environment episode length follows the window-derived value."
        )
    dataset_start_date, dataset_end_date, start_year_month, end_year_month = resolve_dataset_dates(
        args.start_date,
        args.end_date,
    )
    checkpoint_start_date, checkpoint_end_date, _, _ = resolve_dataset_dates(
        getattr(args, "load_model_start_date", None) or dataset_start_date,
        getattr(args, "load_model_end_date", None) or getattr(args, "load_model_start_date", None) or dataset_end_date,
    )
    strategy_map = {s["name"]: s for s in STRATEGIES}
    strategy_map["ILP-K"] = strategy_map["MCMF-K"]  # Backward-compatible legacy label.
    selected_strategies = [strategy_map[n] for n in args.strategies]
    checkpoint_selection = args.checkpoint_selection.replace("-", "_")
    if args.load_best_loss:
        checkpoint_selection = "best_loss"
    elif checkpoint_selection not in {"latest", "best_reward", "best_loss"}:
        checkpoint_selection = "best_reward"
    ifload_bestloss = checkpoint_selection == "best_loss"
    print("=" * 80)
    print("Model Evaluation")
    print(f"   Strategies: {[s['name'] for s in selected_strategies]}")
    print(f"   Modes: {args.transportation_modes}")
    print(f"   Seeds: {args.seeds}")
    print(f"   Episodes per config: {args.episodes}")
    print(f"   Vehicles: {args.num_vehicles} (EV={args.num_ev})")
    print(f"   Grid: {args.grid_size}x{args.grid_size}")
    print(f"   Dataset dates: {dataset_start_date} -> {dataset_end_date}")
    print(f"   Checkpoint dates: {checkpoint_start_date} -> {checkpoint_end_date}")
    print(f"   Checkpoint selection: {checkpoint_selection}")
    print(f"   Time window: {args.start_hour:.1f} -> {args.stop_hour:.1f} ({expected_steps} steps @ {args.epoch_length:.1f}s)")
    print(f"   Zone scope: {'Manhattan only' if args.only_manhattan_zones else 'full NYC CSV zones'}")
    print(f"   Demand: {'Yellow + non-pooled HVFHV' if args.full_demand else 'Yellow only'}")
    print(f"   Distribution mode: {zone_distribution_mode}")
    if zone_distribution_mode in LEARNER_VARIANTS:
        print(f"   GAT neighbour number: {args.gat_neighbour_number}")
    if zone_distribution_mode == "optimization_anchored_residual":
        print(f"   Post-demand Q weight: {args.post_demand_q_weight:g}")
    if zone_distribution_mode == "optimization_anchored_residual":
        print(f"   MASAC target entropy ratio: {args.masac_target_entropy_ratio:g}")
    print(f"   iftransformer: {args.iftransformer}")
    print("=" * 80)

    # ── 1. Pre-check: checkpoint availability (only needed for ADP strategies) ──
    need_ckpt = any(s["load_ckpt"] for s in selected_strategies)
    checkpoint_tags = sorted({
        strategy.get("checkpoint_assign_tag", "gurobi")
        for strategy in selected_strategies
        if strategy["load_ckpt"]
    })
    ckpt_available = {}  # (mode, training assignment tag) -> bool
    if need_ckpt:
        checkpoint_preference = checkpoint_selection.replace("_", " ")
        print(
            "\nCheckpoint check "
            f"(training tags = {checkpoint_tags}, prefer {checkpoint_preference}):"
        )
        for mode in args.transportation_modes:
            for checkpoint_tag in checkpoint_tags:
                ok = True
                for vtype in ("aev", "ev"):
                    ckpt_dir = build_checkpoint_dir(
                        checkpoint_tag,
                        mode,
                        args.num_ev,
                        intense,
                        vtype,
                        checkpoint_start_date,
                        checkpoint_end_date,
                        zone_distribution_mode,
                        args.only_manhattan_zones,
                        args.full_demand,
                        checkpoint_suffix=(args.checkpoint_suffix + acceptance_lookup_suffix),
                    )
                    latest = ADPTrainer.find_latest_checkpoint(
                        ckpt_dir,
                        prefer_best=checkpoint_selection == "best_reward",
                        prefer_best_loss=ifload_bestloss,
                    )
                    status = latest if latest else "NOT FOUND"
                    print(f"   [{mode}/{checkpoint_tag}/{vtype}] {ckpt_dir} -> {status}")
                    if latest is None:
                        ok = False
                ckpt_available[(mode, checkpoint_tag)] = ok
        missing_count = sum(1 for v in ckpt_available.values() if not v)
        if missing_count:
            print(f"\n⚠  {missing_count} config(s) missing checkpoint — ADP strategies will be skipped for those.")
    print()

    # ── 2. Run evaluation ──
    all_results = []
    demand_baselines = {}

    for mode in args.transportation_modes:
        for strat in selected_strategies:
            checkpoint_assign_tag = strat.get("checkpoint_assign_tag", "gurobi")
            # Skip ADP strategies if checkpoint not available
            if strat["load_ckpt"] and not ckpt_available.get(
                (mode, checkpoint_assign_tag),
                False,
            ):
                print(
                    f"⏭  Skipping {strat['name']} for {mode}: "
                    f"{checkpoint_assign_tag} checkpoint not found"
                )
                continue

            for seed in args.seeds:
                print(f"\n{'─' * 70}")
                print(f"▶ {strat['name']}  mode={mode}  seed={seed}")
                print(f"{'─' * 70}")

                fine_tune = bool(strat.get("train_during_test", False))
                results, env = run_nyc_training(
                    adpvalue=strat["adp"],
                    num_episodes=args.episodes,
                    use_intense_requests=intense,
                    assignmentgurobi=strat["gurobi"],
                    batch_size=args.batch_size,
                    num_vehicles=args.num_vehicles,
                    num_ev=args.num_ev,
                    heuristic_battery_threshold=args.heuristic_battery_threshold,
                    transportation_mode=mode,
                    start_training_episode=0 if fine_tune else 999,
                    usemcmf=strat["usemcmf"],
                    knownreject=strat.get("known_reject", args.known_reject),
                    mcmf_use_gpu=False,
                    mcmf_solver=args.mcmf_solver,
                    mcmf_backend=args.mcmf_backend,
                    mcmf_strict=args.mcmf_strict,
                    mcmf_cost_scale=args.mcmf_cost_scale,
                    mcmf_graph_reduction=args.mcmf_graph_reduction,
                    mcmf_verify=args.mcmf_verify,
                    useauction=strat.get("useauction", False),
                    auction_use_gpu=args.auction_use_gpu,
                    auction_epsilon=args.auction_epsilon,
                    auction_max_rounds=args.auction_max_rounds,
                    auction_top_k=args.auction_top_k,
                    ifloadcheckpoint=strat["load_ckpt"],
                    trainnetwork=fine_tune,
                    prestep=0 if fine_tune else 1440,
                    random_seed=seed,
                    parquet_path=args.parquet_path,
                    full_demand=args.full_demand,
                    hvfhv_parquet_path=args.hvfhv_parquet_path,
                    start_year_month=start_year_month,
                    end_year_month=end_year_month,
                    start_date=dataset_start_date,
                    end_date=dataset_end_date,
                    coord_csv=args.coord_csv,
                    station_csv=args.station_csv,
                    station_capacity_scale=args.station_capacity_scale,
                    start_hour=args.start_hour,
                    stop_hour=args.stop_hour,
                    epoch_length=args.epoch_length,
                    zone_distribution_mode=zone_distribution_mode,
                    learner_variant=zone_distribution_mode,
                    ev_acceptance_feature=args.ev_acceptance_feature,
                    ev_acceptance_model=args.ev_acceptance_model,
                ev_response_anchor=args.ev_response_anchor,
                ev_response_critic_input=args.ev_response_critic_input,
                    checkpoint_suffix=args.checkpoint_suffix,
                    only_manhattan_zones=args.only_manhattan_zones,
                    human_ev_charge_decision_interval_minutes=(
                        args.human_ev_charge_decision_interval_minutes
                    ),
                    load_checkpoint_assign_tag=(
                        checkpoint_assign_tag if strat["load_ckpt"] else None
                    ),
                    load_best_loss=ifload_bestloss if strat["load_ckpt"] else False,
                    checkpoint_selection=checkpoint_selection if strat["load_ckpt"] else "latest",
                    load_checkpoint_start_date=checkpoint_start_date if strat["load_ckpt"] else None,
                    load_checkpoint_end_date=checkpoint_end_date if strat["load_ckpt"] else None,
                    checkpoint_trained_start_hour=args.checkpoint_trained_start_hour if strat["load_ckpt"] else None,
                    checkpoint_trained_stop_hour=args.checkpoint_trained_stop_hour if strat["load_ckpt"] else None,
                    iftransformer=args.iftransformer,
                    gat_neighbour_number=args.gat_neighbour_number,
                    post_demand_q_weight=args.post_demand_q_weight,
                    post_demand_head_lr_multiplier=args.post_demand_head_lr_multiplier,
                    masac_target_entropy_ratio=args.masac_target_entropy_ratio,
                )

                rewards = results.get("episode_rewards", [])
                avg_reward = np.mean(rewards) if rewards else 0.0

                detailed = results.get("episode_detailed_stats", [])
                demand_signature = tuple(
                    (
                        str(d.get("current_real_date")),
                        int(d.get("whole_req_num", d.get("total_orders", 0)) or 0),
                    )
                    for d in detailed
                )
                demand_key = (mode, seed)
                baseline = demand_baselines.setdefault(demand_key, demand_signature)
                if demand_signature != baseline:
                    raise RuntimeError(
                        "Evaluation demand mismatch across strategies for "
                        f"mode={mode}, seed={seed}: expected {baseline}, got {demand_signature} "
                        f"for {strat['name']}"
                    )
                total_accept = sum(d.get("accepted_orders", 0) for d in detailed)
                total_reject = sum(d.get("rejected_orders", 0) for d in detailed)
                total_rejected_requests = sum(d.get("rejected_requests", d.get("rejected_orders", 0)) for d in detailed)
                total_recourse_requests = sum(d.get("recourse_requests", 0) for d in detailed)
                total_lost_requests = sum(d.get("lost_requests", 0) for d in detailed)
                total_complete = sum(d.get("completed_orders", 0) for d in detailed)
                total_orders = sum(d.get("whole_req_num", d.get("total_orders", 0)) for d in detailed)
                mean_ev_completed_orders = _mean_detail_metric(detailed, "completed_ev_orders")
                mean_aev_completed_orders = _mean_detail_metric(detailed, "completed_aev_orders", fallback=_derive_completed_aev_orders)
                mean_assignment_success_rate = _mean_detail_metric(detailed, "assignment_success_rate", fallback=_derive_assignment_success_rate)
                mean_completed_order_value = _mean_detail_metric(detailed, "avg_completed_order_value", "avg_request_value")
                mean_ev_completed_order_value = _mean_detail_metric(
                    detailed,
                    "avg_ev_completed_order_value",
                    fallback=lambda stats: _first_present_value(stats, "avg_completed_order_value", "avg_request_value")
                    if float(stats.get("completed_ev_orders", 0) or 0) > 0
                    else 0.0,
                )
                mean_aev_completed_order_value = _mean_detail_metric(
                    detailed,
                    "avg_aev_completed_order_value",
                    fallback=lambda stats: _first_present_value(stats, "avg_completed_order_value", "avg_request_value")
                    if _derive_completed_aev_orders(stats) > 0
                    else 0.0,
                )
                mean_sample_assign_q = _mean_detail_metric(detailed, "sample_assign_q_value")
                mean_sample_charge_q = _mean_detail_metric(detailed, "sample_charge_q_value")
                mean_sample_idle_q = _mean_detail_metric(detailed, "sample_idle_q_value")
                mean_sample_assign_q_aev = _mean_detail_metric(detailed, "sample_assign_q_value_aev")
                mean_sample_assign_q_ev = _mean_detail_metric(detailed, "sample_assign_q_value_ev")
                mean_drop_off_rate = _mean_detail_metric(detailed, "drop_off_rate")
                mean_service_ratio = _mean_detail_metric(detailed, "service_ratio")
                mean_max_station_pressure = _mean_detail_metric(detailed, "max_station_pressure")
                mean_station_pressure = _mean_detail_metric(detailed, "mean_station_pressure")
                mean_max_station_pressure_ratio = _mean_detail_metric(detailed, "max_station_pressure_ratio")
                mean_station_pressure_ratio = _mean_detail_metric(detailed, "mean_station_pressure_ratio")
                wait_summary = aggregate_wait_metrics(detailed)
                mean_avg_wait = wait_summary["avg_wait"]
                total_waiting_vehicle_count = wait_summary[
                    "waiting_vehicle_count"
                ]
                mean_waiting_vehicle_count = wait_summary[
                    "mean_waiting_vehicle_count"
                ]
                mean_episode_reward_aev = _mean_detail_metric(
                    detailed,
                    "episode_aev_reward",
                    "episode_reward_aev",
                )
                mean_episode_reward_ev = _mean_detail_metric(
                    detailed,
                    "episode_ev_reward",
                    "episode_reward_ev",
                )
                mean_avg_battery_level = _mean_detail_metric(detailed, "avg_battery_level", "avg_battery")
                mean_finished_charge = _mean_detail_metric(detailed, "finished_charge", "charge_finished")
                mean_daily_charges_human_ev = _mean_detail_metric(
                    detailed,
                    "avg_daily_charging_sessions_per_human_ev",
                )
                mean_daily_charges_aev = _mean_detail_metric(
                    detailed,
                    "avg_daily_charging_sessions_per_aev",
                )
                mean_daily_charges_all = _mean_detail_metric(
                    detailed,
                    "avg_daily_charging_sessions_per_vehicle",
                )
                mean_charge_duration_human_ev = _mean_detail_metric(
                    detailed,
                    "avg_charging_session_duration_minutes_human_ev",
                )
                mean_charge_duration_aev = _mean_detail_metric(
                    detailed,
                    "avg_charging_session_duration_minutes_aev",
                )
                mean_charge_duration_all = _mean_detail_metric(
                    detailed,
                    "avg_charging_session_duration_minutes_all",
                )
                hourly_zone_request_completed_orders, hourly_zone_request_completed_rows = _aggregate_hourly_zone_request_completed_orders(detailed)
                daily_zone_request_completion_shares, daily_zone_request_completion_rows = _aggregate_daily_zone_request_completion_shares(detailed)
                hourly_zone_charge_station_counts, hourly_zone_charge_station_rows = _aggregate_hourly_zone_charge_station_counts(detailed)
                hourly_zone_vehicle_counts, hourly_zone_vehicle_rows = _aggregate_hourly_zone_vehicle_counts(detailed)
                hourly_completed_map = {}
                hourly_completed_rows = []
                for episode_idx, episode_detail in enumerate(detailed, start=1):
                    for hourly_row in episode_detail.get("hourly_completed_orders", []) or []:
                        completed_date = hourly_row.get("completed_date")
                        completed_hour = int(hourly_row.get("completed_hour", 0))
                        completed_orders = int(hourly_row.get("completed_orders", 0))
                        completed_ev_orders = int(hourly_row.get("completed_ev_orders", 0))
                        completed_aev_orders = int(hourly_row.get("completed_aev_orders", 0))
                        hourly_completed_rows.append({
                            "strategy": strat["name"],
                            "mode": mode,
                            "seed": seed,
                            "episode_number": episode_idx,
                            "completed_date": completed_date,
                            "completed_hour": completed_hour,
                            "completed_orders": completed_orders,
                            "completed_ev_orders": completed_ev_orders,
                            "completed_aev_orders": completed_aev_orders,
                        })
                        key = (completed_date, completed_hour)
                        bucket = hourly_completed_map.setdefault(
                            key,
                            {
                                "completed_date": completed_date,
                                "completed_hour": completed_hour,
                                "completed_orders": 0,
                                "completed_ev_orders": 0,
                                "completed_aev_orders": 0,
                            },
                        )
                        bucket["completed_orders"] += completed_orders
                        bucket["completed_ev_orders"] += completed_ev_orders
                        bucket["completed_aev_orders"] += completed_aev_orders
                hourly_completed_orders = [
                    hourly_completed_map[key]
                    for key in sorted(hourly_completed_map.keys(), key=lambda item: (item[0] or "", item[1]))
                ]
                if hourly_completed_orders:
                    peak_hour_row = max(hourly_completed_orders, key=lambda row: row["completed_orders"])
                    peak_completed_date = peak_hour_row["completed_date"]
                    peak_completed_hour = peak_hour_row["completed_hour"]
                    peak_completed_orders = peak_hour_row["completed_orders"]
                else:
                    peak_completed_date = None
                    peak_completed_hour = None
                    peak_completed_orders = 0

                entry = {
                    "strategy": strat["name"],
                    "mode": mode,
                    "seed": seed,
                    "avg_reward": avg_reward,
                    "total_reward": sum(rewards),
                    "episodes": len(rewards),
                    "total_orders": total_orders,
                    "accept": total_accept,
                    "reject": total_reject,
                    "rejected_requests": total_rejected_requests,
                    "recourse_requests": total_recourse_requests,
                    "lost_requests": total_lost_requests,
                    "complete": total_complete,
                    "mean_ev_completed_orders": mean_ev_completed_orders,
                    "mean_aev_completed_orders": mean_aev_completed_orders,
                    "mean_assignment_success_rate": mean_assignment_success_rate,
                    "mean_completed_order_value": mean_completed_order_value,
                    "mean_ev_completed_order_value": mean_ev_completed_order_value,
                    "mean_aev_completed_order_value": mean_aev_completed_order_value,
                    "mean_sample_assign_q_value": mean_sample_assign_q,
                    "mean_sample_assign_q_value_aev": mean_sample_assign_q_aev,
                    "mean_sample_assign_q_value_ev": mean_sample_assign_q_ev,
                    "mean_sample_charge_q_value": mean_sample_charge_q,
                    "mean_sample_idle_q_value": mean_sample_idle_q,
                    "mean_drop_off_rate": mean_drop_off_rate,
                    "mean_service_ratio": mean_service_ratio,
                    "max_station_pressure": mean_max_station_pressure,
                    "mean_max_station_pressure": mean_max_station_pressure,
                    "mean_station_pressure": mean_station_pressure,
                    "mean_max_station_pressure_ratio": mean_max_station_pressure_ratio,
                    "mean_station_pressure_ratio": mean_station_pressure_ratio,
                    "avg_wait": mean_avg_wait,
                    "waiting_vehicle_count": total_waiting_vehicle_count,
                    "mean_waiting_vehicle_count": mean_waiting_vehicle_count,
                    "episode_reward_aev": mean_episode_reward_aev,
                    "episode_reward_ev": mean_episode_reward_ev,
                    "episode_aev_reward": mean_episode_reward_aev,
                    "episode_ev_reward": mean_episode_reward_ev,
                    "avg_battery_level": mean_avg_battery_level,
                    "finished_charge": mean_finished_charge,
                    "avg_daily_charging_sessions_per_human_ev": mean_daily_charges_human_ev,
                    "avg_daily_charging_sessions_per_aev": mean_daily_charges_aev,
                    "avg_daily_charging_sessions_per_vehicle": mean_daily_charges_all,
                    "avg_charging_session_duration_minutes_human_ev": mean_charge_duration_human_ev,
                    "avg_charging_session_duration_minutes_aev": mean_charge_duration_aev,
                    "avg_charging_session_duration_minutes_all": mean_charge_duration_all,
                    "hourly_zone_request_completed_orders": hourly_zone_request_completed_orders,
                    "hourly_zone_request_completed_orders_json": _json_dumps(hourly_zone_request_completed_orders),
                    "daily_zone_request_completion_shares": daily_zone_request_completion_shares,
                    "daily_zone_request_completion_shares_json": _json_dumps(daily_zone_request_completion_shares),
                    "hourly_zone_charge_station_counts": hourly_zone_charge_station_counts,
                    "hourly_zone_charge_station_counts_json": _json_dumps(hourly_zone_charge_station_counts),
                    "hourly_zone_vehicle_counts": hourly_zone_vehicle_counts,
                    "hourly_zone_vehicle_counts_json": _json_dumps(hourly_zone_vehicle_counts),
                    "hourly_completed_orders": hourly_completed_orders,
                    "hourly_completed_orders_json": _json_dumps(hourly_completed_orders),
                    "peak_completed_date": peak_completed_date,
                    "peak_completed_hour": peak_completed_hour,
                    "peak_completed_orders": peak_completed_orders,
                }
                all_results.append(entry)

                print(f"   Avg reward: {avg_reward:.2f}  "
                        f"Accept: {entry['accept']}  Reject: {entry['reject']}  "
                        f"RejectedReq: {entry['rejected_requests']}  RecourseReq: {entry['recourse_requests']}  LostReq: {entry['lost_requests']}  "
                        f"Complete: {entry['complete']}  "
                        f"EV complete(avg): {mean_ev_completed_orders:.2f}  "
                        f"AEV complete(avg): {mean_aev_completed_orders:.2f}  "
                        f"AvgWait: {mean_avg_wait:.2f}  "
                        f"WaitVeh: {mean_waiting_vehicle_count:.1f}  "
                        f"Charge/veh-day: {mean_daily_charges_all:.3f}  "
                        f"ChargeTime: {mean_charge_duration_all:.2f} min  "
                        f"DropOff: {mean_drop_off_rate:.4f}  "
                        f"MaxPressure: {mean_max_station_pressure:.2f}")

    # ── 3. Summary table ──
    print("\n" + "=" * 120)
    print("Evaluation Summary")
    print("=" * 120)
    header = f"{'Strategy':<12} {'Mode':<12} {'Seed':<8} {'AvgReward':>10} {'Orders':>8} {'Accept':>8} {'Reject':>8} {'Complete':>10} {'AccRate':>8} {'SvcRate':>8} {'AvgWait':>9} {'DropOff':>8}"
    print(header)
    print("-" * 120)
    for r in all_results:
        acc_rate = r['accept'] / r['total_orders'] * 100 if r['total_orders'] > 0 else 0
        service_rate = r.get('mean_service_ratio', 0.0) * 100.0
        print(f"{r['strategy']:<12} {r['mode']:<12} {r['seed']:<8} "
              f"{r['avg_reward']:>10.2f} {r['total_orders']:>8} {r['accept']:>8} {r['reject']:>8} {r['complete']:>10} {acc_rate:>7.1f}% {service_rate:>7.1f}% {r['avg_wait']:>9.2f} {r['mean_drop_off_rate']:>8.4f}")

    # Aggregate per (strategy, mode)
    print("\n" + "-" * 120)
    print(f"{'Strategy':<12} {'Mode':<12} {'MeanReward':>12} {'StdReward':>12} {'SvcRate':>10} {'AvgWait':>10} {'DropOff':>10} {'Seeds':>6}")
    print("-" * 120)
    seen = {}
    for r in all_results:
        key = (r["strategy"], r["mode"])
        seen.setdefault(key, []).append(r["avg_reward"])
    for (s, m), rews in seen.items():
        mean_drop_off = np.mean([r['mean_drop_off_rate'] for r in all_results if r['strategy'] == s and r['mode'] == m])
        mean_service_ratio = np.mean([r['mean_service_ratio'] for r in all_results if r['strategy'] == s and r['mode'] == m]) * 100.0
        wait_subset = [r for r in all_results if r['strategy'] == s and r['mode'] == m]
        mean_avg_wait = aggregate_wait_metrics(wait_subset)['avg_wait']
        print(f"{s:<12} {m:<12} {np.mean(rews):>12.2f} {np.std(rews):>12.2f} {mean_service_ratio:>9.1f}% {mean_avg_wait:>10.2f} {mean_drop_off:>10.4f} {len(rews):>6}")

    # Save raw results
    out_dir = Path("results/test_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    demand_tag = "_fulldemand" if args.full_demand else ""
    distribution_tag = f"_{zone_distribution_mode}{demand_tag}"
    out_path = out_dir / f"test_results_4way{distribution_tag}.npy"
    np.save(out_path, all_results)
    print(f"\n✓ Raw results saved to {out_path}")

    # ── 4. Save Excel with two sheets: detail + summary ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = out_dir / f"test_results_4way{distribution_tag}_{timestamp}.xlsx"

    # Detail sheet — one row per (strategy, mode, seed)
    df_detail = pd.DataFrame(all_results)
    df_detail["acc_rate"] = df_detail.apply(
        lambda r: r["accept"] / r["total_orders"] * 100 if r["total_orders"] > 0 else 0, axis=1
    )
    df_detail["mean_service_ratio_pct"] = df_detail["mean_service_ratio"] * 100.0
    df_detail["mean_assignment_success_rate_pct"] = df_detail["mean_assignment_success_rate"] * 100.0

    hourly_detail_rows = []
    daily_zone_request_completion_detail_rows = []
    hourly_zone_request_completed_detail_rows = []
    hourly_zone_charge_station_detail_rows = []
    hourly_zone_detail_rows = []
    for result in all_results:
        for zone_row in result.get("daily_zone_request_completion_shares", []) or []:
            daily_zone_request_completion_detail_rows.append({
                "strategy": result["strategy"],
                "mode": result["mode"],
                "seed": result["seed"],
                "request_date": zone_row.get("request_date"),
                "zone_id": zone_row.get("zone_id"),
                "generated_requests": zone_row.get("generated_requests", 0),
                "completed_requests": zone_row.get("completed_requests", 0),
                "completion_ratio": zone_row.get("completion_ratio", 0.0),
            })
        for zone_row in result.get("hourly_zone_request_completed_orders", []) or []:
            hourly_zone_request_completed_detail_rows.append({
                "strategy": result["strategy"],
                "mode": result["mode"],
                "seed": result["seed"],
                "request_date": zone_row.get("request_date"),
                "request_hour": zone_row.get("request_hour"),
                "zone_id": zone_row.get("zone_id"),
                "generated_requests": zone_row.get("generated_requests", 0),
                "completed_requests": zone_row.get("completed_requests", 0),
                "completion_ratio": zone_row.get("completion_ratio", 0.0),
            })
        for zone_row in result.get("hourly_zone_charge_station_counts", []) or []:
            hourly_zone_charge_station_detail_rows.append({
                "strategy": result["strategy"],
                "mode": result["mode"],
                "seed": result["seed"],
                "date": zone_row.get("date"),
                "hour": zone_row.get("hour"),
                "zone_id": zone_row.get("zone_id"),
                "snapshot_count": zone_row.get("snapshot_count", 0),
                "mean_station_count": zone_row.get("mean_station_count", 0.0),
                "mean_total_capacity": zone_row.get("mean_total_capacity", 0.0),
                "mean_queue_vehicle_count": zone_row.get("mean_queue_vehicle_count", 0.0),
                "mean_queue_to_capacity_ratio": zone_row.get("mean_queue_to_capacity_ratio", 0.0),
            })
        for zone_row in result.get("hourly_zone_vehicle_counts", []) or []:
            hourly_zone_detail_rows.append({
                "strategy": result["strategy"],
                "mode": result["mode"],
                "seed": result["seed"],
                "date": zone_row.get("date"),
                "hour": zone_row.get("hour"),
                "zone_id": zone_row.get("zone_id"),
                "snapshot_count": zone_row.get("snapshot_count", 0),
                "mean_total_vehicles": zone_row.get("mean_total_vehicles", 0.0),
                "mean_ev_vehicles": zone_row.get("mean_ev_vehicles", 0.0),
                "mean_aev_vehicles": zone_row.get("mean_aev_vehicles", 0.0),
            })
        for hourly_row in result.get("hourly_completed_orders", []) or []:
            hourly_detail_rows.append({
                "strategy": result["strategy"],
                "mode": result["mode"],
                "seed": result["seed"],
                "completed_date": hourly_row.get("completed_date"),
                "completed_hour": hourly_row.get("completed_hour"),
                "completed_orders": hourly_row.get("completed_orders", 0),
                "completed_ev_orders": hourly_row.get("completed_ev_orders", 0),
                "completed_aev_orders": hourly_row.get("completed_aev_orders", 0),
            })

    # Summary sheet — aggregate per (strategy, mode)
    summary_rows = []
    hourly_summary_rows = []
    daily_zone_request_completion_summary_rows = []
    hourly_zone_request_completed_summary_rows = []
    hourly_zone_charge_station_summary_rows = []
    hourly_zone_summary_rows = []
    for (s, m), rews in seen.items():
        subset = [r for r in all_results if r["strategy"] == s and r["mode"] == m]
        wait_summary = aggregate_wait_metrics(subset)
        total_episodes = sum(int(r.get("episodes", 0) or 0) for r in subset)
        daily_zone_request_completion_summary_map = {}
        for result in subset:
            for zone_row in result.get("daily_zone_request_completion_shares", []) or []:
                key = (zone_row.get("request_date"), int(zone_row.get("zone_id", 0)))
                bucket = daily_zone_request_completion_summary_map.setdefault(
                    key,
                    {
                        "strategy": s,
                        "mode": m,
                        "request_date": zone_row.get("request_date"),
                        "zone_id": int(zone_row.get("zone_id", 0)),
                        "generated_requests": 0.0,
                        "completed_requests": 0.0,
                    },
                )
                bucket["generated_requests"] += float(zone_row.get("generated_requests", 0) or 0)
                bucket["completed_requests"] += float(zone_row.get("completed_requests", 0) or 0)
        mean_daily_zone_request_completion_shares = []
        if daily_zone_request_completion_summary_map:
            for bucket in daily_zone_request_completion_summary_map.values():
                bucket["generated_requests"] /= len(subset)
                bucket["completed_requests"] /= len(subset)
                bucket["completion_ratio"] = (
                    bucket["completed_requests"] / bucket["generated_requests"]
                    if bucket["generated_requests"] > 0
                    else 0.0
                )
            mean_daily_zone_request_completion_shares = [
                daily_zone_request_completion_summary_map[key]
                for key in sorted(daily_zone_request_completion_summary_map.keys(), key=lambda item: (item[0] or "", item[1]))
            ]
            daily_zone_request_completion_summary_rows.extend(mean_daily_zone_request_completion_shares)
        hourly_zone_request_completed_summary_map = {}
        for result in subset:
            for zone_row in result.get("hourly_zone_request_completed_orders", []) or []:
                key = (
                    zone_row.get("request_date"),
                    int(zone_row.get("request_hour", 0)),
                    int(zone_row.get("zone_id", 0)),
                )
                bucket = hourly_zone_request_completed_summary_map.setdefault(
                    key,
                    {
                        "strategy": s,
                        "mode": m,
                        "request_date": zone_row.get("request_date"),
                        "request_hour": int(zone_row.get("request_hour", 0)),
                        "zone_id": int(zone_row.get("zone_id", 0)),
                        "generated_requests": 0.0,
                        "completed_requests": 0.0,
                    },
                )
                bucket["generated_requests"] += float(zone_row.get("generated_requests", 0) or 0)
                bucket["completed_requests"] += float(zone_row.get("completed_requests", 0) or 0)
        mean_hourly_zone_request_completed_orders = []
        if hourly_zone_request_completed_summary_map:
            for bucket in hourly_zone_request_completed_summary_map.values():
                bucket["generated_requests"] /= len(subset)
                bucket["completed_requests"] /= len(subset)
                bucket["completion_ratio"] = (
                    bucket["completed_requests"] / bucket["generated_requests"]
                    if bucket["generated_requests"] > 0
                    else 0.0
                )
            mean_hourly_zone_request_completed_orders = [
                hourly_zone_request_completed_summary_map[key]
                for key in sorted(hourly_zone_request_completed_summary_map.keys(), key=lambda item: (item[0] or "", item[1], item[2]))
            ]
            hourly_zone_request_completed_summary_rows.extend(mean_hourly_zone_request_completed_orders)
        hourly_zone_charge_station_summary_map = {}
        for result in subset:
            for zone_row in result.get("hourly_zone_charge_station_counts", []) or []:
                key = (
                    zone_row.get("date"),
                    int(zone_row.get("hour", 0)),
                    int(zone_row.get("zone_id", 0)),
                )
                bucket = hourly_zone_charge_station_summary_map.setdefault(
                    key,
                    {
                        "strategy": s,
                        "mode": m,
                        "date": zone_row.get("date"),
                        "hour": int(zone_row.get("hour", 0)),
                        "zone_id": int(zone_row.get("zone_id", 0)),
                        "mean_station_count": 0.0,
                        "mean_total_capacity": 0.0,
                        "mean_queue_vehicle_count": 0.0,
                    },
                )
                bucket["mean_station_count"] += float(zone_row.get("mean_station_count", 0.0) or 0.0)
                bucket["mean_total_capacity"] += float(zone_row.get("mean_total_capacity", 0.0) or 0.0)
                bucket["mean_queue_vehicle_count"] += float(zone_row.get("mean_queue_vehicle_count", 0.0) or 0.0)
        mean_hourly_zone_charge_station_counts = []
        if hourly_zone_charge_station_summary_map:
            for bucket in hourly_zone_charge_station_summary_map.values():
                bucket["mean_station_count"] /= len(subset)
                bucket["mean_total_capacity"] /= len(subset)
                bucket["mean_queue_vehicle_count"] /= len(subset)
                bucket["mean_queue_to_capacity_ratio"] = (
                    bucket["mean_queue_vehicle_count"] / bucket["mean_total_capacity"]
                    if bucket["mean_total_capacity"] > 0
                    else 0.0
                )
            mean_hourly_zone_charge_station_counts = [
                hourly_zone_charge_station_summary_map[key]
                for key in sorted(hourly_zone_charge_station_summary_map.keys(), key=lambda item: (item[0] or "", item[1], item[2]))
            ]
            hourly_zone_charge_station_summary_rows.extend(mean_hourly_zone_charge_station_counts)
        hourly_zone_summary_map = {}
        for result in subset:
            for zone_row in result.get("hourly_zone_vehicle_counts", []) or []:
                key = (
                    zone_row.get("date"),
                    int(zone_row.get("hour", 0)),
                    int(zone_row.get("zone_id", 0)),
                )
                bucket = hourly_zone_summary_map.setdefault(
                    key,
                    {
                        "strategy": s,
                        "mode": m,
                        "date": zone_row.get("date"),
                        "hour": int(zone_row.get("hour", 0)),
                        "zone_id": int(zone_row.get("zone_id", 0)),
                        "mean_total_vehicles": 0.0,
                        "mean_ev_vehicles": 0.0,
                        "mean_aev_vehicles": 0.0,
                    },
                )
                bucket["mean_total_vehicles"] += float(zone_row.get("mean_total_vehicles", 0.0) or 0.0)
                bucket["mean_ev_vehicles"] += float(zone_row.get("mean_ev_vehicles", 0.0) or 0.0)
                bucket["mean_aev_vehicles"] += float(zone_row.get("mean_aev_vehicles", 0.0) or 0.0)
        mean_hourly_zone_vehicle_counts = []
        if hourly_zone_summary_map:
            for bucket in hourly_zone_summary_map.values():
                bucket["mean_total_vehicles"] /= len(subset)
                bucket["mean_ev_vehicles"] /= len(subset)
                bucket["mean_aev_vehicles"] /= len(subset)
            mean_hourly_zone_vehicle_counts = [
                hourly_zone_summary_map[key]
                for key in sorted(hourly_zone_summary_map.keys(), key=lambda item: (item[0] or "", item[1], item[2]))
            ]
            hourly_zone_summary_rows.extend(mean_hourly_zone_vehicle_counts)
        hourly_summary_map = {}
        for result in subset:
            for hourly_row in result.get("hourly_completed_orders", []) or []:
                key = (hourly_row.get("completed_date"), int(hourly_row.get("completed_hour", 0)))
                bucket = hourly_summary_map.setdefault(
                    key,
                    {
                        "strategy": s,
                        "mode": m,
                        "completed_date": hourly_row.get("completed_date"),
                        "completed_hour": int(hourly_row.get("completed_hour", 0)),
                        "mean_completed_orders": 0.0,
                        "mean_completed_ev_orders": 0.0,
                        "mean_completed_aev_orders": 0.0,
                    },
                )
                bucket["mean_completed_orders"] += float(hourly_row.get("completed_orders", 0))
                bucket["mean_completed_ev_orders"] += float(hourly_row.get("completed_ev_orders", 0))
                bucket["mean_completed_aev_orders"] += float(hourly_row.get("completed_aev_orders", 0))
        if hourly_summary_map:
            for bucket in hourly_summary_map.values():
                bucket["mean_completed_orders"] /= len(subset)
                bucket["mean_completed_ev_orders"] /= len(subset)
                bucket["mean_completed_aev_orders"] /= len(subset)
            mean_hourly_completed_orders = [
                hourly_summary_map[key]
                for key in sorted(hourly_summary_map.keys(), key=lambda item: (item[0] or "", item[1]))
            ]
            peak_hour_row = max(mean_hourly_completed_orders, key=lambda row: row["mean_completed_orders"])
            peak_completed_date = peak_hour_row["completed_date"]
            peak_completed_hour = peak_hour_row["completed_hour"]
            peak_completed_orders = peak_hour_row["mean_completed_orders"]
            hourly_summary_rows.extend(mean_hourly_completed_orders)
        else:
            mean_hourly_completed_orders = []
            peak_completed_date = None
            peak_completed_hour = None
            peak_completed_orders = 0.0
        summary_rows.append({
            "strategy": s,
            "mode": m,
            "mean_reward": np.mean(rews),
            "std_reward": np.std(rews),
            "mean_accept": np.mean([r["accept"] for r in subset]),
            "mean_reject": np.mean([r["reject"] for r in subset]),
            "mean_rejected_requests": np.mean([r["rejected_requests"] for r in subset]),
            "mean_recourse_requests": np.mean([r["recourse_requests"] for r in subset]),
            "mean_lost_requests": np.mean([r["lost_requests"] for r in subset]),
            "mean_complete": np.mean([r["complete"] for r in subset]),
            "mean_orders": np.mean([r["total_orders"] for r in subset]),
            "mean_acc_rate": np.mean([r["accept"] / r["total_orders"] * 100 if r["total_orders"] > 0 else 0 for r in subset]),
            "mean_ev_completed_orders": np.mean([r["mean_ev_completed_orders"] for r in subset]),
            "mean_aev_completed_orders": np.mean([r["mean_aev_completed_orders"] for r in subset]),
            "mean_assignment_success_rate": np.mean([r["mean_assignment_success_rate"] for r in subset]),
            "mean_assignment_success_rate_pct": np.mean([r["mean_assignment_success_rate"] * 100.0 for r in subset]),
            "mean_completed_order_value": np.mean([r["mean_completed_order_value"] for r in subset]),
            "mean_ev_completed_order_value": np.mean([r["mean_ev_completed_order_value"] for r in subset]),
            "mean_aev_completed_order_value": np.mean([r["mean_aev_completed_order_value"] for r in subset]),
            "mean_sample_assign_q_value": np.mean([r["mean_sample_assign_q_value"] for r in subset]),
            "mean_sample_assign_q_value_aev": np.mean([r["mean_sample_assign_q_value_aev"] for r in subset]),
            "mean_sample_assign_q_value_ev": np.mean([r["mean_sample_assign_q_value_ev"] for r in subset]),
            "mean_sample_charge_q_value": np.mean([r["mean_sample_charge_q_value"] for r in subset]),
            "mean_sample_idle_q_value": np.mean([r["mean_sample_idle_q_value"] for r in subset]),
            "mean_drop_off_rate": np.mean([r["mean_drop_off_rate"] for r in subset]),
            "mean_service_ratio": np.mean([r["mean_service_ratio"] for r in subset]),
            "mean_service_ratio_pct": np.mean([r["mean_service_ratio"] * 100.0 for r in subset]),
            "max_station_pressure": np.mean([r["max_station_pressure"] for r in subset]),
            "mean_max_station_pressure": np.mean([r["mean_max_station_pressure"] for r in subset]),
            "mean_station_pressure": np.mean([r["mean_station_pressure"] for r in subset]),
            "mean_max_station_pressure_ratio": np.mean([r["mean_max_station_pressure_ratio"] for r in subset]),
            "mean_station_pressure_ratio": np.mean([r["mean_station_pressure_ratio"] for r in subset]),
            "avg_wait": wait_summary["avg_wait"],
            "waiting_vehicle_count": wait_summary["waiting_vehicle_count"],
            "mean_waiting_vehicle_count": (
                wait_summary["waiting_vehicle_count"] / total_episodes
                if total_episodes else 0.0
            ),
            "episode_reward_aev": np.mean([r["episode_reward_aev"] for r in subset]),
            "episode_reward_ev": np.mean([r["episode_reward_ev"] for r in subset]),
            "episode_aev_reward": np.mean([r["episode_aev_reward"] for r in subset]),
            "episode_ev_reward": np.mean([r["episode_ev_reward"] for r in subset]),
            "avg_battery_level": np.mean([r["avg_battery_level"] for r in subset]),
            "finished_charge": np.mean([r["finished_charge"] for r in subset]),
            "avg_daily_charging_sessions_per_human_ev": np.mean([r["avg_daily_charging_sessions_per_human_ev"] for r in subset]),
            "avg_daily_charging_sessions_per_aev": np.mean([r["avg_daily_charging_sessions_per_aev"] for r in subset]),
            "avg_daily_charging_sessions_per_vehicle": np.mean([r["avg_daily_charging_sessions_per_vehicle"] for r in subset]),
            "avg_charging_session_duration_minutes_human_ev": np.mean([r["avg_charging_session_duration_minutes_human_ev"] for r in subset]),
            "avg_charging_session_duration_minutes_aev": np.mean([r["avg_charging_session_duration_minutes_aev"] for r in subset]),
            "avg_charging_session_duration_minutes_all": np.mean([r["avg_charging_session_duration_minutes_all"] for r in subset]),
            "peak_completed_date": peak_completed_date,
            "peak_completed_hour": peak_completed_hour,
            "peak_completed_orders": peak_completed_orders,
            "mean_daily_zone_request_completion_shares_json": _json_dumps(mean_daily_zone_request_completion_shares),
            "mean_hourly_zone_request_completed_orders_json": _json_dumps(mean_hourly_zone_request_completed_orders),
            "mean_hourly_zone_charge_station_counts_json": _json_dumps(mean_hourly_zone_charge_station_counts),
            "mean_hourly_zone_vehicle_counts_json": _json_dumps(mean_hourly_zone_vehicle_counts),
            "mean_hourly_completed_orders_json": _json_dumps(mean_hourly_completed_orders),
            "num_seeds": len(rews),
        })
    df_summary = pd.DataFrame(summary_rows)

    df_detail = _drop_all_zero_numeric_columns(df_detail)
    df_summary = _drop_all_zero_numeric_columns(df_summary)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_detail.to_excel(writer, sheet_name="detail", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        if daily_zone_request_completion_detail_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(daily_zone_request_completion_detail_rows)).to_excel(writer, sheet_name="daily_req_completion_detail", index=False)
        if daily_zone_request_completion_summary_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(daily_zone_request_completion_summary_rows)).to_excel(writer, sheet_name="daily_req_completion_summary", index=False)
        if hourly_zone_request_completed_detail_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(hourly_zone_request_completed_detail_rows)).to_excel(writer, sheet_name="hourly_req_completion_detail", index=False)
        if hourly_zone_request_completed_summary_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(hourly_zone_request_completed_summary_rows)).to_excel(writer, sheet_name="hourly_req_completion_summary", index=False)
        if hourly_zone_charge_station_detail_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(hourly_zone_charge_station_detail_rows)).to_excel(writer, sheet_name="hourly_station_queue_detail", index=False)
        if hourly_zone_charge_station_summary_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(hourly_zone_charge_station_summary_rows)).to_excel(writer, sheet_name="hourly_station_queue_summary", index=False)
        if hourly_zone_detail_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(hourly_zone_detail_rows)).to_excel(writer, sheet_name="hourly_zone_vehicle_detail", index=False)
        if hourly_zone_summary_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(hourly_zone_summary_rows)).to_excel(writer, sheet_name="hourly_zone_vehicle_summary", index=False)
        if hourly_detail_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(hourly_detail_rows)).to_excel(writer, sheet_name="hourly_completed_detail", index=False)
        if hourly_summary_rows:
            _drop_all_zero_numeric_columns(pd.DataFrame(hourly_summary_rows)).to_excel(writer, sheet_name="hourly_completed_summary", index=False)
    print(f"✓ Excel saved to {excel_path}")


if __name__ == "__main__":
    main()
