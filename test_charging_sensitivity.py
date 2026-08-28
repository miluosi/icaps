"""NYC driving-energy sensitivity analysis without value-function learning.

The experiment keeps demand, initial vehicle states, and all behavioral random
draws fixed across consumption ratios.  Assignment uses the structured MCMF
baseline (ADP weight zero); no checkpoint is loaded and no network is trained.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_nyctrainer import run_nyc_training
from src.NYCEnvironment import DEFAULT_INITIAL_BATTERY_MEAN
from src.charging_wait_metrics import aggregate_wait_metrics


# The inner points preserve the requested local perturbations.  The 0.80 and
# 1.20 endpoints represent plausible efficient-versus-adverse operating
# conditions and make the response visible without using unrealistic extremes.
DEFAULT_CONSUMPTION_RATIOS = (0.80, 0.95**2, 0.95, 1.0, 1.05, 1.05**2, 1.20)
BASE_CONSUMPTION_WH_PER_MILE = 230.0
TREND_METRICS = {
    "avg_daily_charging_sessions_per_human_ev":
        "Human EV charging sessions per vehicle-day",
    "avg_daily_charging_sessions_per_vehicle":
        "All-fleet charging sessions per vehicle-day",
    "avg_charging_session_duration_minutes_human_ev":
        "Human EV charging-session duration (minutes)",
    "avg_wait_minutes_waiting_charging_vehicles":
        "Positive charger-queue wait (minutes)",
    "mean_completed_orders": "Completed orders",
    "mean_service_ratio": "Service ratio",
}


def validate_consumption_ratios(values: Iterable[float]) -> tuple[float, ...]:
    ratios = tuple(float(value) for value in values)
    if not ratios:
        raise ValueError("at least one battery-consumption ratio is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
        raise ValueError("battery-consumption ratios must be finite and positive")
    if len(set(ratios)) != len(ratios):
        raise ValueError("battery-consumption ratios must be unique")
    return tuple(sorted(ratios))


def _weighted_mean(rows: list[dict], value_key: str, count_key: str) -> float:
    total_count = sum(max(0.0, float(row.get(count_key, 0.0) or 0.0)) for row in rows)
    if total_count <= 0.0:
        return 0.0
    return sum(
        float(row.get(value_key, 0.0) or 0.0)
        * max(0.0, float(row.get(count_key, 0.0) or 0.0))
        for row in rows
    ) / total_count


def _per_vehicle_day(
    rows: list[dict],
    session_key: str,
    vehicle_count_key: str,
) -> float:
    sessions = sum(float(row.get(session_key, 0.0) or 0.0) for row in rows)
    vehicle_days = sum(
        float(row.get(vehicle_count_key, 0.0) or 0.0)
        * float(row.get("charging_observation_days", 0.0) or 0.0)
        for row in rows
    )
    return sessions / vehicle_days if vehicle_days > 0.0 else 0.0


def summarize_run(
    detailed_stats: list[dict],
    *,
    ratio: float,
    seed: int,
    epoch_length_sec: float,
) -> dict:
    if not detailed_stats:
        raise RuntimeError("NYC sensitivity run returned no episode statistics")

    wait = aggregate_wait_metrics(detailed_stats)
    total_observation_days = sum(
        float(row.get("charging_observation_days", 0.0) or 0.0)
        for row in detailed_stats
    )
    human_ev_count = int(detailed_stats[0].get("human_ev_vehicle_count", 0) or 0)
    aev_count = int(detailed_stats[0].get("aev_vehicle_count", 0) or 0)
    all_vehicle_count = int(detailed_stats[0].get("all_vehicle_count", 0) or 0)

    def selected_action_count(fleet_type: str, action_name: str) -> int:
        return int(sum(
            int(
                row.get("mcmf_action_counts_by_type", {})
                .get(fleet_type, {})
                .get(action_name, 0)
            )
            for row in detailed_stats
        ))

    def selected_action_count_at_low_soc(action_name: str) -> int:
        low_soc_bins = ("soc_le_0p15", "soc_0p15_0p20")
        return int(sum(
            int(
                row.get("mcmf_action_counts_by_type_soc_bin", {})
                .get("aev", {})
                .get(soc_bin, {})
                .get(action_name, 0)
            )
            for row in detailed_stats
            for soc_bin in low_soc_bins
        ))

    return {
        "battery_consumption_ratio": float(ratio),
        "seed": int(seed),
        "episodes": len(detailed_stats),
        "observation_days": total_observation_days,
        "human_ev_vehicle_count": human_ev_count,
        "aev_vehicle_count": aev_count,
        "all_vehicle_count": all_vehicle_count,
        "charging_station_count": int(
            detailed_stats[0].get("charging_station_count", 0) or 0
        ),
        "charging_total_capacity": int(
            detailed_stats[0].get("charging_total_capacity", 0) or 0
        ),
        "charging_station_csv": str(
            detailed_stats[0].get("charging_station_csv", "") or ""
        ),
        "base_consumption_wh_per_mile": BASE_CONSUMPTION_WH_PER_MILE,
        "effective_consumption_wh_per_mile": BASE_CONSUMPTION_WH_PER_MILE * ratio,
        "effective_consumption_kwh_per_km": (
            BASE_CONSUMPTION_WH_PER_MILE * ratio / 1000.0 / 1.609344
        ),
        "configured_initial_battery_mean": float(
            detailed_stats[0].get(
                "configured_initial_battery_mean",
                DEFAULT_INITIAL_BATTERY_MEAN,
            )
        ),
        "human_ev_charge_decision_interval_minutes": float(
            detailed_stats[0].get(
                "human_ev_charge_decision_interval_minutes",
                10.0,
            )
        ),
        "realized_initial_battery_mean": float(
            np.mean([
                row.get("realized_initial_battery_mean", DEFAULT_INITIAL_BATTERY_MEAN)
                for row in detailed_stats
            ])
        ),
        "realized_initial_battery_mean_human_ev": float(
            np.mean([
                row.get(
                    "realized_initial_battery_mean_human_ev",
                    DEFAULT_INITIAL_BATTERY_MEAN,
                )
                for row in detailed_stats
            ])
        ),
        "realized_initial_battery_mean_aev": float(
            np.mean([
                row.get(
                    "realized_initial_battery_mean_aev",
                    DEFAULT_INITIAL_BATTERY_MEAN,
                )
                for row in detailed_stats
            ])
        ),
        "avg_daily_charging_sessions_per_human_ev": _per_vehicle_day(
            detailed_stats,
            "human_ev_charging_sessions",
            "human_ev_vehicle_count",
        ),
        "avg_daily_charging_sessions_per_aev": _per_vehicle_day(
            detailed_stats,
            "aev_charging_sessions",
            "aev_vehicle_count",
        ),
        "avg_daily_charging_sessions_per_vehicle": _per_vehicle_day(
            detailed_stats,
            "all_vehicle_charging_sessions",
            "all_vehicle_count",
        ),
        "avg_charging_session_duration_minutes_human_ev": _weighted_mean(
            detailed_stats,
            "avg_charging_session_duration_minutes_human_ev",
            "completed_charging_sessions_with_duration_human_ev",
        ),
        "avg_charging_session_duration_minutes_aev": _weighted_mean(
            detailed_stats,
            "avg_charging_session_duration_minutes_aev",
            "completed_charging_sessions_with_duration_aev",
        ),
        "avg_charging_session_duration_minutes_all": _weighted_mean(
            detailed_stats,
            "avg_charging_session_duration_minutes_all",
            "completed_charging_sessions_with_duration_all",
        ),
        "avg_wait_steps_waiting_charging_vehicles": float(wait["avg_wait"]),
        "avg_wait_minutes_waiting_charging_vehicles": (
            float(wait["avg_wait"]) * float(epoch_length_sec) / 60.0
        ),
        "waiting_charging_vehicle_count": float(wait["waiting_vehicle_count"]),
        "mean_completed_orders": float(
            np.mean([row.get("completed_orders", 0.0) for row in detailed_stats])
        ),
        "avg_reward": float(
            np.mean([row.get("episode_reward", 0.0) for row in detailed_stats])
        ),
        "mean_service_ratio": float(
            np.mean([row.get("service_ratio", 0.0) for row in detailed_stats])
        ),
        "mean_final_battery_soc": float(
            np.mean(
                [
                    row.get("avg_battery_level", row.get("avg_battery", 0.0))
                    for row in detailed_stats
                ]
            )
        ),
        "mean_final_battery_soc_human_ev": float(np.mean([
            row.get("final_battery_mean_human_ev", 0.0)
            for row in detailed_stats
        ])),
        "mean_final_battery_soc_aev": float(np.mean([
            row.get("final_battery_mean_aev", 0.0)
            for row in detailed_stats
        ])),
        "total_distance_km_human_ev": float(sum(
            row.get("total_distance_km_human_ev", 0.0)
            for row in detailed_stats
        )),
        "total_distance_km_aev": float(sum(
            row.get("total_distance_km_aev", 0.0)
            for row in detailed_stats
        )),
        "mcmf_aev_request_selections": selected_action_count("aev", "request"),
        "mcmf_aev_charge_selections": selected_action_count("aev", "charge"),
        "mcmf_aev_wait_selections": selected_action_count("aev", "wait"),
        "mcmf_aev_relocate_selections": selected_action_count("aev", "relocate"),
        "mcmf_low_soc_aev_charge_selections": selected_action_count_at_low_soc("charge"),
        "mcmf_low_soc_aev_wait_selections": selected_action_count_at_low_soc("wait"),
    }


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    interval_column = "human_ev_charge_decision_interval_minutes"
    if interval_column in detail and detail[interval_column].nunique() > 1:
        raise ValueError("Do not pool different Human EV decision intervals in one ratio summary")
    metric_columns = [
        "avg_daily_charging_sessions_per_human_ev",
        "avg_daily_charging_sessions_per_aev",
        "avg_daily_charging_sessions_per_vehicle",
        "avg_charging_session_duration_minutes_human_ev",
        "avg_charging_session_duration_minutes_aev",
        "avg_charging_session_duration_minutes_all",
        "avg_wait_steps_waiting_charging_vehicles",
        "avg_wait_minutes_waiting_charging_vehicles",
        "waiting_charging_vehicle_count",
        "mean_completed_orders",
        "avg_reward",
        "mean_service_ratio",
        "mean_final_battery_soc",
        "mean_final_battery_soc_human_ev",
        "mean_final_battery_soc_aev",
        "total_distance_km_human_ev",
        "total_distance_km_aev",
        "mcmf_aev_request_selections",
        "mcmf_aev_charge_selections",
        "mcmf_aev_wait_selections",
        "mcmf_aev_relocate_selections",
        "mcmf_low_soc_aev_charge_selections",
        "mcmf_low_soc_aev_wait_selections",
    ]
    rows = []
    for ratio, group in detail.groupby("battery_consumption_ratio", sort=True):
        configured_initial_mean = (
            float(group["configured_initial_battery_mean"].iloc[0])
            if "configured_initial_battery_mean" in group
            else DEFAULT_INITIAL_BATTERY_MEAN
        )
        row = {
            "configured_initial_battery_mean": configured_initial_mean,
            "human_ev_charge_decision_interval_minutes": (
                float(group[interval_column].iloc[0])
                if interval_column in group else None
            ),
            "realized_initial_battery_mean": float(
                group["realized_initial_battery_mean"].mean()
                if "realized_initial_battery_mean" in group
                else configured_initial_mean
            ),
            "realized_initial_battery_mean_human_ev": float(
                group["realized_initial_battery_mean_human_ev"].mean()
                if "realized_initial_battery_mean_human_ev" in group
                else configured_initial_mean
            ),
            "realized_initial_battery_mean_aev": float(
                group["realized_initial_battery_mean_aev"].mean()
                if "realized_initial_battery_mean_aev" in group
                else configured_initial_mean
            ),
            "battery_consumption_ratio": float(ratio),
            "effective_consumption_wh_per_mile": float(
                group["effective_consumption_wh_per_mile"].iloc[0]
            ),
            "effective_consumption_kwh_per_km": float(
                group["effective_consumption_kwh_per_km"].iloc[0]
            ),
            "num_seeds": int(group["seed"].nunique()),
            "charging_station_count": int(
                group["charging_station_count"].iloc[0]
                if "charging_station_count" in group
                else 0
            ),
            "charging_total_capacity": int(
                group["charging_total_capacity"].iloc[0]
                if "charging_total_capacity" in group
                else 0
            ),
            "charging_station_csv": str(
                group["charging_station_csv"].iloc[0]
                if "charging_station_csv" in group
                else ""
            ),
        }
        for column in metric_columns:
            if column in group:
                row[column] = float(group[column].mean())
                row[f"std_{column}"] = float(group[column].std(ddof=0))
            else:
                row[column] = 0.0
                row[f"std_{column}"] = 0.0
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("battery_consumption_ratio")
    baseline = summary[np.isclose(summary["battery_consumption_ratio"], 1.0)]
    if not baseline.empty:
        baseline_row = baseline.iloc[0]
        for column in (
            "avg_daily_charging_sessions_per_human_ev",
            "avg_daily_charging_sessions_per_aev",
            "avg_daily_charging_sessions_per_vehicle",
            "avg_wait_minutes_waiting_charging_vehicles",
        ):
            base_value = float(baseline_row[column])
            change_column = f"pct_change_vs_baseline_{column}"
            if abs(base_value) > 1e-12:
                summary[change_column] = (summary[column] / base_value - 1.0) * 100.0
            else:
                summary[change_column] = 0.0
    return summary


def build_trend_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Quantify the direction and visibility of each seven-point response."""
    if len(summary) < 2:
        raise ValueError("at least two consumption ratios are required for trends")

    x = summary["battery_consumption_ratio"].to_numpy(dtype=float)
    rows = []
    for column, label in TREND_METRICS.items():
        y = summary[column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        total_variation = float(np.square(y - y.mean()).sum())
        residual_variation = float(np.square(y - fitted).sum())
        r_squared = (
            1.0 - residual_variation / total_variation
            if total_variation > 1e-12
            else 0.0
        )
        low_value = float(y[0])
        high_value = float(y[-1])
        rows.append({
            "metric": column,
            "display_name": label,
            "low_ratio": float(x[0]),
            "high_ratio": float(x[-1]),
            "low_value": low_value,
            "high_value": high_value,
            "endpoint_change_percent": (
                (high_value / low_value - 1.0) * 100.0
                if abs(low_value) > 1e-12
                else 0.0
            ),
            "linear_slope_per_unit_ratio": float(slope),
            "linear_r_squared": float(r_squared),
        })
    return pd.DataFrame(rows)


def save_plot(summary: pd.DataFrame, output_path: Path) -> None:
    x = summary["battery_consumption_ratio"].to_numpy(dtype=float)
    fig, (count_axis, time_axis) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    count_axis.errorbar(
        x,
        summary["avg_daily_charging_sessions_per_human_ev"],
        yerr=summary["std_avg_daily_charging_sessions_per_human_ev"],
        marker="o",
        linewidth=2.0,
        capsize=3,
        label="Human EV charging sessions/vehicle-day",
    )
    count_axis.errorbar(
        x,
        summary["avg_daily_charging_sessions_per_aev"],
        yerr=summary["std_avg_daily_charging_sessions_per_aev"],
        marker="s",
        linewidth=2.0,
        capsize=3,
        label="AEV charging sessions/vehicle-day",
    )
    count_axis.set_xlabel("Driving-energy consumption ratio")
    count_axis.set_ylabel("Average charging sessions per vehicle-day")
    count_axis.set_title("Charging frequency")
    count_axis.grid(alpha=0.25)
    count_axis.legend(loc="best")

    duration_handle = time_axis.errorbar(
        x,
        summary["avg_charging_session_duration_minutes_human_ev"],
        yerr=summary["std_avg_charging_session_duration_minutes_human_ev"],
        color="tab:blue",
        marker="o",
        linewidth=2.0,
        capsize=3,
        label="Average charging duration",
    )
    wait_axis = time_axis.twinx()
    wait_handle = wait_axis.errorbar(
        x,
        summary["avg_wait_minutes_waiting_charging_vehicles"],
        yerr=summary["std_avg_wait_minutes_waiting_charging_vehicles"],
        color="tab:red",
        marker="^",
        linestyle="--",
        linewidth=2.0,
        capsize=3,
        label="Average charger wait (minutes)",
    )
    time_axis.set_xlabel("Driving-energy consumption ratio")
    time_axis.set_ylabel("Average charging duration (minutes)", color="tab:blue")
    wait_axis.set_ylabel(
        "Average wait among waiting charging vehicles (minutes)", color="tab:red"
    )
    time_axis.set_title("Charging duration and queue wait")
    time_axis.grid(alpha=0.25)

    time_axis.legend(
        [duration_handle, wait_handle],
        ["Average charging duration", "Average charger wait"],
        loc="best",
    )
    fig.suptitle("NYC 200-vehicle battery-consumption sensitivity: exact MCMF")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="NYC 200-vehicle charging sensitivity without learning"
    )
    parser.add_argument(
        "--consumption-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_CONSUMPTION_RATIOS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[256])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--num-vehicles", type=int, default=200)
    parser.add_argument("--num-ev", type=int, default=100)
    parser.add_argument(
        "--initial-battery-mean",
        type=float,
        default=DEFAULT_INITIAL_BATTERY_MEAN,
        help=(
            "Requested mean initial SOC. Vehicles are sampled symmetrically "
            "around this mean; the legacy default is 0.875."
        ),
    )
    parser.add_argument("--start-date", default="2025-12-18")
    parser.add_argument("--end-date", default="2025-12-18")
    parser.add_argument("--start-hour", type=float, default=8.0)
    parser.add_argument("--stop-hour", type=float, default=10.0)
    parser.add_argument("--epoch-length", type=float, default=30.0)
    parser.add_argument(
        "--parquet-path",
        default=str(
            project_root
            / "nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet"
        ),
    )
    parser.add_argument(
        "--station-csv",
        default=None,
        help=(
            "Charging-station CSV. Omit to use the five-zone fallback network, "
            "which produces observable congestion at the 200-vehicle scale."
        ),
    )
    parser.add_argument("--station-capacity-scale", type=float, default=1.0)
    parser.add_argument(
        "--mcmf-backend",
        choices=["gurobi_network", "primal_dual", "ortools", "auto"],
        default="gurobi_network",
    )
    parser.add_argument(
        "--charge-wait-bool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reserve reachable current charger capacity for the lowest-SOC "
            "AEV rows and disable their wait actions; use --no-charge-wait-bool "
            "for the legacy all-one wait column."
        ),
    )
    parser.add_argument(
        "--human-ev-charge-decision-interval-minutes",
        type=float,
        default=120.0,
        help="Minimum interval between Human EV stochastic charge decisions.",
    )
    parser.add_argument("--heuristic-battery-threshold", type=float, default=0.5)
    parser.add_argument("--only-manhattan-zones", action="store_true", default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results/charging_sensitivity",
    )
    return parser.parse_args()


def run_sensitivity(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ratios = validate_consumption_ratios(args.consumption_ratios)
    decision_interval_minutes = float(getattr(
        args, "human_ev_charge_decision_interval_minutes", 120.0
    ))
    if args.num_vehicles != 200:
        raise ValueError("this sensitivity experiment is fixed to 200 vehicles")
    if args.num_ev <= 0 or args.num_ev >= args.num_vehicles:
        raise ValueError("num-ev must leave both human EV and AEV vehicles")
    if args.stop_hour <= args.start_hour:
        raise ValueError("stop-hour must be greater than start-hour")
    if args.epoch_length <= 0.0:
        raise ValueError("epoch-length must be positive")

    rows = []
    for ratio in ratios:
        for seed in args.seeds:
            print(
                "\n"
                + "=" * 88
                + f"\nNYC charging sensitivity: ratio={ratio:.4f}, "
                f"effective={BASE_CONSUMPTION_WH_PER_MILE * ratio:.3f} Wh/mile, "
                f"seed={seed}\n"
                + "=" * 88,
                flush=True,
            )
            results, _ = run_nyc_training(
                adpvalue=0.0,
                num_episodes=args.episodes,
                use_intense_requests=True,
                assignmentgurobi=True,
                batch_size=256,
                num_vehicles=args.num_vehicles,
                num_ev=args.num_ev,
                heuristic_battery_threshold=args.heuristic_battery_threshold,
                transportation_mode="integrated",
                start_training_episode=999,
                usemcmf=True,
                knownreject=False,
                mcmf_use_gpu=False,
                mcmf_solver="exact",
                mcmf_backend=args.mcmf_backend,
                mcmf_strict=True,
                mcmf_cost_scale=10_000,
                mcmf_graph_reduction=True,
                mcmf_verify=False,
                useauction=False,
                auction_use_gpu=False,
                auction_epsilon=1e-3,
                auction_max_rounds=None,
                auction_top_k=None,
                ifloadcheckpoint=False,
                trainnetwork=False,
                random_seed=seed,
                parquet_path=str(Path(args.parquet_path).expanduser().resolve()),
                start_year_month="2025-12",
                end_year_month="2025-12",
                start_date=args.start_date,
                end_date=args.end_date,
                coord_csv=None,
                station_csv=args.station_csv,
                station_capacity_scale=args.station_capacity_scale,
                start_hour=args.start_hour,
                stop_hour=args.stop_hour,
                epoch_length=args.epoch_length,
                zone_distribution_mode="none",
                daily_drop_off=False,
                ifreject=False,
                ifdropoff=False,
                only_manhattan_zones=args.only_manhattan_zones,
                common_random_numbers=True,
                battery_consumption_ratio=ratio,
                initial_battery_mean=args.initial_battery_mean,
                charge_wait_bool=args.charge_wait_bool,
                human_ev_charge_decision_interval_minutes=(
                    decision_interval_minutes
                ),
            )
            row = summarize_run(
                results.get("episode_detailed_stats", []),
                ratio=ratio,
                seed=seed,
                epoch_length_sec=args.epoch_length,
            )
            rows.append(row)
            checkpoint_dir = Path(args.output_dir).expanduser().resolve()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values(
                ["battery_consumption_ratio", "seed"]
            ).to_csv(
                checkpoint_dir
                / "nyc_200_vehicle_mcmf_completed_checkpoint_detail.csv",
                index=False,
            )
            print(
                "Sensitivity result: "
                f"human EV={row['avg_daily_charging_sessions_per_human_ev']:.4f}, "
                f"AEV={row['avg_daily_charging_sessions_per_aev']:.4f}, "
                f"all={row['avg_daily_charging_sessions_per_vehicle']:.4f} "
                "sessions/vehicle-day, "
                f"wait={row['avg_wait_minutes_waiting_charging_vehicles']:.2f} min",
                flush=True,
            )

    detail = pd.DataFrame(rows).sort_values(
        ["battery_consumption_ratio", "seed"]
    )
    summary = build_summary(detail)
    metadata = {
        "experiment": "NYC driving-energy charging sensitivity",
        "learning_enabled": False,
        "assignment_method": "structured exact MCMF",
        "mcmf_backend": args.mcmf_backend,
        "common_random_numbers": True,
        "consumption_ratios": list(ratios),
        "base_consumption_wh_per_mile": BASE_CONSUMPTION_WH_PER_MILE,
        "configured_initial_battery_mean": args.initial_battery_mean,
        "charge_wait_bool": bool(args.charge_wait_bool),
        "human_ev_charge_decision_interval_minutes": float(
            decision_interval_minutes
        ),
        "num_vehicles": args.num_vehicles,
        "num_human_ev": args.num_ev,
        "num_aev": args.num_vehicles - args.num_ev,
        "seeds": list(args.seeds),
        "episodes_per_configuration": args.episodes,
        "date_range": [args.start_date, args.end_date],
        "hour_window": [args.start_hour, args.stop_hour],
        "epoch_length_seconds": args.epoch_length,
        "parquet_path": str(Path(args.parquet_path).expanduser().resolve()),
        "station_csv": args.station_csv,
        "station_capacity_scale": args.station_capacity_scale,
        "average_wait_definition": (
            "vehicle-weighted positive charger-queue wait, including "
            "right-censored vehicles still waiting at episode end"
        ),
    }
    return detail, summary, metadata


def save_results(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"nyc_200_vehicle_mcmf_battery_consumption_sensitivity_{timestamp}"
    paths = {
        "excel": output_dir / f"{stem}.xlsx",
        "detail_csv": output_dir / f"{stem}_detail.csv",
        "summary_csv": output_dir / f"{stem}_summary.csv",
        "trend_csv": output_dir / f"{stem}_trend.csv",
        "json": output_dir / f"{stem}.json",
        "plot": output_dir / f"{stem}.png",
        "latest_excel": output_dir
        / "nyc_200_vehicle_mcmf_battery_consumption_sensitivity_latest.xlsx",
        "latest_summary_csv": output_dir
        / "nyc_200_vehicle_mcmf_battery_consumption_sensitivity_latest.csv",
        "latest_plot": output_dir
        / "nyc_200_vehicle_mcmf_battery_consumption_sensitivity_latest.png",
    }

    metadata_frame = pd.DataFrame(
        {
            "parameter": list(metadata.keys()),
            "value": [
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for value in metadata.values()
            ],
        }
    )
    trend = build_trend_summary(summary)
    for excel_path in (paths["excel"], paths["latest_excel"]):
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            detail.to_excel(writer, sheet_name="detail", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            trend.to_excel(writer, sheet_name="trend", index=False)
            metadata_frame.to_excel(writer, sheet_name="metadata", index=False)
    detail.to_csv(paths["detail_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    trend.to_csv(paths["trend_csv"], index=False)
    summary.to_csv(paths["latest_summary_csv"], index=False)
    paths["json"].write_text(
        json.dumps(
            {
                "metadata": metadata,
                "detail": detail.to_dict(orient="records"),
                "summary": summary.to_dict(orient="records"),
                "trend": trend.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_plot(summary, paths["plot"])
    save_plot(summary, paths["latest_plot"])
    return paths


def main() -> None:
    args = parse_args()
    detail, summary, metadata = run_sensitivity(args)
    paths = save_results(detail, summary, metadata, args.output_dir)
    display_columns = [
        "battery_consumption_ratio",
        "effective_consumption_wh_per_mile",
        "avg_daily_charging_sessions_per_human_ev",
        "avg_daily_charging_sessions_per_aev",
        "avg_daily_charging_sessions_per_vehicle",
        "avg_wait_minutes_waiting_charging_vehicles",
    ]
    print("\nNYC charging sensitivity summary")
    print(summary[display_columns].to_string(index=False))
    print("\nSaved outputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
