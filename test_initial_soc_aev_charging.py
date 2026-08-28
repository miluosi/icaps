"""Check whether lower initial SOC induces AEV charging under exact MCMF.

This diagnostic changes only the initial battery distribution.  It keeps the
assignment method at exact MCMF, disables learning, and fixes the consumption
ratio at its calibrated baseline value of 1.0.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from src.NYCEnvironment import initial_battery_bounds
from test_charging_sensitivity import run_sensitivity


DEFAULT_INITIAL_BATTERY_MEANS = (0.875, 0.45, 0.30, 0.15)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="NYC 200-vehicle initial-SOC AEV charging diagnostic"
    )
    parser.add_argument(
        "--initial-battery-means",
        type=float,
        nargs="+",
        default=list(DEFAULT_INITIAL_BATTERY_MEANS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[256])
    parser.add_argument("--start-date", default="2025-12-18")
    parser.add_argument("--end-date", default="2025-12-18")
    parser.add_argument("--start-hour", type=float, default=0.0)
    parser.add_argument("--stop-hour", type=float, default=24.0)
    parser.add_argument("--epoch-length", type=float, default=120.0)
    parser.add_argument(
        "--mcmf-backend",
        choices=["gurobi_network", "primal_dual", "ortools", "auto"],
        default="gurobi_network",
    )
    parser.add_argument(
        "--charge-wait-bool",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--parquet-path",
        default=(
            "/Users/seinzhou/Desktop/adp_trainer/nyedata/nye_simulation/"
            "parquet/yellow_tripdata_2025-12.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results/initial_soc_aev_charging",
    )
    return parser.parse_args()


def _sensitivity_args(args: argparse.Namespace, initial_mean: float) -> SimpleNamespace:
    return SimpleNamespace(
        consumption_ratios=[1.0],
        seeds=list(args.seeds),
        episodes=1,
        num_vehicles=200,
        num_ev=100,
        initial_battery_mean=float(initial_mean),
        start_date=args.start_date,
        end_date=args.end_date,
        start_hour=args.start_hour,
        stop_hour=args.stop_hour,
        epoch_length=args.epoch_length,
        parquet_path=args.parquet_path,
        station_csv=None,
        station_capacity_scale=1.0,
        mcmf_backend=args.mcmf_backend,
        charge_wait_bool=args.charge_wait_bool,
        heuristic_battery_threshold=0.5,
        only_manhattan_zones=True,
        output_dir=args.output_dir,
    )


def build_initial_soc_summary(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "realized_initial_battery_mean",
        "realized_initial_battery_mean_human_ev",
        "realized_initial_battery_mean_aev",
        "avg_daily_charging_sessions_per_human_ev",
        "avg_daily_charging_sessions_per_aev",
        "avg_daily_charging_sessions_per_vehicle",
        "avg_charging_session_duration_minutes_human_ev",
        "avg_charging_session_duration_minutes_aev",
        "avg_wait_minutes_waiting_charging_vehicles",
        "waiting_charging_vehicle_count",
        "mean_completed_orders",
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
    for initial_mean, group in detail.groupby(
        "configured_initial_battery_mean", sort=True
    ):
        initial_low, initial_high = initial_battery_bounds(float(initial_mean))
        row = {
            "configured_initial_battery_mean": float(initial_mean),
            "initial_battery_low": initial_low,
            "initial_battery_high": initial_high,
            "num_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("configured_initial_battery_mean")


def save_plot(summary: pd.DataFrame, output_path: Path) -> None:
    x = summary["configured_initial_battery_mean"]
    fig, (charge_axis, wait_axis) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    charge_axis.plot(
        x,
        summary["avg_daily_charging_sessions_per_human_ev"],
        marker="o",
        label="Human EV",
    )
    charge_axis.plot(
        x,
        summary["avg_daily_charging_sessions_per_aev"],
        marker="s",
        label="AEV",
    )
    charge_axis.set_xlabel("Configured mean initial SOC")
    charge_axis.set_ylabel("Charging sessions per vehicle-day")
    charge_axis.set_title("Charging-session starts")
    charge_axis.grid(alpha=0.25)
    charge_axis.legend()

    wait_axis.plot(
        x,
        summary["avg_wait_minutes_waiting_charging_vehicles"],
        color="tab:red",
        marker="^",
    )
    wait_axis.set_xlabel("Configured mean initial SOC")
    wait_axis.set_ylabel("Positive charger-queue wait (minutes)")
    wait_axis.set_title("Wait among charging vehicles")
    wait_axis.grid(alpha=0.25)
    for axis in (charge_axis, wait_axis):
        axis.invert_xaxis()
    fig.suptitle("NYC 200 vehicles: lower initial SOC under exact MCMF")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    means = sorted({float(value) for value in args.initial_battery_means}, reverse=True)
    if any(value <= 0.0 or value > 1.0 for value in means):
        raise ValueError("initial battery means must be in (0, 1]")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_parts = []
    for initial_mean in means:
        print(
            f"\nInitial-SOC diagnostic: configured mean={initial_mean:.3f}, "
            "solver=exact MCMF, learning=off",
            flush=True,
        )
        detail, _, _ = run_sensitivity(_sensitivity_args(args, initial_mean))
        detail_parts.append(detail)
        mean_tag = f"{initial_mean:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        detail.to_csv(
            output_dir / f"initial_soc_{mean_tag}_completed_detail.csv",
            index=False,
        )
        pd.concat(detail_parts, ignore_index=True).to_csv(
            output_dir / "initial_soc_completed_checkpoint_detail.csv",
            index=False,
        )

    detail = pd.concat(detail_parts, ignore_index=True).sort_values(
        ["configured_initial_battery_mean", "seed"], ascending=[False, True]
    )
    summary = build_initial_soc_summary(detail)
    metadata = {
        "experiment": "NYC initial-SOC AEV charging diagnostic",
        "assignment_method": "exact MCMF",
        "mcmf_backend": args.mcmf_backend,
        "charge_wait_bool": bool(args.charge_wait_bool),
        "learning_enabled": False,
        "adp_value": 0.0,
        "battery_consumption_ratio": 1.0,
        "num_vehicles": 200,
        "num_human_ev": 100,
        "num_aev": 100,
        "configured_initial_battery_means": means,
        "seeds": list(args.seeds),
        "date_range": [args.start_date, args.end_date],
        "hour_window": [args.start_hour, args.stop_hour],
        "epoch_length_seconds": args.epoch_length,
        "parquet_path": str(Path(args.parquet_path).expanduser().resolve()),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"nyc_200_vehicle_mcmf_initial_soc_aev_charging_{timestamp}"
    detail_path = output_dir / f"{stem}_detail.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    json_path = output_dir / f"{stem}.json"
    plot_path = output_dir / f"{stem}.png"
    latest_summary_path = output_dir / "nyc_200_vehicle_mcmf_initial_soc_aev_charging_latest.csv"
    latest_plot_path = output_dir / "nyc_200_vehicle_mcmf_initial_soc_aev_charging_latest.png"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(latest_summary_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "detail": detail.to_dict(orient="records"),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_plot(summary, plot_path)
    save_plot(summary, latest_plot_path)
    print("\nInitial-SOC AEV charging summary")
    print(summary[[
        "configured_initial_battery_mean",
        "realized_initial_battery_mean_aev",
        "avg_daily_charging_sessions_per_human_ev",
        "avg_daily_charging_sessions_per_aev",
        "mcmf_aev_charge_selections",
        "mcmf_aev_wait_selections",
        "avg_wait_minutes_waiting_charging_vehicles",
        "mean_final_battery_soc_aev",
    ]].to_string(index=False))
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
