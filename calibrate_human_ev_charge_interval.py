"""Calibrate the Human EV charge-decision interval on real Manhattan data.

The Binary Logit charge probability and station-choice MNL are not modified.
Only the minimum real-time interval between Human EV charge decisions varies.
Assignment uses exact MCMF without learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from test_charging_sensitivity import run_sensitivity


DEFAULT_INTERVALS_MINUTES = (60.0, 120.0, 180.0)
DEFAULT_SEEDS = (256, 512, 1024)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calibrate Human EV charge-decision interval on Manhattan"
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        nargs="+",
        default=list(DEFAULT_INTERVALS_MINUTES),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--initial-battery-mean", type=float, default=0.875)
    parser.add_argument("--start-hour", type=float, default=8.0)
    parser.add_argument("--stop-hour", type=float, default=11.0)
    parser.add_argument("--epoch-length", type=float, default=120.0)
    parser.add_argument(
        "--parquet-path",
        default=(
            "/Users/seinzhou/Desktop/adp_trainer/nyedata/nye_simulation/parquet/"
            "yellow_tripdata_2025-12.parquet"
        ),
    )
    parser.add_argument(
        "--station-csv",
        type=Path,
        default=project_root / "nyedata/nyc_all_charging_stations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results/human_ev_charge_interval_calibration",
    )
    return parser.parse_args()


def experiment_args(
    args: argparse.Namespace,
    *,
    interval_minutes: float,
    seed: int,
    scratch_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        consumption_ratios=[1.0],
        seeds=[int(seed)],
        episodes=1,
        num_vehicles=200,
        num_ev=100,
        initial_battery_mean=float(args.initial_battery_mean),
        start_date="2025-12-18",
        end_date="2025-12-18",
        start_hour=float(args.start_hour),
        stop_hour=float(args.stop_hour),
        epoch_length=float(args.epoch_length),
        parquet_path=str(Path(args.parquet_path).expanduser().resolve()),
        station_csv=str(args.station_csv.expanduser().resolve()),
        station_capacity_scale=1.0,
        mcmf_backend="primal_dual",
        charge_wait_bool=True,
        human_ev_charge_decision_interval_minutes=float(interval_minutes),
        heuristic_battery_threshold=0.5,
        only_manhattan_zones=True,
        output_dir=scratch_dir,
    )


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "avg_daily_charging_sessions_per_human_ev",
        "avg_daily_charging_sessions_per_aev",
        "avg_daily_charging_sessions_per_vehicle",
        "avg_charging_session_duration_minutes_human_ev",
        "avg_wait_minutes_waiting_charging_vehicles",
        "mean_completed_orders",
        "mean_service_ratio",
        "mean_final_battery_soc_human_ev",
    ]
    rows = []
    for interval, group in detail.groupby(
        "human_ev_charge_decision_interval_minutes", sort=True
    ):
        row = {
            "human_ev_charge_decision_interval_minutes": float(interval),
            "num_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def save_plot(summary: pd.DataFrame, output_path: Path, observation_hours: float) -> None:
    x = summary["human_ev_charge_decision_interval_minutes"] / 60.0
    y = summary["mean_avg_daily_charging_sessions_per_human_ev"]
    yerr = summary["std_avg_daily_charging_sessions_per_human_ev"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.0, capsize=4)
    ax.axhspan(2.0, 3.0, color="#2ca02c", alpha=0.14, label="Target: 2--3/day")
    ax.set_xlabel("Minimum Human EV charge-decision interval (hours)")
    ax.set_ylabel("Charging sessions per Human EV-day")
    ax.set_title(
        "NYC Manhattan, 200 vehicles, exact MCMF, no learning\n"
        f"{observation_hours:g}-hour observation window"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    intervals = sorted({float(value) for value in args.interval_minutes})
    if any(not np.isfinite(value) or value < 0.0 for value in intervals):
        raise ValueError("intervals must be finite and nonnegative")
    args.station_csv = args.station_csv.expanduser().resolve()
    parquet_path = Path(args.parquet_path).expanduser().resolve()
    if not args.station_csv.is_file():
        raise FileNotFoundError(args.station_csv)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "detail.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "metadata.json"
    plot_path = output_dir / "human_ev_charge_interval_calibration.png"
    detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    if not detail.empty:
        if not np.allclose(
            detail["configured_initial_battery_mean"].astype(float),
            args.initial_battery_mean,
        ) or not np.allclose(
            detail["observation_days"].astype(float),
            (args.stop_hour - args.start_hour) / 24.0,
        ):
            raise ValueError(
                "Existing results use another initial SOC or observation window. "
                "Choose a new --output-dir instead of mixing experiments."
            )

    for interval in intervals:
        for seed in sorted({int(value) for value in args.seeds}):
            if not detail.empty and (
                np.isclose(
                    detail["human_ev_charge_decision_interval_minutes"].astype(float),
                    interval,
                )
                & (detail["seed"].astype(int) == seed)
            ).any():
                print(f"SKIP interval={interval:g} min seed={seed}", flush=True)
                continue
            print(f"RUN interval={interval:g} min seed={seed}", flush=True)
            run_detail, _, _ = run_sensitivity(
                experiment_args(
                    args,
                    interval_minutes=interval,
                    seed=seed,
                    scratch_dir=output_dir / "run_checkpoint",
                )
            )
            run_detail["human_ev_charge_decision_interval_minutes"] = interval
            detail = pd.concat([detail, run_detail], ignore_index=True)
            detail = detail.sort_values(
                ["human_ev_charge_decision_interval_minutes", "seed"]
            ).drop_duplicates(
                ["human_ev_charge_decision_interval_minutes", "seed"],
                keep="last",
            )
            detail.to_csv(detail_path, index=False)

    summary = summarize(detail)
    summary.to_csv(summary_path, index=False)
    save_plot(summary, plot_path, args.stop_hour - args.start_hour)
    target = summary[
        summary["mean_avg_daily_charging_sessions_per_human_ev"].between(2.0, 3.0)
    ]
    selected = (
        float(target.iloc[0]["human_ev_charge_decision_interval_minutes"])
        if not target.empty
        else None
    )
    metadata = {
        "experiment": "Human EV charge-decision interval calibration",
        "date": "2025-12-18",
        "hour_window": [float(args.start_hour), float(args.stop_hour)],
        "projected_day_normalization": (args.stop_hour - args.start_hour) < 24.0,
        "configured_initial_battery_mean": float(args.initial_battery_mean),
        "num_vehicles": 200,
        "num_human_ev": 100,
        "num_aev": 100,
        "assignment": "exact MCMF (primal_dual), no learning",
        "demand": str(parquet_path),
        "station_csv": str(args.station_csv),
        "station_scope": "real NYC stations mapped to Manhattan TLC zones",
        "binary_logit_and_station_mnl_modified": False,
        "candidate_intervals_minutes": sorted(
            detail["human_ev_charge_decision_interval_minutes"].unique().tolist()
        ),
        "selected_interval_minutes": selected,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"SELECTED interval={selected} minutes", flush=True)
    print(f"DONE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
