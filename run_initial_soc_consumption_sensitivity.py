"""Run resumable NYC charging sensitivity for one initial-SOC setting.

This orchestration script deliberately writes CSV and JSON only.  It reuses the
exact-MCMF, no-learning experiment in ``test_charging_sensitivity.py`` and keeps
one durable row per consumption-ratio/seed configuration so an interrupted
full-day experiment can resume without repeating completed simulations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from test_charging_sensitivity import (
    DEFAULT_CONSUMPTION_RATIOS,
    build_summary,
    build_trend_summary,
    run_sensitivity,
)


DEFAULT_SEEDS = (256, 512, 1024)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Resumable full-day NYC charging sensitivity for one initial SOC"
        )
    )
    parser.add_argument("--initial-battery-mean", type=float, required=True)
    parser.add_argument(
        "--consumption-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_CONSUMPTION_RATIOS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
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
        help="NYC AFDC charging-station CSV used by the adp_trainer NYC preset.",
    )
    parser.add_argument("--station-capacity-scale", type=float, default=1.0)
    parser.add_argument("--start-date", default="2025-12-18")
    parser.add_argument("--end-date", default="2025-12-18")
    parser.add_argument("--start-hour", type=float, default=0.0)
    parser.add_argument("--stop-hour", type=float, default=24.0)
    parser.add_argument("--epoch-length", type=float, default=120.0)
    parser.add_argument("--mcmf-backend", default="primal_dual")
    parser.add_argument(
        "--human-ev-charge-decision-interval-minutes",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results/initial_soc_consumption_sensitivity",
    )
    return parser.parse_args()


def _experiment_args(
    args: argparse.Namespace,
    *,
    ratio: float,
    seed: int,
    scratch_output_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        consumption_ratios=[float(ratio)],
        seeds=[int(seed)],
        episodes=1,
        num_vehicles=200,
        num_ev=100,
        initial_battery_mean=float(args.initial_battery_mean),
        start_date=args.start_date,
        end_date=args.end_date,
        start_hour=float(args.start_hour),
        stop_hour=float(args.stop_hour),
        epoch_length=float(args.epoch_length),
        parquet_path=args.parquet_path,
        station_csv=str(args.station_csv.expanduser().resolve()),
        station_capacity_scale=float(args.station_capacity_scale),
        mcmf_backend=args.mcmf_backend,
        charge_wait_bool=True,
        human_ev_charge_decision_interval_minutes=float(
            args.human_ev_charge_decision_interval_minutes
        ),
        heuristic_battery_threshold=0.5,
        only_manhattan_zones=True,
        output_dir=scratch_output_dir,
    )


def _is_completed(detail: pd.DataFrame, ratio: float, seed: int) -> bool:
    if detail.empty:
        return False
    return bool(
        (
            np.isclose(detail["battery_consumption_ratio"].astype(float), ratio)
            & (detail["seed"].astype(int) == seed)
        ).any()
    )


def main() -> None:
    args = parse_args()
    args.station_csv = args.station_csv.expanduser().resolve()
    if not args.station_csv.is_file():
        raise FileNotFoundError(
            f"NYC charging-station CSV does not exist: {args.station_csv}"
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "detail.csv"
    summary_path = output_dir / "summary.csv"
    trend_path = output_dir / "trend.csv"
    json_path = output_dir / "results.json"
    scratch_output_dir = output_dir / "run_checkpoint"

    detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    if not detail.empty:
        interval_column = "human_ev_charge_decision_interval_minutes"
        if interval_column not in detail or not np.allclose(
            detail[interval_column].astype(float),
            float(args.human_ev_charge_decision_interval_minutes),
        ):
            raise ValueError(
                "Existing results use a different or unrecorded Human EV charge "
                "decision interval. Choose a new --output-dir; legacy results "
                "must not be mixed with the new decision-clock experiment."
            )
    ratios = sorted({float(value) for value in args.consumption_ratios})
    seeds = sorted({int(value) for value in args.seeds})
    total = len(ratios) * len(seeds)

    for ratio in ratios:
        for seed in seeds:
            if _is_completed(detail, ratio, seed):
                print(
                    f"SKIP completed ratio={ratio:.4f}, seed={seed}", flush=True
                )
                continue
            print(
                f"RUN ratio={ratio:.4f}, seed={seed}, "
                f"initial_soc={args.initial_battery_mean:.3f}",
                flush=True,
            )
            run_detail, _, _ = run_sensitivity(
                _experiment_args(
                    args,
                    ratio=ratio,
                    seed=seed,
                    scratch_output_dir=scratch_output_dir,
                )
            )
            detail = pd.concat([detail, run_detail], ignore_index=True)
            detail = detail.sort_values(
                ["battery_consumption_ratio", "seed"]
            ).drop_duplicates(
                ["battery_consumption_ratio", "seed"], keep="last"
            )
            detail.to_csv(detail_path, index=False)
            print(f"CHECKPOINT {len(detail)}/{total}: {detail_path}", flush=True)

    summary = build_summary(detail)
    trend = (
        build_trend_summary(summary)
        if summary["battery_consumption_ratio"].nunique() >= 2
        else pd.DataFrame()
    )
    metadata = {
        "experiment": "NYC initial-SOC and energy-consumption sensitivity",
        "learning_enabled": False,
        "assignment_method": "exact MCMF",
        "mcmf_backend": args.mcmf_backend,
        "charge_wait_bool": True,
        "human_ev_charge_decision_interval_minutes": float(
            args.human_ev_charge_decision_interval_minutes
        ),
        "configured_initial_battery_mean": float(args.initial_battery_mean),
        "consumption_ratios": ratios,
        "seeds": seeds,
        "num_vehicles": 200,
        "num_human_ev": 100,
        "num_aev": 100,
        "date_range": [args.start_date, args.end_date],
        "hour_window": [float(args.start_hour), float(args.stop_hour)],
        "epoch_length_seconds": float(args.epoch_length),
        "parquet_path": str(Path(args.parquet_path).expanduser().resolve()),
        "station_csv": str(args.station_csv),
        "station_network": "NYC AFDC stations mapped to Manhattan TLC zones",
        "station_capacity_scale": float(args.station_capacity_scale),
        "completed_configurations": int(len(detail)),
    }
    summary.to_csv(summary_path, index=False)
    trend.to_csv(trend_path, index=False)
    json_path.write_text(
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
    print(f"DONE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
