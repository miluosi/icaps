"""Analyze the three-case NYC charging sensitivity experiment.

The script reads the aggregated sensitivity CSV, maps the simulator's verbose
metric names to the compact comparison schema requested for the paper, writes
the comparison data and LaTeX table code, and renders both a combined H/M/L
figure and one figure for each initial-SOC case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CASE_LABELS = {0.875: "H", 0.45: "M", 0.15: "L"}
COMPARE_LISTS = [
    "initial_battery_case",
    "battery_consumption_ratio",
    "avg_reward",
    "mean_completed_orders",
    "avg_battery_level",
    "avg_charge_num_ev",
    "avg_charge_num_aev",
    "avg_charge_time",
    "avg_wait",
]
SOURCE_COLUMNS = {
    "battery_consumption_ratio": "battery_consumption_ratio",
    "avg_reward": "avg_reward",
    "mean_completed_orders": "mean_completed_orders",
    "avg_battery_level": "mean_final_battery_soc",
    "avg_charge_num_ev": "avg_daily_charging_sessions_per_human_ev",
    "avg_charge_num_aev": "avg_daily_charging_sessions_per_aev",
    "avg_charge_time": "avg_charging_session_duration_minutes_all",
    "avg_wait": "avg_wait_minutes_waiting_charging_vehicles",
}
PLOT_LABELS = {
    "avg_reward": "Average daily reward",
    "mean_completed_orders": "Completed orders",
    "avg_battery_level": "Final battery SOC",
    "avg_charge_num_ev": "Human EV charging sessions/vehicle-day",
    "avg_charge_num_aev": "AEV charging sessions/vehicle-day",
    "avg_charge_time": "Charging-session duration (min)",
    "avg_wait": "Positive charger wait (min)",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_output = root / "outputs/nyc_real_station_charging_sensitivity"
    parser = argparse.ArgumentParser(
        description="Create LaTeX and figures for NYC charging sensitivity"
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=default_output / "combined_summary.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"initial_soc_case", *SOURCE_COLUMNS.values()}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"summary CSV is missing required columns: {missing}")

    comparison = pd.DataFrame()
    rounded_soc = summary["initial_soc_case"].astype(float).round(3)
    comparison["initial_battery_case"] = rounded_soc.map(CASE_LABELS)
    if comparison["initial_battery_case"].isna().any():
        unknown = sorted(rounded_soc[comparison["initial_battery_case"].isna()].unique())
        raise ValueError(f"unknown initial-SOC cases: {unknown}")
    for target, source in SOURCE_COLUMNS.items():
        comparison[target] = pd.to_numeric(summary[source], errors="raise")
    comparison["initial_battery_case"] = pd.Categorical(
        comparison["initial_battery_case"], categories=["H", "M", "L"], ordered=True
    )
    return comparison[COMPARE_LISTS].sort_values(
        ["initial_battery_case", "battery_consumption_ratio"]
    ).reset_index(drop=True)


def latex_table(comparison: pd.DataFrame) -> str:
    headers = [
        "Initial battery",
        "Consumption ratio",
        "Avg. reward",
        "Completed orders",
        "Final SOC",
        "Human EV charges",
        "AEV charges",
        "Charge time (min)",
        "Wait (min)",
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{NYC charging sensitivity with real Manhattan charging stations}",
        r"\label{tab:nyc_real_station_charging_sensitivity}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ccrrrrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    previous_case = None
    for row in comparison.itertuples(index=False):
        if previous_case is not None and row.initial_battery_case != previous_case:
            lines.append(r"\midrule")
        lines.append(
            f"{row.initial_battery_case} & "
            f"{row.battery_consumption_ratio:.4f} & "
            f"{row.avg_reward:.1f} & "
            f"{row.mean_completed_orders:.1f} & "
            f"{row.avg_battery_level:.3f} & "
            f"{row.avg_charge_num_ev:.3f} & "
            f"{row.avg_charge_num_aev:.3f} & "
            f"{row.avg_charge_time:.2f} & "
            f"{row.avg_wait:.2f} \\\\"
        )
        previous_case = row.initial_battery_case
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\footnotesize",
        (
            r"Notes: H, M, and L denote configured initial mean SOC values of "
            r"0.875, 0.45, and 0.15. Human EV and AEV charging counts are actual "
            r"charging starts per vehicle-day. Charge time excludes travel and queueing. "
            r"Wait is the mean positive queue wait across waiting charging vehicles."
        ),
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def _plot_panels(comparison: pd.DataFrame, output_path: Path, case: str | None) -> None:
    metrics = list(PLOT_LABELS)
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 14.5))
    axes = axes.ravel()
    colors = {"H": "#2563eb", "M": "#ea580c", "L": "#15803d"}
    selected_cases = [case] if case is not None else ["H", "M", "L"]
    for axis, metric in zip(axes, metrics):
        for label in selected_cases:
            rows = comparison[comparison["initial_battery_case"].eq(label)]
            axis.plot(
                rows["battery_consumption_ratio"],
                rows[metric],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=colors[label],
                label=label,
            )
        axis.set_title(PLOT_LABELS[metric])
        axis.set_xlabel("Driving-energy consumption ratio")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        if case is None:
            axis.legend(title="Initial SOC", frameon=False)
    axes[-1].axis("off")
    title = (
        "NYC 200-vehicle charging sensitivity: H/M/L comparison"
        if case is None
        else f"NYC 200-vehicle charging sensitivity: initial SOC {case}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = args.summary_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary CSV does not exist: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison(pd.read_csv(summary_path))
    comparison.to_csv(output_dir / "charging_sensitivity_compare.csv", index=False)
    (output_dir / "charging_sensitivity_table.tex").write_text(
        latex_table(comparison), encoding="utf-8"
    )
    _plot_panels(
        comparison,
        output_dir / "charging_sensitivity_HML.png",
        case=None,
    )
    for case in ("H", "M", "L"):
        _plot_panels(
            comparison,
            output_dir / f"charging_sensitivity_{case}.png",
            case=case,
        )
    print(f"comparison rows: {len(comparison)}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
