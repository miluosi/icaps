"""Legacy solver evaluator; not the fixed-policy ICAPS recourse paper runner."""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import src.ADPtrainer as adp_trainer_module
from src.ADPtrainer import ADPTrainer
from src.charging_wait_metrics import aggregate_wait_metrics
from src.synthetic_scenario import (
    DEFAULT_AEV_INITIAL_BATTERY_SCALE,
    DEFAULT_CHARGE_DURATION,
    DEFAULT_CRITICAL_CHARGING_BATTERY,
    DEFAULT_EPISODE_DAYS,
    DEFAULT_GRID_SIZE,
    DEFAULT_NUM_STATIONS,
    DEFAULT_SIMULATION_PERIOD,
    DEFAULT_STATION_CAPACITY,
    DEFAULT_STATION_QUEUE_CAPACITY,
    DEFAULT_SYNTHETIC_DEMAND_PROFILE,
    DEFAULT_SYNTHETIC_DEMAND_SCALE,
    DEFAULT_WAIT_PENALTY_PER_STEP,
    synthetic_checkpoint_suffix,
)


# ── Test strategies ───────────────────────────────────────────────
STRATEGIES = [
    {"name": "MASAC",    "adp": 1.0, "gurobi": True,  "usemcmf": False,  "load_ckpt": True,  "known_reject": False},
    {"name": "ADP-ILP",  "adp": 1.0, "gurobi": True,  "usemcmf": False, "load_ckpt": True,  "known_reject": False},
    {"name": "ADP-ILP-QOFF", "adp": 1.0, "gurobi": True, "usemcmf": False, "load_ckpt": True, "known_reject": False, "disable_queue_predictor": True},
    {"name": "ADP-ILP-DOFF", "adp": 1.0, "gurobi": True, "usemcmf": False, "load_ckpt": True, "known_reject": False, "disable_post_demand_predictor": True},
    {"name": "ADP-ILP-QDOFF", "adp": 1.0, "gurobi": True, "usemcmf": False, "load_ckpt": True, "known_reject": False, "disable_queue_predictor": True, "disable_post_demand_predictor": True},
    {"name": "ILP",      "adp": 0.0, "gurobi": True,  "usemcmf": False, "load_ckpt": False, "known_reject": False},
    {"name": "MCMF-K",   "adp": 0.0, "gurobi": True,  "usemcmf": True,  "load_ckpt": False, "known_reject": True},
    {"name": "ADP-MCMF", "adp": 1.0, "gurobi": True,  "usemcmf": True,  "load_ckpt": True,  "known_reject": False},
    {"name": "ADP-MCMF-QOFF", "adp": 1.0, "gurobi": True, "usemcmf": True, "load_ckpt": True, "known_reject": False, "disable_queue_predictor": True},
    {"name": "ADP-MCMF-K", "adp": 1.0, "gurobi": True,  "usemcmf": True,  "load_ckpt": True,  "known_reject": True},
    {"name": "ADP-MCMF-FT", "adp": 1.0, "gurobi": True, "usemcmf": True, "load_ckpt": True, "known_reject": False, "train_during_test": True},
    {"name": "MCMF",     "adp": 0.0, "gurobi": True,  "usemcmf": True,  "load_ckpt": False, "known_reject": False},
    {"name": "ADP-AUCTION", "adp": 1.0, "gurobi": True, "usemcmf": False, "useauction": True, "load_ckpt": True, "known_reject": False},
    {"name": "AUCTION", "adp": 0.0, "gurobi": True, "usemcmf": False, "useauction": True, "load_ckpt": False, "known_reject": False},
    {"name": "ADP-HEU",  "adp": 1.0, "gurobi": False, "usemcmf": False, "load_ckpt": True, "checkpoint_assign_tag": "gurobi", "known_reject": False},
    {"name": "ADP-HEU-K", "adp": 1.0, "gurobi": False, "usemcmf": False, "load_ckpt": True, "checkpoint_assign_tag": "gurobi", "known_reject": True},
    {"name": "ADP-HEU-HEU", "adp": 1.0, "gurobi": False, "usemcmf": False, "load_ckpt": True, "checkpoint_assign_tag": "heu", "known_reject": False},
    {"name": "ADP-HEU-HEU-K", "adp": 1.0, "gurobi": False, "usemcmf": False, "load_ckpt": True, "checkpoint_assign_tag": "heu", "known_reject": True},
    {"name": "HEU",      "adp": 0.0, "gurobi": False, "usemcmf": False, "load_ckpt": False, "known_reject": False},
    {"name": "HEU-K",      "adp": 0.0, "gurobi": False, "usemcmf": False, "load_ckpt": False, "known_reject": True},
]

STRATEGY_NAMES = tuple(strategy["name"] for strategy in STRATEGIES)
# DEFAULT_STRATEGIES = (
#     "MCMF",
#     "MCMF-K",
#     "HEU",
#     "ADP-MCMF",
#     "ADP-HEU",
#     "ADP-MCMF-FT",
# )

DEFAULT_STRATEGIES = (
    "ADP-MCMF",
    "ADP-HEU",
)



PRESERVED_OUTPUT_COLUMNS = {
    "mean_service_ratio",
    "mean_service_ratio_pct",
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

if len(STRATEGY_NAMES) != len(set(STRATEGY_NAMES)):
    raise RuntimeError("Duplicate strategy names in test_model.STRATEGIES")


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


def _derive_service_ratio(stats: dict) -> float:
    completed_orders = float(stats.get("completed_orders", 0) or 0)
    generated_orders = _first_present_value(
        stats,
        "total_generated_requests",
        "whole_req_num",
    )
    if generated_orders is None:
        generated_orders = stats.get("total_orders", 0)
    generated_orders = float(generated_orders or 0)
    if generated_orders <= 0:
        return 0.0
    return completed_orders / generated_orders


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Q-network checkpoints across ILP, MCMF, and heuristic backends")
    from src.acceptance_features import add_acceptance_arguments
    add_acceptance_arguments(parser)
    parser.add_argument('--checkpoint-suffix', default='', help='Experiment namespace printed by the training CLI, excluding the auto-added EV predictor hash')
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes per seed")
    parser.add_argument("--num-vehicles", type=int, default=200, help="Total vehicles")
    parser.add_argument("--num-ev", type=int, default=100, help="EV vehicles")
    parser.add_argument("--seeds", type=int, nargs="+", default=[256],
                        help="Random seeds for evaluation (different from training seed)")
    parser.add_argument("--transportation-modes", type=str, nargs="+",
                        default=["integrated"],
                        choices=["integrated", "evfirst", "aevfirst"],
                        help="Transportation modes to test")
    parser.add_argument("--demand-patterns", type=str, nargs="+",
                        default=["intense"],
                        choices=["intense", "random"],
                        help="Demand patterns to test")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=list(DEFAULT_STRATEGIES),
                        choices=STRATEGY_NAMES,
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
    parser.add_argument("--heuristic-battery-threshold", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--known-reject", action="store_true")
    parser.set_defaults(known_reject=False)
    parser.add_argument("--num-stations", type=int, default=DEFAULT_NUM_STATIONS, help="Number of charging stations")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE, help="Grid size for the environment (NxN)")
    parser.add_argument("--station-capacity", type=int, default=DEFAULT_STATION_CAPACITY, help="Charging slots per station")
    parser.add_argument("--station-queue-capacity", type=int, default=DEFAULT_STATION_QUEUE_CAPACITY, help="Maximum AEV waiting-room admissions per station")
    parser.add_argument("--charge-duration", type=int, default=DEFAULT_CHARGE_DURATION, help="Steps required for a full charge")
    parser.add_argument(
        "--checkpoint-charge-duration",
        type=int,
        default=None,
        help=(
            "Charge duration encoded in the checkpoint path; defaults to the "
            "evaluation duration. This supports an explicit cross-duration "
            "stress test without relabeling the trained checkpoint."
        ),
    )
    parser.add_argument("--aev-initial-battery-scale", type=float, default=DEFAULT_AEV_INITIAL_BATTERY_SCALE, help="Scale predictive initial AEV state of charge")
    parser.add_argument(
        "--checkpoint-aev-initial-battery-scale",
        type=float,
        default=None,
        help=(
            "Initial-battery scale encoded in the checkpoint path; defaults "
            "to the evaluation value for an in-scenario test."
        ),
    )
    parser.add_argument("--critical-charging-battery", type=float, default=DEFAULT_CRITICAL_CHARGING_BATTERY, help="AEV SoC below which waiting is infeasible")
    parser.add_argument(
        "--distribution-mode",
        type=str,
        default="st_masac_gat_queue_demand_gurobi",
        choices=[
            "bayes", "time-only", "none",
            "integrated_directq", "optimization_anchored_residual",
            "st_masac_gat_post_demand_direct",
            "st_masac_gat_queue_demand_gurobi",
        ],
        help="ICAPS core value-function mode for checkpoint lookup and evaluation",
    )
    parser.add_argument("--simulation-period", type=int, default=DEFAULT_SIMULATION_PERIOD, help="Simulation steps per virtual day")
    parser.add_argument("--episode-days", type=int, default=DEFAULT_EPISODE_DAYS, help="Virtual days per evaluation episode")
    parser.add_argument("--charging-wait-penalty-per-step", type=float, default=DEFAULT_WAIT_PENALTY_PER_STEP, help="Environment reward penalty for each charger-wait step")
    parser.add_argument(
        "--synthetic-demand-profile",
        choices=["predictive", "legacy"],
        default=DEFAULT_SYNTHETIC_DEMAND_PROFILE,
        help="Predictable within-day demand profile or the legacy saturated generator",
    )
    parser.add_argument(
        "--synthetic-demand-scale",
        type=float,
        default=DEFAULT_SYNTHETIC_DEMAND_SCALE,
        help="Multiplier on predictive request-arrival probabilities",
    )
    parser.add_argument(
        "--checkpoint-synthetic-demand-scale",
        type=float,
        default=None,
        help=(
            "Demand scale encoded in the checkpoint path; defaults to the "
            "evaluation demand scale. Use 1.0 to stress-test a scale-1 model "
            "under a slightly different evaluation load."
        ),
    )
    parser.add_argument("--post-demand-q-weight", type=float, default=0.0, help="Initial action-head weight for direct post-demand MASAC")
    parser.add_argument("--post-demand-head-lr-multiplier", type=float, default=10.0, help="Learning-rate multiplier for the direct post-demand action head")
    parser.add_argument(
        "--masac-target-entropy-ratio",
        type=float,
        default=0.9,
        help="Target entropy as a fraction of log(candidate count) for masac_baseline/standard_masac_gat fine-tuning",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=["latest", "best"],
        default="best",
        help="Load the final trained checkpoint or the best post-warmup checkpoint",
    )
    parser.add_argument('--allow-online-adaptation', action='store_true',
                        help='Allow ADP-MCMF-FT as a separately labeled online-adaptation run')
    args = parser.parse_args()
    if 'ADP-MCMF-FT' in args.strategies and not args.allow_online_adaptation:
        parser.error('ADP-MCMF-FT trains during test; use --allow-online-adaptation and exclude it from fixed-policy tables')
    return args


def normalize_distribution_mode(distribution_mode: str | None) -> str:
    mode = distribution_mode or "none"
    aliases = {
        "masac_demand_direct": "st_masac_gat_post_demand_direct",
    }
    return aliases.get(mode, mode)


def get_distribution_suffix(distribution_mode: str) -> str:
    distribution_mode = normalize_distribution_mode(distribution_mode)
    if distribution_mode == "elbo":
        return "_noenc"
    if distribution_mode == "bayes":
        return ""
    if distribution_mode == "bayes_simple":
        return "_bayes_simple"
    if distribution_mode == "none":
        return "_none"
    if distribution_mode == "masac_queue_length":
        return "_masac_queue_length"
    if distribution_mode == "st_masac_gat_former2_queue_feature":
        return "_st_masac_gat_former2_queue_feature"
    if distribution_mode == "st_masac_gat_former2_queue_feature_greedy_alpha":
        return "_st_masac_gat_former2_queue_feature_greedy_alpha"
    if distribution_mode == "st_masac_gat_former2_queue_feature_fixed_alpha":
        return "_st_masac_gat_former2_queue_feature_fixed_alpha"
    if distribution_mode == "masac_baseline":
        return "_masac_baseline"
    if distribution_mode == "st_masac_gat_post_demand_direct":
        return "_st_masac_gat_post_demand_direct"
    if distribution_mode == "st_masac_gat_queue_demand_gurobi":
        return "_st_masac_gat_queue_demand_gurobi"
    if distribution_mode in {"integrated_directq", "optimization_anchored_residual"}:
        return "_" + distribution_mode
    if distribution_mode == "standard_masac_gat":
        return "_standard_masac_gat"
    if distribution_mode == "standard_masac_gat_total_q":
        return "_standard_masac_gat_total_q"
    if distribution_mode == "standard_masac_gat_greedy_alpha":
        return "_standard_masac_gat_greedy_alpha"
    if distribution_mode == "standard_masac_gat_fixed_alpha":
        return "_standard_masac_gat_fixed_alpha"
    return "_noenc"


def build_checkpoint_dir(
    mode: str,
    num_ev: int,
    intense: bool,
    vtype: str,
    distribution_mode: str,
    checkpoint_scenario_suffix: str = "",
    assign_tag: str = "gurobi",
) -> str:
    """Build a checkpoint directory for the selected training assignment."""
    if assign_tag not in {"gurobi", "heu"}:
        raise ValueError("assign_tag must be one of: gurobi, heu")
    enc_suffix = get_distribution_suffix(distribution_mode) + str(
        checkpoint_scenario_suffix or ""
    )
    if mode == "integrated":
        return f"checkpoints/q_networks_{assign_tag}_{mode}_{num_ev}_{intense}_{vtype}{enc_suffix}"
    elif mode == "evfirst":
        return f"checkpoints/q_networksevfirst_{assign_tag}_{mode}_{num_ev}_{intense}_{vtype}{enc_suffix}"
    elif mode == "aevfirst":
        return f"checkpoints/q_networksaevfirst_{assign_tag}_{mode}_{num_ev}_{intense}_{vtype}{enc_suffix}"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    args = parse_args()
    zone_distribution_mode = normalize_distribution_mode(args.distribution_mode)
    checkpoint_scenario_suffix = synthetic_checkpoint_suffix(
        num_stations=args.num_stations,
        station_capacity=args.station_capacity,
        charge_duration=(
            args.checkpoint_charge_duration
            if args.checkpoint_charge_duration is not None
            else args.charge_duration
        ),
        simulation_period=args.simulation_period,
        episode_days=args.episode_days,
        charging_wait_penalty_per_step=args.charging_wait_penalty_per_step,
        station_queue_capacity=args.station_queue_capacity,
        aev_initial_battery_scale=(
            args.checkpoint_aev_initial_battery_scale
            if args.checkpoint_aev_initial_battery_scale is not None
            else args.aev_initial_battery_scale
        ),
        critical_charging_battery=args.critical_charging_battery,
        grid_size=args.grid_size,
        synthetic_demand_profile=args.synthetic_demand_profile,
        synthetic_demand_scale=(
            args.checkpoint_synthetic_demand_scale
            if args.checkpoint_synthetic_demand_scale is not None
            else args.synthetic_demand_scale
        ),
    )
    checkpoint_scenario_suffix = '_'.join(part for part in (checkpoint_scenario_suffix, args.checkpoint_suffix) if part)
    from src.acceptance_features import acceptance_checkpoint_suffix
    checkpoint_lookup_suffix = checkpoint_scenario_suffix + acceptance_checkpoint_suffix(args.ev_acceptance_feature, args.ev_acceptance_model,
        anchor=args.ev_response_anchor, critic_input=args.ev_response_critic_input)
    if zone_distribution_mode in {"integrated_directq", "optimization_anchored_residual"}:
        from src.value_function_registry import get_value_function_class
        adp_trainer_module.PyTorchChargingValueFunction = get_value_function_class(zone_distribution_mode)
    elif zone_distribution_mode == "elbo":
        from src.ValueFunction_pytorch_elbo import PyTorchChargingValueFunction as ELBOPyTorchChargingValueFunction
        adp_trainer_module.PyTorchChargingValueFunction = ELBOPyTorchChargingValueFunction
    elif zone_distribution_mode == "bayes_simple":
        from src.ValueFunction_pytorch_bayessimple import PyTorchChargingValueFunction as BayesSimplePyTorchChargingValueFunction
        adp_trainer_module.PyTorchChargingValueFunction = BayesSimplePyTorchChargingValueFunction
    elif zone_distribution_mode in {
        "masac_queue_length",
        "st_masac_gat_former2_queue_feature",
        "st_masac_gat_former2_queue_feature_greedy_alpha",
        "st_masac_gat_former2_queue_feature_fixed_alpha",
    }:
        from src.ValueFunction_st_masac_gat_former2_queue_feature import PyTorchChargingValueFunction as MASACQueueLengthValueFunction
        adp_trainer_module.PyTorchChargingValueFunction = MASACQueueLengthValueFunction
    elif zone_distribution_mode == "masac_baseline":
        from src.ValueFunction_masac_baseline import PyTorchChargingValueFunction as MASACBaselineValueFunction
        adp_trainer_module.PyTorchChargingValueFunction = MASACBaselineValueFunction
    elif zone_distribution_mode in {
        "st_masac_gat_post_demand_direct",
        "st_masac_gat_queue_demand_gurobi",
    }:
        from src.ValueFunction_st_masac_gat_post_demand_direct import PyTorchChargingValueFunction as MASACPostDemandDirectValueFunction
        adp_trainer_module.PyTorchChargingValueFunction = MASACPostDemandDirectValueFunction
    elif zone_distribution_mode == "standard_masac_gat_total_q":
        from src.ValueFunction_standard_masac_gat_total_q import PyTorchChargingValueFunction as StandardMASACTotalQValueFunction
        adp_trainer_module.PyTorchChargingValueFunction = StandardMASACTotalQValueFunction
    elif zone_distribution_mode in {
        "standard_masac_gat",
        "standard_masac_gat_greedy_alpha",
        "standard_masac_gat_fixed_alpha",
    }:
        from src.ValueFunction_standard_masac_gat import PyTorchChargingValueFunction as StandardMASACGATValueFunction
        adp_trainer_module.PyTorchChargingValueFunction = StandardMASACGATValueFunction
    demand_map = {"intense": True, "random": False}
    strategy_map = {s["name"]: s for s in STRATEGIES}
    selected_strategies = [strategy_map[n] for n in args.strategies]

    print("=" * 80)
    print("Model Evaluation")
    print(f"   Strategies: {[s['name'] for s in selected_strategies]}")
    print(f"   Modes: {args.transportation_modes}")
    print(f"   Demand patterns: {args.demand_patterns}")
    print(f"   Seeds: {args.seeds}")
    print(f"   Episodes per config: {args.episodes}")
    print(f"   Vehicles: {args.num_vehicles} (EV={args.num_ev})")
    print(f"   Grid: {args.grid_size}x{args.grid_size}")
    print(f"   Distribution mode: {zone_distribution_mode}")
    print(f"   Checkpoint selection: {args.checkpoint_selection}")
    print(f"   Synthetic demand profile: {args.synthetic_demand_profile}")
    print(
        "   Queue scenario: "
        f"stations={args.num_stations}, slots={args.station_capacity}, "
        f"charge={args.charge_duration}, horizon="
        f"{args.simulation_period * args.episode_days}, "
        f"wait_cost={args.charging_wait_penalty_per_step:g}"
    )
    print(
        "   Checkpoint scenario suffix: "
        f"{checkpoint_scenario_suffix or '<legacy>'}"
    )
    print("=" * 80)

    # ── 1. Pre-check: checkpoint availability (only needed for ADP strategies) ──
    need_ckpt = any(s["load_ckpt"] for s in selected_strategies)
    checkpoint_tags = sorted({
        strategy.get("checkpoint_assign_tag", "gurobi")
        for strategy in selected_strategies
        if strategy["load_ckpt"]
    })
    ckpt_available = {}  # (mode, intense, training assignment tag) -> bool
    if need_ckpt:
        print(
            "\nCheckpoint check "
            f"(training tags = {checkpoint_tags}, selection = {args.checkpoint_selection}):"
        )
        for mode in args.transportation_modes:
            for dp_name in args.demand_patterns:
                intense = demand_map[dp_name]
                for checkpoint_tag in checkpoint_tags:
                    key = (mode, intense, checkpoint_tag)
                    ev_dir = build_checkpoint_dir(
                        mode, args.num_ev, intense, "ev",
                        zone_distribution_mode, checkpoint_lookup_suffix,
                        assign_tag=checkpoint_tag,
                    )
                    aev_dir = build_checkpoint_dir(
                        mode, args.num_ev, intense, "aev",
                        zone_distribution_mode, checkpoint_lookup_suffix,
                        assign_tag=checkpoint_tag,
                    )
                    ev_checkpoint, aev_checkpoint = ADPTrainer.find_checkpoint_pair(
                        ev_dir,
                        aev_dir,
                        prefer_best=args.checkpoint_selection == "best",
                    )
                    print(f"   [{mode}/{dp_name}/{checkpoint_tag}/aev] {aev_dir} -> {aev_checkpoint or 'NOT FOUND'}")
                    print(f"   [{mode}/{dp_name}/{checkpoint_tag}/ev] {ev_dir} -> {ev_checkpoint or 'NOT FOUND'}")
                    ckpt_available[key] = bool(ev_checkpoint and aev_checkpoint)
        missing_count = sum(1 for v in ckpt_available.values() if not v)
        if missing_count:
            print(f"\n⚠  {missing_count} config(s) missing checkpoint — ADP strategies will be skipped for those.")
    print()

    # ── 2. Run evaluation ──
    all_results = []
    trainer = ADPTrainer()

    for mode in args.transportation_modes:
        for dp_name in args.demand_patterns:
            intense = demand_map[dp_name]

            for strat in selected_strategies:
                checkpoint_assign_tag = strat.get("checkpoint_assign_tag", "gurobi")
                # Skip ADP strategies if checkpoint not available
                if strat["load_ckpt"] and not ckpt_available.get(
                    (mode, intense, checkpoint_assign_tag),
                    False,
                ):
                    print(
                        f"⏭  Skipping {strat['name']} for {mode}/{dp_name}: "
                        f"{checkpoint_assign_tag} checkpoint not found"
                    )
                    continue

                for seed in args.seeds:
                    print(f"\n{'─' * 70}")
                    print(f"▶ {strat['name']}  mode={mode}  demand={dp_name}  seed={seed}")
                    print(f"{'─' * 70}")

                    fine_tune = bool(strat.get("train_during_test", False))
                    results, env = trainer.run_charging_integration_test(
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
                        knownreject=strat.get("known_reject", args.known_reject),
                        ifloadgingValueFunction=strat["load_ckpt"],
                        load_checkpoint_assign_tag=(
                            checkpoint_assign_tag if strat["load_ckpt"] else None
                        ),
                        trainnetwork=fine_tune,
                        evaluatemode=not fine_tune,
                        save_checkpoints=False,
                        random_seed=seed,
                        grid_size=args.grid_size,
                        num_stations=args.num_stations,
                        station_capacity=args.station_capacity,
                        station_queue_capacity=args.station_queue_capacity,
                        charge_duration=args.charge_duration,
                        aev_initial_battery_scale=args.aev_initial_battery_scale,
                        critical_charging_battery=args.critical_charging_battery,
                        zone_distribution_mode=zone_distribution_mode,
                        ev_acceptance_feature=args.ev_acceptance_feature,
                        ev_acceptance_model=args.ev_acceptance_model,
                ev_response_anchor=args.ev_response_anchor,
                ev_response_critic_input=args.ev_response_critic_input,
                        post_demand_q_weight=args.post_demand_q_weight,
                        post_demand_head_lr_multiplier=args.post_demand_head_lr_multiplier,
                        masac_target_entropy_ratio=args.masac_target_entropy_ratio,
                        checkpoint_selection=args.checkpoint_selection,
                        disable_queue_predictor=bool(strat.get("disable_queue_predictor", False)),
                        disable_post_demand_predictor=bool(strat.get("disable_post_demand_predictor", False)),
                        simulation_period=args.simulation_period,
                        episode_length=(
                            args.simulation_period * args.episode_days
                            if args.episode_days is not None else None
                        ),
                        charging_wait_penalty_per_step=args.charging_wait_penalty_per_step,
                        synthetic_demand_profile=args.synthetic_demand_profile,
                        synthetic_demand_scale=args.synthetic_demand_scale,
                        checkpoint_scenario_suffix=checkpoint_scenario_suffix,
                    )

                    rewards = results.get("episode_rewards", [])
                    avg_reward = np.mean(rewards) if rewards else 0.0

                    detailed = results.get("episode_detailed_stats", [])
                    total_accept = sum(d.get("accepted_orders", 0) for d in detailed)
                    total_reject = sum(d.get("rejected_orders", 0) for d in detailed)
                    total_rejected_requests = sum(d.get("rejected_requests", d.get("rejected_orders", 0)) for d in detailed)
                    total_recourse_requests = sum(d.get("recourse_requests", 0) for d in detailed)
                    total_lost_requests = sum(d.get("lost_requests", 0) for d in detailed)
                    total_complete = sum(d.get("completed_orders", 0) for d in detailed)
                    total_orders = sum(d.get("total_orders", 0) for d in detailed)
                    mean_ev_completed_orders = _mean_detail_metric(detailed, "completed_ev_orders")
                    mean_aev_completed_orders = _mean_detail_metric(detailed, "completed_aev_orders", fallback=_derive_completed_aev_orders)
                    mean_assignment_success_rate = _mean_detail_metric(detailed, "assignment_success_rate", fallback=_derive_assignment_success_rate)
                    mean_service_ratio = _mean_detail_metric(
                        detailed,
                        "service_ratio",
                        fallback=_derive_service_ratio,
                    )
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
                    mean_episode_reward_aev = _mean_detail_metric(detailed, "episode_reward_aev")
                    mean_episode_reward_ev = _mean_detail_metric(detailed, "episode_reward_ev")
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
                    mean_charging_wait_penalty = _mean_detail_metric(detailed, "charging_wait_penalty_total")
                    mean_charging_wait_steps = _mean_detail_metric(detailed, "charging_wait_steps")
                    mean_charging_wait_time = _mean_detail_metric(detailed, "avg_charging_wait_time")
                    wait_summary = aggregate_wait_metrics(detailed)
                    mean_avg_wait = wait_summary["avg_wait"]
                    total_waiting_vehicle_count = wait_summary[
                        "waiting_vehicle_count"
                    ]
                    mean_waiting_vehicle_count = wait_summary[
                        "mean_waiting_vehicle_count"
                    ]
                    mean_station_pressure = _mean_detail_metric(detailed, "mean_station_pressure")
                    mean_max_station_pressure = _mean_detail_metric(detailed, "mean_max_station_pressure")
                    max_station_pressure = max(
                        [float(stats.get("max_station_pressure", 0.0) or 0.0) for stats in detailed],
                        default=0.0,
                    )
                    queue_predictor_holdout_mse = _mean_detail_metric(detailed, "queue_predictor_holdout_mse")
                    queue_predictor_training_mse_aev = _mean_detail_metric(detailed, "queue_predictor_training_mse_aev")
                    queue_input_weight_norm_aev = _mean_detail_metric(detailed, "queue_input_weight_norm_aev")

                    entry = {
                        "strategy": strat["name"],
                        "mode": mode,
                        "demand": dp_name,
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
                        "mean_service_ratio": mean_service_ratio,
                        "mean_service_ratio_pct": mean_service_ratio * 100.0,
                        "mean_completed_order_value": mean_completed_order_value,
                        "mean_ev_completed_order_value": mean_ev_completed_order_value,
                        "mean_aev_completed_order_value": mean_aev_completed_order_value,
                        "mean_sample_assign_q_value": mean_sample_assign_q,
                        "mean_sample_assign_q_value_aev": mean_sample_assign_q_aev,
                        "mean_sample_assign_q_value_ev": mean_sample_assign_q_ev,
                        "mean_sample_charge_q_value": mean_sample_charge_q,
                        "mean_sample_idle_q_value": mean_sample_idle_q,
                        "mean_drop_off_rate": mean_drop_off_rate,
                        "episode_reward_aev": mean_episode_reward_aev,
                        "episode_reward_ev": mean_episode_reward_ev,
                        "avg_battery_level": mean_avg_battery_level,
                        "finished_charge": mean_finished_charge,
                        "avg_daily_charging_sessions_per_human_ev": mean_daily_charges_human_ev,
                        "avg_daily_charging_sessions_per_aev": mean_daily_charges_aev,
                        "avg_daily_charging_sessions_per_vehicle": mean_daily_charges_all,
                        "avg_charging_session_duration_minutes_human_ev": mean_charge_duration_human_ev,
                        "avg_charging_session_duration_minutes_aev": mean_charge_duration_aev,
                        "avg_charging_session_duration_minutes_all": mean_charge_duration_all,
                        "charging_wait_penalty_total": mean_charging_wait_penalty,
                        "charging_wait_steps": mean_charging_wait_steps,
                        "avg_charging_wait_time": mean_charging_wait_time,
                        "avg_wait": mean_avg_wait,
                        "waiting_vehicle_count": total_waiting_vehicle_count,
                        "mean_waiting_vehicle_count": mean_waiting_vehicle_count,
                        "mean_station_pressure": mean_station_pressure,
                        "mean_max_station_pressure": mean_max_station_pressure,
                        "max_station_pressure": max_station_pressure,
                        "queue_predictor_holdout_mse": queue_predictor_holdout_mse,
                        "queue_predictor_training_mse_aev": queue_predictor_training_mse_aev,
                        "queue_input_weight_norm_aev": queue_input_weight_norm_aev,
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
                          f"WaitSteps: {mean_charging_wait_steps:.2f}  "
                          f"Charge/veh-day(EV/AEV/All): "
                          f"{mean_daily_charges_human_ev:.3f}/"
                          f"{mean_daily_charges_aev:.3f}/"
                          f"{mean_daily_charges_all:.3f}  "
                          f"ChargeTime(EV/AEV/All min): "
                          f"{mean_charge_duration_human_ev:.1f}/"
                          f"{mean_charge_duration_aev:.1f}/"
                          f"{mean_charge_duration_all:.1f}  "
                          f"MaxPressure: {max_station_pressure:.2f}  "
                          f"DropOff: {mean_drop_off_rate:.4f}")

    # ── 3. Summary table ──
    print("\n" + "=" * 120)
    print("Evaluation Summary")
    print("=" * 120)
    header = f"{'Strategy':<10} {'Mode':<12} {'Demand':<8} {'Seed':<8} {'AvgReward':>10} {'Orders':>8} {'Accept':>8} {'Reject':>8} {'Complete':>10} {'AccRate':>8} {'AvgWait':>9} {'DropOff':>8}"
    print(header)
    print("-" * 120)
    for r in all_results:
        acc_rate = r['accept'] / r['total_orders'] * 100 if r['total_orders'] > 0 else 0
        print(f"{r['strategy']:<10} {r['mode']:<12} {r['demand']:<8} {r['seed']:<8} "
              f"{r['avg_reward']:>10.2f} {r['total_orders']:>8} {r['accept']:>8} {r['reject']:>8} {r['complete']:>10} {acc_rate:>7.1f}% {r['avg_wait']:>9.2f} {r['mean_drop_off_rate']:>8.4f}")

    # Aggregate per (strategy, mode, demand)
    print("\n" + "-" * 120)
    print(f"{'Strategy':<10} {'Mode':<12} {'Demand':<8} {'MeanReward':>12} {'StdReward':>12} {'AvgWait':>10} {'DropOff':>10} {'Seeds':>6}")
    print("-" * 120)
    seen = {}
    for r in all_results:
        key = (r["strategy"], r["mode"], r["demand"])
        seen.setdefault(key, []).append(r["avg_reward"])
    for (s, m, d), rews in seen.items():
        mean_drop_off = np.mean([r['mean_drop_off_rate'] for r in all_results if r['strategy'] == s and r['mode'] == m and r['demand'] == d])
        wait_subset = [r for r in all_results if r['strategy'] == s and r['mode'] == m and r['demand'] == d]
        mean_avg_wait = aggregate_wait_metrics(wait_subset)['avg_wait']
        print(f"{s:<10} {m:<12} {d:<8} {np.mean(rews):>12.2f} {np.std(rews):>12.2f} {mean_avg_wait:>10.2f} {mean_drop_off:>10.4f} {len(rews):>6}")

    # Save raw results
    out_dir = Path("results/test_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution_tag = f"_{zone_distribution_mode}"
    out_path = out_dir / f"test_results_4way{distribution_tag}.npy"
    np.save(out_path, all_results)
    print(f"\n✓ Raw results saved to {out_path}")

    # ── 4. Save Excel with two sheets: detail + summary ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = out_dir / f"test_results_4way{distribution_tag}_{timestamp}.xlsx"

    # Detail sheet — one row per (strategy, mode, demand, seed)
    df_detail = pd.DataFrame(all_results)
    df_detail["acc_rate"] = df_detail.apply(
        lambda r: r["accept"] / r["total_orders"] * 100 if r["total_orders"] > 0 else 0, axis=1
    )
    df_detail["mean_assignment_success_rate_pct"] = df_detail["mean_assignment_success_rate"] * 100.0

    # Summary sheet — aggregate per (strategy, mode, demand)
    summary_rows = []
    for (s, m, d), rews in seen.items():
        subset = [r for r in all_results if r["strategy"] == s and r["mode"] == m and r["demand"] == d]
        wait_summary = aggregate_wait_metrics(subset)
        total_episodes = sum(int(r.get("episodes", 0) or 0) for r in subset)
        summary_rows.append({
            "strategy": s,
            "mode": m,
            "demand": d, 
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
            "mean_service_ratio": np.mean([r["mean_service_ratio"] for r in subset]),
            "mean_service_ratio_pct": np.mean([r["mean_service_ratio_pct"] for r in subset]),
            "mean_completed_order_value": np.mean([r["mean_completed_order_value"] for r in subset]),
            "mean_ev_completed_order_value": np.mean([r["mean_ev_completed_order_value"] for r in subset]),
            "mean_aev_completed_order_value": np.mean([r["mean_aev_completed_order_value"] for r in subset]),
            "mean_sample_assign_q_value": np.mean([r["mean_sample_assign_q_value"] for r in subset]),
            "mean_sample_assign_q_value_aev": np.mean([r["mean_sample_assign_q_value_aev"] for r in subset]),
            "mean_sample_assign_q_value_ev": np.mean([r["mean_sample_assign_q_value_ev"] for r in subset]),
            "mean_sample_charge_q_value": np.mean([r["mean_sample_charge_q_value"] for r in subset]),
            "mean_sample_idle_q_value": np.mean([r["mean_sample_idle_q_value"] for r in subset]),
            "mean_drop_off_rate": np.mean([r["mean_drop_off_rate"] for r in subset]),
            "episode_reward_aev": np.mean([r["episode_reward_aev"] for r in subset]),
            "episode_reward_ev": np.mean([r["episode_reward_ev"] for r in subset]),
            "avg_battery_level": np.mean([r["avg_battery_level"] for r in subset]),
            "finished_charge": np.mean([r["finished_charge"] for r in subset]),
            "avg_daily_charging_sessions_per_human_ev": np.mean([r["avg_daily_charging_sessions_per_human_ev"] for r in subset]),
            "avg_daily_charging_sessions_per_aev": np.mean([r["avg_daily_charging_sessions_per_aev"] for r in subset]),
            "avg_daily_charging_sessions_per_vehicle": np.mean([r["avg_daily_charging_sessions_per_vehicle"] for r in subset]),
            "avg_charging_session_duration_minutes_human_ev": np.mean([r["avg_charging_session_duration_minutes_human_ev"] for r in subset]),
            "avg_charging_session_duration_minutes_aev": np.mean([r["avg_charging_session_duration_minutes_aev"] for r in subset]),
            "avg_charging_session_duration_minutes_all": np.mean([r["avg_charging_session_duration_minutes_all"] for r in subset]),
            "charging_wait_penalty_total": np.mean([r["charging_wait_penalty_total"] for r in subset]),
            "charging_wait_steps": np.mean([r["charging_wait_steps"] for r in subset]),
            "avg_charging_wait_time": np.mean([r["avg_charging_wait_time"] for r in subset]),
            "avg_wait": wait_summary["avg_wait"],
            "waiting_vehicle_count": wait_summary["waiting_vehicle_count"],
            "mean_waiting_vehicle_count": (
                wait_summary["waiting_vehicle_count"] / total_episodes
                if total_episodes else 0.0
            ),
            "mean_station_pressure": np.mean([r["mean_station_pressure"] for r in subset]),
            "mean_max_station_pressure": np.mean([r["mean_max_station_pressure"] for r in subset]),
            "max_station_pressure": np.max([r["max_station_pressure"] for r in subset]),
            "queue_predictor_holdout_mse": np.mean([r["queue_predictor_holdout_mse"] for r in subset]),
            "queue_predictor_training_mse_aev": np.mean([r["queue_predictor_training_mse_aev"] for r in subset]),
            "queue_input_weight_norm_aev": np.mean([r["queue_input_weight_norm_aev"] for r in subset]),
            "num_seeds": len(rews),
        })
    df_summary = pd.DataFrame(summary_rows)

    df_detail = _drop_all_zero_numeric_columns(df_detail)
    df_summary = _drop_all_zero_numeric_columns(df_summary)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_detail.to_excel(writer, sheet_name="detail", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)
    print(f"✓ Excel saved to {excel_path}")


if __name__ == "__main__":
    main()
