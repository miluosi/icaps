"""
Entrypoint to run charging integration training via ADPTrainer.
Mirrors test_integrated_charging main workflow but uses the trainer encapsulation.
"""
import argparse
import os

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run charging integration training with ADPTrainer")
    parser.add_argument("--adp", type=float, default=1.0, help="ADP value (0 disables NN training)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--num-vehicles", type=int, default=200, help="Total vehicles")
    parser.add_argument("--num-ev", type=int, default=100, help="EV vehicles")
    parser.add_argument(
        "--transportation-mode",
        type=str,
        nargs="+",
        default=["integrated", "evfirst", "aevfirst"],
        choices=["integrated", "evfirst", "aevfirst"],
        help="One or more transportation modes",
    )
    parser.add_argument("--use-intense-requests", action="store_true", help="Use intense request pattern")
    parser.add_argument("--no-intense-requests", dest="use_intense_requests", action="store_false", help="Disable intense request pattern")
    parser.set_defaults(use_intense_requests=True)
    parser.add_argument("--assignment-gurobi", action="store_true", help="Use Gurobi assignment")
    parser.add_argument("--assignment-heuristic", dest="assignment_gurobi", action="store_false", help="Use heuristic assignment")
    parser.set_defaults(assignment_gurobi=True)
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--start-training-episode", type=int, default=3, help="Episode index to start NN training")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume synthetic training from an existing scenario checkpoint",
    )
    parser.add_argument(
        "--resume-checkpoint-selection",
        choices=["latest", "best"],
        default="best",
        help="Checkpoint tag used by --resume",
    )
    parser.add_argument("--use-mcmf", action="store_true", dest="usemcmf", help="Use MCMF for routing")
    parser.add_argument("--no-mcmf", action="store_false", dest="usemcmf", help="Do not use MCMF for routing")
    parser.set_defaults(usemcmf=True)
    parser.add_argument("--mcmf-use-gpu", action="store_true", help="Use the existing GPU MCMF solver interface when MCMF is enabled")
    parser.add_argument("--mcmf-use-cpu", dest="mcmf_use_gpu", action="store_false", help="Force the CPU MCMF solver even if GPU kernels are available")
    parser.set_defaults(mcmf_use_gpu=False)
    parser.add_argument(
        "--mcmf-solver", choices=["exact", "legacy", "auction"], default="exact",
        help="MCMF implementation; exact is globally optimal on the configured Q-value grid",
    )
    parser.add_argument(
        "--mcmf-backend", choices=["auto", "ortools", "gurobi_network", "primal_dual"],
        default="gurobi_network", help="Exact MCMF backend (auto tries only exact backends)",
    )
    parser.add_argument(
        "--mcmf-cost-scale",
        type=int,
        default=10_000,
        help="Shared Q-value precision grid; 10000 means four decimal places",
    )
    parser.add_argument("--mcmf-online", dest="mcmf_strict", action="store_false", help="Allow fallback to legacy MCMF if every exact backend fails")
    parser.set_defaults(mcmf_strict=True)
    parser.add_argument("--no-mcmf-graph-reduction", dest="mcmf_graph_reduction", action="store_false")
    parser.set_defaults(mcmf_graph_reduction=True)
    parser.add_argument("--mcmf-verify", action="store_true", help="Run the optional residual optimality certificate")
    parser.add_argument("--use-auction", action="store_true", dest="useauction", help="Use auction solver as the MCMF backend")
    parser.add_argument("--no-auction", action="store_false", dest="useauction", help="Disable auction solver backend")
    parser.set_defaults(useauction=False)
    parser.add_argument("--auction -use-gpu", action="store_true", help="Use GPU auction solver when auction is enabled")
    parser.add_argument("--auction-use-cpu", dest="auction_use_gpu", action="store_false", help="Force CPU auction solver")
    parser.set_defaults(auction_use_gpu=False)
    parser.add_argument("--auction-epsilon", type=float, default=1e-3, help="Auction bidding epsilon")
    parser.add_argument("--auction-max-rounds", type=int, default=None, help="Maximum auction iterations/rounds before falling back to MCMF")
    parser.add_argument("--auction-top-k", type=int, default=None, help="Keep only each vehicle's top-K feasible auction actions before solving")
    parser.add_argument("--known-reject", action="store_true", help="Use known reject feature")
    parser.set_defaults(known_reject=False)
    parser.add_argument("--heuristic-battery-threshold", type=float, default=0.30, help="Battery threshold for heuristic assignment")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE, help="Grid size for the environment (NxN)")
    parser.add_argument("--num-stations", type=int, default=DEFAULT_NUM_STATIONS, help="Number of charging stations")
    parser.add_argument("--station-capacity", type=int, default=DEFAULT_STATION_CAPACITY, help="Charging slots per station")
    parser.add_argument("--station-queue-capacity", type=int, default=DEFAULT_STATION_QUEUE_CAPACITY, help="Maximum AEV waiting-room admissions per station")
    parser.add_argument("--charge-duration", type=int, default=DEFAULT_CHARGE_DURATION, help="Steps required for a full charge")
    parser.add_argument("--aev-initial-battery-scale", type=float, default=DEFAULT_AEV_INITIAL_BATTERY_SCALE, help="Scale predictive initial AEV state of charge")
    parser.add_argument("--critical-charging-battery", type=float, default=DEFAULT_CRITICAL_CHARGING_BATTERY, help="AEV SoC below which waiting is infeasible")
    parser.add_argument("--random-seed", type=int, default=64, help="Random seed for training")
    parser.add_argument(
        "--distribution-mode",
        type=str,
        default="st_masac_gat_queue_demand_gurobi",
        choices=[
            "bayes", "time-only", "none",
            "st_masac_gat_post_demand_direct",
            "st_masac_gat_queue_demand_gurobi",
        ],
        help="ICAPS core value-function mode",
    )
    parser.add_argument("--post-demand-q-weight", type=float, default=0.0, help="Initial action-head weight for direct post-demand MASAC")
    parser.add_argument("--post-demand-head-lr-multiplier", type=float, default=10.0, help="Learning-rate multiplier for the direct post-demand action head")
    parser.add_argument(
        "--masac-target-entropy-ratio",
        type=float,
        default=0.9,
        help=(
            "Target entropy as a fraction of log(candidate count) for "
            "masac_baseline and standard_masac_gat"
        ),
    )
    parser.add_argument("--simulation-period", type=int, default=DEFAULT_SIMULATION_PERIOD, help="Simulation steps per virtual day")
    parser.add_argument("--episode-days", type=int, default=DEFAULT_EPISODE_DAYS, help="Virtual days per episode")
    parser.add_argument("--charging-wait-penalty-per-step", type=float, default=DEFAULT_WAIT_PENALTY_PER_STEP, help="Environment reward penalty for each step spent waiting for a charger")
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
    parser.add_argument("--all-modes", action="store_true", help="Run all transportation modes instead of only the specified mode")
    parser.add_argument("--all-demand-patterns", action="store_true", help="Run both intense and random demand instead of only the specified demand setting")
    parser.add_argument("--daily-drop-off", action="store_true", help="Update satisfaction, dropout, and rejoin at the start of each day")
    parser.set_defaults(daily_drop_off=False)
    parser.add_argument(
        "--fixed-training-vehicle-states",
        dest="randomize_training_vehicle_states",
        action="store_false",
        help="Reuse one vehicle position/battery initialization in every training episode",
    )
    parser.set_defaults(randomize_training_vehicle_states=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.useauction:
        args.usemcmf = True
    zone_distribution_mode = args.distribution_mode or "none"
    if zone_distribution_mode == "elbo":
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

    print("🚗⚡ ADPTrainer charging integration run")
    print(f"ADP={args.adp}, episodes={args.episodes}, vehicles={args.num_vehicles}, ev={args.num_ev}")
    print(f"Mode={args.transportation_mode}, use_intense_requests={args.use_intense_requests}, assignment_gurobi={args.assignment_gurobi}")
    print(f"known_reject={args.known_reject}, heuristic_battery_threshold={args.heuristic_battery_threshold}, grid_size={args.grid_size}, stations={args.num_stations}, station_capacity={args.station_capacity}, charge_duration={args.charge_duration}, seed={args.random_seed}")
    print(f"distribution_mode={zone_distribution_mode}")
    if zone_distribution_mode in {
        "masac_baseline",
        "standard_masac_gat",
        "standard_masac_gat_total_q",
    }:
        print(
            "masac_target_entropy_ratio="
            f"{args.masac_target_entropy_ratio:g}"
        )
    print(f"episode_days={args.episode_days}, simulation_period={args.simulation_period}, charging_wait_penalty_per_step={args.charging_wait_penalty_per_step}")
    print(f"synthetic_demand_profile={args.synthetic_demand_profile}")
    checkpoint_scenario_suffix = synthetic_checkpoint_suffix(
        num_stations=args.num_stations,
        station_capacity=args.station_capacity,
        charge_duration=args.charge_duration,
        simulation_period=args.simulation_period,
        episode_days=args.episode_days,
        charging_wait_penalty_per_step=args.charging_wait_penalty_per_step,
        station_queue_capacity=args.station_queue_capacity,
        aev_initial_battery_scale=args.aev_initial_battery_scale,
        critical_charging_battery=args.critical_charging_battery,
        grid_size=args.grid_size,
        synthetic_demand_profile=args.synthetic_demand_profile,
        synthetic_demand_scale=args.synthetic_demand_scale,
    )
    print(f"checkpoint_scenario_suffix={checkpoint_scenario_suffix or '<legacy>'}")
    print(
        f"mcmf_solver={'auction' if args.useauction else args.mcmf_solver}, "
        f"backend={args.mcmf_backend}, strict={args.mcmf_strict}, "
        f"cost_scale={args.mcmf_cost_scale}"
    )
    if args.useauction:
        print(f"auction_solver={'GPU' if args.auction_use_gpu else 'CPU'}, epsilon={args.auction_epsilon}, max_rounds={args.auction_max_rounds}, top_k={args.auction_top_k}")

    trainer = ADPTrainer()
    demand_pattern_list = [False,True] if args.all_demand_patterns else [args.use_intense_requests]
    transportation_mode_list = ["evfirst", "integrated", "aevfirst"] if args.all_modes else list(dict.fromkeys(args.transportation_mode))
    for demand_pattern in demand_pattern_list:
        for mode in transportation_mode_list:
            print(f"\n--- Starting training: demand_pattern={demand_pattern}, mode={mode} ---")
            results, env = trainer.run_charging_integration_test(
                adpvalue=args.adp,
                num_episodes=args.episodes,
                use_intense_requests=demand_pattern,
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
                ifloadgingValueFunction=args.resume,
                trainnetwork=True,
                random_seed=args.random_seed,
                grid_size=args.grid_size,
                num_stations=args.num_stations,
                station_capacity=args.station_capacity,
                station_queue_capacity=args.station_queue_capacity,
                charge_duration=args.charge_duration,
                aev_initial_battery_scale=args.aev_initial_battery_scale,
                critical_charging_battery=args.critical_charging_battery,
                zone_distribution_mode=zone_distribution_mode,
                post_demand_q_weight=args.post_demand_q_weight,
                post_demand_head_lr_multiplier=args.post_demand_head_lr_multiplier,
                masac_target_entropy_ratio=args.masac_target_entropy_ratio,
                daily_drop_off=args.daily_drop_off,
                simulation_period=args.simulation_period,
                episode_length=(
                    args.simulation_period * args.episode_days
                    if args.episode_days is not None else None
                ),
                charging_wait_penalty_per_step=args.charging_wait_penalty_per_step,
                synthetic_demand_profile=args.synthetic_demand_profile,
                synthetic_demand_scale=args.synthetic_demand_scale,
                randomize_training_vehicle_states=args.randomize_training_vehicle_states,
                checkpoint_selection=(
                    args.resume_checkpoint_selection if args.resume else None
                ),
                checkpoint_scenario_suffix=checkpoint_scenario_suffix,
            )

            # Summary output
            print("\n✅ Training finished")
            print(f"Episodes: {len(results.get('episode_rewards', []))}")
            if results.get("episode_rewards"):
                print(f"Avg reward: {sum(results['episode_rewards'])/len(results['episode_rewards']):.2f}")
                window = min(10, max(1, len(results['episode_rewards']) // 2))
                first_reward = sum(results['episode_rewards'][:window]) / window
                last_reward = sum(results['episode_rewards'][-window:]) / window
                print(f"Reward trend (first {window} -> last {window}): {first_reward:.2f} -> {last_reward:.2f} ({last_reward - first_reward:+.2f})")
            if results.get("value_function_losses"):
                print(f"Avg loss (AEV): {sum(results['value_function_losses'])/len(results['value_function_losses']):.4f}")
            if results.get("value_function_ev_losses"):
                print(f"Avg loss (EV): {sum(results['value_function_ev_losses'])/len(results['value_function_ev_losses']):.4f}")
            detailed = results.get("episode_detailed_stats", [])
            if detailed:
                wait_summary = aggregate_wait_metrics(detailed)
                print(
                    "Avg wait among waiting vehicles: "
                    f"{wait_summary['avg_wait']:.2f} steps"
                )
                print(
                    "Avg request outcomes: "
                    f"rejected={sum(row.get('rejected_requests', 0) for row in detailed) / len(detailed):.2f}, "
                    f"recourse={sum(row.get('recourse_requests', 0) for row in detailed) / len(detailed):.2f}, "
                    f"lost={sum(row.get('lost_requests', 0) for row in detailed) / len(detailed):.2f}"
                )
            if results.get("excel_path"):
                print(f"Stats saved to: {results['excel_path']}")
            if results.get("spatial_image_path"):
                print(f"Spatial plot saved to: {results['spatial_image_path']}")


if __name__ == "__main__":
    main()
