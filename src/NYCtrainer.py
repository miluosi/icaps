import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.ADPtrainer import ADPTrainer
from src.GurobiOptimizer import GurobiOptimizer
from src.charging_wait_metrics import aggregate_wait_metrics


class NYCTrainer:
    """NYC-specific trainer wrapper, mirroring ADPTrainer-style encapsulation."""

    def __init__(
        self,
        *,
        create_environment: Callable[..., Any],
        resolve_parquet_paths: Callable[..., list[str]],
        get_value_function_class: Callable[[str], type],
        set_random_seeds: Callable[[int], None],
        trainer_helper: ADPTrainer | None = None,
    ):
        self._create_environment = create_environment
        self._resolve_parquet_paths = resolve_parquet_paths
        self._get_value_function_class = get_value_function_class
        self._set_random_seeds = set_random_seeds
        self._trainer_helper = trainer_helper or ADPTrainer()

    @staticmethod
    def _select_motion_fn(env: Any, transportation_mode: str) -> Callable[..., tuple[Any, Any, Any]]:
        if transportation_mode == "integrated":
            return env.simulate_motion
        if transportation_mode == "evfirst":
            return env.simulate_motion_evfirst
        if transportation_mode == "aevfirst":
            return env.simulate_motion_aevfirst
        raise ValueError(f"Unsupported transportation mode: {transportation_mode}")

    @staticmethod
    def _checkpoint_dirs(
        transportation_mode: str,
        assignmentgurobi: bool,
        num_ev: int,
        use_intense_requests: bool,
        start_date: str | None = None,
        end_date: str | None = None,
        zone_distribution_mode: str = "none",
        only_manhattan_zones: bool = False,
        full_demand: bool = False,
        checkpoint_suffix: str | None = None,
    ) -> tuple[str, str]:
        assign_tag = "gurobi" if assignmentgurobi else "heu"
        start_date = (start_date or "unknown_start").replace("-", "")
        end_date = (end_date or start_date).replace("-", "")
        date_suffix = f"_{start_date}_{end_date}"
        distribution_suffix = NYCTrainer._distribution_suffix(zone_distribution_mode)
        zone_scope_suffix = "_manhattan" if only_manhattan_zones else ""
        demand_suffix = "_fulldemand" if full_demand else ""
        extra_suffix = ""
        if checkpoint_suffix:
            clean_suffix = "".join(
                ch if ch.isalnum() or ch in {"_", "-"} else "_"
                for ch in str(checkpoint_suffix).strip()
            ).strip("_")
            if clean_suffix:
                extra_suffix = f"_{clean_suffix}"
        if transportation_mode == "integrated":
            return (
                f"checkpoints/q_networks_nyc_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}{date_suffix}{distribution_suffix}{zone_scope_suffix}{demand_suffix}{extra_suffix}_ev",
                f"checkpoints/q_networks_nyc_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}{date_suffix}{distribution_suffix}{zone_scope_suffix}{demand_suffix}{extra_suffix}_aev",
            )
        if transportation_mode == "evfirst":
            return (
                f"checkpoints/q_networksevfirst_nyc_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}{date_suffix}{distribution_suffix}{zone_scope_suffix}{demand_suffix}{extra_suffix}_ev",
                f"checkpoints/q_networksevfirst_nyc_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}{date_suffix}{distribution_suffix}{zone_scope_suffix}{demand_suffix}{extra_suffix}_aev",
            )
        if transportation_mode == "aevfirst":
            return (
                f"checkpoints/q_networksaevfirst_nyc_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}{date_suffix}{distribution_suffix}{zone_scope_suffix}{demand_suffix}{extra_suffix}_ev",
                f"checkpoints/q_networksaevfirst_nyc_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}{date_suffix}{distribution_suffix}{zone_scope_suffix}{demand_suffix}{extra_suffix}_aev",
            )
        raise ValueError(f"Unsupported transportation mode: {transportation_mode}")

    @staticmethod
    def _resolve_zone_distribution_mode(zone_distribution_mode: str | None, encoder: bool = False) -> str:
        return zone_distribution_mode or ("bayes" if encoder else "none")

    @classmethod
    def _distribution_suffix(cls, zone_distribution_mode: str | None, encoder: bool = False) -> str:
        effective_zone_distribution_mode = cls._resolve_zone_distribution_mode(zone_distribution_mode, encoder)
        if effective_zone_distribution_mode == "bayes":
            return "_bayes"
        if effective_zone_distribution_mode == "bayes_simple":
            return "_bayes_simple"
        if effective_zone_distribution_mode == "bayes_simple_pretrain":
            return "_bayes_simple_pretrain"
        if effective_zone_distribution_mode == "neuradp":
            return "_neuradp"
        if effective_zone_distribution_mode == "adp_critic":
            return "_adp_critic"
        if effective_zone_distribution_mode == "sac":
            return "_sac"
        if effective_zone_distribution_mode == "masac_baseline":
            return "_masac_baseline"
        if effective_zone_distribution_mode == "st_masac_gat":
            return "_st_masac_gat"
        if effective_zone_distribution_mode == "st_masac_gat_post_demand":
            return "_st_masac_gat_post_demand"
        if effective_zone_distribution_mode == "st_masac_gat_post_demand_direct":
            return "_st_masac_gat_post_demand_direct"
        if effective_zone_distribution_mode == "standard_masac_gat":
            return "_standard_masac_gat"
        if effective_zone_distribution_mode == "standard_masac_gat_total_q":
            return "_standard_masac_gat_total_q"
        if effective_zone_distribution_mode == "optimization_anchored_residual":
            return "_optimization_anchored_residual"
        if effective_zone_distribution_mode == "standard_masac_gat_greedy_alpha":
            return "_standard_masac_gat_greedy_alpha"
        if effective_zone_distribution_mode == "standard_masac_gat_fixed_alpha":
            return "_standard_masac_gat_fixed_alpha"
        if effective_zone_distribution_mode == "st_masac_gat_former2":
            return "_st_masac_gat_former2"
        if effective_zone_distribution_mode == "st_masac_gat_former2_queue_feature":
            return "_st_masac_gat_former2_queue_feature"
        if effective_zone_distribution_mode == "st_masac_gat_former2_queue_feature_greedy_alpha":
            return "_st_masac_gat_former2_queue_feature_greedy_alpha"
        if effective_zone_distribution_mode == "st_masac_gat_former2_queue_feature_fixed_alpha":
            return "_st_masac_gat_former2_queue_feature_fixed_alpha"
        if effective_zone_distribution_mode == "st_masac_gat_frozen":
            return "_st_masac_gat_frozen"
        if effective_zone_distribution_mode == "st_masac_gat_neighbour_frozen":
            return "_st_masac_gat_neighbour_frozen"
        if effective_zone_distribution_mode == "none":
            return "_none"
        return "_noenc"

    @staticmethod
    def _temporary_heuristic_warmup(env: Any, enabled: bool) -> tuple[bool, bool, float] | None:
        if not enabled:
            return None
        state = (bool(env.usemcmf), bool(env.assignmentgurobi), float(env.adp_value))
        env.usemcmf = False
        env.assignmentgurobi = False
        env.adp_value = 0.0
        return state

    @staticmethod
    def _restore_solver_state(env: Any, state: tuple[bool, bool, float] | None) -> None:
        if state is None:
            return
        env.usemcmf, env.assignmentgurobi, env.adp_value = state

    @staticmethod
    def _configure_pretrained_zone_distributors(
        *,
        value_function: Any,
        value_function_ev: Any,
        env: Any,
        transportation_mode: str,
        pretrained_zone_dir: str,
        only_manhattan_zones: bool,
        full_demand: bool = False,
    ) -> None:
        scope = "manhattan" if only_manhattan_zones else "full_nyc"
        if full_demand:
            scope += "_fulldemand"
        root = Path(pretrained_zone_dir).expanduser() / scope
        required = []
        optional = []
        if transportation_mode == "evfirst":
            required = [
                (value_function_ev, "ev", "leader"),
                (value_function, "aev", "follower"),
            ]
        elif transportation_mode == "aevfirst":
            required = [(value_function_ev, "ev", "follower")]
            optional = [(value_function, "aev", "leader")]
        else:
            raise ValueError("bayes_simple_pretrain supports only evfirst and aevfirst")

        load_specs = list(required)
        for spec in optional:
            _, vehicle_type, role = spec
            checkpoint_path = root / f"{transportation_mode}_{vehicle_type}_{role}.pth"
            if checkpoint_path.exists():
                load_specs.append(spec)
            else:
                print(
                    f"ℹ No pretrained zone distributor for mode={transportation_mode} "
                    f"vehicle={vehicle_type} role={role}; Q zone input for that role will be zero."
                )

        active = {(id(vf), role) for vf, _, role in load_specs}
        for vf in (value_function, value_function_ev):
            vf.pretrained_zone_distribution = True
            for role in ("leader", "follower"):
                predictor_name = f"time_zone_dist_predictor_{role}"
                optimizer_name = f"time_zone_dist_optimizer_{role}"
                predictor = getattr(vf, predictor_name, None)
                if (id(vf), role) not in active:
                    setattr(vf, predictor_name, None)
                    setattr(vf, optimizer_name, None)
                    setattr(vf, f"zone_dist_predictor_{role}", None)
                    setattr(vf, f"zone_dist_optimizer_{role}", None)
                elif predictor is not None:
                    for parameter in predictor.parameters():
                        parameter.requires_grad = False

        expected_zone_ids = [int(zone_id) for zone_id in getattr(env, "aux_zone_ids", [])]
        for vf, vehicle_type, role in load_specs:
            checkpoint_path = root / f"{transportation_mode}_{vehicle_type}_{role}.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing pretrained zone distributor: {checkpoint_path}. "
                    "Run pretrain_zone.py with matching scope and transportation mode first."
                )
            payload = torch.load(checkpoint_path, map_location=vf.device)
            checkpoint_zone_ids = [int(zone_id) for zone_id in payload.get("zone_ids", [])]
            if checkpoint_zone_ids != expected_zone_ids:
                raise ValueError(
                    f"Zone support mismatch for {checkpoint_path}: "
                    f"checkpoint has {len(checkpoint_zone_ids)} zones, environment has {len(expected_zone_ids)}"
                )
            if payload.get("transportation_mode") != transportation_mode:
                raise ValueError(f"Transportation mode mismatch in {checkpoint_path}")
            if payload.get("vehicle_type") != vehicle_type or payload.get("role") != role:
                raise ValueError(f"Vehicle type/role mismatch in {checkpoint_path}")
            predictor = getattr(vf, f"time_zone_dist_predictor_{role}", None)
            if predictor is None:
                raise RuntimeError(f"Missing {role} zone predictor for {vehicle_type}")
            checkpoint_time_bins = int(payload.get("num_time_bins", -1))
            predictor_time_bins = int(getattr(predictor, "num_time_bins", -1))
            if checkpoint_time_bins != predictor_time_bins:
                raise ValueError(
                    f"Time-bin mismatch for {checkpoint_path}: checkpoint={checkpoint_time_bins}, "
                    f"model={predictor_time_bins}"
                )
            load_result = predictor.load_state_dict(payload["state_dict"], strict=False)
            missing_keys = list(getattr(load_result, "missing_keys", []))
            unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
            if missing_keys or unexpected_keys:
                print(
                    f"WARNING: Zone distributor checkpoint loaded with key differences: "
                    f"missing={missing_keys}, unexpected={unexpected_keys}"
                )
            predictor.eval()
            for parameter in predictor.parameters():
                parameter.requires_grad = False
            setattr(vf, f"time_zone_dist_optimizer_{role}", None)
            setattr(vf, f"zone_dist_optimizer_{role}", None)
            vf.time_zone_dist_predictor = predictor
            vf.zone_dist_predictor = predictor
            vf.time_zone_dist_optimizer = None
            vf.zone_dist_optimizer = None
            print(
                f"✓ Loaded frozen zone distributor: mode={transportation_mode} "
                f"vehicle={vehicle_type} role={role} file={checkpoint_path}"
            )
        for vf in (value_function, value_function_ev):
            active_roles = [role for candidate_vf, _, role in load_specs if candidate_vf is vf]
            if active_roles:
                primary_role = active_roles[0]
                vf.time_zone_dist_predictor = getattr(vf, f"time_zone_dist_predictor_{primary_role}", None)
                vf.zone_dist_predictor = getattr(vf, f"zone_dist_predictor_{primary_role}", None)
            else:
                vf.time_zone_dist_predictor = None
                vf.zone_dist_predictor = None
            vf.time_zone_dist_optimizer = None
            vf.zone_dist_optimizer = None

    @staticmethod
    def _write_lines(log_path: Path, lines: list[str]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _loss_output_dir() -> Path:
        output_dir = Path("Npyloss")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _loss_series(value_function: Any) -> dict[str, np.ndarray]:
        if value_function is None:
            return {}
        return {
            "training_losses": np.asarray(getattr(value_function, "training_losses", []), dtype=np.float32),
            "queue_training_losses": np.asarray(getattr(value_function, "queue_training_losses", []), dtype=np.float32),
            "queue_training_mse_losses": np.asarray(
                getattr(value_function, "queue_training_mse_losses", []),
                dtype=np.float32,
            ),
            "post_demand_training_losses": np.asarray(
                getattr(value_function, "post_demand_training_losses", []),
                dtype=np.float32,
            ),
            "post_demand_training_mse_losses": np.asarray(
                getattr(value_function, "post_demand_training_mse_losses", []),
                dtype=np.float32,
            ),
            "post_demand_weight_history": np.asarray(
                getattr(value_function, "post_demand_weight_history", []),
                dtype=object,
            ),
            "normalized_td_losses": np.asarray(
                getattr(value_function, "normalized_td_losses", []),
                dtype=np.float32,
            ),
            "td_error_history": np.asarray(
                getattr(value_function, "td_error_history", []),
                dtype=object,
            ),
            "q_values_history": np.asarray(
                getattr(value_function, "q_values_history", []),
                dtype=object,
            ),
            "rejection_training_losses": np.asarray(
                getattr(value_function, "rejection_training_losses", []),
                dtype=np.float32,
            ),
        }

    @staticmethod
    def _latest_metric(value_function: Any, attr_name: str) -> float:
        values = getattr(value_function, attr_name, None)
        if not values:
            return 0.0
        try:
            value = float(values[-1])
        except (TypeError, ValueError, IndexError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _zone_kl_loss_output_dir() -> Path:
        output_dir = Path("zone_kl_loss")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _zone_kl_loss_array(value_function: Any, role: str | None = None) -> np.ndarray:
        if value_function is None:
            return np.asarray([], dtype=np.float32)

        if role is not None:
            role_losses = getattr(value_function, "zone_dist_role_losses", None)
            if isinstance(role_losses, dict) and role_losses.get(role):
                return np.asarray(role_losses[role], dtype=np.float32)

        for attr_name in ("zone_dist_losses", "zone_dist_kl_losses"):
            values = getattr(value_function, attr_name, None)
            if values:
                return np.asarray(values, dtype=np.float32)

        return np.asarray([], dtype=np.float32)

    @staticmethod
    def _zone_kl_loss_targets(
        transportation_mode: str,
        value_function: Any,
        value_function_ev: Any,
    ) -> list[tuple[str, Any, str | None]]:
        if transportation_mode == "evfirst":
            return [
                ("ev_leader", value_function_ev, "leader"),
                ("aev_follower", value_function, "follower"),
            ]
        if transportation_mode == "aevfirst":
            return [
                ("ev_follower", value_function_ev, "follower"),
            ]
        return [
            ("aev", value_function, None),
            ("ev", value_function_ev, None),
        ]

    def _save_zone_kl_loss_arrays(
        self,
        *,
        transportation_mode: str,
        zone_distribution_mode: str,
        global_step: int,
        value_function: Any,
        value_function_ev: Any,
    ) -> None:
        if zone_distribution_mode == "none":
            return

        output_dir = self._zone_kl_loss_output_dir()
        base_name = f"nyc_{transportation_mode}_{zone_distribution_mode}_step{int(global_step):06d}"
        for target_name, target_value_function, target_role in self._zone_kl_loss_targets(
            transportation_mode,
            value_function,
            value_function_ev,
        ):
            losses = self._zone_kl_loss_array(target_value_function, target_role)
            np.save(output_dir / f"{base_name}_{target_name}_zone_kl.npy", losses)
            np.save(
                output_dir / f"nyc_{transportation_mode}_{zone_distribution_mode}_{target_name}_zone_kl_latest.npy",
                losses,
            )

    def _save_loss_snapshot(
        self,
        *,
        transportation_mode: str,
        zone_distribution_mode: str,
        global_step: int,
        value_function: Any,
        value_function_ev: Any,
    ) -> None:
        output_dir = self._loss_output_dir()
        base_name = f"nyc_{transportation_mode}_{zone_distribution_mode}_step{int(global_step):06d}"
        payloads = {
            f"{base_name}_aev.npy": self._loss_series(value_function),
            f"{base_name}_ev.npy": self._loss_series(value_function_ev),
        }
        for file_name, payload in payloads.items():
            np.save(output_dir / file_name, payload, allow_pickle=True)

        latest_payload = {
            "global_step": int(global_step),
            "transportation_mode": transportation_mode,
            "zone_distribution_mode": zone_distribution_mode,
            "aev": self._loss_series(value_function),
            "ev": self._loss_series(value_function_ev),
        }
        np.save(output_dir / f"nyc_{transportation_mode}_{zone_distribution_mode}_latest.npy", latest_payload, allow_pickle=True)

    @staticmethod
    def _count_solver_graph_edges(
        vehicle_action_matrix: np.ndarray,
        num_requests: int,
        num_charging: int,
        station_vacancies: list[int],
        ev_flags: list[bool],
    ) -> tuple[int, int]:
        num_vehicles = int(vehicle_action_matrix.shape[0])
        num_actions = int(vehicle_action_matrix.shape[1])
        edges = num_vehicles
        action_has_incoming = np.zeros(num_actions, dtype=bool)
        for i in range(num_vehicles):
            for a in range(num_actions):
                if vehicle_action_matrix[i, a] == 0:
                    continue
                if ev_flags[i] and a >= num_requests:
                    continue
                edges += 1
                action_has_incoming[a] = True
        for a in range(num_actions):
            if not action_has_incoming[a]:
                continue
            if a < num_requests:
                edges += 1
            elif a < num_requests + num_charging:
                station_idx = a - num_requests
                if station_idx < len(station_vacancies) and station_vacancies[station_idx] > 0:
                    edges += 1
            else:
                edges += 1
        return 2 + num_vehicles + num_actions, edges

    def run_nyc_solver_benchmark(
        self,
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
        self._set_random_seeds(random_seed)
        parquet_paths = self._resolve_parquet_paths(
            parquet_path,
            start_year_month,
            end_year_month,
            num_episodes,
            full_demand=full_demand,
            hvfhv_parquet_path=hvfhv_parquet_path,
        )
        env = self._create_environment(
            num_vehicles=num_vehicles,
            num_ev=num_ev,
            parquet_paths=parquet_paths,
            full_demand=full_demand,
            coord_csv=coord_csv,
            station_csv=station_csv,
            start_date=start_date,
            end_date=end_date,
            station_capacity_scale=station_capacity_scale,
            epoch_length=epoch_length,
            start_hour=start_hour,
            stop_hour=stop_hour,
            heuristic_battery_threshold=heuristic_battery_threshold,
            use_intense_requests=use_intense_requests,
            assignmentgurobi=True,
            usemcmf=True,
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
            knownreject=knownreject,
            random_seed=random_seed,
            daily_drop_off=daily_drop_off,
            ifreject=ifreject,
            ifdropoff=ifdropoff,
            only_manhattan_zones=only_manhattan_zones,
        )
        env.mcmf_use_gpu = bool(mcmf_use_gpu)
        env.use_cuda_ssp = bool(mcmf_use_gpu)
        env.useauction = bool(useauction or getattr(env, 'ifsolveauctioncuda', False))
        env.mcmf_solver = "auction" if env.useauction else mcmf_solver
        env.mcmf_backend = mcmf_backend
        env.mcmf_strict = bool(mcmf_strict)
        env.mcmf_cost_scale = int(mcmf_cost_scale)
        env.mcmf_graph_reduction = bool(mcmf_graph_reduction)
        env.mcmf_verify = bool(mcmf_verify)
        env.auction_use_gpu = bool(auction_use_gpu)
        env.auction_epsilon = float(auction_epsilon)
        env.auction_max_rounds = auction_max_rounds
        env.auction_top_k = auction_top_k
        env.adp_value = 0.0
        env.value_function = None
        env.value_function_ev = None
        env.evaluatemode = True

        log_lines: list[str] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        solver_tag = (
            "auction" if useauction else
            f"mcmf_{mcmf_solver or 'legacy'}_{mcmf_backend}"
        )
        log_path = Path(log_dir) / f"nyc_solver_benchmark_{num_vehicles}_{solver_tag}_{timestamp}.log"

        def emit(message: str) -> None:
            print(message)
            log_lines.append(message)

        emit("NYC Manhattan Solver Benchmark")
        emit(f"  vehicles={num_vehicles}, ev={num_ev}, warmup_steps={warmup_steps}, benchmark_steps={benchmark_steps}")
        emit(f"  demand={'yellow+hvfhv_nonshared' if full_demand else 'yellow_only'}")
        emit(f"  parquet_files={len(parquet_paths)}")
        for resolved_path in parquet_paths:
            emit(f"    - {resolved_path}")
        emit(f"  relocation_targets={len(env.relocation_target_ids)}, stations={len(env.charging_manager.stations)}")

        env.reset()
        for _ in range(max(0, warmup_steps)):
            env.generate_requests()
            env.current_time += 1

        benchmark_rows: list[dict] = []
        opt = GurobiOptimizer(env)
        for bench_step in range(max(1, benchmark_steps)):
            if bench_step > 0:
                env.generate_requests()
                env.current_time += 1
            active_requests_snapshot = copy.deepcopy(env.active_requests)
            vehicles_to_rebalance = env._build_vehicles_to_rebalance(list(env.vehicles.keys()))
            if not vehicles_to_rebalance:
                vehicles_to_rebalance = list(env.vehicles.keys())
            available_requests = [req for req in active_requests_snapshot.values()]

            matrix_start = time.time()
            vehicle_action_matrix, num_requests, num_stations, num_zones = env.generate_whole_matrix(
                vehicles_to_rebalance,
                rebalance_num=len(vehicles_to_rebalance),
                onlyev=False,
            )
            matrix_time = time.time() - matrix_start

            qvalue_start = time.time()
            batch_q_value = env.generate_vehicle_qvalue_withoutqnetwork(vehicles_to_rebalance)
            qvalue_time = time.time() - qvalue_start

            station_list = list(env.charging_manager.stations.values())
            station_vacancies = [max(0, station.max_capacity - len(station.current_vehicles)) for station in station_list]
            ev_flags = [env.vehicles[vid]["type"] == 1 for vid in vehicles_to_rebalance]
            graph_nodes, graph_edges = self._count_solver_graph_edges(
                vehicle_action_matrix,
                num_requests,
                num_stations,
                station_vacancies,
                ev_flags,
            )

            original_solver = getattr(env, "mcmf_solver", None)
            original_useauction = getattr(env, "useauction", False)
            env.useauction = False
            try:
                env.mcmf_solver = "legacy"
                legacy_mcmf_start = time.time()
                legacy_mcmf_assignments = opt._np_vehicle_rebalancing_network(
                    vehicles_to_rebalance,
                    available_requests,
                    vehicle_action_matrix,
                    batch_q_value,
                    iflp=True,
                )
                legacy_mcmf_time = time.time() - legacy_mcmf_start

                env.mcmf_solver = "exact"
                exact_mcmf_start = time.time()
                exact_mcmf_assignments = opt._np_vehicle_rebalancing_network(
                    vehicles_to_rebalance,
                    available_requests,
                    vehicle_action_matrix,
                    batch_q_value,
                    iflp=True,
                )
                exact_mcmf_time = time.time() - exact_mcmf_start
                exact_stats = dict(getattr(env, "mcmf_last_result", {}))
            finally:
                env.mcmf_solver = original_solver
                env.useauction = original_useauction

            configured_exact = str(original_solver or "legacy").lower() in {
                "exact", "auto", "ortools", "gurobi_network", "primal_dual"
            }
            mcmf_assignments = (
                exact_mcmf_assignments if configured_exact
                else legacy_mcmf_assignments
            )
            mcmf_time = exact_mcmf_time if configured_exact else legacy_mcmf_time

            gurobi_start = time.time()
            gurobi_assignments = opt._gurobi_vehicle_rebalancing_network(
                vehicles_to_rebalance,
                available_requests,
                vehicle_action_matrix,
                batch_q_value,
                iflp=True,
            )
            gurobi_time = time.time() - gurobi_start

            row = {
                "benchmark_step": bench_step,
                "epoch_time": float(env.current_time),
                "vehicles_to_rebalance": len(vehicles_to_rebalance),
                "available_requests": len(available_requests),
                "matrix_rows": int(vehicle_action_matrix.shape[0]),
                "matrix_cols": int(vehicle_action_matrix.shape[1]),
                "request_cols": int(num_requests),
                "charging_cols": int(num_stations),
                "zone_cols": int(num_zones),
                "wait_cols": int(vehicle_action_matrix.shape[1] - num_requests - num_stations - num_zones),
                "request_edges": int(np.sum(vehicle_action_matrix[:, :num_requests])) if num_requests > 0 else 0,
                "charging_edges": int(np.sum(vehicle_action_matrix[:, num_requests:num_requests + num_stations])) if num_stations > 0 else 0,
                "zone_edges": int(np.sum(vehicle_action_matrix[:, num_requests + num_stations:num_requests + num_stations + num_zones])) if num_zones > 0 else 0,
                "total_edges": int(np.sum(vehicle_action_matrix)),
                "mcmf_graph_nodes": int(graph_nodes),
                "mcmf_graph_edges": int(graph_edges),
                "matrix_time_sec": float(matrix_time),
                "qvalue_time_sec": float(qvalue_time),
                "mcmf_time_sec": float(mcmf_time),
                "legacy_mcmf_time_sec": float(legacy_mcmf_time),
                "exact_mcmf_time_sec": float(exact_mcmf_time),
                "exact_mcmf_backend": exact_stats.get("backend", ""),
                "exact_original_edges": exact_stats.get("original_edges", 0),
                "exact_reduced_edges": exact_stats.get("reduced_edges", 0),
                "exact_objective_int": exact_stats.get("objective_int", 0),
                "exact_objective_q": exact_stats.get("objective_q", 0.0),
                "exact_objective_mode": exact_stats.get("objective_mode", ""),
                "qvalue_scale": exact_stats.get("qvalue_scale", 0),
                "qvalue_entries": exact_stats.get("qvalue_entries", 0),
                "qvalue_rounded_entries": exact_stats.get("qvalue_rounded_entries", 0),
                "qvalue_rounding_max_abs": exact_stats.get("qvalue_rounding_max_abs", 0.0),
                "exact_optimal": exact_stats.get("optimal", False),
                "gurobi_time_sec": float(gurobi_time),
                "mcmf_assignments": int(len(mcmf_assignments)),
                "legacy_mcmf_assignments": int(len(legacy_mcmf_assignments)),
                "exact_mcmf_assignments": int(len(exact_mcmf_assignments)),
                "gurobi_assignments": int(len(gurobi_assignments)),
            }
            benchmark_rows.append(row)
            emit(
                f"Step {bench_step}: epoch={int(env.current_time)} vehicles={row['vehicles_to_rebalance']} requests={row['available_requests']} "
                f"matrix={row['matrix_rows']}x{row['matrix_cols']} blocks=req:{row['request_cols']}/chg:{row['charging_cols']}/zone:{row['zone_cols']}/wait:{row['wait_cols']} "
                f"edges=req:{row['request_edges']}/chg:{row['charging_edges']}/zone:{row['zone_edges']}/total:{row['total_edges']} "
                f"qmatrix={row['matrix_time_sec']:.3f}s qvalue={row['qvalue_time_sec']:.3f}s "
                f"legacy_mcmf={row['legacy_mcmf_time_sec']:.3f}s "
                f"exact_mcmf={row['exact_mcmf_time_sec']:.3f}s "
                f"exact_edges={row['exact_original_edges']}->{row['exact_reduced_edges']} "
                f"gurobi={row['gurobi_time_sec']:.3f}s"
            )

        summary = {
            "vehicles": num_vehicles,
            "num_ev": num_ev,
            "warmup_steps": warmup_steps,
            "benchmark_steps": len(benchmark_rows),
            "avg_matrix_time_sec": float(np.mean([row["matrix_time_sec"] for row in benchmark_rows])) if benchmark_rows else 0.0,
            "avg_qvalue_time_sec": float(np.mean([row["qvalue_time_sec"] for row in benchmark_rows])) if benchmark_rows else 0.0,
            "avg_mcmf_time_sec": float(np.mean([row["mcmf_time_sec"] for row in benchmark_rows])) if benchmark_rows else 0.0,
            "avg_legacy_mcmf_time_sec": float(np.mean([row["legacy_mcmf_time_sec"] for row in benchmark_rows])) if benchmark_rows else 0.0,
            "avg_exact_mcmf_time_sec": float(np.mean([row["exact_mcmf_time_sec"] for row in benchmark_rows])) if benchmark_rows else 0.0,
            "avg_gurobi_time_sec": float(np.mean([row["gurobi_time_sec"] for row in benchmark_rows])) if benchmark_rows else 0.0,
        }
        if summary["avg_exact_mcmf_time_sec"] > 0:
            summary["exact_vs_legacy_speedup"] = (
                summary["avg_legacy_mcmf_time_sec"]
                / summary["avg_exact_mcmf_time_sec"]
            )
        else:
            summary["exact_vs_legacy_speedup"] = 0.0
        emit("Summary:")
        emit(json.dumps(summary, ensure_ascii=False, indent=2))
        emit("Rows:")
        emit(json.dumps(benchmark_rows, ensure_ascii=False, indent=2))
        self._write_lines(log_path, log_lines)
        print(f"Benchmark log saved to {log_path}")
        return {"summary": summary, "rows": benchmark_rows, "log_path": str(log_path)}

    def run_nyc_training(
        self,
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
        only_manhattan_zones: bool = False,
        prestep: int = 120,
        training_frequency: int = 10,
        load_best_loss: bool = False,
        checkpoint_selection: str = "auto",
        load_checkpoint_start_date: str | None = None,
        load_checkpoint_end_date: str | None = None,
        checkpoint_trained_start_hour: float | None = None,
        checkpoint_trained_stop_hour: float | None = None,
        checkpoint_suffix: str | None = None,
        pretrained_zone_dir: str = "checkpoints/zone_pretrain",
        iftransformer: bool = False,
        gat_neighbour_number: int = 0,
        post_demand_q_weight: float = 0.0,
        post_demand_head_lr_multiplier: float = 10.0,
        masac_target_entropy_ratio: float = 0.9,
        residual_target_policy: str = "cached_mcmf",
        predictor_variant: str = "p3",
        recourse_variant: str = "legacy",
        rejection_logit_shift: float = 0.0,
        common_random_numbers: bool = False,
    ):
        self._set_random_seeds(random_seed)
        if useauction:
            usemcmf = True
        parquet_paths = self._resolve_parquet_paths(
            parquet_path,
            start_year_month,
            end_year_month,
            num_episodes,
            full_demand=full_demand,
            hvfhv_parquet_path=hvfhv_parquet_path,
        )
        print(f"📦 Training parquet files ({len(parquet_paths)}):")
        for resolved_path in parquet_paths:
            print(f"   - {resolved_path}")

        env = self._create_environment(
            num_vehicles=num_vehicles,
            num_ev=num_ev,
            parquet_paths=parquet_paths,
            full_demand=full_demand,
            coord_csv=coord_csv,
            station_csv=station_csv,
            start_date=start_date,
            end_date=end_date,
            station_capacity_scale=station_capacity_scale,
            epoch_length=epoch_length,
            start_hour=start_hour,
            stop_hour=stop_hour,
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
            random_seed=random_seed,
            daily_drop_off=daily_drop_off,
            ifreject=ifreject,
            ifdropoff=ifdropoff,
            rejection_penalty_base=rejection_penalty_base,
            rejection_penalty_per_km=rejection_penalty_per_km,
            rejection_penalty_final_value_ratio=rejection_penalty_final_value_ratio,
            recourse_variant=recourse_variant,
            rejection_logit_shift=rejection_logit_shift,
            common_random_numbers=common_random_numbers,
            only_manhattan_zones=only_manhattan_zones,
        )
        env.mcmf_use_gpu = bool(mcmf_use_gpu)
        env.use_cuda_ssp = bool(mcmf_use_gpu)
        env.useauction = bool(useauction or getattr(env, 'ifsolveauctioncuda', False))
        env.mcmf_solver = "auction" if env.useauction else mcmf_solver
        env.mcmf_backend = mcmf_backend
        env.mcmf_strict = bool(mcmf_strict)
        env.mcmf_cost_scale = int(mcmf_cost_scale)
        env.mcmf_graph_reduction = bool(mcmf_graph_reduction)
        env.mcmf_verify = bool(mcmf_verify)
        env.auction_use_gpu = bool(auction_use_gpu)
        env.auction_epsilon = float(auction_epsilon)
        env.auction_max_rounds = auction_max_rounds
        env.auction_top_k = auction_top_k
        env.evaluatemode = not trainnetwork

        effective_zone_distribution_mode = self._resolve_zone_distribution_mode(zone_distribution_mode)
        print(f"Path transformer self-attention: {'enabled' if iftransformer else 'disabled'}")
        use_neural_network = self._trainer_helper._should_train_value_function(
            adpvalue,
            trainnetwork,
        )
        encoder_enabled = effective_zone_distribution_mode in {"bayes", "bayes_simple", "bayes_simple_pretrain"}
        value_function_class = self._get_value_function_class(effective_zone_distribution_mode)

        if use_neural_network or ifloadcheckpoint:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            value_function_kwargs = dict(
                grid_size=env.grid_size,
                num_vehicles=num_vehicles,
                device=device,
                episode_length=env.episode_length,
                max_requests=10000,
                env=env,
                encoder=encoder_enabled,
                zone_distribution_mode=effective_zone_distribution_mode,
                iftransformer=iftransformer,
            )
            if effective_zone_distribution_mode in {
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
                value_function_kwargs["neighbour_number"] = max(0, int(gat_neighbour_number))
            if effective_zone_distribution_mode in {
                "st_masac_gat_post_demand_direct",
                "standard_masac_gat",
                "standard_masac_gat_total_q",
                "optimization_anchored_residual",
                "standard_masac_gat_greedy_alpha",
                "standard_masac_gat_fixed_alpha",
            }:
                value_function_kwargs["post_demand_q_weight"] = float(
                    post_demand_q_weight
                )
                value_function_kwargs["post_demand_head_lr_multiplier"] = max(
                    1.0,
                    float(post_demand_head_lr_multiplier),
                )
            if effective_zone_distribution_mode in {
                "masac_baseline",
                "standard_masac_gat",
                "standard_masac_gat_total_q",
                "optimization_anchored_residual",
                "standard_masac_gat_greedy_alpha",
                "standard_masac_gat_fixed_alpha",
            }:
                value_function_kwargs["entropy_target_ratio"] = float(
                    masac_target_entropy_ratio
                )
            if effective_zone_distribution_mode == "optimization_anchored_residual":
                value_function_kwargs["residual_target_policy"] = str(
                    residual_target_policy
                )
                value_function_kwargs["predictor_variant"] = str(predictor_variant)
            value_function = value_function_class(**value_function_kwargs)
            value_function_ev = value_function_class(**value_function_kwargs)
            value_function.debug_name = "AEV"
            value_function_ev.debug_name = "EV"
            env.set_value_function(value_function)
            env.set_value_function_ev(value_function_ev)

            if ifloadcheckpoint and checkpoint_trained_start_hour is not None and checkpoint_trained_stop_hour is not None:
                trained_episode_length = max(
                    1.0,
                    ((float(checkpoint_trained_stop_hour) - float(checkpoint_trained_start_hour)) * 3600.0) / float(epoch_length),
                )
                inference_time_offset = ((float(start_hour) - float(checkpoint_trained_start_hour)) * 3600.0) / float(epoch_length)
                for vf in (value_function, value_function_ev):
                    vf.aligned_inference_episode_length = trained_episode_length
                    vf.aligned_inference_time_offset = inference_time_offset
                print(
                    "✓ Checkpoint time alignment: "
                    f"train_window={float(checkpoint_trained_start_hour):.1f}-{float(checkpoint_trained_stop_hour):.1f}, "
                    f"eval_window={float(start_hour):.1f}-{float(stop_hour):.1f}, "
                    f"offset_steps={inference_time_offset:.0f}, model_episode_steps={trained_episode_length:.0f}"
                )
        else:
            value_function = None
            value_function_ev = None

        if ifloadcheckpoint:
            checkpoint_assign_tag = self._trainer_helper._resolve_checkpoint_assign_tag(
                assignmentgurobi,
                load_checkpoint_assign_tag,
            )
            checkpoint_assignmentgurobi = checkpoint_assign_tag == "gurobi"
            print(
                "Loading value functions trained with "
                f"{checkpoint_assign_tag.upper()} assignment"
            )
            evfile, aevfile = self._checkpoint_dirs(
                transportation_mode=transportation_mode,
                assignmentgurobi=checkpoint_assignmentgurobi,
                num_ev=num_ev,
                use_intense_requests=use_intense_requests,
                start_date=load_checkpoint_start_date or start_date,
                end_date=load_checkpoint_end_date or load_checkpoint_start_date or end_date,
                zone_distribution_mode=effective_zone_distribution_mode,
                only_manhattan_zones=only_manhattan_zones,
                full_demand=full_demand,
                checkpoint_suffix=checkpoint_suffix,
            )
            if load_best_loss:
                checkpoint_selection = "best_loss"
            elif checkpoint_selection == "auto":
                checkpoint_selection = "best_reward" if not trainnetwork else "latest"
            if checkpoint_selection not in {"latest", "best_reward", "best_loss"}:
                raise ValueError(f"Unsupported checkpoint_selection: {checkpoint_selection}")
            prefer_best_checkpoint = checkpoint_selection == "best_reward"
            prefer_best_loss_checkpoint = checkpoint_selection == "best_loss"
            print(f"   Checkpoint selection: {checkpoint_selection}")
            ev_ckpt = self._trainer_helper.find_latest_checkpoint(
                evfile,
                prefer_best=prefer_best_checkpoint,
                prefer_best_loss=prefer_best_loss_checkpoint,
            )
            aev_ckpt = self._trainer_helper.find_latest_checkpoint(
                aevfile,
                prefer_best=prefer_best_checkpoint,
                prefer_best_loss=prefer_best_loss_checkpoint,
            )
            print(f"   AEV checkpoint: {aev_ckpt or 'Not Found'}")
            print(f"   EV  checkpoint: {ev_ckpt or 'Not Found'}")
            missing_checkpoints = []
            if not aev_ckpt:
                missing_checkpoints.append(aevfile)
            if not ev_ckpt:
                missing_checkpoints.append(evfile)
            if missing_checkpoints:
                message = (
                    "Requested checkpoint load but checkpoint(s) were not found: "
                    + ", ".join(missing_checkpoints)
                )
                if not trainnetwork:
                    raise FileNotFoundError(message)
                print(f"⚠ {message}; continuing with newly initialized network(s).")
            if aev_ckpt:
                self._trainer_helper.load_checkpoint(value_function, aev_ckpt)
            if ev_ckpt:
                self._trainer_helper.load_checkpoint(value_function_ev, ev_ckpt)

        if effective_zone_distribution_mode == "bayes_simple_pretrain" and value_function is not None:
            self._configure_pretrained_zone_distributors(
                value_function=value_function,
                value_function_ev=value_function_ev,
                env=env,
                transportation_mode=transportation_mode,
                pretrained_zone_dir=pretrained_zone_dir,
                only_manhattan_zones=only_manhattan_zones,
                full_demand=full_demand,
            )

        env.adp_value = adpvalue
        env.assignmentgurobi = assignmentgurobi
        training_frequency = max(1, int(training_frequency))
        warmup_steps = 64
        loss_save_interval = 60
        prestep = max(0, int(prestep))
        motion_fn = self._select_motion_fn(env, transportation_mode)
        if effective_zone_distribution_mode == "bayes_simple_pretrain" and value_function is not None:
            predictor = getattr(value_function, "time_zone_dist_predictor", None) if value_function else None
            predictor_ev = getattr(value_function_ev, "time_zone_dist_predictor", None) if value_function_ev else None
            print(f"✓ Pretrained zone distribution mode: {effective_zone_distribution_mode}")
            if predictor is not None:
                for parameter in predictor.parameters():
                    parameter.requires_grad = False
            if predictor_ev is not None:
                for parameter in predictor_ev.parameters():
                    parameter.requires_grad = False
            print(f"✓ Freezed zone distribution predictors: {predictor is not None}, {predictor_ev is not None}")
        total_station_capacity = sum(station.max_capacity for station in env.charging_manager.stations.values())
        print(f"✓ NYCEnvironment: {num_vehicles} vehicles, {env.NUM_ZONES} zones")
        print(f"✓ Charging stations: {len(env.charging_manager.stations)} stations, capacity={total_station_capacity}, scale={station_capacity_scale}")
        if env.useauction:
            print(f"✓ MCMF solver: AUCTION ({'GPU' if env.auction_use_gpu else 'CPU'})")
        elif env.mcmf_solver == "exact":
            print(
                f"✓ MCMF solver: EXACT (backend={env.mcmf_backend}, "
                f"strict={env.mcmf_strict}, scale={env.mcmf_cost_scale})"
            )
        else:
            print(f"✓ MCMF solver: {'GPU' if env.mcmf_use_gpu else 'CPU'}")
        if use_neural_network:
            print(f"✓ PyTorchChargingValueFunction({effective_zone_distribution_mode}): {sum(p.numel() for p in value_function.network.parameters())} params")
            print(f"   device: {value_function.device}")
            print(f"✓ Heuristic warmup steps before training: {prestep}")
            print(
                f"✓ Value network training frequency: every {training_frequency} steps "
                f"({training_frequency * float(epoch_length) / 60.0:.1f} min)"
            )

        results = {
            "Idle_average": [],
            "episode_rewards": [],
            "episode_rewards_aev": [],
            "episode_rewards_ev": [],
            "drop_off_rates": [],
            "charging_events": [],
            "episode_detailed_stats": [],
            "episode_rejected_requests": [],
            "episode_recourse_requests": [],
            "episode_lost_requests": [],
            "vehicle_visit_stats": [],
            "battery_levels": [],
            "environment_stats": [],
            "value_function_losses": [],
            "value_function_ev_losses": [],
            "value_function_normalized_td_losses": [],
            "value_function_ev_normalized_td_losses": [],
            "qvalue_losses": [],
            "qvalue_ev_losses": [],
            "queue_wait_losses": [],
            "queue_wait_ev_losses": [],
            "qvalue_normalized_td_losses": [],
            "qvalue_ev_normalized_td_losses": [],
            "episode_times": [],
            "avg_step_times": [],
            "step_timing_rows": [],
        }
        best_reward = float("-inf")
        best_reward_ev = float("-inf")
        best_reward_aev = float("-inf")
        best_loss_aev = float("inf")
        best_loss_ev = float("inf")

        global_step = 0
        combined_best_loss = float("inf")
        for episode in range(num_episodes):
            episode_start = time.time()
            episode_seed = 32 + episode
            env.set_request_generation_seed(episode_seed)

            env.reset()
            episode_reward = 0
            episode_reward_aev = 0
            episode_reward_ev = 0
            episode_charging_events = []
            episode_losses = []
            episode_losses_ev = []
            episode_norm_td_losses = []
            episode_norm_td_losses_ev = []
            episode_queue_losses = []
            episode_queue_losses_ev = []
            idle_list = []
            step_durations = []

            for step in range(env.episode_length):
                step_start = time.time()
                current_requests = list(env.active_requests.values())
                simulate_motion_start = time.time()
                heuristic_phase = use_neural_network and global_step < prestep
                solver_state = self._temporary_heuristic_warmup(env, heuristic_phase)
                try:
                    actions, storeactions, storeactions_ev = motion_fn(
                        agents=[], current_requests=current_requests, rebalance=True
                    )
                finally:
                    self._restore_solver_state(env, solver_state)
                simulate_motion_time = time.time() - simulate_motion_start

                env_step_start = time.time()
                _, rewards, _, _, info = env.step(actions, storeactions, storeactions_ev)
                env_step_time = time.time() - env_step_start
                step_wall_time = time.time() - step_start
                step_durations.append(step_wall_time)
                rebalance_profile = info.get("rebalancing_profile", {})
                simulation_profile = info.get("simulation_profile", {})
                step_profile = info.get("step_profile", {})
                results["step_timing_rows"].append({
                    "episode_number": episode + 1,
                    "step_number": step,
                    "global_step": global_step,
                    "simulate_motion_time_sec": simulate_motion_time,
                    "env_step_time_sec": env_step_time,
                    "step_wall_time_sec": step_wall_time,
                    "solver_name": rebalance_profile.get("solver_name"),
                    "vehicles_to_rebalance": int(rebalance_profile.get("vehicles_to_rebalance", 0) or 0),
                    "available_requests": int(rebalance_profile.get("available_requests", 0) or 0),
                    "qmatrix_time_sec": float(rebalance_profile.get("qmatrix_time_sec", 0.0) or 0.0),
                    "qvalue_time_sec": float(rebalance_profile.get("qvalue_time_sec", 0.0) or 0.0),
                    "solver_time_sec": float(rebalance_profile.get("solver_time_sec", 0.0) or 0.0),
                    "solve_total_time_sec": float(rebalance_profile.get("solve_total_time_sec", 0.0) or 0.0),
                    "simulation_total_time_sec": float(simulation_profile.get("total_time_sec", 0.0) or 0.0),
                    "simulation_rebalancing_time_sec": float(simulation_profile.get("rebalancing_time_sec", 0.0) or 0.0),
                    "simulation_fallback_time_sec": float(simulation_profile.get("fallback_time_sec", 0.0) or 0.0),
                    "execute_actions_time_sec": float(step_profile.get("execute_actions_time_sec", 0.0) or 0.0),
                    "update_environment_time_sec": float(step_profile.get("update_environment_time_sec", 0.0) or 0.0),
                    "q_learning_aev_time_sec": float(step_profile.get("q_learning_aev_time_sec", 0.0) or 0.0),
                    "q_learning_ev_time_sec": float(step_profile.get("q_learning_ev_time_sec", 0.0) or 0.0),
                    "step_profile_total_time_sec": float(step_profile.get("total_time_sec", 0.0) or 0.0),
                })

                if step % 25 == 0:
                    stats = env.get_stats()
                    vehicle_status_count = {
                        "charging": 0, "onboard": 0, "to_pickup": 0,
                        "to_charge": 0, "idle_moving": 0, "fully_idle": 0,
                    }
                    for vid, vehicle in env.vehicles.items():
                        if vehicle["charging_station"] is not None:
                            status = "charging"
                        elif vehicle["passenger_onboard"] is not None:
                            status = "onboard"
                        elif vehicle["assigned_request"] is not None:
                            status = "to_pickup"
                        elif vehicle.get("charging_target") is not None:
                            status = "to_charge"
                        elif vehicle.get("idle_target") is not None or vehicle.get("target_location") is not None:
                            status = "idle_moving"
                        else:
                            status = "fully_idle"
                        vehicle_status_count[status] += 1
                    step_reward = sum(rewards.values())
                    avg_step_wall_time = float(np.mean(step_durations)) if step_durations else 0.0
                    warmup_label = "heuristic" if heuristic_phase else "training"
                    print(
                        f"  Step {step}: phase={warmup_label}, active_requests={len(env.active_requests)}, "
                        f"new_requests={stats.get('generated_requests_last_step', 0)}, "
                        f"total_generated={stats.get('total_generated_requests', 0)}, "
                        f"demand_date={stats.get('current_real_date', 'unknown')}, "
                        f"hour={stats.get('current_real_hour', 0.0):.2f}, "
                        f"min_battery={stats.get('min_vehicle_battery', 0.0):.3f}, "
                        f"cannot_reach_charge={stats.get('vehicles_unable_to_reach_charging', 0)}, "
                        f"reward={step_reward:.2f}, "
                        f"rej_pen={info.get('step_rejection_reward_total', 0.0):.2f}/"
                        f"{info.get('step_rejection_reward_count', 0)}, "
                        f"status={vehicle_status_count}, step_wall={step_wall_time:.3f}s, avg_step_wall={avg_step_wall_time:.3f}s"
                    )
                    if rebalance_profile:
                        print(
                            f"    RebalanceTiming: solver={rebalance_profile.get('solver_name', 'n/a')} "
                            f"matrix={int(rebalance_profile.get('matrix_rows', 0))}x{int(rebalance_profile.get('action_columns', 0))} "
                            f"blocks=req:{int(rebalance_profile.get('request_matrix_cols', 0))}/chg:{int(rebalance_profile.get('charging_matrix_cols', 0))}/zone:{int(rebalance_profile.get('zone_matrix_cols', 0))}/wait:{int(rebalance_profile.get('wait_matrix_cols', 0))} "
                            f"qmatrix={rebalance_profile.get('qmatrix_time_sec', 0.0):.3f}s "
                            f"qvalue={rebalance_profile.get('qvalue_time_sec', 0.0):.3f}s "
                            f"solve={rebalance_profile.get('solver_time_sec', 0.0):.3f}s "
                            f"total={rebalance_profile.get('solve_total_time_sec', 0.0):.3f}s "
                            f"vehicles={int(rebalance_profile.get('vehicles_to_rebalance', 0))} "
                            f"reqs={int(rebalance_profile.get('available_requests', 0))} "
                            f"req_edges={int(rebalance_profile.get('feasible_request_edges', 0))} "
                            f"charge_edges={int(rebalance_profile.get('feasible_charging_edges', 0))} "
                            f"zone_edges={int(rebalance_profile.get('feasible_zone_edges', 0))}",
                            flush=True,
                        )
                    if simulation_profile:
                        print(
                            f"    SimTiming: simulate_motion={simulate_motion_time:.3f}s internal_total={simulation_profile.get('total_time_sec', 0.0):.3f}s "
                            f"rebalance={simulation_profile.get('rebalancing_time_sec', 0.0):.3f}s "
                            f"fallback={simulation_profile.get('fallback_time_sec', 0.0):.3f}s",
                            flush=True,
                        )
                    if step_profile:
                        print(
                            f"    StepTiming: env_step={env_step_time:.3f}s execute={step_profile.get('execute_actions_time_sec', 0.0):.3f}s "
                            f"update_env={step_profile.get('update_environment_time_sec', 0.0):.3f}s "
                            f"qlearn_aev={step_profile.get('q_learning_aev_time_sec', 0.0):.3f}s "
                            f"qlearn_ev={step_profile.get('q_learning_ev_time_sec', 0.0):.3f}s",
                            flush=True,
                        )
                    idle_list.append(vehicle_status_count["fully_idle"])

                def _value_training_ready(value_fn) -> bool:
                    if value_fn is None:
                        return False
                    replay_ready = len(getattr(value_fn, "experience_buffer", [])) >= warmup_steps
                    queue_ready = (
                        hasattr(value_fn, "train_queue_predictor")
                        and len(getattr(value_fn, "queue_experience_buffer", [])) >= 4
                    )
                    return bool(replay_ready or queue_ready)

                aev_training_ready = _value_training_ready(value_function)
                ev_training_ready = _value_training_ready(value_function_ev)
                if (
                    use_neural_network
                    and global_step >= prestep
                    and (aev_training_ready or ev_training_ready)
                    and episode >= start_training_episode
                    and step % training_frequency == 0
                ):
                    loss_aev = value_function.train_step(batch_size=batch_size, ifEV=False) if aev_training_ready else 0.0

                    if loss_aev > 0:
                        episode_losses.append(loss_aev)
                        latest_norm_td = self._latest_metric(value_function, "normalized_td_losses")
                        if latest_norm_td > 0:
                            episode_norm_td_losses.append(latest_norm_td)
                        latest_queue_loss = self._latest_metric(value_function, "queue_training_losses")
                        if latest_queue_loss > 0:
                            episode_queue_losses.append(latest_queue_loss)
                    loss_ev = value_function_ev.train_step(batch_size=batch_size, ifEV=True) if ev_training_ready else 0.0
                    if loss_ev > 0:
                        episode_losses_ev.append(loss_ev)
                        latest_norm_td_ev = self._latest_metric(value_function_ev, "normalized_td_losses")
                        if latest_norm_td_ev > 0:
                            episode_norm_td_losses_ev.append(latest_norm_td_ev)
                        latest_queue_loss_ev = self._latest_metric(value_function_ev, "queue_training_losses")
                        if latest_queue_loss_ev > 0:
                            episode_queue_losses_ev.append(latest_queue_loss_ev)

                    if not ifloadcheckpoint:
                        evfile, aevfile = self._checkpoint_dirs(
                            transportation_mode=transportation_mode,
                            assignmentgurobi=assignmentgurobi,
                            num_ev=num_ev,
                            use_intense_requests=use_intense_requests,
                            start_date=start_date,
                            end_date=end_date,
                            zone_distribution_mode=effective_zone_distribution_mode,
                            only_manhattan_zones=only_manhattan_zones,
                            full_demand=full_demand,
                            checkpoint_suffix=checkpoint_suffix,
                        )

                        loss_aev_f = float(loss_aev)
                        loss_ev_f = float(loss_ev)
                        if loss_aev_f > 0 and loss_ev_f > 0:
                            combined_loss_f = loss_aev_f + loss_ev_f
                            if np.isfinite(combined_loss_f) and combined_loss_f < float(combined_best_loss):
                                combined_best_loss = combined_loss_f
                                print(
                                    f"🏆 New best combined loss: {combined_best_loss:.6f} at episode {episode + 1}, step {step}"
                                )
                                self._trainer_helper._save_q_network_checkpoint(
                                    value_function,
                                    episode + 1,
                                    checkpoint_dir=aevfile,
                                    checkpoint_tag="best_combined_loss",
                                )
                                self._trainer_helper._save_q_network_checkpoint(
                                    value_function_ev,
                                    episode + 1,
                                    checkpoint_dir=evfile,
                                    checkpoint_tag="best_combined_loss",
                                )

                        if (
                            np.isfinite(loss_aev_f)
                            and loss_aev_f > 0
                            and loss_aev_f < best_loss_aev
                        ):
                            best_loss_aev = loss_aev_f
                            print(
                                f"🏆 New best AEV loss: {best_loss_aev:.6f} at episode {episode + 1}, step {step}, saving AEV checkpoint..."
                            )
                            self._trainer_helper._save_q_network_checkpoint(
                                value_function,
                                episode + 1,
                                checkpoint_dir=aevfile,
                                checkpoint_tag="best_loss",
                            )
                        if (
                            np.isfinite(loss_ev_f)
                            and loss_ev_f > 0
                            and loss_ev_f < best_loss_ev
                        ):
                            best_loss_ev = loss_ev_f
                            print(
                                f"🏆 New best EV loss: {best_loss_ev:.6f} at episode {episode + 1}, step {step}, saving EV checkpoint..."
                            )
                            self._trainer_helper._save_q_network_checkpoint(
                                value_function_ev,
                                episode + 1,
                                checkpoint_dir=evfile,
                                checkpoint_tag="best_loss",
                            )
                    if (
                        global_step > 0
                        and global_step % loss_save_interval == 0
                    ):
                        has_queue_loss = bool(
                            getattr(value_function, "queue_training_mse_losses", [])
                            or getattr(value_function_ev, "queue_training_mse_losses", [])
                            or getattr(value_function, "queue_training_losses", [])
                            or getattr(value_function_ev, "queue_training_losses", [])
                        )
                        if float(loss_aev) > 0 or float(loss_ev) > 0 or has_queue_loss:
                            self._save_loss_snapshot(
                                transportation_mode=transportation_mode,
                                zone_distribution_mode=effective_zone_distribution_mode,
                                global_step=global_step,
                                value_function=value_function,
                                value_function_ev=value_function_ev,
                            )
                        predictor = getattr(value_function, "time_zone_dist_predictor", None) if value_function is not None else None
                        predictor_ev = getattr(value_function_ev, "time_zone_dist_predictor", None) if value_function_ev is not None else None
                        predictor_trainable = False
                        if predictor is not None:
                            predictor_trainable = predictor_trainable or any(
                                parameter.requires_grad for parameter in predictor.parameters()
                            )
                        if predictor_ev is not None:
                            predictor_trainable = predictor_trainable or any(
                                parameter.requires_grad for parameter in predictor_ev.parameters()
                            )
                        if predictor_trainable and effective_zone_distribution_mode == "bayes_simple_pretrain":
                            print("⚠ Warning: Pretrained zone distribution predictor parameters are still trainable.")
                            self._save_zone_kl_loss_arrays(
                                transportation_mode=transportation_mode,
                                zone_distribution_mode=effective_zone_distribution_mode,
                                global_step=global_step,
                                value_function=value_function,
                                value_function_ev=value_function_ev,
                            )
                        elif predictor_trainable and effective_zone_distribution_mode in {
                            "bayes",
                            "bayes_simple",
                            "pretrain_zonepredictor",
                        }:
                            self._save_zone_kl_loss_arrays(
                                transportation_mode=transportation_mode,
                                zone_distribution_mode=effective_zone_distribution_mode,
                                global_step=global_step,
                                value_function=value_function,
                                value_function_ev=value_function_ev,
                            )

                episode_reward += sum(rewards.values())
                episode_reward_aev += sum(
                    reward for vehicle_id, reward in rewards.items() if env.vehicles.get(vehicle_id, {}).get("type") == 2
                )
                episode_reward_ev += sum(
                    reward for vehicle_id, reward in rewards.items() if env.vehicles.get(vehicle_id, {}).get("type") == 1
                )
                episode_charging_events.extend(info.get("charging_events", []))
                global_step += 1

            if use_neural_network and not ifloadcheckpoint:
                evfile, aevfile = self._checkpoint_dirs(
                    transportation_mode=transportation_mode,
                    assignmentgurobi=assignmentgurobi,
                    num_ev=num_ev,
                    use_intense_requests=use_intense_requests,
                    start_date=start_date,
                    end_date=end_date,
                    zone_distribution_mode=effective_zone_distribution_mode,
                    only_manhattan_zones=only_manhattan_zones,
                    full_demand=full_demand,
                    checkpoint_suffix=checkpoint_suffix,
                )

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    print(f"🏆 New best total reward: {best_reward:.2f} at episode {episode + 1}, saving paired checkpoints...")
                    self._trainer_helper._save_q_network_checkpoint(
                        value_function,
                        episode + 1,
                        checkpoint_dir=aevfile,
                        checkpoint_tag="best",
                    )
                    self._trainer_helper._save_q_network_checkpoint(
                        value_function_ev,
                        episode + 1,
                        checkpoint_dir=evfile,
                        checkpoint_tag="best",
                    )

                if episode_reward_ev > best_reward_ev:
                    best_reward_ev = episode_reward_ev
                    print(f"🏆 New best EV reward: {best_reward_ev:.2f} at episode {episode + 1}, saving EV-only best checkpoint...")
                    self._trainer_helper._save_q_network_checkpoint(
                        value_function_ev,
                        episode + 1,
                        checkpoint_dir=evfile,
                        checkpoint_tag="best_ev",
                    )

                if episode_reward_aev > best_reward_aev:
                    best_reward_aev = episode_reward_aev
                    print(f"🏆 New best AEV reward: {best_reward_aev:.2f} at episode {episode + 1}, saving AEV-only best checkpoint...")
                    self._trainer_helper._save_q_network_checkpoint(
                        value_function,
                        episode + 1,
                        checkpoint_dir=aevfile,
                        checkpoint_tag="best_aev",
                    )

            if use_neural_network and not ifloadcheckpoint:
                evfile, aevfile = self._checkpoint_dirs(
                    transportation_mode=transportation_mode,
                    assignmentgurobi=assignmentgurobi,
                    num_ev=num_ev,
                    use_intense_requests=use_intense_requests,
                    start_date=start_date,
                    end_date=end_date,
                    zone_distribution_mode=effective_zone_distribution_mode,
                    only_manhattan_zones=only_manhattan_zones,
                    full_demand=full_demand,
                    checkpoint_suffix=checkpoint_suffix,
                )
                self._trainer_helper._save_q_network_checkpoint(
                    value_function,
                    episode + 1,
                    checkpoint_dir=aevfile,
                )
                self._trainer_helper._save_q_network_checkpoint(
                    value_function_ev,
                    episode + 1,
                    checkpoint_dir=evfile,
                )

            avg_loss_aev = np.mean(episode_losses) if episode_losses else 0.0
            avg_loss_ev = np.mean(episode_losses_ev) if episode_losses_ev else 0.0
            avg_norm_td_aev = np.mean(episode_norm_td_losses) if episode_norm_td_losses else 0.0
            avg_norm_td_ev = np.mean(episode_norm_td_losses_ev) if episode_norm_td_losses_ev else 0.0
            avg_queue_loss_aev = np.mean(episode_queue_losses) if episode_queue_losses else 0.0
            avg_queue_loss_ev = np.mean(episode_queue_losses_ev) if episode_queue_losses_ev else 0.0
            episode_time = time.time() - episode_start
            avg_step_time = float(np.mean(step_durations)) if step_durations else 0.0

            results["Idle_average"].append(sum(idle_list) / len(idle_list) if idle_list else 0)
            results["episode_rewards"].append(episode_reward)
            results["episode_rewards_aev"].append(episode_reward_aev)
            results["episode_rewards_ev"].append(episode_reward_ev)
            results["charging_events"].extend(episode_charging_events)
            results["value_function_losses"].append(avg_loss_aev)
            results["value_function_ev_losses"].append(avg_loss_ev)
            results["value_function_normalized_td_losses"].append(avg_norm_td_aev)
            results["value_function_ev_normalized_td_losses"].append(avg_norm_td_ev)
            results["qvalue_losses"].extend(episode_losses)
            results["qvalue_ev_losses"].extend(episode_losses_ev)
            results["queue_wait_losses"].extend(episode_queue_losses)
            results["queue_wait_ev_losses"].extend(episode_queue_losses_ev)
            results["qvalue_normalized_td_losses"].extend(episode_norm_td_losses)
            results["qvalue_ev_normalized_td_losses"].extend(episode_norm_td_losses_ev)
            results["episode_times"].append(episode_time)
            results["avg_step_times"].append(avg_step_time)

            stats = env.get_stats()
            results["environment_stats"].append(stats)
            results["battery_levels"].append(stats["average_battery"])

            episode_stats = env.get_episode_stats()
            episode_stats["episode_number"] = episode + 1
            episode_stats["episode_reward"] = episode_reward
            episode_stats["episode_reward_aev"] = episode_reward_aev
            episode_stats["episode_reward_ev"] = episode_reward_ev
            episode_stats["episode_aev_reward"] = episode_reward_aev
            episode_stats["episode_ev_reward"] = episode_reward_ev
            episode_stats["charging_events_count"] = len(episode_charging_events)
            episode_stats["neural_network_loss"] = avg_loss_aev
            episode_stats["neural_evnetwork_loss"] = avg_loss_ev
            episode_stats["normalized_td_loss_aev"] = avg_norm_td_aev
            episode_stats["normalized_td_loss_ev"] = avg_norm_td_ev
            episode_stats["queue_wait_loss_aev"] = avg_queue_loss_aev
            episode_stats["queue_wait_loss_ev"] = avg_queue_loss_ev
            episode_stats["queue_wait_mse_loss_aev"] = avg_queue_loss_aev
            episode_stats["queue_wait_mse_loss_ev"] = avg_queue_loss_ev
            episode_stats["episode_time_sec"] = episode_time
            episode_stats["avg_step_time_sec"] = avg_step_time
            episode_stats["avg_step_time_ms"] = avg_step_time * 1000.0
            results["episode_detailed_stats"].append(episode_stats)
            results["drop_off_rates"].append(episode_stats.get("drop_off_rate", 0.0))
            results["episode_rejected_requests"].append(episode_stats.get("rejected_requests", 0))
            results["episode_recourse_requests"].append(episode_stats.get("recourse_requests", 0))
            results["episode_lost_requests"].append(episode_stats.get("lost_requests", 0))

            print(
                f"{episode + 1:4d} | reward={episode_reward:8.2f} | AEV={episode_reward_aev:8.2f} | EV={episode_reward_ev:8.2f} | loss(AEV)={avg_loss_aev:.4f} "
                f"| loss(EV)={avg_loss_ev:.4f} "
                f"| normTD(AEV)={avg_norm_td_aev:.4f} "
                f"| normTD(EV)={avg_norm_td_ev:.4f} "
                f"| accept={episode_stats.get('accepted_orders', 0)} "
                f"| reject={episode_stats.get('rejected_orders', 0)} "
                f"| rejected_requests={episode_stats.get('rejected_requests', 0)} "
                f"| recourse_requests={episode_stats.get('recourse_requests', 0)} "
                f"| lost_requests={episode_stats.get('lost_requests', 0)} "
                f"| rej_pen={episode_stats.get('rejection_reward_total', 0.0):.2f}/"
                f"{episode_stats.get('rejection_reward_count', 0)} "
                f"| complete={episode_stats.get('completed_orders', 0)} "
                f"| avg_wait={episode_stats.get('avg_wait', 0.0):.2f} "
                f"| wait_vehicles={episode_stats.get('waiting_vehicle_count', 0)} "
                f"| online={episode_stats.get('online_vehicles', 0)} "
                f"| offline={episode_stats.get('offline_vehicles', 0)} "
                f"| dropoff={episode_stats.get('drop_off_rate', 0.0):.3f} "
                f"| battery={episode_stats.get('avg_battery_level', 0):.2f} "
                f"| episode_time={episode_time:.1f}s "
                f"| sim_epoch_time={avg_step_time:.4f}s ({avg_step_time * 1000.0:.1f}ms)"
            )

            torch.cuda.empty_cache()

        print("\n=== NYC Training Complete ===")
        print(f"Episodes: {num_episodes}, Avg reward: {np.mean(results['episode_rewards']):.2f}")
        if results["episode_rewards_aev"]:
            print(f"Avg AEV reward: {np.mean(results['episode_rewards_aev']):.2f}")
        if results["episode_rewards_ev"]:
            print(f"Avg EV reward: {np.mean(results['episode_rewards_ev']):.2f}")
        if results["episode_times"]:
            print(f"Avg episode time: {np.mean(results['episode_times']):.2f}s")
        if results["avg_step_times"]:
            avg_epoch_time = np.mean(results["avg_step_times"])
            print(f"Avg simulation epoch wall time: {avg_epoch_time:.4f}s ({avg_epoch_time * 1000.0:.1f}ms)")
        if results["episode_detailed_stats"]:
            wait_summary = aggregate_wait_metrics(results["episode_detailed_stats"])
            print(
                "Avg wait among waiting vehicles: "
                f"{wait_summary['avg_wait']:.2f} steps"
            )

        if use_neural_network:
            self._save_loss_snapshot(
                transportation_mode=transportation_mode,
                zone_distribution_mode=effective_zone_distribution_mode,
                global_step=global_step,
                value_function=value_function,
                value_function_ev=value_function_ev,
            )
            self._save_zone_kl_loss_arrays(
                transportation_mode=transportation_mode,
                zone_distribution_mode=effective_zone_distribution_mode,
                global_step=global_step,
                value_function=value_function,
                value_function_ev=value_function_ev,
            )

        results_dir = Path("results/nyc_tests") if assignmentgurobi else Path("results/nyc_tests_h")
        results_dir.mkdir(parents=True, exist_ok=True)
        excel_path, spatial_path = self._trainer_helper._save_episode_stats_to_excel(
            env,
            results["episode_detailed_stats"],
            results_dir,
            transportation_mode=transportation_mode,
            zone_distribution_mode=effective_zone_distribution_mode,
        )
        results["excel_path"] = excel_path
        results["spatial_image_path"] = spatial_path

        return results, env
