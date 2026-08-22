"""
NYCEnvironment: NYC taxi zone-based environment with real trip data and charging.

Uses NYC TLC taxi zone IDs (1-263) as locations, loads real yellow taxi parquet
data (optionally combined with non-pooled HVFHV trips) for demand generation,
and replicates ChargingIntegratedEnvironment's
charging / vehicle-movement / Gurobi-rebalancing interface so that existing
trainers (run_trainer.py etc.) can use it with minimal changes.
"""

from __future__ import annotations

from datetime import date
import hashlib
import inspect
import math
import os
import random
import re
import tempfile
import time
import urllib.request
import zipfile
from collections import deque
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import torch
import numpy as np
import pandas as pd
from shapely import contains_xy

from src.Action import Action, ChargingAction, IdleAction, ServiceAction
from src.charging_station import ChargingStation, ChargingStationManager
from src.charging_wait_metrics import positive_wait_metrics
from src.qvalue_precision import qvalue_rounding_diagnostics, round_qvalue_matrix
from src.Request import Request

# ---------------------------------------------------------------------------
# Haversine helper (km)
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================================
# NYCEnvironment
# ============================================================================

class NYCEnvironment:
    """
    Grid-free NYC environment operating on official TLC taxi-zone geometry.

    Key differences from ChargingIntegratedEnvironment:
    * Locations are official zone IDs (1-263), not grid cells.
    * Demand is loaded from real NYC yellow taxi parquet files.
    * Passenger trips use TLC route distance at the configured simulator speed;
      deadhead and relocation legs use official zone centroids.
    * Charging logic, vehicle dict layout, Action classes, step() / reset()
      / simulate_motion() interfaces mirror ChargingIntegratedEnvironment
      so that the same trainer code can drive both environments.
    """

    MANHATTAN_TLC_ZONE_IDS = {
        4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88,
        90, 100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128,
        137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162,
        163, 164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231,
        232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262,
        263,
    }
    TLC_TAXI_ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    MILES_TO_KM = 1.609344

    def _round_assignment_qvalues(self, values):
        """Put every NYC assignment Q matrix on the shared solver grid."""

        rounded = round_qvalue_matrix(
            values,
            int(getattr(self, 'mcmf_cost_scale', 10_000)),
        )
        self.qvalue_precision_last = qvalue_rounding_diagnostics(
            values,
            rounded,
            int(getattr(self, 'mcmf_cost_scale', 10_000)),
        )
        self.qvalue_precision_last["shape"] = tuple(rounded.shape)
        return rounded

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_vehicles: int = 25,
        num_stations: int = 10,
        ev_num_vehicles: int | None = None,
        parquet_path: str | list[str] | None = None,
        full_demand: bool = False,
        coord_csv: str | None = None,
        station_csv: str | None = None,
        epoch_length_sec: float = 30.0,
        start_hour: float = 0.0,
        stop_hour: float = 24.0,
        episode_length: int = 2880,
        heuristic_battery_threshold: float = 0.5,
        use_intense_requests: bool = True,
        assignmentgurobi: bool = True,
        usemcmf: bool = True,
        useauction: bool = False,
        auction_use_gpu: bool = False,
        auction_epsilon: float = 1e-3,
        auction_max_rounds: int | None = None,
        auction_top_k: int | None = None,
        mcmf_solver: str | None = "exact",
        mcmf_backend: str = "gurobi_network",
        mcmf_strict: bool = True,
        mcmf_cost_scale: int = 10_000,
        mcmf_graph_reduction: bool = True,
        mcmf_verify: bool = False,
        mcmf_fallback_value: float | None = None,
        knownreject: bool = False,
        gurobi_network: bool = True,
        gurobi_network_lp: bool = True,
        daily_drop_off: bool = True,
        ifreject: bool = True,
        ifdropoff: bool = False,
        record_time: bool = False,
        random_seed: int | None = None,
        multi_gpu_devices=None,
        station_capacity_scale: float = 1.0,
        zone_dataset = None,
        ifonlymanhatten: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
        test_request_less_aev: bool = False,
        ifsolveauctioncuda: bool = False,
        rejection_penalty_base: float = 4.0,
        rejection_penalty_per_km: float = 0.35,
        rejection_penalty_final_value_ratio: float | None = 0.25,
        operating_cost_per_km: float = 0.08,
    ):
        # --- base directory of data files ---
        _base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nyedata", "nye_simulation")

        # --- load zone coordinates ---
        # ``coord_csv`` remains accepted for CLI compatibility, but the
        # simulator no longer trusts a separately generated coordinate table.
        # All zone centroids and borough labels come from the official TLC
        # taxi-zone geometry for both Manhattan and full-NYC runs.
        self.coord_csv = coord_csv
        geometry_candidate = (
            Path(coord_csv).expanduser().resolve().parent / "taxi_zones.geojson"
            if coord_csv is not None
            else Path(_base) / "taxi_zones.geojson"
        )
        default_geometry = Path(_base) / "taxi_zones.geojson"
        self.zone_geometry_path = str(
            geometry_candidate if geometry_candidate.exists() else default_geometry
        )
        self.ifonlymanhatten = ifonlymanhatten
        self._load_zone_coordinates()
        if self.ifonlymanhatten and not self.manhattan_zone_ids:
            raise RuntimeError(
                "ifonlymanhatten=True but no Manhattan TLC zone ids were loaded. "
                "Check that taxi_zones.geojson/geopandas is available, or disable --only-manhattan-zones."
            )
        print(
            f"✓ Zone scope requested: {'Manhattan only' if self.ifonlymanhatten else 'full NYC'} "
            f"(geometry_zones={len(self.real_zone_ids)}, manhattan_zones={len(self.manhattan_zone_ids)})"
        )

        self.NUM_ZONES = len(self.zone_coords)
        self.NUM_LOCATIONS = self.NUM_ZONES
        self.MAX_CAPACITY = 4
        self.EPOCH_LENGTH = epoch_length_sec
        self.NUM_AGENTS = num_vehicles

        if torch.cuda.is_available() and ifsolveauctioncuda:
            print("✓ CUDA is available. Auction solver will use GPU.")
            self.ifsolveauctioncuda = True
        else:
            self.ifsolveauctioncuda = False

        # --- demand data ---
        self.parquet_path = parquet_path or self._find_default_parquet(_base)
        self.full_demand = bool(full_demand)
        self._demand_df: pd.DataFrame | None = None  # lazy loaded
        self._warned_empty_demand_dates: set[str] = set()
        self._demand_cursor: int = 0
        self._demand_day_cache: pd.DataFrame | None = None
        self._demand_day_cache_label = None
        self._available_demand_dates: list = []
        self.episode_day_index = -1
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date) if end_date is not None else self.start_date
        if self.start_date is not None and self.end_date is not None and self.end_date < self.start_date:
            raise ValueError(f"end_date {end_date} must be on or after start_date {start_date}")

        # --- time window ---
        self.start_hour = start_hour
        self.stop_hour = stop_hour
        self.START_EPOCH = start_hour * 3600
        self.STOP_EPOCH = stop_hour * 3600
        daily_window_sec = max(self.EPOCH_LENGTH, self.STOP_EPOCH - self.START_EPOCH)
        self.simulation_period = max(1, int(math.ceil(daily_window_sec / self.EPOCH_LENGTH)))
        self.days_per_week = 7

        # --- RNG ---
        self.initial_random_seed = random_seed
        self.request_generation_seed = random_seed
        if random_seed is not None:
            self.set_random_seed(random_seed)

        # --- vehicle counts ---
        self.num_vehicles = num_vehicles
        self.ev_num_vehicles = ev_num_vehicles if ev_num_vehicles is not None else num_vehicles // 2
        self.num_stations = num_stations
        self.station_capacity_scale = max(0.0, float(1.0 if station_capacity_scale is None else station_capacity_scale))

        # --- flags ---
        self.use_intense_requests = use_intense_requests
        self.multi_gpu_devices = multi_gpu_devices
        self.assignmentgurobi = assignmentgurobi
        self.usemcmf = usemcmf
        self.useauction = bool(useauction or self.ifsolveauctioncuda)
        self.mcmf_solver = "auction" if self.useauction else mcmf_solver
        self.mcmf_backend = str(mcmf_backend)
        self.mcmf_strict = bool(mcmf_strict)
        self.mcmf_cost_scale = int(mcmf_cost_scale)
        self.mcmf_graph_reduction = bool(mcmf_graph_reduction)
        self.mcmf_verify = bool(mcmf_verify)
        self.mcmf_fallback_value = mcmf_fallback_value
        self.auction_use_gpu = bool(auction_use_gpu)
        self.auction_epsilon = float(auction_epsilon)
        self.auction_max_rounds = auction_max_rounds
        self.auction_top_k = auction_top_k
        self.usenetworkx = False
        self.gurobi_network = gurobi_network
        self.knownreject = knownreject
        self.battery_first = False
        self.gurobi_network_lp = gurobi_network_lp
        self.daily_drop_off = daily_drop_off
        self.ifreject = ifreject
        self.ifdropoff = ifdropoff
        self.record_time = record_time
        self.evaluatemode = True
        self.test_request_less_aev = test_request_less_aev
        # --- tuning knobs (same defaults as ChargingIntegratedEnvironment) ---
        # Reward rates are expressed per hour and converted to this environment's
        # real 30-second epoch. The synthetic environment used the same numeric
        # penalties per abstract step, which overwhelms NYC fare revenue.
        self.reward_epoch_hours = self.EPOCH_LENGTH / 3600.0
        self.charging_penalty_per_hour = 2.0
        self.idle_penalty_per_hour = 4.0
        self.charging_penalty = self.charging_penalty_per_hour * self.reward_epoch_hours
        self.idle_penalty = self.idle_penalty_per_hour * self.reward_epoch_hours
        self.charging_reward_noise = 0.2 * self.reward_epoch_hours
        self.movement_reward_noise_std = 0.005
        self.service_event_reward_noise_std = 0.02
        self.ev_charging_extra_penalty = 0.1 * self.reward_epoch_hours
        self.aev_charging_extra_penalty = 0.01 * self.reward_epoch_hours
        self.idle_vehicle_reward = -self.idle_penalty
        self.waiting_vehicle_reward = -self.idle_penalty
        self.adp_value = 1.0
        self.unserved_penalty = 50
        self.idle_vehicle_requirement = 1
        self.ev_model_basis = "Tesla Model 3 standard range"
        self.battery_capacity_kwh = 51.25
        self.battery_degradation = 0.10
        self.ev_consumption_wh_per_mile = 230.0
        self.ev_consumption_kwh_per_km = (self.ev_consumption_wh_per_mile / 1000.0) / 1.609344
        self.average_velocity_mph = 11.21
        self.average_velocity_kmph = self.average_velocity_mph * 1.609344
        self.ev_charge_rate_kw = 20.0
        self.charge_target_soc = 0.80
        self.charge_topup_soc = 0.05
        self.min_charging_session_minutes = 5.0
        self.max_charging_session_minutes = 120.0
        self.chargeincrease_per_epoch = (
            self.ev_charge_rate_kw * (self.EPOCH_LENGTH / 3600.0)
        ) / self.battery_capacity_kwh
        self.min_charging_session_epochs = max(
            1, int(math.ceil(self.min_charging_session_minutes * 60.0 / self.EPOCH_LENGTH))
        )
        self.max_charging_session_epochs = max(
            self.min_charging_session_epochs,
            int(math.ceil(self.max_charging_session_minutes * 60.0 / self.EPOCH_LENGTH)),
        )
        self.min_battery_level = 0.2
        self.charge_duration = min(
            self.max_charging_session_epochs,
            max(
                self.min_charging_session_epochs,
                int(math.ceil(
                    max(0.0, self.charge_target_soc - self.min_battery_level)
                    / max(self.chargeincrease_per_epoch, 1e-6)
                )),
            ),
        )
        self.chargeincrease_whole = self.chargeincrease_per_epoch * self.charge_duration
        self.charge_finished = 0.0
        self.penalty_reject_requestnum = 3
        self.rejection_loss: list = []
        self.rejection_pretrained = False
        self.minimum_charging_level = 0.2
        self.rebalance_battery_threshold = 0.3
        self.must_charge_battery_threshold = 0.15
        self.no_reloc_battery_threshold = self.must_charge_battery_threshold
        self.heuristic_battery_threshold = heuristic_battery_threshold
        self.battery_consum = self.ev_consumption_kwh_per_km / self.battery_capacity_kwh
        self.operating_cost_per_km = float(operating_cost_per_km)
        if not math.isfinite(self.operating_cost_per_km) or self.operating_cost_per_km < 0.0:
            raise ValueError("operating_cost_per_km must be a finite nonnegative value")
        # Compatibility alias for older value functions and checkpoints.
        # Costs are positive parameters; rewards carry the negative sign.
        self.movingpenalty = -self.operating_cost_per_km
        self.learning_reloc_penalty_base = 5.0
        self.learning_reloc_penalty_per_km = 0.5
        self.charging_wait_penalty_per_hour = 8.0
        self.charging_wait_penalty_per_step = self.charging_wait_penalty_per_hour * self.reward_epoch_hours
        self.learning_wait_penalty = self.charging_wait_penalty_per_step
        self.charging_wait_penalty_total = 0.0
        self.charging_wait_steps = 0
        self.charging_wait_observations: list[dict] = []
        self._charging_queue_arrivals: dict[tuple[int, int], dict] = {}
        self.qmatrix_diagnostic_interval = 50
        self._last_qmatrix_diagnostic_keys: set[tuple[int, int, bool]] = set()
        self.myopic_trip_cost_per_km = self.operating_cost_per_km
        self.myopic_time_cost_per_epoch = 0.05
        self.unserve_penalty = -0.5
        self.complete_ratio_reward = 0.5
        self.request_generation_rate = 0.8
        self.episode_length = max(1, int(episode_length)) if episode_length is not None else self.simulation_period
        self.decision_mode = "integrated"
        self.decision_mode_set = {"integrated", "aev_first", "ev_first"}
        self.use_range_requests = True  # True: range-based, False: all feasible
        self.assignmentrange = 5.0  # radius in km for range-based request filtering
        self.request_top_k = 128  # cap per-vehicle feasible requests to keep q-value batches bounded
        self.charge_action_range_km = 5
        self.zone_action_range_km = 10.0
        self.charge_top_k = 20
        self.zone_top_k = 8
        self.heuristic_use_scale = True

        # --- precompute distance matrix (km) ---
        self._build_distance_matrix()

        # --- charging stations ---
        self.charging_manager = ChargingStationManager()
        self.station_csv = station_csv or os.path.join(_base, "nyc_charging_stations.csv")
        self._setup_charging_stations()
        self.num_stations = len(self.charging_manager.stations)
        self._last_matrix_request_ids: list[int] = []
        self._last_matrix_charge_station_ids: list[int] = []
        self._last_matrix_zone_indices: list[int] = []
        self._last_matrix_zone_target_ids: list[int] = []
        self._last_matrix_num_requests = 0
        self._last_matrix_num_stations = 0
        self._last_matrix_num_zones = 0
        self._last_vehicle_action_graph_neighbours: dict[int, list[dict]] = {}
        self._last_vehicle_action_graph_neighbour_step: int | None = None
        self._last_vehicle_action_graph_neighbour_signature = None
        # The EV matrix exposes one outside action (the final ``wait`` column),
        # but executing that action means relocating according to the EV MNL
        # policy.  Sample its destination before Q evaluation and reuse it at
        # execution time so the learned edge and the realised transition agree.
        self._ev_default_relocation_cache_step: float | None = None
        self._ev_default_relocation_targets: dict[int, int] = {}
        self._ev_default_relocation_probabilities: dict[int, dict[int, float]] = {}
        self.charge_stats: Dict[int, list] = {}

        # --- zone system ---
        self.num_zones = 4
        self._prior_zone_dist_target = [1.0 / self.num_zones] * self.num_zones
        self.zoneinfo = {"1": "Surge", "2": "HighDemand", "3": "CityCenter", "4": "Normal"}
        self.loc_to_zone: Dict[int, int] = {}
        self.zone_to_locs: Dict[int, list] = {}
        self.surge_zone_locs: set = set()
        self.high_demand_zone_locs: set = set()
        self.city_center_zone_locs: set = set()
        self._init_zones()

        # --- hotspot locations (zone_ids of high-demand centroids) ---
        self.hotspot_locations: list = []  # compatibility alias for relocation targets
        self.hotspot_locations_num = 0
        self.relocation_target_ids: list[int] = []
        self.aux_zone_ids: list[int] = []
        self.aux_zone_id_to_index: Dict[int, int] = {}
        self.aux_zone_dim: int = 0

        # --- vehicles ---
        self.vehicles: Dict[int, dict] = {}
        self.storeactions: Dict[int, object | None] = {}
        self.storeactions_ev: Dict[int, object | None] = {}

        # --- request state ---
        self.active_requests: Dict[int, Request] = {}
        self.completed_requests: list = []
        self.completed_requests_ev: list = []
        self.completed_request_time_records: list = []
        self.rejected_requests: list = []
        self.ev_requests: list = []
        self.request_counter = 0
        self.whole_req = 0
        self.whole_req_num = 0
        self.request_value_sum = 0.0
        self.reject_number: Dict[int, int] = {}
        self.assignmentnumber: Dict[int, int] = {}
        # --- tracking ---
        self.request_generation_history: list = []
        self.last_generated_requests = 0
        self.last_generated_request_time = None
        self.vehicle_position_history: Dict[int, list] = {}
        self.charging_usage_history: list = []
        self.rebalancing_assignments_per_step: list = []
        self.rebalancing_whole: list = []
        self.total_rebalancing_calls = 0
        self.current_online = 0
        self.daily_online_history: list = []
        self.period_dropout_counts: list = []
        self.hourly_zone_vehicle_snapshots: dict = {}
        self.hourly_zone_charge_station_snapshots: dict = {}
        self.current_period_dropout_count = 0
        self.idle_charging_num: Dict[int, int] = {}
        self.station_pressure_snapshot_count = 0
        self.station_pressure_mean_sum = 0.0
        self.station_pressure_ratio_mean_sum = 0.0
        self.max_station_pressure = 0
        self.max_station_pressure_station_id = -1
        self.max_station_pressure_ratio = 0.0
        self.max_station_pressure_ratio_station_id = -1
        # --- Q-learning / NN ---
        self.q_table: dict = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.value_function = None
        self.value_function_ev = None
        self.zone_dataset = zone_dataset
        # --- EV behaviour ---
        self.heuevfirst = False
        self.ev_last_completed_time: dict = {}
        self.ev_last_accepted_time: dict = {}
        self.ev_current_idle_start_time: dict = {}
        self.ev_idle_durations: list = []
        self.ev_consecutive_rejections: dict = {}
        self.ev_penalty_until_time: dict = {}
        self.ev_penalty_duration = 2
        self.ride_acceptance_asc = 1.810
        self.ride_acceptance_beta_idle_min = -0.017
        self.ride_acceptance_beta_pickup_min = -0.050
        self.ride_acceptance_beta_surge = 0.101
        self.ride_acceptance_noise_std = 0.0
        self.recourse_variant = "legacy"
        self.rejection_logit_shift = 0.0
        self.common_random_numbers = False
        self._recourse_experiment_seed = int(random_seed or 0)
        self.surge_min_multiplier = 1.0
        self.surge_max_multiplier = 5.0
        self.rejection_penalty_base = float(rejection_penalty_base)
        self.rejection_penalty_per_km = float(rejection_penalty_per_km)
        if rejection_penalty_final_value_ratio is None or float(rejection_penalty_final_value_ratio) < 0.0:
            self.rejection_penalty_final_value_ratio = None
        else:
            self.rejection_penalty_final_value_ratio = float(rejection_penalty_final_value_ratio)
        self.rejection_reward_total = 0.0
        self.rejection_reward_count = 0
        self.step_rejection_reward_total = 0.0
        self.step_rejection_reward_count = 0
        self.ev_offer_count = 0
        self.ev_eligible_decision_count = 0
        self.ev_rejected_request_ids: set = set()
        self.ev_rejection_times: dict = {}
        self.ev_rejected_recovered_same_epoch_ids: set = set()
        self.ev_rejected_picked_up_by_aev_ids: set = set()
        self.ev_rejected_completed_ids: set = set()
        self.recovery_delays: list[float] = []
        self.aev_request_assignment_count = 0
        self.aev_recourse_assignment_count = 0
        self.residual_request_count = 0
        self.residual_request_served_count = 0
        self.rejected_request_served_count = 0
        self.unoffered_request_count = 0
        self.unoffered_request_served_count = 0
        self._current_ev_stage_request_ids: set = set()
        self._current_ev_offered_request_ids: set = set()
        self._stage1_recourse_target_by_transition: dict = {}
        self._same_epoch_blocked_request_ids: set = set()
        self.ev_charge_soc_threshold = 0.25
        self.ev_charge_soc_slope = 12.0
        self.ev_station_choice_beta = 1.0
        # Xie, Liu, and Chen (2023), Eq. (20) simulation-policy parameters.
        self.relocation_beta_match = 0.08
        self.relocation_beta_cost = 0.1
        self.relocation_cost_u = 0.6 * 0.92
        self.relocation_beta = self.relocation_beta_match  # compatibility alias
        self.ev_basesalary = 15
        self.dropoff_probability_rate = 0.4
        self.ev_dropoff_threshold = 0.35
        self.ev_dropoff_beta_0 = -1.0
        self.ev_dropoff_beta_idle = 0.12
        self.ev_dropoff_beta_satisfaction = -1.0
        self.ev_rejoin_gamma_0 = -1.6
        self.ev_rejoin_gamma_satisfaction = 0.8
        self.reject_number: Dict[int, int] = {}
        # --- time recording ---
        self.time_stats: Dict[str, list] = {
            'qvalue_with_network': [],
            'qvalue_without_network': [],
            'gurobi_solve': [],
            'gurobi_variables': [],
            'gurobi_constraints': [],
            'qvalue_scale_1': [],
            'qvalue_scale_2': [],
        }
        self._last_rebalancing_profile: Dict[str, float] = {}
        self._last_simulation_profile: Dict[str, float] = {}
        self._last_step_profile: Dict[str, float] = {}

        # --- initialise ---
        self.current_time: float = 0.0
        self.initialise_environment()

        total_station_capacity = sum(station.max_capacity for station in self.charging_manager.stations.values())
        print(f"✓ NYCEnvironment: {num_vehicles} vehicles, {self.num_stations} stations, "
              f"capacity={total_station_capacity}, coord_zones={self.NUM_ZONES}, "
              f"relocation_zones={self.hotspot_locations_num}, epoch={epoch_length_sec}s")

    # ==================================================================
    # Data loading helpers
    # ==================================================================


    def _load_zone_coordinates(self):
        """Load centroids, polygons, and boroughs from official TLC geometry."""
        try:
            import geopandas as gpd
        except Exception as exc:
            raise RuntimeError(
                "NYCEnvironment requires geopandas to load official TLC taxi-zone geometry"
            ) from exc

        geo_path = Path(self.zone_geometry_path)
        if not geo_path.exists():
            shp_dir = Path(__file__).resolve().parents[1] / "nyedata" / "taxi_zones_shp"
            shp_candidates = list(shp_dir.rglob("*.shp")) if shp_dir.exists() else []
            if not shp_candidates:
                shp_candidates = [self._download_official_taxi_zone_geometry(geo_path.parent)]
            geo_path = shp_candidates[0]

        zones_gdf = gpd.read_file(geo_path)
        if zones_gdf.crs is None or zones_gdf.crs.to_epsg() != 4326:
            zones_gdf = zones_gdf.to_crs(epsg=4326)

        loc_col = next((c for c in zones_gdf.columns if c.lower() in ("locationid", "location_id")), None)
        if loc_col is None:
            loc_col = [c for c in zones_gdf.columns if c != "geometry"][0]
        borough_col = next((c for c in zones_gdf.columns if c.lower() in ("borough", "boro_name", "boroname")), None)

        zones_gdf["zone_id"] = pd.to_numeric(zones_gdf[loc_col], errors="coerce")
        zones_gdf = zones_gdf[zones_gdf["zone_id"].notna()].copy()
        zones_gdf["zone_id"] = zones_gdf["zone_id"].astype(int)
        zones_gdf = zones_gdf[
            zones_gdf.geometry.notna() & ~zones_gdf.geometry.is_empty
        ].copy()

        centroid_gdf = zones_gdf.to_crs(epsg=3857)
        centroids = centroid_gdf.geometry.centroid.to_crs(epsg=4326)
        self.zone_coords = {
            int(zone_id): (float(centroid.y), float(centroid.x))
            for zone_id, centroid in zip(zones_gdf["zone_id"].astype(int), centroids)
        }
        self.real_zone_ids = set(self.zone_coords)
        self.real_zone_count = len(self.real_zone_ids)
        self.zone_boroughs: Dict[int, str] = {}
        self.manhattan_zone_ids: set[int] = set()

        if borough_col is not None:
            self.zone_boroughs = {
                int(row.zone_id): str(getattr(row, borough_col)).strip()
                for row in zones_gdf.itertuples(index=False)
            }
            self.manhattan_zone_ids = {
                zid for zid, borough in self.zone_boroughs.items()
                if borough.lower() == "manhattan"
            }
        else:
            self.manhattan_zone_ids = self.MANHATTAN_TLC_ZONE_IDS & self.real_zone_ids

        if not self.manhattan_zone_ids:
            self.manhattan_zone_ids = self.MANHATTAN_TLC_ZONE_IDS & self.real_zone_ids

        self.zone_geometries = zones_gdf[["zone_id", "geometry"]].copy()

    def _download_official_taxi_zone_geometry(self, data_dir: Path) -> Path:
        """Download and cache the official TLC taxi-zone shapefile.

        Servers do not need a pre-generated coordinate CSV or a committed copy
        of the geometry.  Only the known shapefile sidecars are extracted from
        TLC's archive, so untrusted archive paths can never escape the cache
        directory.
        """
        cache_dir = Path(data_dir) / "taxi_zones_shp" / "taxi_zones"
        shapefile_path = cache_dir / "taxi_zones.shp"
        required_suffixes = (".shp", ".shx", ".dbf", ".prj")
        if shapefile_path.exists() and all(
            (cache_dir / f"taxi_zones{suffix}").exists()
            for suffix in required_suffixes
        ):
            return shapefile_path

        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        print(
            "⬇ Official TLC taxi-zone geometry is missing; downloading "
            f"{self.TLC_TAXI_ZONE_SHAPEFILE_URL}",
            flush=True,
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix="taxi-zones-", dir=str(cache_dir.parent)
            ) as temporary_dir:
                temporary_path = Path(temporary_dir)
                archive_path = temporary_path / "taxi_zones.zip"
                request = urllib.request.Request(
                    self.TLC_TAXI_ZONE_SHAPEFILE_URL,
                    headers={"User-Agent": "adp-trainer/1.0"},
                )
                with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)

                extracted: dict[str, bytes] = {}
                allowed_suffixes = set(required_suffixes) | {".cpg"}
                with zipfile.ZipFile(archive_path) as archive:
                    for member in archive.infolist():
                        member_name = Path(member.filename).name
                        member_path = Path(member_name)
                        if (
                            member_path.stem.lower() == "taxi_zones"
                            and member_path.suffix.lower() in allowed_suffixes
                        ):
                            extracted[member_path.suffix.lower()] = archive.read(member)

                missing = [suffix for suffix in required_suffixes if suffix not in extracted]
                if missing:
                    raise RuntimeError(
                        "downloaded TLC archive is missing required files: "
                        + ", ".join(missing)
                    )

                cache_dir.mkdir(parents=True, exist_ok=True)
                for suffix, contents in extracted.items():
                    staged_path = temporary_path / f"taxi_zones{suffix}"
                    staged_path.write_bytes(contents)
                    staged_path.replace(cache_dir / f"taxi_zones{suffix}")
        except Exception as exc:
            raise FileNotFoundError(
                "Official TLC taxi-zone geometry is unavailable locally and automatic "
                f"download failed from {self.TLC_TAXI_ZONE_SHAPEFILE_URL}: {exc}"
            ) from exc

        print(f"✓ Cached official TLC taxi-zone geometry at {shapefile_path}", flush=True)
        return shapefile_path


    def map_zone(self, location_or_lat, lon: float | None = None) -> int:
        """Map either a real location id or a coordinate pair to a TLC zone id."""
        if lon is None and isinstance(location_or_lat, (int, np.integer)):
            loc_id = int(location_or_lat)
            return loc_id if loc_id in self.real_zone_ids else -1

        if lon is None and isinstance(location_or_lat, (tuple, list)) and len(location_or_lat) >= 2:
            lat = float(location_or_lat[0])
            lon = float(location_or_lat[1])
        else:
            lat = float(location_or_lat)
            lon = float(lon)

        if self.zone_geometries is not None and not self.zone_geometries.empty:
            mask = contains_xy(self.zone_geometries.geometry.values, lon, lat)
            if np.any(mask):
                return int(self.zone_geometries.loc[mask, "zone_id"].iloc[0])
        return self._nearest_zone(lat, lon, prefer_polygon=False)


    def prepare_zone_dataset(self):
        if self.zone_dataset is None:
            return
        print(f"📖 Loading preprocessed zone dataset from {self.zone_dataset} ...")
        self.zone_demand_profiles = np.load(self.zone_dataset, allow_pickle=True).item()
        print(f"   ✓ Loaded demand profiles for {len(self.zone_demand_profiles)} zones")








    @staticmethod
    def _find_default_parquet(base_dir: str) -> str:
        """Pick the first available parquet file, auto-downloading if none found."""
        parquet_dir = os.path.join(base_dir, 'parquet')
        if os.path.isdir(parquet_dir):
            for f in sorted(os.listdir(parquet_dir)):
                if f.endswith('.parquet'):
                    return os.path.join(parquet_dir, f)

        # No parquet found — try auto-download (default: 2025-01)
        print("⚠ No parquet file found, attempting auto-download (2025-01) ...")
        try:
            import requests as _req
            url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
            os.makedirs(parquet_dir, exist_ok=True)
            target = os.path.join(parquet_dir, "yellow_tripdata_2025-01.parquet")
            resp = _req.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            with open(target, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            print(f"\r   Downloading: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB "
                                  f"({100 * downloaded / total:.0f}%)", end='', flush=True)
            print(f"\n✓ Downloaded {target}")
            return target
        except Exception as e:
            raise FileNotFoundError(
                f"No parquet found under {base_dir} and auto-download failed: {e}\n"
                f"Run: cd nyedata/nye_simulation && python download_data.py"
            )

    @staticmethod
    def _read_normalized_demand_file(
        parquet_path: str | Path,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Read one Yellow or HVFHV parquet into the Yellow-compatible schema.

        HVFHV request time is preferred over pickup time because it represents
        when the passenger entered the dispatch system.  Trips that actually
        matched another independently booked passenger are excluded; unmatched
        shared-product requests remain valid one-vehicle/one-request trips.
        """

        yellow_columns = [
            'tpep_pickup_datetime',
            'PULocationID',
            'DOLocationID',
            'fare_amount',
            'trip_distance',
        ]
        start_ts = pd.Timestamp(start_date) if start_date is not None else None
        stop_ts = (
            pd.Timestamp(end_date if end_date is not None else start_date)
            + pd.Timedelta(days=1)
            if start_date is not None
            else None
        )
        yellow_filters = None
        if start_ts is not None and stop_ts is not None:
            yellow_filters = [
                ('tpep_pickup_datetime', '>=', start_ts),
                ('tpep_pickup_datetime', '<', stop_ts),
            ]
        try:
            frame = pd.read_parquet(
                parquet_path,
                columns=yellow_columns,
                filters=yellow_filters,
            )
        except Exception:
            hvfhv_columns = [
                'request_datetime',
                'pickup_datetime',
                'PULocationID',
                'DOLocationID',
                'base_passenger_fare',
                'trip_miles',
                'shared_match_flag',
            ]
            hvfhv_filters = None
            if start_ts is not None and stop_ts is not None:
                hvfhv_filters = [
                    ('request_datetime', '>=', start_ts),
                    ('request_datetime', '<', stop_ts),
                ]
            try:
                hvfhv = pd.read_parquet(
                    parquet_path,
                    columns=hvfhv_columns,
                    filters=hvfhv_filters,
                )
            except Exception as hvfhv_error:
                raise ValueError(
                    f"Unsupported NYC demand parquet schema: {parquet_path}. "
                    "Expected Yellow Taxi or HVFHV trip-record columns."
                ) from hvfhv_error

            shared_match = (
                hvfhv['shared_match_flag']
                .fillna('N')
                .astype(str)
                .str.strip()
                .str.upper()
            )
            hvfhv = hvfhv.loc[shared_match.ne('Y')].copy()
            request_datetime = pd.to_datetime(
                hvfhv['request_datetime'], errors='coerce'
            )
            pickup_datetime = pd.to_datetime(
                hvfhv['pickup_datetime'], errors='coerce'
            )
            normalized = pd.DataFrame({
                'tpep_pickup_datetime': request_datetime.fillna(pickup_datetime),
                'PULocationID': hvfhv['PULocationID'],
                'DOLocationID': hvfhv['DOLocationID'],
                'fare_amount': hvfhv['base_passenger_fare'],
                'trip_distance': hvfhv['trip_miles'],
            })
            normalized['demand_source'] = 'hvfhv_nonshared'
            return normalized

        frame = frame.copy()
        frame['demand_source'] = 'yellow'
        return frame

    def _load_demand_data(self, parquet_path: str | list[str] | None = None) -> pd.DataFrame:
        """Load, normalize, and clean Yellow plus optional non-pooled HVFHV."""
        path = parquet_path or self.parquet_path
        path_list = list(path) if isinstance(path, (list, tuple)) else [path]
        cache_key = (
            tuple(str(Path(single_path).expanduser().resolve()) for single_path in path_list),
            bool(getattr(self, 'full_demand', False)),
            bool(self.ifonlymanhatten),
            tuple(sorted(int(zone_id) for zone_id in self.real_zone_ids)),
            tuple(sorted(int(zone_id) for zone_id in self.manhattan_zone_ids)) if self.ifonlymanhatten else (),
            str(self.start_date),
            str(self.end_date),
        )
        demand_cache = getattr(NYCEnvironment, "_demand_data_cache", {})
        cached = demand_cache.get(cache_key)
        if cached is not None:
            return cached.copy(deep=False)

        if isinstance(path, (list, tuple)):
            frames = [
                self._read_normalized_demand_file(
                    single_path,
                    self.start_date,
                    self.end_date,
                )
                for single_path in path
            ]
            df = pd.concat(frames, ignore_index=True)
        else:
            df = self._read_normalized_demand_file(
                path,
                self.start_date,
                self.end_date,
            )
        # basic cleaning
        df = df[df['trip_distance'] > 0]
        df = df[df['fare_amount'] > 0]
        df = df[df['trip_distance'] <= 50]
        df = df[df['fare_amount'] <= 500]
        if self.real_zone_ids:
            df = df[df['PULocationID'].isin(self.real_zone_ids)]
            df = df[df['DOLocationID'].isin(self.real_zone_ids)]
        if self.ifonlymanhatten:
            if not self.manhattan_zone_ids:
                raise RuntimeError("Manhattan-only demand requested, but Manhattan zone ids are empty.")
            df = df[df['PULocationID'].isin(self.manhattan_zone_ids)]
            df = df[df['DOLocationID'].isin(self.manhattan_zone_ids)]
        df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        expected_year_months = set()
        for single_path in path_list:
            match = re.search(r'(\d{4})-(\d{2})', Path(single_path).name)
            if match is not None:
                expected_year_months.add(f"{match.group(1)}-{match.group(2)}")
        if expected_year_months:
            pickup_year_month = df['pickup_datetime'].dt.strftime('%Y-%m')
            df = df[pickup_year_month.isin(expected_year_months)]
        df['pickup_date'] = df['pickup_datetime'].dt.normalize()
        if self.start_date is not None:
            start_ts = pd.Timestamp(self.start_date)
            end_ts = pd.Timestamp(self.end_date)
            df = df[(df['pickup_date'] >= start_ts) & (df['pickup_date'] <= end_ts)]
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_minute'] = df['pickup_datetime'].dt.minute
        df['pickup_second_of_day'] = (
            df['pickup_datetime'].dt.hour * 3600
            + df['pickup_datetime'].dt.minute * 60
            + df['pickup_datetime'].dt.second
        )
        # Historical duration is deliberately not used.  The simulator has a
        # configured travel speed; TLC ``trip_distance`` determines both trip
        # duration and energy consumption consistently under that speed.
        cleaned = df.sort_values('pickup_datetime').reset_index(drop=True)
        demand_cache[cache_key] = cleaned
        NYCEnvironment._demand_data_cache = demand_cache
        return cleaned.copy(deep=False)

    def _ensure_demand_loaded(self):
        if self._demand_df is None:
            if isinstance(self.parquet_path, (list, tuple)):
                file_names = ", ".join(Path(path).name for path in self.parquet_path)
                print(f"📖 Loading demand data from {len(self.parquet_path)} parquet files: {file_names}")
            else:
                print(f"📖 Loading demand data from {self.parquet_path} ...")
            self._demand_df = self._load_demand_data()
            self._demand_df['pickup_date'] = pd.to_datetime(self._demand_df['pickup_date']).dt.normalize()
            date_counts = self._demand_df.groupby('pickup_date').size().sort_index()
            self._available_demand_dates = [
                self._normalize_demand_date(date_value) for date_value in date_counts.index.tolist()
            ]
            print(f"   ✓ {len(self._demand_df):,} trips loaded")
            if 'demand_source' in self._demand_df.columns:
                source_counts = self._demand_df.groupby('demand_source').size().sort_index()
                source_summary = ", ".join(
                    f"{source}:{int(count):,}"
                    for source, count in source_counts.items()
                )
                print(f"   Demand sources: {source_summary}")
            if not date_counts.empty:
                date_summary = ", ".join(
                    f"{pd.Timestamp(date_value).date()}:{int(count):,}"
                    for date_value, count in date_counts.items()
                )
                print(f"   Demand dates ({len(date_counts)}): {date_summary}")

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if value is None:
            return None
        return pd.to_datetime(value).date()

    @staticmethod
    def _normalize_demand_date(value):
        if value is None:
            return None
        return pd.Timestamp(value).normalize()

    def _current_day_index(self, current_time=None) -> int:
        if current_time is None:
            current_time = self.current_time
        demand_step = max(0.0, float(current_time) - 1.0)
        return int(self.episode_day_index + int(demand_step // max(1, self.simulation_period)))

    def _day_step_offset(self, current_time=None) -> float:
        if current_time is None:
            current_time = self.current_time
        demand_step = max(0.0, float(current_time) - 1.0)
        return demand_step % max(1, self.simulation_period)

    def _current_date_label(self, current_time=None):
        self._ensure_demand_loaded()
        if not self._available_demand_dates:
            return None
        day_index = self._current_day_index(current_time)
        return self._available_demand_dates[day_index % len(self._available_demand_dates)]

    # ==================================================================
    # Distance / travel time
    # ==================================================================

    def _build_distance_matrix(self):
        """Precompute pairwise distances (km) between zone centroids."""
        ids = sorted(self.zone_coords.keys())
        max_id = max(ids)
        self.distance_matrix = np.zeros((max_id + 1, max_id + 1), dtype=np.float32)
        for i in ids:
            lat_i, lon_i = self.zone_coords[i]
            for j in ids:
                if i != j:
                    lat_j, lon_j = self.zone_coords[j]
                    self.distance_matrix[i, j] = _haversine_km(lat_i, lon_i, lat_j, lon_j)

        self.travel_time_matrix = self.distance_matrix / self.average_velocity_kmph * 60.0

    def get_travel_time(self, source: int, destination: int) -> float:
        """Travel time in *epochs* (each epoch = EPOCH_LENGTH seconds)."""
        if source == destination:
            return 0.0
        try:
            minutes = float(self.travel_time_matrix[source, destination])
        except (IndexError, KeyError):
            return 1.0
        return max(1.0, minutes * 60.0 / self.EPOCH_LENGTH)  # at least 1 epoch

    def get_travel_time_minutes(self, source: int, destination: int) -> float:
        """Travel time in minutes."""
        if source == destination:
            return 0.0
        try:
            return float(self.travel_time_matrix[source, destination])
        except (IndexError, KeyError):
            return 3.0

    def get_distance_km(self, source: int, destination: int) -> float:
        if source == destination:
            return 0.0
        try:
            return float(self.distance_matrix[source, destination])
        except (IndexError, KeyError):
            return 1.0

    def get_next_location(self, source: int, destination: int) -> int:
        """One-hop greedy routing: move to the neighbour zone closest to destination."""
        if source == destination:
            return source
        # find the zone among source's neighbours that is closest to destination
        dest_lat, dest_lon = self.zone_coords.get(destination, (0, 0))
        best_zone = destination
        best_dist = float('inf')
        for zid, (lat, lon) in self.zone_coords.items():
            if zid == source:
                continue
            d_from_src = self.distance_matrix[source, zid] if source < self.distance_matrix.shape[0] and zid < self.distance_matrix.shape[1] else 999
            d_to_dest = self.distance_matrix[zid, destination] if zid < self.distance_matrix.shape[0] and destination < self.distance_matrix.shape[1] else 999
            # pick neighbour within ~3 km that minimises remaining distance
            if d_from_src <= 3.0 and d_to_dest < best_dist:
                best_dist = d_to_dest
                best_zone = zid
        return best_zone

    # ==================================================================
    # Zone system
    # ==================================================================

    def _init_zones(self):
        """Keep compatibility semantic groups, but derive them from real demand hotspots."""
        self.loc_to_zone = {}
        self.zone_to_locs = {group: [] for group in range(self.num_zones)}

        self._ensure_demand_loaded()
        demand_rank = self._demand_df.groupby('PULocationID').size().sort_values(ascending=False)
        ranked_zone_ids = [int(zid) for zid in demand_rank.index if int(zid) in self.real_zone_ids]
        if not ranked_zone_ids:
            ranked_zone_ids = sorted(self.real_zone_ids)

        surge_cut = max(1, min(16, len(ranked_zone_ids)))
        high_cut = max(surge_cut, min(surge_cut + 32, len(ranked_zone_ids)))
        center_cut = max(high_cut, min(high_cut + 48, len(ranked_zone_ids)))

        self.surge_zone_locs = set(ranked_zone_ids[:surge_cut])
        self.high_demand_zone_locs = set(ranked_zone_ids[surge_cut:high_cut])
        self.city_center_zone_locs = set(ranked_zone_ids[high_cut:center_cut])
        normal_locs = set(self.real_zone_ids) - self.surge_zone_locs - self.high_demand_zone_locs - self.city_center_zone_locs

        for group_idx, locs in enumerate([
            self.surge_zone_locs,
            self.high_demand_zone_locs,
            self.city_center_zone_locs,
            normal_locs,
        ]):
            self.zone_to_locs[group_idx] = sorted(locs)
            for zid in locs:
                self.loc_to_zone[int(zid)] = group_idx

    def get_zone_id(self, location_id: int) -> int:
        return self.loc_to_zone.get(int(location_id), 0)

    def get_real_zone_id(self, location_or_lat, lon: float | None = None) -> int:
        return self.map_zone(location_or_lat, lon)

    def get_aux_zone_id(self, location_or_lat, lon: float | None = None) -> int:
        real_zone_id = self.get_real_zone_id(location_or_lat, lon)
        return self.aux_zone_id_to_index.get(int(real_zone_id), 0)

    def get_distribution_zone_index(self, location_or_lat, lon: float | None = None) -> int | None:
        """Return the 0-based zone index used by zone-distribution vectors."""
        if self.aux_zone_dim > 0:
            real_zone_id = self.get_real_zone_id(location_or_lat, lon)
            zone_idx = self.aux_zone_id_to_index.get(int(real_zone_id))
            return int(zone_idx) if zone_idx is not None else None

        zone_id = self.get_zone_id(location_or_lat)
        zone_id = int(zone_id)
        return zone_id if 0 <= zone_id < self.num_zones else None

    def get_zone_embedding_id(self, location_or_lat, lon: float | None = None) -> int:
        """Return the 1-based zone id consumed by the neural zone embedding."""
        zone_idx = self.get_distribution_zone_index(location_or_lat, lon)
        return int(zone_idx) + 1 if zone_idx is not None else 0

    def get_hour_of_day(self, current_time: float = None) -> float:
        """Return the real-world hour of day (0-24) for a given simulation step.

        Args:
            current_time: simulation epoch index.  None → self.current_time.
        Returns:
            Hour of day as float, e.g. 7.5 means 07:30.
        """
        if current_time is None:
            current_time = self.current_time
        second_of_day = self.START_EPOCH + self._day_step_offset(current_time) * self.EPOCH_LENGTH
        return second_of_day / 3600.0  # convert seconds → hours

    def get_zone_locations(self, zone_id: int) -> list:
        return list(self.zone_to_locs.get(int(zone_id), []))

    # ==================================================================
    # Charging stations
    # ==================================================================

    def _setup_charging_stations(self):
        """Load charging stations from CSV and map them into real TLC zones."""
        if self.ifonlymanhatten:
            if not self.manhattan_zone_ids:
                raise RuntimeError("Manhattan-only station setup requested, but Manhattan zone ids are empty.")
            allowed_zone_ids = self.manhattan_zone_ids
        else:
            allowed_zone_ids = self.real_zone_ids
        if os.path.exists(self.station_csv):
            df = pd.read_csv(self.station_csv)
            if {'station_id', 'lat', 'lon'}.issubset(df.columns):
                for _, row in df.iterrows():
                    sid = int(row['station_id'])
                    lat, lon = float(row['lat']), float(row['lon'])
                    cap = max(1, int(round(4 * self.station_capacity_scale)))
                    zone_id = self.map_zone(lat, lon)
                    if zone_id < 0 or zone_id not in allowed_zone_ids:
                        continue
                    self.charging_manager.add_station(sid, zone_id, cap)
            elif {'Latitude', 'Longitude'}.issubset(df.columns):
                city_names = {
                    'NEW YORK', 'MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'THE BRONX',
                    'STATEN ISLAND', 'LONG ISLAND CITY', 'LIC', 'ASTORIA', 'FLUSHING',
                    'JAMAICA', 'RIDGEWOOD', 'FAR ROCKAWAY', 'BAYSIDE', 'COLLEGE POINT',
                    'CORONA', 'FOREST HILLS', 'FRESH MEADOWS', 'JACKSON HEIGHTS',
                    'REGO PARK', 'ROCKAWAY PARK', 'WHITESTONE', 'EAST ELMHURST',
                    'ELMHURST', 'KEW GARDENS', 'MIDDLE VILLAGE', 'OZONE PARK',
                    'RICHMOND HILL', 'WOODSIDE', 'MIDLAND BEACH', 'SOUTH BEACH',
                    'RED HOOK',
                }
                if 'City' in df.columns:
                    city = df['City'].fillna('').astype(str).str.upper().str.strip()
                    df = df[city.isin(city_names)]
                for station_index, (_, row) in enumerate(df.iterrows(), start=1):
                    lat = pd.to_numeric(row.get('Latitude'), errors='coerce')
                    lon = pd.to_numeric(row.get('Longitude'), errors='coerce')
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    level1 = pd.to_numeric(row.get('EV Level1 EVSE Num', 0), errors='coerce')
                    level2 = pd.to_numeric(row.get('EV Level2 EVSE Num', 0), errors='coerce')
                    dc_fast = pd.to_numeric(row.get('EV DC Fast Count', 0), errors='coerce')
                    connector_count = sum(0 if pd.isna(v) else float(v) for v in (level1, level2, dc_fast))
                    cap = max(1, int(round(4 * self.station_capacity_scale)))
                    sid = int(row['ID']) if 'ID' in row and pd.notna(row.get('ID')) else station_index
                    zone_id = self.map_zone(float(lat), float(lon))
                    if zone_id < 0 or zone_id not in allowed_zone_ids:
                        continue
                    self.charging_manager.add_station(sid, zone_id, cap)
            else:
                raise ValueError(
                    f"Unsupported charging station CSV schema for {self.station_csv}. "
                    "Expected station_id/lat/lon or Latitude/Longitude columns."
                )
        else:
            # fallback: create evenly distributed stations
            zone_ids = sorted(allowed_zone_ids)
            step = max(1, len(zone_ids) // self.num_stations)
            for i in range(self.num_stations):
                zid = zone_ids[min(i * step, len(zone_ids) - 1)]
                cap = max(1, int(round(4 * self.station_capacity_scale)))
                self.charging_manager.add_station(i + 1, zid, cap)
        self.station_zone_ids = np.array(
            [self.charging_manager.stations[sid].location for sid in sorted(self.charging_manager.stations.keys())],
            dtype=np.int32,
        ) if self.charging_manager.stations else np.array([], dtype=np.int32)
        self.nearest_charging_distance = np.zeros(self.distance_matrix.shape[0], dtype=np.float32)
        if self.station_zone_ids.size > 0:
            self.nearest_charging_distance = np.min(self.distance_matrix[:, self.station_zone_ids], axis=1)
        self.idle_charging_num = {sid: s.max_capacity for sid, s in self.charging_manager.stations.items()}
        self.charge_stats = {sid: [] for sid in self.charging_manager.stations}

    def _nearest_zone(self, lat: float, lon: float, prefer_polygon: bool = True) -> int:
        """Return the containing zone if possible, otherwise the nearest centroid zone."""
        if prefer_polygon and self.zone_geometries is not None and not self.zone_geometries.empty:
            mask = contains_xy(self.zone_geometries.geometry.values, lon, lat)
            if np.any(mask):
                return int(self.zone_geometries.loc[mask, "zone_id"].iloc[0])
        best, best_d = 1, float('inf')
        for zid, (z_lat, z_lon) in self.zone_coords.items():
            d = (lat - z_lat) ** 2 + (lon - z_lon) ** 2
            if d < best_d:
                best_d = d
                best = zid
        return best

    # ==================================================================
    # Vehicle setup
    # ==================================================================

    def _setup_vehicles(self):
        saved_state = random.getstate()
        saved_np = np.random.get_state()
        if self.initial_random_seed is not None:
            day_offset = max(0, int(getattr(self, 'episode_day_index', 0) or 0))
            setup_seed = int(self.initial_random_seed) + day_offset
            random.seed(setup_seed)
            np.random.seed(setup_seed)

        if self.ifonlymanhatten:
            if not self.manhattan_zone_ids:
                raise RuntimeError(
                    "Manhattan-only vehicle setup requested, but Manhattan zone ids are empty."
                )
            valid_zones = sorted(self.manhattan_zone_ids)
        else:
            valid_zones = sorted(self.real_zone_ids)
        for i in range(self.num_vehicles):
            zone = random.choice(valid_zones)
            vtype = 1 if i < self.ev_num_vehicles else 2
            vtype_name = 'EV' if vtype == 1 else 'AEV'
            self.vehicles[i] = {
                'type': vtype,
                'type_name': vtype_name,
                'location': zone,
                'coordinates': self.zone_coords[zone],  # (lat, lon) for viz
            'zone_id': zone,
                'battery': random.uniform(0.8, 0.95),
                'charging_station': None,
                'charging_time_left': 0,
                'total_distance': 0.0,
                'charging_count': 0,
                'assigned_request': None,
                'passenger_onboard': None,
                'passenger_trip_distance_total': 0.0,
                'passenger_trip_distance_remaining': 0.0,
                'passenger_trip_distance_travelled': 0.0,
                'passenger_trip_start_coordinates': None,
                'service_earnings': 0.0,
                'daily_salary': 0.0,
                'salary_ratio': 0.0 if vtype == 1 else 1e4,
                'satisfaction': 0.0,
                'is_online': True,
                'offline_until_time': None,
                'whether_finishrequest': False,
                'rejected_requests': 0,
                'unserved_penalty': 0,
                'is_stationary': False,
                'stationary_duration': 0,
                'target_location': None,
                'charging_target': None,
                'idle_target': None,
                'idle_timer': 0,
                'continual_reject': 0,
                'penalty_timer': 0,
                'needs_emergency_charging': False,
                'no_charge_cooldown_until': 0,
            }
            self.storeactions[i] = None
            self.storeactions_ev[i] = None

        random.setstate(saved_state)
        np.random.set_state(saved_np)

    # ==================================================================
    # Initialise / reset
    # ==================================================================

    def initialise_environment(self):
        self.current_time = 0.0
        self._setup_vehicles()
        # build relocation target ids from active real TLC zones
        self._ensure_demand_loaded()
        if self.ifonlymanhatten:
            if not self.manhattan_zone_ids:
                raise RuntimeError("Manhattan-only relocation requested, but Manhattan zone ids are empty.")
            active_zone_ids = sorted(self.manhattan_zone_ids)
        else:
            active_zone_ids = sorted(self.real_zone_ids)
        self.relocation_target_ids = active_zone_ids
        self.hotspot_locations = list(self.relocation_target_ids)
        self.hotspot_locations_num = len(self.relocation_target_ids)
        self.aux_zone_ids = list(self.relocation_target_ids)
        self.aux_zone_id_to_index = {zid: idx for idx, zid in enumerate(self.aux_zone_ids)}
        self.aux_zone_dim = len(self.aux_zone_ids)
        if self.aux_zone_dim > 0:
            self._prior_zone_dist_target = [1.0 / self.aux_zone_dim] * self.aux_zone_dim
        print(f"   Relocation target zones: {self.hotspot_locations_num}")

    def reset(self):
        if self._available_demand_dates:
            self.episode_day_index = (self.episode_day_index + 1) % len(self._available_demand_dates)
        else:
            self.episode_day_index = 0
        self.current_time = 0.0
        self.request_value_sum = 0.0
        self.whole_req = 0
        self.ev_requests = []
        self.active_requests = {}
        self.whole_req_num = 0
        self.completed_requests = []
        self.completed_requests_ev = []
        self.completed_request_time_records = []
        self.rejected_requests = []
        self.request_counter = 0
        self.charge_finished = 0.0
        self.charge_stats = {sid: [] for sid in self.charging_manager.stations}
        self.charging_wait_penalty_total = 0.0
        self.charging_wait_steps = 0
        self.charging_wait_observations = []
        self._charging_queue_arrivals = {}
        self.rebalancing_assignments_per_step = []
        self.rebalancing_whole = []
        self.total_rebalancing_calls = 0
        self.storeactions = {}
        self.storeactions_ev = {}
        self.request_generation_history = []
        self.last_generated_requests = 0
        self.last_generated_request_time = None
        self.vehicle_position_history = {}
        self.charging_usage_history = []
        self._demand_cursor = 0
        self._demand_day_cache = None
        self._demand_day_cache_label = None
        self._setup_vehicles()
        self.current_online = len(self.vehicles)
        self.daily_online_history = [self.current_online]
        self.period_dropout_counts = []
        self.hourly_zone_vehicle_snapshots = {}
        self.hourly_zone_charge_station_snapshots = {}
        self.station_pressure_snapshot_count = 0
        self.station_pressure_mean_sum = 0.0
        self.station_pressure_ratio_mean_sum = 0.0
        self.max_station_pressure = 0
        self.max_station_pressure_station_id = -1
        self.max_station_pressure_ratio = 0.0
        self.max_station_pressure_ratio_station_id = -1
        self._bayes_step_contexts = {}
        self._last_qmatrix_diagnostic_keys = set()
        self._ev_default_relocation_cache_step = None
        self._ev_default_relocation_targets = {}
        self._ev_default_relocation_probabilities = {}
        self.current_period_dropout_count = 0
        # Reset EV tracking
        self.ev_last_completed_time = {}
        self.ev_last_accepted_time = {}
        self.ev_current_idle_start_time = {}
        self.ev_idle_durations = []
        self.ev_consecutive_rejections = {}
        self.ev_penalty_until_time = {}
        self.rejection_reward_total = 0.0
        self.rejection_reward_count = 0
        self.step_rejection_reward_total = 0.0
        self.step_rejection_reward_count = 0
        self.ev_offer_count = 0
        self.ev_eligible_decision_count = 0
        self.ev_rejected_request_ids = set()
        self.ev_rejection_times = {}
        self.ev_rejected_recovered_same_epoch_ids = set()
        self.ev_rejected_picked_up_by_aev_ids = set()
        self.ev_rejected_completed_ids = set()
        self.recovery_delays = []
        self.aev_request_assignment_count = 0
        self.aev_recourse_assignment_count = 0
        self.residual_request_count = 0
        self.residual_request_served_count = 0
        self.rejected_request_served_count = 0
        self.unoffered_request_count = 0
        self.unoffered_request_served_count = 0
        self._current_ev_stage_request_ids = set()
        self._current_ev_offered_request_ids = set()
        self._stage1_recourse_target_by_transition = {}
        self._same_epoch_blocked_request_ids = set()
        current_date_label = self._current_date_label()
        if current_date_label is not None:
            self._prepare_day_demand(current_date_label)
            print(
                f"✓ Episode demand date: {current_date_label.date()} "
                f"({len(self._demand_day_cache):,} trips)"
            )
        self._record_hourly_zone_vehicle_snapshot(current_time=self.current_time)
        return self.get_initial_states()

    def _compute_real_zone_vehicle_type_counts(self):
        zone_total_counts = {int(zone_id): 0 for zone_id in sorted(self.real_zone_ids)}
        zone_ev_counts = {int(zone_id): 0 for zone_id in sorted(self.real_zone_ids)}
        zone_aev_counts = {int(zone_id): 0 for zone_id in sorted(self.real_zone_ids)}

        for vehicle in self.vehicles.values():
            if not vehicle.get('is_online', True):
                continue
            real_zone_id = vehicle.get('zone_id', vehicle.get('location', -1))
            real_zone_id = self.get_real_zone_id(real_zone_id)
            if real_zone_id not in zone_total_counts:
                continue
            zone_total_counts[real_zone_id] += 1
            vehicle_type = vehicle.get('type', 0)
            if vehicle_type == 1:
                zone_ev_counts[real_zone_id] += 1
            elif vehicle_type == 2:
                zone_aev_counts[real_zone_id] += 1

        return zone_total_counts, zone_ev_counts, zone_aev_counts

    def _compute_zone_charge_station_counts(self):
        zone_station_counts = {}
        for station in self.charging_manager.stations.values():
            zone_id = int(station.location)
            bucket = zone_station_counts.setdefault(
                zone_id,
                {
                    'station_count': 0,
                    'total_capacity': 0,
                    'queue_vehicle_count': 0,
                },
            )
            bucket['station_count'] += 1
            bucket['total_capacity'] += int(station.max_capacity)
            bucket['queue_vehicle_count'] += int(len(station.charging_queue) + len(station.charging_queue_notarrived))
        return zone_station_counts

    def _compute_station_pressure_snapshot(self):
        station_count = 0
        pressure_sum = 0.0
        pressure_ratio_sum = 0.0
        max_pressure = 0
        max_pressure_station_id = -1
        max_pressure_ratio = 0.0
        max_pressure_ratio_station_id = -1

        for sid, station in self.charging_manager.stations.items():
            pressure = int(
                len(getattr(station, 'current_vehicles', []) or [])
                + len(getattr(station, 'charging_queue', []) or [])
                + len(getattr(station, 'charging_queue_notarrived', []) or [])
            )
            capacity = max(1, int(getattr(station, 'max_capacity', 1) or 1))
            pressure_ratio = float(pressure) / float(capacity)
            station_id = int(getattr(station, 'id', sid))

            station_count += 1
            pressure_sum += float(pressure)
            pressure_ratio_sum += pressure_ratio

            if pressure > max_pressure:
                max_pressure = pressure
                max_pressure_station_id = station_id
            if pressure_ratio > max_pressure_ratio:
                max_pressure_ratio = pressure_ratio
                max_pressure_ratio_station_id = station_id

        divisor = float(station_count) if station_count > 0 else 1.0
        return {
            'station_count': station_count,
            'mean_station_pressure': pressure_sum / divisor,
            'mean_station_pressure_ratio': pressure_ratio_sum / divisor,
            'max_station_pressure': max_pressure,
            'max_station_pressure_station_id': max_pressure_station_id,
            'max_station_pressure_ratio': max_pressure_ratio,
            'max_station_pressure_ratio_station_id': max_pressure_ratio_station_id,
        }

    def _record_station_pressure_snapshot(self):
        snapshot = self._compute_station_pressure_snapshot()
        self.station_pressure_snapshot_count += 1
        self.station_pressure_mean_sum += float(snapshot['mean_station_pressure'])
        self.station_pressure_ratio_mean_sum += float(snapshot['mean_station_pressure_ratio'])

        if int(snapshot['max_station_pressure']) > int(self.max_station_pressure):
            self.max_station_pressure = int(snapshot['max_station_pressure'])
            self.max_station_pressure_station_id = int(snapshot['max_station_pressure_station_id'])
        if float(snapshot['max_station_pressure_ratio']) > float(self.max_station_pressure_ratio):
            self.max_station_pressure_ratio = float(snapshot['max_station_pressure_ratio'])
            self.max_station_pressure_ratio_station_id = int(snapshot['max_station_pressure_ratio_station_id'])

        return snapshot

    def _record_hourly_zone_vehicle_snapshot(self, current_time=None):
        current_date = self._current_date_label(current_time)
        if current_date is None:
            return

        current_hour = int(self.get_hour_of_day(current_time))
        zone_total_counts, zone_ev_counts, zone_aev_counts = self._compute_real_zone_vehicle_type_counts()
        for zone_id in sorted(zone_total_counts.keys()):
            key = (current_date, current_hour, int(zone_id))
            bucket = self.hourly_zone_vehicle_snapshots.setdefault(
                key,
                {
                    'date': current_date,
                    'hour': current_hour,
                    'zone_id': int(zone_id),
                    'snapshot_count': 0,
                    'total_vehicles_sum': 0.0,
                    'ev_vehicles_sum': 0.0,
                    'aev_vehicles_sum': 0.0,
                },
            )
            bucket['snapshot_count'] += 1
            bucket['total_vehicles_sum'] += float(zone_total_counts.get(zone_id, 0))
            bucket['ev_vehicles_sum'] += float(zone_ev_counts.get(zone_id, 0))
            bucket['aev_vehicles_sum'] += float(zone_aev_counts.get(zone_id, 0))

    def _record_hourly_zone_charge_station_snapshot(self, current_time=None):
        current_date = self._current_date_label(current_time)
        if current_date is None:
            return

        current_hour = int(self.get_hour_of_day(current_time))
        zone_station_counts = self._compute_zone_charge_station_counts()
        for zone_id in sorted(zone_station_counts.keys()):
            zone_stats = zone_station_counts[zone_id]
            key = (current_date, current_hour, int(zone_id))
            bucket = self.hourly_zone_charge_station_snapshots.setdefault(
                key,
                {
                    'date': current_date,
                    'hour': current_hour,
                    'zone_id': int(zone_id),
                    'snapshot_count': 0,
                    'station_count_sum': 0.0,
                    'total_capacity_sum': 0.0,
                    'queue_vehicle_count_sum': 0.0,
                },
            )
            bucket['snapshot_count'] += 1
            bucket['station_count_sum'] += float(zone_stats.get('station_count', 0))
            bucket['total_capacity_sum'] += float(zone_stats.get('total_capacity', 0))
            bucket['queue_vehicle_count_sum'] += float(zone_stats.get('queue_vehicle_count', 0))

    # ==================================================================
    # Demand generation from real data
    # ==================================================================

    def _prepare_day_demand(self, date_label=None):
        """Cache demand for a specific real pickup date sorted by second-of-day."""
        self._ensure_demand_loaded()
        df = self._demand_df
        if date_label is None:
            date_label = self._current_date_label()
        date_label = self._normalize_demand_date(date_label)
        if date_label is None:
            if len(self._available_demand_dates) > 1:
                raise RuntimeError(
                    "Demand date is missing while multiple dates are loaded; refusing to merge "
                    f"{len(self._available_demand_dates)} days into one episode."
                )
            day_df = df.copy()
        else:
            day_df = df[df['pickup_date'].eq(date_label)].copy()
            if day_df.empty:
                date_key = str(date_label.date())
                if date_key not in self._warned_empty_demand_dates:
                    self._warned_empty_demand_dates.add(date_key)
                    print(f"⚠ No demand rows found for {date_key}; this day will generate 0 requests.")
        cached_dates = pd.to_datetime(day_df['pickup_date']).dt.normalize().dropna().unique()
        if len(cached_dates) > 1:
            raise RuntimeError(
                "Episode demand cache contains multiple pickup dates: "
                + ", ".join(str(pd.Timestamp(value).date()) for value in cached_dates[:10])
            )
        if len(cached_dates) == 1 and date_label is not None:
            cached_date = pd.Timestamp(cached_dates[0]).normalize()
            if cached_date != date_label:
                raise RuntimeError(
                    f"Episode demand cache date mismatch: requested {date_label.date()}, "
                    f"cached {cached_date.date()}."
                )
        day_df = day_df.sort_values('pickup_second_of_day').reset_index(drop=True)
        self._demand_day_cache = day_df
        self._demand_day_cache_label = date_label
        self._demand_cursor = 0

    def _available_vehicle_supply_by_zone(self) -> Dict[int, int]:
        """Return real-time idle supply used by the zonal surge calculation."""
        supply: Dict[int, int] = {}
        for vehicle in self.vehicles.values():
            if not vehicle.get('is_online', True):
                continue
            if vehicle.get('assigned_request') is not None or vehicle.get('passenger_onboard') is not None:
                continue
            if vehicle.get('charging_station') is not None or vehicle.get('charging_target') is not None:
                continue
            if float(vehicle.get('penalty_timer', 0.0)) > 0.0:
                continue
            zone_id = int(vehicle.get('location', 0))
            supply[zone_id] = supply.get(zone_id, 0) + 1
        return supply

    def _queued_request_demand_by_zone(self, pending_requests: list | None = None) -> Dict[int, int]:
        """Return unassigned request demand, including a not-yet-finalized batch."""
        occupied_request_ids = {
            int(request_id)
            for vehicle in self.vehicles.values()
            for request_id in (vehicle.get('assigned_request'), vehicle.get('passenger_onboard'))
            if request_id is not None
        }
        demand: Dict[int, int] = {}
        for request in self.active_requests.values():
            if int(getattr(request, 'request_id', -1)) in occupied_request_ids:
                continue
            zone_id = int(getattr(request, 'pickup', getattr(request, 'source', 0)))
            demand[zone_id] = demand.get(zone_id, 0) + 1
        for request in pending_requests or []:
            zone_id = int(getattr(request, 'pickup', getattr(request, 'source', 0)))
            demand[zone_id] = demand.get(zone_id, 0) + 1
        return demand

    def _dynamic_surge_by_zone(self, pending_requests: list | None = None) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Compute Ashkrof et al. (2025) zonal D/S ratios and 1x--5x multipliers.

        A one-vehicle denominator floor keeps zero-supply zones finite while still
        allowing multiple queued requests in such a zone to trigger surge pricing.
        """
        demand = self._queued_request_demand_by_zone(pending_requests)
        supply = self._available_vehicle_supply_by_zone()
        zone_ids = set(getattr(self, 'real_zone_ids', set())) | set(demand) | set(supply)
        ratios = {
            int(zone_id): float(demand.get(zone_id, 0)) / float(max(1, supply.get(zone_id, 0)))
            for zone_id in zone_ids
        }
        max_ratio = max(ratios.values(), default=1.0)
        min_multiplier = float(getattr(self, 'surge_min_multiplier', 1.0))
        max_multiplier = float(getattr(self, 'surge_max_multiplier', 5.0))
        multipliers: Dict[int, float] = {}
        for zone_id, ratio in ratios.items():
            if ratio <= 1.0 or max_ratio <= 1.0:
                multiplier = min_multiplier
            else:
                multiplier = min_multiplier + (max_multiplier - min_multiplier) * (
                    (ratio - 1.0) / (max_ratio - 1.0)
                )
            multipliers[int(zone_id)] = float(np.clip(multiplier, min_multiplier, max_multiplier))
        return ratios, multipliers

    def _apply_dynamic_surge_to_requests(
        self,
        requests: list,
        request_history: list[dict] | None = None,
    ) -> None:
        """Price a request batch using the current zonal demand--supply state."""
        if not requests:
            return
        ratios, multipliers = self._dynamic_surge_by_zone(requests)
        if request_history is None:
            request_history = [{} for _ in requests]
        for request, history in zip(requests, request_history):
            zone_id = int(getattr(request, 'pickup', getattr(request, 'source', 0)))
            base_value = float(getattr(request, 'value', 0.0))
            multiplier = float(multipliers.get(zone_id, 1.0))
            final_value = base_value * multiplier
            surge_bonus = max(0.0, final_value - base_value)
            request.final_value = final_value
            request.surge_multiplier = multiplier
            request.surge_bonus = surge_bonus
            request.demand_supply_ratio = float(ratios.get(zone_id, 0.0))
            history['surge_multiplier'] = multiplier
            history['surge_bonus'] = surge_bonus
            history['demand_supply_ratio'] = request.demand_supply_ratio

    def generate_requests(self, day=None) -> list:
        """
        Generate requests for the current epoch from real taxi trip data.

        Each epoch covers [current_time * EPOCH_LENGTH, (current_time+1) * EPOCH_LENGTH)
        seconds of the day.
        """
        date_label = self._normalize_demand_date(day if day is not None else self._current_date_label())
        if self._demand_day_cache is None or self._demand_day_cache_label != date_label:
            self._prepare_day_demand(date_label)

        epoch_start_sec = self.START_EPOCH + self._day_step_offset() * self.EPOCH_LENGTH
        epoch_end_sec = epoch_start_sec + self.EPOCH_LENGTH

        # filter outside operating hours
        if epoch_start_sec >= self.STOP_EPOCH:
            self.last_generated_requests = 0
            self.last_generated_request_time = self.current_time
            return []

        cache = self._demand_day_cache
        generated: list = []
        request_history: list[dict] = []

        # advance cursor to epoch_start_sec
        while self._demand_cursor < len(cache) and cache.iloc[self._demand_cursor]['pickup_second_of_day'] < epoch_start_sec:
            self._demand_cursor += 1

        idx = self._demand_cursor
        while idx < len(cache):
            row = cache.iloc[idx]
            sod = row['pickup_second_of_day']
            if sod >= epoch_end_sec:
                break

            pu = int(row['PULocationID'])
            do = int(row['DOLocationID'])
            fare = float(row['fare_amount'])
            trip_distance_miles = float(row['trip_distance'])
            trip_distance_km = trip_distance_miles * self.MILES_TO_KM

            # skip zones not in our coordinate set
            if pu not in self.zone_coords or do not in self.zone_coords:
                idx += 1
                continue

            self.request_counter += 1
            # TLC distance supplies the route length.  Duration follows the
            # simulator's configured speed, not the recorded trip duration.
            travel_time_epochs = max(
                1.0,
                trip_distance_km
                / max(float(self.average_velocity_kmph), 1e-9)
                * 3600.0
                / float(self.EPOCH_LENGTH),
            )

            # Dynamic surge is applied consistently to the complete epoch batch
            # after all valid requests have been collected.
            base_value = max(5.0, fare)

            req = Request(
                request_id=self.request_counter,
                source=pu,
                destination=do,
                current_time=self.current_time,
                travel_time=travel_time_epochs,
                value=base_value,
                final_value=base_value,
                trip_distance_km=trip_distance_km,
            )
            req.trip_distance_miles = trip_distance_miles
            generated.append(req)
            request_history.append({
                'pickup_zone': pu,
                'dropoff_zone': do,
                'time': self.current_time,
                'fare': fare,
                'trip_distance_miles': trip_distance_miles,
                'trip_distance_km': trip_distance_km,
            })
            idx += 1

        self._demand_cursor = idx
        self._apply_dynamic_surge_to_requests(generated, request_history)
        return self._finalize_generated_requests(generated, request_history)

    def _finalize_generated_requests(self, generated: list, request_history: list[dict] | None = None) -> list:
        if not generated:
            self.last_generated_requests = 0
            self.last_generated_request_time = self.current_time
            return []

        if request_history is None:
            request_history = [
                {
                    'pickup_zone': req.pickup,
                    'dropoff_zone': req.dropoff,
                    'time': self.current_time,
                    'fare': req.value,
                }
                for req in generated
            ]

        for req, history in zip(generated, request_history):
            self.active_requests[req.request_id] = req
            self.request_generation_history.append(history)

        self.whole_req_num += len(generated)
        self.last_generated_requests = len(generated)
        self.last_generated_request_time = self.current_time
        return generated

    def _build_request_feasibility_matrix(self, requests: list, vehicle_ids: list[int]) -> np.ndarray:
        if not requests or not vehicle_ids:
            return np.zeros((len(vehicle_ids), len(requests)), dtype=bool)

        use_range = getattr(self, 'use_range_requests', False)
        range_radius = getattr(self, 'assignmentrange', 5.0)
        vehicle_locations = np.array([self.vehicles[vid]['location'] for vid in vehicle_ids], dtype=np.int32)
        vehicle_battery = np.array([self.vehicles[vid]['battery'] for vid in vehicle_ids], dtype=np.float32)
        pickup_ids = np.array([req.pickup for req in requests], dtype=np.int32)
        dropoff_ids = np.array([req.dropoff for req in requests], dtype=np.int32)
        pickup_dists = self.distance_matrix[vehicle_locations[:, None], pickup_ids[None, :]]
        trip_dists = np.asarray([
            self._request_trip_distance_km(request) for request in requests
        ], dtype=np.float32)
        total_dists = pickup_dists + trip_dists[None, :]

        if dropoff_ids.size > 0 and np.max(dropoff_ids) < self.nearest_charging_distance.shape[0]:
            reserve = np.maximum(
                self.min_battery_level,
                self.nearest_charging_distance[dropoff_ids] * self.battery_consum + 0.01,
            )
        else:
            reserve = np.array([self._post_action_battery_reserve(req.dropoff) for req in requests], dtype=np.float32)

        feasible = vehicle_battery[:, None] - total_dists * self.battery_consum >= reserve[None, :]
        if use_range:
            feasible &= pickup_dists <= range_radius

        return feasible

    def _select_assignable_requests(
        self,
        requests: list,
        request_history: list[dict],
        vehicle_ids: list[int],
        sample_num: int | None,
    ) -> tuple[list, list[dict]]:
        if not requests or not vehicle_ids:
            return [], []

        feasible = self._build_request_feasibility_matrix(requests, vehicle_ids)
        if feasible.size == 0:
            return [], []

        request_order = list(range(len(requests)))
        random.shuffle(request_order)
        adjacency: dict[int, list[int]] = {}
        for req_idx in request_order:
            feasible_vehicle_rows = np.flatnonzero(feasible[:, req_idx]).tolist()
            random.shuffle(feasible_vehicle_rows)
            adjacency[req_idx] = feasible_vehicle_rows

        vehicle_to_request = [-1] * len(vehicle_ids)

        def _augment(req_idx: int, seen_vehicle_rows: set[int]) -> bool:
            for vehicle_row in adjacency[req_idx]:
                if vehicle_row in seen_vehicle_rows:
                    continue
                seen_vehicle_rows.add(vehicle_row)
                matched_req = vehicle_to_request[vehicle_row]
                if matched_req == -1 or _augment(matched_req, seen_vehicle_rows):
                    vehicle_to_request[vehicle_row] = req_idx
                    return True
            return False

        for req_idx in request_order:
            _augment(req_idx, set())

        matched_request_indices = [req_idx for req_idx in vehicle_to_request if req_idx != -1]
        if not matched_request_indices:
            return [], []

        matched_request_indices = list(dict.fromkeys(matched_request_indices))
        if sample_num is not None:
            sample_size = max(0, min(int(sample_num), len(matched_request_indices), len(vehicle_ids)))
            if sample_size == 0:
                return [], []
            if sample_size < len(matched_request_indices):
                matched_request_indices = random.sample(matched_request_indices, sample_size)

        return (
            [requests[idx] for idx in matched_request_indices],
            [request_history[idx] for idx in matched_request_indices],
        )

    def generate_requests_time(
        self,
        day=None,
        vehicle_ids: list[int] | None = None,
        rand=None,
        sample_num: int | None = None,
        snapshot_window_seconds: float | None = None,
    ) -> list:
        """
        Generate requests for one epoch from real taxi trip data.

        When ``rand`` is provided, this samples one non-empty epoch from the
        requested day.  This is used by the fleet-ratio snapshot experiment.
        Without ``rand`` it uses the current simulation epoch, like
        ``generate_requests``.

        ``snapshot_window_seconds`` models request retention at a decision
        snapshot: all requests arriving in [t - window, t) are active at t.
        """
        date_label = self._normalize_demand_date(day if day is not None else self._current_date_label())
        if self._demand_day_cache is None or self._demand_day_cache_label != date_label:
            self._prepare_day_demand(date_label)
        cache = self._demand_day_cache
        if cache is None or cache.empty:
            self.last_generated_requests = 0
            self.last_generated_request_time = self.current_time
            if rand is not None:
                print("generate_requests_time sampled 0 requests: empty day demand", flush=True)
            return []

        window_seconds = None
        if snapshot_window_seconds is not None:
            window_seconds = max(float(self.EPOCH_LENGTH), float(snapshot_window_seconds))

        if rand is not None:
            rng = np.random.default_rng(int(rand))
            valid = cache[
                (cache['pickup_second_of_day'] >= self.START_EPOCH)
                & (cache['pickup_second_of_day'] < self.STOP_EPOCH)
                & (cache['PULocationID'].astype(int).isin(self.zone_coords))
                & (cache['DOLocationID'].astype(int).isin(self.zone_coords))
            ].copy()
            if valid.empty:
                self.last_generated_requests = 0
                self.last_generated_request_time = self.current_time
                print("generate_requests_time sampled 0 requests: no valid trips in time window", flush=True)
                return []
            if window_seconds is None:
                epoch_bins = np.floor(
                    (valid['pickup_second_of_day'].to_numpy(dtype=float) - float(self.START_EPOCH))
                    / float(self.EPOCH_LENGTH)
                ).astype(int)
                unique_bins = np.unique(epoch_bins)
                chosen_bin = int(rng.choice(unique_bins))
            else:
                pickup_seconds = np.sort(valid['pickup_second_of_day'].to_numpy(dtype=float))
                min_bin = int(math.ceil(window_seconds / float(self.EPOCH_LENGTH)))
                max_bin = int(math.floor((float(self.STOP_EPOCH) - float(self.START_EPOCH)) / float(self.EPOCH_LENGTH))) - 1
                possible_bins = np.arange(min_bin, max_bin + 1, dtype=int)
                if possible_bins.size <= 0:
                    self.last_generated_requests = 0
                    self.last_generated_request_time = self.current_time
                    print("generate_requests_time sampled 0 requests: no valid snapshot bins", flush=True)
                    return []
                window_ends = float(self.START_EPOCH) + possible_bins.astype(float) * float(self.EPOCH_LENGTH)
                left = np.searchsorted(pickup_seconds, window_ends - window_seconds, side='left')
                right = np.searchsorted(pickup_seconds, window_ends, side='left')
                nonempty_bins = possible_bins[(right - left) > 0]
                if nonempty_bins.size <= 0:
                    self.last_generated_requests = 0
                    self.last_generated_request_time = self.current_time
                    print("generate_requests_time sampled 0 requests: no non-empty snapshot windows", flush=True)
                    return []
                chosen_bin = int(rng.choice(nonempty_bins))
            self.current_time = float(chosen_bin)
        else:
            chosen_bin = int(self._day_step_offset())

        if window_seconds is None:
            epoch_start_sec = self.START_EPOCH + chosen_bin * self.EPOCH_LENGTH
            epoch_end_sec = epoch_start_sec + self.EPOCH_LENGTH
        else:
            epoch_end_sec = self.START_EPOCH + chosen_bin * self.EPOCH_LENGTH
            epoch_start_sec = max(float(self.START_EPOCH), epoch_end_sec - window_seconds)

        # filter outside operating hours
        if epoch_start_sec >= self.STOP_EPOCH or epoch_end_sec <= self.START_EPOCH:
            self.last_generated_requests = 0
            self.last_generated_request_time = self.current_time
            return []

        generated: list = []
        request_history: list[dict] = []

        rows = cache[
            (cache['pickup_second_of_day'] >= epoch_start_sec)
            & (cache['pickup_second_of_day'] < epoch_end_sec)
        ]
        if rand is not None and not rows.empty:
            rows = rows.sample(frac=1.0, random_state=int(rand))

        for _, row in rows.iterrows():

            pu = int(row['PULocationID'])
            do = int(row['DOLocationID'])
            fare = float(row['fare_amount'])
            trip_distance_miles = float(row['trip_distance'])
            trip_distance_km = trip_distance_miles * self.MILES_TO_KM

            # skip zones not in our coordinate set
            if pu not in self.zone_coords or do not in self.zone_coords:
                continue

            next_request_id = self.request_counter + len(generated) + 1
            travel_time_epochs = max(
                1.0,
                trip_distance_km
                / max(float(self.average_velocity_kmph), 1e-9)
                * 3600.0
                / float(self.EPOCH_LENGTH),
            )

            # Dynamic surge is applied after optional request sampling so the
            # multiplier reflects the demand that actually enters this snapshot.
            base_value = max(5.0, fare)

            req = Request(
                request_id=next_request_id,
                source=pu,
                destination=do,
                current_time=self.current_time,
                travel_time=travel_time_epochs,
                value=base_value,
                final_value=base_value,
                trip_distance_km=trip_distance_km,
            )
            req.trip_distance_miles = trip_distance_miles
            generated.append(req)
            request_history.append({
                'pickup_zone': pu,
                'dropoff_zone': do,
                'time': self.current_time,
                'fare': fare,
                'trip_distance_miles': trip_distance_miles,
                'trip_distance_km': trip_distance_km,
            })

        if sample_num is not None:
            if vehicle_ids is not None:
                generated, request_history = self._select_assignable_requests(
                    generated,
                    request_history,
                    vehicle_ids,
                    sample_num,
                )
            else:
                sample_size = max(0, min(int(sample_num), len(generated)))
                if sample_size < len(generated):
                    rng = random.Random(rand)
                    sampled_indices = rng.sample(range(len(generated)), sample_size)
                    generated = [generated[idx] for idx in sampled_indices]
                    request_history = [request_history[idx] for idx in sampled_indices]

        if generated:
            self.request_counter = max(req.request_id for req in generated)

        self._apply_dynamic_surge_to_requests(generated, request_history)
        finalized = self._finalize_generated_requests(generated, request_history)
        if rand is not None or sample_num is not None:
            date_desc = date_label.date() if hasattr(date_label, 'date') else date_label
            start_hour = float(epoch_start_sec) / 3600.0
            end_hour = float(epoch_end_sec) / 3600.0
            window_minutes = (float(epoch_end_sec) - float(epoch_start_sec)) / 60.0
            print(
                f"generate_requests_time sampled {len(finalized)} requests "
                f"(date={date_desc}, step={chosen_bin}, "
                f"window={window_minutes:.1f}m, hours={start_hour:.2f}-{end_hour:.2f}, rand={rand})",
                flush=True,
            )
        return finalized



    # keep compatibility alias
    def _generate_intense_requests(self):
        return self.generate_requests()

    def _generate_random_requests(self):
        return self.generate_requests()

    # ==================================================================
    # Request batch (abstract method compat)
    # ==================================================================

    def get_request_batch(self):
        requests = list(self.active_requests.values())
        for vid, v in self.vehicles.items():
            if v['battery'] < 0.005 and v['charging_station'] is None:
                requests.append(Request(
                    request_id=f"charge_{vid}",
                    source=v['location'],
                    destination=v['location'],
                    current_time=self.current_time,
                    travel_time=0,
                    value=0.5,
                ))
        return requests

    # ==================================================================
    # Vehicle state
    # ==================================================================

    def _get_vehicle_state(self, vehicle_id: int):
        v = self.vehicles[vehicle_id]
        lat, lon = v['coordinates']
        return np.array([
            lat / 41.0,
            lon / -74.0,
            v['battery'],
            float(v['charging_station'] is not None),
            self.current_time / max(1, self.episode_length),
        ], dtype=np.float32)

    def get_initial_states(self, num_agents=None, is_training=True):
        if num_agents is None:
            num_agents = self.num_vehicles
        return {vid: self._get_vehicle_state(vid) for vid in range(num_agents) if vid in self.vehicles}

    # ==================================================================
    # EV helpers  (mirror ChargingIntegratedEnvironment)
    # ==================================================================

    def _is_ev(self, vehicle_id: int) -> bool:
        v = self.vehicles.get(vehicle_id)
        return v is not None and v.get('type', 0) == 1

    def _in_ev_penalty(self, vehicle_id: int) -> bool:
        if not self._is_ev(vehicle_id):
            return False
        return float(self.current_time) < float(self.ev_penalty_until_time.get(vehicle_id, -1.0))

    def _record_ev_acceptance(self, vehicle_id: int):
        if not self._is_ev(vehicle_id):
            return
        now = float(self.current_time)
        self.ev_last_accepted_time[vehicle_id] = now
        idle_start = self.ev_current_idle_start_time.get(vehicle_id)
        if idle_start is not None:
            self.ev_idle_durations.append(max(0.0, now - float(idle_start)))
        self.ev_current_idle_start_time[vehicle_id] = None
        self.ev_consecutive_rejections[vehicle_id] = 0

    def _record_ev_completion(self, vehicle_id: int):
        if not self._is_ev(vehicle_id):
            return
        now = float(self.current_time)
        self.ev_last_completed_time[vehicle_id] = now
        self.ev_current_idle_start_time[vehicle_id] = now

    def _record_ev_rejection(self, vehicle_id: int):
        if not self._is_ev(vehicle_id):
            return
        cnt = int(self.ev_consecutive_rejections.get(vehicle_id, 0)) + 1
        self.ev_consecutive_rejections[vehicle_id] = cnt
        if cnt >= 2:
            self.ev_penalty_until_time[vehicle_id] = float(self.current_time) + float(self.ev_penalty_duration)
            self.ev_consecutive_rejections[vehicle_id] = 0

    def update_satisfaction(self, vehicle_id: int):
        base_salary = max(float(self.ev_basesalary), 1e-6)
        vehicle = self.vehicles[vehicle_id]
        satisfaction = float(vehicle.get('satisfaction', 0.0))
        daily_salary = float(vehicle.get('daily_salary', 0.0))
        baseline_gap = (daily_salary - base_salary) / base_salary
        updated = satisfaction * self.dropoff_probability_rate + (1 - self.dropoff_probability_rate) * baseline_gap
        vehicle['satisfaction'] = float(np.clip(updated, -1.0, 1.0))

    def _update_all_ev_satisfaction(self):
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle.get('type') == 1:
                self.update_satisfaction(vehicle_id)

    def calculate_dropoff_probability(self, vehicle_id: int) -> float:
        vehicle = self.vehicles[vehicle_id]
        idle_time = float(vehicle.get('idle_timer', 0.0))
        satisfaction = float(vehicle.get('satisfaction', 0.0))
        logit = (
            self.ev_dropoff_beta_0
            + self.ev_dropoff_beta_idle * idle_time
            + self.ev_dropoff_beta_satisfaction * satisfaction
        )
        if self.ifdropoff:
            return float(1.0 / (1.0 + np.exp(-logit)))
        return 0.0

    def calculate_rejoin_probability(self, vehicle_id: int) -> float:
        satisfaction = float(self.vehicles[vehicle_id].get('satisfaction', 0.0))
        logit = self.ev_rejoin_gamma_0 + self.ev_rejoin_gamma_satisfaction * satisfaction
        if self.ifdropoff:
            return float(1.0 / (1.0 + np.exp(-logit)))
        return 0.0

    def _mark_ev_pending_dropout_penalty(self, vehicle_id: int) -> float:
        if not self._is_ev(vehicle_id):
            return 0.0
        pending_action = self.storeactions_ev.get(vehicle_id)
        if pending_action is None or not isinstance(pending_action, ServiceAction):
            return 0.0
        if getattr(pending_action, 'ev_dropout_after_action', False):
            return float(getattr(pending_action, 'dropout_penalty', 0.0))
        dropout_penalty = float(self.ev_basesalary)
        pending_action.ev_dropout_after_action = True
        pending_action.dropout_penalty = dropout_penalty
        pending_action.is_vehicle_done = True
        return 0.0

    def _set_vehicle_offline_until_next_real_day(self, vehicle_id: int):
        vehicle = self.vehicles[vehicle_id]
        self._mark_ev_pending_dropout_penalty(vehicle_id)
        next_day_time = (self._current_day_index() + 1) * self.simulation_period
        vehicle['is_online'] = False
        vehicle['offline_until_time'] = next_day_time
        vehicle['assigned_request'] = None
        vehicle['passenger_onboard'] = None
        vehicle['target_location'] = None
        vehicle['idle_target'] = None
        vehicle['charging_target'] = None
        vehicle['is_stationary'] = False
        vehicle['stationary_duration'] = 0
        vehicle['whether_finishrequest'] = False
        self.current_period_dropout_count += 1

    def _handle_vehicle_dropout_event(self, vehicle_id: int):
        vehicle = self.vehicles[vehicle_id]
        if vehicle.get('type') != 1 or not vehicle.get('is_online', True):
            return
        dropoff_probability = self.calculate_dropoff_probability(vehicle_id)
        if dropoff_probability > self.ev_dropoff_threshold:
            self._set_vehicle_offline_until_next_real_day(vehicle_id)

    def _can_daily_dropout(self, vehicle: dict) -> bool:
        return (
            vehicle.get('type') == 1
            and vehicle.get('is_online', True)
            and vehicle.get('assigned_request') is None
            and vehicle.get('passenger_onboard') is None
            and vehicle.get('charging_station') is None
        )

    def _refresh_daily_driver_states(self):
        previous_period_dropout_count = self.current_period_dropout_count
        self.current_period_dropout_count = 0
        self._update_all_ev_satisfaction()
        daily_online = 0
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle['idle_timer'] = 0
            was_online = vehicle.get('is_online', True)
            if self.daily_drop_off and self._can_daily_dropout(vehicle):
                self._handle_vehicle_dropout_event(vehicle_id)
            elif not was_online:
                rejoin_probability = self.calculate_rejoin_probability(vehicle_id)
                if random.random() < rejoin_probability:
                    vehicle['is_online'] = True
                    vehicle['offline_until_time'] = None
            if vehicle.get('is_online', True):
                daily_online += 1
            vehicle['daily_salary'] = 0.0
            vehicle['salary_ratio'] = 0.0 if vehicle.get('type') == 1 else 1e4
        self.current_online = daily_online
        self.daily_online_history.append(daily_online)
        if self.daily_drop_off:
            self.period_dropout_counts.append(previous_period_dropout_count + self.current_period_dropout_count)
            self.current_period_dropout_count = 0
        else:
            self.period_dropout_counts.append(previous_period_dropout_count)

    # ==================================================================
    # EV behaviour models (charge / relocation / prior features)
    # ==================================================================

    def compute_ev_charge_probability(self, vehicle_id: int) -> Tuple[float, Dict[int, float]]:
        """Zhang et al. (2023) Binary Logit + MNL charge model (same as ChargingIntegrated)."""
        if not self._is_ev(vehicle_id) or not hasattr(self, 'charging_manager'):
            return 0.0, {}
        vehicle = self.vehicles[vehicle_id]
        if vehicle.get('no_charge_cooldown_until', 0) > self.current_time and vehicle.get('battery', 1.0) > 0.2:
            return 0.0, {}
        soc = float(vehicle.get('battery', 1.0))
        d_deadhead = 1.0 if vehicle.get('total_distance', 0) > 100 else 0.0
        eta = np.random.normal(0, 1.760)
        V_swap = 3.5 - 9.5 * soc + 0.69 * d_deadhead + eta
        p_charge = 1.0 / (1.0 + np.exp(-V_swap))
        if not self.charging_manager.stations:
            return float(p_charge), {}
        vehicle_loc = int(vehicle.get('location', 0))
        station_utilities: Dict[int, float] = {}
        for sid, station in self.charging_manager.stations.items():
            if not self._can_reach_charging_station(vehicle_id, int(sid)):
                continue
            s_loc = int(station.location)
            d_detour = float(self._manhattan_distance_loc(vehicle_loc, s_loc))
            n_battery = float(station.available_slots)
            l_queue = float(len(getattr(station, 'charging_queue', [])))
            station_capacity = max(float(getattr(station, 'max_capacity', 4)), 1.0)
            queue_within_capacity = min(l_queue, station_capacity)
            queue_overflow = max(0.0, l_queue - station_capacity)
            queue_penalty = 0.111 * queue_within_capacity + 2.5 * queue_overflow
            cost = 1.0
            if s_loc in self.surge_zone_locs:
                pop = 5.0
            elif s_loc in self.high_demand_zone_locs:
                pop = 4.0
            elif s_loc in self.city_center_zone_locs:
                pop = 3.0
            else:
                pop = 2.0
            xi = np.random.normal(0, 1.900)
            V_i = -0.325 * d_detour + 0.0529 * n_battery - queue_penalty - 1.020 * cost + 0.0927 * pop + xi
            station_utilities[int(sid)] = V_i
        if station_utilities:
            station_utilities = self._localize_charging_station_utilities(vehicle_loc, station_utilities)
            sids = list(station_utilities.keys())
            utils = np.array([station_utilities[s] for s in sids])
            exp_utils = np.exp(utils - np.max(utils))
            probs = exp_utils / np.sum(exp_utils)
            station_probs = {sids[i]: float(probs[i]) for i in range(len(sids))}
        else:
            station_probs = {}
        return float(p_charge), station_probs

    def _localize_charging_station_utilities(
        self,
        vehicle_loc: int,
        station_utilities: Dict[int, float],
        *,
        max_candidates: int = 5,
        distance_slack_km: float = 1.5,
    ) -> Dict[int, float]:
        """Keep EV charging choice stochastic, but only among a few nearby reachable stations."""
        if len(station_utilities) <= max_candidates:
            return station_utilities

        distance_pairs = []
        for sid in station_utilities.keys():
            station = self.charging_manager.stations.get(int(sid))
            if station is None:
                continue
            distance_pairs.append((self.get_distance_km(vehicle_loc, int(station.location)), int(sid)))

        if not distance_pairs:
            return station_utilities

        distance_pairs.sort(key=lambda item: item[0])
        nearest_distance = distance_pairs[0][0]
        nearby_ids = [sid for dist, sid in distance_pairs if dist <= nearest_distance + distance_slack_km]
        if len(nearby_ids) < min(max_candidates, len(distance_pairs)):
            nearby_ids = [sid for _, sid in distance_pairs[:max_candidates]]
        elif len(nearby_ids) > max_candidates:
            nearby_ids = nearby_ids[:max_candidates]
        return {sid: station_utilities[sid] for sid in nearby_ids if sid in station_utilities}

    def compute_ev_relocation_distribution(self, vehicle_id: int) -> Dict[int, float]:
        """Return the Xie et al. (2023) MNL relocation distribution."""
        vehicle = self.vehicles.get(vehicle_id, {})
        cur_loc = int(vehicle.get('location', 0))
        beta1 = float(getattr(self, 'relocation_beta_match', 0.08))
        beta2 = float(getattr(self, 'relocation_beta_cost', 0.1))
        reloc_cost_u = float(getattr(self, 'relocation_cost_u', 0.6 * 0.92))
        # Build candidate set from nearby relocation targets: self + nearest relocation targets.
        candidate_zone_ids = list(self.relocation_target_ids) if self.relocation_target_ids else sorted(self.zone_coords.keys())
        dists = [(self.get_distance_km(cur_loc, z), z) for z in candidate_zone_ids if z != cur_loc]
        dists.sort()
        neighbours = [cur_loc] + [z for _, z in dists[:4]]
        neighbours = list(dict.fromkeys(neighbours))
        # Supply: idle vehicle counts per zone
        idle_counts: Dict[int, int] = {}
        for vid, v in self.vehicles.items():
            if v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None:
                loc = int(v.get('location', 0))
                idle_counts[loc] = idle_counts.get(loc, 0) + 1
        # Demand: active request counts per pickup zone
        request_counts: Dict[int, int] = {}
        if hasattr(self, 'active_requests'):
            for req in self.active_requests.values():
                pu = int(getattr(req, 'pickup', getattr(req, 'source', 0)))
                request_counts[pu] = request_counts.get(pu, 0) + 1
        utilities = {}
        for loc_j in neighbours:
            O_j = request_counts.get(loc_j, 0)
            A_j = max(idle_counts.get(loc_j, 0), 1)
            s_j = min(O_j / A_j, 1.0)
            RC_ij = 0.0 if loc_j == cur_loc else reloc_cost_u
            utilities[loc_j] = beta1 * s_j + beta2 * RC_ij
        locs = list(utilities.keys())
        u_arr = np.array([utilities[l] for l in locs])
        exp_u = np.exp(u_arr - np.max(u_arr))
        probs = exp_u / np.sum(exp_u)
        return {locs[i]: float(probs[i]) for i in range(len(locs))}

    def compute_ev_relocation_probability(self, vehicle_id: int) -> Tuple[int, Dict[int, float]]:
        """Sample one destination from the NYC EV relocation distribution."""
        prob_dict = self.compute_ev_relocation_distribution(vehicle_id)
        locs = list(prob_dict)
        probs = np.asarray([prob_dict[loc] for loc in locs], dtype=np.float64)
        chosen = int(np.random.choice(locs, p=probs))
        return chosen, prob_dict

    def _sample_ev_default_relocation_target(self, vehicle_id: int) -> int:
        """Sample-once target for the EV matrix's final wait/reloc action.

        The sampled target is cached for the current decision epoch.  Q-value
        generation and action execution therefore see the same concrete
        relocation edge, while the action matrix keeps its single outside
        column and the existing stochastic MNL direction policy.
        """

        step = float(getattr(self, 'current_time', 0.0) or 0.0)
        if getattr(self, '_ev_default_relocation_cache_step', None) != step:
            self._ev_default_relocation_cache_step = step
            self._ev_default_relocation_targets = {}
            self._ev_default_relocation_probabilities = {}

        targets = getattr(self, '_ev_default_relocation_targets', {})
        if int(vehicle_id) in targets:
            return int(targets[int(vehicle_id)])

        vehicle = self.vehicles[vehicle_id]
        current_location = int(vehicle['location'])
        battery_blocked = float(vehicle.get('battery', 1.0)) <= (
            float(self.min_battery_level) + 2.0 * float(self.battery_consum)
        )
        if battery_blocked:
            target = current_location
            probabilities = {current_location: 1.0}
        else:
            target, probabilities = self.compute_ev_relocation_probability(vehicle_id)

        self._ev_default_relocation_targets[int(vehicle_id)] = int(target)
        self._ev_default_relocation_probabilities[int(vehicle_id)] = {
            int(zone): float(probability)
            for zone, probability in probabilities.items()
        }
        return int(target)

    def _handle_ev_rejection_relocation(self, vehicle_id: int):
        """EV rejection relocation — returns (target_zone_id, prob_dict)."""
        chosen_zone, prob_dict = self.compute_ev_relocation_probability(vehicle_id)
        return chosen_zone, prob_dict

    def _build_prior_features(self, vehicle_ids, actions):
        """Build prior feature vectors for cross-attention (same structure as ChargingIntegrated)."""
        feats = []
        norm = max(1, self.NUM_LOCATIONS)
        target_dim = self.aux_zone_dim if self.aux_zone_dim > 0 else self.num_zones
        zone_counts = [0] * target_dim
        for vid in vehicle_ids:
            v = self.vehicles[vid]
            loc_norm = float(v['location']) / norm
            battery = float(v.get('battery', 1.0))
            idle_time = float(v.get('idle_timer', 0)) / max(1.0, float(self.episode_length))
            act = actions.get(vid)
            if act is None:
                target_loc_norm = loc_norm
                action_type_float = 0.0
                tgt_loc_raw = v['location']
            elif isinstance(act, ServiceAction):
                tgt = getattr(act, 'target_location', v['location'])
                if tgt is None and getattr(act, 'request_id', None) in self.active_requests:
                    tgt = self.active_requests[act.request_id].pickup
                tgt_loc_raw = tgt if isinstance(tgt, (int, float)) else v['location']
                target_loc_norm = float(tgt_loc_raw) / norm
                action_type_float = 1.0
            elif isinstance(act, ChargingAction):
                station = self.charging_manager.stations.get(getattr(act, 'charging_station_id', None))
                tgt_loc_raw = station.location if station is not None else v['location']
                target_loc_norm = float(tgt_loc_raw) / norm
                action_type_float = 2.0
            else:
                tgt = v.get('target_location', v.get('idle_target', v['location']))
                tgt_loc_raw = tgt if isinstance(tgt, (int, float)) else v['location']
                target_loc_norm = float(tgt_loc_raw) / norm
                action_type_float = 3.0
            feats.append([loc_norm, battery, idle_time, target_loc_norm, action_type_float])
            zone_idx = self.get_distribution_zone_index(int(tgt_loc_raw))
            if zone_idx is not None and 0 <= zone_idx < target_dim:
                zone_counts[zone_idx] += 1
        # Build zone distribution target (normalized counts)
        total = sum(zone_counts)
        if total > 0:
            self._prior_zone_dist_target = [c / total for c in zone_counts]
        else:
            self._prior_zone_dist_target = [1.0 / target_dim] * target_dim
        return np.array(feats, dtype=np.float32) if feats else np.zeros((0, 5), dtype=np.float32)

    def _zone_distribution_from_actions(self, vehicle_ids, actions):
        """Return the simulator action-target distribution on aux_zone_ids."""
        target_dim = self.aux_zone_dim if self.aux_zone_dim > 0 else self.num_zones
        counts = np.zeros(target_dim, dtype=np.float32)
        for vid in vehicle_ids:
            action = actions.get(vid)
            vehicle = self.vehicles[vid]
            target = vehicle.get('target_location', vehicle.get('idle_target', vehicle['location']))
            if isinstance(action, ServiceAction):
                target = getattr(action, 'target_location', None)
                if target is None and getattr(action, 'request_id', None) in self.active_requests:
                    target = self.active_requests[action.request_id].pickup
            elif isinstance(action, ChargingAction):
                station = self.charging_manager.stations.get(getattr(action, 'charging_station_id', None))
                target = station.location if station is not None else vehicle['location']
            if not isinstance(target, (int, float)):
                target = vehicle['location']
            zone_idx = self.get_distribution_zone_index(int(target))
            if zone_idx is not None and 0 <= zone_idx < target_dim:
                counts[zone_idx] += 1.0
        total = float(counts.sum())
        if total <= 0.0:
            return [1.0 / max(target_dim, 1)] * target_dim
        return (counts / total).tolist()

    def _vehicle_zone_distribution(self, vehicle_ids):
        """Return the current fleet-state distribution on aux_zone_ids."""
        target_dim = self.aux_zone_dim if self.aux_zone_dim > 0 else self.num_zones
        counts = np.zeros(target_dim, dtype=np.float32)
        for vid in vehicle_ids:
            location = int(self.vehicles[vid].get('location', 0))
            zone_idx = self.get_distribution_zone_index(location)
            if zone_idx is not None and 0 <= zone_idx < target_dim:
                counts[zone_idx] += 1.0
        total = float(counts.sum())
        if total <= 0.0:
            return [1.0 / max(target_dim, 1)] * target_dim
        return (counts / total).tolist()

    def _set_bayes_context(self, *, role, leader_is_ev, state_dist, peer_dist,
                           target_dist=None, prior_features=None, skip_training=False):
        self._bayes_context_role = role
        self._leader_is_ev = bool(leader_is_ev)
        self._bayes_state_posterior = state_dist
        self._bayes_external_prior = peer_dist
        self._bayes_external_posterior = peer_dist
        self._prior_zone_dist_target = target_dist
        self._prior_features_for_posterior = prior_features
        self._skip_bayes_distribution_training = bool(skip_training)

    def _activate_bayes_step_context(self, vehicle_type):
        context = getattr(self, '_bayes_step_contexts', {}).get(vehicle_type)
        if context:
            self._set_bayes_context(**context)

    # ==================================================================
    # Rejection model  (paper-aligned ride acceptance utility)
    # ==================================================================

    def _request_surge_bonus(self, request) -> float:
        explicit_bonus = getattr(request, 'surge_bonus', None)
        if explicit_bonus is not None:
            return max(0.0, float(explicit_bonus))
        return max(0.0, float(getattr(request, 'final_value', 0.0)) - float(getattr(request, 'value', 0.0)))

    def _calculate_rejection_reward(
        self,
        vehicle_id: int,
        request=None,
        *,
        pickup_location: int | None = None,
        vehicle_location: int | None = None,
    ) -> float:
        if pickup_location is None and request is not None:
            pickup_location = int(getattr(request, 'pickup', vehicle_location if vehicle_location is not None else 0))
        if vehicle_location is None:
            vehicle_location = int(self.vehicles.get(vehicle_id, {}).get('location', pickup_location or 0))
        if pickup_location is None:
            pickup_location = vehicle_location
        ratio = getattr(self, 'rejection_penalty_final_value_ratio', None)
        if request is not None and ratio is not None:
            final_value = float(getattr(request, 'final_value', getattr(request, 'value', 0.0)) or 0.0)
            if final_value > 0.0:
                return -float(ratio) * final_value
        distance_km = self.get_distance_km(int(vehicle_location), int(pickup_location))
        return -float(
            self.rejection_penalty_base
            + self.rejection_penalty_per_km * distance_km
        )

    def configure_recourse_experiment(
        self,
        variant: str = "legacy",
        *,
        rejection_logit_shift: float = 0.0,
        common_random_numbers: bool = False,
    ) -> None:
        """Configure the additive R0--R4 experiment without changing APIs."""
        variant = str(variant or "legacy").strip().lower()
        if variant not in {"legacy", "r0", "r1", "r2", "r3", "r4"}:
            raise ValueError(
                "recourse variant must be legacy or one of r0, r1, r2, r3, r4"
            )
        shift = float(rejection_logit_shift)
        if not math.isfinite(shift):
            raise ValueError("rejection_logit_shift must be finite")
        self.recourse_variant = variant
        self.rejection_logit_shift = shift
        self.common_random_numbers = bool(common_random_numbers)

    def _acceptance_uniform(self, vehicle_id: int, request) -> float:
        """Return an offer-keyed uniform shared by paired experiment runs."""
        if not bool(getattr(self, "common_random_numbers", False)):
            return random.random()
        request_id = getattr(request, "request_id", "unknown")
        day_index = int(getattr(self, "episode_day_index", 0) or 0)
        key = (
            f"{int(getattr(self, '_recourse_experiment_seed', 0))}|"
            f"{day_index}|{float(self.current_time):.9f}|"
            f"{int(vehicle_id)}|{request_id}"
        ).encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        integer = int.from_bytes(digest, byteorder="big", signed=False)
        return (integer + 0.5) / float(2**64)

    def _calculate_known_rejection_probability(self, vehicle_id: int, request, *, sample_noise: bool = False) -> float:
        vehicle = self.vehicles[vehicle_id]
        if vehicle['type'] == 2:
            return 0.0
        epsilon_t = (
            np.random.normal(0.0, self.ride_acceptance_noise_std)
            if sample_noise and self.ride_acceptance_noise_std > 0.0
            else 0.0
        )
        idle_steps = float(vehicle.get('idle_timer', 0))
        idle_time_minutes = idle_steps * float(self.EPOCH_LENGTH) / 60.0
        pickup_time_minutes = self.get_travel_time_minutes(vehicle['location'], request.pickup)
        utility = (
            self.ride_acceptance_asc
            + self.ride_acceptance_beta_idle_min * idle_time_minutes
            + self.ride_acceptance_beta_pickup_min * pickup_time_minutes
            + self.ride_acceptance_beta_surge * self._request_surge_bonus(request)
            + float(getattr(self, "rejection_logit_shift", 0.0))
            + epsilon_t
        )
        accept_prob = 1.0 / (1.0 + np.exp(-np.clip(utility, -60.0, 60.0)))
        return float(np.clip(1.0 - accept_prob, 0.0, 0.99))

    def _calculate_rejection_probabilityreal(self, vehicle_id: int, request) -> float:
        return self._calculate_known_rejection_probability(vehicle_id, request, sample_noise=True)

    def _should_reject_request(self, vehicle_id: int, request) -> bool:
        if not self.ifreject or getattr(self, "recourse_variant", "legacy") == "r0":
            return False
        return self._acceptance_uniform(vehicle_id, request) < (
            self._calculate_rejection_probabilityreal(vehicle_id, request)
        )

    def _request_action_outcome_features(self, vehicle_id: int, request, value_function=None) -> dict:
        vehicle_location = int(self.vehicles[vehicle_id]['location'])
        pickup_location = int(getattr(request, 'pickup', vehicle_location))
        dropoff_location = int(getattr(request, 'dropoff', pickup_location))
        pickup_distance = self.get_distance_km(vehicle_location, pickup_location)
        trip_distance = self._request_trip_distance_km(request)
        pickup_duration = self.get_travel_time(vehicle_location, pickup_location)
        trip_duration = float(getattr(request, 'travel_time', self.get_travel_time(pickup_location, dropoff_location)))
        return {
            'post_action_location': dropoff_location,
            'post_action_distance': float(pickup_distance + trip_distance),
            'post_action_duration': float(pickup_duration + trip_duration),
            'post_action_zoneid': int(self.get_zone_embedding_id(dropoff_location)),
        }

    def _annotate_service_action_features(self, action, vehicle_id: int, request, value_function=None):
        if action is None or request is None:
            return
        features = self._request_action_outcome_features(vehicle_id, request, value_function=value_function)
        action.pickup_location = int(getattr(request, 'pickup', self.vehicles[vehicle_id]['location']))
        action.dropoff_location = int(getattr(request, 'dropoff', action.pickup_location))
        action.value_target_location = action.dropoff_location
        action.request_value = float(getattr(request, 'final_value', 0.0))
        for key, value in features.items():
            setattr(action, key, value)

    def _coerce_location_for_candidate(self, location, fallback: int) -> int:
        if location is None:
            return int(fallback)
        if isinstance(location, (int, np.integer)):
            return int(location)
        if isinstance(location, (tuple, list)) and len(location) >= 2:
            mapped = self.map_zone(location)
            return int(mapped) if mapped >= 0 else int(fallback)
        return int(fallback)

    def _candidate_from_request(self, vehicle_id: int, request) -> dict:
        vehicle_location = int(self.vehicles[vehicle_id]['location'])
        pickup_location = int(getattr(request, 'pickup', vehicle_location))
        dropoff_location = int(getattr(request, 'dropoff', pickup_location))
        pickup_distance = float(self.get_distance_km(vehicle_location, pickup_location))
        trip_distance = self._request_trip_distance_km(request)
        pickup_duration = float(self.get_travel_time(vehicle_location, pickup_location))
        trip_duration = float(getattr(request, 'travel_time', self.get_travel_time(pickup_location, dropoff_location)))
        return {
            'action_type': f"assign_{int(getattr(request, 'request_id', 0))}",
            'target_location': pickup_location,
            'request_value': float(getattr(request, 'final_value', getattr(request, 'value', 0.0))),
            'target_distance': pickup_distance,
            'target_zoneid': int(self.get_zone_embedding_id(pickup_location)),
            'post_action_location': dropoff_location,
            'post_action_distance': pickup_distance + trip_distance,
            'post_action_duration': pickup_duration + trip_duration,
            'post_action_zoneid': int(self.get_zone_embedding_id(dropoff_location)),
            'vehicle_idle_time': float(self.vehicles[vehicle_id].get('idle_timer', 0.0)),
            'num_requests': float(len(self.active_requests)),
        }

    def _candidate_from_action(self, vehicle_id: int, action) -> dict | None:
        vehicle_location = int(self.vehicles[vehicle_id]['location'])
        if isinstance(action, ServiceAction) and hasattr(action, 'request_id'):
            request = self.active_requests.get(action.request_id)
            if request is not None:
                return self._candidate_from_request(vehicle_id, request)
            target = self._coerce_location_for_candidate(getattr(action, 'target_location', None), vehicle_location)
            return {
                'action_type': f"assign_{int(getattr(action, 'request_id', 0))}",
                'target_location': target,
                'request_value': float(getattr(action, 'request_value', 0.0)),
                'target_distance': float(self.get_distance_km(vehicle_location, target)),
                'target_zoneid': int(self.get_zone_embedding_id(target)),
                'post_action_location': int(getattr(action, 'post_action_location', target) or target),
                'post_action_distance': float(getattr(action, 'post_action_distance', 0.0)),
                'post_action_duration': float(getattr(action, 'post_action_duration', 0.0)),
                'post_action_zoneid': int(getattr(action, 'post_action_zoneid', self.get_zone_embedding_id(target)) or 0),
                'vehicle_idle_time': float(self.vehicles[vehicle_id].get('idle_timer', 0.0)),
                'num_requests': float(len(self.active_requests)),
            }
        if isinstance(action, ChargingAction) and hasattr(action, 'charging_station_id'):
            station = self.charging_manager.stations.get(action.charging_station_id)
            if station is None:
                return None
            target = int(station.location)
            distance = float(self.get_distance_km(vehicle_location, target))
            charge_duration = float(getattr(
                action,
                'charging_duration',
                self._charge_duration_for_vehicle(vehicle_id),
            ))
            duration = float(self.get_travel_time(vehicle_location, target) + charge_duration)
            return {
                'action_type': f"charge_{int(action.charging_station_id)}",
                'target_station_id': int(action.charging_station_id),
                'target_location': target,
                'request_value': 0.0,
                'target_distance': distance,
                'target_zoneid': int(self.get_zone_embedding_id(target)),
                'post_action_location': target,
                'post_action_distance': distance,
                'post_action_duration': duration,
                'post_action_zoneid': int(self.get_zone_embedding_id(target)),
                'vehicle_idle_time': float(self.vehicles[vehicle_id].get('idle_timer', 0.0)),
                'num_requests': float(len(self.active_requests)),
            }
        target = self._coerce_location_for_candidate(getattr(action, 'target_location', None), vehicle_location)
        distance = float(self.get_distance_km(vehicle_location, target))
        idle_kind = getattr(action, 'learning_action_type', None)
        if idle_kind not in {'idle', 'reloc'}:
            idle_kind = 'reloc' if distance > 1e-9 else 'idle'
        return {
            'action_type': idle_kind,
            'target_location': target,
            'request_value': 0.0,
            'target_distance': distance,
            'target_zoneid': int(self.get_zone_embedding_id(target)),
            'post_action_location': target,
            'post_action_distance': distance,
            'post_action_duration': float(self.get_travel_time(vehicle_location, target)) if distance > 0.0 else 0.0,
            'post_action_zoneid': int(self.get_zone_embedding_id(target)),
            'vehicle_idle_time': float(self.vehicles[vehicle_id].get('idle_timer', 0.0)),
            'num_requests': float(len(self.active_requests)),
        }

    def _build_bootstrap_action_candidates(self, vehicle_id: int, selected_action=None, max_candidates: int = 48) -> list[dict]:
        if vehicle_id not in self.vehicles:
            return []
        vehicle = self.vehicles[vehicle_id]
        candidates = []
        selected_candidate = self._candidate_from_action(vehicle_id, selected_action) if selected_action is not None else None
        if selected_candidate is not None:
            candidates.append(selected_candidate)

        assigned = set()
        for vid, other_vehicle in self.vehicles.items():
            if vid == vehicle_id:
                continue
            if other_vehicle['assigned_request'] is not None:
                assigned.add(other_vehicle['assigned_request'])
            if other_vehicle['passenger_onboard'] is not None:
                assigned.add(other_vehicle['passenger_onboard'])

        request_candidates = []
        vehicle_location = int(vehicle['location'])
        vehicle_battery = float(vehicle.get('battery', 1.0))
        for request in self.active_requests.values():
            if request.request_id in assigned:
                continue
            pickup_distance = float(self.get_distance_km(vehicle_location, request.pickup))
            trip_distance = self._request_trip_distance_km(request)
            reserve = self._post_action_battery_reserve(request.dropoff)
            if vehicle_battery - (pickup_distance + trip_distance) * self.battery_consum < reserve:
                continue
            candidate = self._candidate_from_request(vehicle_id, request)
            score = candidate['request_value'] - 0.15 * pickup_distance
            request_candidates.append((score, candidate))
        request_candidates.sort(key=lambda item: item[0], reverse=True)
        candidates.extend(candidate for _, candidate in request_candidates[:max_candidates])

        if not self._is_ev(vehicle_id):
            charge_candidates = []
            for station_id, station in self.charging_manager.stations.items():
                if not self._can_reach_charging_station(vehicle_id, station_id):
                    continue
                total_reserved = len(station.current_vehicles) + len(station.charging_queue_notarrived)
                if total_reserved >= station.max_capacity:
                    continue
                candidate = self._candidate_from_action(
                    vehicle_id,
                    ChargingAction([], station_id, self._charge_duration_for_vehicle(vehicle_id), vehicle_location, vehicle_battery),
                )
                if candidate is not None:
                    charge_candidates.append((-candidate['target_distance'], candidate))
            charge_candidates.sort(key=lambda item: item[0], reverse=True)
            candidates.extend(candidate for _, candidate in charge_candidates[:3])

            zone_candidates = []
            for zone_id in list(self.relocation_target_ids)[:16]:
                distance = float(self.get_distance_km(vehicle_location, int(zone_id)))
                if distance * self.battery_consum > max(0.0, vehicle_battery - self._post_action_battery_reserve(zone_id)):
                    continue
                zone_candidates.append((-distance, {
                    'action_type': 'reloc',
                    'target_location': int(zone_id),
                    'request_value': 0.0,
                    'target_distance': distance,
                    'target_zoneid': int(self.get_zone_embedding_id(zone_id)),
                    'post_action_location': int(zone_id),
                    'post_action_distance': distance,
                    'post_action_duration': float(self.get_travel_time(vehicle_location, int(zone_id))) if distance > 0.0 else 0.0,
                    'post_action_zoneid': int(self.get_zone_embedding_id(zone_id)),
                    'vehicle_idle_time': float(vehicle.get('idle_timer', 0.0)),
                    'num_requests': float(len(self.active_requests)),
                }))
            zone_candidates.sort(key=lambda item: item[0], reverse=True)
            candidates.extend(candidate for _, candidate in zone_candidates[:4])

        wait_candidate = self._candidate_from_action(
            vehicle_id,
            IdleAction([], vehicle.get('coordinates'), vehicle.get('coordinates'), vehicle_location, vehicle_battery),
        )
        if wait_candidate is not None:
            candidates.append(wait_candidate)

        deduped = []
        seen = set()
        for candidate in candidates:
            key = (candidate.get('action_type'), int(candidate.get('target_location', -1)))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
            if len(deduped) >= max_candidates:
                break
        return deduped

    def _attach_bootstrap_candidates(self, action, vehicle_id: int):
        if action is not None:
            action.bootstrap_candidates = self._build_bootstrap_action_candidates(vehicle_id, action)
            cache_step = getattr(self, '_last_vehicle_action_graph_neighbour_step', None)
            if cache_step == int(getattr(self, 'current_time', 0) or 0):
                cached_neighbours = getattr(self, '_last_vehicle_action_graph_neighbours', {}).get(
                    int(vehicle_id),
                    [],
                )
            else:
                cached_neighbours = []
            action.graph_neighbour_candidates = [dict(candidate) for candidate in cached_neighbours]
            selected_candidate = self._candidate_from_action(vehicle_id, action)
            if selected_candidate is not None:
                action.next_value = float(selected_candidate.get('request_value', 0.0))
                action.next_target_location = int(selected_candidate.get('target_location', self.vehicles[vehicle_id]['location']))
                action.next_target_distance = float(selected_candidate.get('target_distance', 0.0))
                action.next_target_zoneid = int(selected_candidate.get('target_zoneid', 0) or 0)
                action.next_post_action_distance = float(selected_candidate.get('post_action_distance', 0.0))
                action.next_post_action_duration = float(selected_candidate.get('post_action_duration', 0.0))
                action.next_post_action_zoneid = int(selected_candidate.get('post_action_zoneid', 0) or 0)

    def _next_action_training_features(self, vehicle_id: int, next_action, fallback_target: int, fallback_value: float = 0.0) -> dict:
        candidate = self._candidate_from_action(vehicle_id, next_action)
        if candidate is None:
            target = self._coerce_location_for_candidate(fallback_target, self.vehicles[vehicle_id]['location'])
            candidate = {
                'action_type': 'reloc' if target != int(self.vehicles[vehicle_id]['location']) else 'idle',
                'target_location': target,
                'request_value': float(fallback_value),
                'target_distance': float(self.get_distance_km(self.vehicles[vehicle_id]['location'], target)),
                'target_zoneid': int(self.get_zone_embedding_id(target)),
                'post_action_location': target,
                'post_action_distance': 0.0,
                'post_action_duration': 0.0,
                'post_action_zoneid': int(self.get_zone_embedding_id(target)),
            }
        return candidate

    def _build_reject_classifier_offer_sample(self, vehicle_id: int, request, *, vehicle_snapshot=None, was_rejected: bool = False) -> dict:
        vehicle = vehicle_snapshot if vehicle_snapshot is not None else self.vehicles.get(vehicle_id, {})
        vehicle_location = int(vehicle.get('location', getattr(request, 'pickup', 0)))
        pickup_location = int(getattr(request, 'pickup', vehicle_location))
        dropoff_location = int(getattr(request, 'dropoff', pickup_location))
        request_value = float(getattr(request, 'final_value', getattr(request, 'value', 0.0)))
        base_value = float(getattr(request, 'value', request_value))
        return {
            'pickup_distance_km': float(self.get_distance_km(vehicle_location, pickup_location)),
            'pickup_time_minutes': float(self.get_travel_time_minutes(vehicle_location, pickup_location)),
            'vehicle_idle_time': float(vehicle.get('idle_timer', 0.0)),
            'battery_level': float(vehicle.get('battery', 1.0)),
            'current_time': float(self.current_time),
            'num_requests': float(len(self.active_requests)),
            'request_value': request_value,
            'surge_value': max(0.0, request_value - base_value),
            'trip_distance_km': self._request_trip_distance_km(request),
            'trip_duration_epochs': float(getattr(request, 'travel_time', self.get_travel_time(pickup_location, dropoff_location))),
            'vehicle_type': int(vehicle.get('type', 1)),
            'pickup_location': pickup_location,
            'dropoff_location': dropoff_location,
            'was_rejected': bool(was_rejected),
        }

    # ==================================================================
    # Assign / pickup / dropoff
    # ==================================================================

    def _record_same_epoch_recourse_if_applicable(
        self,
        vehicle_id: int,
        request_id: int,
    ) -> bool:
        """Record an EV-rejected request accepted by an AEV in this epoch."""
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None or int(vehicle.get('type', 1)) != 2:
            return False
        rejected_at = self.ev_rejection_times.get(request_id)
        if rejected_at is None or float(rejected_at) != float(self.current_time):
            return False
        recourse_ids = self.ev_rejected_recovered_same_epoch_ids
        is_new = request_id not in recourse_ids
        recourse_ids.add(request_id)
        return is_new

    def _assign_request_to_vehicle(self, vehicle_id: int, request_id: int) -> bool:
        if request_id not in self.active_requests or vehicle_id not in self.vehicles:
            return False
        vehicle = self.vehicles[vehicle_id]
        request = self.active_requests[request_id]
        if not vehicle.get('is_online', True):
            return False

        # already assigned elsewhere?
        for other_vid, other_v in self.vehicles.items():
            if other_vid != vehicle_id:
                if other_v['assigned_request'] == request_id or other_v['passenger_onboard'] == request_id:
                    return False

        if vehicle['penalty_timer'] > 0:
            return False

        if vehicle['assigned_request'] is not None or vehicle['passenger_onboard'] is not None:
            return False

        # rejection check
        if self._should_reject_request(vehicle_id, request):
            vehicle['rejected_requests'] += 1
            self._record_ev_rejection(vehicle_id)
            if self._is_ev(vehicle_id):
                self.ev_rejected_request_ids.add(request_id)
                self.ev_rejection_times[request_id] = float(self.current_time)
            vehicle['assigned_request'] = request_id
            self.rejected_requests.append(request)
            vehicle['idle_target'] = None
            vehicle['is_stationary'] = False
            vehicle['stationary_duration'] = 0
            return False

        # accept
        vehicle['assigned_request'] = request_id
        self._record_ev_acceptance(vehicle_id)
        self._record_same_epoch_recourse_if_applicable(vehicle_id, request_id)
        vehicle['idle_target'] = None
        vehicle['is_stationary'] = False
        vehicle['stationary_duration'] = 0
        if vehicle['type'] == 1:
            self.ev_requests.append(request)
        return True

    def _pickup_passenger(self, vehicle_id: int) -> bool:
        vehicle = self.vehicles[vehicle_id]
        if vehicle['battery'] <= 0.0:
            self._clear_vehicle_assignments(vehicle_id)
            return False
        if vehicle['assigned_request'] is None:
            return False
        if vehicle['assigned_request'] not in self.active_requests:
            vehicle['assigned_request'] = None
            return False
        request = self.active_requests[vehicle['assigned_request']]
        if vehicle['location'] == request.pickup:
            # check not already picked up by another
            for ov, oveh in self.vehicles.items():
                if ov != vehicle_id and oveh['passenger_onboard'] == vehicle['assigned_request']:
                    vehicle['assigned_request'] = None
                    return False
            vehicle['passenger_onboard'] = vehicle['assigned_request']
            if (
                vehicle.get('type') == 2
                and vehicle['passenger_onboard'] in self.ev_rejected_request_ids
                and vehicle['passenger_onboard']
                not in self.ev_rejected_picked_up_by_aev_ids
            ):
                rejected_request_id = vehicle['passenger_onboard']
                self.ev_rejected_picked_up_by_aev_ids.add(rejected_request_id)
                rejected_at = float(
                    self.ev_rejection_times.get(
                        rejected_request_id,
                        self.current_time,
                    )
                )
                self.recovery_delays.append(
                    max(0.0, float(self.current_time) - rejected_at)
                )
            vehicle['assigned_request'] = None
            route_distance = self._request_trip_distance_km(request)
            vehicle['passenger_trip_distance_total'] = route_distance
            vehicle['passenger_trip_distance_remaining'] = route_distance
            vehicle['passenger_trip_distance_travelled'] = 0.0
            vehicle['passenger_trip_start_coordinates'] = tuple(
                vehicle.get('coordinates', self.zone_coords[request.pickup])
            )
            return True
        return False

    def _dropoff_passenger(self, vehicle_id: int) -> float:
        vehicle = self.vehicles[vehicle_id]
        if vehicle['battery'] <= 0.0:
            self._clear_vehicle_assignments(vehicle_id)
            return self.unserve_penalty
        if vehicle['passenger_onboard'] is None:
            return 0.0
        if vehicle['passenger_onboard'] not in self.active_requests:
            vehicle['passenger_onboard'] = None
            vehicle['assigned_request'] = None
            self._reset_passenger_trip_state(vehicle)
            return 0.0
        request = self.active_requests[vehicle['passenger_onboard']]
        trip_remaining = float(vehicle.get('passenger_trip_distance_remaining', 0.0) or 0.0)
        if vehicle['location'] == request.dropoff and trip_remaining <= 1e-9:
            completed = self.active_requests.pop(vehicle['passenger_onboard'])
            completed_real_hour = float(self.get_hour_of_day())
            completed_hour_bucket = int(completed_real_hour) % 24
            completed_date = str(self._current_date_label())
            self.completed_requests.append(completed)
            if completed.request_id in self.ev_rejected_request_ids:
                self.ev_rejected_completed_ids.add(completed.request_id)
            if vehicle['type'] == 1:
                self.completed_requests_ev.append(completed)
            completed.completed_real_hour = completed_real_hour
            completed.completed_hour_bucket = completed_hour_bucket
            completed.completed_date = completed_date
            self.completed_request_time_records.append({
                'request_id': getattr(completed, 'request_id', None),
                'vehicle_id': int(vehicle_id),
                'vehicle_type': int(vehicle.get('type', 0)),
                'completed_date': completed_date,
                'completed_real_hour': completed_real_hour,
                'completed_hour': completed_hour_bucket,
                'completed_orders': 1,
                'completed_ev_orders': 1 if vehicle.get('type') == 1 else 0,
                'completed_aev_orders': 1 if vehicle.get('type') == 2 else 0,
                'request_value': float(getattr(completed, 'final_value', 0.0)),
                'pickup_zone': getattr(completed, 'pickup', None),
                'dropoff_zone': getattr(completed, 'dropoff', None),
            })
            self.request_value_sum += completed.final_value
            earnings = completed.final_value
            vehicle['service_earnings'] += earnings
            vehicle['daily_salary'] += earnings
            vehicle['salary_ratio'] = vehicle['daily_salary'] / self.ev_basesalary if self.ev_basesalary > 0 else 0.0
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['whether_finishrequest'] = True
            vehicle['passenger_onboard'] = None
            self._reset_passenger_trip_state(vehicle)
            self._record_ev_completion(vehicle_id)
            if not self.daily_drop_off:
                self._handle_vehicle_dropout_event(vehicle_id)
            return earnings
        return 0.0

    def _clear_vehicle_assignments(self, vehicle_id: int):
        v = self.vehicles[vehicle_id]
        v['target_location'] = None
        v['idle_target'] = None
        v['assigned_request'] = None
        v['passenger_onboard'] = None
        v['charging_target'] = None
        self._reset_passenger_trip_state(v)

    @staticmethod
    def _reset_passenger_trip_state(vehicle: dict):
        vehicle['passenger_trip_distance_total'] = 0.0
        vehicle['passenger_trip_distance_remaining'] = 0.0
        vehicle['passenger_trip_distance_travelled'] = 0.0
        vehicle['passenger_trip_start_coordinates'] = None

    # ==================================================================
    # Movement helpers (zone-to-zone)
    # ==================================================================

    def _coerce_target_zone_id(self, target_zone) -> int | None:
        if target_zone is None:
            return None
        if isinstance(target_zone, (int, np.integer)):
            return int(target_zone)
        if isinstance(target_zone, (tuple, list)) and len(target_zone) >= 2:
            mapped_zone = self.map_zone(target_zone)
            return int(mapped_zone) if mapped_zone >= 0 else None
        return None

    def _move_vehicle_one_step(self, vehicle_id: int, target_zone: int) -> float:
        """Move a vehicle by one epoch worth of distance toward the target zone centroid."""
        vehicle = self.vehicles[vehicle_id]
        target_zone = self._coerce_target_zone_id(target_zone)
        if target_zone is None:
            self._clear_vehicle_assignments(vehicle_id)
            return 0.0
        cur = int(vehicle['location'])
        if cur == target_zone:
            vehicle['zone_id'] = cur
            vehicle['coordinates'] = self.zone_coords.get(cur, vehicle['coordinates'])
            return 0.0

        cur_lat, cur_lon = vehicle.get('coordinates', self.zone_coords.get(cur, (0.0, 0.0)))
        target_lat, target_lon = self.zone_coords.get(int(target_zone), (cur_lat, cur_lon))
        remaining_km = _haversine_km(cur_lat, cur_lon, target_lat, target_lon)
        max_step_km = max(0.0, self.average_velocity_kmph * (self.EPOCH_LENGTH / 3600.0))

        if remaining_km <= 1e-9 or max_step_km <= 0.0:
            new_lat, new_lon = target_lat, target_lon
            moved_km = 0.0
        else:
            moved_km = min(max_step_km, remaining_km)
            ratio = moved_km / remaining_km
            new_lat = cur_lat + (target_lat - cur_lat) * ratio
            new_lon = cur_lon + (target_lon - cur_lon) * ratio
            if moved_km >= remaining_km - 1e-6:
                new_lat, new_lon = target_lat, target_lon

        vehicle['coordinates'] = (float(new_lat), float(new_lon))
        mapped_zone = self.map_zone(vehicle['coordinates'])
        if mapped_zone < 0:
            if moved_km >= remaining_km - 1e-6:
                mapped_zone = int(target_zone)
            else:
                mapped_zone = self._nearest_zone(new_lat, new_lon, prefer_polygon=False)
        vehicle['location'] = int(mapped_zone)
        vehicle['zone_id'] = int(mapped_zone)
        vehicle['total_distance'] += moved_km
        vehicle['battery'] -= moved_km * self.battery_consum
        vehicle['battery'] = max(0.0, vehicle['battery'])
        if vehicle['battery'] <= 0.0:
            vehicle['needs_emergency_charging'] = True
            self._clear_vehicle_assignments(vehicle_id)
        # track
        if vehicle_id not in self.vehicle_position_history:
            self.vehicle_position_history[vehicle_id] = []
        self.vehicle_position_history[vehicle_id].append({
            'zone': int(mapped_zone),
            'coordinates': vehicle['coordinates'],
            'time': self.current_time,
        })
        return moved_km

    def _move_passenger_trip_one_step(self, vehicle_id: int, request) -> float:
        """Advance an occupied trip using TLC route length and simulator speed."""
        vehicle = self.vehicles[vehicle_id]
        total_km = float(
            vehicle.get('passenger_trip_distance_total', 0.0)
            or self._request_trip_distance_km(request)
        )
        remaining_km = float(
            vehicle.get('passenger_trip_distance_remaining', total_km)
        )
        travelled_km = float(
            vehicle.get('passenger_trip_distance_travelled', total_km - remaining_km)
        )
        max_step_km = max(
            0.0,
            float(self.average_velocity_kmph) * (float(self.EPOCH_LENGTH) / 3600.0),
        )
        moved_km = min(max_step_km, max(0.0, remaining_km))
        remaining_km = max(0.0, remaining_km - moved_km)
        travelled_km = min(total_km, travelled_km + moved_km)

        start_lat, start_lon = vehicle.get(
            'passenger_trip_start_coordinates',
            self.zone_coords.get(request.pickup, vehicle.get('coordinates', (0.0, 0.0))),
        ) or self.zone_coords.get(request.pickup, vehicle.get('coordinates', (0.0, 0.0)))
        target_lat, target_lon = self.zone_coords.get(
            int(request.dropoff), vehicle.get('coordinates', (start_lat, start_lon))
        )
        progress = 1.0 if total_km <= 1e-9 else min(1.0, travelled_km / total_km)
        new_lat = start_lat + (target_lat - start_lat) * progress
        new_lon = start_lon + (target_lon - start_lon) * progress
        if remaining_km <= 1e-9:
            new_lat, new_lon = target_lat, target_lon
            mapped_zone = int(request.dropoff)
        else:
            mapped_zone = self.map_zone((new_lat, new_lon))
            if mapped_zone < 0:
                mapped_zone = self._nearest_zone(new_lat, new_lon, prefer_polygon=False)

        vehicle['passenger_trip_distance_total'] = total_km
        vehicle['passenger_trip_distance_remaining'] = remaining_km
        vehicle['passenger_trip_distance_travelled'] = travelled_km
        vehicle['coordinates'] = (float(new_lat), float(new_lon))
        vehicle['location'] = int(mapped_zone)
        vehicle['zone_id'] = int(mapped_zone)
        vehicle['total_distance'] += moved_km
        vehicle['battery'] = max(0.0, vehicle['battery'] - moved_km * self.battery_consum)
        if vehicle['battery'] <= 0.0:
            vehicle['needs_emergency_charging'] = True
            self._clear_vehicle_assignments(vehicle_id)

        self.vehicle_position_history.setdefault(vehicle_id, []).append({
            'zone': int(mapped_zone),
            'coordinates': vehicle['coordinates'],
            'time': self.current_time,
        })
        return moved_km

    def _execute_movement_towards_target(self, vehicle_id: int) -> float:
        vehicle = self.vehicles[vehicle_id]
        if vehicle['charging_station'] is not None:
            return -0.2

        target_zone: int | None = None
        if vehicle['passenger_onboard'] is not None:
            req = self.active_requests.get(vehicle['passenger_onboard'])
            if req:
                dist = self._move_passenger_trip_one_step(vehicle_id, req)
                return self._movement_cost(dist) if dist > 0 else -0.05
        elif vehicle['assigned_request'] is not None:
            req = self.active_requests.get(vehicle['assigned_request'])
            if req:
                target_zone = req.pickup
        elif vehicle.get('target_location') is not None:
            target_zone = self._coerce_target_zone_id(vehicle['target_location'])

        if target_zone is not None:
            dist = self._move_vehicle_one_step(vehicle_id, target_zone)
            return self._movement_cost(dist) if dist > 0 else -0.05
        return 0.0

    def _execute_movement_towards_charging_station(self, vehicle_id: int, station_id: int) -> float:
        vehicle = self.vehicles[vehicle_id]
        station = self.charging_manager.stations[station_id]
        station_zone = station.location

        if vehicle['location'] == station_zone:
            vehicle['charging_target'] = None
            self._clear_aev_notarrived_if_arrived(vehicle_id, station_id)
            self._mark_charging_queue_arrival(vehicle_id, station_id)
            success = station.start_charging(str(vehicle_id))
            if success:
                self._mark_charging_started(vehicle_id, station_id)
                vehicle['charging_station'] = station_id
                self._set_vehicle_charging_session(vehicle_id)
                vehicle['charging_count'] += 1
                vehicle['target_location'] = None
                return -self.charging_penalty
            return self._charging_wait_step_penalty(vehicle_id, station_id)

        dist = self._move_vehicle_one_step(vehicle_id, station_zone)
        # arrived after move?
        if vehicle['location'] == station_zone:
            vehicle['charging_target'] = None
            self._clear_aev_notarrived_if_arrived(vehicle_id, station_id)
            self._mark_charging_queue_arrival(vehicle_id, station_id)
            success = station.start_charging(str(vehicle_id))
            if success:
                self._mark_charging_started(vehicle_id, station_id)
                vehicle['charging_station'] = station_id
                self._set_vehicle_charging_session(vehicle_id)
                vehicle['charging_count'] += 1
                vehicle['target_location'] = None
                return -self.charging_penalty
            return self._charging_wait_step_penalty(vehicle_id, station_id)
        return self._movement_cost(dist)

    def _battery_required_to_location(self, vehicle_id: int, target_zone: int) -> float:
        vehicle = self.vehicles[vehicle_id]
        target_zone = self._coerce_target_zone_id(target_zone)
        if target_zone is None:
            return 0.0
        return self.get_distance_km(vehicle['location'], target_zone) * self.battery_consum

    def _min_battery_required_to_reach_charging_from_location(self, location: int) -> float:
        if not self.charging_manager.stations:
            return 0.0
        nearest_distance = min(
            self.get_distance_km(location, station.location)
            for station in self.charging_manager.stations.values()
        )
        return nearest_distance * self.battery_consum

    def _post_action_battery_reserve(self, location: int) -> float:
        if 0 <= int(location) < self.nearest_charging_distance.shape[0]:
            nearest_distance = float(self.nearest_charging_distance[int(location)])
            return max(self.min_battery_level, nearest_distance * self.battery_consum + 0.01)
        return max(
            self.min_battery_level,
            self._min_battery_required_to_reach_charging_from_location(location) + 0.01,
        )

    def _charge_target_for_battery(self, battery: float) -> float:
        battery = float(np.clip(battery, 0.0, 1.0))
        target = float(getattr(self, 'charge_target_soc', 0.80))
        if battery < target:
            return target
        return min(1.0, battery + float(getattr(self, 'charge_topup_soc', 0.05)))

    def _charge_duration_for_battery(self, battery: float) -> int:
        target = self._charge_target_for_battery(battery)
        deficit = max(0.0, target - float(battery))
        raw_epochs = int(math.ceil(deficit / max(float(self.chargeincrease_per_epoch), 1e-6)))
        return int(min(
            int(getattr(self, 'max_charging_session_epochs', max(1, raw_epochs))),
            max(int(getattr(self, 'min_charging_session_epochs', 1)), raw_epochs),
        ))

    def _charge_duration_for_vehicle(self, vehicle_id: int) -> int:
        vehicle = self.vehicles.get(vehicle_id, {})
        return self._charge_duration_for_battery(float(vehicle.get('battery', self.min_battery_level)))

    def _set_vehicle_charging_session(self, vehicle_id: int):
        vehicle = self.vehicles[vehicle_id]
        vehicle['charge_target_soc'] = self._charge_target_for_battery(float(vehicle.get('battery', 0.0)))
        vehicle['charging_time_left'] = self._charge_duration_for_battery(float(vehicle.get('battery', 0.0)))

    def _can_reach_location_with_battery(self, vehicle_id: int, target_zone: int, reserve: float = 0.01) -> bool:
        vehicle = self.vehicles[vehicle_id]
        required = self._battery_required_to_location(vehicle_id, target_zone)
        return required <= max(0.0, float(vehicle.get('battery', 0.0)) - reserve)

    def _can_reach_charging_station(self, vehicle_id: int, station_id: int, reserve: float = 0.01) -> bool:
        station = self.charging_manager.stations.get(station_id)
        if station is None:
            return False
        return self._can_reach_location_with_battery(vehicle_id, station.location, reserve=reserve)

    def _reachable_charging_station_probs(self, vehicle_id: int, station_probs: Dict[int, float]) -> Dict[int, float]:
        reachable = {
            int(sid): float(prob)
            for sid, prob in station_probs.items()
            if self._can_reach_charging_station(vehicle_id, int(sid))
        }
        total_prob = sum(reachable.values())
        if total_prob <= 0:
            return {}
        return {sid: prob / total_prob for sid, prob in reachable.items()}

    def _count_vehicles_unable_to_reach_charging(self) -> int:
        count = 0
        for vid, vehicle in self.vehicles.items():
            if not vehicle.get('is_online', True):
                continue
            if not any(self._can_reach_charging_station(vid, sid) for sid in self.charging_manager.stations):
                count += 1
        return count

    def _execute_movement_towards_idle(self, vehicle_id: int, target_zone) -> float:
        target_zone = self._coerce_target_zone_id(target_zone)
        if target_zone is None:
            self.vehicles[vehicle_id]['idle_target'] = None
            self.vehicles[vehicle_id]['target_location'] = None
            return 0.0
        vehicle = self.vehicles[vehicle_id]
        vehicle['idle_target'] = target_zone
        vehicle['target_location'] = target_zone
        if vehicle['charging_station'] is not None:
            return 0.0
        if vehicle['location'] == target_zone:
            vehicle['idle_target'] = None
            vehicle['target_location'] = None
            return 0.0
        dist = self._move_vehicle_one_step(vehicle_id, target_zone)
        if vehicle['location'] == target_zone:
            vehicle['idle_target'] = None
            vehicle['target_location'] = None
        return self._movement_cost(dist)

    def _move_vehicle_to_charging_station(self, vehicle_id: int, station_id: int):
        if vehicle_id in self.vehicles and station_id in self.charging_manager.stations:
            vehicle = self.vehicles[vehicle_id]
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = station_id
            vehicle['idle_target'] = None
            vehicle['charging_station'] = None
            station = self.charging_manager.stations[station_id]
            vehicle['target_location'] = station.location
            self._register_aev_notarrived_reservation(vehicle_id, station_id)

    def _clear_aev_notarrived_reservations(self, vehicle_id: int):
        if self._is_ev(vehicle_id):
            return
        vehicle_key = str(vehicle_id)
        for station in self.charging_manager.stations.values():
            station.charging_queue_notarrived = [
                queued_vehicle
                for queued_vehicle in station.charging_queue_notarrived
                if str(queued_vehicle) != vehicle_key
            ]

    def _register_aev_notarrived_reservation(self, vehicle_id: int, station_id: int):
        if self._is_ev(vehicle_id):
            return
        self._clear_aev_notarrived_reservations(vehicle_id)
        station = self.charging_manager.stations.get(station_id)
        if station is None:
            return
        vehicle_key = str(vehicle_id)
        existing = {str(queued_vehicle) for queued_vehicle in station.charging_queue_notarrived}
        if vehicle_key not in existing:
            station.charging_queue_notarrived.append(vehicle_key)

    def _clear_aev_notarrived_if_arrived(self, vehicle_id: int, station_id: int):
        if self._is_ev(vehicle_id):
            return
        station = self.charging_manager.stations.get(station_id)
        if station is None:
            return
        vehicle_key = str(vehicle_id)
        station.charging_queue_notarrived = [
            queued_vehicle
            for queued_vehicle in station.charging_queue_notarrived
            if str(queued_vehicle) != vehicle_key
        ]

    def _queue_vehicle_key(self, vehicle_id: int) -> str:
        return str(vehicle_id)

    def _is_vehicle_waiting_for_charger(self, vehicle_id: int, station_id: int | None = None) -> bool:
        vehicle_key = self._queue_vehicle_key(vehicle_id)
        stations = (
            [self.charging_manager.stations.get(station_id)]
            if station_id is not None else self.charging_manager.stations.values()
        )
        for station in stations:
            if station is None:
                continue
            if vehicle_key in {str(queued) for queued in getattr(station, 'charging_queue', [])}:
                return True
        return False

    def _charging_queue_feature_snapshot(self, vehicle_id: int, station_id: int, current_time=None):
        value_function = getattr(self, 'value_function', None)
        if value_function is None or not hasattr(value_function, '_queue_features'):
            return None
        station = self.charging_manager.stations.get(station_id)
        if station is None:
            return None
        vehicle = self.vehicles.get(vehicle_id, {})
        try:
            return value_function._queue_features(
                station_id=station_id,
                target_location=station.location,
                vehicle_id=vehicle_id,
                vehicle_location=vehicle.get('location', station.location),
                current_time=float(self.current_time if current_time is None else current_time),
                num_requests=float(len(getattr(self, 'active_requests', {}))),
                travel_duration=0.0,
            )
        except Exception:
            return None

    def _mark_charging_queue_arrival(self, vehicle_id: int, station_id: int):
        if station_id not in self.charging_manager.stations:
            return
        key = (int(vehicle_id), int(station_id))
        if key in self._charging_queue_arrivals:
            return
        self._charging_queue_arrivals[key] = {
            'arrival_time': float(self.current_time),
            'features': self._charging_queue_feature_snapshot(vehicle_id, station_id),
            'vehicle_location': self.vehicles.get(vehicle_id, {}).get('location'),
        }

    def _mark_charging_started(self, vehicle_id: int, station_id: int):
        key = (int(vehicle_id), int(station_id))
        arrival = self._charging_queue_arrivals.pop(key, None)
        arrival_time = float(arrival.get('arrival_time', self.current_time)) if arrival else float(self.current_time)
        observed_wait = max(0.0, float(self.current_time) - arrival_time)
        self.charging_wait_observations.append({
            'vehicle_id': int(vehicle_id),
            'station_id': int(station_id),
            'arrival_time': arrival_time,
            'start_time': float(self.current_time),
            'observed_wait': observed_wait,
        })
        for value_function in (getattr(self, 'value_function', None), getattr(self, 'value_function_ev', None)):
            if value_function is None or not hasattr(value_function, 'store_queue_experience'):
                continue
            try:
                station = self.charging_manager.stations.get(station_id)
                value_function.store_queue_experience(
                    vehicle_id=vehicle_id,
                    station_id=station_id,
                    target_location=getattr(station, 'location', None),
                    vehicle_location=(arrival or {}).get('vehicle_location', self.vehicles.get(vehicle_id, {}).get('location')),
                    current_time=arrival_time,
                    num_requests=len(getattr(self, 'active_requests', {})),
                    observed_wait=observed_wait,
                    features=(arrival or {}).get('features'),
                )
            except Exception:
                continue

    def _charging_wait_step_penalty(self, vehicle_id: int, station_id: int) -> float:
        self._mark_charging_queue_arrival(vehicle_id, station_id)
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is not None:
            vehicle['charging_target'] = station_id
            vehicle['target_location'] = None
        penalty = max(0.0, float(getattr(self, 'charging_wait_penalty_per_step', 0.5)))
        self.charging_wait_steps += 1
        self.charging_wait_penalty_total += penalty
        return -penalty

    # ==================================================================
    # Execute action  (mirrors ChargingIntegratedEnvironment)
    # ==================================================================

    def _execute_action(self, vehicle_id: int, action) -> Tuple[float, float]:
        vehicle = self.vehicles[vehicle_id]
        if not vehicle.get('is_online', True):
            return 0.0, 0.0
        if vehicle_id not in self.storeactions or self.storeactions[vehicle_id] is None:
            self.storeactions[vehicle_id] = action
            self.storeactions[vehicle_id].dur_reward = 0
            self.storeactions[vehicle_id].current_time = self.current_time

        reward = 0.0
        dur_reward = 0.0

        if isinstance(action, ChargingAction):
            vehicle['idle_target'] = None
            if vehicle['charging_station'] is None:
                sid = action.charging_station_id
                if sid in self.charging_manager.stations:
                    if vehicle['location'] == self.charging_manager.stations[sid].location:
                        vehicle['charging_target'] = None
                        self._clear_aev_notarrived_if_arrived(vehicle_id, sid)
                        self._mark_charging_queue_arrival(vehicle_id, sid)
                        success = self.charging_manager.stations[sid].start_charging(str(vehicle_id))
                        if success:
                            self._mark_charging_started(vehicle_id, sid)
                            vehicle['charging_station'] = sid
                            self._set_vehicle_charging_session(vehicle_id)
                            vehicle['charging_count'] += 1
                            vehicle['target_location'] = None
                            reward = -self.charging_penalty - np.random.random() * self.charging_reward_noise
                        else:
                            reward = self._charging_wait_step_penalty(vehicle_id, sid)
                    else:
                        vehicle['target_charging_station'] = sid
                        reward = self._execute_movement_towards_charging_station(vehicle_id, sid)
                else:
                    reward = -self.charging_penalty
            else:
                reward = -self.charging_penalty
            if vehicle['type'] == 1:
                reward -= self.ev_charging_extra_penalty
                if self.storeactions_ev.get(vehicle_id) is not None:
                    self.storeactions_ev[vehicle_id].dur_reward += reward
            else:
                reward -= self.aev_charging_extra_penalty
                if self.storeactions.get(vehicle_id) is not None:
                    self.storeactions[vehicle_id].dur_reward += reward
            dur_reward = getattr(action, 'dur_reward', 0)

        elif isinstance(action, ServiceAction):
            if vehicle['idle_target'] is not None and vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:
                movement_reward = self._execute_movement_towards_idle(vehicle_id, vehicle.get('idle_target'))
                reward = movement_reward
                if getattr(action, 'was_rejected', False) and not getattr(action, 'rejection_reward_applied', False):
                    reject_reward = getattr(action, 'rejection_reward', None)
                    if reject_reward is None:
                        reject_reward = self._calculate_rejection_reward(
                            vehicle_id,
                            pickup_location=getattr(action, 'target_location', vehicle.get('location')),
                            vehicle_location=getattr(action, 'vehicle_loc', vehicle.get('location')),
                        )
                    reject_reward = float(reject_reward)
                    reward += reject_reward
                    self.rejection_reward_total += reject_reward
                    self.rejection_reward_count += 1
                    self.step_rejection_reward_total += reject_reward
                    self.step_rejection_reward_count += 1
                    action.rejection_reward_applied = True
                self._update_dur_reward(vehicle_id, reward)
                return reward, self._get_dur_reward(vehicle_id)
            elif vehicle.get('target_location') is not None and vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:
                vehicle['idle_target'] = None
                vehicle['target_location'] = None
                return 0.0, 0.0
            elif vehicle['assigned_request'] is not None:
                vehicle['idle_target'] = None
                if self._pickup_passenger(vehicle_id):
                    reward = 0.5 + np.random.normal(
                        0, self.service_event_reward_noise_std
                    )
                elif vehicle['battery'] <= 0.0:
                    self._clear_vehicle_assignments(vehicle_id)
                else:
                    reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(
                        0, self.movement_reward_noise_std
                    )
                self._update_dur_reward(vehicle_id, reward)
            elif vehicle['passenger_onboard'] is not None:
                vehicle['idle_target'] = None
                earnings = self._dropoff_passenger(vehicle_id)
                if earnings > 0:
                    reward = earnings + np.random.normal(
                        0, self.service_event_reward_noise_std
                    )
                elif vehicle['battery'] <= 0.0:
                    self._clear_vehicle_assignments(vehicle_id)
                else:
                    reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(
                        0, self.movement_reward_noise_std
                    )
                self._update_dur_reward(vehicle_id, reward)

        elif isinstance(action, IdleAction):
            if vehicle.get('is_stationary', False):
                idle_pen = -self.idle_penalty
                self._update_dur_reward(vehicle_id, idle_pen)
                return idle_pen, idle_pen
            else:
                idle_pen = -self.idle_penalty
                if vehicle['idle_target'] is None:
                    movement_reward = 0.0
                else:
                    movement_reward = self._execute_movement_towards_idle(vehicle_id, vehicle.get('idle_target'))
                reward = movement_reward + idle_pen
                self._update_dur_reward(vehicle_id, reward)
                dur_reward = self._get_dur_reward(vehicle_id)

        return reward, dur_reward

    def _update_dur_reward(self, vehicle_id: int, reward: float):
        vtype = self.vehicles[vehicle_id]['type']
        if vtype == 1:
            sa = self.storeactions_ev.get(vehicle_id)
            if sa is not None:
                sa.dur_reward += reward
        else:
            sa = self.storeactions.get(vehicle_id)
            if sa is not None:
                sa.dur_reward += reward

    def _get_dur_reward(self, vehicle_id: int) -> float:
        vtype = self.vehicles[vehicle_id]['type']
        sa = self.storeactions_ev.get(vehicle_id) if vtype == 1 else self.storeactions.get(vehicle_id)
        return getattr(sa, 'dur_reward', 0.0) if sa else 0.0

    # ==================================================================
    # step()
    # ==================================================================

    def step(self, actions, storeactions, storeactions_ev=None):
        step_index = self.current_time
        step_start = time.time()
        rewards = {}
        dur_rewards = {}
        next_states = {}
        charging_events = []
        reward_breakdown = {
            'service': 0.0,
            'charging': 0.0,
            'idle': 0.0,
            'other': 0.0,
        }
        self.step_assignments = 0
        self.step_rejections = 0
        self.step_rejection_reward_total = 0.0
        self.step_rejection_reward_count = 0

        execute_actions_start = time.time()
        for vehicle_id, action in actions.items():
            reward, dur_reward = self._execute_action(vehicle_id, action)
            rewards[vehicle_id] = reward
            dur_rewards[vehicle_id] = dur_reward
            next_states[vehicle_id] = self._get_vehicle_state(vehicle_id)
            if isinstance(action, ChargingAction):
                reward_breakdown['charging'] += float(reward)
                charging_events.append({
                    'vehicle_id': vehicle_id,
                    'station_id': action.charging_station_id,
                    'duration': action.charging_duration,
                    'time': self.current_time,
                })
            elif isinstance(action, ServiceAction):
                reward_breakdown['service'] += float(reward)
            elif isinstance(action, IdleAction):
                reward_breakdown['idle'] += float(reward)
            else:
                reward_breakdown['other'] += float(reward)
        execute_actions_time = time.time() - execute_actions_start

        update_env_start = time.time()
        self._update_environment()
        update_env_time = time.time() - update_env_start

        dead_battery_start = time.time()
        self._check_dead_battery_vehicles()
        dead_battery_time = time.time() - dead_battery_start

        q_learning_aev_start = time.time()
        self._activate_bayes_step_context('aev')
        aev_bayes_context = getattr(self, '_bayes_step_contexts', {}).get('aev')
        if self.value_function is not None and hasattr(self.value_function, 'remember_zone_distribution_context'):
            self.value_function.remember_zone_distribution_context(self.current_time, aev_bayes_context)
        self._update_q_learning(storeactions, False)
        q_learning_aev_time = time.time() - q_learning_aev_start

        q_learning_ev_start = time.time()
        self._activate_bayes_step_context('ev')
        ev_bayes_context = getattr(self, '_bayes_step_contexts', {}).get('ev')
        if self.value_function_ev is not None and hasattr(self.value_function_ev, 'remember_zone_distribution_context'):
            self.value_function_ev.remember_zone_distribution_context(self.current_time, ev_bayes_context)
        self._update_q_learning(storeactions_ev, True)
        q_learning_ev_time = time.time() - q_learning_ev_start

        record_usage_start = time.time()
        self._record_charging_usage()
        self._record_hourly_zone_vehicle_snapshot(current_time=self.current_time)
        record_usage_time = time.time() - record_usage_start
        total_step_time = time.time() - step_start
        self._last_step_profile = {
            'step': float(step_index),
            'execute_actions_time_sec': execute_actions_time,
            'update_environment_time_sec': update_env_time,
            'dead_battery_time_sec': dead_battery_time,
            'q_learning_aev_time_sec': q_learning_aev_time,
            'q_learning_ev_time_sec': q_learning_ev_time,
            'record_usage_time_sec': record_usage_time,
            'total_time_sec': total_step_time,
        }
        if self._should_log_timing(step_index):
            print(
                f"⏱ env.step step={int(step_index)} total={total_step_time:.3f}s execute={execute_actions_time:.3f}s "
                f"update_env={update_env_time:.3f}s dead_battery={dead_battery_time:.3f}s "
                f"qlearn_aev={q_learning_aev_time:.3f}s qlearn_ev={q_learning_ev_time:.3f}s usage={record_usage_time:.3f}s",
                flush=True,
            )

        done = self.current_time >= self.episode_length
        ev_idle_mean = float(np.mean(self.ev_idle_durations)) if self.ev_idle_durations else 0.0
        ev_idle_count = len(self.ev_idle_durations)
        ev_in_penalty = sum(1 for vid in self.vehicles if self._in_ev_penalty(vid))
        return next_states, rewards, dur_rewards, done, {
            'charging_events': charging_events,
            'ev_idle_mean': ev_idle_mean,
            'ev_idle_count': ev_idle_count,
            'ev_in_penalty': ev_in_penalty,
            'reward_breakdown': reward_breakdown,
            'step_rejection_reward_total': float(self.step_rejection_reward_total),
            'step_rejection_reward_count': int(self.step_rejection_reward_count),
            'episode_rejection_reward_total': float(self.rejection_reward_total),
            'episode_rejection_reward_count': int(self.rejection_reward_count),
            'rebalancing_profile': dict(self._last_rebalancing_profile),
            'simulation_profile': dict(self._last_simulation_profile),
            'step_profile': dict(self._last_step_profile),
        }

    # ==================================================================
    # _update_environment
    # ==================================================================

    def _update_environment(self):
        previous_date_label = self._current_date_label()
        self.current_time += 1
        current_date_label = self._current_date_label()
        if current_date_label != previous_date_label:
            self._demand_day_cache = None
            self._demand_day_cache_label = None
            self._refresh_daily_driver_states()

        self.reject_number[self.current_time] = 0
        self.assignmentnumber[self.current_time] = 0
        actions = {}
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles}

        charging_phase_start = time.time()
        self._ev_charging_phase(actions, storeactions_ev)
        leftover = [vid for vid in self.vehicles if vid not in actions]



        vehicles_to_rebalance = self._build_vehicles_to_rebalance(leftover)
        aev_to_rebalance = [vid for vid in vehicles_to_rebalance if not self._is_ev(vid)]
        aev_to_rebalance_num = len(aev_to_rebalance)
        
        if self.test_request_less_aev:
            new_requests = self.generate_requests_time(
                sample_num=aev_to_rebalance_num,
                vehicle_ids=aev_to_rebalance,
            )
        else:
            new_requests = self.generate_requests()
        self.whole_req_num += len(new_requests) - len(new_requests)  # already counted in generate_requests

        # Idle timer & penalty timer
        for vid, v in self.vehicles.items():
            if not v.get('is_online', True):
                continue
            if v['assigned_request'] is None and v['passenger_onboard'] is None:
                v['idle_timer'] += 1
            if v['penalty_timer'] > 0:
                v['penalty_timer'] -= 1

        # Charging progress
        for vid, v in self.vehicles.items():
            if not v.get('is_online', True):
                continue
            if v['charging_station'] is not None:
                v['charging_time_left'] -= 1
                charge_target = float(v.get('charge_target_soc', getattr(self, 'charge_target_soc', 0.80)))
                v['battery'] = min(charge_target, 1.0, v['battery'] + self.chargeincrease_per_epoch)
                if v['battery'] >= charge_target - 1e-6:
                    v['charging_time_left'] = 0
                if v['charging_time_left'] <= 0:
                    sid = v['charging_station']
                    if sid in self.charging_manager.stations:
                        self.charging_manager.stations[sid].stop_charging(str(vid))
                    v['charging_station'] = None
                    v.pop('charge_target_soc', None)
                    self.charge_finished += 1
                    self.charge_stats.setdefault(sid, []).append(self.current_time)

        # Sync station state
        for sid, station in self.charging_manager.stations.items():
            for cv in station.current_vehicles:
                cvid = int(cv)
                if cvid in self.vehicles:
                    v = self.vehicles[cvid]
                    v['charging_target'] = None
                    if v['charging_station'] is None:
                        self._mark_charging_started(cvid, sid)
                        v['charging_station'] = sid
                        self._set_vehicle_charging_session(cvid)
            self.idle_charging_num[sid] = station.max_capacity - len(station.current_vehicles)

        self.current_online = sum(1 for v in self.vehicles.values() if v.get('is_online', True))

        # Expire old requests
        expired = [rid for rid, r in self.active_requests.items() if self.current_time > r.pickup_deadline]
        for rid in expired:
            being_served = any(
                v['assigned_request'] == rid or v['passenger_onboard'] == rid
                for v in self.vehicles.values()
            )
            if not being_served:
                self.active_requests.pop(rid, None)

    # ==================================================================
    # Charging / battery helpers
    # ==================================================================

    def _check_dead_battery_vehicles(self) -> list:
        dead = []
        for vid, v in self.vehicles.items():
            if v['battery'] <= 0.0:
                self._clear_vehicle_assignments(vid)
                if v['charging_station'] is None:
                    dead.append(vid)
        return dead

    def _record_charging_usage(self):
        if not self.charging_manager.stations:
            return
        pressure_snapshot = self._record_station_pressure_snapshot()
        total_occ = sum(len(s.current_vehicles) for s in self.charging_manager.stations.values())
        total_st = len(self.charging_manager.stations)
        self.charging_usage_history.append({
            'time': self.current_time,
            'total_occupied': total_occ,
            'total_stations': total_st,
            'vehicles_per_station': total_occ / max(1, total_st),
            'mean_station_pressure': pressure_snapshot['mean_station_pressure'],
            'max_station_pressure': pressure_snapshot['max_station_pressure'],
            'max_station_pressure_ratio': pressure_snapshot['max_station_pressure_ratio'],
            'station_details': {
                sid: len(s.current_vehicles) for sid, s in self.charging_manager.stations.items()
            },
        })
        self._record_hourly_zone_charge_station_snapshot(current_time=self.current_time)

    # ==================================================================
    # Q-learning update (placeholder – real logic in value_function)
    # ==================================================================

    def _update_q_learning(self, actions, ifev=False):
        """Store experiences into value function buffer (mirrors ChargingIntegrated)."""
        # Skip experience storage and filtering entirely in pure evaluation mode
        # (only ADP-MCMF-FT / training runs need to keep populating the buffer).
        if getattr(self, 'evaluatemode', False):
            return
        valuefunction = self.value_function
        valuefunction_ev = self.value_function_ev
        if valuefunction is None or not hasattr(valuefunction, 'experience_buffer'):
            return

        if ifev:
            offlinsedatalen = valuefunction_ev.experience_buffer.__len__() if valuefunction_ev and hasattr(valuefunction_ev, 'experience_buffer') else 0
            if self.current_time % 100 == 0:
                print(f"🔄 Updating EV Q-learning – buffer size: {offlinsedatalen}")
        else:
            offlinsedatalen = valuefunction.experience_buffer.__len__()
            if self.current_time % 100 == 0:
                print(f"🔄 Updating Q-learning – buffer size: {offlinsedatalen}")

        if actions is None:
            return

        for vehicle_id in actions.keys():
            action = actions[vehicle_id]
            if action is None:
                continue
            if not self.vehicles[vehicle_id].get('is_online', True):
                continue
            batterynow = self.vehicles[vehicle_id]['battery']
            current_location = action.vehicle_loc
            current_battery = action.vehicle_battery
            current_request_num = getattr(action, 'req_num', 0)
            veh_curloc = self.vehicles[vehicle_id]['location']
            veh_type = self.vehicles[vehicle_id]['type']
            action_start_time = float(getattr(action, 'current_time', self.current_time))
            default_action_duration = max(1.0, float(self.current_time) - action_start_time)
            action_dur_time = float(getattr(action, 'dur_time', default_action_duration) or default_action_duration)

            if veh_type == 2:
                if not (self.value_function and hasattr(self.value_function, 'store_experience')):
                    continue
                other_vehicles = len([v for v in self.vehicles.values() if v['assigned_request'] is not None])
                num_requests = len(self.active_requests)
                store_threshold = 5
                next_action = getattr(action, 'next_action', None)

                def store_aev_experience(exp_kwargs):
                    post_demand_observation_valid = bool(
                        exp_kwargs.pop('post_demand_observation_valid', False)
                    )
                    post_location = exp_kwargs.get(
                        'post_action_location',
                        exp_kwargs.get('next_vehicle_location', exp_kwargs.get('target_location')),
                    )
                    if post_demand_observation_valid and post_location is not None:
                        exp_kwargs.setdefault(
                            'observed_post_demand',
                            self._active_request_count_at_location(int(post_location)),
                        )
                    exp_kwargs.setdefault(
                        'post_demand_num_requests_at_start',
                        float(current_request_num or 0.0),
                    )
                    exp_kwargs.setdefault('post_demand_current_zone_count', 0.0)
                    exp_kwargs.setdefault('post_demand_snapshot_available', 0.0)
                    exp_kwargs.setdefault(
                        'graph_neighbour_candidates',
                        list(getattr(action, 'graph_neighbour_candidates', []) or []),
                    )
                    if getattr(
                        self.value_function,
                        'standard_entropy_tuning',
                        False,
                    ):
                        exp_kwargs.setdefault(
                            'candidate_actions',
                            list(getattr(action, 'bootstrap_candidates', []) or []),
                        )
                    exp_kwargs.setdefault(
                        'next_graph_neighbour_candidates',
                        list(getattr(next_action, 'graph_neighbour_candidates', []) or []) if next_action is not None else [],
                    )
                    filtered_experiences = self.filter_aev_experiences_for_aev_value_function([exp_kwargs])
                    for filtered_exp in filtered_experiences:
                        self.value_function.store_experience(**filtered_exp)

                if isinstance(action, ServiceAction) and hasattr(action, 'request_id') and next_action is not None and action.dur_reward > store_threshold:
                    r_exec = actions[vehicle_id].dur_reward
                    req = action.target_location
                    next_features = self._next_action_training_features(vehicle_id, next_action, req, fallback_value=r_exec)
                    next_value = next_features['request_value']
                    next_target = next_features['target_location']
                    next_battery = batterynow
                    next_action_type = next_features['action_type']
                    request_obj = self.active_requests.get(action.request_id) if action.request_id in self.active_requests else None
                    req_final_value = request_obj.final_value if request_obj else r_exec
                    store_aev_experience({
                        'vehicle_id': vehicle_id,
                        'action_type': f"assign_{action.request_id}",
                        'vehicle_location': actions[vehicle_id].vehicle_loc,
                        'target_location': req,
                        'current_time': action_start_time,
                        'reward': r_exec,
                        'next_vehicle_location': actions[vehicle_id].vehicle_loc_post,
                        'next_target_location': next_target,
                        'battery_level': current_battery,
                        'next_battery_level': next_battery,
                        'other_vehicles': other_vehicles,
                        'num_requests': num_requests,
                        'request_value': req_final_value,
                        'next_action_type': next_action_type,
                        'next_request_value': next_value,
                        'dur_time': action_dur_time,
                        'is_system_done': getattr(self, 'done', False),
                        'vehicle_idle_time': getattr(action, 'idle_time', 0),
                        'next_vehicle_idle_time': self.vehicles[vehicle_id]['idle_timer'],
                        'post_action_location': getattr(action, 'post_action_location', None),
                        'post_action_distance': getattr(action, 'post_action_distance', None),
                        'post_action_duration': getattr(action, 'post_action_duration', None),
                        'post_action_zoneid': getattr(action, 'post_action_zoneid', None),
                        'next_post_action_distance': next_features.get('post_action_distance', 0.0),
                        'next_post_action_duration': next_features.get('post_action_duration', 0.0),
                        'next_post_action_zoneid': next_features.get('post_action_zoneid', 0),
                        'next_candidate_actions': getattr(next_action, 'bootstrap_candidates', []),
                        'post_demand_observation_valid': True,
                    })

                elif isinstance(action, ChargingAction) and hasattr(action, 'charging_station_id'):
                    st_id = action.charging_station_id
                    next_action = getattr(action, 'next_action', None)
                    if hasattr(self, 'charging_manager') and st_id in self.charging_manager.stations and batterynow > self.chargeincrease_per_epoch and next_action is not None:
                        station_loc = self.charging_manager.stations[st_id].location
                        r_exec = actions[vehicle_id].dur_reward
                        next_features = self._next_action_training_features(vehicle_id, next_action, station_loc, fallback_value=0.0)
                        next_value = next_features['request_value']
                        next_target = next_features['target_location']
                        next_action_type = next_features['action_type']
                        charge_experience = {
                            'vehicle_id': vehicle_id,
                            'action_type': f"charge_{st_id}",
                            'vehicle_location': actions[vehicle_id].vehicle_loc,
                            'target_location': station_loc,
                            'current_time': action_start_time,
                            'reward': r_exec,
                            'next_vehicle_location': actions[vehicle_id].vehicle_loc_post,
                            'next_target_location': next_target,
                            'battery_level': current_battery,
                            'next_battery_level': batterynow,
                            'other_vehicles': other_vehicles,
                            'num_requests': num_requests,
                            'request_value': 0.0,
                            'next_action_type': next_action_type,
                            'next_request_value': next_value,
                            'dur_time': action_dur_time,
                            'is_system_done': getattr(self, 'done', False),
                            'vehicle_idle_time': getattr(action, 'idle_time', 0),
                            'next_vehicle_idle_time': self.vehicles[vehicle_id]['idle_timer'],
                            'next_post_action_distance': next_features.get('post_action_distance', 0.0),
                            'next_post_action_duration': next_features.get('post_action_duration', 0.0),
                            'next_post_action_zoneid': next_features.get('post_action_zoneid', 0),
                            'next_candidate_actions': getattr(next_action, 'bootstrap_candidates', []),
                            'post_demand_observation_valid': True,
                        }
                        masac_mode = getattr(
                            getattr(self, 'value_function', None),
                            'zone_distribution_mode',
                            '',
                        )
                        legacy_former2 = masac_mode == 'st_masac_gat_former2'
                        former2_queue_feature = masac_mode in {
                            'st_masac_gat_former2_queue_feature',
                            'st_masac_gat_former2_queue_feature_greedy_alpha',
                            'st_masac_gat_former2_queue_feature_fixed_alpha',
                        }
                        if former2_queue_feature:
                            queue_snapshot = self._charging_queue_feature_snapshot(
                                vehicle_id,
                                st_id,
                                current_time=action_start_time,
                            )
                            charge_experience['queue_features'] = queue_snapshot
                            freeze_queue_feature = getattr(
                                self.value_function,
                                'queue_wait_feature_from_snapshot',
                                None,
                            )
                            if callable(freeze_queue_feature):
                                charge_experience['queue_wait_feature'] = freeze_queue_feature(
                                    queue_snapshot
                                )
                        elif not legacy_former2:
                            charge_source_loc = actions[vehicle_id].vehicle_loc
                            if charge_source_loc is None:
                                charge_source_loc = veh_curloc
                            charge_distance = float(self.get_distance_km(charge_source_loc, station_loc))
                            charge_travel_duration = float(self.get_travel_time(charge_source_loc, station_loc)) if charge_distance > 0.0 else 0.0
                            charge_session_duration = float(self._charge_duration_for_vehicle(vehicle_id))
                            charge_zoneid = int(self.get_zone_embedding_id(station_loc))
                            charge_experience.update({
                                'target_station_id': int(st_id),
                                'vehicle_location': charge_source_loc,
                                'target_distance': charge_distance,
                                'target_zoneid': charge_zoneid,
                                'post_action_location': station_loc,
                                'post_action_distance': charge_distance,
                                'post_action_duration': charge_travel_duration + charge_session_duration,
                                'post_action_zoneid': charge_zoneid,
                                'queue_features': self._charging_queue_feature_snapshot(
                                    vehicle_id,
                                    st_id,
                                    current_time=action_start_time,
                                ),
                            })
                        store_aev_experience(charge_experience)

                elif isinstance(action, IdleAction):
                    next_action = getattr(action, 'next_action', None)
                    idle_target = self._coerce_location_for_candidate(
                        getattr(action, 'target_location', None),
                        veh_curloc,
                    )
                    idle_distance = float(self.get_distance_km(veh_curloc, idle_target))
                    idle_duration = float(self.get_travel_time(veh_curloc, idle_target)) if idle_distance > 0.0 else 0.0
                    idle_kind = getattr(action, 'learning_action_type', None)
                    if idle_kind not in {'idle', 'reloc'}:
                        idle_kind = 'reloc' if idle_distance > 1e-9 else 'idle'
                    if next_action is not None:
                        next_features = self._next_action_training_features(vehicle_id, next_action, idle_target, fallback_value=actions[vehicle_id].dur_reward)
                        next_value = next_features['request_value']
                        next_action_type = next_features['action_type']
                        next_target = next_features['target_location']
                    else:
                        next_value = actions[vehicle_id].dur_reward
                        next_action_type = "idle"
                        next_features = {}
                        next_target = veh_curloc
                    r_exec = actions[vehicle_id].dur_reward
                    learning_reward = float(r_exec)
                    if idle_kind == 'reloc':
                        learning_reward -= float(self.learning_reloc_penalty_base)
                        learning_reward -= float(self.learning_reloc_penalty_per_km) * idle_distance
                    else:
                        learning_reward -= float(self.learning_wait_penalty)
                    store_aev_experience({
                        'vehicle_id': vehicle_id,
                        'action_type': idle_kind,
                        'vehicle_location': actions[vehicle_id].vehicle_loc,
                        'target_location': idle_target,
                        'current_time': action_start_time,
                        'reward': learning_reward,
                        'next_vehicle_location': actions[vehicle_id].vehicle_loc_post,
                        'next_target_location': next_target,
                        'battery_level': current_battery,
                        'next_battery_level': batterynow,
                        'other_vehicles': other_vehicles,
                        'num_requests': num_requests,
                        'request_value': 0.0,
                        'next_action_type': next_action_type,
                        'next_request_value': next_value,
                        'dur_time': action_dur_time,
                        'is_system_done': getattr(self, 'done', False),
                        'vehicle_idle_time': getattr(action, 'idle_time', 0),
                        'next_vehicle_idle_time': self.vehicles[vehicle_id]['idle_timer'],
                        'post_action_location': idle_target,
                        'post_action_distance': getattr(action, 'post_action_distance', idle_distance),
                        'post_action_duration': getattr(action, 'post_action_duration', idle_duration),
                        'post_action_zoneid': getattr(action, 'post_action_zoneid', self.get_zone_embedding_id(idle_target)),
                        'next_post_action_distance': next_features.get('post_action_distance', 0.0),
                        'next_post_action_duration': next_features.get('post_action_duration', 0.0),
                        'next_post_action_zoneid': next_features.get('post_action_zoneid', 0),
                        'next_candidate_actions': getattr(next_action, 'bootstrap_candidates', []) if next_action is not None else [],
                        'post_demand_observation_valid': next_action is not None,
                    })

            else:  # EV (type==1)
                if not (self.value_function_ev and hasattr(self.value_function_ev, 'store_experience')):
                    continue
                other_vehicles = len([v for v in self.vehicles.values() if v['assigned_request'] is not None])
                num_requests = len(self.active_requests)
                store_threshold = 5
                next_action = getattr(action, 'next_action', None)
                next_target = action.next_target_location if hasattr(action, 'next_target_location') else 0
                was_rejected = bool(getattr(action, 'was_rejected', False))
                if isinstance(action, ServiceAction) and hasattr(action, 'request_id') and was_rejected:
                    reject_target = getattr(action, 'target_location', current_location)
                    reject_distance = self._manhattan_distance_loc(current_location, reject_target)
                    if hasattr(self.value_function_ev, 'store_rejection_experience'):
                        self.value_function_ev.store_rejection_experience(
                            vehicle_id=vehicle_id,
                            request_id=action.request_id,
                            vehicle_location=current_location,
                            pickup_location=reject_target,
                            current_time=action_start_time,
                            distance=reject_distance,
                            rejection_reason=getattr(action, 'rejection_reason', 'driver_reject'),
                            rejection_sample=getattr(action, 'rejection_sample', None),
                        )

                elif isinstance(action, ServiceAction) and hasattr(action, 'request_id') and next_action is not None and action.dur_reward > store_threshold:
                    r_exec = actions[vehicle_id].dur_reward
                    next_features = self._next_action_training_features(vehicle_id, next_action, next_target, fallback_value=r_exec)
                    next_value = next_features['request_value']
                    next_target = next_features['target_location']
                    next_action_type = next_features['action_type']
                    request_obj = self.active_requests.get(action.request_id) if action.request_id in self.active_requests else None
                    req_final_value = request_obj.final_value if request_obj else r_exec
                    if hasattr(self.value_function_ev, 'store_acceptance_experience'):
                        accept_target = getattr(action, 'target_location', current_location)
                        accept_distance = self._manhattan_distance_loc(current_location, accept_target)
                        self.value_function_ev.store_acceptance_experience(
                            vehicle_id=vehicle_id,
                            request_id=action.request_id,
                            vehicle_location=current_location,
                            pickup_location=accept_target,
                            current_time=action_start_time,
                            distance=accept_distance,
                            rejection_sample=getattr(action, 'rejection_sample', None),
                        )
                    self.value_function_ev.store_experience(
                        vehicle_id=vehicle_id, action_type=f"assign_{action.request_id}",
                        vehicle_location=actions[vehicle_id].vehicle_loc, target_location=action.target_location,
                        current_time=action_start_time, reward=r_exec,
                        next_vehicle_location=actions[vehicle_id].vehicle_loc_post, next_target_location=next_target,
                        battery_level=current_battery, next_battery_level=batterynow,
                        other_vehicles=other_vehicles, num_requests=num_requests,
                        request_value=req_final_value, next_action_type=next_action_type,
                        next_request_value=next_value, dur_time=action_dur_time,
                        is_system_done=getattr(self, 'done', False),
                        vehicle_idle_time=getattr(action, 'idle_time', 0),
                        next_vehicle_idle_time=self.vehicles[vehicle_id]['idle_timer'],
                        post_action_location=getattr(action, 'post_action_location', None),
                        post_action_distance=getattr(action, 'post_action_distance', None),
                        post_action_duration=getattr(action, 'post_action_duration', None),
                        post_action_zoneid=getattr(action, 'post_action_zoneid', None),
                        next_post_action_distance=next_features.get('post_action_distance', 0.0),
                        next_post_action_duration=next_features.get('post_action_duration', 0.0),
                        next_post_action_zoneid=next_features.get('post_action_zoneid', 0),
                        graph_neighbour_candidates=list(getattr(action, 'graph_neighbour_candidates', []) or []),
                        next_graph_neighbour_candidates=list(getattr(next_action, 'graph_neighbour_candidates', []) or []),
                        next_candidate_actions=getattr(next_action, 'bootstrap_candidates', []),
                        was_rejected=False,
                        **(
                            {
                                'candidate_actions': getattr(
                                    action,
                                    'bootstrap_candidates',
                                    [],
                                )
                            }
                            if getattr(
                                self.value_function_ev,
                                'standard_entropy_tuning',
                                False,
                            )
                            else {}
                        ))

                elif isinstance(action, ServiceAction) and hasattr(action, 'request_id'):
                    next_action = getattr(action, 'next_action', None)
                    if next_action is not None and action.dur_reward < store_threshold and not was_rejected:
                        r_exec = 0
                        next_features = self._next_action_training_features(vehicle_id, next_action, next_target, fallback_value=r_exec)
                        next_value = next_features['request_value']
                        next_target = next_features['target_location']
                        next_action_type = next_features['action_type']
                        request_obj = self.active_requests.get(action.request_id) if action.request_id in self.active_requests else None
                        req_final_value = request_obj.final_value if request_obj else r_exec
                        self.value_function_ev.store_experience(
                            vehicle_id=vehicle_id, action_type=f"assign_{action.request_id}",
                            vehicle_location=actions[vehicle_id].vehicle_loc, target_location=action.target_location,
                            current_time=action_start_time, reward=r_exec,
                            next_vehicle_location=actions[vehicle_id].vehicle_loc_post, next_target_location=next_target,
                            battery_level=current_battery, next_battery_level=batterynow,
                            other_vehicles=other_vehicles, num_requests=num_requests,
                            request_value=req_final_value, next_action_type=next_action_type,
                            next_request_value=next_value, dur_time=action_dur_time,
                            is_system_done=getattr(self, 'done', False),
                            vehicle_idle_time=getattr(action, 'idle_time', 0),
                            next_vehicle_idle_time=self.vehicles[vehicle_id]['idle_timer'],
                            post_action_location=getattr(action, 'post_action_location', None),
                            post_action_distance=getattr(action, 'post_action_distance', None),
                            post_action_duration=getattr(action, 'post_action_duration', None),
                            post_action_zoneid=getattr(action, 'post_action_zoneid', None),
                            next_post_action_distance=next_features.get('post_action_distance', 0.0),
                            next_post_action_duration=next_features.get('post_action_duration', 0.0),
                            next_post_action_zoneid=next_features.get('post_action_zoneid', 0),
                            graph_neighbour_candidates=list(getattr(action, 'graph_neighbour_candidates', []) or []),
                            next_graph_neighbour_candidates=list(getattr(next_action, 'graph_neighbour_candidates', []) or []),
                            next_candidate_actions=getattr(next_action, 'bootstrap_candidates', []),
                            was_rejected=False,
                            **(
                                {
                                    'candidate_actions': getattr(
                                        action,
                                        'bootstrap_candidates',
                                        [],
                                    )
                                }
                                if getattr(
                                    self.value_function_ev,
                                    'standard_entropy_tuning',
                                    False,
                                )
                            else {}
                            ))

                elif isinstance(action, IdleAction):
                    idle_target = self._coerce_location_for_candidate(
                        getattr(action, 'target_location', None),
                        current_location,
                    )
                    idle_distance = float(self.get_distance_km(current_location, idle_target))
                    idle_duration = (
                        float(self.get_travel_time(current_location, idle_target))
                        if idle_distance > 0.0 else 0.0
                    )
                    next_action = getattr(action, 'next_action', None)
                    if next_action is not None:
                        next_features = self._next_action_training_features(
                            vehicle_id,
                            next_action,
                            idle_target,
                            fallback_value=actions[vehicle_id].dur_reward,
                        )
                        next_value = next_features['request_value']
                        next_action_type = next_features['action_type']
                        next_target = next_features['target_location']
                    else:
                        next_features = {}
                        next_value = actions[vehicle_id].dur_reward
                        next_action_type = 'reloc'
                        next_target = idle_target

                    self.value_function_ev.store_experience(
                        vehicle_id=vehicle_id,
                        action_type='reloc',
                        vehicle_location=actions[vehicle_id].vehicle_loc,
                        target_location=idle_target,
                        current_time=action_start_time,
                        reward=float(actions[vehicle_id].dur_reward),
                        next_vehicle_location=actions[vehicle_id].vehicle_loc_post,
                        next_target_location=next_target,
                        battery_level=current_battery,
                        next_battery_level=batterynow,
                        other_vehicles=other_vehicles,
                        num_requests=num_requests,
                        request_value=0.0,
                        next_action_type=next_action_type,
                        next_request_value=next_value,
                        dur_time=action_dur_time,
                        is_system_done=getattr(self, 'done', False),
                        vehicle_idle_time=getattr(action, 'idle_time', 0),
                        next_vehicle_idle_time=self.vehicles[vehicle_id]['idle_timer'],
                        target_distance=idle_distance,
                        target_zoneid=int(self.get_zone_embedding_id(idle_target)),
                        post_action_location=idle_target,
                        post_action_distance=getattr(action, 'post_action_distance', idle_distance),
                        post_action_duration=getattr(action, 'post_action_duration', idle_duration),
                        post_action_zoneid=getattr(
                            action,
                            'post_action_zoneid',
                            self.get_zone_embedding_id(idle_target),
                        ),
                        next_post_action_distance=next_features.get('post_action_distance', 0.0),
                        next_post_action_duration=next_features.get('post_action_duration', 0.0),
                        next_post_action_zoneid=next_features.get('post_action_zoneid', 0),
                        graph_neighbour_candidates=list(
                            getattr(action, 'graph_neighbour_candidates', []) or []
                        ),
                        next_graph_neighbour_candidates=list(
                            getattr(next_action, 'graph_neighbour_candidates', []) or []
                        ) if next_action is not None else [],
                        next_candidate_actions=getattr(
                            next_action,
                            'bootstrap_candidates',
                            [],
                        ) if next_action is not None else [],
                        **(
                            {
                                'candidate_actions': getattr(
                                    action,
                                    'bootstrap_candidates',
                                    [],
                                )
                            }
                            if getattr(
                                self.value_function_ev,
                                'standard_entropy_tuning',
                                False,
                            )
                            else {}
                        ),
                    )

    # ==================================================================
    # Q-value accessors
    # ==================================================================

    def set_value_function(self, vf):
        self.value_function = vf

    def set_value_function_ev(self, vf):
        self.value_function_ev = vf

    def get_assignment_q_value(self, vehicle_id, target_id, vehicle_location, target_location) -> float:
        if self.value_function and hasattr(self.value_function, 'get_assignment_q_value'):
            vehicle = self.vehicles.get(vehicle_id, {})
            battery = vehicle.get('battery', 1.0)
            other = len([v for v in self.vehicles.values() if v['assigned_request'] is not None])
            nr = len(self.active_requests)
            rv = 0.0
            if target_id in self.active_requests:
                rv = self.active_requests[target_id].final_value
            return self.value_function.get_assignment_q_value(
                vehicle_id, target_id, vehicle_location, target_location,
                self.current_time, other, nr, battery, rv,
            )
        return 0.0

    def get_charging_q_value(self, vehicle_id, station_id, vehicle_location, station_location) -> float:
        if self.value_function and hasattr(self.value_function, 'get_charging_q_value'):
            vehicle = self.vehicles.get(vehicle_id, {})
            battery = vehicle.get('battery', 1.0)
            idle_count = len([v for v in self.vehicles.values()
                              if v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
            return self.value_function.get_charging_q_value(
                vehicle_id, station_id, vehicle_location, station_location,
                self.current_time, max(0, idle_count - 1), len(self.active_requests), battery,
            )
        dist = self.get_distance_km(vehicle_location, station_location)
        charging_epochs = float(self._charge_duration_for_vehicle(vehicle_id))
        return self._movement_cost(dist) - self.charging_penalty * charging_epochs

    # ==================================================================
    # Distance helper (compat with ChargingIntegratedEnvironment)
    # ==================================================================

    def _manhattan_distance_loc(self, a: int, b: int) -> float:
        """Return distance in km between two zone IDs (used by matrix generators)."""
        return self.get_distance_km(a, b)

    def _movement_cost(self, distance_km: float) -> float:
        """Return the common dollar reward adjustment for driven distance."""
        operating_cost = getattr(self, 'operating_cost_per_km', None)
        if operating_cost is None:
            operating_cost = abs(float(getattr(self, 'movingpenalty', -0.08)))
        return -float(operating_cost) * max(
            0.0,
            float(distance_km or 0.0),
        )

    def _request_trip_distance_km(self, request) -> float:
        """Return TLC route distance, falling back for synthetic/legacy requests."""
        route_distance = getattr(request, 'trip_distance_km', None)
        if route_distance is not None:
            try:
                route_distance = float(route_distance)
                if math.isfinite(route_distance) and route_distance >= 0.0:
                    return route_distance
            except (TypeError, ValueError):
                pass
        return float(self.get_distance_km(request.pickup, request.dropoff))

    def _manhattan_distance_loc_time(self, a: int, b: int) -> float:
        return self.get_travel_time_minutes(a, b)

    def _loc_to_xy(self, loc: int):
        c = self.zone_coords.get(loc, (40.75, -73.98))
        return c

    # ==================================================================
    # simulate_motion  helpers & full implementations
    # ==================================================================

    def _build_vehicles_to_rebalance(self, leftover_list):
        """Build filtered list of vehicles eligible for rebalancing (same logic as ChargingIntegrated)."""
        idle_1 = [vid for vid, v in self.vehicles.items()
                if v.get('is_online', True) and v['assigned_request'] is None and v['passenger_onboard'] is None
                and v['charging_station'] is None and v['target_location'] is None and v['penalty_timer'] == 0]
        idle_2 = [vid for vid, v in self.vehicles.items()
                if v.get('is_online', True) and v['needs_emergency_charging']]
        idle_wait = [vid for vid, v in self.vehicles.items()
                   if v.get('is_online', True) and v['is_stationary']
                   and vid not in idle_1 and vid not in idle_2 and v['penalty_timer'] == 0]
        idle_v = [vid for vid, v in self.vehicles.items()
                if v.get('is_online', True) and self._is_ev(vid)
                and v['assigned_request'] is None and v['passenger_onboard'] is None
                and v['charging_station'] is None and v['idle_target'] is not None
                and vid not in idle_2 and vid not in idle_1 and vid not in idle_wait and v['penalty_timer'] == 0]
        idle_all = idle_1 + idle_2 + idle_wait + idle_v

        vehicles_to_rebalance = []
        for vid, v in self.vehicles.items():
            if vid in leftover_list:
                if vid in idle_all:
                    vehicles_to_rebalance.append(vid)
                elif v['battery'] <= self.rebalance_battery_threshold and v['passenger_onboard'] is None and v['assigned_request'] is None:
                    vehicles_to_rebalance.append(vid)

        # Remove ineligible
        vehicles_to_rebalance = [
            vid for vid in vehicles_to_rebalance
            if self.vehicles[vid].get('is_online', True)
            and self.vehicles[vid]['assigned_request'] is None
            and self.vehicles[vid]['passenger_onboard'] is None
            and self.vehicles[vid]['charging_station'] is None
            and (self._is_ev(vid) or self.vehicles[vid]['target_location'] is None)]
        return vehicles_to_rebalance

    def _active_request_count_at_location(self, location: int) -> int:
        location = int(location)
        return sum(
            1
            for request in self.active_requests.values()
            if int(getattr(request, 'pickup', getattr(request, 'source', -1))) == location
        )

    def _get_available_requests(self):
        assigned = []
        for vid in self.vehicles:
            if self.vehicles[vid]['assigned_request'] is not None:
                assigned.append(self.vehicles[vid]['assigned_request'])
            if self.vehicles[vid]['passenger_onboard'] is not None:
                assigned.append(self.vehicles[vid]['passenger_onboard'])
        return [r for r in self.active_requests.values() if r.request_id not in assigned]

    def _should_log_timing(self, step_index=None):
        if step_index is None:
            step_index = self.current_time
        try:
            step_num = int(step_index)
        except (TypeError, ValueError):
            step_num = 0
        return step_num < 3 or step_num % 50 == 0

    def _combine_rebalancing_profiles(self, profiles, solver_name):
        profiles = [dict(profile) for profile in profiles if profile]
        if not profiles:
            return {}
        combined = dict(profiles[-1])
        sum_fields = [
            'vehicles_to_rebalance',
            'matrix_rows',
            'feasible_request_edges',
            'feasible_charging_edges',
            'feasible_zone_edges',
            'qmatrix_time_sec',
            'qvalue_time_sec',
            'solver_time_sec',
            'solve_total_time_sec',
        ]
        max_fields = [
            'available_requests',
            'action_columns',
            'request_matrix_cols',
            'charging_matrix_cols',
            'zone_matrix_cols',
            'request_columns',
            'charging_columns',
            'zone_columns',
        ]
        for field in sum_fields:
            combined[field] = sum(float(profile.get(field, 0.0) or 0.0) for profile in profiles)
        for field in max_fields:
            combined[field] = max(float(profile.get(field, 0.0) or 0.0) for profile in profiles)
        combined['phase_count'] = float(len(profiles))
        combined['solver_name'] = solver_name
        combined['onlyev'] = 0.0
        qvalue_modes = {profile.get('qvalue_mode') for profile in profiles if profile.get('qvalue_mode')}
        combined['qvalue_mode'] = '+'.join(sorted(qvalue_modes)) if qvalue_modes else combined.get('qvalue_mode')
        return combined

    def _solve_rebalancing(self, vehicles_to_rebalance, available_requests, onlyev=False):
        """Run the appropriate solver and return rebalancing_assignments dict."""
        solve_start = time.time()
        matrix_start = time.time()
        vam, nr, ns, nz = self.generate_whole_matrix(vehicles_to_rebalance,
                                                       rebalance_num=len(vehicles_to_rebalance), onlyev=onlyev)
        matrix_time = time.time() - matrix_start
        qvalue_start = time.time()
        if self.adp_value > 0 and self.value_function is not None:
            bqv = self.generate_vehicle_qvalue(vehicles_to_rebalance, onlyev=onlyev)
            qvalue_mode = 'network'
        else:
            bqv = self.generate_vehicle_qvalue_withoutqnetwork(vehicles_to_rebalance)
            qvalue_mode = 'fallback'
        qvalue_time = time.time() - qvalue_start

        solver_use_mcmf = self.usemcmf
        solver_name = 'mcmf' if solver_use_mcmf else ('gurobi' if self.assignmentgurobi else 'heuristic')
        if solver_use_mcmf and (getattr(self, 'useauction', False) or getattr(self, 'mcmf_solver', None) == 'auction'):
            solver_name = 'auction'
        solver_start = time.time()
        if solver_use_mcmf:
            if onlyev:
                result = self.gurobi_optimizer._np_vehicle_rebalancing_network_ev(
                    vehicles_to_rebalance, available_requests, vam, bqv, iflp=True)
            else:
                result = self.gurobi_optimizer._np_vehicle_rebalancing_network(
                    vehicles_to_rebalance, available_requests, vam, bqv, iflp=True)
        else:
            if self.assignmentgurobi:
                if self.gurobi_network:
                    if onlyev:
                        result = self.gurobi_optimizer._gurobi_vehicle_rebalancing_network_ev(
                            vehicles_to_rebalance, available_requests, vam, bqv, self.gurobi_network_lp)
                    else:
                        result = self.gurobi_optimizer._gurobi_vehicle_rebalancing_network(
                            vehicles_to_rebalance, available_requests, vam, bqv, self.gurobi_network_lp)
                else:
                    if onlyev:
                        result = self.gurobi_optimizer._gurobi_vehicle_rebalancing_ev(
                            vehicles_to_rebalance, available_requests)
                        result = result[0] if isinstance(result, tuple) else result
                    else:
                        result = self.gurobi_optimizer.optimize_vehicle_rebalancing_integrated(vehicles_to_rebalance)
            else:
                charging_stations = [s for s in self.charging_manager.stations.values() if s.available_slots > 0]
                heuristic_action_matrix = vam if getattr(self, 'heuristic_use_scale', True) else None
                if onlyev:
                    if self.adp_value > 0 and self.value_function_ev is not None:
                        result = self.gurobi_optimizer._heuristic_assignment_fastqvalue(
                            vehicles_to_rebalance, charging_stations, vam, bqv)
                    else:
                        ar = list(self.active_requests.values()) if self.active_requests else []
                        result = self.gurobi_optimizer._heuristic_assignment_with_reject(
                            vehicles_to_rebalance, ar, charging_stations, heuristic_action_matrix)
                elif self.adp_value > 0 and self.value_function is not None:
                    result = self.gurobi_optimizer._heuristic_assignment_fastqvalue(
                        vehicles_to_rebalance, charging_stations, vam, bqv)
                else:
                    ar = list(self.active_requests.values()) if self.active_requests else []
                    result = self.gurobi_optimizer._heuristic_assignment_with_reject(
                        vehicles_to_rebalance, ar, charging_stations, heuristic_action_matrix)

        solver_time = time.time() - solver_start
        total_time = time.time() - solve_start
        self.time_stats['gurobi_solve'].append(solver_time)
        if qvalue_mode == 'network':
            self.time_stats['qvalue_with_network'].append(qvalue_time)
        else:
            self.time_stats['qvalue_without_network'].append(qvalue_time)
        self._last_rebalancing_profile = {
            'step': float(self.current_time),
            'vehicles_to_rebalance': float(len(vehicles_to_rebalance)),
            'available_requests': float(len(available_requests)),
            'matrix_rows': float(vam.shape[0]) if hasattr(vam, 'shape') else 0.0,
            'action_columns': float(vam.shape[1]) if hasattr(vam, 'shape') else 0.0,
            'request_matrix_cols': float(nr),
            'charging_matrix_cols': float(ns),
            'zone_matrix_cols': float(nz),
            'wait_matrix_cols': 1.0,
            'feasible_request_edges': float(np.sum(vam[:, :nr])) if hasattr(vam, 'shape') and nr > 0 else 0.0,
            'feasible_charging_edges': float(np.sum(vam[:, nr:nr + ns])) if hasattr(vam, 'shape') and ns > 0 else 0.0,
            'feasible_zone_edges': float(np.sum(vam[:, nr + ns:nr + ns + nz])) if hasattr(vam, 'shape') and nz > 0 else 0.0,
            'request_columns': float(nr),
            'charging_columns': float(ns),
            'zone_columns': float(nz),
            'qmatrix_time_sec': matrix_time,
            'qvalue_time_sec': qvalue_time,
            'solver_time_sec': solver_time,
            'solve_total_time_sec': total_time,
            'solver_name': solver_name,
            'qvalue_mode': qvalue_mode,
            'onlyev': float(bool(onlyev)),
        }
        if self._should_log_timing():
            print(
                f"⏱ Rebalance profile step={int(self.current_time)} solver={solver_name} onlyev={onlyev} "
                f"matrix={vam.shape[0]}x{vam.shape[1]} blocks=req:{nr}/chg:{ns}/zone:{nz}/wait:1 "
                f"vehicles={len(vehicles_to_rebalance)} reqs={len(available_requests)} "
                f"qmatrix={matrix_time:.3f}s qvalue={qvalue_time:.3f}s solve={solver_time:.3f}s total={total_time:.3f}s",
                flush=True,
            )
        return result

    def _update_storeaction_full(self, vehicle_id, action, storeactions_dict, sa_store,
                                  target_loc=None, idle_time_val=None):
        """Full storeaction update mirroring ChargingIntegrated patterns."""
        vehicle_location = self.vehicles[vehicle_id]['location']
        vehicle_battery = self.vehicles[vehicle_id]['battery']
        idle_t = idle_time_val if idle_time_val is not None else self.vehicles[vehicle_id]['idle_timer']
        if target_loc is not None:
            target_zone = self._coerce_target_zone_id(target_loc)
            action.target_location = target_zone if target_zone is not None else target_loc
        if isinstance(action, IdleAction):
            target_zone = self._coerce_location_for_candidate(getattr(action, 'target_location', None), vehicle_location)
            distance = float(self.get_distance_km(vehicle_location, target_zone))
            learning_action_type = getattr(action, 'learning_action_type', None)
            if learning_action_type not in {'idle', 'reloc'}:
                learning_action_type = 'reloc' if distance > 1e-9 else 'idle'
            action.learning_action_type = learning_action_type
            action.target_location = int(target_zone)
            action.post_action_location = int(target_zone)
            action.post_action_distance = distance
            action.post_action_duration = float(self.get_travel_time(vehicle_location, target_zone)) if distance > 0.0 else 0.0
            action.post_action_zoneid = int(self.get_zone_embedding_id(target_zone))
        self._attach_bootstrap_candidates(action, vehicle_id)
        if storeactions_dict.get(vehicle_id) is None:
            storeactions_dict[vehicle_id] = action
            storeactions_dict[vehicle_id].idle_time = idle_t
            if target_loc is not None:
                storeactions_dict[vehicle_id].target_location = getattr(action, 'target_location', target_loc)
            sa_store[vehicle_id] = action
            sa_store[vehicle_id].dur_reward = 0
            sa_store[vehicle_id].current_time = self.current_time
            sa_store[vehicle_id].idle_time = idle_t
            if target_loc is not None:
                sa_store[vehicle_id].target_location = getattr(action, 'target_location', target_loc)
        else:
            old_t = getattr(storeactions_dict[vehicle_id], 'current_time', self.current_time)
            storeactions_dict[vehicle_id].next_action = action
            storeactions_dict[vehicle_id].next_idle_time = idle_t
            storeactions_dict[vehicle_id].vehicle_loc_post = vehicle_location
            storeactions_dict[vehicle_id].vehicle_battery_post = vehicle_battery
            if target_loc is not None:
                storeactions_dict[vehicle_id].next_target_location = getattr(action, 'target_location', target_loc)
            sa_store[vehicle_id] = None
            sa_store[vehicle_id] = action
            sa_store[vehicle_id].dur_reward = 0
            sa_store[vehicle_id].dur_time = self.current_time - old_t
            sa_store[vehicle_id].current_time = self.current_time
            sa_store[vehicle_id].idle_time = idle_t
            if target_loc is not None:
                sa_store[vehicle_id].target_location = getattr(action, 'target_location', target_loc)

    def _bump_assignment_counter(self, amount: int = 1):
        key = self.current_time
        self.assignmentnumber[key] = self.assignmentnumber.get(key, 0) + int(amount)

    def _bump_reject_counter(self, amount: int = 1):
        key = self.current_time
        self.reject_number[key] = self.reject_number.get(key, 0) + int(amount)

    def _process_integrated_assignments(self, rebalancing_assignments, actions,
                                         storeactions, storeactions_ev):
        """Process rebalancing_assignments dict for integrated / AEV-phase mode.
        Handles charge_X, Request, waiting, idle_at_, reloc, None."""
        new_assignments = 0
        charging_assignments = 0
        quest_num_now = len(self.active_requests)

        for vid, target_request in rebalancing_assignments.items():
            vehicle = self.vehicles[vid]
            vehicle_location = vehicle['location']
            vehicle_battery = vehicle['battery']
            vehicle['needs_emergency_charging'] = False
            vehicle['is_stationary'] = False

            if not self._is_ev(vid) and not (
                isinstance(target_request, str) and target_request.startswith("charge_")
            ):
                self._clear_aev_notarrived_reservations(vid)

            if target_request:
                if isinstance(target_request, str) and target_request.startswith("charge_"):
                    station_id = int(target_request.replace("charge_", ""))
                    self._register_aev_notarrived_reservation(vid, station_id)
                    self._move_vehicle_to_charging_station(vid, station_id)
                    charging_assignments += 1
                    actions[vid] = ChargingAction([], station_id, self._charge_duration_for_vehicle(vid),
                                                   vehicle_location, vehicle_battery, req_num=quest_num_now)
                    tgt = vehicle['target_location']
                    self._update_storeaction_full(vid, actions[vid], storeactions, self.storeactions, target_loc=tgt)

                elif isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                    self._bump_assignment_counter()
                    vehicle['is_stationary'] = False
                    offer_vehicle_snapshot = dict(vehicle)
                    offer_reject_sample = self._build_reject_classifier_offer_sample(
                        vid,
                        target_request,
                        vehicle_snapshot=offer_vehicle_snapshot,
                        was_rejected=False,
                    )
                    if self._assign_request_to_vehicle(vid, target_request.request_id):
                        new_assignments += 1
                        vehicle['continual_reject'] = 0
                        vehicle['penalty_timer'] = 0
                        vehicle['idle_target'] = None
                        vehicle['idle_timer'] = 0
                        actions[vid] = ServiceAction([], target_request.request_id,
                                                      vehicle_location, vehicle_battery, req_num=quest_num_now)
                        req_obj = self.active_requests[target_request.request_id]
                        tgt_loc = req_obj.pickup
                        self._annotate_service_action_features(actions[vid], vid, req_obj, self.value_function_ev)
                        actions[vid].rejection_sample = dict(offer_reject_sample)
                        if vehicle['type'] == 1:
                            self._update_storeaction_full(vid, actions[vid], storeactions_ev, self.storeactions_ev, target_loc=tgt_loc)
                        else:
                            self._update_storeaction_full(vid, actions[vid], storeactions, self.storeactions, target_loc=tgt_loc)
                    else:
                        # Rejection
                        vehicle['continual_reject'] += 1
                        vehicle['assigned_request'] = None
                        if vehicle['continual_reject'] >= self.penalty_reject_requestnum:
                            vehicle['penalty_timer'] = self.ev_penalty_duration
                            vehicle['continual_reject'] = 0
                        if self._is_ev(vid):
                            target_zone, _ = self._handle_ev_rejection_relocation(vid)
                            if vehicle['battery'] <= self.min_battery_level + 2 * self.battery_consum:
                                target_zone = vehicle['location']
                            vehicle['idle_target'] = target_zone
                            rej_req_id = target_request.request_id
                            actions[vid] = ServiceAction([], rej_req_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
                            req_obj = self.active_requests[rej_req_id]
                            tgt_loc = req_obj.pickup
                            actions[vid].was_rejected = True
                            self._annotate_service_action_features(actions[vid], vid, req_obj, self.value_function_ev)
                            rejected_offer_sample = dict(offer_reject_sample)
                            rejected_offer_sample['was_rejected'] = True
                            actions[vid].rejection_sample = rejected_offer_sample
                            actions[vid].rejection_reward = self._calculate_rejection_reward(
                                vid,
                                target_request,
                                pickup_location=tgt_loc,
                                vehicle_location=vehicle_location,
                            )
                            self._bump_reject_counter()
                            actions[vid].rejection_reason = 'driver_reject'
                            self._update_storeaction_full(vid, actions[vid], storeactions_ev, self.storeactions_ev, target_loc=tgt_loc)

                elif isinstance(target_request, str) and target_request == "waiting":
                    vehicle['is_stationary'] = True
                    current_coords = vehicle['coordinates']
                    actions[vid] = IdleAction([], current_coords, current_coords,
                                              vehicle_location, vehicle_battery, req_num=quest_num_now)
                    self._update_storeaction_full(vid, actions[vid], storeactions, self.storeactions, target_loc=vehicle_location)

                elif isinstance(target_request, str) and target_request.startswith("idle_at_"):
                    vehicle['is_stationary'] = False
                    zone_idx = int(target_request.replace("idle_at_", ""))
                    hotspot_zone = self.hotspot_locations[zone_idx]  # zone_id (int)
                    idle_target = hotspot_zone
                    vehicle['assigned_request'] = None
                    vehicle['passenger_onboard'] = None
                    vehicle['charging_station'] = None
                    vehicle['target_location'] = idle_target
                    vehicle['idle_target'] = idle_target
                    current_coords = vehicle['coordinates']
                    target_coords = self.zone_coords.get(hotspot_zone, current_coords)
                    actions[vid] = IdleAction([], current_coords, target_coords,
                                              vehicle_location, vehicle_battery, req_num=quest_num_now)
                    self._update_storeaction_full(vid, actions[vid], storeactions, self.storeactions,
                                                  target_loc=vehicle['target_location'])

                elif isinstance(target_request, str) and target_request.startswith("reloc"):
                    vehicle['is_stationary'] = False
                    if self._is_ev(vid):
                        target_zone = self._sample_ev_default_relocation_target(vid)
                        vehicle['idle_target'] = target_zone
                        vehicle['target_location'] = target_zone
                        current_coords = vehicle['coordinates']
                        target_coords = self.zone_coords.get(target_zone, current_coords)
                        actions[vid] = IdleAction([], current_coords, target_coords,
                                                   vehicle_location, vehicle_battery, req_num=quest_num_now)
                        actions[vid].learning_action_type = 'reloc'
                        self._update_storeaction_full(
                            vid,
                            actions[vid],
                            storeactions_ev,
                            self.storeactions_ev,
                            target_loc=target_zone,
                        )

                else:
                    # Unknown target – idle at current location
                    vehicle['is_stationary'] = False
                    current_coords = vehicle['coordinates']
                    target_coords = vehicle.get('idle_target', current_coords)
                    if isinstance(target_coords, int):
                        target_coords = self.zone_coords.get(target_coords, current_coords)
                    actions[vid] = IdleAction([], current_coords, target_coords,
                                              vehicle_location, vehicle_battery, req_num=quest_num_now)
                    if self._is_ev(vid):
                        self._update_storeaction_full(vid, actions[vid], storeactions_ev, self.storeactions_ev,
                                                      target_loc=vehicle.get('idle_target', vehicle_location))
                    else:
                        self._update_storeaction_full(vid, actions[vid], storeactions, self.storeactions,
                                                      target_loc=vehicle.get('idle_target', vehicle_location))
            else:
                # None assignment – idle
                vehicle['is_stationary'] = False
                idle_target = vehicle['location']
                vehicle['target_location'] = idle_target
                vehicle['idle_target'] = idle_target
                current_coords = vehicle['coordinates']
                actions[vid] = IdleAction([], current_coords, current_coords,
                                           vehicle_location, vehicle_battery)

        return new_assignments, charging_assignments

    def _process_ev_only_assignments(self, rebalancing_assignments_ev, actions,
                                      storeactions_ev):
        """Process EV-only assignments (Phase 1/2 of evfirst/aevfirst)."""
        new_assignments = 0
        quest_num_now = len(self.active_requests)

        for vid, target_request in rebalancing_assignments_ev.items():
            vehicle = self.vehicles[vid]
            vehicle_location = vehicle['location']
            vehicle_battery = vehicle['battery']

            if isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                vehicle['is_stationary'] = False
                self._bump_assignment_counter()
                self.ev_offer_count += 1
                self._current_ev_offered_request_ids.add(
                    target_request.request_id
                )
                offer_vehicle_snapshot = dict(vehicle)
                offer_reject_sample = self._build_reject_classifier_offer_sample(
                    vid,
                    target_request,
                    vehicle_snapshot=offer_vehicle_snapshot,
                    was_rejected=False,
                )
                if self._assign_request_to_vehicle(vid, target_request.request_id):
                    new_assignments += 1
                    vehicle['idle_timer'] = 0
                    vehicle['continual_reject'] = 0
                    vehicle['penalty_timer'] = 0
                    vehicle['idle_target'] = None
                    actions[vid] = ServiceAction([], target_request.request_id,
                                                  vehicle_location, vehicle_battery, req_num=quest_num_now)
                    req_obj = self.active_requests[target_request.request_id]
                    tgt_loc = req_obj.pickup
                    self._annotate_service_action_features(actions[vid], vid, req_obj, self.value_function_ev)
                    actions[vid].rejection_sample = dict(offer_reject_sample)
                    self._update_storeaction_full(vid, actions[vid], storeactions_ev, self.storeactions_ev, target_loc=tgt_loc)
                else:
                    vehicle['continual_reject'] += 1
                    vehicle['assigned_request'] = None
                    if vehicle['continual_reject'] >= self.penalty_reject_requestnum:
                        vehicle['penalty_timer'] = self.ev_penalty_duration
                        vehicle['continual_reject'] = 0
                    target_zone, _ = self._handle_ev_rejection_relocation(vid)
                    if vehicle['battery'] <= self.min_battery_level + 2 * self.battery_consum:
                        target_zone = vehicle['location']
                    vehicle['idle_target'] = target_zone
                    rej_req_id = target_request.request_id
                    actions[vid] = ServiceAction([], rej_req_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
                    req_obj = self.active_requests[rej_req_id]
                    tgt_loc = req_obj.pickup
                    actions[vid].was_rejected = True
                    self._annotate_service_action_features(actions[vid], vid, req_obj, self.value_function_ev)
                    rejected_offer_sample = dict(offer_reject_sample)
                    rejected_offer_sample['was_rejected'] = True
                    actions[vid].rejection_sample = rejected_offer_sample
                    actions[vid].rejection_reward = self._calculate_rejection_reward(
                        vid,
                        target_request,
                        pickup_location=tgt_loc,
                        vehicle_location=vehicle_location,
                    )
                    self._bump_reject_counter()
                    actions[vid].rejection_reason = 'driver_reject'
                    self._update_storeaction_full(vid, actions[vid], storeactions_ev, self.storeactions_ev, target_loc=tgt_loc)
            else:
                # No request assigned – relocate
                vehicle['is_stationary'] = False
                target_zone = self._sample_ev_default_relocation_target(vid)
                vehicle['idle_target'] = target_zone
                vehicle['target_location'] = target_zone
                current_coords = vehicle['coordinates']
                target_coords = self.zone_coords.get(target_zone, current_coords)
                actions[vid] = IdleAction([], current_coords, target_coords,
                                           vehicle_location, vehicle_battery, req_num=quest_num_now)
                actions[vid].learning_action_type = 'reloc'
                self._update_storeaction_full(vid, actions[vid], storeactions_ev, self.storeactions_ev,
                                              target_loc=target_zone)

        return new_assignments

    def _attach_stage2_recourse_targets(
        self,
        *,
        actions: dict,
        selected_assignments: dict,
        ev_stage_action_ids: list[int],
    ) -> None:
        """Allocate the joint stage-2 value over selected stage-1 edges.

        The residual critic is an additive edge surrogate.  Giving each of
        the ``n`` selected EV edges ``V2/n`` makes their summed Bellman target
        equal ``sum(R1-G1) + V2`` without counting the recourse value ``n``
        times.
        """
        value_function = getattr(self, 'value_function', None)
        component_fn = getattr(
            value_function,
            'target_components_for_candidate',
            None,
        )
        if not callable(component_fn) or not ev_stage_action_ids:
            return

        other_vehicles = sum(
            vehicle.get('assigned_request') is not None
            or vehicle.get('passenger_onboard') is not None
            for vehicle in self.vehicles.values()
        )
        total_structured_value = 0.0
        total_target_residual1 = 0.0
        total_target_residual2 = 0.0
        selected_edge_count = 0
        for vehicle_id in selected_assignments:
            if self._is_ev(vehicle_id):
                continue
            action = actions.get(vehicle_id)
            if action is None:
                continue
            candidate = self._candidate_from_action(vehicle_id, action)
            if candidate is None:
                continue
            structured, residual1, residual2 = component_fn(
                vehicle_id=int(vehicle_id),
                candidate=candidate,
                current_time=float(self.current_time),
                other_vehicles=float(other_vehicles),
                num_requests=float(len(self.active_requests)),
            )
            total_structured_value += float(structured)
            total_target_residual1 += float(residual1)
            total_target_residual2 += float(residual2)
            selected_edge_count += 1

        if selected_edge_count == 0:
            total_stage2_value = 0.0
        else:
            # Clip the two joint additive residual sums, not each edge
            # independently.  This is G2 + min(Delta2_1, Delta2_2).
            total_stage2_value = total_structured_value + min(
                total_target_residual1,
                total_target_residual2,
            )
        value_share = total_stage2_value / float(len(ev_stage_action_ids))
        target_map = getattr(self, '_stage1_recourse_target_by_transition', None)
        if target_map is None:
            target_map = {}
            self._stage1_recourse_target_by_transition = target_map
        for vehicle_id in ev_stage_action_ids:
            target_map[(int(vehicle_id), float(self.current_time))] = value_share

    def _build_fallback_actions(self, actions):
        """Generate actions for vehicles not yet assigned (same as ChargingIntegrated)."""
        for vid, v in self.vehicles.items():
            if vid not in actions:
                vloc = v['location']
                vbat = v['battery']
                if not v.get('is_online', True):
                    actions[vid] = IdleAction([], v['coordinates'], v['coordinates'], vloc, vbat)
                elif v.get('is_stationary', False):
                    actions[vid] = IdleAction([], v['coordinates'], v['coordinates'], vloc, vbat)
                elif v['charging_station'] is not None:
                    actions[vid] = ChargingAction([], v['charging_station'], v.get('charging_time_left', self.charge_duration), vloc, vbat)
                elif v['assigned_request'] is not None:
                    actions[vid] = ServiceAction([], v['assigned_request'], vloc, vbat)
                elif v['passenger_onboard'] is not None:
                    actions[vid] = ServiceAction([], v['passenger_onboard'], vloc, vbat)
                elif v.get('charging_target') is not None:
                    actions[vid] = ChargingAction([], v['charging_target'], self._charge_duration_for_vehicle(vid), vloc, vbat)
                elif v.get('target_location') is not None:
                    tgt = v['target_location']
                    tc = self.zone_coords.get(tgt, v['coordinates']) if isinstance(tgt, int) else tgt
                    actions[vid] = IdleAction([], v['coordinates'], tc, vloc, vbat)
                else:
                    actions[vid] = IdleAction([], v['coordinates'], v['coordinates'], vloc, vbat)

        if len(actions) != len(self.vehicles):
            missing = [vid for vid in self.vehicles if vid not in actions]
            raise RuntimeError(f"Action generation failed at step {self.current_time} – {len(missing)} missing")

    def _ev_charging_phase(self, actions, storeactions_ev):
        """Pre-assignment EV charging probability decision (same as ChargingIntegrated)."""
        for vid, v in self.vehicles.items():
            if (v.get('is_online', True) and self._is_ev(vid) and v['charging_station'] is None and v['assigned_request'] is None
                    and v['passenger_onboard'] is None and v['idle_target'] is None and v['target_location'] is None):
                p_charge, station_probs = self.compute_ev_charge_probability(vid)
                station_probs = self._reachable_charging_station_probs(vid, station_probs)
                must_charge = float(v.get('battery', 1.0)) <= float(
                    getattr(self, 'must_charge_battery_threshold', self.rebalance_battery_threshold)
                )
                if must_charge and not station_probs:
                    v['needs_emergency_charging'] = True
                    continue
                if station_probs and (must_charge or random.random() < p_charge):
                    r = random.random()
                    acc = 0.0
                    chosen_station = next(iter(station_probs.keys())) if station_probs else None
                    if chosen_station is None:
                        v['needs_emergency_charging'] = True
                        continue
                    for sid, prob in station_probs.items():
                        acc += float(prob)
                        if r <= acc:
                            chosen_station = int(sid)
                            break
                    vloc = v['location']
                    vbat = v['battery']
                    self._move_vehicle_to_charging_station(vid, chosen_station)
                    actions[vid] = ChargingAction([], chosen_station, self._charge_duration_for_vehicle(vid), vloc, vbat)
                    self._update_storeaction(vid, actions[vid], storeactions_ev, is_ev=True)
                else:
                    v['no_charge_cooldown_until'] = self.current_time + 5

    # ------------------------------------------------------------------
    # simulate_motion  (integrated mode)
    # ------------------------------------------------------------------

    def simulate_motion(self, agents=None, current_requests=None, rebalance=True):
        if agents is None:
            agents = []
        simulate_start = time.time()
        actions = {}
        self._bayes_step_contexts = {}
        self.decision_mode = "integrated"
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles}

        charging_phase_start = time.time()
        self._ev_charging_phase(actions, storeactions_ev)
        charging_phase_time = time.time() - charging_phase_start
        leftover = [vid for vid in self.vehicles if vid not in actions]
        rebalancing_time = 0.0

        if rebalance and leftover:
            vehicles_to_rebalance = self._build_vehicles_to_rebalance(leftover)
            aev_to_rebalance = [vid for vid in vehicles_to_rebalance if not self._is_ev(vid)]
            aev_to_rebalance_num = len(aev_to_rebalance)
            
            if self.current_time % 50 == 0:
                print(f"🔄 Rebalancing Step {self.current_time}: {len(vehicles_to_rebalance)} vehicles", flush=True)

            if len(vehicles_to_rebalance) > 0:
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)

                available_requests = self._get_available_requests()
                rebalancing_start = time.time()
                rebalancing_assignments = self._solve_rebalancing(vehicles_to_rebalance, available_requests)
                rebalancing_time = time.time() - rebalancing_start
                self.total_rebalancing_calls += 1

                new_a, ch_a = self._process_integrated_assignments(
                    rebalancing_assignments, actions, storeactions, storeactions_ev)
                self.rebalancing_assignments_per_step.append(new_a)
                self.rebalancing_whole.append(len(rebalancing_assignments))

        fallback_start = time.time()
        self._build_fallback_actions(actions)
        fallback_time = time.time() - fallback_start
        if current_requests:
            self.update_recent_requests(current_requests)
        total_time = time.time() - simulate_start
        self._last_simulation_profile = {
            'step': float(self.current_time),
            'charging_phase_time_sec': charging_phase_time,
            'rebalancing_time_sec': rebalancing_time,
            'fallback_time_sec': fallback_time,
            'total_time_sec': total_time,
            'leftover_vehicles': float(len(leftover)),
        }
        if self._should_log_timing():
            rebalance_desc = self._last_rebalancing_profile.get('solver_name', 'n/a') if self._last_rebalancing_profile else 'n/a'
            print(
                f"⏱ simulate_motion step={int(self.current_time)} total={total_time:.3f}s charging={charging_phase_time:.3f}s "
                f"rebalance={rebalancing_time:.3f}s fallback={fallback_time:.3f}s solver={rebalance_desc}",
                flush=True,
            )
        return actions, storeactions, storeactions_ev

    # ------------------------------------------------------------------
    # simulate_motion_evfirst  (EV first, then AEV with prior features)
    # ------------------------------------------------------------------

    def simulate_motion_evfirst(self, agents=None, current_requests=None, rebalance=True):
        if agents is None:
            agents = []
        simulate_start = time.time()
        actions = {}
        self._prior_features_for_posterior = None
        self._bayes_step_contexts = {}
        self.decision_mode = "ev_first"
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles}

        charging_phase_start = time.time()
        self._ev_charging_phase(actions, storeactions_ev)
        charging_phase_time = time.time() - charging_phase_start
        leftover = [vid for vid in self.vehicles if vid not in actions]
        rebalancing_time = 0.0

        if rebalance and leftover:
            vehicles_to_rebalance = self._build_vehicles_to_rebalance(leftover)
            if self.current_time % 50 == 0:
                print(f"🔄 Rebalancing Step {self.current_time}: {len(vehicles_to_rebalance)} vehicles")

            if len(vehicles_to_rebalance) > 0:
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)

                new_assignments = 0
                charging_assignments = 0
                self.total_rebalancing_calls += 1
                re_assignments_len = len(vehicles_to_rebalance)

                # --- Phase 1: EV only ---
                rebalancing_start = time.time()
                vehicles_ev = [vid for vid in vehicles_to_rebalance if self._is_ev(vid)]
                self.ev_eligible_decision_count += len(vehicles_ev)
                vehicles_aev = [vid for vid in vehicles_to_rebalance if vid not in vehicles_ev]
                all_ev_ids = [
                    vid for vid, vehicle in self.vehicles.items()
                    if self._is_ev(vid) and vehicle.get('is_online', True)
                ]
                all_aev_ids = [
                    vid for vid, vehicle in self.vehicles.items()
                    if not self._is_ev(vid) and vehicle.get('is_online', True)
                ]
                ev_state_dist = self._vehicle_zone_distribution(all_ev_ids)
                aev_state_dist = self._vehicle_zone_distribution(all_aev_ids)
                self._set_bayes_context(
                    role='leader', leader_is_ev=True, state_dist=ev_state_dist,
                    peer_dist=aev_state_dist, prior_features=None,
                )
                available_requests = self._get_available_requests()
                self._current_ev_stage_request_ids = {
                    request.request_id for request in available_requests
                }
                self._current_ev_offered_request_ids = set()
                rebalancing_ev = self._solve_rebalancing(vehicles_ev, available_requests, onlyev=True)
                ev_profile = dict(self._last_rebalancing_profile)
                ev_new = self._process_ev_only_assignments(rebalancing_ev, actions, storeactions_ev)
                new_assignments += ev_new

                # Build prior features from EV leader for AEV follower
                ev_action_ids = [vid for vid in actions if self._is_ev(vid)]
                ev_prior_features = self._build_prior_features(ev_action_ids, actions)
                ev_target_dist = self._zone_distribution_from_actions(ev_action_ids, actions)

                # --- Phase 2: AEV ---
                self._set_bayes_context(
                    role='follower', leader_is_ev=True, state_dist=aev_state_dist,
                    peer_dist=ev_target_dist, prior_features=ev_prior_features,
                )
                available_requests = self._get_available_requests()
                residual_ids = {
                    request.request_id for request in available_requests
                }
                rejected_residual_ids = residual_ids & self.ev_rejected_request_ids
                unoffered_residual_ids = (
                    residual_ids
                    & self._current_ev_stage_request_ids
                    - self._current_ev_offered_request_ids
                )
                self.residual_request_count += len(residual_ids)
                self.unoffered_request_count += len(unoffered_residual_ids)

                recourse_variant = getattr(self, 'recourse_variant', 'legacy')
                if recourse_variant == 'r1':
                    # R1 keeps rejected requests active for future epochs but
                    # removes them only from the current AEV recourse graph.
                    available_requests = [
                        request
                        for request in available_requests
                        if request.request_id not in rejected_residual_ids
                    ]

                force_structured_only = recourse_variant == 'r2'
                old_force_structured = None
                if self.value_function is not None:
                    old_force_structured = getattr(
                        self.value_function,
                        'force_structured_only',
                        False,
                    )
                    self.value_function.force_structured_only = force_structured_only
                try:
                    self._same_epoch_blocked_request_ids = (
                        set(rejected_residual_ids)
                        if recourse_variant == 'r1'
                        else set()
                    )
                    rebalancing_aev = self._solve_rebalancing(
                        vehicles_aev,
                        available_requests,
                    )
                finally:
                    self._same_epoch_blocked_request_ids = set()
                    if self.value_function is not None:
                        self.value_function.force_structured_only = old_force_structured
                aev_profile = dict(self._last_rebalancing_profile)
                aev_new, aev_ch = self._process_integrated_assignments(
                    rebalancing_aev, actions, storeactions, storeactions_ev)
                selected_aev_request_ids = {
                    target.request_id
                    for vehicle_id, target in rebalancing_aev.items()
                    if not self._is_ev(vehicle_id) and isinstance(target, Request)
                }
                served_rejected = selected_aev_request_ids & rejected_residual_ids
                served_unoffered = selected_aev_request_ids & unoffered_residual_ids
                self.aev_request_assignment_count += len(selected_aev_request_ids)
                self.aev_recourse_assignment_count += len(served_rejected)
                self.residual_request_served_count += len(
                    selected_aev_request_ids & residual_ids
                )
                self.rejected_request_served_count += len(served_rejected)
                self.unoffered_request_served_count += len(served_unoffered)
                if recourse_variant == 'r4':
                    self._attach_stage2_recourse_targets(
                        actions=actions,
                        selected_assignments=rebalancing_aev,
                        ev_stage_action_ids=list(rebalancing_ev),
                    )
                new_assignments += aev_new
                charging_assignments += aev_ch
                aev_action_ids = [vid for vid in actions if not self._is_ev(vid)]
                aev_target_dist = self._zone_distribution_from_actions(aev_action_ids, actions)
                self._bayes_step_contexts = {
                    'ev': {
                        'role': 'leader', 'leader_is_ev': True,
                        'state_dist': ev_state_dist, 'peer_dist': aev_state_dist,
                        'target_dist': ev_target_dist, 'prior_features': None,
                        'skip_training': False,
                    },
                    'aev': {
                        'role': 'follower', 'leader_is_ev': True,
                        'state_dist': aev_state_dist, 'peer_dist': ev_target_dist,
                        'target_dist': aev_target_dist, 'prior_features': ev_prior_features,
                        'skip_training': False,
                    },
                }
                rebalancing_time = time.time() - rebalancing_start
                self._last_rebalancing_profile = self._combine_rebalancing_profiles(
                    [ev_profile, aev_profile],
                    solver_name='evfirst_two_stage',
                )

                self.rebalancing_assignments_per_step.append(new_assignments)
                self.rebalancing_whole.append(re_assignments_len)

        fallback_start = time.time()
        self._build_fallback_actions(actions)
        fallback_time = time.time() - fallback_start
        if current_requests:
            self.update_recent_requests(current_requests)
        total_time = time.time() - simulate_start
        self._last_simulation_profile = {
            'step': float(self.current_time),
            'charging_phase_time_sec': charging_phase_time,
            'rebalancing_time_sec': rebalancing_time,
            'fallback_time_sec': fallback_time,
            'total_time_sec': total_time,
            'leftover_vehicles': float(len(leftover)),
        }
        if self._should_log_timing():
            rebalance_desc = self._last_rebalancing_profile.get('solver_name', 'n/a') if self._last_rebalancing_profile else 'n/a'
            print(
                f"⏱ simulate_motion_evfirst step={int(self.current_time)} total={total_time:.3f}s "
                f"charging={charging_phase_time:.3f}s rebalance={rebalancing_time:.3f}s "
                f"fallback={fallback_time:.3f}s solver={rebalance_desc}",
                flush=True,
            )
        return actions, storeactions, storeactions_ev

    # ------------------------------------------------------------------
    # simulate_motion_aevfirst  (AEV first, then EV with prior features)
    # ------------------------------------------------------------------

    def simulate_motion_aevfirst(self, agents=None, current_requests=None, rebalance=True):
        if agents is None:
            agents = []
        simulate_start = time.time()
        actions = {}
        self._prior_features_for_posterior = None
        self._bayes_step_contexts = {}
        self.decision_mode = "aev_first"
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles}

        charging_phase_start = time.time()
        self._ev_charging_phase(actions, storeactions_ev)
        charging_phase_time = time.time() - charging_phase_start
        leftover = [vid for vid in self.vehicles if vid not in actions]
        rebalancing_time = 0.0

        if rebalance and leftover:
            vehicles_to_rebalance = self._build_vehicles_to_rebalance(leftover)
            if self.current_time % 50 == 0:
                print(f"🔄 Rebalancing Step {self.current_time}: {len(vehicles_to_rebalance)} vehicles")

            if len(vehicles_to_rebalance) > 0:
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)

                new_assignments = 0
                charging_assignments = 0
                self.total_rebalancing_calls += 1
                re_assignments_len = len(vehicles_to_rebalance)

                # --- Phase 1: AEV only ---
                rebalancing_start = time.time()
                vehicles_aev = [vid for vid in vehicles_to_rebalance if not self._is_ev(vid)]
                vehicles_ev = [vid for vid in vehicles_to_rebalance if self._is_ev(vid)]
                all_aev_ids = [
                    vid for vid, vehicle in self.vehicles.items()
                    if not self._is_ev(vid) and vehicle.get('is_online', True)
                ]
                all_ev_ids = [
                    vid for vid, vehicle in self.vehicles.items()
                    if self._is_ev(vid) and vehicle.get('is_online', True)
                ]
                aev_state_dist = self._vehicle_zone_distribution(all_aev_ids)
                ev_state_dist = self._vehicle_zone_distribution(all_ev_ids)
                self._set_bayes_context(
                    role='leader', leader_is_ev=False, state_dist=aev_state_dist,
                    peer_dist=ev_state_dist, prior_features=None,
                )
                available_requests = self._get_available_requests()
                rebalancing_aev = self._solve_rebalancing(vehicles_aev, available_requests)
                aev_profile = dict(self._last_rebalancing_profile)
                aev_new, aev_ch = self._process_integrated_assignments(
                    rebalancing_aev, actions, storeactions, storeactions_ev)
                new_assignments += aev_new
                charging_assignments += aev_ch

                # Build prior features from AEV leader for EV follower
                aev_action_ids = [vid for vid in actions if not self._is_ev(vid)]
                aev_prior_features = self._build_prior_features(aev_action_ids, actions)
                aev_target_dist = self._zone_distribution_from_actions(aev_action_ids, actions)

                # --- Phase 2: EV only ---
                self._set_bayes_context(
                    role='follower', leader_is_ev=False, state_dist=ev_state_dist,
                    peer_dist=aev_target_dist, prior_features=aev_prior_features,
                )
                available_requests = self._get_available_requests()
                rebalancing_ev = self._solve_rebalancing(vehicles_ev, available_requests, onlyev=True)
                ev_profile = dict(self._last_rebalancing_profile)
                ev_new = self._process_ev_only_assignments(rebalancing_ev, actions, storeactions_ev)
                new_assignments += ev_new
                ev_action_ids = [vid for vid in actions if self._is_ev(vid)]
                ev_target_dist = self._zone_distribution_from_actions(ev_action_ids, actions)
                self._bayes_step_contexts = {
                    'aev': {
                        'role': 'leader', 'leader_is_ev': False,
                        'state_dist': aev_state_dist, 'peer_dist': ev_state_dist,
                        'target_dist': aev_target_dist, 'prior_features': None,
                        'skip_training': True,
                    },
                    'ev': {
                        'role': 'follower', 'leader_is_ev': False,
                        'state_dist': ev_state_dist, 'peer_dist': aev_target_dist,
                        'target_dist': ev_target_dist, 'prior_features': aev_prior_features,
                        'skip_training': False,
                    },
                }
                rebalancing_time = time.time() - rebalancing_start
                self._last_rebalancing_profile = self._combine_rebalancing_profiles(
                    [aev_profile, ev_profile],
                    solver_name='aevfirst_two_stage',
                )

                self.rebalancing_assignments_per_step.append(new_assignments)
                self.rebalancing_whole.append(re_assignments_len)

        fallback_start = time.time()
        self._build_fallback_actions(actions)
        fallback_time = time.time() - fallback_start
        if current_requests:
            self.update_recent_requests(current_requests)
        total_time = time.time() - simulate_start
        self._last_simulation_profile = {
            'step': float(self.current_time),
            'charging_phase_time_sec': charging_phase_time,
            'rebalancing_time_sec': rebalancing_time,
            'fallback_time_sec': fallback_time,
            'total_time_sec': total_time,
            'leftover_vehicles': float(len(leftover)),
        }
        if self._should_log_timing():
            rebalance_desc = self._last_rebalancing_profile.get('solver_name', 'n/a') if self._last_rebalancing_profile else 'n/a'
            print(
                f"⏱ simulate_motion_aevfirst step={int(self.current_time)} total={total_time:.3f}s "
                f"charging={charging_phase_time:.3f}s rebalance={rebalancing_time:.3f}s "
                f"fallback={fallback_time:.3f}s solver={rebalance_desc}",
                flush=True,
            )
        return actions, storeactions, storeactions_ev

    def update_recent_requests(self, requests):
        """Compat stub for request tracking."""
        pass

    # ==================================================================
    # Matrix generators (for Gurobi / MinCostMaxFlowGPU)
    # ==================================================================

    def generate_vehicle_requests(self, vehicle_ids):
        assigned = set()
        for vid, v in self.vehicles.items():
            if v['assigned_request'] is not None:
                assigned.add(v['assigned_request'])
            if v['passenger_onboard'] is not None:
                assigned.add(v['passenger_onboard'])
        blocked = set(getattr(self, '_same_epoch_blocked_request_ids', set()))
        avail = [
            r
            for r in self.active_requests.values()
            if r.request_id not in assigned and r.request_id not in blocked
        ]
        self._last_matrix_request_ids = [r.request_id for r in avail]
        mat = np.zeros((len(vehicle_ids), len(avail)), dtype=np.float32)
        if not vehicle_ids or not avail:
            return mat
        use_range = getattr(self, 'use_range_requests', False)
        range_radius = getattr(self, 'assignmentrange', 5.0)
        vehicle_locations = np.array([self.vehicles[vid]['location'] for vid in vehicle_ids], dtype=np.int32)
        vehicle_battery = np.array([self.vehicles[vid]['battery'] for vid in vehicle_ids], dtype=np.float32)
        pickup_ids = np.array([req.pickup for req in avail], dtype=np.int32)
        dropoff_ids = np.array([req.dropoff for req in avail], dtype=np.int32)
        pickup_dists = self.distance_matrix[vehicle_locations[:, None], pickup_ids[None, :]]
        trip_dists = np.asarray([
            self._request_trip_distance_km(request) for request in avail
        ], dtype=np.float32)
        total_dists = pickup_dists + trip_dists[None, :]
        if dropoff_ids.size > 0 and np.max(dropoff_ids) < self.nearest_charging_distance.shape[0]:
            reserve = np.maximum(self.min_battery_level, self.nearest_charging_distance[dropoff_ids] * self.battery_consum + 0.01)
        else:
            reserve = np.array([self._post_action_battery_reserve(req.dropoff) for req in avail], dtype=np.float32)
        feasible = vehicle_battery[:, None] - total_dists * self.battery_consum >= reserve[None, :]
        if use_range:
            feasible &= pickup_dists <= range_radius
        request_top_k = getattr(self, 'request_top_k', None)
        if request_top_k is not None and request_top_k > 0 and feasible.shape[1] > request_top_k:
            for row_idx in range(feasible.shape[0]):
                row_mask = feasible[row_idx]
                feasible_idx = np.flatnonzero(row_mask)
                if feasible_idx.size > request_top_k:
                    keep_local = feasible_idx[np.argpartition(pickup_dists[row_idx, feasible_idx], request_top_k - 1)[:request_top_k]]
                    row_mask[:] = False
                    row_mask[keep_local] = True
        mat[feasible] = 1.0
        return mat



    def generate_vehicle_zone(self, vehicle_ids, distance_threshold=None):
        zones = list(self.relocation_target_ids) if self.relocation_target_ids else sorted(self.zone_to_locs.keys())
        mat = np.zeros((len(vehicle_ids), len(zones)), dtype=np.float32)
        if not vehicle_ids or not zones:
            return mat
        if distance_threshold is None:
            distance_threshold = getattr(self, 'zone_action_range_km', None)
        vehicle_locations = np.array([self.vehicles[vid]['location'] for vid in vehicle_ids], dtype=np.int32)
        vehicle_battery = np.array([self.vehicles[vid]['battery'] for vid in vehicle_ids], dtype=np.float32)
        zone_ids = np.array(zones, dtype=np.int32)
        zone_dists = self.distance_matrix[vehicle_locations[:, None], zone_ids[None, :]]
        if np.max(zone_ids) < self.nearest_charging_distance.shape[0]:
            reserve = np.maximum(self.min_battery_level, self.nearest_charging_distance[zone_ids] * self.battery_consum + 0.01)
        else:
            reserve = np.array([self._post_action_battery_reserve(zone_id) for zone_id in zones], dtype=np.float32)
        feasible = vehicle_battery[:, None] - zone_dists * self.battery_consum >= reserve[None, :]
        if distance_threshold is not None and distance_threshold > 0:
            feasible &= zone_dists <= float(distance_threshold)
        zone_top_k = getattr(self, 'zone_top_k', None)
        if zone_top_k is not None and zone_top_k > 0 and feasible.shape[1] > zone_top_k:
            for row_idx in range(feasible.shape[0]):
                row_mask = feasible[row_idx]
                feasible_idx = np.flatnonzero(row_mask)
                if feasible_idx.size > zone_top_k:
                    keep_local = feasible_idx[np.argpartition(zone_dists[row_idx, feasible_idx], zone_top_k - 1)[:zone_top_k]]
                    row_mask[:] = False
                    row_mask[keep_local] = True
        mat[feasible] = 1.0
        return mat

    def generate_vehicle_chargerange(self, vehicle_ids):
        stations = sorted(self.charging_manager.stations.keys())
        mat = np.zeros((len(vehicle_ids), len(stations)), dtype=np.float32)
        if not vehicle_ids or not stations:
            return mat
        vehicle_locations = np.array([self.vehicles[vid]['location'] for vid in vehicle_ids], dtype=np.int32)
        vehicle_battery = np.array([self.vehicles[vid]['battery'] for vid in vehicle_ids], dtype=np.float32)
        station_zone_ids = self.station_zone_ids
        charge_dists = self.distance_matrix[vehicle_locations[:, None], station_zone_ids[None, :]]
        feasible = charge_dists * self.battery_consum <= np.maximum(0.0, vehicle_battery[:, None] - 0.01)
        charge_range_km = getattr(self, 'charge_action_range_km', None)
        if charge_range_km is not None and charge_range_km > 0:
            feasible &= charge_dists <= float(charge_range_km)
        mat[feasible] = 1.0
        for col_idx, station_id in enumerate(stations):
            station = self.charging_manager.stations[station_id]
            total_reserved = len(station.current_vehicles) + len(station.charging_queue_notarrived)
            if total_reserved >= station.max_capacity:
                for row_idx, vehicle_id in enumerate(vehicle_ids):
                    if not self._is_ev(vehicle_id):
                        mat[row_idx, col_idx] = 0.0
        charge_top_k = getattr(self, 'charge_top_k', None)
        if charge_top_k is not None and charge_top_k > 0 and mat.shape[1] > charge_top_k:
            for row_idx in range(mat.shape[0]):
                feasible_idx = np.flatnonzero(mat[row_idx] > 0)
                if feasible_idx.size > charge_top_k:
                    keep_local = feasible_idx[np.argpartition(charge_dists[row_idx, feasible_idx], charge_top_k - 1)[:charge_top_k]]
                    mat[row_idx, :] = 0.0
                    mat[row_idx, keep_local] = 1.0
        return mat

    def generate_vehicle_wait(self, vehicle_ids, rebalance_num=0):

        carindex = self.findchargerange_c(rebalance_num)
        vehicle_wait = np.zeros((len(vehicle_ids), 1))
        
        for i, vehicle_id in enumerate(vehicle_ids):
            if self.vehicles[vehicle_id]['type'] == 1:
                vehicle_wait[i][0] = 1  
            else:
                if carindex[vehicle_id] <= 0:
                    vehicle_wait[i][0] = 1  # 附近没有充电容量，可以等待
                else:
                    vehicle_wait[i][0] = 0  # 附近有充电容量，不应该等待
        return vehicle_wait

    def _active_gat_neighbour_number(self) -> int:
        neighbour_numbers = []
        for value_function in (getattr(self, 'value_function', None), getattr(self, 'value_function_ev', None)):
            graph_encoder = getattr(value_function, 'graph_encoder', None)
            if graph_encoder is not None and hasattr(graph_encoder, 'neighbour_number'):
                neighbour_numbers.append(max(0, int(graph_encoder.neighbour_number)))
        return max(neighbour_numbers, default=0)

    def _cache_vehicle_action_graph_neighbours(
        self,
        vehicle_ids,
        vehicle_action_matrix,
        num_requests: int,
        num_stations: int,
        num_zones: int,
    ) -> None:
        """Cache each vehicle's nearest unique graph nodes from its feasible action row."""
        neighbour_number = self._active_gat_neighbour_number()
        current_step = int(getattr(self, 'current_time', 0) or 0)
        signature = (
            current_step,
            neighbour_number,
            tuple(int(vehicle_id) for vehicle_id in vehicle_ids),
            tuple(getattr(self, '_last_matrix_request_ids', [])),
            tuple(getattr(self, '_last_matrix_charge_station_ids', [])),
            tuple(getattr(self, '_last_matrix_zone_target_ids', [])),
        )
        if signature == getattr(self, '_last_vehicle_action_graph_neighbour_signature', None):
            return
        self._last_vehicle_action_graph_neighbour_signature = signature
        self._last_vehicle_action_graph_neighbours = {}
        self._last_vehicle_action_graph_neighbour_step = current_step
        if neighbour_number <= 0 or len(vehicle_ids) == 0:
            return

        request_ids = list(getattr(self, '_last_matrix_request_ids', []))[:num_requests]
        request_by_id = {request.request_id: request for request in self.active_requests.values()}
        request_pickups = np.full(num_requests, -1, dtype=np.int32)
        for column, request_id in enumerate(request_ids):
            if request_id in request_by_id:
                request_pickups[column] = int(request_by_id[request_id].pickup)
        station_ids = list(getattr(self, '_last_matrix_charge_station_ids', []))[:num_stations]
        station_locations = np.asarray([
            int(self.charging_manager.stations[station_id].location)
            for station_id in station_ids
        ], dtype=np.int32)
        zone_locations = np.asarray(
            list(getattr(self, '_last_matrix_zone_target_ids', []))[:num_zones],
            dtype=np.int32,
        )

        for row_index, vehicle_id in enumerate(vehicle_ids):
            vehicle_location = int(self.vehicles[vehicle_id]['location'])
            raw_candidates = []

            request_columns = np.flatnonzero(vehicle_action_matrix[row_index, :num_requests] > 0)
            valid_request_columns = request_columns[request_pickups[request_columns] >= 0] if request_columns.size else request_columns
            if valid_request_columns.size:
                distances = self.distance_matrix[vehicle_location, request_pickups[valid_request_columns]]
                raw_candidates.extend(
                    (
                        float(distance),
                        'zone',
                        int(request_pickups[column]),
                        int(request_pickups[column]),
                    )
                    for column, distance in zip(valid_request_columns, distances)
                )

            station_start = num_requests
            station_columns = np.flatnonzero(
                vehicle_action_matrix[row_index, station_start:station_start + num_stations] > 0
            )
            if station_columns.size:
                distances = self.distance_matrix[vehicle_location, station_locations[station_columns]]
                raw_candidates.extend(
                    (
                        float(distance),
                        'station',
                        int(station_ids[column]),
                        int(station_locations[column]),
                    )
                    for column, distance in zip(station_columns, distances)
                )

            zone_start = num_requests + num_stations
            zone_columns = np.flatnonzero(
                vehicle_action_matrix[row_index, zone_start:zone_start + num_zones] > 0
            )
            if zone_columns.size:
                distances = self.distance_matrix[vehicle_location, zone_locations[zone_columns]]
                raw_candidates.extend(
                    (
                        float(distance),
                        'zone',
                        int(zone_locations[column]),
                        int(zone_locations[column]),
                    )
                    for column, distance in zip(zone_columns, distances)
                )

            neighbours = []
            seen_nodes = set()
            for distance, node_type, node_id, target_location in sorted(raw_candidates, key=lambda item: item[0]):
                node_key = (node_type, node_id)
                if node_key in seen_nodes:
                    continue
                seen_nodes.add(node_key)
                neighbours.append({
                    'node_type': node_type,
                    'node_id': node_id,
                    'target_location': target_location,
                    'distance': distance,
                })
                if len(neighbours) >= neighbour_number:
                    break
            self._last_vehicle_action_graph_neighbours[int(vehicle_id)] = neighbours


    def generate_whole_matrix(self, vehicle_ids, rebalance_num=0, onlyev=False):
        req_mat = self.generate_vehicle_requests(vehicle_ids)
        ev_rows = np.array([self._is_ev(vehicle_id) for vehicle_id in vehicle_ids], dtype=bool)
        zone_target_ids = list(self.relocation_target_ids) if self.relocation_target_ids else sorted(self.zone_to_locs.keys())
        zone_indices = list(range(len(zone_target_ids)))
        zone_mat = self.generate_vehicle_zone(vehicle_ids)
        if zone_mat.size > 0 and np.any(ev_rows):
            zone_mat[ev_rows, :] = 0.0
        if zone_mat.size > 0:
            keep_zone_cols = np.any(zone_mat > 0, axis=0)
            zone_mat = zone_mat[:, keep_zone_cols]
            zone_indices = [idx for idx, keep in zip(zone_indices, keep_zone_cols) if keep]
            zone_target_ids = [zone_id for zone_id, keep in zip(zone_target_ids, keep_zone_cols) if keep]
        charge_mat = self.generate_vehicle_chargerange(vehicle_ids)
        if charge_mat.size > 0 and np.any(ev_rows):
            charge_mat[ev_rows, :] = 0.0
        if charge_mat.size > 0 and zone_mat.size > 0:
            vehicle_battery = np.asarray([self.vehicles[vid]['battery'] for vid in vehicle_ids], dtype=np.float32)
            no_reloc_rows = (
                (~ev_rows)
                & (vehicle_battery <= float(getattr(self, 'no_reloc_battery_threshold', 0.15)))
            )
            if np.any(no_reloc_rows):
                zone_mat[no_reloc_rows, :] = 0.0
        charge_station_ids = sorted(self.charging_manager.stations.keys())
        if charge_mat.size > 0:
            keep_charge_cols = np.any(charge_mat > 0, axis=0)
            charge_mat = charge_mat[:, keep_charge_cols]
            charge_station_ids = [sid for sid, keep in zip(charge_station_ids, keep_charge_cols) if keep]
        self._last_matrix_charge_station_ids = charge_station_ids
        self._last_matrix_zone_indices = zone_indices
        self._last_matrix_zone_target_ids = zone_target_ids
        self._last_matrix_num_requests = req_mat.shape[1]
        self._last_matrix_num_stations = charge_mat.shape[1]
        self._last_matrix_num_zones = zone_mat.shape[1]
        wait_mat = self.generate_vehicle_wait(vehicle_ids, rebalance_num)
        total = np.hstack([req_mat, charge_mat, zone_mat, wait_mat])
        self._cache_vehicle_action_graph_neighbours(
            vehicle_ids,
            total,
            req_mat.shape[1],
            charge_mat.shape[1],
            zone_mat.shape[1],
        )
        return total, req_mat.shape[1], charge_mat.shape[1], zone_mat.shape[1]


    def filter_aev_experiences_for_aev_value_function(self, experiences):
        """Filter AEV experiences from AEV value function training data (same as ChargingIntegrated)."""
        valuefunction = getattr(self, 'value_function', None)
        if not experiences or valuefunction is None or not hasattr(valuefunction, 'experience_buffer'):
            return experiences

        experience_buffer = valuefunction.experience_buffer
        buffer_size = len(experience_buffer)
        buffer_capacity = getattr(experience_buffer, 'maxlen', None) or max(buffer_size, 1)
        training_step = int(getattr(valuefunction, 'training_step', 0) or 0)

        min_buffer_before_filter = min(buffer_capacity, 2048)
        if buffer_size < min_buffer_before_filter:
            return experiences

        def action_bucket(action_type: str) -> str:
            action_type = str(action_type)
            if action_type == 'reloc' or action_type.startswith('reloc'):
                return 'reloc'
            if action_type == 'idle':
                return 'idle'
            if action_type.startswith('charge'):
                return 'charge'
            if action_type.startswith('assign'):
                return 'assign'
            return 'other'

        desired_ratios = {
            'assign': 0.60,
            'charge': 0.20,
            'reloc': 0.20,
            'idle': 0.02,
            'other': 0.05,
        }
        fill_ratio = buffer_size / max(buffer_capacity, 1)
        if fill_ratio <= 0.55:
            global_accept_prob = 1.0
        else:
            buffer_pressure = min(1.0, max(0.0, (fill_ratio - 0.55) / 0.45))
            global_accept_prob = max(0.10, 1.0 - 0.90 * buffer_pressure)
        ratio_margin = 0.12 if fill_ratio < 0.5 else (0.08 if training_step < 200 else 0.05)

        cache = getattr(self, '_aev_filter_bucket_cache', None)
        buffer_id = id(experience_buffer)
        if (
            cache is not None
            and cache.get('buffer_id') == buffer_id
            and cache.get('buffer_size') == buffer_size
        ):
            bucket_counts = dict(cache.get('bucket_counts', {}))
            for bucket in ('assign', 'charge', 'reloc', 'idle', 'other'):
                bucket_counts.setdefault(bucket, 0)
        else:
            bucket_counts = {'assign': 0, 'charge': 0, 'reloc': 0, 'idle': 0, 'other': 0}
            indexed_counts = getattr(experience_buffer, 'action_bucket_counts', None)
            if callable(indexed_counts):
                bucket_counts.update(indexed_counts())
            else:
                for exp in experience_buffer:
                    bucket_counts[action_bucket(exp.get('action_type', 'other'))] += 1

        filtered = []
        filtered_out_counts = {'assign': 0, 'charge': 0, 'reloc': 0, 'idle': 0, 'other': 0, 'fill_gate': 0}
        projected_total = max(buffer_size, 1)

        for exp in experiences:
            if global_accept_prob < 1.0 and random.random() > global_accept_prob:
                filtered_out_counts['fill_gate'] += 1
                continue

            bucket = action_bucket(exp.get('action_type', 'other'))
            desired_ratio = desired_ratios.get(bucket, desired_ratios['other'])
            max_allowed = int((desired_ratio + ratio_margin) * projected_total)
            min_keep = max(64, int(desired_ratio * min_buffer_before_filter * 0.5))
            current_count = bucket_counts.get(bucket, 0)

            if current_count >= max_allowed and current_count >= min_keep:
                filtered_out_counts[bucket] = filtered_out_counts.get(bucket, 0) + 1
                continue

            filtered.append(exp)
            bucket_counts[bucket] = current_count + 1
            projected_total += 1

        if experiences and len(filtered) != len(experiences):
            agg = getattr(self, '_aev_filter_log_agg', None)
            if agg is None:
                agg = {
                    'calls': 0,
                    'kept': 0,
                    'seen': 0,
                    'dropped': {'assign': 0, 'charge': 0, 'reloc': 0, 'idle': 0, 'other': 0, 'fill_gate': 0},
                }
                self._aev_filter_log_agg = agg
            agg['calls'] += 1
            agg['kept'] += len(filtered)
            agg['seen'] += len(experiences)
            for k, v in filtered_out_counts.items():
                agg['dropped'][k] = agg['dropped'].get(k, 0) + v

            if agg['calls'] >= 1000:
                print(
                    "🧹 Filtered AEV experiences (agg over "
                    f"{agg['calls']} calls): kept={agg['kept']}/{agg['seen']} "
                    f"buffer={buffer_size}/{buffer_capacity} "
                    f"fill_ratio={fill_ratio:.3f} "
                    f"accept_prob={global_accept_prob:.2f} "
                    f"train_step={training_step} "
                    f"dropped={agg['dropped']}",
                    flush=True,
                )
                self._aev_filter_log_agg = None

        if buffer_size + len(filtered) < buffer_capacity:
            self._aev_filter_bucket_cache = {
                'buffer_id': buffer_id,
                'buffer_size': buffer_size + len(filtered),
                'bucket_counts': dict(bucket_counts),
            }
        else:
            self._aev_filter_bucket_cache = None

        return filtered
    
    def generate_vehicle_qvalue_withoutqnetwork(self, vehicles_to_rebalance):
        n = len(vehicles_to_rebalance)
        invalid_q = -1e6
        is_ev_rows = np.array([self._is_ev(vid) for vid in vehicles_to_rebalance], dtype=bool)
        assigned = set()
        for vid, v in self.vehicles.items():
            if v['assigned_request'] is not None:
                assigned.add(v['assigned_request'])
            if v['passenger_onboard'] is not None:
                assigned.add(v['passenger_onboard'])
        active_avail = [r for r in self.active_requests.values() if r.request_id not in assigned]
        request_ids = list(getattr(self, '_last_matrix_request_ids', []))
        if request_ids or int(getattr(self, '_last_matrix_num_requests', 0) or 0) > 0:
            request_by_id = {r.request_id: r for r in active_avail}
            avail = [request_by_id[rid] for rid in request_ids if rid in request_by_id]
        else:
            avail = active_avail
        nr = int(getattr(self, '_last_matrix_num_requests', len(avail)) or 0)
        if nr == 0 and avail:
            nr = len(avail)

        # These lists may legitimately be empty after generate_whole_matrix filters
        # infeasible charge/zone columns.  Do not fall back to the full station/zone
        # set here, or q columns shift away from the action matrix.
        charge_station_ids = list(getattr(self, '_last_matrix_charge_station_ids', []))
        zone_target_ids = list(getattr(self, '_last_matrix_zone_target_ids', []))
        ns = len(charge_station_ids)
        nz = len(zone_target_ids)
        cols = nr + ns + nz + 1
        q = np.full((n, cols), invalid_q, dtype=np.float32)
        for i, vid in enumerate(vehicles_to_rebalance):
            vloc = self.vehicles[vid]['location']
            for j, req in enumerate(avail[:nr]):
                # No-network exact MCMF uses the observed/surged table value
                # directly.  Operating cost is realized by simula tion motion,
                # not subtracted a second time in this assignment objective.
                # MCMF-K additionally optimizes expected accepted value using
                # the deterministic known-rejection probability.
                request_q = float(
                    getattr(req, 'final_value', getattr(req, 'value', 0.0)) or 0.0
                )
                if getattr(self, 'knownreject', False):
                    reject_prob = self._calculate_known_rejection_probability(
                        int(vid), req
                    )
                    accept_prob = 1.0 - max(0.0, min(1.0, float(reject_prob)))
                    request_q *= accept_prob
                q[i, j] = request_q
            if not is_ev_rows[i]:
                for k, sid in enumerate(charge_station_ids):
                    sloc = self.charging_manager.stations[sid].location
                    d = self.get_distance_km(vloc, sloc)
                    charging_epochs = float(self._charge_duration_for_vehicle(int(vid)))
                    q[i, nr + k] = (
                        self._movement_cost(d)
                        - self.charging_penalty * charging_epochs
                    )
                for m, zone_id in enumerate(zone_target_ids):
                    rep = zone_id
                    d = self.get_distance_km(vloc, rep)
                    relocation_epochs = max(1.0, float(self.get_travel_time(vloc, rep)))
                    q[i, nr + ns + m] = (
                        self._movement_cost(d)
                        - self.idle_penalty * relocation_epochs
                    )
                q[i, -1] = -self.idle_penalty
            else:
                wait_target = self._sample_ev_default_relocation_target(int(vid))
                wait_distance = self.get_distance_km(vloc, wait_target)
                relocation_epochs = max(
                    1.0, float(self.get_travel_time(vloc, wait_target))
                )
                q[i, -1] = (
                    self._movement_cost(wait_distance)
                    - self.idle_penalty * relocation_epochs
                )
        return self._round_assignment_qvalues(q)

    def generate_vehicle_qvalue(self, vehicles_to_rebalance, onlyev=False, prior_features=None):
        del prior_features
        vf = self.value_function_ev if onlyev else self.value_function
        if vf is None or int(getattr(vf, 'training_step', 0) or 0) <= 0:
            return self.generate_vehicle_qvalue_withoutqnetwork(vehicles_to_rebalance)
        vf_ev_for_split = self.value_function_ev if (
            not onlyev
            and self.value_function_ev is not None
            and int(getattr(self.value_function_ev, 'training_step', 0) or 0) > 0
        ) else None

        assigned = set()
        for vid, vehicle in self.vehicles.items():
            if vehicle['assigned_request'] is not None:
                assigned.add(vehicle['assigned_request'])
            if vehicle['passenger_onboard'] is not None:
                assigned.add(vehicle['passenger_onboard'])
        active_avail = [req for req in self.active_requests.values() if req.request_id not in assigned]

        n = len(vehicles_to_rebalance)
        invalid_q = -1e6
        current_time = float(self.current_time) if hasattr(self, 'current_time') else 0.0
        other_vehicles = len([
            vehicle for vehicle in self.vehicles.values()
            if vehicle['assigned_request'] is None
            and vehicle['passenger_onboard'] is None
            and vehicle['charging_station'] is None
        ])
        num_reqs = len(self.active_requests)

        vehicle_action_matrix, nr, ns, nz = self.generate_whole_matrix(
            vehicles_to_rebalance, rebalance_num=n, onlyev=onlyev
        )
        total_cols = vehicle_action_matrix.shape[1]
        batch_q_value = np.full((n, total_cols), invalid_q, dtype=np.float32)
        if total_cols == 0:
            return self._round_assignment_qvalues(batch_q_value)

        request_ids = list(getattr(self, '_last_matrix_request_ids', []))
        if request_ids or nr > 0:
            request_by_id = {req.request_id: req for req in active_avail}
            avail = [request_by_id[request_id] for request_id in request_ids[:nr] if request_id in request_by_id]
        else:
            avail = active_avail

        vehicle_ids_arr = np.asarray(vehicles_to_rebalance, dtype=np.int32)
        vehicle_locations = np.asarray([self.vehicles[vid]['location'] for vid in vehicles_to_rebalance], dtype=np.int32)
        vehicle_battery = np.asarray([self.vehicles[vid]['battery'] for vid in vehicles_to_rebalance], dtype=np.float32)
        vehicle_idle = np.asarray([float(self.vehicles[vid].get('idle_timer', 0)) for vid in vehicles_to_rebalance], dtype=np.float32)
        is_ev_rows = np.asarray([self._is_ev(vid) for vid in vehicles_to_rebalance], dtype=bool)
        charge_station_ids = list(getattr(self, '_last_matrix_charge_station_ids', []))[:ns]
        zone_target_ids = list(getattr(self, '_last_matrix_zone_target_ids', []))[:nz]

        edge_row_parts = []
        edge_col_parts = []
        edge_action_parts = []
        edge_target_location_parts = []
        edge_request_value_parts = []
        edge_target_distance_parts = []
        edge_target_zoneid_parts = []
        edge_target_station_id_parts = []
        edge_post_action_location_parts = []
        edge_post_action_distance_parts = []
        edge_post_action_duration_parts = []
        edge_post_action_zoneid_parts = []

        if nr > 0:
            request_pickups = np.asarray([req.pickup for req in avail], dtype=np.int32)
            request_dropoffs = np.asarray([req.dropoff for req in avail], dtype=np.int32)
            request_values = np.asarray([req.final_value for req in avail], dtype=np.float32)
            request_zoneids = np.asarray([
                self.get_zone_embedding_id(req.pickup) if hasattr(self, 'get_zone_embedding_id') else 0 for req in avail
            ], dtype=np.int64)
            request_post_zoneids = np.asarray([
                self.get_zone_embedding_id(req.dropoff) if hasattr(self, 'get_zone_embedding_id') else 0 for req in avail
            ], dtype=np.int64)
            request_trip_distances = np.asarray([
                self._request_trip_distance_km(req) for req in avail
            ], dtype=np.float32)
            request_trip_durations = np.asarray([
                float(getattr(req, 'travel_time', self.get_travel_time(req.pickup, req.dropoff))) for req in avail
            ], dtype=np.float32)
            req_rows, req_cols = np.nonzero(vehicle_action_matrix[:, :nr] == 1)
            if req_rows.size > 0:
                pickup_distances = self.distance_matrix[vehicle_locations[req_rows], request_pickups[req_cols]].astype(np.float32)
                pickup_minutes = self.travel_time_matrix[vehicle_locations[req_rows], request_pickups[req_cols]].astype(np.float32)
                pickup_durations = np.where(
                    pickup_distances > 0.0,
                    np.maximum(1.0, pickup_minutes * 60.0 / float(self.EPOCH_LENGTH)),
                    0.0,
                ).astype(np.float32)
                edge_row_parts.append(req_rows)
                edge_col_parts.append(req_cols)
                edge_action_parts.append(np.full(req_rows.shape, 2, dtype=np.int64))
                edge_target_location_parts.append(request_pickups[req_cols])
                edge_request_value_parts.append(request_values[req_cols])
                edge_target_distance_parts.append(pickup_distances)
                edge_target_zoneid_parts.append(request_zoneids[req_cols])
                edge_target_station_id_parts.append(np.full(req_rows.shape, -1, dtype=np.int64))
                edge_post_action_location_parts.append(request_dropoffs[req_cols])
                edge_post_action_distance_parts.append(pickup_distances + request_trip_distances[req_cols])
                edge_post_action_duration_parts.append(pickup_durations + request_trip_durations[req_cols])
                edge_post_action_zoneid_parts.append(request_post_zoneids[req_cols])

        if ns > 0:
            station_locations = np.asarray(
                [self.charging_manager.stations[sid].location for sid in charge_station_ids],
                dtype=np.int32,
            )
            station_zoneids = np.asarray([self.get_zone_embedding_id(loc) for loc in station_locations], dtype=np.int64)
            charge_rows, charge_cols = np.nonzero(vehicle_action_matrix[:, nr:nr + ns] == 1)
            if charge_rows.size > 0:
                keep_charge_rows = ~is_ev_rows[charge_rows]
                charge_rows = charge_rows[keep_charge_rows]
                charge_cols = charge_cols[keep_charge_rows]
            if charge_rows.size > 0:
                edge_row_parts.append(charge_rows)
                edge_col_parts.append(nr + charge_cols)
                edge_action_parts.append(np.full(charge_rows.shape, 3, dtype=np.int64))
                edge_target_location_parts.append(station_locations[charge_cols])
                edge_request_value_parts.append(np.zeros(charge_rows.shape, dtype=np.float32))
                charge_distances = self.distance_matrix[vehicle_locations[charge_rows], station_locations[charge_cols]].astype(np.float32)
                charge_minutes = self.travel_time_matrix[vehicle_locations[charge_rows], station_locations[charge_cols]].astype(np.float32)
                charge_travel_durations = np.where(
                    charge_distances > 0.0,
                    np.maximum(1.0, charge_minutes * 60.0 / float(self.EPOCH_LENGTH)),
                    0.0,
                ).astype(np.float32)
                edge_target_distance_parts.append(charge_distances)
                edge_target_zoneid_parts.append(station_zoneids[charge_cols])
                station_ids_arr = np.asarray(charge_station_ids, dtype=np.int64)
                edge_target_station_id_parts.append(station_ids_arr[charge_cols])
                edge_post_action_location_parts.append(station_locations[charge_cols])
                edge_post_action_distance_parts.append(charge_distances)
                charge_session_durations = np.asarray([
                    float(self._charge_duration_for_vehicle(int(vehicles_to_rebalance[row_idx])))
                    for row_idx in charge_rows
                ], dtype=np.float32)
                edge_post_action_duration_parts.append(charge_travel_durations + charge_session_durations)
                edge_post_action_zoneid_parts.append(station_zoneids[charge_cols])

        if nz > 0:
            relocation_targets = np.asarray(zone_target_ids[:nz], dtype=np.int32)
            relocation_zoneids = np.asarray([self.get_zone_embedding_id(loc) for loc in relocation_targets], dtype=np.int64)
            reloc_rows, reloc_cols = np.nonzero(vehicle_action_matrix[:, nr + ns:nr + ns + nz] == 1)
            if reloc_rows.size > 0:
                keep_reloc_rows = ~is_ev_rows[reloc_rows]
                reloc_rows = reloc_rows[keep_reloc_rows]
                reloc_cols = reloc_cols[keep_reloc_rows]
            if reloc_rows.size > 0:
                edge_row_parts.append(reloc_rows)
                edge_col_parts.append(nr + ns + reloc_cols)
                edge_action_parts.append(np.full(reloc_rows.shape, 1, dtype=np.int64))
                edge_target_location_parts.append(relocation_targets[reloc_cols])
                edge_request_value_parts.append(np.zeros(reloc_rows.shape, dtype=np.float32))
                reloc_distances = self.distance_matrix[vehicle_locations[reloc_rows], relocation_targets[reloc_cols]].astype(np.float32)
                reloc_minutes = self.travel_time_matrix[vehicle_locations[reloc_rows], relocation_targets[reloc_cols]].astype(np.float32)
                reloc_durations = np.where(
                    reloc_distances > 0.0,
                    np.maximum(1.0, reloc_minutes * 60.0 / float(self.EPOCH_LENGTH)),
                    0.0,
                ).astype(np.float32)
                edge_target_distance_parts.append(reloc_distances)
                edge_target_zoneid_parts.append(relocation_zoneids[reloc_cols])
                edge_target_station_id_parts.append(np.full(reloc_rows.shape, -1, dtype=np.int64))
                edge_post_action_location_parts.append(relocation_targets[reloc_cols])
                edge_post_action_distance_parts.append(reloc_distances)
                edge_post_action_duration_parts.append(reloc_durations)
                edge_post_action_zoneid_parts.append(relocation_zoneids[reloc_cols])

        wait_rows = np.flatnonzero(vehicle_action_matrix[:, -1] == 1)
        if wait_rows.size > 0:
            wait_targets = vehicle_locations[wait_rows].copy()
            for local_index, row_index in enumerate(wait_rows):
                vehicle_id = int(vehicles_to_rebalance[int(row_index)])
                if is_ev_rows[int(row_index)]:
                    wait_targets[local_index] = self._sample_ev_default_relocation_target(
                        vehicle_id
                    )
            wait_distances = self.distance_matrix[
                vehicle_locations[wait_rows], wait_targets
            ].astype(np.float32)
            wait_minutes = self.travel_time_matrix[
                vehicle_locations[wait_rows], wait_targets
            ].astype(np.float32)
            wait_durations = np.where(
                wait_distances > 0.0,
                np.maximum(
                    1.0,
                    wait_minutes * 60.0 / float(self.EPOCH_LENGTH),
                ),
                0.0,
            ).astype(np.float32)
            wait_zoneids = np.asarray(
                [self.get_zone_embedding_id(int(loc)) for loc in wait_targets],
                dtype=np.int64,
            )
            edge_row_parts.append(wait_rows)
            edge_col_parts.append(np.full(wait_rows.shape, total_cols - 1, dtype=np.int64))
            edge_action_parts.append(np.full(wait_rows.shape, 1, dtype=np.int64))
            edge_target_location_parts.append(wait_targets)
            edge_request_value_parts.append(np.zeros(wait_rows.shape, dtype=np.float32))
            edge_target_distance_parts.append(wait_distances)
            edge_target_zoneid_parts.append(wait_zoneids)
            edge_target_station_id_parts.append(np.full(wait_rows.shape, -1, dtype=np.int64))
            edge_post_action_location_parts.append(wait_targets)
            edge_post_action_distance_parts.append(wait_distances)
            edge_post_action_duration_parts.append(wait_durations)
            edge_post_action_zoneid_parts.append(wait_zoneids)

        if not edge_row_parts:
            self._maybe_print_qmatrix_diagnostic(
                batch_q_value,
                vehicle_action_matrix,
                nr,
                ns,
                nz,
                onlyev=onlyev,
                is_ev_rows=is_ev_rows,
            )
            return self._round_assignment_qvalues(batch_q_value)

        edge_rows = np.concatenate(edge_row_parts)
        edge_cols = np.concatenate(edge_col_parts)
        action_type_ids = np.concatenate(edge_action_parts)
        target_locations = np.concatenate(edge_target_location_parts)
        request_values = np.concatenate(edge_request_value_parts)
        target_distances = np.concatenate(edge_target_distance_parts)
        target_zoneids = np.concatenate(edge_target_zoneid_parts)
        target_station_ids = np.concatenate(edge_target_station_id_parts)
        post_action_locations = np.concatenate(edge_post_action_location_parts)
        post_action_distances = np.concatenate(edge_post_action_distance_parts)
        post_action_durations = np.concatenate(edge_post_action_duration_parts)
        post_action_zoneids = np.concatenate(edge_post_action_zoneid_parts)

        q_vals = np.zeros(edge_rows.shape, dtype=np.float32)
        ev_edge_mask = is_ev_rows[edge_rows]
        aev_edge_mask = ~ev_edge_mask
        common_kwargs = dict(
            current_time=current_time,
            other_vehicles=other_vehicles,
            num_reqs=num_reqs,
        )

        def _forward(net, sel_mask):
            sel_idx = np.flatnonzero(sel_mask)
            if sel_idx.size == 0:
                return
            forward_kwargs = dict(
                vehicle_ids=vehicle_ids_arr[edge_rows[sel_idx]],
                vehicle_locations=vehicle_locations[edge_rows[sel_idx]],
                target_locations=target_locations[sel_idx],
                current_times=np.full(sel_idx.shape, common_kwargs['current_time'], dtype=np.float32),
                other_vehicles=np.full(sel_idx.shape, common_kwargs['other_vehicles'], dtype=np.float32),
                num_requests=np.full(sel_idx.shape, common_kwargs['num_reqs'], dtype=np.float32),
                battery_levels=vehicle_battery[edge_rows[sel_idx]],
                request_values=request_values[sel_idx],
                target_distances=target_distances[sel_idx],
                target_zoneids=target_zoneids[sel_idx],
                vehicle_idle_times=vehicle_idle[edge_rows[sel_idx]],
                action_type_ids=action_type_ids[sel_idx],
                post_action_distances=post_action_distances[sel_idx],
                post_action_durations=post_action_durations[sel_idx],
                post_action_zoneids=post_action_zoneids[sel_idx],
            )
            if getattr(net, 'uses_post_action_locations', False):
                forward_kwargs['post_action_locations'] = post_action_locations[sel_idx]
            if getattr(net, 'uses_queue_wait_loss', False):
                forward_kwargs['target_station_ids'] = target_station_ids[sel_idx]
            graph_encoder = getattr(net, 'graph_encoder', None)
            if graph_encoder is not None and hasattr(graph_encoder, 'neighbour_number'):
                forward_kwargs['vehicle_neighbour_candidates'] = getattr(
                    self,
                    '_last_vehicle_action_graph_neighbours',
                    {},
                )
            out = net.batch_get_mixed_q_values(**forward_kwargs)
            q_vals[sel_idx] = np.asarray(out, dtype=np.float32)

        if onlyev:
            _forward(vf, np.ones_like(ev_edge_mask, dtype=bool))
        else:
            _forward(vf, aev_edge_mask)
            ev_net = vf_ev_for_split if vf_ev_for_split is not None else vf
            _forward(ev_net, ev_edge_mask)
        if getattr(self, 'knownreject', False) and nr > 0:
            request_edge_idx = np.flatnonzero(action_type_ids == 2)
            if request_edge_idx.size > 0:
                accept_multipliers = np.ones(request_edge_idx.shape, dtype=np.float32)
                for local_idx, edge_idx in enumerate(request_edge_idx):
                    req_col = int(edge_cols[edge_idx])
                    if req_col < 0 or req_col >= len(avail):
                        continue
                    vid = int(vehicle_ids_arr[edge_rows[edge_idx]])
                    reject_prob = self._calculate_known_rejection_probability(vid, avail[req_col])
                    accept_multipliers[local_idx] = max(0.0, min(1.0, 1.0 - float(reject_prob)))
                q_vals[request_edge_idx] *= accept_multipliers
        batch_q_value[edge_rows, edge_cols] = q_vals
        self._maybe_print_qmatrix_diagnostic(
            batch_q_value,
            vehicle_action_matrix,
            nr,
            ns,
            nz,
            onlyev=onlyev,
            is_ev_rows=is_ev_rows,
        )
        return self._round_assignment_qvalues(batch_q_value)

    def _maybe_print_qmatrix_diagnostic(
        self,
        q_matrix,
        action_matrix,
        nr,
        ns,
        nz,
        onlyev=False,
        is_ev_rows=None,
    ):
        interval = int(getattr(self, 'qmatrix_diagnostic_interval', 0) or 0)
        if interval <= 0:
            return
        step = int(getattr(self, 'current_time', 0) or 0)
        if step % interval != 0:
            return
        key = (step, bool(onlyev))
        seen = getattr(self, '_last_qmatrix_diagnostic_keys', set())
        if key in seen:
            return
        seen.add(key)
        if len(seen) > 256:
            seen.clear()
            seen.add(key)
        self._last_qmatrix_diagnostic_keys = seen

        invalid_q = -1e5

        def summarize(start_col, end_col, row_mask=None):
            if end_col <= start_col:
                return None
            if row_mask is None:
                sub_q = q_matrix[:, start_col:end_col]
                sub_mask = action_matrix[:, start_col:end_col] == 1
            else:
                row_mask = np.asarray(row_mask, dtype=bool)
                if row_mask.size != q_matrix.shape[0] or not np.any(row_mask):
                    return None
                sub_q = q_matrix[row_mask, start_col:end_col]
                sub_mask = action_matrix[row_mask, start_col:end_col] == 1
            finite = np.isfinite(sub_q) & (sub_q > invalid_q) & sub_mask
            if not np.any(finite):
                return None
            values = sub_q[finite].astype(np.float64)
            return {
                'n': int(values.size),
                'mean': float(np.mean(values)),
                'p50': float(np.percentile(values, 50)),
                'max': float(np.max(values)),
            }

        def fmt(name, stats):
            if not stats:
                return f"{name}=NA"
            return (
                f"{name}(n={stats['n']},mean={stats['mean']:.3f},"
                f"p50={stats['p50']:.3f},max={stats['max']:.3f})"
            )

        def scope_line(scope_name, row_mask=None):
            req_stats = summarize(0, nr, row_mask=row_mask)
            charge_stats = summarize(nr, nr + ns, row_mask=row_mask)
            reloc_stats = summarize(nr + ns, nr + ns + nz, row_mask=row_mask)
            wait_stats = summarize(nr + ns + nz, nr + ns + nz + 1, row_mask=row_mask)
            diff_parts = []
            if req_stats and wait_stats:
                diff_parts.append(f"wait_minus_request={wait_stats['mean'] - req_stats['mean']:.3f}")
            if req_stats and reloc_stats:
                diff_parts.append(f"reloc_minus_request={reloc_stats['mean'] - req_stats['mean']:.3f}")
            diff_text = " ".join(diff_parts) if diff_parts else "diff=NA"
            print(
                "ADPQMatrix "
                f"step={step} scope={scope_name} onlyev={bool(onlyev)} "
                f"{fmt('request', req_stats)} "
                f"{fmt('charge', charge_stats)} "
                f"{fmt('reloc', reloc_stats)} "
                f"{fmt('wait', wait_stats)} "
                f"{diff_text}",
                flush=True,
            )

        scope_line("all", row_mask=None)
        if is_ev_rows is not None:
            ev_rows = np.asarray(is_ev_rows, dtype=bool)
            if ev_rows.size == q_matrix.shape[0]:
                if np.any(ev_rows):
                    scope_line("EV", row_mask=ev_rows)
                if np.any(~ev_rows):
                    scope_line("AEV", row_mask=~ev_rows)

    # ==================================================================
    # Various compat helpers
    # ==================================================================

    def findchargerange_v(self):
        ret = {}
        for sid, station in self.charging_manager.stations.items():
            ret[sid] = []
            for vid, v in self.vehicles.items():
                d = self.get_distance_km(v['location'], station.location)
                if d <= 5.0:
                    ret[sid].append(vid)
        return ret

    def findchargerange_c(self, rebalance_num=0):
        ret = {}
        for vid, v in self.vehicles.items():
            cap = 0
            for station in self.charging_manager.stations.values():
                d = self.get_distance_km(v['location'], station.location)
                if d * self.battery_consum <= v['battery'] - 0.01:
                    cap += station.max_capacity - len(station.current_vehicles) - len(station.charging_queue_notarrived)
            ret[vid] = max(cap - rebalance_num, 0)
        return ret

    def return_nearest_idle_target(self, vehicle_id):
        if not self.hotspot_locations:
            return self.vehicles[vehicle_id]['location']
        vloc = self.vehicles[vehicle_id]['location']
        best = min(self.hotspot_locations, key=lambda z: self.get_distance_km(vloc, z))
        return best

    def return_nearest_hotspot_index(self, vehicle_id):
        if not self.hotspot_locations:
            return 0
        vloc = self.vehicles[vehicle_id]['location']
        dists = [(self.get_distance_km(vloc, z), idx) for idx, z in enumerate(self.hotspot_locations)]
        return min(dists, key=lambda x: x[0])[1]

    def _update_storeaction(self, vehicle_id, action, storeactions_dict, is_ev=False):
        self._attach_bootstrap_candidates(action, vehicle_id)
        sa = self.storeactions_ev if is_ev else self.storeactions
        if sa.get(vehicle_id) is None:
            storeactions_dict[vehicle_id] = action
            sa[vehicle_id] = action
            sa[vehicle_id].dur_reward = 0
            sa[vehicle_id].current_time = self.current_time
        else:
            storeactions_dict[vehicle_id].next_action = action
            storeactions_dict[vehicle_id].vehicle_loc_post = self.vehicles[vehicle_id]['location']
            storeactions_dict[vehicle_id].vehicle_battery_post = self.vehicles[vehicle_id]['battery']
            old_t = getattr(storeactions_dict[vehicle_id], 'current_time', self.current_time)
            sa[vehicle_id] = action
            sa[vehicle_id].dur_reward = 0
            sa[vehicle_id].dur_time = self.current_time - old_t
            sa[vehicle_id].current_time = self.current_time

    # ==================================================================
    # Episode stats
    # ==================================================================

    def get_episode_stats(self):
        total_bat = sum(v['battery'] for v in self.vehicles.values())
        avg_bat = total_bat / max(1, len(self.vehicles))
        total_rejected = len(self.rejected_requests)
        total_ev_req = len(self.ev_requests)

        if self.charging_usage_history:
            avg_per_station = sum(u['vehicles_per_station'] for u in self.charging_usage_history) / len(self.charging_usage_history)
            avg_occ = sum(u['total_occupied'] for u in self.charging_usage_history) / len(self.charging_usage_history)
            total_cap = sum(s.max_capacity for s in self.charging_manager.stations.values())
            avg_util = avg_occ / max(1, total_cap)
        else:
            avg_per_station = 0
            avg_util = 0

        completed = len(self.completed_requests)
        completed_ev = len(self.completed_requests_ev)
        rejected_requests = len({
            request.request_id for request in self.rejected_requests
        })
        recourse_requests = len(self.ev_rejected_recovered_same_epoch_ids)
        lost_requests = max(0, int(self.whole_req_num) - completed)
        service_ratio = completed / self.whole_req_num if self.whole_req_num > 0 else 0
        avg_val = self.request_value_sum / completed if completed > 0 else 0
        rejection_reward_count = int(getattr(self, 'rejection_reward_count', 0))
        rejection_reward_total = float(getattr(self, 'rejection_reward_total', 0.0))
        avg_rejection_reward = (
            rejection_reward_total / rejection_reward_count
            if rejection_reward_count > 0 else 0.0
        )
        online_vehicles = sum(1 for v in self.vehicles.values() if v.get('is_online', True))
        offline_vehicles = sum(1 for v in self.vehicles.values() if not v.get('is_online', True))
        ev_count = sum(1 for v in self.vehicles.values() if v.get('type') == 1)
        total_dropoffs = sum(self.period_dropout_counts) + self.current_period_dropout_count
        drop_off_rate = total_dropoffs / max(1, ev_count)

        ev_vehicles = [v for v in self.vehicles.values() if v['type'] == 1]
        aev_vehicles = [v for v in self.vehicles.values() if v['type'] == 2]
        ev_rej = sum(v['rejected_requests'] for v in ev_vehicles)
        aev_rej = sum(v['rejected_requests'] for v in aev_vehicles)
        ev_offers = int(getattr(self, 'ev_offer_count', 0))
        rejected_unique = len(getattr(self, 'ev_rejected_request_ids', set()))
        recovered_same_epoch = len(
            getattr(self, 'ev_rejected_recovered_same_epoch_ids', set())
        )
        rejected_completed = len(
            getattr(self, 'ev_rejected_completed_ids', set())
        )
        recovery_delays = list(getattr(self, 'recovery_delays', []))
        residual_count = int(getattr(self, 'residual_request_count', 0))
        unoffered_count = int(getattr(self, 'unoffered_request_count', 0))
        aev_request_assignments = int(
            getattr(self, 'aev_request_assignment_count', 0)
        )

        avg_reb = 0
        total_reb = 0
        avg_reb_whole = 0
        if self.rebalancing_assignments_per_step:
            total_reb = sum(self.rebalancing_assignments_per_step)
            avg_reb = total_reb / len(self.rebalancing_assignments_per_step)
            avg_reb_whole = sum(self.rebalancing_whole) / len(self.rebalancing_whole) if self.rebalancing_whole else 0

        hourly_completed_map = {}
        for record in self.completed_request_time_records:
            key = (record.get('completed_date'), int(record.get('completed_hour', 0)))
            bucket = hourly_completed_map.setdefault(
                key,
                {
                    'completed_date': record.get('completed_date'),
                    'completed_hour': int(record.get('completed_hour', 0)),
                    'completed_orders': 0,
                    'completed_ev_orders': 0,
                    'completed_aev_orders': 0,
                },
            )
            bucket['completed_orders'] += int(record.get('completed_orders', 0))
            bucket['completed_ev_orders'] += int(record.get('completed_ev_orders', 0))
            bucket['completed_aev_orders'] += int(record.get('completed_aev_orders', 0))
        hourly_completed_orders = [
            hourly_completed_map[key]
            for key in sorted(hourly_completed_map.keys(), key=lambda item: (item[0] or '', item[1]))
        ]
        hourly_zone_request_map = {}
        for record in self.request_generation_history:
            request_time = float(record.get('time', self.current_time))
            request_date = str(self._current_date_label(request_time))
            request_hour = int(self.get_hour_of_day(request_time)) % 24
            pickup_zone = record.get('pickup_zone')
            if pickup_zone is None:
                continue
            key = (request_date, request_hour, int(pickup_zone))
            bucket = hourly_zone_request_map.setdefault(
                key,
                {
                    'request_date': request_date,
                    'request_hour': request_hour,
                    'zone_id': int(pickup_zone),
                    'generated_requests': 0,
                    'completed_requests': 0,
                },
            )
            bucket['generated_requests'] += 1

        daily_zone_request_map = {}
        for record in self.request_generation_history:
            request_time = float(record.get('time', self.current_time))
            request_date = str(self._current_date_label(request_time))
            pickup_zone = record.get('pickup_zone')
            if pickup_zone is None:
                continue
            key = (request_date, int(pickup_zone))
            bucket = daily_zone_request_map.setdefault(
                key,
                {
                    'request_date': request_date,
                    'zone_id': int(pickup_zone),
                    'generated_requests': 0,
                    'completed_requests': 0,
                },
            )
            bucket['generated_requests'] += 1

        for record in self.completed_request_time_records:
            completed_date = record.get('completed_date')
            completed_hour = int(record.get('completed_hour', 0))
            pickup_zone = record.get('pickup_zone')
            if pickup_zone is None:
                continue
            hourly_key = (completed_date, completed_hour, int(pickup_zone))
            hourly_bucket = hourly_zone_request_map.setdefault(
                hourly_key,
                {
                    'request_date': completed_date,
                    'request_hour': completed_hour,
                    'zone_id': int(pickup_zone),
                    'generated_requests': 0,
                    'completed_requests': 0,
                },
            )
            hourly_bucket['completed_requests'] += int(record.get('completed_orders', 0))

            daily_key = (completed_date, int(pickup_zone))
            daily_bucket = daily_zone_request_map.setdefault(
                daily_key,
                {
                    'request_date': completed_date,
                    'zone_id': int(pickup_zone),
                    'generated_requests': 0,
                    'completed_requests': 0,
                },
            )
            daily_bucket['completed_requests'] += int(record.get('completed_orders', 0))

        hourly_zone_request_completed_orders = []
        for key in sorted(hourly_zone_request_map.keys(), key=lambda item: (item[0] or '', item[1], item[2])):
            bucket = hourly_zone_request_map[key]
            generated_requests = int(bucket.get('generated_requests', 0))
            completed_requests = int(bucket.get('completed_requests', 0))
            hourly_zone_request_completed_orders.append({
                'request_date': bucket.get('request_date'),
                'request_hour': int(bucket.get('request_hour', 0)),
                'zone_id': int(bucket.get('zone_id', 0)),
                'generated_requests': generated_requests,
                'completed_requests': completed_requests,
                'completion_ratio': completed_requests / generated_requests if generated_requests > 0 else 0.0,
            })

        daily_zone_request_completion_shares = []
        for key in sorted(daily_zone_request_map.keys(), key=lambda item: (item[0] or '', item[1])):
            bucket = daily_zone_request_map[key]
            generated_requests = int(bucket.get('generated_requests', 0))
            completed_requests = int(bucket.get('completed_requests', 0))
            daily_zone_request_completion_shares.append({
                'request_date': bucket.get('request_date'),
                'zone_id': int(bucket.get('zone_id', 0)),
                'generated_requests': generated_requests,
                'completed_requests': completed_requests,
                'completion_ratio': completed_requests / generated_requests if generated_requests > 0 else 0.0,
            })
        hourly_zone_vehicle_counts = []
        if self.hourly_zone_vehicle_snapshots:
            for key in sorted(self.hourly_zone_vehicle_snapshots.keys(), key=lambda item: (item[0] or '', item[1], item[2])):
                bucket = self.hourly_zone_vehicle_snapshots[key]
                snapshot_count = int(bucket.get('snapshot_count', 0))
                divisor = float(snapshot_count) if snapshot_count > 0 else 1.0
                hourly_zone_vehicle_counts.append({
                    'date': bucket.get('date'),
                    'hour': int(bucket.get('hour', 0)),
                    'zone_id': int(bucket.get('zone_id', 0)),
                    'snapshot_count': snapshot_count,
                    'mean_total_vehicles': float(bucket.get('total_vehicles_sum', 0.0)) / divisor,
                    'mean_ev_vehicles': float(bucket.get('ev_vehicles_sum', 0.0)) / divisor,
                    'mean_aev_vehicles': float(bucket.get('aev_vehicles_sum', 0.0)) / divisor,
                })
        hourly_zone_charge_station_counts = []
        if self.hourly_zone_charge_station_snapshots:
            for key in sorted(self.hourly_zone_charge_station_snapshots.keys(), key=lambda item: (item[0] or '', item[1], item[2])):
                bucket = self.hourly_zone_charge_station_snapshots[key]
                snapshot_count = int(bucket.get('snapshot_count', 0))
                divisor = float(snapshot_count) if snapshot_count > 0 else 1.0
                mean_station_count = float(bucket.get('station_count_sum', 0.0)) / divisor
                mean_total_capacity = float(bucket.get('total_capacity_sum', 0.0)) / divisor
                mean_queue_vehicle_count = float(bucket.get('queue_vehicle_count_sum', 0.0)) / divisor
                hourly_zone_charge_station_counts.append({
                    'date': bucket.get('date'),
                    'hour': int(bucket.get('hour', 0)),
                    'zone_id': int(bucket.get('zone_id', 0)),
                    'snapshot_count': snapshot_count,
                    'mean_station_count': mean_station_count,
                    'mean_total_capacity': mean_total_capacity,
                    'mean_queue_vehicle_count': mean_queue_vehicle_count,
                    'mean_queue_to_capacity_ratio': mean_queue_vehicle_count / mean_total_capacity if mean_total_capacity > 0 else 0.0,
                })
        avg_charging_wait_time = (
            float(np.mean([obs['observed_wait'] for obs in self.charging_wait_observations]))
            if self.charging_wait_observations else 0.0
        )
        wait_metrics = positive_wait_metrics(
            self.charging_wait_observations,
            active_arrivals=getattr(self, '_charging_queue_arrivals', {}),
            current_time=float(self.current_time),
        )
        current_pressure_snapshot = self._compute_station_pressure_snapshot()
        pressure_snapshot_count = int(self.station_pressure_snapshot_count)
        if pressure_snapshot_count > 0:
            mean_station_pressure = self.station_pressure_mean_sum / float(pressure_snapshot_count)
            mean_station_pressure_ratio = self.station_pressure_ratio_mean_sum / float(pressure_snapshot_count)
            current_max_pressure = int(current_pressure_snapshot['max_station_pressure'])
            if current_max_pressure > int(self.max_station_pressure):
                max_station_pressure = current_max_pressure
                max_station_pressure_station_id = int(current_pressure_snapshot['max_station_pressure_station_id'])
            else:
                max_station_pressure = int(self.max_station_pressure)
                max_station_pressure_station_id = int(self.max_station_pressure_station_id)

            current_max_pressure_ratio = float(current_pressure_snapshot['max_station_pressure_ratio'])
            if current_max_pressure_ratio > float(self.max_station_pressure_ratio):
                max_station_pressure_ratio = current_max_pressure_ratio
                max_station_pressure_ratio_station_id = int(current_pressure_snapshot['max_station_pressure_ratio_station_id'])
            else:
                max_station_pressure_ratio = float(self.max_station_pressure_ratio)
                max_station_pressure_ratio_station_id = int(self.max_station_pressure_ratio_station_id)
        else:
            mean_station_pressure = float(current_pressure_snapshot['mean_station_pressure'])
            mean_station_pressure_ratio = float(current_pressure_snapshot['mean_station_pressure_ratio'])
            max_station_pressure = int(current_pressure_snapshot['max_station_pressure'])
            max_station_pressure_station_id = int(current_pressure_snapshot['max_station_pressure_station_id'])
            max_station_pressure_ratio = float(current_pressure_snapshot['max_station_pressure_ratio'])
            max_station_pressure_ratio_station_id = int(current_pressure_snapshot['max_station_pressure_ratio_station_id'])

        return {
            'episode_time': self.current_time,
            'total_orders': len(self.active_requests) + completed + total_rejected,
            'accepted_orders': len(self.active_requests) + completed,
            'active_orders': len(self.active_requests),
            'rejected_orders': total_rejected,
            'rejected_requests': rejected_requests,
            'recourse_requests': recourse_requests,
            'lost_requests': lost_requests,
            'rejection_reward_total': rejection_reward_total,
            'rejection_reward_count': rejection_reward_count,
            'avg_rejection_reward': avg_rejection_reward,
            'ev_accept': total_ev_req,
            'completed_orders': completed,
            'completed_ev_orders': completed_ev,
            'service_ratio': service_ratio,
            'avg_request_value': avg_val,
            'avg_battery': avg_bat,
            'charge_finished': self.charge_finished,
            'charging_wait_penalty_total': float(self.charging_wait_penalty_total),
            'charging_wait_steps': int(self.charging_wait_steps),
            'avg_charging_wait_time': avg_charging_wait_time,
            'avg_wait': float(wait_metrics['avg_wait']),
            'waiting_vehicle_count': int(wait_metrics['waiting_vehicle_count']),
            'completed_waiting_vehicle_count': int(
                wait_metrics['completed_waiting_vehicle_count']
            ),
            'ongoing_waiting_vehicle_count': int(
                wait_metrics['ongoing_waiting_vehicle_count']
            ),
            'charging_wait_observations': list(self.charging_wait_observations),
            'max_station_pressure': int(max_station_pressure),
            'max_station_pressure_station_id': int(max_station_pressure_station_id),
            'mean_station_pressure': float(mean_station_pressure),
            'max_station_pressure_ratio': float(max_station_pressure_ratio),
            'max_station_pressure_ratio_station_id': int(max_station_pressure_ratio_station_id),
            'mean_station_pressure_ratio': float(mean_station_pressure_ratio),
            'station_pressure_snapshot_count': pressure_snapshot_count,
            'avg_station_utilization': avg_util,
            'avg_vehicles_per_station': avg_per_station,
            'ev_rejected': ev_rej,
            'aev_rejected': aev_rej,
            'recourse_variant': getattr(self, 'recourse_variant', 'legacy'),
            'ev_offer_count': ev_offers,
            'ev_offer_rate': ev_offers / max(
                1, int(getattr(self, 'ev_eligible_decision_count', 0))
            ),
            'ev_conditional_acceptance_rate': (
                max(0, ev_offers - ev_rej) / ev_offers if ev_offers else 0.0
            ),
            'ev_conditional_rejection_rate': ev_rej / ev_offers if ev_offers else 0.0,
            'rejected_request_recovery_rate_same_epoch': (
                recovered_same_epoch / rejected_unique
                if rejected_unique else 0.0
            ),
            'rejected_request_recovered_same_epoch': recovered_same_epoch,
            'rejected_request_unique_count': rejected_unique,
            'rejected_request_lost_count': max(0, rejected_unique - rejected_completed),
            'lost_after_rejection_rate': (
                max(0, rejected_unique - rejected_completed) / rejected_unique
                if rejected_unique else 0.0
            ),
            'mean_recovery_delay_epochs': (
                float(np.mean(recovery_delays)) if recovery_delays else 0.0
            ),
            'residual_request_count': residual_count,
            'residual_request_served_count': int(
                getattr(self, 'residual_request_served_count', 0)
            ),
            'residual_request_coverage': (
                int(getattr(self, 'residual_request_served_count', 0))
                / residual_count if residual_count else 0.0
            ),
            'unoffered_request_count': unoffered_count,
            'unoffered_request_served_count': int(
                getattr(self, 'unoffered_request_served_count', 0)
            ),
            'unoffered_request_coverage': (
                int(getattr(self, 'unoffered_request_served_count', 0))
                / unoffered_count if unoffered_count else 0.0
            ),
            'aev_recourse_assignment_count': int(
                getattr(self, 'aev_recourse_assignment_count', 0)
            ),
            'aev_recourse_share': (
                int(getattr(self, 'aev_recourse_assignment_count', 0))
                / aev_request_assignments if aev_request_assignments else 0.0
            ),
            'ev_vehicles': len(ev_vehicles),
            'aev_vehicles': len(aev_vehicles),
            'whole_req_num': self.whole_req_num,
            'avg_rebalancing_assignments': avg_reb,
            'total_rebalancing_assignments': total_reb,
            'total_rebalancing_calls': self.total_rebalancing_calls,
            'avg_rebalance_whole': avg_reb_whole,
            'avg_battery_level': avg_bat,
            'online_vehicles': online_vehicles,
            'offline_vehicles': offline_vehicles,
            'drop_off_rate': drop_off_rate,
            'dropoff_driver_count': total_dropoffs,
            'daily_online_history': list(self.daily_online_history),
            'period_dropout_counts': list(self.period_dropout_counts) + [self.current_period_dropout_count],
            'hourly_zone_request_completed_orders': hourly_zone_request_completed_orders,
            'daily_zone_request_completion_shares': daily_zone_request_completion_shares,
            'hourly_zone_vehicle_counts': hourly_zone_vehicle_counts,
            'hourly_zone_charge_station_counts': hourly_zone_charge_station_counts,
            'hourly_completed_orders': hourly_completed_orders,
            'current_real_date': str(self._current_date_label()),
        }

    def get_stats(self):
        avg_bat = sum(v['battery'] for v in self.vehicles.values()) / max(1, len(self.vehicles))
        return {
            'vehicles': len(self.vehicles),
            'online_vehicles': sum(1 for v in self.vehicles.values() if v.get('is_online', True)),
            'offline_vehicles': sum(1 for v in self.vehicles.values() if not v.get('is_online', True)),
            'active_requests': len(self.active_requests),
            'generated_requests_last_step': self.last_generated_requests,
            'last_generated_request_time': self.last_generated_request_time,
            'total_generated_requests': self.whole_req_num,
            'current_real_date': str(self._current_date_label().date()) if self._current_date_label() is not None else None,
            'current_real_hour': self.get_hour_of_day(),
            'min_vehicle_battery': min((v['battery'] for v in self.vehicles.values()), default=0.0),
            'vehicles_unable_to_reach_charging': self._count_vehicles_unable_to_reach_charging(),
            'completed_requests': len(self.completed_requests),
            'completed_orders_req': len(self.completed_requests),
            'rejected_requests': len({
                request.request_id for request in self.rejected_requests
            }),
            'recourse_requests': len(self.ev_rejected_recovered_same_epoch_ids),
            'lost_requests': max(
                0,
                int(self.whole_req_num) - len(self.completed_requests),
            ),
            'avg_battery': avg_bat,
            'average_battery': avg_bat,
        }

    # ==================================================================
    # Random seed helpers
    # ==================================================================

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def set_request_generation_seed(self, seed):
        self.request_generation_seed = seed

    # ==================================================================
    # Misc compat (ChargingIntegratedEnvironment attributes trainers may access)
    # ==================================================================

    @property
    def grid_size(self):
        """Compat: some code references env.grid_size.  Return sqrt(NUM_LOCATIONS)."""
        return int(math.ceil(math.sqrt(self.NUM_LOCATIONS)))

    def evaluate_service_option(self, vehicle_id, request, ifEVQvalue=False):
        if request is None:
            return 0.0
        vf = self.value_function_ev if ifEVQvalue else self.value_function
        if vf and hasattr(vf, 'get_assignment_q_value'):
            v = self.vehicles.get(vehicle_id, {})
            vehicle_location = int(v.get('location', getattr(request, 'pickup', 0)))
            pickup_location = int(getattr(request, 'pickup', vehicle_location))
            request_value = float(getattr(request, 'final_value', getattr(request, 'value', 0.0)) or 0.0)
            pickup_dist = float(self.get_distance_km(vehicle_location, pickup_location))
            pick_zone = int(self.get_zone_embedding_id(pickup_location)) if hasattr(self, 'get_zone_embedding_id') else 0
            assignment_kwargs = {
                'vehicle_id': vehicle_id,
                'target_id': getattr(request, 'request_id', 0),
                'vehicle_location': vehicle_location,
                'target_reject': pickup_location,
                'target_location': pickup_location,
                'current_time': self.current_time,
                'other_vehicles': len(self.vehicles),
                'num_requests': len(self.active_requests),
                'battery_level': float(v.get('battery', 1.0)),
                'request_value': request_value,
                'pickup_dist': pickup_dist,
                'pick_zone': pick_zone,
            }
            try:
                signature = inspect.signature(vf.get_assignment_q_value)
                accepts_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in signature.parameters.values()
                )
                if not accepts_kwargs:
                    assignment_kwargs = {
                        key: value
                        for key, value in assignment_kwargs.items()
                        if key in signature.parameters
                    }
            except (TypeError, ValueError):
                pass
            return vf.get_assignment_q_value(**assignment_kwargs)
        return 0.0

    def batch_evaluate_service_options(self, pairs, ifEVQvalue=False):
        return [self.evaluate_service_option(vid, req, ifEVQvalue) for vid, req in pairs]

    def heuristic_find_nearest_v(self, vehicle_ids):
        assignments = {}
        avail = list(self.active_requests.values())
        for vid in vehicle_ids:
            vloc = self.vehicles[vid]['location']
            best, best_d = None, float('inf')
            for req in avail:
                d = self.get_distance_km(vloc, req.pickup)
                if d < best_d:
                    best_d = d
                    best = req
            if best is not None:
                assignments[vid] = best
                avail.remove(best)
        return assignments

    def save_time_stats(self, file_path=None):
        if file_path is None:
            file_path = 'results/time_analysis/nyc_time_stats.json'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        import json
        with open(file_path, 'w') as f:
            json.dump({k: [float(x) for x in v] for k, v in self.time_stats.items()}, f, indent=2)
