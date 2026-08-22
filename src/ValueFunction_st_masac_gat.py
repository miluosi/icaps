"""ST-ADP-MASAC-MCMF first-pass value function.

This implementation keeps the current MCMF assignment layer untouched.  It
returns an edge-additive score

    g_myopic + beta * clip(w_type(H) * q_res(edge, H))

where ``g_myopic`` is the simple MCMF baseline score and ``q_res`` is a learned
residual value.  The actor/temperature modules are trained for the MASAC-style
objective, but ``eta_pi`` defaults to 0.0 so the actor prior does not perturb
MCMF during the initial debugging runs.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TopKNeighbourContext(nn.Module):
    """Attention-pool the nearest feasible action nodes for each vehicle."""

    def __init__(self, hidden_dim: int, neighbour_number: int):
        super().__init__()
        self.neighbour_number = max(0, int(neighbour_number))
        self.nei_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        vehicle_emb: torch.Tensor,
        neighbour_emb: torch.Tensor,
        neighbour_distances: torch.Tensor,
        neighbour_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vehicle_emb: [B, H] source-zone embeddings.
            neighbour_emb: [B, M, H] feasible action-node embeddings.
            neighbour_distances: [B, M] source-to-node distances in km.
            neighbour_mask: [B, M] valid candidate mask.
        """
        if self.neighbour_number <= 0 or neighbour_emb.shape[1] == 0:
            return torch.zeros_like(vehicle_emb)

        candidate_count = int(neighbour_emb.shape[1])
        k = min(self.neighbour_number, candidate_count)
        masked_distances = neighbour_distances.masked_fill(~neighbour_mask, float("inf"))
        knn_dist, knn_idx = torch.topk(masked_distances, k=k, dim=-1, largest=False)
        gather_idx = knn_idx.unsqueeze(-1).expand(-1, -1, neighbour_emb.shape[-1])
        h_nei = torch.gather(neighbour_emb, dim=1, index=gather_idx)
        valid_knn = torch.gather(neighbour_mask, dim=1, index=knn_idx)

        h_ego = vehicle_emb.unsqueeze(1).expand(-1, k, -1)
        valid_distances = neighbour_distances.masked_fill(~neighbour_mask, 0.0)
        mean_distance = (
            valid_distances.sum(dim=1, keepdim=True)
            / neighbour_mask.sum(dim=1, keepdim=True).clamp_min(1)
        ).detach().clamp_min(1e-6)
        safe_knn_dist = torch.where(valid_knn, knn_dist, torch.zeros_like(knn_dist))
        dist_feature = (safe_knn_dist / mean_distance).unsqueeze(-1)

        logits = self.attn(torch.cat([h_ego, h_nei, dist_feature], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~valid_knn, -1e9)
        alpha = torch.softmax(logits, dim=-1) * valid_knn.to(logits.dtype)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (alpha.unsqueeze(-1) * self.nei_proj(h_nei)).sum(dim=1)


class _GraphAttentionEncoder(nn.Module):
    """All-node graph encoder plus vehicle-specific feasible-neighbour context."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        neighbour_number: int = 5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, heads, batch_first=True) for _ in range(layers)]
        )
        self.neighbour_number = max(0, int(neighbour_number))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.demand_head = nn.Linear(hidden_dim, 1)
        self.queue_head = nn.Linear(hidden_dim, 1)
        self.peak_head = nn.Linear(hidden_dim, 1)
        if self.neighbour_number > 0:
            self.topk_neighbour_context = _TopKNeighbourContext(hidden_dim, self.neighbour_number)
            self.neighbour_norm = nn.LayerNorm(hidden_dim)
        else:
            # Keep neighbour=0 initialization identical to the pre-neighbour model.
            # Constructing unused layers still consumes RNG and changes every later
            # critic/actor weight under the same training seed.
            self.topk_neighbour_context = None
            self.neighbour_norm = None
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.ndim == 2:
            node_features = node_features.unsqueeze(0)
        h = self.input_proj(node_features)
        for attn, norm, ffn, ffn_norm in zip(self.layers, self.norms, self.ffns, self.ffn_norms):
            attn_out, _ = attn(h, h, h, need_weights=False)
            h = norm(h + attn_out)
            h = ffn_norm(h + ffn(h))
        return h.squeeze(0)

    def add_neighbour_context(
        self,
        vehicle_emb: torch.Tensor,
        neighbour_emb: torch.Tensor,
        neighbour_distances: torch.Tensor,
        neighbour_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.topk_neighbour_context is None or self.neighbour_norm is None:
            return vehicle_emb
        context = self.topk_neighbour_context(
            vehicle_emb,
            neighbour_emb,
            neighbour_distances,
            neighbour_mask,
        )
        enriched = self.neighbour_norm(vehicle_emb + context)
        has_neighbour = neighbour_mask.any(dim=1, keepdim=True)
        return torch.where(has_neighbour, enriched, vehicle_emb)


class _Mixer(nn.Module):
    def __init__(self, graph_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.ReLU(),
            nn.Linear(graph_dim, 3),
        )
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(pooled)
        w_ev = F.softplus(out[..., 0]) + 1e-3
        w_aev = F.softplus(out[..., 1]) + 1e-3
        baseline = out[..., 2]
        return w_ev, w_aev, baseline


class PyTorchChargingValueFunction:
    """MCMF-compatible residual ST-GAT + MASAC value provider."""

    uses_post_action_locations = True
    uses_queue_wait_loss = True
    uses_queue_wait_feature = True

    def __init__(
        self,
        grid_size: int = 10,
        num_vehicles: int = 8,
        log_dir: str = "logs/st_masac_gat",
        device: str = "cpu",
        episode_length: int = 300,
        max_requests: int = 1000,
        env=None,
        encoder: bool = False,
        zone_distribution_mode: str | None = None,
        replay_buffer_size: int = 500000,
        iftransformer: bool = False,
        neighbour_number: int = 5,
    ):
        del log_dir, encoder, iftransformer
        self.grid_size = int(grid_size)
        self.num_vehicles = int(num_vehicles)
        self.episode_length = max(1.0, float(episode_length))
        self.max_requests = max(1.0, float(max_requests))
        self.env = env
        self.device = torch.device(device)
        self.zone_distribution_mode = zone_distribution_mode or "st_masac_gat"
        self.neighbour_number = max(0, int(neighbour_number))
        self.freeze_graph_encoder = self.zone_distribution_mode == "st_masac_gat_frozen"
        self.freeze_neighbour_context = (
            self.zone_distribution_mode == "st_masac_gat_neighbour_frozen"
        )

        self.gamma = 0.95
        self.tau = 0.005
        self.learning_rate = 1e-3
        self.beta_max = 0.30
        self.beta_warmup_steps = 500
        self.eta_pi = 0.0
        self.residual_clip_rho = 0.30
        self.lambda_actor = 1.0
        self.lambda_alpha = 1.0
        self.lambda_orth = 0.05
        self.lambda_cql = 0.0
        self.hidden_dim = 96
        self.graph_node_dim = 16
        self.edge_local_dim = 18
        self.edge_dim = self.edge_local_dim + self.hidden_dim * 2
        self.queue_feature_dim = 9

        self.graph_encoder = _GraphAttentionEncoder(
            self.graph_node_dim,
            self.hidden_dim,
            neighbour_number=self.neighbour_number,
        ).to(self.device)
        if self.freeze_graph_encoder:
            self.graph_encoder.requires_grad_(False)
        elif self.freeze_neighbour_context and self.graph_encoder.topk_neighbour_context is not None:
            self.graph_encoder.topk_neighbour_context.requires_grad_(False)
            self.graph_encoder.neighbour_norm.requires_grad_(False)
        self.mixer = _Mixer(self.hidden_dim).to(self.device)
        self.network = _MLP(self.edge_dim, hidden_dim=128).to(self.device)
        self.critic2 = _MLP(self.edge_dim, hidden_dim=128).to(self.device)
        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self.actor = _MLP(self.edge_dim, hidden_dim=128).to(self.device)
        self.queue_predictor = _MLP(self.queue_feature_dim, hidden_dim=64).to(self.device)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.05), dtype=torch.float32, device=self.device))

        params = (
            [parameter for parameter in self.graph_encoder.parameters() if parameter.requires_grad]
            + list(self.mixer.parameters())
            + list(self.network.parameters())
            + list(self.critic2.parameters())
            + list(self.actor.parameters())
            + [self.log_alpha]
        )
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        self.queue_optimizer = optim.Adam(self.queue_predictor.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.queue_loss_fn = nn.MSELoss()
        self.experience_buffer = deque(maxlen=int(replay_buffer_size))
        self.queue_experience_buffer = deque(maxlen=int(replay_buffer_size))
        self.training_losses: list[float] = []
        self.normalized_td_losses: list[float] = []
        self.td_error_history: list[dict[str, float]] = []
        self.rejection_training_losses: list[float] = []
        self.queue_training_losses: list[float] = []
        self.queue_training_mse_losses: list[float] = []
        self.recent_station_waits: dict[int, float] = {}
        self.queue_loss_weight = 1.0
        self.queue_edge_loss_weight = 1.0
        self.queue_predictor_trained = False
        self.q_values_history: list[dict[str, float]] = []
        self.training_step = 0
        self.debug_name = "ADP"
        self._graph_cache_key = None
        self._graph_cache = None
        print(
            f"✓ ST-ADP-MASAC-GAT value function initialized: beta_max={self.beta_max}, "
            f"eta_pi={self.eta_pi}, residual_clip={self.residual_clip_rho}*std(g), "
            f"neighbours={self.neighbour_number}, graph_frozen={self.freeze_graph_encoder}, "
            f"neighbour_frozen={self.freeze_neighbour_context}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Basic feature helpers
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(1e-4, 10.0)

    def _target_entropy(self, candidate_count: int) -> float:
        """Return the legacy entropy target without changing existing runs.

        The historical ST-MASAC modes used a negative categorical entropy
        target.  Keep that behavior here for checkpoint and experiment
        reproducibility; ``standard_masac_gat`` overrides this hook with the
        standard positive discrete-SAC target.
        """
        return -math.log(max(1, int(candidate_count)))

    def _beta(self) -> float:
        if self.beta_warmup_steps <= 0:
            return float(self.beta_max)
        return float(min(self.beta_max, self.beta_max * self.training_step / self.beta_warmup_steps))

    def _zone_norm(self, location: Any) -> float:
        if self.env is not None and hasattr(self.env, "get_distribution_zone_index"):
            try:
                idx = self.env.get_distribution_zone_index(int(location))
                dim = max(1, int(getattr(self.env, "aux_zone_dim", 1) or 1))
                if idx is not None and idx >= 0:
                    return float(idx) / float(max(1, dim - 1))
            except Exception:
                pass
        num_locations = max(1, int(getattr(self.env, "NUM_LOCATIONS", self.grid_size * self.grid_size)))
        return float(np.clip(float(location or 0) / float(max(1, num_locations - 1)), 0.0, 1.0))

    def _zone_index(self, location: Any) -> int:
        if self.env is not None and hasattr(self.env, "get_distribution_zone_index"):
            try:
                idx = self.env.get_distribution_zone_index(int(location))
                if idx is not None and idx >= 0:
                    return int(idx)
            except Exception:
                pass
        return 0

    def _time_norm(self, current_time: float) -> float:
        if self.env is not None and hasattr(self.env, "get_hour_of_day"):
            try:
                return float(self.env.get_hour_of_day(float(current_time))) / 24.0
            except Exception:
                pass
        return float(np.clip(float(current_time) / self.episode_length, 0.0, 1.5))

    def _vehicle_type(self, vehicle_id: int) -> int:
        if self.env is not None and hasattr(self.env, "vehicles") and vehicle_id in self.env.vehicles:
            return int(self.env.vehicles[vehicle_id].get("type", 1))
        return 1

    def _post_battery(self, battery: float, action_type_id: int, distance: float) -> float:
        battery_after = float(battery)
        if self.env is not None:
            battery_after -= float(distance) * float(getattr(self.env, "battery_consum", 0.0))
            if int(action_type_id) == 3:
                battery_after += float(getattr(self.env, "chargeincrease_whole", 0.0))
        return float(np.clip(battery_after, 0.0, 1.0))

    def _action_id(self, action_type: str | int) -> int:
        if isinstance(action_type, (int, np.integer)):
            return int(action_type)
        action = str(action_type)
        if action == "idle" or action == "reloc" or action.startswith("reloc"):
            return 1
        if action.startswith("charge"):
            return 3
        return 2

    def _myopic_score(
        self,
        action_type_id: int,
        request_value: float,
        target_distance: float,
        post_action_distance: float | None = None,
    ) -> float:
        """Return the immediate MCMF score used as the residual-Q baseline.

        Keep this baseline on the same reward scale as ``Environment``.  The
        previous constants made charging and relocation profitable in their
        own right (``5 - .5 d`` and ``2 - .3 d``), even though both actions
        incur costs when executed.  That forced residual Q-learning to first
        learn a large negative correction before it could improve on MCMF.

        Request actions use ``post_action_distance`` (pickup plus passenger
        trip) when available.  Other actions use their target/deadhead
        distance.  This keeps the residual baseline on the same dollar-per-km
        operating-cost definition as execution and the assignment solvers.
        """
        action_type_id = int(action_type_id)
        dist = max(0.0, float(target_distance or 0.0))
        operating_cost_per_km = getattr(
            self.env,
            "operating_cost_per_km",
            None,
        )
        if operating_cost_per_km is None:
            moving_penalty = float(getattr(self.env, "movingpenalty", -5e-3))
        else:
            moving_penalty = -abs(float(operating_cost_per_km))
        charging_penalty = float(getattr(self.env, "charging_penalty", 0.25))
        if action_type_id == 2:
            request_distance = (
                dist
                if post_action_distance is None
                else max(0.0, float(post_action_distance or 0.0))
            )
            return float(request_value or 0.0) + moving_penalty * request_distance
        if action_type_id == 3:
            return moving_penalty * dist - charging_penalty
        if dist <= 1e-9:
            return moving_penalty
        return moving_penalty * max(1.0, dist)

    def _state_features(
        self,
        *,
        location,
        current_time,
        battery,
        idle_time=0.0,
        other_vehicles=0.0,
        num_requests=0.0,
        vehicle_type=1,
    ) -> list[float]:
        vehicle_type = int(vehicle_type or 1)
        time_norm = self._time_norm(float(current_time))
        hour = time_norm * 2.0 * math.pi
        return [
            self._zone_norm(location),
            time_norm,
            math.sin(hour),
            math.cos(hour),
            float(np.clip(battery, 0.0, 1.2)),
            float(np.clip(float(idle_time) / max(self.episode_length, 1.0), 0.0, 1.0)),
            float(np.clip(float(other_vehicles) / max(float(self.num_vehicles), 1.0), 0.0, 2.0)),
            float(np.clip(float(num_requests) / max(float(self.max_requests), 1.0), 0.0, 2.0)),
            1.0 if vehicle_type == 1 else 0.0,
            1.0 if vehicle_type == 2 else 0.0,
        ]

    def _local_edge_features(
        self,
        state_features: list[float],
        *,
        action_type_id: int,
        target_distance: float,
        post_action_duration: float,
        post_action_distance: float,
        battery_level: float,
        vehicle_type: int,
        queue_wait_feature: float = 0.0,
    ) -> list[float]:
        action_type_id = int(action_type_id)
        post_soc = self._post_battery(float(battery_level), action_type_id, float(post_action_distance))
        return state_features + [
            1.0 if action_type_id == 1 else 0.0,
            1.0 if action_type_id == 2 else 0.0,
            1.0 if action_type_id == 3 else 0.0,
            float(np.clip(float(target_distance) / 30.0, 0.0, 5.0)),
            float(np.clip(float(post_action_distance) / 50.0, 0.0, 5.0)),
            float(np.clip(float(post_action_duration) / max(self.episode_length, 1.0), 0.0, 2.0)),
            post_soc,
            1.0 + float(np.clip(float(queue_wait_feature), 0.0, 4.0)),
        ]

    def _resolve_station(self, station_id: Any = None, target_location: Any = None):
        stations = getattr(getattr(self.env, "charging_manager", None), "stations", {}) if self.env is not None else {}
        if station_id is not None:
            try:
                station = stations.get(int(station_id))
                if station is not None:
                    return station
            except Exception:
                pass
        if target_location is not None:
            try:
                target_location = int(target_location)
                for station in stations.values():
                    if int(getattr(station, "location", -1)) == target_location:
                        return station
            except Exception:
                pass
        return None

    def _distance_to_station(self, source_location: Any, station) -> float:
        if station is None:
            return 0.0
        target_location = int(getattr(station, "location", source_location or 0))
        source_location = int(source_location if source_location is not None else target_location)
        if self.env is not None and hasattr(self.env, "get_distance_km"):
            try:
                return float(self.env.get_distance_km(source_location, target_location))
            except Exception:
                pass
        return abs(float(source_location) - float(target_location))

    def _queue_features(
        self,
        *,
        station_id: Any = None,
        target_location: Any = None,
        vehicle_id: Any = None,
        vehicle_location: Any = None,
        current_time: float = 0.0,
        num_requests: float = 0.0,
        travel_duration: float = 0.0,
    ) -> list[float]:
        del vehicle_id
        station = self._resolve_station(station_id, target_location)
        num_vehicles = max(1.0, float(self.num_vehicles))
        max_requests = max(1.0, float(self.max_requests))
        episode_length = max(1.0, float(self.episode_length))
        if station is None:
            return [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                float(np.clip(float(num_requests) / max_requests, 0.0, 2.0)),
                self._time_norm(float(current_time)),
                0.0,
                float(np.clip(float(travel_duration) / episode_length, 0.0, 2.0)),
            ]

        cap = max(1.0, float(getattr(station, "max_capacity", 1) or 1))
        queue_len = float(len(getattr(station, "charging_queue", []) or []))
        occupied = float(len(getattr(station, "current_vehicles", []) or []))
        inbound = float(len(getattr(station, "charging_queue_notarrived", []) or []))
        recent_wait = float(self.recent_station_waits.get(int(getattr(station, "id", 0)), 0.0))

        low_soc_near = 0.0
        if self.env is not None and hasattr(self.env, "vehicles"):
            threshold = float(getattr(self.env, "min_battery_level", 0.2)) + 0.15
            radius = float(getattr(self.env, "charge_action_range_km", getattr(self.env, "chargeassignnum", 5)))
            for vehicle in getattr(self.env, "vehicles", {}).values():
                try:
                    if float(vehicle.get("battery", 1.0)) > threshold:
                        continue
                    if vehicle.get("charging_station") is not None:
                        continue
                    dist = self._distance_to_station(vehicle.get("location", target_location), station)
                    if dist <= radius:
                        low_soc_near += 1.0
                except Exception:
                    continue

        return [
            float(np.clip(queue_len / num_vehicles, 0.0, 2.0)),
            float(np.clip(occupied / cap, 0.0, 2.0)),
            float(np.clip(cap / max(num_vehicles, cap), 0.0, 1.0)),
            float(np.clip(inbound / num_vehicles, 0.0, 2.0)),
            float(np.clip(low_soc_near / num_vehicles, 0.0, 2.0)),
            float(np.clip(float(num_requests) / max_requests, 0.0, 2.0)),
            self._time_norm(float(current_time) + float(travel_duration)),
            float(np.clip(recent_wait / episode_length, 0.0, 2.0)),
            float(np.clip(float(travel_duration) / episode_length, 0.0, 2.0)),
        ]

    def store_queue_experience(self, **kwargs):
        observed_wait = float(kwargs.get("observed_wait", kwargs.get("wait_time", 0.0)) or 0.0)
        station_id = kwargs.get("station_id", None)
        features = kwargs.get("features")
        if features is None:
            features = self._queue_features(
                station_id=station_id,
                target_location=kwargs.get("target_location"),
                vehicle_id=kwargs.get("vehicle_id"),
                vehicle_location=kwargs.get("vehicle_location"),
                current_time=float(kwargs.get("current_time", 0.0) or 0.0),
                num_requests=float(kwargs.get("num_requests", 0.0) or 0.0),
                travel_duration=float(kwargs.get("travel_duration", 0.0) or 0.0),
            )
        self.queue_experience_buffer.append({
            "features": [float(x) for x in features],
            "observed_wait": max(0.0, observed_wait),
            "station_id": int(station_id) if station_id is not None else -1,
        })
        if station_id is not None:
            self.recent_station_waits[int(station_id)] = max(0.0, observed_wait)

    def _queue_loss_from_samples(self, batch_size: int) -> torch.Tensor | None:
        if len(self.queue_experience_buffer) < 4:
            return None
        batch = random.sample(
            list(self.queue_experience_buffer),
            min(max(4, batch_size // 2), len(self.queue_experience_buffer)),
        )
        rows = [sample["features"] for sample in batch]
        labels = [[float(sample["observed_wait"])] for sample in batch]
        preds = self.queue_predictor(torch.tensor(rows, dtype=torch.float32, device=self.device))
        target = torch.tensor(labels, dtype=torch.float32, device=self.device)
        return self.queue_loss_fn(preds, target)

    def train_queue_predictor(self, batch_size: int = 64) -> float:
        queue_loss = self._queue_loss_from_samples(batch_size)
        if queue_loss is None:
            return 0.0
        self.queue_optimizer.zero_grad()
        queue_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.queue_predictor.parameters(), max_norm=10.0)
        self.queue_optimizer.step()
        self.queue_predictor_trained = True
        loss_value = float(queue_loss.item())
        self.queue_training_losses.append(loss_value)
        self.queue_training_mse_losses.append(loss_value)
        return loss_value

    def predict_queue_waits(
        self,
        *,
        station_ids=None,
        target_locations=None,
        vehicle_ids=None,
        vehicle_locations=None,
        current_times=None,
        num_requests=None,
        travel_durations=None,
    ) -> np.ndarray:
        size = 0
        for seq in (station_ids, target_locations, vehicle_ids, vehicle_locations, current_times):
            if seq is not None:
                size = len(seq)
                break
        if size == 0:
            return np.zeros(0, dtype=np.float32)
        if not self.queue_predictor_trained:
            return np.zeros(size, dtype=np.float32)

        def _at(seq, idx, default=None):
            if seq is None:
                return default
            return seq[idx]

        rows = []
        for idx in range(size):
            rows.append(
                self._queue_features(
                    station_id=_at(station_ids, idx, None),
                    target_location=_at(target_locations, idx, None),
                    vehicle_id=_at(vehicle_ids, idx, None),
                    vehicle_location=_at(vehicle_locations, idx, None),
                    current_time=float(_at(current_times, idx, 0.0) or 0.0),
                    num_requests=float(_at(num_requests, idx, 0.0) or 0.0),
                    travel_duration=float(_at(travel_durations, idx, 0.0) or 0.0),
                )
            )
        with torch.no_grad():
            waits = self.queue_predictor(torch.tensor(rows, dtype=torch.float32, device=self.device)).squeeze(1)
            waits = torch.relu(waits)
        return waits.cpu().numpy().astype(np.float32)

    def _normalize_queue_wait(self, wait_steps: float | np.ndarray) -> float | np.ndarray:
        denom = max(1.0, float(getattr(self.env, "charge_duration", 1.0) if self.env is not None else 1.0))
        return np.clip(np.asarray(wait_steps, dtype=np.float32) / denom, 0.0, 5.0)

    def _station_id_from_action_type(self, action_type: Any) -> int | None:
        text = str(action_type or "")
        if not text.startswith("charge_"):
            return None
        try:
            return int(text.split("_", 1)[1])
        except (TypeError, ValueError, IndexError):
            return None

    def _queue_wait_from_feature_snapshot(self, features: Any) -> float | None:
        if features is None or not self.queue_predictor_trained:
            return None
        try:
            row = [float(x) for x in features]
        except (TypeError, ValueError):
            return None
        if len(row) != self.queue_feature_dim:
            return None
        with torch.no_grad():
            tensor = torch.tensor([row], dtype=torch.float32, device=self.device)
            wait = self.queue_predictor(tensor).squeeze(1)
            wait = torch.relu(wait).item()
        return float(wait)

    def _queue_wait_features_for_edges(
        self,
        *,
        action_type_ids,
        target_locations,
        vehicle_ids,
        vehicle_locations,
        current_times,
        num_requests,
        post_action_durations,
        target_station_ids=None,
    ) -> np.ndarray:
        action_type_ids = np.asarray(action_type_ids, dtype=np.int64)
        features = np.zeros(action_type_ids.shape[0], dtype=np.float32)
        if action_type_ids.size == 0 or not np.any(action_type_ids == 3):
            return features
        if not self.queue_predictor_trained:
            return features
        charge_idx = np.flatnonzero(action_type_ids == 3)
        charge_duration = float(getattr(self.env, "charge_duration", 0.0)) if self.env is not None else 0.0
        post_durations = np.asarray(post_action_durations, dtype=np.float32)
        travel_durations = np.maximum(0.0, post_durations[charge_idx] - charge_duration)
        station_ids = None
        if target_station_ids is not None:
            target_station_ids = np.asarray(target_station_ids, dtype=np.int64)
            station_ids = target_station_ids[charge_idx]
        waits = self.predict_queue_waits(
            station_ids=station_ids,
            target_locations=np.asarray(target_locations)[charge_idx],
            vehicle_ids=np.asarray(vehicle_ids)[charge_idx],
            vehicle_locations=np.asarray(vehicle_locations)[charge_idx],
            current_times=np.asarray(current_times, dtype=np.float32)[charge_idx],
            num_requests=np.asarray(num_requests, dtype=np.float32)[charge_idx],
            travel_durations=travel_durations,
        )
        features[charge_idx] = self._normalize_queue_wait(waits).astype(np.float32)
        return features

    def _queue_wait_feature_from_experience(self, exp: dict, candidate: dict | None = None) -> float:
        source = exp if candidate is None else candidate
        action_type = source.get("action_type", exp.get("action_type", "idle"))
        action_id = self._action_id(action_type)
        if action_id != 3:
            return 0.0
        if "queue_wait_feature" in source:
            try:
                return float(np.clip(float(source["queue_wait_feature"]), 0.0, 5.0))
            except (TypeError, ValueError):
                pass
        wait = self._queue_wait_from_feature_snapshot(source.get("queue_features"))
        if wait is not None:
            return float(self._normalize_queue_wait(wait))
        for key in ("predicted_queue_wait", "observed_queue_wait", "observed_wait", "wait_time"):
            if key in source:
                try:
                    return float(self._normalize_queue_wait(float(source[key])))
                except (TypeError, ValueError):
                    pass

        station_id = source.get("target_station_id", source.get("station_id"))
        if station_id is None:
            station_id = self._station_id_from_action_type(action_type)
        current_time = float(exp.get("current_time", 0.0) or 0.0)
        vehicle_location = exp.get("vehicle_location", 0)
        if candidate is not None:
            current_time += float(exp.get("dur_time", 1.0) or 1.0)
            vehicle_location = exp.get("next_vehicle_location", vehicle_location)
        post_duration = float(source.get("post_action_duration", source.get("dur_time", 0.0)) or 0.0)
        charge_duration = float(getattr(self.env, "charge_duration", 0.0)) if self.env is not None else 0.0
        travel_duration = max(0.0, post_duration - charge_duration)
        waits = self.predict_queue_waits(
            station_ids=[station_id] if station_id is not None else None,
            target_locations=[source.get("target_location", source.get("post_action_location", vehicle_location))],
            vehicle_ids=[int(exp.get("vehicle_id", -1))],
            vehicle_locations=[vehicle_location],
            current_times=[current_time],
            num_requests=[float(source.get("num_requests", exp.get("num_requests", 0.0)) or 0.0)],
            travel_durations=[travel_duration],
        )
        if waits.size == 0:
            return 0.0
        return float(self._normalize_queue_wait(float(waits[0])))

    def _queue_penalty_per_step(self) -> float:
        if self.env is not None:
            return float(getattr(self.env, "learning_wait_penalty", getattr(self.env, "charging_wait_penalty_per_step", 0.5)))
        return 0.5

    def _queue_edge_losses(
        self,
        *,
        action_type_ids,
        target_locations,
        vehicle_ids,
        vehicle_locations,
        current_times,
        num_requests,
        post_action_durations,
        target_station_ids=None,
    ) -> np.ndarray:
        action_type_ids = np.asarray(action_type_ids, dtype=np.int64)
        losses = np.zeros(action_type_ids.shape[0], dtype=np.float32)
        if action_type_ids.size == 0 or not np.any(action_type_ids == 3):
            return losses
        if not self.queue_predictor_trained:
            return losses
        charge_idx = np.flatnonzero(action_type_ids == 3)
        charge_duration = float(getattr(self.env, "charge_duration", 0.0)) if self.env is not None else 0.0
        post_durations = np.asarray(post_action_durations, dtype=np.float32)
        travel_durations = np.maximum(0.0, post_durations[charge_idx] - charge_duration)
        station_ids = None
        if target_station_ids is not None:
            target_station_ids = np.asarray(target_station_ids, dtype=np.int64)
            station_ids = target_station_ids[charge_idx]
        waits = self.predict_queue_waits(
            station_ids=station_ids,
            target_locations=np.asarray(target_locations)[charge_idx],
            vehicle_ids=np.asarray(vehicle_ids)[charge_idx],
            vehicle_locations=np.asarray(vehicle_locations)[charge_idx],
            current_times=np.asarray(current_times, dtype=np.float32)[charge_idx],
            num_requests=np.asarray(num_requests, dtype=np.float32)[charge_idx],
            travel_durations=travel_durations,
        )
        penalty_per_step = self._queue_penalty_per_step()
        losses[charge_idx] = (
            np.maximum(0.0, waits)
            * float(self.queue_edge_loss_weight)
            * float(penalty_per_step)
        )
        return losses

    # ------------------------------------------------------------------
    # Resource graph snapshot
    # ------------------------------------------------------------------

    def _graph_locations(self) -> list[int]:
        if self.env is None:
            return list(range(max(1, self.grid_size * self.grid_size)))
        zones = list(getattr(self.env, "aux_zone_ids", []) or getattr(self.env, "relocation_target_ids", []) or [])
        if not zones:
            zones = sorted(getattr(self.env, "zone_coords", {}).keys())
        return [int(z) for z in zones[: max(1, len(zones))]]

    def _build_graph_node_features(self) -> tuple[torch.Tensor, dict[int, int], dict[int, int], int]:
        zones = self._graph_locations()
        zone_to_row = {int(z): idx for idx, z in enumerate(zones)}
        rows: list[list[float]] = []
        current_time = float(getattr(self.env, "current_time", 0.0) if self.env is not None else 0.0)
        time_norm = self._time_norm(current_time)
        hour_angle = time_norm * 2.0 * math.pi
        active_requests = list(getattr(self.env, "active_requests", {}).values()) if self.env is not None else []
        vehicles = getattr(self.env, "vehicles", {}) if self.env is not None else {}

        demand = {z: 0.0 for z in zones}
        for req in active_requests:
            zid = int(getattr(req, "pickup", 0))
            if zid in demand:
                demand[zid] += 1.0

        ev_count = {z: 0.0 for z in zones}
        aev_count = {z: 0.0 for z in zones}
        ev_soc_sum = {z: 0.0 for z in zones}
        aev_soc_sum = {z: 0.0 for z in zones}
        low_soc = {z: 0.0 for z in zones}
        low_thr = float(getattr(self.env, "heuristic_battery_threshold", 0.3) if self.env is not None else 0.3)
        for vehicle in vehicles.values():
            if not vehicle.get("is_online", True):
                continue
            loc = int(vehicle.get("location", 0))
            if loc not in zone_to_row:
                continue
            battery = float(vehicle.get("battery", 1.0))
            if int(vehicle.get("type", 1)) == 1:
                ev_count[loc] += 1.0
                ev_soc_sum[loc] += battery
            else:
                aev_count[loc] += 1.0
                aev_soc_sum[loc] += battery
            if battery <= low_thr:
                low_soc[loc] += 1.0

        for idx, zone in enumerate(zones):
            ev_n = ev_count[zone]
            aev_n = aev_count[zone]
            ev_soc = ev_soc_sum[zone] / max(1.0, ev_n)
            aev_soc = aev_soc_sum[zone] / max(1.0, aev_n)
            rows.append([
                float(idx) / float(max(1, len(zones) - 1)),
                float(np.clip(demand[zone] / 100.0, 0.0, 5.0)),
                float(np.clip(ev_n / max(1.0, self.num_vehicles), 0.0, 2.0)),
                float(np.clip(aev_n / max(1.0, self.num_vehicles), 0.0, 2.0)),
                ev_soc,
                aev_soc,
                float(np.clip(low_soc[zone] / max(1.0, self.num_vehicles), 0.0, 2.0)),
                0.0,
                time_norm,
                math.sin(hour_angle),
                math.cos(hour_angle),
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ])

        station_to_row: dict[int, int] = {}
        stations = getattr(getattr(self.env, "charging_manager", None), "stations", {}) if self.env is not None else {}
        for sid, station in sorted(stations.items()):
            station_to_row[int(sid)] = len(rows)
            capacity = float(getattr(station, "max_capacity", 1) or 1)
            current = float(len(getattr(station, "current_vehicles", []) or []))
            queue = float(len(getattr(station, "charging_queue_notarrived", []) or []))
            loc = int(getattr(station, "location", 0))
            near_demand = demand.get(loc, 0.0)
            rows.append([
                self._zone_norm(loc),
                float(np.clip(near_demand / 100.0, 0.0, 5.0)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                float(np.clip(queue / max(1.0, capacity), 0.0, 5.0)),
                time_norm,
                math.sin(hour_angle),
                math.cos(hour_angle),
                0.0,
                1.0,
                float(np.clip(capacity / max(1.0, self.num_vehicles), 0.0, 2.0)),
                float(np.clip((current + queue) / max(1.0, capacity), 0.0, 5.0)),
                1.0,
            ])

        global_row = len(rows)
        rows.append([
            0.0,
            float(np.clip(len(active_requests) / max(1.0, self.max_requests), 0.0, 5.0)),
            float(np.clip(sum(ev_count.values()) / max(1.0, self.num_vehicles), 0.0, 2.0)),
            float(np.clip(sum(aev_count.values()) / max(1.0, self.num_vehicles), 0.0, 2.0)),
            0.0,
            0.0,
            float(np.clip(sum(low_soc.values()) / max(1.0, self.num_vehicles), 0.0, 2.0)),
            0.0,
            time_norm,
            math.sin(hour_angle),
            math.cos(hour_angle),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ])
        tensor = torch.tensor(rows, dtype=torch.float32, device=self.device)
        return tensor, zone_to_row, station_to_row, global_row

    def _graph_context(self):
        key = (
            id(self.env),
            int(getattr(self.env, "current_time", 0) if self.env is not None else 0),
            len(getattr(self.env, "active_requests", {}) if self.env is not None else {}),
        )
        if self._graph_cache_key == key and self._graph_cache is not None:
            return self._graph_cache
        node_features, zone_to_row, station_to_row, global_row = self._build_graph_node_features()
        embeddings = self.graph_encoder(node_features)
        pooled = embeddings.mean(dim=0)
        w_ev, w_aev, baseline = self.mixer(pooled)
        self._graph_cache_key = key
        self._graph_cache = {
            "embeddings": embeddings,
            "zone_to_row": zone_to_row,
            "station_to_row": station_to_row,
            "global_row": global_row,
            "pooled": pooled,
            "w_ev": w_ev,
            "w_aev": w_aev,
            "baseline": baseline,
        }
        return self._graph_cache

    def _graph_row_for_location(self, graph: dict, location: Any) -> int:
        loc = int(location or 0)
        row = graph["zone_to_row"].get(loc)
        if row is None:
            idx = self._zone_index(loc)
            zones = list(graph["zone_to_row"].values())
            row = zones[idx] if zones and 0 <= idx < len(zones) else graph["global_row"]
        return int(row)

    def _graph_embedding_for_location(self, graph: dict, location: Any) -> torch.Tensor:
        return graph["embeddings"][self._graph_row_for_location(graph, location)]

    def _graph_row_for_neighbour(self, graph: dict, candidate: dict) -> int:
        if candidate.get("node_type") == "station":
            try:
                station_row = graph["station_to_row"].get(int(candidate.get("node_id", -1)))
            except (TypeError, ValueError):
                station_row = None
            if station_row is not None:
                return int(station_row)
        location = candidate.get("target_location", candidate.get("node_id", 0))
        return self._graph_row_for_location(graph, location)

    def _vehicle_source_embeddings(
        self,
        graph: dict,
        vehicle_ids,
        vehicle_locations,
        vehicle_neighbour_candidates: dict[int, list[dict]] | None,
    ) -> dict[int, torch.Tensor]:
        unique_vehicle_locations: dict[int, int] = {}
        for vehicle_id, location in zip(vehicle_ids, vehicle_locations):
            unique_vehicle_locations.setdefault(int(vehicle_id), int(location))

        source_embeddings = {
            vehicle_id: self._graph_embedding_for_location(graph, location)
            for vehicle_id, location in unique_vehicle_locations.items()
        }
        if not vehicle_neighbour_candidates or self.neighbour_number <= 0:
            return source_embeddings

        vehicle_order = list(unique_vehicle_locations)
        candidate_embeddings: list[torch.Tensor] = []
        candidate_distances: list[torch.Tensor] = []
        max_candidates = 0
        for vehicle_id in vehicle_order:
            candidates = list(vehicle_neighbour_candidates.get(vehicle_id, []) or [])
            rows = []
            distances = []
            seen_rows = set()
            for candidate in candidates:
                row = self._graph_row_for_neighbour(graph, candidate)
                if row in seen_rows:
                    continue
                seen_rows.add(row)
                rows.append(graph["embeddings"][row])
                distances.append(max(0.0, float(candidate.get("distance", 0.0) or 0.0)))
            if rows:
                candidate_embeddings.append(torch.stack(rows))
                candidate_distances.append(torch.tensor(distances, dtype=torch.float32, device=self.device))
                max_candidates = max(max_candidates, len(rows))
            else:
                candidate_embeddings.append(graph["embeddings"].new_zeros((0, self.hidden_dim)))
                candidate_distances.append(torch.zeros(0, dtype=torch.float32, device=self.device))

        if max_candidates == 0:
            return source_embeddings

        padded_embeddings = []
        padded_distances = []
        masks = []
        for embeddings, distances in zip(candidate_embeddings, candidate_distances):
            pad_size = max_candidates - int(embeddings.shape[0])
            padded_embeddings.append(F.pad(embeddings, (0, 0, 0, pad_size)))
            padded_distances.append(F.pad(distances, (0, pad_size)))
            masks.append(
                torch.arange(max_candidates, device=self.device) < int(embeddings.shape[0])
            )

        source_batch = torch.stack([source_embeddings[vehicle_id] for vehicle_id in vehicle_order])
        enriched_batch = self.graph_encoder.add_neighbour_context(
            source_batch,
            torch.stack(padded_embeddings),
            torch.stack(padded_distances),
            torch.stack(masks),
        )
        return {
            vehicle_id: enriched_batch[index]
            for index, vehicle_id in enumerate(vehicle_order)
        }

    def _edge_tensor_from_arrays(
        self,
        *,
        vehicle_ids,
        vehicle_locations,
        target_locations,
        current_times,
        other_vehicles,
        num_requests,
        battery_levels,
        target_distances,
        vehicle_idle_times,
        action_type_ids,
        post_action_distances,
        post_action_durations,
        post_action_locations,
        target_station_ids=None,
        queue_wait_features=None,
        vehicle_neighbour_candidates: dict[int, list[dict]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph = self._graph_context()
        source_embeddings = self._vehicle_source_embeddings(
            graph,
            vehicle_ids,
            vehicle_locations,
            vehicle_neighbour_candidates,
        )
        if queue_wait_features is None:
            queue_wait_features = self._queue_wait_features_for_edges(
                action_type_ids=action_type_ids,
                target_locations=target_locations,
                vehicle_ids=vehicle_ids,
                vehicle_locations=vehicle_locations,
                current_times=current_times,
                num_requests=num_requests,
                post_action_durations=post_action_durations,
                target_station_ids=target_station_ids,
            )
        else:
            queue_wait_features = np.asarray(queue_wait_features, dtype=np.float32)
        rows = []
        type_weights = []
        for i in range(len(vehicle_ids)):
            vehicle_type = self._vehicle_type(int(vehicle_ids[i]))
            state = self._state_features(
                location=int(vehicle_locations[i]),
                current_time=float(current_times[i]),
                battery=float(battery_levels[i]),
                idle_time=float(vehicle_idle_times[i]),
                other_vehicles=float(other_vehicles[i]),
                num_requests=float(num_requests[i]),
                vehicle_type=vehicle_type,
            )
            local = self._local_edge_features(
                state,
                action_type_id=int(action_type_ids[i]),
                target_distance=float(target_distances[i]),
                post_action_duration=float(post_action_durations[i]),
                post_action_distance=float(post_action_distances[i]),
                battery_level=float(battery_levels[i]),
                vehicle_type=vehicle_type,
                queue_wait_feature=float(queue_wait_features[i]),
            )
            source_h = source_embeddings[int(vehicle_ids[i])]
            target_h = self._graph_embedding_for_location(graph, int(post_action_locations[i]))
            local_t = torch.tensor(local, dtype=torch.float32, device=self.device)
            rows.append(torch.cat([local_t, source_h, target_h], dim=0))
            type_weights.append(graph["w_ev"] if vehicle_type == 1 else graph["w_aev"])
        return torch.stack(rows), torch.stack(type_weights).unsqueeze(1), graph["baseline"]

    # ------------------------------------------------------------------
    # Inference for MCMF edge scores
    # ------------------------------------------------------------------

    def _execution_scores_from_residual(
        self,
        g_t: torch.Tensor,
        residual: torch.Tensor,
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Compose the legacy bounded residual score used by assignment.

        Subclasses can override this hook without duplicating the vectorized
        edge/feature construction in :meth:`batch_get_mixed_q_values`.
        """
        bounded_residual = torch.maximum(
            torch.minimum(residual, bounds),
            -bounds,
        )
        return g_t + self._beta() * bounded_residual

    def _selection_residual(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        type_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return the residual used by the deployed assignment oracle.

        Historical modes keep their conservative per-edge minimum.  A
        solver-consistent residual mode can override this small hook to use
        the online-critic mean for action selection while retaining clipped
        double evaluation for Bellman targets.
        """
        return torch.minimum(q1, q2) * type_weights

    def batch_get_mixed_q_values(
        self,
        *,
        vehicle_ids,
        vehicle_locations,
        target_locations,
        current_times,
        other_vehicles,
        num_requests,
        battery_levels,
        request_values,
        target_distances,
        target_zoneids,
        vehicle_idle_times,
        action_type_ids,
        post_action_distances=None,
        post_action_durations=None,
        post_action_zoneids=None,
        post_action_locations=None,
        target_station_ids=None,
        vehicle_neighbour_candidates: dict[int, list[dict]] | None = None,
    ):
        del target_zoneids, post_action_zoneids
        size = len(vehicle_ids)
        if size == 0:
            return []
        myopic_action_distances = (
            np.asarray(target_distances, dtype=np.float32)
            if post_action_distances is None
            else np.asarray(post_action_distances, dtype=np.float32)
        )
        post_action_distances = np.zeros(size, dtype=np.float32) if post_action_distances is None else np.asarray(post_action_distances, dtype=np.float32)
        post_action_durations = np.zeros(size, dtype=np.float32) if post_action_durations is None else np.asarray(post_action_durations, dtype=np.float32)
        post_action_locations = np.asarray(target_locations if post_action_locations is None else post_action_locations, dtype=np.int64)
        action_type_ids = np.asarray(action_type_ids, dtype=np.int64)
        vehicle_ids = np.asarray(vehicle_ids)
        vehicle_locations = np.asarray(vehicle_locations)
        target_locations = np.asarray(target_locations)
        current_times = np.asarray(current_times, dtype=np.float32)
        other_vehicles = np.asarray(other_vehicles)
        num_requests = np.asarray(num_requests, dtype=np.float32)
        battery_levels = np.asarray(battery_levels)
        vehicle_idle_times = np.asarray(vehicle_idle_times)
        target_station_ids = (
            np.full(size, -1, dtype=np.int64)
            if target_station_ids is None
            else np.asarray(target_station_ids, dtype=np.int64)
        )
        request_values = np.asarray(request_values, dtype=np.float32)
        target_distances = np.asarray(target_distances, dtype=np.float32)
        queue_wait_features = self._queue_wait_features_for_edges(
            action_type_ids=action_type_ids,
            target_locations=target_locations,
            vehicle_ids=vehicle_ids,
            vehicle_locations=vehicle_locations,
            current_times=current_times,
            num_requests=num_requests,
            post_action_durations=post_action_durations,
            target_station_ids=target_station_ids,
        )
        g = np.asarray(
            [
                self._myopic_score(
                    action_type_ids[i],
                    request_values[i],
                    target_distances[i],
                    myopic_action_distances[i],
                )
                for i in range(size)
            ],
            dtype=np.float32,
        )
        with torch.no_grad():
            edge_t, type_w, _ = self._edge_tensor_from_arrays(
                vehicle_ids=vehicle_ids,
                vehicle_locations=vehicle_locations,
                target_locations=target_locations,
                current_times=current_times,
                other_vehicles=other_vehicles,
                num_requests=num_requests,
                battery_levels=battery_levels,
                target_distances=target_distances,
                vehicle_idle_times=vehicle_idle_times,
                action_type_ids=action_type_ids,
                post_action_distances=post_action_distances,
                post_action_durations=post_action_durations,
                post_action_locations=post_action_locations,
                target_station_ids=target_station_ids,
                queue_wait_features=queue_wait_features,
                vehicle_neighbour_candidates=vehicle_neighbour_candidates,
            )
            q1 = self.network(edge_t)
            q2 = self.critic2(edge_t)
            residual = self._selection_residual(q1, q2, type_w)
            g_t = torch.tensor(g, dtype=torch.float32, device=self.device).unsqueeze(1)
            sigma_g = torch.std(g_t, unbiased=False).clamp_min(1.0)
            base_bound = float(self.residual_clip_rho) * sigma_g
            bounds = torch.full_like(residual, float(base_bound.item()))
            charge_mask = torch.tensor(
                action_type_ids == 3,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(1)
            if torch.any(charge_mask):
                charge_duration = max(
                    1.0,
                    float(getattr(self.env, "charge_duration", 1.0) if self.env is not None else 1.0),
                )
                charging_cost_per_step = float(
                    getattr(
                        self.env,
                        "charging_penalty_per_step",
                        getattr(self.env, "charging_penalty", 0.0),
                    ) if self.env is not None else 0.0
                )
                wait_cost_per_step = float(self._queue_penalty_per_step())
                predicted_wait_steps = torch.tensor(
                    queue_wait_features,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(1) * charge_duration
                charge_cost_bound = (
                    charging_cost_per_step * charge_duration
                    + wait_cost_per_step * predicted_wait_steps
                )
                bounds = torch.where(
                    charge_mask,
                    torch.maximum(bounds, charge_cost_bound),
                    bounds,
                )
            scores = self._execution_scores_from_residual(
                g_t,
                residual,
                bounds,
            )
            if self.eta_pi != 0.0:
                scores = scores + float(self.eta_pi) * F.logsigmoid(self.actor(edge_t))
        values = scores.squeeze(1).cpu().numpy()
        self._last_adp_score_stats = {
            "mode": self.zone_distribution_mode,
            "g_mean": float(np.mean(g)),
            "beta": float(self._beta()),
            "score_mean": float(np.mean(values)),
            "queue_wait_feature_mean": float(np.mean(queue_wait_features)) if queue_wait_features.size else 0.0,
            "neighbour_count_mean": (
                float(np.mean([
                    len(vehicle_neighbour_candidates.get(int(vehicle_id), []) or [])
                    for vehicle_id in dict.fromkeys(vehicle_ids.tolist())
                ]))
                if vehicle_neighbour_candidates
                else 0.0
            ),
        }
        return values.astype(np.float32).tolist()

    def get_assignment_q_value(self, vehicle_id: int, target_id: int, vehicle_location: int, target_reject=None,
                               target_location: int | None = None, current_time: float = 0.0,
                               other_vehicles: int = 0, num_requests: int = 0, battery_level: float = 1.0,
                               request_value: float = 0.0, pickup_dist: float | None = None,
                               pick_zone: int | None = None) -> float:
        del target_reject, pick_zone
        target_location = vehicle_location if target_location is None else target_location
        pickup_distance = float(pickup_dist or 0.0)
        post_action_location = target_location
        post_action_distance = pickup_distance
        post_action_duration = 1.0
        request = (
            getattr(self.env, "active_requests", {}).get(target_id)
            if self.env is not None
            else None
        )
        if request is not None:
            post_action_location = self._location_or_fallback(
                getattr(request, "dropoff", target_location),
                target_location,
            )
            if hasattr(self.env, "_request_trip_distance_km"):
                trip_distance = float(self.env._request_trip_distance_km(request))
            elif getattr(request, "trip_distance_km", None) is not None:
                trip_distance = float(request.trip_distance_km)
            elif hasattr(self.env, "get_distance_km"):
                trip_distance = float(
                    self.env.get_distance_km(
                        int(target_location),
                        int(post_action_location),
                    )
                )
            else:
                trip_distance = 0.0
            post_action_distance += trip_distance
            post_action_duration = max(
                1.0,
                float(getattr(request, "travel_time", 1.0) or 1.0),
            )
        out = self.batch_get_mixed_q_values(
            vehicle_ids=[vehicle_id],
            vehicle_locations=[vehicle_location],
            target_locations=[target_location],
            current_times=[current_time],
            other_vehicles=[other_vehicles],
            num_requests=[num_requests],
            battery_levels=[battery_level],
            request_values=[request_value],
            target_distances=[pickup_distance],
            target_zoneids=[0],
            vehicle_idle_times=[0.0],
            action_type_ids=[2],
            post_action_distances=[post_action_distance],
            post_action_durations=[post_action_duration],
            post_action_zoneids=[0],
            post_action_locations=[post_action_location],
        )
        return float(out[0])

    def _batch_get_action_q_values(
        self,
        batch_inputs: list[dict[str, Any]],
        action_type_id: int,
    ) -> list[float]:
        """Adapt the environment's legacy batch API to the unified edge critic."""
        if not batch_inputs:
            return []

        vehicle_ids = []
        vehicle_locations = []
        target_locations = []
        current_times = []
        other_vehicles = []
        num_requests = []
        battery_levels = []
        request_values = []
        target_distances = []
        target_zoneids = []
        vehicle_idle_times = []
        post_action_distances = []
        post_action_durations = []
        post_action_zoneids = []
        post_action_locations = []
        target_station_ids = []

        active_requests = (
            getattr(self.env, "active_requests", {}) if self.env is not None else {}
        )
        charge_duration = max(
            1.0,
            float(getattr(self.env, "charge_duration", 1.0) if self.env is not None else 1.0),
        )

        def distance(origin: int, destination: int) -> float:
            if self.env is not None and hasattr(self.env, "_manhattan_distance_loc"):
                return float(self.env._manhattan_distance_loc(origin, destination))
            ox, oy = origin % self.grid_size, origin // self.grid_size
            dx, dy = destination % self.grid_size, destination // self.grid_size
            return float(abs(ox - dx) + abs(oy - dy))

        for item in batch_inputs:
            vehicle_id = int(item.get("vehicle_id", -1))
            vehicle_location = self._location_or_fallback(
                item.get("vehicle_location"),
                0,
            )
            target_location = vehicle_location
            post_location = vehicle_location
            request_value = 0.0
            target_distance = 0.0
            post_distance = 0.0
            post_duration = 1.0
            station_id = -1

            if int(action_type_id) == 2:
                target_location = self._location_or_fallback(
                    item.get("target_location"),
                    vehicle_location,
                )
                request_value = float(item.get("request_value", 0.0) or 0.0)
                target_distance = float(
                    item.get("pickup_dist", distance(vehicle_location, target_location)) or 0.0
                )
                post_location = target_location
                post_distance = target_distance
                request = active_requests.get(item.get("target_id"))
                if request is not None:
                    pickup = self._location_or_fallback(
                        getattr(request, "pickup", target_location),
                        target_location,
                    )
                    post_location = self._location_or_fallback(
                        getattr(request, "dropoff", pickup),
                        pickup,
                    )
                    post_distance = target_distance + distance(pickup, post_location)
                post_duration = max(1.0, post_distance)
            elif int(action_type_id) == 3:
                target_location = self._location_or_fallback(
                    item.get("station_location"),
                    vehicle_location,
                )
                post_location = target_location
                target_distance = distance(vehicle_location, target_location)
                post_distance = target_distance
                post_duration = max(0.0, target_distance) + charge_duration
                station_id = self._location_or_fallback(item.get("station_id"), -1)
            else:
                target_location = self._location_or_fallback(
                    item.get("target_location"),
                    vehicle_location,
                )
                post_location = target_location
                target_distance = distance(vehicle_location, target_location)
                post_distance = target_distance
                post_duration = max(1.0, target_distance)

            vehicle_ids.append(vehicle_id)
            vehicle_locations.append(vehicle_location)
            target_locations.append(target_location)
            current_times.append(float(item.get("current_time", 0.0) or 0.0))
            other_vehicles.append(float(item.get("other_vehicles", 0.0) or 0.0))
            num_requests.append(float(item.get("num_requests", 0.0) or 0.0))
            battery_levels.append(float(item.get("battery_level", 1.0) or 0.0))
            request_values.append(request_value)
            target_distances.append(target_distance)
            target_zoneids.append(int(item.get("pick_zone", 0) or 0))
            vehicle_idle_times.append(float(item.get("vehicle_idle_time", 0.0) or 0.0))
            post_action_distances.append(post_distance)
            post_action_durations.append(post_duration)
            post_action_zoneids.append(0)
            post_action_locations.append(post_location)
            target_station_ids.append(station_id)

        return self.batch_get_mixed_q_values(
            vehicle_ids=vehicle_ids,
            vehicle_locations=vehicle_locations,
            target_locations=target_locations,
            current_times=current_times,
            other_vehicles=other_vehicles,
            num_requests=num_requests,
            battery_levels=battery_levels,
            request_values=request_values,
            target_distances=target_distances,
            target_zoneids=target_zoneids,
            vehicle_idle_times=vehicle_idle_times,
            action_type_ids=[int(action_type_id)] * len(batch_inputs),
            post_action_distances=post_action_distances,
            post_action_durations=post_action_durations,
            post_action_zoneids=post_action_zoneids,
            post_action_locations=post_action_locations,
            target_station_ids=target_station_ids,
        )

    def batch_get_assignment_q_value(self, batch_inputs, multi_gpu_devices=None):
        del multi_gpu_devices
        return self._batch_get_action_q_values(batch_inputs, action_type_id=2)

    def batch_get_charging_q_value(self, batch_inputs):
        return self._batch_get_action_q_values(batch_inputs, action_type_id=3)

    def batch_get_idle_q_value(self, batch_inputs):
        return self._batch_get_action_q_values(batch_inputs, action_type_id=1)

    def batch_get_waiting_q_value(self, batch_inputs):
        return self._batch_get_action_q_values(batch_inputs, action_type_id=1)

    def get_idle_q_value(
        self,
        vehicle_id: int,
        vehicle_location: int,
        target_location: int,
        battery_level: float,
        current_time: float = 0.0,
        other_vehicles: int = 0,
        num_requests: int = 0,
    ) -> float:
        values = self.batch_get_idle_q_value([{
            "vehicle_id": vehicle_id,
            "vehicle_location": vehicle_location,
            "target_location": target_location,
            "battery_level": battery_level,
            "current_time": current_time,
            "other_vehicles": other_vehicles,
            "num_requests": num_requests,
        }])
        return float(values[0])

    def get_waiting_q_value(
        self,
        vehicle_id: int,
        vehicle_location: int,
        battery_level: float,
        current_time: float = 0.0,
        other_vehicles: int = 0,
        num_requests: int = 0,
    ) -> float:
        return self.get_idle_q_value(
            vehicle_id=vehicle_id,
            vehicle_location=vehicle_location,
            target_location=vehicle_location,
            battery_level=battery_level,
            current_time=current_time,
            other_vehicles=other_vehicles,
            num_requests=num_requests,
        )

    def get_charging_q_value(
        self,
        vehicle_id: int,
        station_id: int = -1,
        vehicle_location: int = 0,
        station_location: int | None = None,
        current_time: float = 0.0,
        other_vehicles: int = 0,
        num_requests: int = 0,
        battery_level: float = 1.0,
    ) -> float:
        values = self.batch_get_charging_q_value([{
            "vehicle_id": vehicle_id,
            "station_id": station_id,
            "vehicle_location": vehicle_location,
            "station_location": vehicle_location if station_location is None else station_location,
            "battery_level": battery_level,
            "current_time": current_time,
            "other_vehicles": other_vehicles,
            "num_requests": num_requests,
        }])
        return float(values[0])

    def get_q_value(
        self,
        vehicle_id: int,
        action_type: str | int,
        vehicle_location: int,
        target_location: int | None = None,
        current_time: float = 0.0,
        other_vehicles: int = 0,
        num_requests: int = 0,
        battery_level: float = 1.0,
        request_value: float = 0.0,
        target_distance: float | None = None,
        **kwargs,
    ) -> float:
        """Compatibility entry point used by trainer diagnostics and legacy callers."""
        del kwargs
        action = str(action_type)
        action_type_id = self._action_id(action_type)
        resolved_target = vehicle_location if target_location is None else target_location
        item = {
            "vehicle_id": vehicle_id,
            "vehicle_location": vehicle_location,
            "target_location": resolved_target,
            "current_time": current_time,
            "other_vehicles": other_vehicles,
            "num_requests": num_requests,
            "battery_level": battery_level,
            "request_value": request_value,
        }
        if target_distance is not None:
            item["pickup_dist"] = target_distance
        if action_type_id == 3:
            item["station_location"] = resolved_target
            try:
                item["station_id"] = int(action.rsplit("_", 1)[-1])
            except (TypeError, ValueError):
                item["station_id"] = -1
        values = self._batch_get_action_q_values([item], action_type_id=action_type_id)
        return float(values[0])

    # ------------------------------------------------------------------
    # Replay storage
    # ------------------------------------------------------------------

    def store_experience(self, **kwargs):
        experience = dict(kwargs)
        action_type = experience.get("action_type", "idle")
        experience["action_type_id"] = self._action_id(action_type)
        if experience["action_type_id"] == 3 and experience.get("target_station_id") is None:
            station_id = self._station_id_from_action_type(action_type)
            if station_id is not None:
                experience["target_station_id"] = station_id
        if "vehicle_type" not in experience:
            experience["vehicle_type"] = self._vehicle_type(int(experience.get("vehicle_id", -1)))
        if experience.get("vehicle_location") is None:
            experience["vehicle_location"] = 0
        if experience.get("target_location") is None:
            experience["target_location"] = experience["vehicle_location"]
        if experience.get("post_action_location") is None:
            experience["post_action_location"] = (
                experience.get("next_vehicle_location")
                if experience.get("next_vehicle_location") is not None
                else experience.get("target_location")
            )
            if experience["post_action_location"] is None:
                experience["post_action_location"] = experience.get("vehicle_location", 0)
        if experience.get("post_action_distance") is None:
            experience["post_action_distance"] = experience.get("target_distance", 0.0)
        if experience.get("post_action_duration") is None:
            experience["post_action_duration"] = experience.get("dur_time", 1.0)
        if experience.get("post_action_zoneid") is None:
            experience["post_action_zoneid"] = experience.get("target_zoneid", 0)
        experience["myopic_score"] = self._myopic_score(
            int(experience["action_type_id"]),
            float(experience.get("request_value", 0.0) or 0.0),
            float(experience.get("target_distance", 0.0) or 0.0),
            float(experience.get("post_action_distance", 0.0) or 0.0),
        )
        self.experience_buffer.append(experience)

    def store_rejection_experience(self, *args, **kwargs):
        return None

    def store_acceptance_experience(self, *args, **kwargs):
        return None

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _location_or_fallback(value: Any, fallback: Any = 0) -> int:
        """Convert optional action locations without letting None enter a tensor."""
        resolved = fallback if value is None else value
        if resolved is None:
            resolved = 0
        try:
            return int(resolved)
        except (TypeError, ValueError, OverflowError):
            return int(fallback) if fallback is not None else 0

    def _experience_location(self, value: Any, fallback: Any = 0) -> int:
        """Normalize scalar location ids and environment ``(x, y)`` coordinates."""
        resolved = fallback if value is None else value
        if isinstance(resolved, (tuple, list, np.ndarray)):
            coordinates = np.asarray(resolved).reshape(-1)
            if coordinates.size >= 2:
                x, y = int(coordinates[0]), int(coordinates[1])
                return y * self.grid_size + x
            if coordinates.size == 1:
                resolved = coordinates[0]
            else:
                resolved = fallback
        try:
            return int(resolved)
        except (TypeError, ValueError, OverflowError):
            return 0 if fallback is None else self._experience_location(fallback, 0)

    @classmethod
    def _neighbour_candidates_from_actions(cls, candidates: list[dict]) -> list[dict]:
        neighbours = []
        for candidate in candidates or []:
            station_id = candidate.get("target_station_id")
            target_location = cls._location_or_fallback(
                candidate.get("target_location"),
                candidate.get("post_action_location", 0),
            )
            neighbours.append({
                "node_type": "station" if station_id is not None else "zone",
                "node_id": int(station_id if station_id is not None else target_location),
                "target_location": target_location,
                "distance": max(0.0, float(candidate.get("target_distance", 0.0) or 0.0)),
            })
        return neighbours

    def _edge_tensor_from_experience(self, exp: dict, *, next_state: bool = False, candidate: dict | None = None):
        if candidate is None:
            action_id = int(exp.get("action_type_id", self._action_id(exp.get("action_type", "idle"))))
            action_type = exp.get("action_type", "idle")
            vehicle_location = self._experience_location(exp.get("vehicle_location"), 0)
            target_location = self._experience_location(
                exp.get("target_location"),
                vehicle_location,
            )
            post_location = self._experience_location(
                exp.get("post_action_location"),
                exp.get("next_vehicle_location")
                if exp.get("next_vehicle_location") is not None
                else target_location,
            )
            current_time = float(exp.get("current_time", 0.0))
            battery = float(exp.get("battery_level", 1.0))
            idle = float(exp.get("vehicle_idle_time", 0.0))
            target_distance = float(exp.get("target_distance", 0.0) or 0.0)
            post_distance = float(exp.get("post_action_distance", target_distance) or 0.0)
            post_duration = float(exp.get("post_action_duration", exp.get("dur_time", 1.0)) or 0.0)
            request_value = float(exp.get("request_value", 0.0) or 0.0)
            target_station_id = exp.get("target_station_id")
            neighbour_candidates = exp.get("graph_neighbour_candidates") or []
        elif next_state:
            action_type = candidate.get("action_type", "idle")
            action_id = self._action_id(action_type)
            vehicle_location = self._experience_location(
                exp.get("next_vehicle_location"),
                exp.get("vehicle_location", 0),
            )
            target_location = self._experience_location(
                candidate.get("target_location"),
                vehicle_location,
            )
            post_location = self._experience_location(
                candidate.get("post_action_location"),
                target_location,
            )
            current_time = float(exp.get("current_time", 0.0)) + float(exp.get("dur_time", 1.0))
            battery = float(exp.get("next_battery_level", exp.get("battery_level", 1.0)))
            idle = float(exp.get("next_vehicle_idle_time", 0.0))
            target_distance = float(candidate.get("target_distance", 0.0) or 0.0)
            post_distance = float(candidate.get("post_action_distance", target_distance) or 0.0)
            post_duration = float(candidate.get("post_action_duration", 0.0) or 0.0)
            request_value = float(candidate.get("request_value", 0.0) or 0.0)
            target_station_id = candidate.get("target_station_id")
            neighbour_candidates = exp.get("next_graph_neighbour_candidates") or []
            if not neighbour_candidates:
                neighbour_candidates = self._neighbour_candidates_from_actions(
                    exp.get("next_candidate_actions") or []
                )
        else:
            # Standard discrete SAC evaluates its actor on the replay state
            # s_t while the Bellman target uses s_{t+1}.  Legacy callers only
            # request candidate edges with next_state=True, so this branch is
            # isolated to the standard mode.
            action_type = candidate.get("action_type", "idle")
            action_id = self._action_id(action_type)
            vehicle_location = self._experience_location(
                exp.get("vehicle_location"),
                0,
            )
            target_location = self._experience_location(
                candidate.get("target_location"),
                vehicle_location,
            )
            post_location = self._experience_location(
                candidate.get("post_action_location"),
                target_location,
            )
            current_time = float(exp.get("current_time", 0.0))
            battery = float(exp.get("battery_level", 1.0))
            idle = float(exp.get("vehicle_idle_time", 0.0))
            target_distance = float(candidate.get("target_distance", 0.0) or 0.0)
            post_distance = float(candidate.get("post_action_distance", target_distance) or 0.0)
            post_duration = float(candidate.get("post_action_duration", 0.0) or 0.0)
            request_value = float(candidate.get("request_value", 0.0) or 0.0)
            target_station_id = candidate.get("target_station_id")
            neighbour_candidates = exp.get("graph_neighbour_candidates") or []
            if not neighbour_candidates:
                neighbour_candidates = self._neighbour_candidates_from_actions(
                    exp.get("candidate_actions") or []
                )
        if target_station_id is None:
            target_station_id = self._station_id_from_action_type(action_type)
        queue_wait_feature = self._queue_wait_feature_from_experience(exp, candidate)
        vehicle_id = int(exp.get("vehicle_id", -1))
        edge_t, type_w, _ = self._edge_tensor_from_arrays(
            vehicle_ids=np.asarray([vehicle_id]),
            vehicle_locations=np.asarray([vehicle_location]),
            target_locations=np.asarray([target_location]),
            current_times=np.asarray([current_time], dtype=np.float32),
            other_vehicles=np.asarray([float(exp.get("other_vehicles", 0.0))], dtype=np.float32),
            num_requests=np.asarray([float(exp.get("num_requests", 0.0))], dtype=np.float32),
            battery_levels=np.asarray([battery], dtype=np.float32),
            target_distances=np.asarray([target_distance], dtype=np.float32),
            vehicle_idle_times=np.asarray([idle], dtype=np.float32),
            action_type_ids=np.asarray([action_id], dtype=np.int64),
            post_action_distances=np.asarray([post_distance], dtype=np.float32),
            post_action_durations=np.asarray([post_duration], dtype=np.float32),
            post_action_locations=np.asarray([post_location], dtype=np.int64),
            target_station_ids=np.asarray([target_station_id if target_station_id is not None else -1], dtype=np.int64),
            queue_wait_features=np.asarray([queue_wait_feature], dtype=np.float32),
            vehicle_neighbour_candidates={vehicle_id: list(neighbour_candidates)},
        )
        g = self._myopic_score(
            action_id,
            request_value,
            target_distance,
            post_distance,
        )
        return edge_t, type_w, g

    def _next_soft_values(self, batch: list[dict]) -> torch.Tensor:
        values = []
        actor_losses = []
        alpha_terms = []
        with torch.no_grad():
            target_net = self.target_network
            target_net2 = self.target_critic2
        for exp in batch:
            if exp.get("is_system_done", False) or exp.get("is_vehicle_done", False):
                values.append(torch.zeros((), dtype=torch.float32, device=self.device))
                continue
            candidates = exp.get("next_candidate_actions") or []
            if not candidates:
                values.append(torch.zeros((), dtype=torch.float32, device=self.device))
                continue
            edge_rows = []
            type_ws = []
            for cand in candidates:
                edge_t, type_w, _ = self._edge_tensor_from_experience(exp, next_state=True, candidate=cand)
                edge_rows.append(edge_t.squeeze(0))
                type_ws.append(type_w.squeeze(0))
            edges = torch.stack(edge_rows)
            weights = torch.stack(type_ws)
            with torch.no_grad():
                q_next = torch.minimum(target_net(edges), target_net2(edges)) * weights
            logits = self.actor(edges).squeeze(1)
            logp = F.log_softmax(logits, dim=0)
            probs = logp.exp()
            soft_value = torch.sum(probs * (q_next.squeeze(1) - self.alpha.detach() * logp))
            values.append(soft_value)
            q_actor = torch.minimum(self.network(edges), self.critic2(edges)).detach().squeeze(1)
            actor_losses.append(torch.sum(probs * (self.alpha.detach() * logp - q_actor)))
            target_entropy = self._target_entropy(len(candidates))
            alpha_terms.append(torch.sum(probs.detach() * logp) + target_entropy)
        if actor_losses:
            self._last_actor_loss_tensor = torch.stack(actor_losses).mean()
            self._last_alpha_term_tensor = torch.stack(alpha_terms).mean()
        else:
            self._last_actor_loss_tensor = torch.zeros((), dtype=torch.float32, device=self.device)
            self._last_alpha_term_tensor = torch.zeros((), dtype=torch.float32, device=self.device)
        return torch.stack(values).unsqueeze(1)

    def _orthogonality_loss(self, q: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        q = q.reshape(-1)
        g = g.reshape(-1)
        g_std = torch.std(g, unbiased=False)
        q_std = torch.std(q, unbiased=False)
        if q.numel() < 2 or g_std < 1e-6 or q_std < 1e-6:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        qn = (q - q.mean()) / (q_std + 1e-6)
        gn = (g - g.mean()) / (g_std + 1e-6)
        return torch.mean(qn * gn).pow(2)

    def _record_td_diagnostics(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        q1: torch.Tensor,
        q2: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            scale = target.detach().abs().clamp_min(1.0)
            norm_loss = self.loss_fn(q1.detach() / scale, target.detach() / scale)
            norm_loss = norm_loss + self.loss_fn(q2.detach() / scale, target.detach() / scale)
            normalized_loss = float(norm_loss.item())
            normalized_abs = torch.abs(pred.detach() - target.detach()) / scale
            raw_abs = torch.abs(pred.detach() - target.detach())
            self.normalized_td_losses.append(normalized_loss)
            self.td_error_history.append({
                "normalized_td_loss": normalized_loss,
                "normalized_td_abs_mean": float(normalized_abs.mean().item()),
                "normalized_td_abs_max": float(normalized_abs.max().item()),
                "td_abs_mean": float(raw_abs.mean().item()),
                "td_abs_max": float(raw_abs.max().item()),
                "td_bias_mean": float((pred.detach() - target.detach()).mean().item()),
                "target_scale_mean": float(scale.mean().item()),
            })
            return normalized_loss

    def train_step(self, batch_size: int = 64, tau: float | None = None, ifEV: bool = False) -> float:
        del ifEV
        if len(self.experience_buffer) < max(8, batch_size // 2):
            queue_only_loss = self.train_queue_predictor(batch_size=batch_size)
            if queue_only_loss > 0:
                self.training_losses.append(float(self.queue_loss_weight) * queue_only_loss)
            return float(self.queue_loss_weight) * queue_only_loss
        self._graph_cache_key = None
        self._graph_cache = None
        batch = random.sample(list(self.experience_buffer), min(batch_size, len(self.experience_buffer)))
        tau = self.tau if tau is None else float(tau)

        edge_rows = []
        type_ws = []
        g_values = []
        rewards = []
        durs = []
        masks = []
        for exp in batch:
            edge_t, type_w, g = self._edge_tensor_from_experience(exp)
            edge_rows.append(edge_t.squeeze(0))
            type_ws.append(type_w.squeeze(0))
            g_values.append(float(g))
            rewards.append(float(exp.get("reward", 0.0)) - float(g))
            durs.append(float(exp.get("dur_time", 1.0)))
            masks.append(0.0 if (exp.get("is_system_done", False) or exp.get("is_vehicle_done", False)) else 1.0)

        edges = torch.stack(edge_rows)
        weights = torch.stack(type_ws)
        g_t = torch.tensor(g_values, dtype=torch.float32, device=self.device).unsqueeze(1)
        residual_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        durs_t = torch.tensor(durs, dtype=torch.float32, device=self.device).unsqueeze(1)
        masks_t = torch.tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)

        q1 = self.network(edges) * weights
        q2 = self.critic2(edges) * weights
        next_v = self._next_soft_values(batch)
        target = residual_rewards + (self.gamma ** durs_t) * next_v * masks_t
        critic_loss = self.loss_fn(q1, target.detach()) + self.loss_fn(q2, target.detach())
        actor_loss = getattr(self, "_last_actor_loss_tensor", torch.zeros((), dtype=torch.float32, device=self.device))
        alpha_term = getattr(self, "_last_alpha_term_tensor", torch.zeros((), dtype=torch.float32, device=self.device))
        alpha_loss = -(self.log_alpha * alpha_term.detach())
        orth_loss = self._orthogonality_loss(torch.minimum(q1, q2).detach(), g_t)
        loss = (
            critic_loss
            + self.lambda_actor * actor_loss
            + self.lambda_alpha * alpha_loss
            + self.lambda_orth * orth_loss
        )
        queue_loss = self._queue_loss_from_samples(batch_size)
        if queue_loss is not None:
            loss = loss + float(self.queue_loss_weight) * queue_loss

        self.optimizer.zero_grad()
        self.queue_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.graph_encoder.parameters() if parameter.requires_grad]
            + list(self.mixer.parameters())
            + list(self.network.parameters())
            + list(self.critic2.parameters())
            + list(self.actor.parameters()),
            max_norm=10.0,
        )
        torch.nn.utils.clip_grad_norm_(self.queue_predictor.parameters(), max_norm=10.0)
        self.optimizer.step()
        if queue_loss is not None:
            self.queue_optimizer.step()
            self.queue_predictor_trained = True
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(1e-4), math.log(10.0))
        self._soft_update(self.network, self.target_network, tau)
        self._soft_update(self.critic2, self.target_critic2, tau)

        objective_value = float(loss.item())
        loss_value = float(critic_loss.detach().item())
        queue_loss_value = float(queue_loss.detach().item()) if queue_loss is not None else 0.0
        self.training_losses.append(loss_value)
        if queue_loss is not None:
            self.queue_training_losses.append(queue_loss_value)
            self.queue_training_mse_losses.append(queue_loss_value)
        self.training_step += 1
        pred = torch.minimum(q1, q2)
        with torch.no_grad():
            norm_td_loss = self._record_td_diagnostics(
                pred=pred,
                target=target,
                q1=q1,
                q2=q2,
            )
            self.q_values_history.append({
                "mean": float(pred.mean().item()),
                "std": float(pred.std().item()) if pred.numel() > 1 else 0.0,
                "target_mean": float(target.mean().item()),
                "target_std": float(target.std().item()) if target.numel() > 1 else 0.0,
                "normalized_td_loss": float(norm_td_loss),
                "beta": float(self._beta()),
                "alpha": float(self.alpha.item()),
                "objective": objective_value,
                "critic": loss_value,
                "actor": float(actor_loss.detach().item()),
                "alpha_loss": float(alpha_loss.detach().item()),
                "orth": float(orth_loss.item()),
                "queue_loss": queue_loss_value,
            })
        if self.training_step % 100 == 0:
            print(
                f"[{self.debug_name}] ST-MASAC-GAT step={self.training_step} critic_loss={loss_value:.4f} "
                f"queue={queue_loss_value:.4f} norm_td={norm_td_loss:.4f} objective={objective_value:.4f} actor={actor_loss.item():.4f} "
                f"alpha={self.alpha.item():.4f} beta={self._beta():.3f} "
                f"res_target_mean={target.mean().item():.3f}",
                flush=True,
            )
        return loss_value

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    # ------------------------------------------------------------------
    # Checkpoint extras
    # ------------------------------------------------------------------

    def extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "graph_encoder_state_dict": self.graph_encoder.state_dict(),
            "mixer_state_dict": self.mixer.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "queue_predictor_state_dict": self.queue_predictor.state_dict(),
            "queue_optimizer_state_dict": self.queue_optimizer.state_dict(),
            "queue_predictor_trained": bool(self.queue_predictor_trained),
            "queue_training_losses": list(self.queue_training_losses),
            "queue_training_mse_losses": list(self.queue_training_mse_losses),
            "recent_station_waits": dict(self.recent_station_waits),
            "queue_loss_weight": float(self.queue_loss_weight),
            "queue_edge_loss_weight": float(self.queue_edge_loss_weight),
            "log_alpha": self.log_alpha.detach().cpu(),
            "zone_distribution_mode": self.zone_distribution_mode,
            "neighbour_number": int(self.neighbour_number),
            "freeze_graph_encoder": bool(self.freeze_graph_encoder),
            "freeze_neighbour_context": bool(self.freeze_neighbour_context),
            "beta_max": self.beta_max,
            "eta_pi": self.eta_pi,
            "residual_clip_rho": self.residual_clip_rho,
        }

    def load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        if "critic2_state_dict" in state:
            self.critic2.load_state_dict(state["critic2_state_dict"], strict=False)
        if "target_critic2_state_dict" in state:
            self.target_critic2.load_state_dict(state["target_critic2_state_dict"], strict=False)
        if "graph_encoder_state_dict" in state:
            self.graph_encoder.load_state_dict(state["graph_encoder_state_dict"], strict=False)
        if "mixer_state_dict" in state:
            self.mixer.load_state_dict(state["mixer_state_dict"], strict=False)
        if "actor_state_dict" in state:
            self.actor.load_state_dict(state["actor_state_dict"], strict=False)
        if "queue_predictor_state_dict" in state:
            self.queue_predictor.load_state_dict(state["queue_predictor_state_dict"], strict=False)
        if "queue_optimizer_state_dict" in state:
            try:
                self.queue_optimizer.load_state_dict(state["queue_optimizer_state_dict"])
            except Exception:
                pass
        self.queue_predictor_trained = bool(state.get("queue_predictor_trained", self.queue_predictor_trained))
        self.queue_training_losses = list(state.get("queue_training_losses", self.queue_training_losses))
        self.queue_training_mse_losses = list(state.get("queue_training_mse_losses", self.queue_training_mse_losses))
        self.recent_station_waits = dict(state.get("recent_station_waits", self.recent_station_waits))
        self.queue_loss_weight = float(state.get("queue_loss_weight", self.queue_loss_weight))
        self.queue_edge_loss_weight = float(state.get("queue_edge_loss_weight", self.queue_edge_loss_weight))
        if "log_alpha" in state:
            value = state["log_alpha"]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
            self.log_alpha.data.copy_(value.to(self.device).reshape(()))

    def add_to_logs(self, *args, **kwargs):
        return None

    def remember(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return self.train_step()
