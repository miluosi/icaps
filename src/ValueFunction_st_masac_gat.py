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

from src.recourse.replay import PrioritizedJointReplayBuffer
from src.recourse.target_builder import RecourseTargetBuilder
from src.recourse.types import (
    ActionType,
    FeasibleEdgeSnapshot,
    FeasibleGraphSnapshot,
    RecourseTransition,
    SystemSnapshot,
)


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


from src.acceptance_features import AcceptanceFeatureMixin, insert_zero_input


class PyTorchChargingValueFunction(AcceptanceFeatureMixin):
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
        checkpoint_replay: str = "recent",
        checkpoint_replay_recent: int = 5_000,
        iftransformer: bool = False,
        neighbour_number: int = 5,
    ):
        del log_dir, encoder, iftransformer
        self.grid_size = int(grid_size)
        self.num_vehicles = int(num_vehicles)
        self.episode_length = max(1.0, float(episode_length))
        self.max_requests = max(1.0, float(max_requests))
        self.env = env
        self._init_acceptance_feature()
        self.device = torch.device(device)
        self.zone_distribution_mode = zone_distribution_mode or "st_masac_gat"
        self.neighbour_number = max(0, int(neighbour_number))
        self.freeze_graph_encoder = self.zone_distribution_mode == "st_masac_gat_frozen"
        self.freeze_neighbour_context = (
            self.zone_distribution_mode == "st_masac_gat_neighbour_frozen"
        )

        self.gamma = 0.95
        self.within_epoch_gamma = 1.0
        self.tau = 0.005
        self.learning_rate = 1e-3
        self.huber_kappa = 1.0
        self.gradient_clip_norm = 10.0
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
        self.target_graph_encoder = copy.deepcopy(self.graph_encoder).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.target_graph_encoder.requires_grad_(False)
        self.target_mixer.requires_grad_(False)
        self.network = _MLP(self.edge_dim, hidden_dim=128).to(self.device)
        self.critic2 = _MLP(self.edge_dim, hidden_dim=128).to(self.device)
        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self.actor = _MLP(self.edge_dim, hidden_dim=128).to(self.device)
        self.queue_predictor = _MLP(self.queue_feature_dim, hidden_dim=64).to(self.device)
        self.target_queue_predictor = copy.deepcopy(self.queue_predictor).to(self.device)
        self.target_queue_predictor.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.05), dtype=torch.float32, device=self.device))

        self.acceptance_input_index = self.edge_local_dim
        self.rejection_input_index = self.edge_local_dim
        if self.acceptance_input_enabled:
            for module in (self.network, self.critic2, self.target_network, self.target_critic2):
                module.net[0] = insert_zero_input(module.net[0], self.rejection_input_index, count=2)
            self.edge_local_dim += 2
            self.edge_dim += 2

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
        self.loss_fn = nn.SmoothL1Loss(beta=self.huber_kappa)
        self.queue_loss_fn = nn.MSELoss()
        self.experience_buffer = deque(maxlen=int(replay_buffer_size))
        self.rejection_buffer = deque(maxlen=int(replay_buffer_size))
        self.joint_replay_buffer = PrioritizedJointReplayBuffer(
            capacity=max(1, int(replay_buffer_size) // 5),
            seed=int(getattr(env, "initial_random_seed", 0) or 0),
        )
        self._owns_joint_replay_payload = True
        self.checkpoint_replay = str(checkpoint_replay)
        self.checkpoint_replay_recent = max(1, int(checkpoint_replay_recent))
        self.target_builder = RecourseTargetBuilder()
        self.planning_objective_mode = "learned"
        self.state_variant = "joint_state_shared_critic"
        self.learner_variant = getattr(
            self, "learner_variant", "optimization_anchored_residual"
        )
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
        self.joint_training_step = 0
        self.optimizer_steps_total = 0
        self.optimizer_steps_joint = 0
        self.optimizer_steps_edge = 0
        self.optimizer_steps_queue = 0
        self.debug_name = "ADP"
        self._graph_cache_key = None
        self._graph_cache = None
        self._target_graph_cache_key = None
        self._target_graph_cache = None
        self._replay_collection_context = None
        self._joint_critic_router: dict[int, "PyTorchChargingValueFunction"] = {
            1: self,
            2: self,
        }
        self._follower_target_provider = None
        self.joint_training_diagnostics: list[dict[str, float | int | str]] = []
        self.next_transition_link_misses = 0
        self.next_transition_link_lookups = 0
        self._target_component_cache: dict[tuple, Any] = {}
        # The actor is not part of the deployed MCMF policy when eta_pi=0.
        # Keep it diagnostic-only instead of allowing its gradients to change
        # the shared graph encoder used by assignment.
        if self.eta_pi == 0.0:
            self.lambda_actor = 0.0
            self.lambda_alpha = 0.0
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
        policy_step = max(
            int(self.training_step), int(getattr(self, "joint_training_step", 0))
        )
        return float(
            min(self.beta_max, self.beta_max * policy_step / self.beta_warmup_steps)
        )

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
        if action in {"wait", "waiting", "continue_wait"}:
            return 0
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
        rejection_probability: float = 0.0,
        human_response_mask: float = 0.0,
    ) -> list[float]:
        action_type_id = int(action_type_id)
        post_soc = self._post_battery(float(battery_level), action_type_id, float(post_action_distance))
        features = state_features + [
            1.0 if action_type_id == 1 else 0.0,
            1.0 if action_type_id == 2 else 0.0,
            1.0 if action_type_id == 3 else 0.0,
            float(np.clip(float(target_distance) / 30.0, 0.0, 5.0)),
            float(np.clip(float(post_action_distance) / 50.0, 0.0, 5.0)),
            float(np.clip(float(post_action_duration) / max(self.episode_length, 1.0), 0.0, 2.0)),
            post_soc,
            1.0 + float(np.clip(float(queue_wait_feature), 0.0, 4.0)),
        ]
        if self.acceptance_input_enabled:
            from src.rejection_anchor import expected_structured_score
            expected_structured_score(0., 0., rejection_probability, human_response_mask)
            if human_response_mask and (vehicle_type != 1 or action_type_id != 2):
                raise ValueError('Only unanswered EV service edges can carry a response mask')
            features.extend([float(rejection_probability), float(human_response_mask)])
        return features

    def _actor_features(self, edges):
        if not self.acceptance_input_enabled:
            return edges
        index = self.rejection_input_index
        return torch.cat([edges[:, :index], edges[:, index + 2:]], dim=1)

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
        self.optimizer_steps_queue += 1
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

    def _queue_wait_from_feature_snapshot(
        self,
        features: Any,
        *,
        target_context: bool = False,
    ) -> float | None:
        if features is None or not self.queue_predictor_trained:
            return None
        try:
            row = [float(x) for x in features]
        except (TypeError, ValueError):
            return None
        if len(row) != self.queue_feature_dim:
            return None
        predictor = (
            self.target_queue_predictor
            if target_context
            else self.queue_predictor
        )
        with torch.no_grad():
            tensor = torch.tensor([row], dtype=torch.float32, device=self.device)
            wait = predictor(tensor).squeeze(1)
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

    def _queue_wait_feature_from_experience(
        self,
        exp: dict,
        candidate: dict | None = None,
        *,
        target_context: bool = False,
    ) -> float:
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
        wait = self._queue_wait_from_feature_snapshot(
            source.get("queue_features"), target_context=target_context
        )
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

    def _graph_locations(self, snapshot: SystemSnapshot | None = None) -> list[int]:
        if snapshot is not None:
            return list(snapshot.zone_ids)
        if self.env is None:
            return list(range(max(1, self.grid_size * self.grid_size)))
        zones = list(getattr(self.env, "aux_zone_ids", []) or getattr(self.env, "relocation_target_ids", []) or [])
        if not zones:
            zones = sorted(getattr(self.env, "zone_coords", {}).keys())
        return [int(z) for z in zones[: max(1, len(zones))]]

    def _build_graph_node_features(
        self,
        snapshot: SystemSnapshot | None = None,
    ) -> tuple[torch.Tensor, dict[int, int], dict[int, int], int]:
        zones = self._graph_locations(snapshot)
        zone_to_row = {int(z): idx for idx, z in enumerate(zones)}
        rows: list[list[float]] = []
        current_time = (
            float(snapshot.current_time)
            if snapshot is not None
            else float(getattr(self.env, "current_time", 0.0) if self.env is not None else 0.0)
        )
        time_norm = self._time_norm(current_time)
        hour_angle = time_norm * 2.0 * math.pi
        if snapshot is not None:
            active_requests = list(snapshot.requests)
            vehicles = {
                vehicle.vehicle_id: {
                    "type": vehicle.vehicle_type,
                    "location": vehicle.location,
                    "battery": vehicle.battery,
                    "is_online": vehicle.online,
                }
                for vehicle in snapshot.vehicles
            }
        else:
            active_requests = list(getattr(self.env, "active_requests", {}).values()) if self.env is not None else []
            vehicles = getattr(self.env, "vehicles", {}) if self.env is not None else {}

        demand = {z: 0.0 for z in zones}
        for req in active_requests:
            pickup_zone = getattr(req, "pickup_zone_id", None)
            zid = int(
                getattr(req, "pickup", 0)
                if pickup_zone is None
                else pickup_zone
            )
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
        if snapshot is not None:
            stations = {station.station_id: station for station in snapshot.stations}
        else:
            stations = getattr(getattr(self.env, "charging_manager", None), "stations", {}) if self.env is not None else {}
        for sid, station in sorted(stations.items()):
            station_to_row[int(sid)] = len(rows)
            capacity = float(
                getattr(station, "capacity", getattr(station, "max_capacity", 1)) or 1
            )
            current = float(
                getattr(
                    station,
                    "occupied",
                    len(getattr(station, "current_vehicles", []) or []),
                )
            )
            inbound = float(
                getattr(
                    station,
                    "inbound",
                    len(getattr(station, "charging_queue_notarrived", []) or []),
                )
            )
            queued = float(
                getattr(
                    station,
                    "queued",
                    len(getattr(station, "charging_queue", []) or []),
                )
            )
            queue = inbound + queued
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

    def _graph_context(
        self,
        snapshot: SystemSnapshot | None = None,
        *,
        target: bool = False,
    ):
        key = (
            "snapshot",
            hash(snapshot),
        ) if snapshot is not None else (
            "live",
            id(self.env),
            int(getattr(self.env, "current_time", 0) if self.env is not None else 0),
            len(getattr(self.env, "active_requests", {}) if self.env is not None else {}),
        )
        cache_key_name = "_target_graph_cache_key" if target else "_graph_cache_key"
        cache_name = "_target_graph_cache" if target else "_graph_cache"
        if getattr(self, cache_key_name) == key and getattr(self, cache_name) is not None:
            return getattr(self, cache_name)
        node_features, zone_to_row, station_to_row, global_row = self._build_graph_node_features(snapshot)
        encoder = self.target_graph_encoder if target else self.graph_encoder
        mixer = self.target_mixer if target else self.mixer
        embeddings = encoder(node_features)
        pooled = embeddings.mean(dim=0)
        w_ev, w_aev, baseline = mixer(pooled)
        cache = {
            "embeddings": embeddings,
            "zone_to_row": zone_to_row,
            "station_to_row": station_to_row,
            "global_row": global_row,
            "pooled": pooled,
            "w_ev": w_ev,
            "w_aev": w_aev,
            "baseline": baseline,
            "encoder": encoder,
            "vehicle_type_by_id": (
                {vehicle.vehicle_id: vehicle.vehicle_type for vehicle in snapshot.vehicles}
                if snapshot is not None
                else {}
            ),
        }
        setattr(self, cache_key_name, key)
        setattr(self, cache_name, cache)
        return cache

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
        enriched_batch = graph["encoder"].add_neighbour_context(
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
        graph_snapshot: SystemSnapshot | None = None,
        target_context: bool = False,
        vehicle_types=None,
        post_demand_features=None,
        rejection_probabilities=None,
        human_response_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del post_demand_features
        graph = self._graph_context(graph_snapshot, target=target_context)
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
            if vehicle_types is not None:
                vehicle_type = int(vehicle_types[i])
            else:
                vehicle_type = int(
                    graph["vehicle_type_by_id"].get(
                        int(vehicle_ids[i]),
                        self._vehicle_type(int(vehicle_ids[i])),
                    )
                )
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
                rejection_probability=(0.0 if rejection_probabilities is None else rejection_probabilities[i]),
                human_response_mask=(0.0 if human_response_masks is None else human_response_masks[i]),
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
        request_ids=None,
        rejection_probabilities=None,
    ):
        del target_zoneids, post_action_zoneids
        size = len(vehicle_ids)
        if size == 0:
            return []
        rejection_probabilities = self.rejection_for_live_edges(
            vehicle_ids, action_type_ids, request_ids, rejection_probabilities
        )
        human_response_masks = self.response_masks_for_live_edges(vehicle_ids, action_type_ids)
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
        success_scores = g.copy()
        for i in range(size):
            g[i], _ = self.response_anchor(g[i], request_values[i], target_distances[i],
                                           rejection_probabilities[i], human_response_masks[i])
        if self.planning_objective_mode == "structured_only":
            self._last_adp_score_stats = {
                "mode": "structured_only",
                "g_mean": float(np.mean(g)),
                "beta": 0.0,
                "score_mean": float(np.mean(g)),
            }
            return g.astype(np.float32).tolist()
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
                rejection_probabilities=rejection_probabilities,
                human_response_masks=human_response_masks,
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
                scores = scores + float(self.eta_pi) * F.logsigmoid(self.actor(self._actor_features(edge_t)))
        values = scores.squeeze(1).cpu().numpy()
        self._last_adp_score_stats = {
            "mode": self.zone_distribution_mode,
            "g_mean": float(np.mean(g)),
            "beta": float(self._beta()),
            "score_mean": float(np.mean(values)),
            "q_reject_mean": float(np.mean(rejection_probabilities)),
            "human_response_edges": int(np.sum(human_response_masks)),
            "success_score_mean": float(np.mean(success_scores)),
            "risk_deduction_mean": float(np.mean(success_scores - g)),
            "expected_anchor_mean": float(np.mean(g)),
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
            request_ids=[target_id],
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
            request_ids=[row.get("target_id", -1) for row in batch_inputs],
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

    def set_replay_collection_context(self, action: Any | None) -> None:
        """Bind immutable action metadata for the next storage call."""

        self._replay_collection_context = action

    def store_recourse_transition(self, transition: RecourseTransition) -> None:
        if not bool(getattr(self, "_owns_joint_replay_payload", True)):
            return
        self._validate_response_transition(transition)
        self.joint_replay_buffer.add(transition)

    def _validate_response_transition(self, transition):
        for graph in (transition.ev_stage_graph, transition.aev_stage_graph):
            if graph is not None:
                for edge in graph.edges:
                    if edge.response_model_hash != self.response_model_hash:
                        raise ValueError('Replay rejection predictor hash mismatch')

    def set_planning_objective_mode(self, mode: str) -> str:
        if mode not in {"learned", "structured_only"}:
            raise ValueError(f"invalid planning objective mode: {mode}")
        previous = self.planning_objective_mode
        self.planning_objective_mode = mode
        return previous

    def store_experience(self, **kwargs):
        if bool(getattr(self.env, "evaluatemode", False)):
            return
        experience = copy.deepcopy(dict(kwargs))
        action = self._replay_collection_context
        metadata = getattr(action, "metadata", None)
        if metadata is not None:
            request_snapshot = getattr(metadata, "request_snapshot", None)
            if request_snapshot is not None:
                experience.setdefault("request_id", request_snapshot.request_id)
            experience.setdefault("transition_id", metadata.transition_id)
            experience.setdefault("stage_id", int(metadata.stage_id))
            experience.setdefault("acceptance_outcome", metadata.acceptance_outcome)
            experience.setdefault("residual_category", metadata.residual_category)
            experience.setdefault("state_snapshot", metadata.state_snapshot)
            experience.setdefault(
                "feasible_graph_snapshot", metadata.feasible_graph_snapshot
            )
            experience.setdefault(
                "residual_state_snapshot", metadata.residual_state_snapshot
            )
            experience.setdefault("next_state_snapshot", metadata.next_state_snapshot)
            experience.setdefault("joint_action_snapshot", metadata.joint_action_snapshot)
            for key, value in metadata.extras.items():
                experience.setdefault(key, value)
            next_metadata = getattr(getattr(action, "next_action", None), "metadata", None)
            if next_metadata is not None:
                experience.setdefault(
                    "next_feasible_graph_snapshot",
                    next_metadata.feasible_graph_snapshot,
                )
                experience.setdefault(
                    "next_state_snapshot", next_metadata.state_snapshot
                )
        experience.setdefault("schema_version", 1)
        experience.setdefault("mode", getattr(self.env, "decision_mode", "integrated"))
        experience.setdefault(
            "recourse_variant", getattr(self.env, "recourse_variant", "legacy")
        )
        experience.setdefault("state_variant", self.state_variant)
        experience.setdefault("learner_variant", self.learner_variant)
        experience.setdefault(
            "solver_backend", getattr(self.env, "mcmf_backend", "unknown")
        )
        action_type = experience.get("action_type", "idle")
        if metadata is not None:
            canonical_type = ActionType(metadata.canonical_type)
            action_type = canonical_type.name.lower()
            experience["action_type"] = action_type
            experience["action_type_id"] = int(canonical_type)
            if (
                canonical_type == ActionType.CHARGE
                and experience.get("target_station_id") is None
            ):
                experience["target_station_id"] = getattr(
                    action, "charging_station_id", None
                )
        else:
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
        q, mask = self.response_from_experience(experience)
        experience.update(rejection_probability=q, human_response_mask=mask,
                          response_model_hash=self.response_model_hash, response_schema_version=3)
        experience['success_structured_score'] = experience['myopic_score']
        experience['myopic_score'], experience['rejection_structured_score'] = self.response_anchor(
            experience['myopic_score'], float(experience.get('request_value', 0.) or 0.),
            float(experience.get('target_distance', 0.) or 0.), q, mask)
        self.experience_buffer.append(experience)
        self._replay_collection_context = None

    def store_rejection_experience(self, *args, **kwargs):
        del args
        sample = copy.deepcopy(dict(kwargs))
        sample["was_rejected"] = True
        sample["rejection_label"] = 1.0
        self.rejection_buffer.append(sample)
        return sample

    def store_acceptance_experience(self, *args, **kwargs):
        del args
        sample = copy.deepcopy(dict(kwargs))
        sample["was_rejected"] = False
        sample["rejection_label"] = 0.0
        self.rejection_buffer.append(sample)
        return sample

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

    def _edge_tensor_from_experience(
        self,
        exp: dict,
        *,
        next_state: bool = False,
        candidate: dict | None = None,
        target_context: bool = False,
        state_snapshot: SystemSnapshot | None = None,
    ):
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
        queue_wait_feature = self._queue_wait_feature_from_experience(
            exp, candidate, target_context=target_context
        )
        feature_source = exp if candidate is None else candidate
        post_demand_feature = feature_source.get(
            "post_demand_feature", exp.get("post_demand_feature")
        )
        vehicle_id = int(exp.get("vehicle_id", -1))
        if state_snapshot is None:
            state_snapshot = exp.get(
                "next_state_snapshot" if next_state else "state_snapshot"
            )
        if isinstance(state_snapshot, SystemSnapshot):
            vehicle_type_for_mask = int(exp.get("vehicle_type", self._vehicle_type(vehicle_id)))
            state_snapshot = state_snapshot.masked(
                str(exp.get("state_variant", self.state_variant)),
                vehicle_type=vehicle_type_for_mask,
            )
            vehicle_type = next(
                (
                    vehicle.vehicle_type
                    for vehicle in state_snapshot.vehicles
                    if vehicle.vehicle_id == vehicle_id
                ),
                vehicle_type_for_mask,
            )
        else:
            vehicle_type = int(exp.get("vehicle_type", self._vehicle_type(vehicle_id)))
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
            graph_snapshot=state_snapshot,
            target_context=target_context,
            vehicle_types=np.asarray([vehicle_type], dtype=np.int64),
            post_demand_features=(
                None
                if post_demand_feature is None
                else np.asarray([float(post_demand_feature)], dtype=np.float32)
            ),
            rejection_probabilities=[self.rejection_from_experience(exp, candidate, next_state=next_state)],
            human_response_masks=[self.response_mask_from_experience(exp, candidate, next_state=next_state)],
        )
        g = self._myopic_score(
            action_id,
            request_value,
            target_distance,
            post_distance,
        )
        q, mask = self.response_from_experience(exp, candidate, next_state=next_state)
        g, _ = self.response_anchor(g, request_value, target_distance, q, mask)
        return edge_t, type_w, g

    def set_joint_critic_router(
        self,
        *,
        ev_value_function: "PyTorchChargingValueFunction",
        aev_value_function: "PyTorchChargingValueFunction",
    ) -> None:
        """Route every graph edge through the critic deployed for its fleet."""

        self._joint_critic_router = {
            1: ev_value_function,
            2: aev_value_function,
        }

    def set_follower_target_provider(self, provider) -> None:
        """Install the lagged AEV stage-value provider used by an R4 leader."""

        self._follower_target_provider = provider

    def validate_recourse_wiring(self) -> None:
        router = getattr(self, "_joint_critic_router", {})
        if set(router) != {1, 2}:
            raise RuntimeError("recourse critic router was not fully wired")
        if str(getattr(self, "state_variant", "")).endswith("separate_critics") and (
            router[1] is router[2]
        ):
            raise RuntimeError("separate critics were not wired")
        if (
            str(getattr(self, "recourse_variant", "legacy")) == "r4"
            and router.get(1) is self
            and self._follower_target_provider is None
        ):
            raise RuntimeError("R4 follower target provider was not wired")
        self._recourse_wiring_validated = True

    def _provider_for_edge(
        self, edge: FeasibleEdgeSnapshot
    ) -> "PyTorchChargingValueFunction":
        router = getattr(self, "_joint_critic_router", {})
        provider = router.get(int(edge.vehicle_type))
        if provider is None:
            raise RuntimeError(
                f"no critic provider wired for vehicle type {edge.vehicle_type}"
            )
        return provider

    @staticmethod
    def _edge_experience(
        graph: FeasibleGraphSnapshot,
        edge: FeasibleEdgeSnapshot,
        *,
        state_variant: str,
    ) -> dict:
        vehicle = next(
            item for item in graph.state.vehicles if item.vehicle_id == edge.vehicle_id
        )
        return {
            "vehicle_id": edge.vehicle_id,
            "vehicle_type": edge.vehicle_type,
            "action_type": edge.action_id,
            "action_type_id": int(edge.action_type),
            "vehicle_location": vehicle.location,
            "target_location": edge.target_location,
            "post_action_location": edge.post_action_location,
            "current_time": graph.state.current_time,
            "battery_level": vehicle.battery,
            "vehicle_idle_time": vehicle.idle_time,
            "other_vehicles": sum(item.online for item in graph.state.vehicles) - 1,
            "num_requests": len(graph.state.requests),
            "request_value": edge.request_value,
            "target_distance": edge.target_distance,
            "post_action_distance": edge.post_action_distance,
            "post_action_duration": edge.post_action_duration,
            "target_station_id": edge.station_id,
            "queue_features": edge.queue_features,
            "post_demand_feature": edge.post_demand_feature,
            "rejection_probability": edge.rejection_probability,
            "human_response_mask": edge.human_response_mask,
            "response_model_hash": edge.response_model_hash,
            "state_snapshot": graph.state,
            "state_variant": state_variant,
        }

    def _correction_bound(
        self,
        graph: FeasibleGraphSnapshot,
        edge: FeasibleEdgeSnapshot,
        *,
        target_context: bool,
    ) -> float:
        bounds = self._correction_bounds_for_edges(
            graph, (edge,), target_context=target_context
        )
        return float(bounds[0].detach().item())

    def _correction_bounds_for_edges(
        self,
        graph: FeasibleGraphSnapshot,
        edges: tuple[FeasibleEdgeSnapshot, ...] | list[FeasibleEdgeSnapshot],
        *,
        target_context: bool,
    ) -> torch.Tensor:
        """Build all deployment bounds with at most one queue forward pass."""

        structured = np.asarray(
            [float(item.structured_score) for item in graph.edges],
            dtype=np.float32,
        )
        sigma_g = max(1.0, float(np.std(structured)))
        bounds = torch.full(
            (len(edges),),
            float(self.residual_clip_rho) * sigma_g,
            dtype=torch.float32,
            device=self.device,
        )
        charge_positions = [
            index
            for index, edge in enumerate(edges)
            if edge.action_type == ActionType.CHARGE
        ]
        if not charge_positions:
            return bounds
        charge_duration = max(
            1.0,
            float(
                getattr(self.env, "charge_duration", 1.0)
                if self.env is not None
                else 1.0
            ),
        )
        charging_cost = float(
            getattr(
                self.env,
                "charging_penalty_per_step",
                getattr(self.env, "charging_penalty", 0.0),
            )
            if self.env is not None
            else 0.0
        )
        predicted_waits = torch.zeros(
            len(charge_positions), dtype=torch.float32, device=self.device
        )
        if self.queue_predictor_trained:
            valid_offsets = []
            feature_rows = []
            for offset, edge_index in enumerate(charge_positions):
                row = tuple(float(value) for value in edges[edge_index].queue_features)
                if len(row) != self.queue_feature_dim:
                    continue
                valid_offsets.append(offset)
                feature_rows.append(row)
            if feature_rows:
                predictor = (
                    self.target_queue_predictor
                    if target_context
                    else self.queue_predictor
                )
                feature_tensor = torch.tensor(
                    feature_rows, dtype=torch.float32, device=self.device
                )
                waits = torch.relu(predictor(feature_tensor).squeeze(1))
                predicted_waits[
                    torch.tensor(
                        valid_offsets, dtype=torch.long, device=self.device
                    )
                ] = waits
        charge_bounds = (
            charging_cost * charge_duration
            + float(self._queue_penalty_per_step()) * predicted_waits
        )
        positions = torch.tensor(
            charge_positions, dtype=torch.long, device=self.device
        )
        bounds[positions] = torch.maximum(bounds[positions], charge_bounds)
        return bounds

    def _deployed_correction(
        self,
        raw_value: torch.Tensor,
        graph: FeasibleGraphSnapshot,
        edge: FeasibleEdgeSnapshot,
        *,
        target_context: bool,
    ) -> torch.Tensor:
        if getattr(self, "direct_q", False):
            return raw_value
        bound = self._correction_bound(
            graph, edge, target_context=target_context
        )
        return float(self._beta()) * torch.clamp(raw_value, -bound, bound)

    def _edge_raw_tensors(
        self,
        graph: FeasibleGraphSnapshot,
        edge: FeasibleEdgeSnapshot,
        *,
        target_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge.response_model_hash != self.response_model_hash:
            raise ValueError('Replay rejection predictor hash mismatch')
        exp = self._edge_experience(
            graph, edge, state_variant=self.state_variant
        )
        edge_tensor, type_weight, _ = self._edge_tensor_from_experience(
            exp,
            target_context=target_context,
            state_snapshot=graph.state,
        )
        critic1 = self.target_network if target_context else self.network
        critic2 = self.target_critic2 if target_context else self.critic2
        raw1 = critic1(edge_tensor) * type_weight
        raw2 = critic2(edge_tensor) * type_weight
        return raw1, raw2, type_weight

    def _edge_correction_tensors(
        self,
        graph: FeasibleGraphSnapshot,
        edge: FeasibleEdgeSnapshot,
        *,
        target_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw1, raw2, type_weight = self._edge_raw_tensors(
            graph, edge, target_context=target_context
        )
        return (
            self._deployed_correction(
                raw1, graph, edge, target_context=target_context
            ),
            self._deployed_correction(
                raw2, graph, edge, target_context=target_context
            ),
            type_weight,
        )

    def _graph_edge_scores(
        self,
        graph: FeasibleGraphSnapshot,
        *,
        target_context: bool,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return rollout scores or unbounded lagged Bellman corrections.

        Neural inference is grouped by fleet provider, avoiding one forward
        pass and one device synchronization per feasible edge.
        """

        full_scores: dict[str, float] = {}
        correction_scores: dict[str, float] = {}
        grouped: dict[int, tuple[Any, list[FeasibleEdgeSnapshot]]] = {}
        for edge in graph.edges:
            provider = self._provider_for_edge(edge)
            if edge.response_model_hash != provider.response_model_hash:
                raise ValueError('Replay rejection predictor hash mismatch')
            grouped.setdefault(id(provider), (provider, []))[1].append(edge)

        for provider, provider_edges in grouped.values():
            edge_rows = []
            type_rows = []
            for edge in provider_edges:
                exp = provider._edge_experience(
                    graph, edge, state_variant=provider.state_variant
                )
                edge_tensor, type_weight, _ = provider._edge_tensor_from_experience(
                    exp,
                    target_context=target_context,
                    state_snapshot=graph.state,
                )
                edge_rows.append(edge_tensor.squeeze(0))
                type_rows.append(type_weight.reshape(()))
            edges_tensor = torch.stack(edge_rows)
            type_weights = torch.stack(type_rows).reshape(-1, 1)
            critic1 = provider.target_network if target_context else provider.network
            critic2 = provider.target_critic2 if target_context else provider.critic2
            raw1 = critic1(edges_tensor) * type_weights
            raw2 = critic2(edges_tensor) * type_weights
            if target_context:
                # Target evaluation is deliberately not beta-scaled or
                # clamped: raw critics fit the unbounded Bellman residual.
                corrections = torch.minimum(raw1, raw2).reshape(-1)
            else:
                selected_raw = provider._selection_residual(
                    raw1,
                    raw2,
                    torch.ones_like(type_weights),
                ).reshape(-1)
                if getattr(provider, "direct_q", False):
                    corrections = selected_raw
                else:
                    bounds = provider._correction_bounds_for_edges(
                        graph, provider_edges, target_context=False
                    )
                    if provider.planning_objective_mode != "structured_only":
                        provider.deployment_edges_scored = getattr(provider, "deployment_edges_scored", 0) + len(provider_edges)
                        provider.deployment_edges_clipped = getattr(provider, "deployment_edges_clipped", 0) + int(
                            (selected_raw.detach().abs() > bounds).sum().item())
                    corrections = float(provider._beta()) * torch.clamp(
                        selected_raw, min=-bounds, max=bounds
                    )
            correction_values = corrections.detach().cpu().numpy()
            for edge, correction_value in zip(
                provider_edges, correction_values
            ):
                correction_value = float(correction_value)
                correction_scores[edge.edge_id] = correction_value
                if provider.planning_objective_mode == "structured_only":
                    full_scores[edge.edge_id] = float(edge.structured_score)
                elif getattr(provider, "direct_q", False):
                    full_scores[edge.edge_id] = correction_value
                else:
                    full_scores[edge.edge_id] = float(
                        edge.structured_score + correction_value
                    )
        return full_scores, correction_scores

    def target_components_for_graph(
        self,
        graph: FeasibleGraphSnapshot | None,
        *,
        structured_only: bool = False,
    ):
        if graph is None or not graph.edges:
            from src.recourse.target_builder import TargetComponents

            return TargetComponents((), 0.0, 0.0, 0.0, solver_status="empty")
        routed_providers = {
            id(self._provider_for_edge(edge)): self._provider_for_edge(edge)
            for edge in graph.edges
        }
        model_signature = tuple(
            sorted(
                (
                    provider_id,
                    int(getattr(provider, "training_step", 0)),
                    int(getattr(provider, "joint_training_step", 0)),
                    int(getattr(provider, "optimizer_steps_queue", 0)),
                )
                for provider_id, provider in routed_providers.items()
            )
        )
        cache_key = (
            graph.graph_id,
            bool(structured_only),
            model_signature,
            int(graph.objective_cost_scale),
        )
        component_cache = getattr(self, "_target_component_cache", None)
        if component_cache is None:
            component_cache = {}
            self._target_component_cache = component_cache
        cached = component_cache.get(cache_key)
        if cached is not None:
            return cached
        with torch.no_grad():
            online_full, _ = self._graph_edge_scores(
                graph, target_context=False
            )
            _, target_correction = self._graph_edge_scores(
                graph, target_context=True
            )
            direct_q = bool(getattr(self, "direct_q", False))
            components = self.target_builder.double_q_target(
                graph,
                online_scores=online_full,
                target_scores=target_correction,
                structured_only=structured_only,
                direct_q=direct_q,
            )
        if len(component_cache) >= 2_048:
            component_cache.clear()
        component_cache[cache_key] = components
        return components

    def _solver_consistent_residual_value(
        self,
        graph: FeasibleGraphSnapshot | None,
        *,
        structured_only: bool = False,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        components = self.target_components_for_graph(
            graph, structured_only=structured_only
        )
        value = torch.tensor(
            components.target_full_value,
            dtype=torch.float32,
            device=self.device,
        )
        return value, components.selected_edge_ids

    def _next_soft_values(self, batch: list[dict]) -> torch.Tensor:
        values = []
        actor_losses = []
        alpha_terms = []
        with torch.no_grad():
            target_net = self.target_network
            target_net2 = self.target_critic2
        for exp in batch:
            stage_id = int(exp.get("stage_id", 0) or 0)
            recourse_variant = str(exp.get("recourse_variant", "legacy"))
            if stage_id == 1 and recourse_variant == "r4":
                follower_graph = exp.get("aev_stage_graph")
                components = self._r4_follower_components(follower_graph)
                follower_value = torch.tensor(
                    components.target_full_value,
                    dtype=torch.float32,
                    device=self.device,
                )
                joint_action = exp.get("joint_action_snapshot")
                leader_edges = len(
                    getattr(joint_action, "selected_edge_ids", ()) or ()
                )
                values.append(follower_value / float(max(1, leader_edges)))
                continue
            if exp.get("is_system_done", False) or exp.get("is_vehicle_done", False):
                values.append(torch.zeros((), dtype=torch.float32, device=self.device))
                continue
            next_graph = exp.get("next_feasible_graph_snapshot")
            if isinstance(next_graph, FeasibleGraphSnapshot):
                joint_value, selected = self._solver_consistent_residual_value(
                    next_graph
                )
                # Edge replay is retained for backwards compatibility.  The
                # primary joint loss below consumes the undivided value.
                values.append(joint_value / float(max(1, len(selected))))
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
            logits = self.actor(self._actor_features(edges)).squeeze(1)
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

    def _selected_correction_tensors(
        self,
        graph: FeasibleGraphSnapshot,
        selected_edge_ids: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple["PyTorchChargingValueFunction", ...]]:
        selected = set(selected_edge_ids)
        values1 = []
        values2 = []
        providers: dict[int, "PyTorchChargingValueFunction"] = {}
        for edge in graph.edges:
            if edge.edge_id not in selected:
                continue
            provider = self._provider_for_edge(edge)
            providers[id(provider)] = provider
            provider._graph_cache_key = None
            provider._graph_cache = None
            correction1, correction2, _ = provider._edge_correction_tensors(
                graph, edge, target_context=False
            )
            values1.append(correction1.reshape(()))
            values2.append(correction2.reshape(()))
        if not values1:
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, zero, tuple(providers.values())
        return (
            torch.stack(values1).sum(),
            torch.stack(values2).sum(),
            tuple(providers.values()),
        )

    def _selected_raw_tensors(
        self,
        graph: FeasibleGraphSnapshot,
        selected_edge_ids: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple["PyTorchChargingValueFunction", ...]]:
        """Return the unbounded online twin predictions used by TD loss."""

        selected = set(selected_edge_ids)
        values1 = []
        values2 = []
        providers: dict[int, "PyTorchChargingValueFunction"] = {}
        for edge in graph.edges:
            if edge.edge_id not in selected:
                continue
            provider = self._provider_for_edge(edge)
            providers[id(provider)] = provider
            provider._graph_cache_key = None
            provider._graph_cache = None
            raw1, raw2, _ = provider._edge_raw_tensors(
                graph, edge, target_context=False
            )
            values1.append(raw1.reshape(()))
            values2.append(raw2.reshape(()))
        if not values1:
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, zero, tuple(providers.values())
        return (
            torch.stack(values1).sum(),
            torch.stack(values2).sum(),
            tuple(providers.values()),
        )

    def _selected_residual_tensor(
        self,
        graph: FeasibleGraphSnapshot,
        selected_edge_ids: tuple[str, ...],
    ) -> torch.Tensor:
        """Compatibility wrapper returning the conservative online correction."""

        q1, q2, _ = self._selected_correction_tensors(
            graph, selected_edge_ids
        )
        return torch.minimum(q1, q2)

    def _temporal_successor_graph(
        self,
        transition: RecourseTransition,
    ) -> FeasibleGraphSnapshot | None:
        self.next_transition_link_lookups += 1
        next_id = transition.next_transition_id
        if not next_id:
            self.next_transition_link_misses += int(not transition.done)
            return None
        candidate = self.joint_replay_buffer.get_by_transition_id(next_id)
        if candidate is None:
            self.next_transition_link_misses += 1
            return None
        expected = (
            transition.run_id,
            transition.cumulative_episode_id,
            transition.transition_sequence_index + 1,
            transition.mode,
            transition.state_variant,
            transition.learner_variant,
        )
        actual = (
            candidate.run_id,
            candidate.cumulative_episode_id,
            candidate.transition_sequence_index,
            candidate.mode,
            candidate.state_variant,
            candidate.learner_variant,
        )
        if actual != expected:
            raise AssertionError(
                "direct replay transition link has incompatible identity: "
                f"expected={expected}, actual={actual}"
            )
        if candidate.mode in {"integrated", "integrated_repair"}:
            return candidate.ev_stage_graph
        if candidate.mode == "ev_first":
            # The AEV follower ends at the next epoch's EV leader phase.
            return candidate.ev_stage_graph
        if candidate.mode == "aev_first":
            # Symmetrically, the EV follower ends at the next AEV leader.
            return candidate.aev_stage_graph
        raise ValueError(f"unsupported temporal mode: {candidate.mode}")

    def _next_transition_graph(
        self,
        transition: RecourseTransition,
        *,
        fleet: str | None = None,
    ) -> FeasibleGraphSnapshot | None:
        """Compatibility wrapper; phase, not fleet, determines succession."""

        del fleet
        return self._temporal_successor_graph(transition)

    def _joint_stage_payload(
        self,
        transition: RecourseTransition,
        *,
        ifEV: bool,
    ):
        if transition.mode in {"integrated", "integrated_repair"}:
            if ifEV:
                return None
            return (
                transition.ev_stage_graph,
                transition.ev_joint_action,
                float(transition.reward_system),
                "system",
            )
        if transition.mode == "ev_first":
            if ifEV:
                return (
                    transition.ev_stage_graph,
                    transition.ev_joint_action,
                    float(transition.reward_system if transition.recourse_target_family == "macro_realized" else transition.reward_ev),
                    "ev_leader",
                )
            if transition.recourse_variant == "r2":
                return None
            return (
                transition.aev_stage_graph,
                transition.aev_joint_action,
                float(transition.reward_aev),
                "aev_follower",
            )
        if transition.mode == "aev_first":
            if ifEV:
                return (
                    transition.ev_stage_graph,
                    transition.ev_joint_action,
                    float(transition.reward_ev),
                    "ev_follower",
                )
            return (
                transition.aev_stage_graph,
                transition.aev_joint_action,
                float(transition.reward_aev),
                "aev_leader",
            )
        raise ValueError(f"unsupported joint transition mode: {transition.mode}")

    def _joint_row_ready(
        self,
        transition: RecourseTransition,
        *,
        ifEV: bool,
    ) -> bool:
        payload = self._joint_stage_payload(transition, ifEV=ifEV)
        if payload is None:
            return False
        graph, action, _reward, phase = payload
        if graph is None or action is None or not action.selected_edge_ids:
            return False
        if transition.done:
            return True
        if (
            transition.mode == "ev_first"
            and phase == "ev_leader"
            and transition.recourse_target_family == "nested_follower"
        ):
            return transition.aev_stage_graph is not None
        if transition.mode == "aev_first" and phase == "aev_leader":
            return transition.ev_stage_graph is not None
        next_id = transition.next_transition_id
        return bool(
            next_id
            and self.joint_replay_buffer.get_by_transition_id(next_id) is not None
        )

    def has_trainable_joint_rows(self, *, ifEV: bool) -> bool:
        return any(
            self._joint_row_ready(transition, ifEV=ifEV)
            for transition in self.joint_replay_buffer
        )

    def _r4_follower_components(self, graph: FeasibleGraphSnapshot | None):
        provider = getattr(self, "_follower_target_provider", None)
        if provider is None:
            raise RuntimeError("R4 follower target provider was not wired")
        return provider(graph, structured_only=False)

    def _train_joint_step(self, batch_size: int, *, ifEV: bool) -> float:
        if len(self.joint_replay_buffer) == 0:
            return 0.0
        sample = self.joint_replay_buffer.sample_ready(
            min(batch_size, len(self.joint_replay_buffer)),
            predicate=lambda transition: self._joint_row_ready(
                transition, ifEV=ifEV
            ),
        )
        if not sample.transitions:
            return 0.0
        losses = []
        td_errors = []
        used_indices = []
        providers_to_step: dict[int, "PyTorchChargingValueFunction"] = {}
        diagnostics = []
        max_edges_per_update = max(
            1,
            int(
                getattr(self.env, "max_joint_target_edges_per_update", 20_000)
                if self.env is not None
                else 20_000
            ),
        )
        consumed_edges = 0
        for transition, replay_index, importance_weight in zip(
            sample.transitions, sample.indices, sample.weights
        ):
            fleet = "ev" if ifEV else "aev"
            payload = self._joint_stage_payload(transition, ifEV=ifEV)
            if payload is None:
                continue
            graph, action, reward, phase = payload
            if transition.state_variant != self.state_variant:
                raise ValueError(
                    "joint replay state variant mismatch: "
                    f"row={transition.state_variant}, learner={self.state_variant}"
                )
            if transition.learner_variant != self.learner_variant:
                raise ValueError(
                    "joint replay learner variant mismatch: "
                    f"row={transition.learner_variant}, learner={self.learner_variant}"
                )
            if graph is None or action is None or not action.selected_edge_ids:
                continue
            graph_edge_count = len(graph.edges)
            if losses and consumed_edges + graph_edge_count > max_edges_per_update:
                break
            consumed_edges += graph_edge_count
            prediction1, prediction2, providers = self._selected_raw_tensors(
                graph, action.selected_edge_ids
            )
            for provider in providers:
                providers_to_step[id(provider)] = provider
            structured_value = float(action.structured_value)

            continuation_full_value = 0.0
            continuation_discount = 0.0
            target_components = None
            target_graph_for_diagnostics = None
            if (
                transition.mode == "ev_first"
                and phase == "ev_leader"
                and transition.recourse_target_family == "nested_follower"
            ):
                # The follower action is inside the current epoch and is
                # evaluated by the explicitly wired AEV lagged target critic.
                target_components = self._r4_follower_components(
                    transition.aev_stage_graph
                )
                target_graph_for_diagnostics = transition.aev_stage_graph
                continuation_full_value = float(
                    target_components.target_full_value
                )
                continuation_discount = self.within_epoch_gamma
            elif transition.mode == "aev_first" and phase == "aev_leader":
                target_graph_for_diagnostics = transition.ev_stage_graph
                target_components = self.target_components_for_graph(
                    transition.ev_stage_graph,
                    structured_only=False,
                )
                continuation_full_value = float(
                    target_components.target_full_value
                )
                continuation_discount = self.within_epoch_gamma
            elif not transition.done:
                next_graph = self._temporal_successor_graph(transition)
                if next_graph is None:
                    # A nonterminal miss is never interpreted as terminal.
                    continue
                target_graph_for_diagnostics = next_graph
                target_components = self.target_components_for_graph(
                    next_graph,
                    structured_only=(
                        transition.recourse_variant == "r2"
                        and fleet == "aev"
                    ),
                )
                continuation_full_value = float(
                    target_components.target_full_value
                )
                continuation_discount = self.gamma ** float(
                    transition.elapsed_epochs
                )
            full_target_value = float(reward) + (
                continuation_discount * continuation_full_value
            )
            target_value = self.target_builder.correction_bellman_target(
                reward=float(reward),
                discount=continuation_discount,
                next_components=target_components,
                current_structured_value=structured_value,
                direct_q=bool(getattr(self, "direct_q", False)),
            )
            target = torch.tensor(target_value, dtype=torch.float32, device=self.device)
            loss1 = F.smooth_l1_loss(
                prediction1, target.detach(), beta=self.huber_kappa
            )
            loss2 = F.smooth_l1_loss(
                prediction2, target.detach(), beta=self.huber_kappa
            )
            losses.append(float(importance_weight) * (loss1 + loss2))
            td1 = target.detach() - prediction1.detach()
            td2 = target.detach() - prediction2.detach()
            td_errors.append(
                0.5 * (float(abs(td1.item())) + float(abs(td2.item())))
            )
            used_indices.append(int(replay_index))
            diagnostics.append({
                "joint_q1_loss": float(loss1.detach().item()),
                "joint_q2_loss": float(loss2.detach().item()),
                "joint_target_full": float(full_target_value),
                "leader_recourse_credit": float(transition.reward_aev) if phase == "ev_leader" and transition.recourse_target_family == "macro_realized" else 0.0,
                "recourse_target_family": transition.recourse_target_family,
                "phase": phase,
                "joint_residual_target": float(target_value),
                "joint_prediction_abs": float(0.5 * (prediction1.detach().abs() + prediction2.detach().abs()).item()),
                "follower_target": float(full_target_value) if phase == "aev_follower" else 0.0,
                "follower_residual": float(prediction1.detach().item()) if phase == "aev_follower" else 0.0,
                "joint_target_structured": float(
                    0.0
                    if target_components is None
                    else target_components.target_structured_value
                ),
                "joint_target_correction": float(
                    0.0
                    if target_components is None
                    else target_components.target_correction_value
                ),
                "r4_follower_full_value": float(
                    continuation_full_value
                    if transition.recourse_variant == "r4" and ifEV
                    else 0.0
                ),
                "target_projection_runtime": float(
                    0.0
                    if target_components is None
                    else target_components.solver_runtime_seconds
                ),
                "joint_target_projection_time": float(
                    0.0
                    if target_components is None
                    else target_components.solver_runtime_seconds
                ),
                "edges_per_update": int(
                    0
                    if target_graph_for_diagnostics is None
                    else len(target_graph_for_diagnostics.edges)
                ),
                "milps_per_update": 0,
                "target_rollout_action_agreement": float(
                    0.0
                    if target_components is None
                    or not getattr(
                        target_graph_for_diagnostics,
                        "selected_edge_ids",
                        (),
                    )
                    else set(target_components.selected_edge_ids)
                    == set(
                        getattr(
                            target_graph_for_diagnostics,
                            "selected_edge_ids",
                            (),
                        )
                    )
                ),
                "next_transition_link_miss_rate": float(
                    self.next_transition_link_misses
                    / max(1, self.next_transition_link_lookups)
                ),
            })
        if not losses:
            return 0.0
        loss = torch.stack(losses).mean()
        for provider in providers_to_step.values():
            provider.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for provider in providers_to_step.values():
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in provider.graph_encoder.parameters()
                    if parameter.requires_grad
                ]
                + list(provider.mixer.parameters())
                + list(provider.network.parameters())
                + list(provider.critic2.parameters()),
                max_norm=self.gradient_clip_norm,
            )
            provider.joint_gradient_clip_count = getattr(provider, "joint_gradient_clip_count", 0) + int(
                float(grad_norm) > self.gradient_clip_norm)
            provider.optimizer.step()
            provider.joint_training_step = int(
                getattr(provider, "joint_training_step", 0)
            ) + 1
            provider.optimizer_steps_joint = int(
                getattr(provider, "optimizer_steps_joint", 0)
            ) + 1
            provider.optimizer_steps_total = int(
                getattr(provider, "optimizer_steps_total", 0)
            ) + 1
            provider._soft_update(provider.network, provider.target_network, provider.tau)
            provider._soft_update(provider.critic2, provider.target_critic2, provider.tau)
            provider._soft_update(
                provider.graph_encoder, provider.target_graph_encoder, provider.tau
            )
            provider._soft_update(provider.mixer, provider.target_mixer, provider.tau)
            provider._soft_update(
                provider.queue_predictor,
                provider.target_queue_predictor,
                provider.tau,
            )
        self.joint_replay_buffer.update_priorities(used_indices, td_errors)
        self.joint_replay_buffer.advance_beta()
        self.joint_training_diagnostics.extend(diagnostics)
        if len(self.joint_training_diagnostics) > 10_000:
            del self.joint_training_diagnostics[:-10_000]
        return float(loss.detach().item())

    def train_step(self, batch_size: int = 64, tau: float | None = None, ifEV: bool = False) -> float:
        if not ifEV and str(getattr(self, 'recourse_variant', 'legacy')) == 'r2':
            # Repair Only must not update the follower, including auxiliaries.
            return 0.0
        if str(getattr(self, 'recourse_variant', 'legacy')) in {'r2', 'r3', 'r4', 'recourse_macro'} or str(self.learner_variant) in {
            "optimization_anchored_residual",
            "integrated_directq",
        }:
            # These learners have one identifiable Bellman source: the full
            # joint transition.  Legacy edge rows remain available to
            # auxiliary predictors and diagnostics, but never impose an
            # arbitrary joint-value/edge-count TD target on the same critic.
            queue_loss = (
                self.train_queue_predictor(batch_size=batch_size)
                if (
                    not ifEV
                    or not bool(
                        getattr(self, "_owns_joint_replay_payload", True)
                    )
                )
                else 0.0
            )
            joint_loss = self._train_joint_step(batch_size, ifEV=ifEV)
            weighted_queue_loss = float(self.queue_loss_weight) * queue_loss
            if queue_loss > 0:
                self.training_losses.append(weighted_queue_loss)
            return weighted_queue_loss + joint_loss
        if len(self.experience_buffer) < max(8, batch_size // 2):
            queue_only_loss = self.train_queue_predictor(batch_size=batch_size)
            joint_loss = self._train_joint_step(batch_size, ifEV=ifEV)
            if queue_only_loss > 0:
                self.training_losses.append(float(self.queue_loss_weight) * queue_only_loss)
            return float(self.queue_loss_weight) * queue_only_loss + joint_loss
        self._graph_cache_key = None
        self._graph_cache = None
        fleet_type = 1 if ifEV else 2
        fleet_rows = [
            exp
            for exp in self.experience_buffer
            if int(exp.get("vehicle_type", fleet_type)) == fleet_type
        ]
        if not fleet_rows:
            return self._train_joint_step(batch_size, ifEV=ifEV)
        batch = random.sample(fleet_rows, min(batch_size, len(fleet_rows)))
        if not ifEV:
            batch = [
                exp
                for exp in batch
                if not (
                    int(exp.get("stage_id", 0) or 0) == 2
                    and str(exp.get("recourse_variant", "legacy")) == "r2"
                )
            ]
        if not batch:
            return self._train_joint_step(batch_size, ifEV=ifEV)
        tau = self.tau if tau is None else float(tau)

        edge_rows = []
        type_ws = []
        g_values = []
        rewards = []
        durs = []
        masks = []
        action_ids = []
        queue_wait_features = []
        for exp in batch:
            edge_t, type_w, g = self._edge_tensor_from_experience(exp)
            edge_rows.append(edge_t.squeeze(0))
            type_ws.append(type_w.squeeze(0))
            g_values.append(float(g))
            rewards.append(float(exp.get("reward", 0.0)) - float(g))
            durs.append(float(exp.get("dur_time", 1.0)))
            is_stage_coupled = (
                int(exp.get("stage_id", 0) or 0) == 1
                and str(exp.get("recourse_variant", "legacy")) == "r4"
            )
            masks.append(
                1.0
                if is_stage_coupled
                else (
                    0.0
                    if (
                        exp.get("is_system_done", False)
                        or exp.get("is_vehicle_done", False)
                    )
                    else 1.0
                )
            )
            action_ids.append(
                int(
                    exp.get(
                        "action_type_id",
                        self._action_id(exp.get("action_type", "wait")),
                    )
                )
            )
            queue_wait_features.append(
                self._queue_wait_feature_from_experience(exp)
            )

        edges = torch.stack(edge_rows)
        weights = torch.stack(type_ws)
        g_t = torch.tensor(g_values, dtype=torch.float32, device=self.device).unsqueeze(1)
        residual_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        durs_t = torch.tensor(durs, dtype=torch.float32, device=self.device).unsqueeze(1)
        masks_t = torch.tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)

        raw_q1 = self.network(edges) * weights
        raw_q2 = self.critic2(edges) * weights
        sigma_g = torch.std(g_t, unbiased=False).clamp_min(1.0)
        bounds = torch.full_like(
            raw_q1, float(self.residual_clip_rho) * float(sigma_g.item())
        )
        charge_mask = torch.tensor(
            np.asarray(action_ids) == int(ActionType.CHARGE),
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(1)
        if torch.any(charge_mask):
            charge_duration = max(
                1.0,
                float(
                    getattr(self.env, "charge_duration", 1.0)
                    if self.env is not None
                    else 1.0
                ),
            )
            charging_cost = float(
                getattr(
                    self.env,
                    "charging_penalty_per_step",
                    getattr(self.env, "charging_penalty", 0.0),
                )
                if self.env is not None
                else 0.0
            )
            waits = torch.tensor(
                queue_wait_features,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1) * charge_duration
            charge_bounds = charging_cost * charge_duration + (
                self._queue_penalty_per_step() * waits
            )
            bounds = torch.where(
                charge_mask, torch.maximum(bounds, charge_bounds), bounds
            )
        score1 = self._execution_scores_from_residual(g_t, raw_q1, bounds)
        score2 = self._execution_scores_from_residual(g_t, raw_q2, bounds)
        if getattr(self, "direct_q", False):
            q1 = score1
            q2 = score2
        else:
            q1 = score1 - g_t
            q2 = score2 - g_t
        next_v = self._next_soft_values(batch)
        bootstrap_discounts = []
        for exp, duration in zip(batch, durs):
            if int(exp.get("stage_id", 0) or 0) == 1 and str(
                exp.get("recourse_variant", "legacy")
            ) == "r4":
                bootstrap_discounts.append(1.0)
            else:
                bootstrap_discounts.append(self.gamma ** float(duration))
        discount_t = torch.tensor(
            bootstrap_discounts, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        target = residual_rewards + discount_t * next_v * masks_t
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
        self.optimizer_steps_edge += 1
        self.optimizer_steps_total += 1
        if queue_loss is not None:
            self.queue_optimizer.step()
            self.optimizer_steps_queue += 1
            self.queue_predictor_trained = True
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(1e-4), math.log(10.0))
        self._soft_update(self.network, self.target_network, tau)
        self._soft_update(self.critic2, self.target_critic2, tau)
        self._soft_update(self.graph_encoder, self.target_graph_encoder, tau)
        self._soft_update(self.mixer, self.target_mixer, tau)
        self._soft_update(
            self.queue_predictor, self.target_queue_predictor, tau
        )

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
        joint_loss = self._train_joint_step(batch_size, ifEV=ifEV)
        if self.q_values_history:
            self.q_values_history[-1]["joint_loss"] = joint_loss
        return loss_value + joint_loss

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    # ------------------------------------------------------------------
    # Checkpoint extras
    # ------------------------------------------------------------------

    def extra_checkpoint_state(self) -> dict[str, Any]:
        joint_replay_state = self.joint_replay_buffer.state_dict(
            mode=(
                self.checkpoint_replay
                if bool(getattr(self, "_owns_joint_replay_payload", True))
                else "none"
            ),
            recent_count=self.checkpoint_replay_recent,
        )
        return {
            "critic2_state_dict": self.critic2.state_dict(),
            "ev_response": self.acceptance_checkpoint_state(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "graph_encoder_state_dict": self.graph_encoder.state_dict(),
            "target_graph_encoder_state_dict": self.target_graph_encoder.state_dict(),
            "mixer_state_dict": self.mixer.state_dict(),
            "target_mixer_state_dict": self.target_mixer.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "queue_predictor_state_dict": self.queue_predictor.state_dict(),
            "target_queue_predictor_state_dict": self.target_queue_predictor.state_dict(),
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
            "joint_replay_schema_version": 3,
            "joint_replay_state_dict": joint_replay_state,
            "joint_replay_hash": joint_replay_state["content_hash"],
            "checkpoint_replay": self.checkpoint_replay,
            "checkpoint_replay_recent": self.checkpoint_replay_recent,
            "state_variant": self.state_variant,
            "learner_variant": self.learner_variant,
            "recourse_variant": getattr(self, "recourse_variant", "legacy"),
            "joint_training_diagnostics": list(self.joint_training_diagnostics),
            "next_transition_link_misses": int(self.next_transition_link_misses),
            "next_transition_link_lookups": int(self.next_transition_link_lookups),
            "joint_training_step": int(self.joint_training_step),
            "optimizer_steps_total": int(self.optimizer_steps_total),
            "optimizer_steps_joint": int(self.optimizer_steps_joint),
            "optimizer_steps_edge": int(self.optimizer_steps_edge),
            "optimizer_steps_queue": int(self.optimizer_steps_queue),
        }

    def load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.load_acceptance_checkpoint_state(state)
        if state.get('joint_replay_schema_version', 3) != 3:
            raise ValueError('Legacy joint replay schema; rejected=1 v3 checkpoint required')
        if not state:
            return
        if "critic2_state_dict" in state:
            self.critic2.load_state_dict(state["critic2_state_dict"], strict=False)
        if "target_critic2_state_dict" in state:
            self.target_critic2.load_state_dict(state["target_critic2_state_dict"], strict=False)
        if "graph_encoder_state_dict" in state:
            self.graph_encoder.load_state_dict(state["graph_encoder_state_dict"], strict=False)
        if "target_graph_encoder_state_dict" in state:
            self.target_graph_encoder.load_state_dict(
                state["target_graph_encoder_state_dict"], strict=False
            )
        else:
            self.target_graph_encoder.load_state_dict(
                self.graph_encoder.state_dict(), strict=False
            )
        if "mixer_state_dict" in state:
            self.mixer.load_state_dict(state["mixer_state_dict"], strict=False)
        if "target_mixer_state_dict" in state:
            self.target_mixer.load_state_dict(
                state["target_mixer_state_dict"], strict=False
            )
        else:
            self.target_mixer.load_state_dict(self.mixer.state_dict(), strict=False)
        if "actor_state_dict" in state:
            self.actor.load_state_dict(state["actor_state_dict"], strict=False)
        if "queue_predictor_state_dict" in state:
            self.queue_predictor.load_state_dict(state["queue_predictor_state_dict"], strict=False)
        if "target_queue_predictor_state_dict" in state:
            self.target_queue_predictor.load_state_dict(
                state["target_queue_predictor_state_dict"], strict=False
            )
        elif "queue_predictor_state_dict" in state:
            self.target_queue_predictor.load_state_dict(
                state["queue_predictor_state_dict"], strict=False
            )
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
        for field_name in ("state_variant", "learner_variant", "recourse_variant"):
            if field_name not in state:
                continue
            stored = str(state[field_name])
            requested = str(getattr(self, field_name, stored))
            if stored != requested:
                raise ValueError(
                    f"checkpoint {field_name} mismatch: "
                    f"requested={requested}, stored={stored}"
                )
        self.joint_training_diagnostics = list(
            state.get("joint_training_diagnostics", self.joint_training_diagnostics)
        )
        self.next_transition_link_misses = int(
            state.get("next_transition_link_misses", self.next_transition_link_misses)
        )
        self.next_transition_link_lookups = int(
            state.get("next_transition_link_lookups", self.next_transition_link_lookups)
        )
        self.joint_training_step = int(
            state.get("joint_training_step", self.joint_training_step)
        )
        self.optimizer_steps_total = int(
            state.get("optimizer_steps_total", self.optimizer_steps_total)
        )
        self.optimizer_steps_joint = int(
            state.get("optimizer_steps_joint", self.optimizer_steps_joint)
        )
        self.optimizer_steps_edge = int(
            state.get("optimizer_steps_edge", self.optimizer_steps_edge)
        )
        self.optimizer_steps_queue = int(
            state.get("optimizer_steps_queue", self.optimizer_steps_queue)
        )
        joint_replay_state = state.get("joint_replay_state_dict")
        if joint_replay_state is not None:
            self.joint_replay_buffer.load_state_dict(joint_replay_state)
            for transition in self.joint_replay_buffer:
                self._validate_response_transition(transition)

    def add_to_logs(self, *args, **kwargs):
        return None

    def remember(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return self.train_step()
