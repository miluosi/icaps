"""MASAC-GAT value function with predicted post-action zone demand.

This is an isolated extension of ``ValueFunction_st_masac_gat``.  The
predictor estimates the number of active requests in the post-action zone at
the action completion time.  Its normalized prediction is appended to every
request, relocation, and charging edge.  No demand bonus is added to either
the environment reward or the final edge score.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from src.ValueFunction_st_masac_gat import (
    PyTorchChargingValueFunction as _BaseMASAC,
    _MLP,
)


def _expand_mlp_input(
    source: _MLP,
    input_dim: int,
    inserted_index: int,
    device: torch.device,
) -> _MLP:
    """Insert a zero-weight input while preserving every existing output."""
    expanded = _MLP(input_dim, hidden_dim=128).to(device)
    with torch.no_grad():
        source_first = source.net[0]
        expanded_first = expanded.net[0]
        expanded_first.weight[:, :inserted_index].copy_(
            source_first.weight[:, :inserted_index]
        )
        expanded_first.weight[:, inserted_index].zero_()
        expanded_first.weight[:, inserted_index + 1 :].copy_(
            source_first.weight[:, inserted_index:]
        )
        expanded_first.bias.copy_(source_first.bias)
        for layer_index in (2, 4):
            expanded.net[layer_index].weight.copy_(source.net[layer_index].weight)
            expanded.net[layer_index].bias.copy_(source.net[layer_index].bias)
    return expanded


class PyTorchChargingValueFunction(_BaseMASAC):
    uses_post_demand_feature = True
    post_demand_feature_version = 1

    def __init__(self, *args, post_demand_scale: float = 100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.zone_distribution_mode = kwargs.get(
            "zone_distribution_mode",
            "st_masac_gat_post_demand",
        ) or "st_masac_gat_post_demand"
        self.post_demand_scale = max(1.0, float(post_demand_scale))
        self.post_demand_feature_dim = 8
        self.post_demand_input_index = int(self.edge_local_dim)

        # Add exactly one critic input and initially reproduce the base model.
        base_network = self.network
        base_critic2 = self.critic2
        base_target_network = self.target_network
        base_target_critic2 = self.target_critic2
        base_actor = self.actor
        self.edge_local_dim += 1
        self.edge_dim += 1
        self.network = _expand_mlp_input(
            base_network, self.edge_dim, self.post_demand_input_index, self.device
        )
        self.critic2 = _expand_mlp_input(
            base_critic2, self.edge_dim, self.post_demand_input_index, self.device
        )
        self.target_network = _expand_mlp_input(
            base_target_network, self.edge_dim, self.post_demand_input_index, self.device
        )
        self.target_critic2 = _expand_mlp_input(
            base_target_critic2, self.edge_dim, self.post_demand_input_index, self.device
        )
        self.actor = _expand_mlp_input(
            base_actor, self.edge_dim, self.post_demand_input_index, self.device
        )

        self.post_demand_predictor = _MLP(
            self.post_demand_feature_dim,
            hidden_dim=64,
        ).to(self.device)
        initial_normalized_count = 1.0 / self.post_demand_scale
        self.post_demand_output_bias = math.log(
            math.expm1(initial_normalized_count)
        )
        self.post_demand_optimizer = optim.Adam(
            self.post_demand_predictor.parameters(),
            lr=self.learning_rate,
        )
        self.post_demand_loss_fn = nn.MSELoss()
        self.post_demand_experience_buffer: deque = deque(
            maxlen=int(kwargs.get("replay_buffer_size", 500_000))
        )
        self.post_demand_training_losses: list[float] = []
        self.post_demand_training_mse_losses: list[float] = []
        self.post_demand_predictor_trained = False
        self._last_post_demand_features = np.zeros(0, dtype=np.float32)
        self.reloc_request_score_margin = 1e-3
        self._last_reloc_request_cap_count = 0
        self.reloc_request_cap_total = 0

        params = (
            [parameter for parameter in self.graph_encoder.parameters() if parameter.requires_grad]
            + list(self.mixer.parameters())
            + list(self.network.parameters())
            + list(self.critic2.parameters())
            + list(self.actor.parameters())
            + [self.log_alpha]
        )
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        print(
            "✓ ST-ADP-MASAC-GAT post-demand initialized: "
            f"edge_dim={self.edge_dim}, predictor_dim={self.post_demand_feature_dim}, "
            f"demand_scale={self.post_demand_scale:g}",
            flush=True,
        )

    def _positive_post_demand(self, raw_prediction: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw_prediction + self.post_demand_output_bias)

    def _current_zone_demand(self, location: int) -> float:
        demand = 0.0
        requests = getattr(self.env, "active_requests", {}) if self.env is not None else {}
        for request in requests.values():
            pickup = int(getattr(request, "pickup", getattr(request, "source", -1)))
            if pickup == int(location):
                demand += 1.0
        return demand

    def _post_demand_predictor_rows(
        self,
        *,
        current_times,
        post_action_durations,
        post_action_locations,
        num_requests,
        current_zone_demands=None,
        snapshot_available=None,
    ) -> np.ndarray:
        current_times = np.asarray(current_times, dtype=np.float32)
        durations = np.asarray(post_action_durations, dtype=np.float32)
        locations = np.asarray(post_action_locations, dtype=np.int64)
        global_demand = np.asarray(num_requests, dtype=np.float32)
        size = locations.size
        if current_zone_demands is None:
            current_zone_demands = np.zeros(size, dtype=np.float32)
            snapshot_available = np.zeros(size, dtype=np.float32)
        else:
            current_zone_demands = np.asarray(current_zone_demands, dtype=np.float32)
            if snapshot_available is None:
                snapshot_available = np.ones(size, dtype=np.float32)
        snapshot_available = np.asarray(snapshot_available, dtype=np.float32)

        rows = []
        for index in range(size):
            completion_time = float(current_times[index] + durations[index])
            time_norm = self._time_norm(completion_time)
            angle = 2.0 * math.pi * time_norm
            rows.append([
                self._zone_norm(int(locations[index])),
                float(time_norm),
                math.sin(angle),
                math.cos(angle),
                float(np.clip(durations[index] / max(self.episode_length, 1.0), 0.0, 2.0)),
                float(np.clip(current_zone_demands[index] / self.post_demand_scale, 0.0, 5.0)),
                float(np.clip(global_demand[index] / self.post_demand_scale, 0.0, 10.0)),
                float(np.clip(snapshot_available[index], 0.0, 1.0)),
            ])
        return np.asarray(rows, dtype=np.float32)

    def predict_post_action_demand(
        self,
        *,
        current_times,
        post_action_durations,
        post_action_locations,
        num_requests,
        current_zone_demands=None,
        snapshot_available=None,
    ) -> np.ndarray:
        size = len(post_action_locations)
        if not self.post_demand_predictor_trained:
            return np.zeros(size, dtype=np.float32)
        rows = self._post_demand_predictor_rows(
            current_times=current_times,
            post_action_durations=post_action_durations,
            post_action_locations=post_action_locations,
            num_requests=num_requests,
            current_zone_demands=current_zone_demands,
            snapshot_available=snapshot_available,
        )
        with torch.no_grad():
            features = torch.tensor(rows, dtype=torch.float32, device=self.device)
            prediction = self._positive_post_demand(
                self.post_demand_predictor(features)
            ).squeeze(1)
        return np.clip(prediction.cpu().numpy(), 0.0, 5.0).astype(np.float32)

    def store_post_demand_experience(
        self,
        *,
        current_time: float,
        post_action_duration: float,
        post_action_location: int,
        num_requests: float,
        current_zone_demand: float,
        observed_post_demand: float,
        snapshot_available: float = 1.0,
    ) -> None:
        row = self._post_demand_predictor_rows(
            current_times=[current_time],
            post_action_durations=[post_action_duration],
            post_action_locations=[post_action_location],
            num_requests=[num_requests],
            current_zone_demands=[current_zone_demand],
            snapshot_available=[snapshot_available],
        )[0]
        target = float(np.clip(observed_post_demand / self.post_demand_scale, 0.0, 5.0))
        self.post_demand_experience_buffer.append((row.tolist(), target))

    def train_post_demand_predictor(self, batch_size: int = 64) -> float:
        if len(self.post_demand_experience_buffer) < max(8, batch_size // 2):
            return 0.0
        batch = random.sample(
            list(self.post_demand_experience_buffer),
            min(batch_size, len(self.post_demand_experience_buffer)),
        )
        features = torch.tensor(
            [item[0] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        targets = torch.tensor(
            [item[1] for item in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        prediction = self._positive_post_demand(self.post_demand_predictor(features))
        loss = self.post_demand_loss_fn(prediction, targets)
        self.post_demand_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.post_demand_predictor.parameters(), 5.0)
        self.post_demand_optimizer.step()
        value = float(loss.item())
        self.post_demand_training_losses.append(value)
        self.post_demand_training_mse_losses.append(
            value * self.post_demand_scale * self.post_demand_scale
        )
        self.post_demand_predictor_trained = True
        return value

    def store_experience(self, **kwargs):
        super().store_experience(**kwargs)
        observed_post_demand = kwargs.get("observed_post_demand")
        if observed_post_demand is None:
            return
        post_location = kwargs.get(
            "post_action_location",
            kwargs.get("next_vehicle_location", kwargs.get("target_location", 0)),
        )
        if post_location is None:
            return
        duration = float(
            kwargs.get("post_action_duration", kwargs.get("dur_time", 1.0)) or 0.0
        )
        self.store_post_demand_experience(
            current_time=float(
                kwargs.get(
                    "post_demand_current_time",
                    kwargs.get("current_time", 0.0),
                )
            ),
            post_action_duration=duration,
            post_action_location=int(post_location),
            num_requests=float(
                kwargs.get(
                    "post_demand_num_requests_at_start",
                    kwargs.get("num_requests", 0.0),
                ) or 0.0
            ),
            current_zone_demand=float(
                kwargs.get("post_demand_current_zone_count", 0.0) or 0.0
            ),
            observed_post_demand=float(observed_post_demand),
            snapshot_available=float(
                kwargs.get("post_demand_snapshot_available", 0.0) or 0.0
            ),
        )

    def post_demand_mse(self, rows: list[tuple[list[float], float]]) -> float:
        if not rows or not self.post_demand_predictor_trained:
            return float("nan")
        features = torch.tensor(
            [row for row, _ in rows],
            dtype=torch.float32,
            device=self.device,
        )
        labels = np.asarray([label for _, label in rows], dtype=np.float32)
        with torch.no_grad():
            prediction = self._positive_post_demand(
                self.post_demand_predictor(features)
            ).squeeze(1)
        raw_prediction = prediction.cpu().numpy() * self.post_demand_scale
        return float(np.mean((raw_prediction - labels) ** 2))

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
        post_demand_features=None,
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
        if post_demand_features is None:
            post_demand_features = self.predict_post_action_demand(
                current_times=current_times,
                post_action_durations=post_action_durations,
                post_action_locations=post_action_locations,
                num_requests=num_requests,
            )
        else:
            post_demand_features = np.asarray(post_demand_features, dtype=np.float32)
        self._last_post_demand_features = np.asarray(post_demand_features, dtype=np.float32)

        rows = []
        type_weights = []
        for index in range(len(vehicle_ids)):
            vehicle_type = self._vehicle_type(int(vehicle_ids[index]))
            state = self._state_features(
                location=int(vehicle_locations[index]),
                current_time=float(current_times[index]),
                battery=float(battery_levels[index]),
                idle_time=float(vehicle_idle_times[index]),
                other_vehicles=float(other_vehicles[index]),
                num_requests=float(num_requests[index]),
                vehicle_type=vehicle_type,
            )
            local = self._local_edge_features(
                state,
                action_type_id=int(action_type_ids[index]),
                target_distance=float(target_distances[index]),
                post_action_duration=float(post_action_durations[index]),
                post_action_distance=float(post_action_distances[index]),
                battery_level=float(battery_levels[index]),
                vehicle_type=vehicle_type,
                queue_wait_feature=float(queue_wait_features[index]),
            )
            local.append(float(post_demand_features[index]))
            source_h = source_embeddings[int(vehicle_ids[index])]
            target_h = self._graph_embedding_for_location(
                graph,
                int(post_action_locations[index]),
            )
            local_t = torch.tensor(local, dtype=torch.float32, device=self.device)
            rows.append(torch.cat([local_t, source_h, target_h], dim=0))
            type_weights.append(
                graph["w_ev"] if vehicle_type == 1 else graph["w_aev"]
            )
        return torch.stack(rows), torch.stack(type_weights).unsqueeze(1), graph["baseline"]

    def _cap_relocation_scores_below_requests(
        self,
        values: np.ndarray,
        vehicle_ids: np.ndarray,
        action_type_ids: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).copy()
        cap_count = 0
        for vehicle_id in np.unique(vehicle_ids):
            vehicle_mask = vehicle_ids == vehicle_id
            request_idx = np.flatnonzero(vehicle_mask & (action_type_ids == 2))
            reloc_idx = np.flatnonzero(vehicle_mask & (action_type_ids == 1))
            if request_idx.size == 0 or reloc_idx.size == 0:
                continue
            request_ceiling = float(np.max(values[request_idx])) - self.reloc_request_score_margin
            over = values[reloc_idx] > request_ceiling
            if np.any(over):
                values[reloc_idx[over]] = request_ceiling
                cap_count += int(np.sum(over))
        self._last_reloc_request_cap_count = cap_count
        self.reloc_request_cap_total += cap_count
        return values

    def batch_get_mixed_q_values(self, **kwargs):
        values = np.asarray(super().batch_get_mixed_q_values(**kwargs), dtype=np.float32)
        vehicle_ids = np.asarray(kwargs["vehicle_ids"], dtype=np.int64)
        action_type_ids = np.asarray(kwargs["action_type_ids"], dtype=np.int64)
        values = self._cap_relocation_scores_below_requests(
            values,
            vehicle_ids,
            action_type_ids,
        )
        return values.astype(np.float32).tolist()

    def train_step(self, batch_size: int = 64, tau: float | None = None, ifEV: bool = False) -> float:
        loss = super().train_step(batch_size=batch_size, tau=tau, ifEV=ifEV)
        self.train_post_demand_predictor(batch_size=batch_size)
        return loss

    def extra_checkpoint_state(self) -> dict[str, Any]:
        state = super().extra_checkpoint_state()
        state.update({
            "post_demand_predictor_state_dict": self.post_demand_predictor.state_dict(),
            "post_demand_optimizer_state_dict": self.post_demand_optimizer.state_dict(),
            "post_demand_predictor_trained": self.post_demand_predictor_trained,
            "post_demand_training_losses": list(self.post_demand_training_losses),
            "post_demand_training_mse_losses": list(self.post_demand_training_mse_losses),
            "post_demand_scale": self.post_demand_scale,
            "post_demand_output_bias": self.post_demand_output_bias,
            "post_demand_feature_version": self.post_demand_feature_version,
            "reloc_request_score_margin": self.reloc_request_score_margin,
            "reloc_request_cap_total": self.reloc_request_cap_total,
        })
        return state

    def load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super().load_extra_checkpoint_state(state)
        predictor_state = state.get("post_demand_predictor_state_dict")
        if predictor_state:
            self.post_demand_predictor.load_state_dict(predictor_state, strict=False)
        optimizer_state = state.get("post_demand_optimizer_state_dict")
        if optimizer_state:
            try:
                self.post_demand_optimizer.load_state_dict(optimizer_state)
            except (ValueError, RuntimeError):
                pass
        self.post_demand_predictor_trained = bool(
            state.get("post_demand_predictor_trained", False)
        )
        self.post_demand_training_losses = list(
            state.get("post_demand_training_losses", [])
        )
        self.post_demand_training_mse_losses = list(
            state.get("post_demand_training_mse_losses", [])
        )
        self.post_demand_output_bias = float(
            state.get("post_demand_output_bias", self.post_demand_output_bias)
        )
        self.reloc_request_score_margin = float(
            state.get("reloc_request_score_margin", self.reloc_request_score_margin)
        )
        self.reloc_request_cap_total = int(
            state.get("reloc_request_cap_total", self.reloc_request_cap_total)
        )
