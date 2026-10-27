"""NeurADP external baseline (Shah, Lowalekar & Varakantham, AAAI 2020).

Reference: "Neural Approximate Dynamic Programming for On-Demand Ride-Pooling",
https://arxiv.org/abs/1911.08842, official code
https://github.com/sanketkshah/NeurADP-for-Ride-Pooling.

The algorithm is kept as in the paper; only the inputs needed by this
mixed-fleet electric simulator are added:

* every vehicle has a post-decision value ``V(post_i)`` given by one shared
  network (path LSTM over location embeddings, time embedding, and fleet
  context), extended with post-decision SoC and a vehicle-type indicator;
* the joint action maximizes ``sum_i r(a_i) + gamma * V(post_i(a_i))`` with
  the same exact assignment oracle used by every other method (the paper uses
  an ILP; both return the same optimum);
* HEV offers are scored as if they will be accepted, as in the original model,
  so there is no recourse and no same-epoch repair.  ``response_aware=True``
  gives the "NeurADP + p" variant, which takes the expectation over the
  acceptance probability without any recourse;
* learning is per-agent TD.  The target for ``V(post_i,t)`` is the score of
  vehicle ``i``'s action in the next epoch's joint solution, re-solved with the
  target network, then one supervised MSE epoch (minibatch 32), a soft target
  update (tau = 0.1), prioritized replay, and decaying additive Gaussian
  exploration noise on the assignment scores.

The immediate reward ``r(a)`` is the edge's structured (myopic) score, credited
at the decision.  Continuing (in-progress) actions earn zero, except the
per-epoch waiting cost.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.recourse.types import ActionType
from src.ValueFunction_st_masac_gat import (
    PyTorchChargingValueFunction as _GraphValueFunctionBase,
)


NEURADP_DEFAULTS = {
    "gamma": 0.95,
    "learning_rate": 1e-3,
    "target_tau": 0.1,
    "samples_per_update": 3,
    "fit_batch_size": 32,
    "noise_scale": 1.0,
    "response_aware": False,
    "num_locations": 1024,
}
NEURADP_CONFIG_KEYS = tuple(NEURADP_DEFAULTS)
_SCALAR_DIM = 9


class NeurADPNetwork(nn.Module):
    """Post-decision value network following NeurADP's PathBasedNN."""

    def __init__(
        self,
        num_locations: int,
        scalar_dim: int = _SCALAR_DIM,
        location_dim: int = 100,
        path_dim: int = 200,
        time_dim: int = 100,
        hidden_dim: int = 300,
    ):
        super().__init__()
        self.num_locations = int(num_locations)
        self.location_embedding = nn.Embedding(
            self.num_locations, location_dim, padding_idx=0
        )
        self.path_lstm = nn.LSTM(location_dim + 1, path_dim, batch_first=True)
        self.time_embedding = nn.Sequential(nn.Linear(3, time_dim), nn.ELU())
        self.state_embedding = nn.Sequential(
            nn.Linear(path_dim + time_dim + scalar_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        path_locations: torch.Tensor,
        path_delays: torch.Tensor,
        time_features: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        path = torch.cat(
            [self.location_embedding(path_locations), path_delays.unsqueeze(-1)],
            dim=-1,
        )
        # NeurADP reads the path backwards (Keras go_backwards=True).
        _, (hidden, _) = self.path_lstm(torch.flip(path, dims=(1,)))
        state = torch.cat(
            [hidden[-1], self.time_embedding(time_features), scalars], dim=-1
        )
        return self.output(self.state_embedding(state)).reshape(-1)


class PyTorchChargingValueFunction(_GraphValueFunctionBase):
    learner_variant = "neuradp"
    direct_q = True
    uses_solver_consistent_targets = True
    collects_legacy_experiences = False

    def __init__(self, *args, **kwargs):
        overrides = {
            key: kwargs.pop(f"neuradp_{key}")
            for key in NEURADP_CONFIG_KEYS
            if kwargs.get(f"neuradp_{key}") is not None
        }
        for key in NEURADP_CONFIG_KEYS:
            kwargs.pop(f"neuradp_{key}", None)
        # Constructor knobs that belong to other learners are irrelevant here.
        for key in (
            "neighbour_number", "post_demand_q_weight",
            "post_demand_head_lr_multiplier", "entropy_target_ratio",
            "residual_target_policy", "predictor_variant",
        ):
            kwargs.pop(key, None)
        kwargs["zone_distribution_mode"] = "neuradp"
        super().__init__(*args, **kwargs)
        env = kwargs.get("env")
        config = dict(NEURADP_DEFAULTS)
        for key in NEURADP_CONFIG_KEYS:
            env_value = getattr(env, f"neuradp_{key}", None) if env is not None else None
            if env_value is not None:
                config[key] = env_value
        config.update(overrides)
        self.neuradp_config = {
            "gamma": float(config["gamma"]),
            "learning_rate": float(config["learning_rate"]),
            "target_tau": float(config["target_tau"]),
            "samples_per_update": max(1, int(config["samples_per_update"])),
            "fit_batch_size": max(1, int(config["fit_batch_size"])),
            "noise_scale": max(0.0, float(config["noise_scale"])),
            "response_aware": bool(config["response_aware"]),
            "num_locations": max(2, int(config["num_locations"])),
        }
        if not 0.0 < self.neuradp_config["gamma"] <= 1.0:
            raise ValueError("NeurADP gamma must be in (0, 1]")
        self.gamma = self.neuradp_config["gamma"]
        self.tau = self.neuradp_config["target_tau"]
        self.learning_rate = self.neuradp_config["learning_rate"]
        self.network = NeurADPNetwork(self.neuradp_config["num_locations"]).to(self.device)
        self.target_network = NeurADPNetwork(self.neuradp_config["num_locations"]).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.requires_grad_(False)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        seed = int(getattr(env, "initial_random_seed", 0) or 0) if env is not None else 0
        self._noise_rng = np.random.default_rng(seed + 20_200_507)
        self.neuradp_td_errors: list[float] = []
        print(
            "✓ NeurADP value function initialized: "
            + ", ".join(f"{key}={value}" for key, value in self.neuradp_config.items())
            + f", params={sum(p.numel() for p in self.network.parameters())}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def _location_token(self, location: Any) -> int:
        try:
            value = int(location)
        except (TypeError, ValueError):
            return 0
        limit = self.neuradp_config["num_locations"] - 1
        return value + 1 if 0 <= value < limit else 0

    @staticmethod
    def _is_continuing(edge) -> bool:
        return bool(dict(edge.metadata).get("continuing", False))

    @staticmethod
    def _is_anchor_only(edge) -> bool:
        return bool(dict(edge.metadata).get("matrix_anchor_only", False))

    def _post_rows(self, graph, edges, *, stay: bool = False):
        """Feature arrays of each edge's post-decision state.

        ``stay=True`` describes the state after a rejected HEV offer: the
        vehicle stays at its current location for one epoch.
        """
        state = graph.state
        vehicles = {vehicle.vehicle_id: vehicle for vehicle in state.vehicles}
        online = [vehicle for vehicle in state.vehicles if vehicle.online]
        online_count = max(1, len(online))
        zone_counts = Counter(int(vehicle.location) for vehicle in online)
        fleet_size = max(1.0, float(self.num_vehicles))
        request_norm = float(len(state.requests)) / fleet_size
        online_norm = float(len(online)) / fleet_size
        time_norm = self._time_norm(float(state.current_time))
        angle = time_norm * 2.0 * np.pi
        count = len(edges)
        path_locations = np.zeros((count, 2), dtype=np.int64)
        path_delays = np.zeros((count, 2), dtype=np.float32)
        time_features = np.tile(
            np.asarray([time_norm, np.sin(angle), np.cos(angle)], dtype=np.float32),
            (count, 1),
        )
        scalars = np.zeros((count, _SCALAR_DIM), dtype=np.float32)
        horizon = max(1.0, float(self.episode_length))
        for row, edge in enumerate(edges):
            vehicle = vehicles.get(edge.vehicle_id)
            location = vehicle.location if vehicle is not None else edge.target_location
            battery = float(vehicle.battery) if vehicle is not None else 0.0
            if stay:
                target, post = location, location
                target_distance = post_distance = 0.0
                duration = 1.0
                post_soc = battery
            else:
                target, post = edge.target_location, edge.post_action_location
                target_distance = float(edge.target_distance)
                post_distance = float(edge.post_action_distance)
                duration = float(edge.post_action_duration)
                action_type_id = int(
                    dict(edge.metadata).get("feature_action_type_id", edge.action_type)
                )
                post_soc = self._post_battery(battery, action_type_id, post_distance)
            path_locations[row] = (self._location_token(target), self._location_token(post))
            path_delays[row] = (
                np.clip(target_distance / 30.0, 0.0, 5.0),
                np.clip(duration / horizon, 0.0, 2.0),
            )
            nearby = zone_counts.get(int(post), 0) - int(
                vehicle is not None and vehicle.online and int(location) == int(post)
            )
            scalars[row] = (
                np.clip(post_soc, 0.0, 1.2),
                1.0 if int(edge.vehicle_type) == 1 else 0.0,
                1.0 if int(edge.vehicle_type) == 2 else 0.0,
                np.clip(max(0, nearby) / online_count, 0.0, 1.0),
                np.clip(request_norm, 0.0, 5.0),
                np.clip(online_norm, 0.0, 2.0),
                np.clip(duration / horizon, 0.0, 2.0),
                np.clip(post_distance / 50.0, 0.0, 5.0),
                1.0 if self._is_continuing(edge) else 0.0,
            )
        return path_locations, path_delays, time_features, scalars

    def _values(self, graph, edges, *, target: bool, stay: bool = False, grad: bool = False):
        if not edges:
            return torch.zeros(0, device=self.device)
        arrays = self._post_rows(graph, edges, stay=stay)
        network = self.target_network if target else self.network
        tensors = (
            torch.as_tensor(arrays[0], device=self.device),
            torch.as_tensor(arrays[1], device=self.device),
            torch.as_tensor(arrays[2], device=self.device),
            torch.as_tensor(arrays[3], device=self.device),
        )
        if grad:
            return network(*tensors)
        chunk = max(1, int(getattr(self, "qvalue_inference_batch_size", 8192) or 8192))
        with torch.no_grad():
            return torch.cat([
                network(*(tensor[start:start + chunk] for tensor in tensors))
                for start in range(0, len(edges), chunk)
            ])

    # ------------------------------------------------------------------
    # Scoring (r(a) + gamma * V(post))
    # ------------------------------------------------------------------

    def _immediate_reward(self, edge, *, rejected: bool = False) -> float:
        if self._is_continuing(edge):
            return float(edge.structured_score) if edge.action_type == ActionType.WAIT else 0.0
        if rejected:
            return float(edge.rejection_structured_score)
        if edge.success_structured_score is not None and (
            edge.expected_response_anchor or edge.human_response_mask
        ):
            return float(edge.success_structured_score)
        return float(edge.structured_score)

    def _offer_rejection_probability(self, edge) -> float:
        # Only unanswered human offers carry a response model (q > 0 requires
        # the acceptance feature, e.g. --ev-acceptance-feature predicted).
        if (
            not self.neuradp_config["response_aware"]
            or not edge.human_response_mask
            or int(edge.vehicle_type) != 1
            or edge.action_type != ActionType.SERVICE
            or self._is_continuing(edge)
        ):
            return 0.0
        return float(np.clip(edge.rejection_probability, 0.0, 1.0))

    def _neuradp_scores(self, graph, *, target: bool) -> dict[str, float]:
        scores: dict[str, float] = {}
        learned = [edge for edge in graph.edges if not self._is_anchor_only(edge)]
        for edge in graph.edges:
            if self._is_anchor_only(edge):
                scores[edge.edge_id] = float(edge.structured_score)
        values = self._values(graph, learned, target=target).detach().cpu().numpy()
        gamma = self.neuradp_config["gamma"]
        offers = [index for index, edge in enumerate(learned)
                  if self._offer_rejection_probability(edge) > 0.0]
        stay_values = {}
        if offers:
            stay = self._values(graph, [learned[i] for i in offers], target=target, stay=True)
            stay_values = dict(zip(offers, stay.detach().cpu().numpy().tolist()))
        for index, (edge, value) in enumerate(zip(learned, values)):
            score = self._immediate_reward(edge) + gamma * float(value)
            q = self._offer_rejection_probability(edge)
            if q > 0.0:
                rejected = self._immediate_reward(edge, rejected=True) + gamma * stay_values[index]
                score = (1.0 - q) * score + q * rejected
            scores[edge.edge_id] = float(score)
        return scores

    def _exploration_noise(self, graph) -> dict[str, float]:
        """NeurADP's decaying |N(0, sd)| bonus; zero for in-progress actions."""
        scale = self.neuradp_config["noise_scale"]
        if scale <= 0.0 or self.env is None or bool(getattr(self.env, "evaluatemode", True)):
            return {}
        episode = int(getattr(self.env, "cumulative_episode_index", 0) or 0)
        agents = max(1, int(self.num_vehicles))
        noise = {}
        for edge in graph.edges:
            if self._is_continuing(edge) or self._is_anchor_only(edge):
                continue
            base = 4000.0 if edge.action_type == ActionType.WAIT else 1000.0
            sd = 1.0 + base / ((episode + 1) * agents)
            noise[edge.edge_id] = scale * abs(float(self._noise_rng.normal(0.0, sd)))
        return noise

    def _graph_edge_scores(self, graph, *, target_context: bool):
        """Rollout scores for the exact assignment oracle (plus diagnostics)."""
        scores = self._neuradp_scores(graph, target=bool(target_context))
        corrections = {
            edge.edge_id: scores[edge.edge_id] - float(edge.structured_score)
            for edge in graph.edges
        }
        if not target_context:
            for edge_id, bonus in self._exploration_noise(graph).items():
                scores[edge_id] += bonus
        return scores, corrections

    # ------------------------------------------------------------------
    # Per-agent TD learning
    # ------------------------------------------------------------------

    def _next_epoch_targets(self, next_graph) -> dict[int, float]:
        """Score of each vehicle's action in the target-network re-solve."""
        scores = self._neuradp_scores(next_graph, target=True)
        selection = self.target_builder.project(next_graph, scores)
        selected = set(selection)
        targets: dict[int, float] = {}
        for edge in next_graph.edges:
            if edge.edge_id in selected:
                targets[int(edge.vehicle_id)] = float(scores[edge.edge_id])
        return targets

    def train_step(self, batch_size: int = 64, tau: float | None = None, ifEV: bool = False) -> float:
        del tau
        if ifEV:
            # Integrated control has one system learner; the HEV twin never trains.
            return 0.0
        return self._train_neuradp(batch_size)

    def _train_joint_step(self, batch_size: int, *, ifEV: bool) -> float:
        return self.train_step(batch_size=batch_size, ifEV=ifEV)

    def _train_neuradp(self, batch_size: int) -> float:
        if len(self.joint_replay_buffer) == 0:
            return 0.0
        count = min(int(batch_size), self.neuradp_config["samples_per_update"],
                    len(self.joint_replay_buffer))
        sample = self.joint_replay_buffer.sample_ready(
            max(1, count),
            ready_fleet=False,
            predicate=lambda transition: self._joint_row_ready(transition, ifEV=False),
        )
        if not sample.transitions:
            return 0.0
        graphs, edge_rows, target_rows, weight_rows, owners = [], [], [], [], []
        used_indices = []
        for transition, replay_index, weight in zip(
            sample.transitions, sample.indices, sample.weights
        ):
            if transition.learner_variant != self.learner_variant:
                raise ValueError(
                    "joint replay learner variant mismatch: "
                    f"row={transition.learner_variant}, learner={self.learner_variant}"
                )
            payload = self._joint_stage_payload(transition, ifEV=False)
            if payload is None:
                continue
            graph, action, _reward, _phase = payload
            if graph is None or action is None or not action.selected_edge_ids:
                continue
            chosen = set(action.selected_edge_ids)
            edges = [edge for edge in graph.edges
                     if edge.edge_id in chosen and not self._is_anchor_only(edge)]
            if transition.done:
                targets = {int(edge.vehicle_id): 0.0 for edge in edges}
            else:
                next_graph = self._temporal_successor_graph(transition)
                if next_graph is None:
                    continue
                targets = self._next_epoch_targets(next_graph)
            edges = [edge for edge in edges if int(edge.vehicle_id) in targets]
            if not edges:
                continue
            owner = len(used_indices)
            used_indices.append(int(replay_index))
            graphs.append((graph, edges))
            target_rows.extend(targets[int(edge.vehicle_id)] for edge in edges)
            weight_rows.extend(float(weight) for _ in edges)
            owners.extend(owner for _ in edges)
            edge_rows.extend(range(len(edges)))
        if not graphs:
            return 0.0
        arrays = [self._post_rows(graph, edges) for graph, edges in graphs]
        tensors = [
            torch.as_tensor(np.concatenate([item[index] for item in arrays]), device=self.device)
            for index in range(4)
        ]
        targets = torch.as_tensor(target_rows, dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(weight_rows, dtype=torch.float32, device=self.device)
        rows = targets.numel()
        fit_batch = self.neuradp_config["fit_batch_size"]
        order = torch.as_tensor(self._noise_rng.permutation(rows), device=self.device)
        losses = []
        self.network.train()
        for start in range(0, rows, fit_batch):
            index = order[start:start + fit_batch]
            prediction = self.network(*(tensor.index_select(0, index) for tensor in tensors))
            loss = (weights.index_select(0, index)
                    * F.mse_loss(prediction, targets.index_select(0, index), reduction="none")).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
        self._soft_update(self.network, self.target_network, self.tau)
        with torch.no_grad():
            td = (targets - self.network(*tensors)).abs().cpu().numpy()
        owner_array = np.asarray(owners)
        td_errors = [float(td[owner_array == owner].mean()) for owner in range(len(used_indices))]
        self.joint_replay_buffer.update_priorities(used_indices, td_errors)
        self.joint_replay_buffer.advance_beta()
        mean_loss = float(np.mean(losses))
        self.training_losses.append(mean_loss)
        self.neuradp_td_errors.append(float(np.mean(td_errors)))
        if len(self.neuradp_td_errors) > 10_000:
            del self.neuradp_td_errors[:-10_000]
        steps = len(losses)
        self.training_step += 1
        self.joint_training_step += 1
        self.optimizer_steps_joint += steps
        self.optimizer_steps_total += steps
        return mean_loss

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def inference_checkpoint_state(self) -> dict[str, Any]:
        state = super().inference_checkpoint_state()
        state["neuradp_config"] = dict(self.neuradp_config)
        return state

    def extra_checkpoint_state(self) -> dict[str, Any]:
        state = super().extra_checkpoint_state()
        state["neuradp_config"] = dict(self.neuradp_config)
        state["neuradp_td_errors"] = list(self.neuradp_td_errors)
        state["neuradp_target_network_state_dict"] = self.target_network.state_dict()
        return state

    def load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        stored = dict(state.get("neuradp_config", {}) or {})
        if stored and int(stored.get("num_locations", -1)) != self.neuradp_config["num_locations"]:
            raise ValueError("NeurADP checkpoint location-embedding size mismatch")
        super().load_extra_checkpoint_state(state)
        self.neuradp_td_errors = list(state.get("neuradp_td_errors", self.neuradp_td_errors))
        target_state = state.get("neuradp_target_network_state_dict")
        # Inference checkpoints store only the online network.
        self.target_network.load_state_dict(
            target_state if target_state is not None else self.network.state_dict()
        )
