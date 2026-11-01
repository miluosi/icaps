"""NeurADP external baseline (Shah, Lowalekar & Varakantham, AAAI 2020).

Reference: "Neural Approximate Dynamic Programming for On-Demand Ride-Pooling",
https://arxiv.org/abs/1911.08842, official code
https://github.com/sanketkshah/NeurADP-for-Ride-Pooling.

This is a mixed-fleet electric adaptation of the paper's per-agent value
learning and joint assignment, using compact route features:

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

Non-wait immediate rewards use the structured action score, credited at the
decision. Waiting uses the simulator's per-epoch idle cost and the project's
learning-only low-SoC penalty. Neither changes the reported system reward.
Continuing actions earn zero, except the per-epoch waiting cost.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.recourse.types import ActionType
from src.qvalue_inference import bounded_edge_inference
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
NEURADP_IMPLEMENTATION_VERSION = 2
_WAIT_CONFIG_KEYS = (
    "learning_soc_wait_threshold", "learning_soc_wait_reference",
    "learning_soc_wait_penalty_per_hour", "learning_soc_wait_exponent",
)


def _validated_config(config):
    result = {key: config[key] for key in NEURADP_DEFAULTS}
    for key in ("gamma", "learning_rate", "target_tau", "noise_scale"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]):
            raise ValueError(f"NeurADP {key} must be finite")
    if not 0 < result["gamma"] <= 1:
        raise ValueError("NeurADP gamma must be in (0, 1]")
    if result["learning_rate"] <= 0 or not 0 <= result["target_tau"] <= 1 or result["noise_scale"] < 0:
        raise ValueError("Invalid NeurADP learning rate, target tau or noise scale")
    for key in ("samples_per_update", "fit_batch_size", "num_locations"):
        value = float(result[key])
        minimum = 2 if key == "num_locations" else 1
        if not math.isfinite(value) or not value.is_integer() or value < minimum:
            raise ValueError(f"NeurADP {key} must be an integer >= {minimum}")
        result[key] = int(value)
    result["response_aware"] = bool(result["response_aware"])
    return result


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
    neuradp_implementation_version = NEURADP_IMPLEMENTATION_VERSION
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
        env = self.env
        config = dict(NEURADP_DEFAULTS)
        for key in NEURADP_CONFIG_KEYS:
            env_value = getattr(env, f"neuradp_{key}", None) if env is not None else None
            if env_value is not None:
                config[key] = env_value
        config.update(overrides)
        self.neuradp_config = _validated_config(config)
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

    def _post_battery(self, battery, action_type_id, distance):
        """Use NYC's target and timeout instead of adding a fixed full session."""
        if int(action_type_id) != ActionType.CHARGE or not callable(
            getattr(self.env, "_charge_duration_for_battery", None)
        ):
            return super()._post_battery(battery, action_type_id, distance)
        arrival = float(battery) - float(distance) * float(self.env.battery_consum)
        duration = self.env._charge_duration_for_battery(arrival)
        target = self.env._charge_target_for_battery(arrival)
        return float(np.clip(min(target, arrival + duration*self.env.chargeincrease_per_epoch), 0, 1))

    def snapshot_action_metadata(self, *, vehicle, action_type, station_id,
                                 post_action_distance, post_action_duration):
        """Freeze charge outcomes at collection, never consult live cars in TD."""
        if action_type != ActionType.CHARGE or not callable(
            getattr(self.env, "_charge_duration_for_battery", None)
        ):
            return ()
        if vehicle.charging_station is not None:
            arrival = float(vehicle.battery)
            duration = float(vehicle.charging_time_left)
            target = float(self.env.vehicles[vehicle.vehicle_id].get(
                "charge_target_soc", self.env.charge_target_soc))
            travel = 0.
        else:
            expansion = getattr(self.env, "_last_expected_charge_expansion", None) or {}
            if float(expansion.get("current_time", -1)) != float(self.env.current_time):
                expansion = {}
            window = expansion.get("candidate_windows", {}).get((vehicle.vehicle_id, station_id))
            if window is None:
                window = self.env.calculate_expected_entercharge_station_time_battery(
                    vehicle.vehicle_id, station_id)
            if window is None:
                raise ValueError("NeurADP charge edge has no reachable charging outcome")
            arrival = float(window["expected_battery_after_travel"])
            duration = float(window["charging_duration"])
            travel = float(window["travel_epochs"])
            target = float(self.env._charge_target_for_battery(arrival))
        post_soc = min(1., target, arrival + duration*self.env.chargeincrease_per_epoch)
        return (("neuradp_post_charge_soc", max(0., float(post_soc))),
                ("neuradp_charge_duration", travel + duration))

    def _post_context(self, graph):
        vehicles = {vehicle.vehicle_id: vehicle for vehicle in graph.state.vehicles}
        online = [vehicle for vehicle in graph.state.vehicles if vehicle.online]
        return vehicles, max(1, len(online)), Counter(int(v.location) for v in online)

    def _post_rows(self, graph, edges, *, stay: bool = False, context=None):
        """Feature arrays of each edge's post-decision state.

        ``stay=True`` describes the state after a rejected HEV offer: the
        vehicle stays at its current location for one epoch.
        """
        state = graph.state
        vehicles, online_count, zone_counts = context or self._post_context(graph)
        fleet_size = max(1.0, float(self.num_vehicles))
        request_norm = float(len(state.requests)) / fleet_size
        online_norm = float(sum(zone_counts.values())) / fleet_size
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
        horizon = max(1., float(getattr(self, "aligned_inference_episode_length",
                                       getattr(self.env, "model_episode_length", self.episode_length))))
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
                metadata = dict(edge.metadata)
                if "neuradp_post_charge_soc" in metadata:
                    post_soc = float(metadata["neuradp_post_charge_soc"])
                    duration = float(metadata["neuradp_charge_duration"])
                else:
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
            return torch.zeros(0, device=self.device if grad else "cpu")
        network = self.target_network if target else self.network
        if grad:
            arrays = self._post_rows(graph, edges, stay=stay)
            return network(*(torch.as_tensor(array, device=self.device) for array in arrays))
        context = self._post_context(graph)
        def evaluate(part):
            arrays = self._post_rows(graph, edges[part], stay=stay, context=context)
            return network(*(torch.as_tensor(array, device=self.device) for array in arrays)).cpu().numpy()
        with torch.no_grad():
            values, self._last_qvalue_inference_stats = bounded_edge_inference(
                len(edges), min(self.qvalue_inference_batch_size,
                                getattr(self, '_neuradp_safe_batch_size', self.qvalue_inference_batch_size)),
                evaluate, device=self.device)
        if self._last_qvalue_inference_stats['oom_retries']:
            self._neuradp_safe_batch_size = self._last_qvalue_inference_stats['effective_batch_size']
        return torch.from_numpy(values)

    def _predict_arrays(self, arrays):
        def evaluate(part):
            return self.network(*(torch.as_tensor(array[part], device=self.device)
                                  for array in arrays)).cpu().numpy()
        with torch.no_grad():
            values, self._last_neuradp_td_inference_stats = bounded_edge_inference(
                len(arrays[0]), min(self.qvalue_inference_batch_size,
                                   getattr(self, '_neuradp_safe_batch_size', self.qvalue_inference_batch_size)),
                evaluate, device=self.device)
        if self._last_neuradp_td_inference_stats['oom_retries']:
            self._neuradp_safe_batch_size = self._last_neuradp_td_inference_stats['effective_batch_size']
        return values

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
        vehicles = {v.vehicle_id: v for v in graph.state.vehicles}
        offers = [index for index, edge in enumerate(learned)
                  if self._offer_rejection_probability(edge) > 0.0]
        stay_values = {}
        if offers:
            stay = self._values(graph, [learned[i] for i in offers], target=target, stay=True)
            stay_values = dict(zip(offers, stay.detach().cpu().numpy().tolist()))
        for index, (edge, value) in enumerate(zip(learned, values)):
            immediate = self._immediate_reward(edge)
            if edge.action_type == ActionType.WAIT and self.env is not None:
                immediate = -float(getattr(self.env, "idle_penalty", -immediate))
                vehicle = vehicles.get(edge.vehicle_id)
                penalty = getattr(self.env, "soc_wait_learning_penalty", None)
                if vehicle is not None and vehicle.online and vehicle.vehicle_type == 2 and callable(penalty):
                    immediate -= float(penalty(vehicle.battery))
            score = immediate + gamma * float(value)
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
        if self.env is not None and bool(getattr(self.env, "evaluatemode", False)):
            return 0.0
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
        graphs, target_rows, weight_rows, owners = [], [], [], []
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
        if not graphs:
            return 0.0
        batches = [self._post_rows(graph, edges) for graph, edges in graphs]
        arrays = [np.concatenate([item[index] for item in batches]) for index in range(4)]
        targets = np.asarray(target_rows, dtype=np.float32)
        weights = np.asarray(weight_rows, dtype=np.float32)
        rows = len(targets)
        fit_batch = self.neuradp_config["fit_batch_size"]
        order = self._noise_rng.permutation(rows)
        losses = []
        self.network.train()
        for start in range(0, rows, fit_batch):
            index = order[start:start + fit_batch]
            prediction = self.network(*(torch.as_tensor(array[index], device=self.device) for array in arrays))
            target_batch = torch.as_tensor(targets[index], device=self.device)
            weight_batch = torch.as_tensor(weights[index], device=self.device)
            loss = (weight_batch * F.mse_loss(prediction, target_batch, reduction="none")).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
        self._soft_update(self.network, self.target_network, self.tau)
        td = np.abs(targets - self._predict_arrays(arrays))
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
        state["neuradp_implementation_version"] = NEURADP_IMPLEMENTATION_VERSION
        state["neuradp_wait_shaping"] = {key: getattr(self.env, key) for key in _WAIT_CONFIG_KEYS
                                         if hasattr(self.env, key)}
        return state

    def extra_checkpoint_state(self) -> dict[str, Any]:
        state = super().extra_checkpoint_state()
        state["neuradp_config"] = dict(self.neuradp_config)
        state["neuradp_implementation_version"] = NEURADP_IMPLEMENTATION_VERSION
        state["neuradp_wait_shaping"] = {key: getattr(self.env, key) for key in _WAIT_CONFIG_KEYS
                                         if hasattr(self.env, key)}
        state["neuradp_td_errors"] = list(self.neuradp_td_errors)
        state["neuradp_target_network_state_dict"] = self.target_network.state_dict()
        return state

    def load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        version = int(state.get("neuradp_implementation_version", 1))
        if not 1 <= version <= NEURADP_IMPLEMENTATION_VERSION:
            raise ValueError(f"Unsupported NeurADP checkpoint implementation version: {version}")
        stored = dict(state.get("neuradp_config", {}) or {})
        restored = _validated_config({**self.neuradp_config, **stored})
        if restored["num_locations"] != self.neuradp_config["num_locations"]:
            raise ValueError("NeurADP checkpoint location-embedding size mismatch")
        super().load_extra_checkpoint_state(state)
        self.neuradp_config = restored
        self.gamma, self.tau = restored["gamma"], restored["target_tau"]
        self.learning_rate = restored["learning_rate"]
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        for key, value in dict(state.get("neuradp_wait_shaping", {}) or {}).items():
            if key in _WAIT_CONFIG_KEYS and self.env is not None:
                setattr(self.env, key, float(value))
        if version < NEURADP_IMPLEMENTATION_VERSION:
            print("NeurADP legacy weights loaded; corrected charging features and wait rewards require retraining for a corrected baseline.", flush=True)
        self.neuradp_td_errors = list(state.get("neuradp_td_errors", self.neuradp_td_errors))
        target_state = state.get("neuradp_target_network_state_dict")
        # Inference checkpoints store only the online network.
        self.target_network.load_state_dict(
            target_state if target_state is not None else self.network.state_dict()
        )
