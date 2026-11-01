"""NeurADP's author-code fitted state-value iteration, adapted to NYC.

Author: https://github.com/sanketkshah/NeurADP-for-Ride-Pooling
For each saved current graph, recompute r_NYC + gamma*U_target(next epoch),
solve its exact joint assignment, and regress U_online(current vehicle state)
on the selected per-vehicle scores. Candidate outcomes use one physical NYC
30s epoch, not whole-action completion. A shared path LSTM includes current
location, remaining route, actual SoC, time and fleet context. NYC's actual
costs/fare timing and learning-only SoC wait shaping enter the model reward.
Reward/response randomness is integrated without consuming simulator RNG.
Current and real_conservative admission/execution and OR-Tools+SSG are kept.
Gamma defaults to this project's .95 (not the author's epoch-dependent default).
"""

from __future__ import annotations

from collections import Counter
import json
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.neuradp_nyc_transition import freeze_epoch_outcomes, vehicle_record
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
    "response_aware": True,
    "num_locations": 1024,
}
NEURADP_CONFIG_KEYS = tuple(NEURADP_DEFAULTS)
_SCALAR_DIM = 10
NEURADP_IMPLEMENTATION_VERSION = 3
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
    """Current vehicle-state value network following NeurADP PathBasedNN."""

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

    def _time_norm(self, current_time):
        # One physical epoch means +30s in both independent-day training and
        # continuous testing. Demand's legacy t-1 indexing is not state time.
        if self.env is not None and hasattr(self.env, 'START_EPOCH'):
            return ((self.env.START_EPOCH+float(current_time)*self.env.EPOCH_LENGTH) % 86400)/86400
        return super()._time_norm(current_time)

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

    def snapshot_action_metadata(self, *, vehicle, action_type, station_id,
                                 target_location, request_id, continuing,
                                 post_action_distance, post_action_duration):
        del post_action_distance, post_action_duration
        if self.env is None or not hasattr(self.env, '_advance_charging_progress'):
            raise ValueError('NeurADP v3 requires the NYC one-epoch transition model')
        key = (self.env.current_time, id(getattr(self.env, '_active_stage_state_snapshot', None)))
        if key != getattr(self, '_preview_cache_key', None):
            self._preview_cache_key = key
            self._preview_relocation_cache = {}
        return freeze_epoch_outcomes(self.env, vehicle=vehicle, action_type=action_type,
            target_location=target_location, request_id=request_id,
            station_id=station_id, continuing=continuing,
            relocation_cache=self._preview_relocation_cache)

    def _post_context(self, graph):
        vehicles = {vehicle.vehicle_id: vehicle for vehicle in graph.state.vehicles}
        online = [vehicle for vehicle in graph.state.vehicles if vehicle.online]
        return vehicles, max(1, len(online)), Counter(int(v.location) for v in online)

    @staticmethod
    def _outcomes(edge):
        encoded = dict(edge.metadata).get('neuradp_epoch_outcomes')
        if encoded is None:
            raise ValueError('NeurADP replay lacks v3 NYC one-epoch outcomes; recollect replay')
        return json.loads(encoded)

    def _record_rows(self, graph, vehicles_and_records, *, next_epoch, context=None):
        vehicles, online_count, zone_counts = context or self._post_context(graph)
        fleet_size = max(1., float(self.num_vehicles))
        request_norm = len(graph.state.requests)/fleet_size
        time_norm = self._time_norm(float(graph.state.current_time)+int(next_epoch))
        angle = time_norm*2*np.pi
        count = len(vehicles_and_records)
        paths = np.zeros((count, 3), dtype=np.int64)
        delays = np.zeros((count, 3), dtype=np.float32)
        times = np.tile(np.asarray([time_norm, np.sin(angle), np.cos(angle)], dtype=np.float32), (count, 1))
        scalars = np.zeros((count, _SCALAR_DIM), dtype=np.float32)
        horizon = max(1., float(getattr(self, 'aligned_inference_episode_length',
            getattr(self.env, 'model_episode_length', self.episode_length))))
        station_locations = {s.station_id: s.location for s in graph.state.stations}
        for row, (vid, record) in enumerate(vehicles_and_records):
            (loc, soc, pickup, dropoff, pickup_time, trip_time, charge_left,
             reloc, reloc_time, idle_left, online, station, charge_target,
             target, service_distance, deadline, request_value) = record
            own = vehicles[vid]
            if pickup is not None:
                route = [loc, pickup, dropoff]
                duration = pickup_time+trip_time
            elif dropoff is not None:
                route = [loc, dropoff, None]
                duration = trip_time
            elif station is not None or charge_target is not None:
                sid = station if station is not None else charge_target
                route = [loc, station_locations.get(sid, target), None]
                duration = charge_left
            else:
                route = [loc, reloc if reloc is not None else target, None]
                duration = max(reloc_time, idle_left)
            paths[row] = [0 if x is None else self._location_token(x) for x in route]
            delays[row] = [1., np.clip(deadline/horizon, 0, 2), np.clip(duration/horizon, 0, 2)]
            nearby = max(0, zone_counts.get(int(loc), 0)-int(own.online and own.location == loc))
            scalars[row] = (np.clip(soc, 0, 1.2), own.vehicle_type == 1,
                own.vehicle_type == 2, nearby/online_count, np.clip(request_norm, 0, 5),
                online_count/fleet_size, np.clip(duration/horizon, 0, 2),
                np.clip(service_distance/50, 0, 5),
                (-1. if not online else float(dropoff is not None or station is not None or charge_target is not None)),
                np.clip(request_value/100., 0, 10))
        return paths, delays, times, scalars

    def _current_rows(self, graph, vehicle_ids):
        vehicles = {v.vehicle_id: v for v in graph.state.vehicles}
        return self._record_rows(graph, [(vid, vehicle_record(vehicles[vid])) for vid in vehicle_ids], next_epoch=False)

    def _post_rows(self, graph, edges, *, stay=False, context=None):
        rows = []
        for edge in edges:
            _, outcomes = self._outcomes(edge)
            outcome = next((x for x in outcomes if bool(x[4]) == bool(stay)), outcomes[0])
            rows.append((edge.vehicle_id, outcome[1]))
        return self._record_rows(graph, rows, next_epoch=True, context=context)

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
    # Scoring (expected NYC epoch reward + gamma * U(next epoch))
    # ------------------------------------------------------------------

    def _immediate_reward(self, edge, *, rejected=False):
        _, outcomes = self._outcomes(edge)
        eligible = [x for x in outcomes if bool(x[4]) == bool(rejected)]
        mass = sum(x[0] for x in eligible)
        return sum(x[0]*(x[2]-x[3]) for x in eligible)/mass if mass else 0.

    def _neuradp_scores(self, graph, *, target):
        # Both rollout and replay re-solve use r_NYC + gamma*U(next state).
        learned = [e for e in graph.edges if not self._is_anchor_only(e)]
        scores = {e.edge_id: float(e.structured_score) for e in graph.edges if self._is_anchor_only(e)}
        context = self._post_context(graph)
        def evaluate(part):
            edges = learned[part]
            rows, owners, probabilities, rewards, terminals = [], [], [], [], []
            for owner, edge in enumerate(edges):
                terminal, outcomes = self._outcomes(edge)
                if not self.neuradp_config['response_aware']:
                    outcomes = [x for x in outcomes if not x[4]]
                    mass = sum(x[0] for x in outcomes)
                    if mass <= 0:
                        outcomes = [[1., *outcomes[0][1:]]]
                    else:
                        outcomes = [[x[0]/mass, *x[1:]] for x in outcomes]
                for probability, record, physical, shaping, _ in outcomes:
                    rows.append((edge.vehicle_id, record)); owners.append(owner)
                    probabilities.append(probability); rewards.append(physical-shaping); terminals.append(terminal)
            # Bound inference by actual outcome rows, including rejection branches.
            values, inference_stats = bounded_edge_inference(len(rows),
                min(self.qvalue_inference_batch_size,
                    getattr(self, '_neuradp_safe_batch_size', self.qvalue_inference_batch_size)),
                lambda sub: self._network_records(graph, rows[sub], target, context), device=self.device)
            if inference_stats['oom_retries']:
                self._neuradp_safe_batch_size = inference_stats['effective_batch_size']
            q = np.asarray(probabilities)*(np.asarray(rewards)+self.gamma*values*(1-np.asarray(terminals, dtype=float)))
            return np.bincount(owners, weights=q, minlength=len(edges)).astype(np.float32)
        with torch.no_grad():
            values, self._last_qvalue_inference_stats = bounded_edge_inference(
                len(learned), self.qvalue_inference_batch_size, evaluate, device=self.device)
        scores.update((edge.edge_id, float(value)) for edge, value in zip(learned, values))
        return scores

    def _network_records(self, graph, rows, target, context):
        arrays = self._record_rows(graph, rows, next_epoch=True, context=context)
        network = self.target_network if target else self.network
        return network(*(torch.as_tensor(x, device=self.device) for x in arrays)).cpu().numpy()

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

    def _saved_graph_targets(self, graph) -> dict[int, float]:
        """Re-solve this saved current graph, not its successor."""
        scores = self._neuradp_scores(graph, target=True)
        selection = self.target_builder.project(graph, scores)
        selected = set(selection)
        targets: dict[int, float] = {}
        for edge in graph.edges:
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

    def _joint_row_ready(self, transition, *, ifEV):
        return (not ifEV and transition.mode == 'integrated'
                and transition.ev_stage_graph is not None
                and bool(transition.ev_stage_graph.edges))

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
        losses, priorities, used_indices = [], [], []
        steps = 0
        # Author code updates each sampled scenario sequentially, including a
        # target-network soft update before resolving the next scenario.
        for transition, replay_index, weight in zip(sample.transitions, sample.indices, sample.weights):
            if transition.learner_variant != self.learner_variant:
                raise ValueError('joint replay learner variant mismatch')
            graph = transition.ev_stage_graph
            targets = self._saved_graph_targets(graph)
            vids = sorted(targets)
            if not vids:
                continue
            arrays = self._current_rows(graph, vids)
            target_values = np.asarray([targets[vid] for vid in vids], dtype=np.float32)
            order = self._noise_rng.permutation(len(vids))
            fit_batch = self.neuradp_config['fit_batch_size']
            self.network.train()
            for start in range(0, len(vids), fit_batch):
                index = order[start:start+fit_batch]
                prediction = self.network(*(torch.as_tensor(array[index], device=self.device) for array in arrays))
                target_batch = torch.as_tensor(target_values[index], device=self.device)
                loss = float(weight)*F.mse_loss(prediction, target_batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                losses.append(float(loss.detach().item()))
                steps += 1
            # Original replay priority is mean squared error, not mean absolute TD.
            error = target_values-self._predict_arrays(arrays)
            priorities.append(float(np.mean(error**2)+1e-6))
            used_indices.append(int(replay_index))
            self._soft_update(self.network, self.target_network, self.tau)
        if not losses:
            return 0.
        self.joint_replay_buffer.update_priorities(used_indices, priorities)
        self.joint_replay_buffer.advance_beta()
        mean_loss = float(np.mean(losses))
        self.training_losses.append(mean_loss)
        self.neuradp_td_errors.append(float(np.mean(priorities)))
        if len(self.neuradp_td_errors) > 10_000:
            del self.neuradp_td_errors[:-10_000]
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
        if version != NEURADP_IMPLEMENTATION_VERSION:
            raise ValueError(f"NeurADP checkpoint version {version} uses different value-state semantics; retrain with implementation v{NEURADP_IMPLEMENTATION_VERSION}")
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
        self.neuradp_td_errors = list(state.get("neuradp_td_errors", self.neuradp_td_errors))
        target_state = state.get("neuradp_target_network_state_dict")
        # Inference checkpoints store only the online network.
        self.target_network.load_state_dict(
            target_state if target_state is not None else self.network.state_dict()
        )
