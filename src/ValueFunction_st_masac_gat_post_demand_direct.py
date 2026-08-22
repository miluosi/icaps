"""MASAC-GAT with a TD-learned post-action demand value head.

The post-demand predictor estimates active requests at the action completion
time and zone.  Its output is not consumed by the nonlinear critic or actor.
Instead, each twin critic has an additive linear demand-value head whose
action-specific coefficients are learned by the ordinary residual TD loss.
No environment reward or relocation cost is modified by this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn, optim

from src.ValueFunction_st_masac_gat import (
    PyTorchChargingValueFunction as _BaseMASAC,
)
from src.ValueFunction_st_masac_gat_post_demand import (
    PyTorchChargingValueFunction as _PostDemandFeatureMASAC,
)


class _DemandCorrectedCritic(nn.Module):
    """Keep demand out of the MLP and add an interpretable linear value head."""

    def __init__(
        self,
        base: nn.Module,
        *,
        demand_index: int,
        action_start_index: int,
        initial_weight: float,
        demand_clip: float,
    ):
        super().__init__()
        self.base = base
        self.demand_index = int(demand_index)
        self.action_start_index = int(action_start_index)
        self.demand_clip = max(0.0, float(demand_clip))
        self.action_weights = nn.Parameter(
            torch.full((3,), float(initial_weight), dtype=torch.float32)
        )

    def demand_correction(self, features: torch.Tensor) -> torch.Tensor:
        demand = torch.clamp(
            features[:, self.demand_index : self.demand_index + 1],
            0.0,
            self.demand_clip,
        )
        action_one_hot = features[
            :, self.action_start_index : self.action_start_index + 3
        ]
        action_weight = torch.sum(
            action_one_hot * self.action_weights.unsqueeze(0),
            dim=1,
            keepdim=True,
        )
        return demand * action_weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base_features = features.clone()
        base_features[:, self.demand_index] = 0.0
        return self.base(base_features) + self.demand_correction(features)


class _DemandMaskedActor(nn.Module):
    """Prevent the auxiliary demand prediction from becoming an actor feature."""

    def __init__(self, base: nn.Module, demand_index: int):
        super().__init__()
        self.base = base
        self.demand_index = int(demand_index)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base_features = features.clone()
        base_features[:, self.demand_index] = 0.0
        return self.base(base_features)


class PyTorchChargingValueFunction(_PostDemandFeatureMASAC):
    uses_post_demand_feature = False
    uses_post_demand_direct_q = True
    post_demand_direct_version = 2

    # _local_edge_features contains ten state values followed by the
    # reloc/request/charge one-hot values.
    action_one_hot_start_index = 10

    def __init__(
        self,
        *args,
        post_demand_q_weight: float = 0.0,
        post_demand_q_clip: float = 5.0,
        post_demand_head_lr_multiplier: float = 10.0,
        **kwargs,
    ):
        kwargs["zone_distribution_mode"] = "st_masac_gat_post_demand_direct"
        super().__init__(*args, **kwargs)
        self.zone_distribution_mode = "st_masac_gat_post_demand_direct"
        self.post_demand_q_initial_weight = float(post_demand_q_weight)
        self.post_demand_q_clip = max(0.0, float(post_demand_q_clip))
        self.post_demand_head_lr_multiplier = max(
            1.0,
            float(post_demand_head_lr_multiplier),
        )

        self.network = self._wrap_critic(self.network)
        self.critic2 = self._wrap_critic(self.critic2)
        self.target_network = self._wrap_critic(self.target_network)
        self.target_critic2 = self._wrap_critic(self.target_critic2)
        self.actor = _DemandMaskedActor(
            self.actor,
            self.post_demand_input_index,
        ).to(self.device)
        self.post_demand_weight_history: list[dict[str, float]] = []

        head_params = [
            self.network.action_weights,
            self.critic2.action_weights,
        ]
        head_param_ids = {id(parameter) for parameter in head_params}
        base_params = (
            [parameter for parameter in self.graph_encoder.parameters() if parameter.requires_grad]
            + list(self.mixer.parameters())
            + list(self.network.parameters())
            + list(self.critic2.parameters())
            + list(self.actor.parameters())
            + [self.log_alpha]
        )
        base_params = [
            parameter
            for parameter in base_params
            if id(parameter) not in head_param_ids
        ]
        self.optimizer = optim.Adam(
            [
                {"params": base_params, "lr": self.learning_rate},
                {
                    "params": head_params,
                    "lr": self.learning_rate * self.post_demand_head_lr_multiplier,
                },
            ]
        )
        print(
            "ST-ADP-MASAC-GAT TD-learned post-demand head initialized: "
            f"initial_weight={self.post_demand_q_initial_weight:g}, "
            f"clip={self.post_demand_q_clip:g}, "
            f"head_lr={self.learning_rate * self.post_demand_head_lr_multiplier:g}",
            flush=True,
        )

    def _wrap_critic(self, critic: nn.Module) -> _DemandCorrectedCritic:
        return _DemandCorrectedCritic(
            critic,
            demand_index=self.post_demand_input_index,
            action_start_index=self.action_one_hot_start_index,
            initial_weight=self.post_demand_q_initial_weight,
            demand_clip=self.post_demand_q_clip,
        ).to(self.device)

    def _conservative_action_weights(self) -> np.ndarray:
        with torch.no_grad():
            weights = torch.minimum(
                self.network.action_weights,
                self.critic2.action_weights,
            )
        return weights.detach().cpu().numpy().astype(np.float32)

    def batch_get_mixed_q_values(self, **kwargs):
        # The critic wrappers apply the demand correction before the standard
        # residual scaling and clipping.  No hand-authored action cap follows.
        values = _BaseMASAC.batch_get_mixed_q_values(self, **kwargs)
        demand = np.asarray(self._last_post_demand_features, dtype=np.float32)
        action_type_ids = np.asarray(kwargs["action_type_ids"], dtype=np.int64)
        weights = self._conservative_action_weights()
        action_weight = np.zeros(action_type_ids.size, dtype=np.float32)
        valid = (action_type_ids >= 1) & (action_type_ids <= 3)
        action_weight[valid] = weights[action_type_ids[valid] - 1]
        correction = np.clip(demand, 0.0, self.post_demand_q_clip) * action_weight

        score_stats = dict(getattr(self, "_last_adp_score_stats", {}) or {})
        score_stats.update({
            "mode": self.zone_distribution_mode,
            "post_demand_nonlinear_critic_input": 0.0,
            "post_demand_prediction_normalized_mean": (
                float(np.mean(demand)) if demand.size else 0.0
            ),
            "post_demand_prediction_count_mean": (
                float(np.mean(demand) * self.post_demand_scale) if demand.size else 0.0
            ),
            "post_demand_head_correction_mean": (
                float(np.mean(correction)) if correction.size else 0.0
            ),
            "post_demand_weight_reloc": float(weights[0]),
            "post_demand_weight_request": float(weights[1]),
            "post_demand_weight_charge": float(weights[2]),
        })
        self._last_adp_score_stats = score_stats
        return values

    def train_step(
        self,
        batch_size: int = 64,
        tau: float | None = None,
        ifEV: bool = False,
    ) -> float:
        loss = super().train_step(batch_size=batch_size, tau=tau, ifEV=ifEV)
        weights = self._conservative_action_weights()
        row = {
            "training_step": float(self.training_step),
            "reloc": float(weights[0]),
            "request": float(weights[1]),
            "charge": float(weights[2]),
        }
        self.post_demand_weight_history.append(row)
        if self.q_values_history:
            self.q_values_history[-1].update({
                "post_demand_weight_reloc": row["reloc"],
                "post_demand_weight_request": row["request"],
                "post_demand_weight_charge": row["charge"],
            })
        return loss

    def extra_checkpoint_state(self) -> dict[str, Any]:
        state = super().extra_checkpoint_state()
        state.pop("reloc_request_score_margin", None)
        state.pop("reloc_request_cap_total", None)
        state.update({
            "post_demand_direct_version": self.post_demand_direct_version,
            "post_demand_q_initial_weight": self.post_demand_q_initial_weight,
            "post_demand_q_clip": self.post_demand_q_clip,
            "post_demand_head_lr_multiplier": self.post_demand_head_lr_multiplier,
            "post_demand_weight_history": list(self.post_demand_weight_history),
            "post_demand_q1_action_weights": self.network.action_weights.detach().cpu(),
            "post_demand_q2_action_weights": self.critic2.action_weights.detach().cpu(),
            "post_demand_target_q1_action_weights": self.target_network.action_weights.detach().cpu(),
            "post_demand_target_q2_action_weights": self.target_critic2.action_weights.detach().cpu(),
        })
        return state

    @staticmethod
    def _restore_action_weights(module: _DemandCorrectedCritic, value: Any) -> None:
        if value is None:
            return
        tensor = torch.as_tensor(value, dtype=torch.float32, device=module.action_weights.device)
        if tensor.numel() == 3:
            module.action_weights.data.copy_(tensor.reshape(3))

    def load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super().load_extra_checkpoint_state(state)
        self.post_demand_q_clip = max(
            0.0,
            float(state.get("post_demand_q_clip", self.post_demand_q_clip)),
        )
        for module in (
            self.network,
            self.critic2,
            self.target_network,
            self.target_critic2,
        ):
            module.demand_clip = self.post_demand_q_clip
        self.post_demand_weight_history = list(
            state.get("post_demand_weight_history", self.post_demand_weight_history)
        )
        self._restore_action_weights(
            self.network,
            state.get("post_demand_q1_action_weights"),
        )
        self._restore_action_weights(
            self.critic2,
            state.get("post_demand_q2_action_weights"),
        )
        self._restore_action_weights(
            self.target_network,
            state.get("post_demand_target_q1_action_weights"),
        )
        self._restore_action_weights(
            self.target_critic2,
            state.get("post_demand_target_q2_action_weights"),
        )
