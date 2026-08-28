"""Main optimization-anchored residual learner.

This class keeps the structured edge score in the deployed objective and learns
its continuation correction. When an EV response model is enabled, unanswered
human service edges use the calibrated rejection-mixture anchor. The ordinary
single-stage Bellman reward remains the realized execution reward; existing
optional recourse behavior in the shared base is not expanded here.
"""

from __future__ import annotations

import torch

from src.ValueFunction_st_masac_gat_post_demand_direct import (
    PyTorchChargingValueFunction as _PostDemandResidual,
)


class PyTorchChargingValueFunction(_PostDemandResidual):
    learner_variant = "optimization_anchored_residual"
    uses_solver_consistent_targets = True
    uses_response_aware_anchor = True

    def __init__(self, *args, **kwargs):
        # These experiment-manifest fields describe this learner but are not
        # constructor arguments of the historical post-demand base class.
        self.entropy_target_ratio = float(
            kwargs.pop("entropy_target_ratio", 0.98)
        )
        self.residual_target_policy = str(
            kwargs.pop("residual_target_policy", "joint_projection")
        )
        self.predictor_variant = str(
            kwargs.pop("predictor_variant", "default")
        )
        super().__init__(*args, **kwargs)
        self.zone_distribution_mode = "optimization_anchored_residual"

    def _selection_residual(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        type_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Online critics select the feasible action; clipped target critics
        # evaluate it in the Bellman target (double-Q semantics).
        return 0.5 * (q1 + q2) * type_weights
