"""Main optimization-anchored residual learner.

This class keeps the exact structured edge score in the deployed objective
and learns only its continuation correction.  Joint target selection and R4
stage coupling are implemented in the shared ST-MASAC base.
"""

from __future__ import annotations

import torch

from src.ValueFunction_st_masac_gat_post_demand_direct import (
    PyTorchChargingValueFunction as _PostDemandResidual,
)


class PyTorchChargingValueFunction(_PostDemandResidual):
    learner_variant = "optimization_anchored_residual"
    uses_solver_consistent_targets = True

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
