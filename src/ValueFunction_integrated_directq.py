"""Integrated DirectQ comparator with the same graph and projection."""

from __future__ import annotations

import torch

from src.ValueFunction_st_masac_gat import (
    PyTorchChargingValueFunction as _ResidualValueFunction,
)


class PyTorchChargingValueFunction(_ResidualValueFunction):
    learner_variant = "integrated_directq"
    direct_q = True
    uses_solver_consistent_targets = True

    def __init__(self, *args, **kwargs):
        kwargs["zone_distribution_mode"] = "integrated_directq"
        super().__init__(*args, **kwargs)

    def _execution_scores_from_residual(
        self,
        g_t: torch.Tensor,
        residual: torch.Tensor,
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        del g_t, bounds
        return residual

