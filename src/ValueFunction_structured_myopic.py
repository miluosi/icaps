"""Pure structured assignment comparator on the canonical recourse graph.

The class deliberately reuses the production graph encoder/checkpoint shape so
the environment, feasible graph, exact projection, state mask, and physical
Macro recourse path are identical to the learned comparators.  Neural tensors
are never consulted by deployment and no optimizer is stepped.
"""

from __future__ import annotations

from src.ValueFunction_optimization_anchored_residual import (
    PyTorchChargingValueFunction as _ResidualValueFunction,
)


class PyTorchChargingValueFunction(_ResidualValueFunction):
    learner_variant = "structured_myopic"
    direct_q = False
    uses_solver_consistent_targets = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_objective_mode = "structured_only"

    def set_planning_objective_mode(self, mode: str) -> str:
        if mode not in {"learned", "structured_only"}:
            raise ValueError(f"invalid planning objective mode: {mode}")
        # EV-first temporarily switches the follower objective by recourse
        # policy.  A pure myopic arm must remain structured at both stages.
        previous = self.planning_objective_mode
        self.planning_objective_mode = "structured_only"
        return previous

    def train_step(self, *args, **kwargs) -> float:
        del args, kwargs
        return 0.0

    def train_queue_predictor(self, *args, **kwargs) -> float:
        del args, kwargs
        return 0.0
