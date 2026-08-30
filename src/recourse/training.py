"""Shared trainer scheduling rules for edge, auxiliary, and joint replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingReadiness:
    edge_ready: bool
    queue_ready: bool
    joint_ready: bool
    predictor_ready: bool = False
    follower_critic_ready: bool = False

    @property
    def any_ready(self) -> bool:
        return bool(self.edge_ready or self.queue_ready or self.joint_ready)


def training_readiness(
    value_function: Any | None,
    *,
    ifEV: bool,
    edge_warmup: int,
    queue_warmup: int = 4,
) -> TrainingReadiness:
    if value_function is None:
        return TrainingReadiness(False, False, False, False, False)
    variant = str(getattr(value_function, 'recourse_variant', 'legacy'))
    follower_frozen = bool(
        not ifEV and variant in {'r1_structured', 'r2'}
    )
    predictors_frozen = bool(
        variant in {'r1_structured', 'r2', 'r3'}
        and getattr(value_function, 'freeze_causal_predictors', True)
    )
    edge_ready = len(
        getattr(value_function, "experience_buffer", ())
    ) >= max(0, int(edge_warmup))
    predictor_ready = bool(
        not predictors_frozen
        and
        hasattr(value_function, "train_queue_predictor")
        and len(getattr(value_function, "queue_experience_buffer", ()))
        >= max(0, int(queue_warmup))
    )
    follower_critic_ready = bool(
        not follower_frozen
        and
        hasattr(value_function, "has_trainable_joint_rows")
        and value_function.has_trainable_joint_rows(ifEV=ifEV)
    )
    return TrainingReadiness(
        edge_ready and not follower_frozen,
        predictor_ready,
        follower_critic_ready,
        predictor_ready,
        follower_critic_ready,
    )
