"""Critic-pair identity rules shared by synthetic and NYC trainers."""

from __future__ import annotations

from typing import Any


SHARED_CRITIC_STATE_VARIANTS = {
    "joint_state_shared_critic",
    "fleet_local_shared_critic",
}


def uses_shared_critic(state_variant: str) -> bool:
    return str(state_variant) in SHARED_CRITIC_STATE_VARIANTS


def enforce_critic_identity(
    aev_value_function: Any,
    ev_value_function: Any,
    *,
    state_variant: str,
) -> tuple[Any, Any]:
    """Return the deployed pair, preserving object identity for shared modes."""

    if uses_shared_critic(state_variant):
        return aev_value_function, aev_value_function
    if aev_value_function is ev_value_function:
        raise ValueError(
            "separate-critic state variant requires distinct EV and AEV critics"
        )
    return aev_value_function, ev_value_function
