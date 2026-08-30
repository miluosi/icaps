"""Critic-pair identity rules shared by synthetic and NYC trainers."""

from __future__ import annotations

from typing import Any


SHARED_CRITIC_STATE_VARIANTS = {
    "joint_state_shared_critic",
    "fleet_local_shared_critic",
    "strict_fleet_local_shared_critic",
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


def wire_recourse_critics(
    aev_value_function: Any,
    ev_value_function: Any,
    *,
    state_variant: str,
) -> tuple[Any, Any]:
    """Deploy and validate the fleet router and the lagged R4 follower.

    This function is intentionally called after checkpoint restoration.  A
    checkpoint may replace replay/module state, but it must never be allowed
    to leave the production rollout with the constructor's self-routing
    defaults.
    """

    aev_value_function, ev_value_function = enforce_critic_identity(
        aev_value_function,
        ev_value_function,
        state_variant=state_variant,
    )
    unique_value_functions = {
        id(aev_value_function): aev_value_function,
        id(ev_value_function): ev_value_function,
    }.values()
    for value_function in unique_value_functions:
        setter = getattr(value_function, "set_joint_critic_router", None)
        if callable(setter):
            setter(
                ev_value_function=ev_value_function,
                aev_value_function=aev_value_function,
            )

    # Joint transition payloads are immutable and fleet-agnostic.  Keep one
    # in-memory/checkpoint copy even when the fleet critics are separate.
    aev_replay = getattr(aev_value_function, "joint_replay_buffer", None)
    ev_replay = getattr(ev_value_function, "joint_replay_buffer", None)
    if aev_replay is not None:
        aev_value_function._owns_joint_replay_payload = True
        if ev_value_function is not aev_value_function and ev_replay is not None:
            ev_value_function.joint_replay_buffer = aev_replay
            ev_value_function._owns_joint_replay_payload = False

    follower_setter = getattr(
        ev_value_function, "set_follower_target_provider", None
    )
    follower_provider = getattr(
        aev_value_function, "target_components_for_graph", None
    )
    if callable(follower_setter) and callable(follower_provider):
        follower_setter(follower_provider)

    for value_function in unique_value_functions:
        validator = getattr(value_function, "validate_recourse_wiring", None)
        if callable(validator):
            validator()
    return aev_value_function, ev_value_function
