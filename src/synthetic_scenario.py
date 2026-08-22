"""Shared synthetic congestion settings for training and evaluation.

The legacy synthetic defaults (four plugs per station, eight-step charging,
one 50-step day) rarely create persistent queues.  The calibrated defaults use
a 24-by-24 map, six two-slot stations, fourteen-step charging, and a two-day
200-step horizon.  Predictive demand is scaled to 0.97: queues remain material
without making the myopic MCMF and HEU baselines incomparable.
"""

from __future__ import annotations


DEFAULT_NUM_STATIONS = 6
DEFAULT_STATION_CAPACITY = 2
DEFAULT_STATION_QUEUE_CAPACITY = 3
DEFAULT_CHARGE_DURATION = 14
DEFAULT_SIMULATION_PERIOD = 100
DEFAULT_EPISODE_DAYS = 2
DEFAULT_WAIT_PENALTY_PER_STEP = 0.5
DEFAULT_AEV_INITIAL_BATTERY_SCALE = 0.68
DEFAULT_CRITICAL_CHARGING_BATTERY = 0.22
DEFAULT_GRID_SIZE = 24
DEFAULT_SYNTHETIC_DEMAND_PROFILE = "predictive"
DEFAULT_SYNTHETIC_DEMAND_SCALE = 0.97

LEGACY_NUM_STATIONS = 3
LEGACY_STATION_CAPACITY = 4
LEGACY_STATION_QUEUE_CAPACITY = 0
LEGACY_CHARGE_DURATION = 8
LEGACY_SIMULATION_PERIOD = 50
LEGACY_EPISODE_DAYS = 1
LEGACY_WAIT_PENALTY_PER_STEP = 1.0
LEGACY_AEV_INITIAL_BATTERY_SCALE = 1.0
LEGACY_CRITICAL_CHARGING_BATTERY = 0.15


def _number_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def synthetic_checkpoint_suffix(
    *,
    num_stations: int,
    station_capacity: int,
    charge_duration: int,
    simulation_period: int,
    episode_days: int,
    charging_wait_penalty_per_step: float,
    station_queue_capacity: int = LEGACY_STATION_QUEUE_CAPACITY,
    aev_initial_battery_scale: float = LEGACY_AEV_INITIAL_BATTERY_SCALE,
    critical_charging_battery: float = LEGACY_CRITICAL_CHARGING_BATTERY,
    grid_size: int | None = None,
    synthetic_demand_profile: str | None = None,
    synthetic_demand_scale: float = 1.0,
) -> str:
    """Identify the environment used to train a synthetic checkpoint.

    The former default intentionally keeps an empty suffix so existing legacy
    checkpoints remain discoverable when all former parameters are supplied.
    Any other configuration receives an explicit tag, preventing evaluation
    from silently loading a model trained under different queue dynamics.
    """
    legacy = (
        int(num_stations) == LEGACY_NUM_STATIONS
        and int(station_capacity) == LEGACY_STATION_CAPACITY
        and int(charge_duration) == LEGACY_CHARGE_DURATION
        and int(simulation_period) == LEGACY_SIMULATION_PERIOD
        and int(episode_days) == LEGACY_EPISODE_DAYS
        and float(charging_wait_penalty_per_step)
        == LEGACY_WAIT_PENALTY_PER_STEP
        and int(station_queue_capacity) == LEGACY_STATION_QUEUE_CAPACITY
        and float(aev_initial_battery_scale)
        == LEGACY_AEV_INITIAL_BATTERY_SCALE
        and float(critical_charging_battery)
        == LEGACY_CRITICAL_CHARGING_BATTERY
    )
    if legacy:
        return ""
    horizon = int(simulation_period) * int(episode_days)
    suffix = (
        f"_synq_n{int(num_stations)}"
        f"_c{int(station_capacity)}"
        f"_d{int(charge_duration)}"
        f"_h{horizon}"
        f"_w{_number_tag(charging_wait_penalty_per_step)}"
        f"_q{int(station_queue_capacity)}"
        f"_ab{_number_tag(aev_initial_battery_scale)}"
        f"_cb{_number_tag(critical_charging_battery)}"
    )
    # Older callers intentionally omit these fields to retain legacy checkpoint
    # discovery.  Current synthetic entrypoints always supply them so a model
    # trained on a different map or demand process cannot be loaded silently.
    if grid_size is not None:
        suffix += f"_g{int(grid_size)}"
    if synthetic_demand_profile is not None:
        profile = "".join(
            ch for ch in str(synthetic_demand_profile).strip().lower()
            if ch.isalnum()
        ) or "unknown"
        suffix += f"_p{profile}"
    if abs(float(synthetic_demand_scale) - 1.0) > 1e-9:
        suffix += f"_ds{_number_tag(synthetic_demand_scale)}"
    return suffix
