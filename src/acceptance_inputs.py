"""Pre-offer neural acceptance inputs, identical for live state and replay.

No response, oracle, random draw, driver/request identifier or future demand is
a predictor input. NYC uses minutes/km; synthetic uses steps/grid distances.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math


FEATURE_VERSION = 2
SCHEMAS = {"synthetic_steps", "nyc_minutes"}
FEATURE_NAMES = (
    "idle_time", "pickup_time", "surge_bonus", "pickup_distance", "trip_distance",
    "trip_time", "base_fare", "offered_fare", "surge_multiplier", "battery_level",
    "request_age", "pickup_slack", "dropoff_slack", "time_of_day_sin", "time_of_day_cos",
    "vehicle_x", "vehicle_y", "pickup_x", "pickup_y", "dropoff_x", "dropoff_y",
    "available_ev_count", "available_aev_count", "pending_request_count",
    "pickup_pending_requests", "pickup_available_vehicles", "dropoff_pending_requests",
    "dropoff_available_vehicles", "pickup_demand_supply_ratio", "dropoff_demand_supply_ratio",
)


def vehicle_value(vehicle, key, default=None):
    if isinstance(vehicle, Mapping):
        return vehicle.get(key, default)
    aliases = {"type": "vehicle_type", "idle_timer": "idle_time", "is_online": "online"}
    return getattr(vehicle, aliases.get(key, key), default)


def offer_context(env, *, snapshot=None):
    """Capture dynamic time/supply/demand once per decision batch.

    When a snapshot is supplied, live vehicles/requests/time are never read.
    Zone grouping is static map information, available in both execution paths.
    """
    vehicles = snapshot.vehicles if snapshot is not None else env.vehicles.values()
    requests = snapshot.requests if snapshot is not None else getattr(env, 'active_requests', {}).values()
    now = float(snapshot.current_time if snapshot is not None else getattr(env, 'current_time', 0.0))
    occupied, supply, kinds = set(), Counter(), Counter()
    for vehicle in vehicles:
        for key in ('assigned_request', 'passenger_onboard'):
            request_id = vehicle_value(vehicle, key)
            if request_id is not None:
                occupied.add(int(request_id))
        available = (vehicle_value(vehicle, 'is_online', True)
                     and all(vehicle_value(vehicle, key) is None for key in
                             ('assigned_request', 'passenger_onboard', 'charging_station', 'charging_target'))
                     and float(vehicle_value(vehicle, 'penalty_timer', 0.0)) <= 0)
        if available:
            supply[_zone(env, vehicle_value(vehicle, 'location', 0))] += 1
            kinds[int(vehicle_value(vehicle, 'type', 1))] += 1
    demand = Counter(_zone(env, request.pickup) for request in requests
                     if int(request.request_id) not in occupied)
    return dict(current_time=now, supply=supply, demand=demand,
                available_ev_count=kinds[1], available_aev_count=kinds[2],
                pending_request_count=sum(demand.values()))


def _zone(env, location):
    location = int(location)
    if callable(getattr(env, 'get_travel_time_minutes', None)):
        return location  # NYC pricing uses individual TLC zones, not zone groups.
    return getattr(env, 'loc_to_zone', {}).get(location, location)


def _coordinates(env, location, nyc):
    if nyc:
        coordinates = getattr(env, 'zone_coords', {})
        if int(location) not in coordinates:
            raise ValueError(f'Missing static NYC coordinates for location {location}')
        latitude, longitude = coordinates[int(location)]
        return float(longitude), float(latitude)
    width = max(1, int(getattr(env, 'grid_size', 1)))
    return float(int(location) % width), float(int(location) // width)


def offer_features(env, vehicle_id, request, *, vehicle=None, context=None, snapshot=None):
    """Build every feature before the driver's response; fail on missing schema.

    A snapshot vehicle requires the full snapshot/context, since dynamic market
    features cannot safely be reconstructed from today's environment.
    """
    if context is None:
        if vehicle is not None and not isinstance(vehicle, Mapping) and snapshot is None:
            raise ValueError('Snapshot vehicle requires pre-offer snapshot/context')
        context = offer_context(env, snapshot=snapshot)
    if vehicle is None:
        vehicle = (next(v for v in snapshot.vehicles if v.vehicle_id == int(vehicle_id))
                   if snapshot is not None else env.vehicles[int(vehicle_id)])
    nyc = callable(getattr(env, 'get_travel_time_minutes', None))
    conversion = float(env.EPOCH_LENGTH) / 60.0 if nyc else 1.0
    now = context['current_time']
    location = int(vehicle_value(vehicle, 'location'))
    pickup, dropoff = int(request.pickup), int(request.dropoff)
    distance = env.get_distance_km if nyc else env._manhattan_distance_loc
    pickup_distance = float(distance(location, pickup))
    trip_distance = getattr(request, 'trip_distance_km', None) if nyc else None
    if trip_distance is None:
        trip_distance = float(distance(pickup, dropoff))
    pickup_time = float(env.get_travel_time_minutes(location, pickup)) if nyc else pickup_distance
    base, offered = float(request.value), float(request.final_value)
    age = max(0.0, now - float(request.created_time)) * conversion
    # Missing/infinite deadlines are not accepted as silently fabricated inputs.
    pickup_slack = (float(request.pickup_deadline) - now) * conversion
    dropoff_slack = (float(request.dropoff_deadline) - now) * conversion
    if nyc:
        # NYC resets the same demand window on each simulated day. Use its
        # pure time mapping with an explicit snapshot time, never live time.
        if callable(getattr(env, 'get_hour_of_day', None)):
            day_fraction = float(env.get_hour_of_day(now)) / 24.0
        else:
            day_fraction = ((float(getattr(env, 'START_EPOCH', 0.0)) + now * float(env.EPOCH_LENGTH)) % 86400.0) / 86400.0
    else:
        period = max(1.0, float(getattr(env, 'simulation_period', 1.0)))
        day_fraction = (now % period) / period
    vx, vy = _coordinates(env, location, nyc)
    px, py = _coordinates(env, pickup, nyc)
    dx, dy = _coordinates(env, dropoff, nyc)
    demand, supply = context['demand'], context['supply']
    pz, dz = _zone(env, pickup), _zone(env, dropoff)
    row = dict(
        feature_version=FEATURE_VERSION, feature_schema='nyc_minutes' if nyc else 'synthetic_steps',
        idle_time=float(vehicle_value(vehicle, 'idle_timer', 0.0)) * conversion,
        pickup_time=pickup_time, surge_bonus=float(env._request_surge_bonus(request)),
        pickup_distance=pickup_distance, trip_distance=float(trip_distance),
        trip_time=float(request.travel_time) * conversion, base_fare=base, offered_fare=offered,
        surge_multiplier=offered / base if base > 0 else 1.0,
        battery_level=float(vehicle_value(vehicle, 'battery', 1.0)),
        request_age=age, pickup_slack=pickup_slack, dropoff_slack=dropoff_slack,
        time_of_day_sin=math.sin(2 * math.pi * day_fraction), time_of_day_cos=math.cos(2 * math.pi * day_fraction),
        vehicle_x=vx, vehicle_y=vy, pickup_x=px, pickup_y=py, dropoff_x=dx, dropoff_y=dy,
        available_ev_count=float(context['available_ev_count']), available_aev_count=float(context['available_aev_count']),
        pending_request_count=float(context['pending_request_count']),
        pickup_pending_requests=float(demand.get(pz, 0)), pickup_available_vehicles=float(supply.get(pz, 0)),
        dropoff_pending_requests=float(demand.get(dz, 0)), dropoff_available_vehicles=float(supply.get(dz, 0)),
        pickup_demand_supply_ratio=float(demand.get(pz, 0)) / max(1, supply.get(pz, 0)),
        dropoff_demand_supply_ratio=float(demand.get(dz, 0)) / max(1, supply.get(dz, 0)),
    )
    if not all(math.isfinite(row[name]) for name in FEATURE_NAMES):
        raise ValueError('Pre-offer neural features must all be finite')
    return row
