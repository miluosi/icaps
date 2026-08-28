"""Builders for immutable simulator and feasible-graph snapshots."""

from __future__ import annotations

from src.acceptance_features import predicted_acceptance
from src.acceptance_inputs import offer_context

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .types import (
    ActionType,
    FeasibleEdgeSnapshot,
    FeasibleGraphSnapshot,
    RequestSnapshot,
    StationSnapshot,
    SystemSnapshot,
    VehicleSnapshot,
)


class StateSnapshotBuilder:
    @staticmethod
    def epoch_id(env: Any) -> int:
        explicit = getattr(env, "epoch_id", None)
        if explicit is not None:
            return int(explicit)
        return int(round(float(getattr(env, "current_time", 0.0))))

    @classmethod
    def build(
        cls,
        env: Any,
        *,
        request_labels: Mapping[int, str] | None = None,
    ) -> SystemSnapshot:
        active_requests = getattr(env, "active_requests", {})
        vehicles = []
        for vehicle_id, vehicle in sorted(getattr(env, "vehicles", {}).items()):
            request_id = vehicle.get("passenger_onboard")
            if request_id is None:
                request_id = vehicle.get("assigned_request")
            vehicles.append(
                VehicleSnapshot.from_vehicle(
                    vehicle_id,
                    vehicle,
                    request=active_requests.get(request_id),
                    env=env,
                    current_time=float(getattr(env, "current_time", 0.0)),
                )
            )
        vehicles = tuple(vehicles)
        zone_candidates = list(
            getattr(env, "aux_zone_ids", ())
            or getattr(env, "relocation_target_ids", ())
            or sorted(getattr(env, "zone_coords", {}).keys())
        )
        requests = []
        for _, request in sorted(
            getattr(env, "active_requests", {}).items(),
            key=lambda item: int(item[0]),
        ):
            snapshot = RequestSnapshot.from_request(request)
            requests.append(
                replace(
                    snapshot,
                    pickup_zone_id=_zone_for_location(
                        env, snapshot.pickup, zone_candidates
                    ),
                )
            )
        station_rows = []
        stations = getattr(getattr(env, "charging_manager", None), "stations", {})
        for station_id, station in sorted(stations.items()):
            occupied = len(getattr(station, "current_vehicles", ()) or ())
            inbound = len(
                getattr(station, "charging_queue_notarrived", ()) or ()
            )
            queued = len(getattr(station, "charging_queue", ()) or ())
            physical_capacity = int(
                getattr(station, "max_capacity", 0) or 0
            )
            queue_capacity = max(
                0, int(getattr(env, "station_queue_capacity", 0) or 0)
            )
            reserve_inbound = bool(
                getattr(env, "reserve_inbound_charging_capacity", False)
            )
            reserved = occupied + (
                inbound + queued if reserve_inbound else 0
            )
            remaining = max(
                0, physical_capacity + queue_capacity - reserved
            )
            station_rows.append(
                StationSnapshot(
                    station_id=int(station_id),
                    location=int(getattr(station, "location", 0)),
                    capacity=physical_capacity,
                    occupied=occupied,
                    inbound=inbound,
                    queued=queued,
                    physical_capacity=physical_capacity,
                    queue_admission_capacity=queue_capacity,
                    reserve_inbound_capacity=reserve_inbound,
                    remaining_admission_capacity=remaining,
                )
            )
        zones = zone_candidates
        if not zones:
            zones = list(range(max(1, int(getattr(env, "NUM_LOCATIONS", 1)))))
        day_id = ""
        current_date = getattr(env, "_current_date_label", None)
        if callable(current_date):
            try:
                day_id = str(current_date())
            except Exception:
                day_id = ""
        context = (
            ("day_id", day_id),
            ("decision_mode", str(getattr(env, "decision_mode", "integrated"))),
        )
        return SystemSnapshot(
            epoch_id=cls.epoch_id(env),
            current_time=float(getattr(env, "current_time", 0.0)),
            zone_ids=tuple(int(zone) for zone in zones),
            vehicles=vehicles,
            requests=tuple(requests),
            stations=tuple(station_rows),
            request_labels=tuple(
                sorted((int(key), str(value)) for key, value in (request_labels or {}).items())
            ),
            exogenous_context=context,
        )

    @classmethod
    def feasible_graph_from_matrix(
        cls,
        env: Any,
        vehicle_ids: list[int],
        action_matrix: np.ndarray,
        score_matrix: np.ndarray,
        structured_matrix: np.ndarray,
        *,
        num_requests: int,
        num_stations: int,
        num_zones: int,
        stage_id: int,
        solver_backend: str,
        state: SystemSnapshot | None = None,
    ) -> FeasibleGraphSnapshot:
        state = state or cls.build(env)
        request_ids = list(getattr(env, "_last_matrix_request_ids", ()))[:num_requests]
        request_map = {request.request_id: request for request in state.requests}
        station_ids = list(getattr(env, "_last_matrix_charge_station_ids", ()))[:num_stations]
        station_map = {station.station_id: station for station in state.stations}
        zone_indices = list(getattr(env, "_last_matrix_zone_indices", ()))[:num_zones]
        zone_ids = list(getattr(env, "_last_matrix_zone_target_ids", ()))[:num_zones]
        vehicle_map = {vehicle.vehicle_id: vehicle for vehicle in state.vehicles}
        acceptance_context = (offer_context(env, snapshot=state)
                              if getattr(env, 'ev_acceptance_feature', 'off') == 'predicted' else None)
        edges: list[FeasibleEdgeSnapshot] = []

        for row, vehicle_id in enumerate(vehicle_ids):
            vehicle = vehicle_map[int(vehicle_id)]
            feasible_columns = np.flatnonzero(np.asarray(action_matrix[row]) > 0)
            if vehicle.vehicle_type == 1 and num_stations + num_zones > 0:
                final_column = int(action_matrix.shape[1]) - 1
                feasible_columns = np.asarray(
                    [
                        column
                        for column in feasible_columns
                        if int(column) < num_requests
                        or int(column) == final_column
                    ],
                    dtype=np.int64,
                )
            for column in feasible_columns:
                column = int(column)
                request_id = None
                station_id = None
                resource_type = None
                resource_id = None
                resource_capacity = 1
                request_value = 0.0
                target_location = vehicle.location
                post_location = vehicle.location
                target_distance = 0.0
                post_distance = 0.0
                post_duration = 0.0
                target_zoneid = 0
                post_zoneid = 0
                queue_features: tuple[float, ...] = ()
                edge_metadata: tuple[
                    tuple[str, float | int | str | bool | None], ...
                ] = ()

                if column < num_requests:
                    if column >= len(request_ids) or request_ids[column] not in request_map:
                        continue
                    request_id = int(request_ids[column])
                    request = request_map[request_id]
                    action_type = ActionType.SERVICE
                    action_id = f"assign_{request_id}"
                    target_location = request.pickup
                    post_location = request.dropoff
                    request_value = request.final_value
                    target_distance = _distance(env, vehicle.location, request.pickup)
                    trip_distance = (
                        request.trip_distance_km
                        if request.trip_distance_km is not None
                        else _distance(env, request.pickup, request.dropoff)
                    )
                    post_distance = target_distance + float(trip_distance)
                    pickup_duration = _travel_time(
                        env, vehicle.location, request.pickup
                    )
                    trip_duration = _travel_time(
                        env, request.pickup, request.dropoff
                    )
                    post_duration = pickup_duration + trip_duration
                    resource_type = "request"
                    resource_id = request_id
                elif column < num_requests + num_stations:
                    local_column = column - num_requests
                    if local_column >= len(station_ids):
                        continue
                    station_id = int(station_ids[local_column])
                    station = station_map.get(station_id)
                    if station is None:
                        continue
                    action_type = ActionType.CHARGE
                    action_id = f"charge_{station_id}"
                    target_location = station.location
                    post_location = station.location
                    target_distance = _distance(env, vehicle.location, station.location)
                    post_distance = target_distance
                    station_travel_duration = _travel_time(
                        env, vehicle.location, station.location
                    )
                    post_duration = station_travel_duration + float(
                        getattr(env, "charge_duration", 1.0)
                    )
                    resource_type = "station"
                    resource_id = station_id
                    resource_capacity = int(
                        station.remaining_admission_capacity
                    )
                elif column < num_requests + num_stations + num_zones:
                    local_column = column - num_requests - num_stations
                    if local_column >= len(zone_ids):
                        continue
                    target_location = int(zone_ids[local_column])
                    post_location = target_location
                    action_type = ActionType.RELOCATE
                    action_id = f"reloc_{target_location}"
                    target_distance = _distance(env, vehicle.location, target_location)
                    post_distance = target_distance
                    post_duration = _travel_time(env, vehicle.location, target_location)
                    serialized_zone_index = (
                        int(zone_indices[local_column])
                        if local_column < len(zone_indices)
                        else int(local_column)
                    )
                    edge_metadata = (
                        ("zone_column", int(local_column)),
                        ("zone_index", serialized_zone_index),
                    )
                else:
                    if vehicle.vehicle_type == 1:
                        target_location = int(
                            getattr(env, "_ev_default_relocation_targets", {}).get(
                                int(vehicle_id), vehicle.location
                            )
                        )
                        post_location = target_location
                        target_distance = _distance(
                            env, vehicle.location, target_location
                        )
                        post_distance = target_distance
                        post_duration = _travel_time(
                            env, vehicle.location, target_location
                        )
                        action_type = ActionType.RELOCATE
                        action_id = f"reloc_{target_location}"
                    else:
                        action_type = ActionType.WAIT
                        action_id = "wait"

                zone_fn = getattr(env, "get_zone_embedding_id", None)
                if callable(zone_fn):
                    try:
                        target_zoneid = int(zone_fn(target_location))
                        post_zoneid = int(zone_fn(post_location))
                    except Exception:
                        target_zoneid = post_zoneid = 0
                value_function = _value_function_for_vehicle(
                    env, vehicle.vehicle_type
                )
                queue_feature_builder = getattr(
                    value_function, "_queue_features", None
                )
                if action_type == ActionType.CHARGE and callable(
                    queue_feature_builder
                ):
                    try:
                        queue_features = tuple(
                            float(value)
                            for value in queue_feature_builder(
                                station_id=station_id,
                                target_location=target_location,
                                vehicle_id=int(vehicle_id),
                                vehicle_location=vehicle.location,
                                current_time=state.current_time,
                                num_requests=float(len(state.requests)),
                                travel_duration=station_travel_duration,
                            )
                        )
                    except (TypeError, ValueError, RuntimeError):
                        queue_features = ()
                edge_id = f"{stage_id}:{int(vehicle_id)}:{column}:{action_id}"
                edges.append(
                    FeasibleEdgeSnapshot(
                        edge_id=edge_id,
                        vehicle_id=int(vehicle_id),
                        vehicle_type=vehicle.vehicle_type,
                        action_type=action_type,
                        action_id=action_id,
                        target_location=int(target_location),
                        post_action_location=int(post_location),
                        request_id=request_id,
                        station_id=station_id,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        resource_capacity=int(resource_capacity),
                        structured_score=float(structured_matrix[row, column]),
                        collection_score=float(score_matrix[row, column]),
                        request_value=float(request_value),
                        target_distance=float(target_distance),
                        post_action_distance=float(post_distance),
                        post_action_duration=float(post_duration),
                        target_zoneid=target_zoneid,
                        post_action_zoneid=post_zoneid,
                        queue_features=queue_features,
                        acceptance_probability=(
                            predicted_acceptance(env, vehicle_id, request_map[request_id], vehicle=vehicle,
                                                 context=acceptance_context, snapshot=state)
                            if request_id is not None and vehicle.vehicle_type == 1 else 0.0
                        ),
                        metadata=edge_metadata,
                    )
                )
        cls._append_continuing_edges(
            env,
            state,
            edges,
            matrix_vehicle_ids={int(vehicle_id) for vehicle_id in vehicle_ids},
            stage_id=int(stage_id),
        )

        # Post-demand predictors use mutable active-request counts.  Evaluate
        # them once at collection and persist the scalar on every edge so a
        # replayed graph never consults the later live environment.
        predictor_groups: dict[int, tuple[Any, list[int]]] = {}
        for index, edge in enumerate(edges):
            value_function = _value_function_for_vehicle(
                env, edge.vehicle_type
            )
            predictor = getattr(value_function, "predict_post_action_demand", None)
            if callable(predictor):
                key = id(value_function)
                predictor_groups.setdefault(key, (value_function, []))[1].append(index)
        for value_function, indices in predictor_groups.values():
            predictor = value_function.predict_post_action_demand
            try:
                serialized_zones = list(state.zone_ids)
                edge_zones = {
                    index: _zone_for_location(
                        env,
                        edges[index].post_action_location,
                        serialized_zones,
                    )
                    for index in indices
                }
                zone_demand = {
                    zone_id: sum(
                        int(request.pickup_zone_id == zone_id)
                        for request in state.requests
                    )
                    for zone_id in set(edge_zones.values())
                }
                predictions = predictor(
                    current_times=[state.current_time for _ in indices],
                    post_action_durations=[
                        edges[index].post_action_duration for index in indices
                    ],
                    post_action_locations=[
                        edges[index].post_action_location for index in indices
                    ],
                    num_requests=[float(len(state.requests)) for _ in indices],
                    current_zone_demands=[
                        float(zone_demand[edge_zones[index]])
                        for index in indices
                    ],
                    snapshot_available=[1.0 for _ in indices],
                )
                for index, prediction in zip(indices, predictions):
                    edges[index] = replace(
                        edges[index], post_demand_feature=float(prediction)
                    )
            except (TypeError, ValueError, RuntimeError):
                # Models without the snapshot-aware predictor signature keep
                # the explicit ``None`` marker; they are never silently fed a
                # value calculated from a future mutable state.
                pass
        pending = getattr(
            getattr(env, "recourse_coordinator", None), "pending", None
        )
        base_graph_id = (
            getattr(pending, "transition_id", None)
            or f"{getattr(env, 'recourse_run_id', 'run')}:{state.epoch_id}"
        )
        return FeasibleGraphSnapshot(
            graph_id=f"{base_graph_id}:graph:{int(stage_id)}",
            epoch_id=state.epoch_id,
            stage_id=int(stage_id),
            solver_backend=str(solver_backend),
            state=state,
            edges=tuple(edges),
            objective_cost_scale=max(
                1, int(getattr(env, "mcmf_cost_scale", 10_000) or 10_000)
            ),
            objective_precision_mode="integer_q_grid",
        )

    @staticmethod
    def selected_edge_ids(
        graph: FeasibleGraphSnapshot,
        assignments: Mapping[int, Any],
    ) -> tuple[str, ...]:
        selected = []
        for vehicle_id, target in assignments.items():
            target_request_id = getattr(target, "request_id", None)
            target_station_id = None
            target_location = None
            target_zone_index = None
            text = str(target)
            if isinstance(target, str) and target.startswith("charge_"):
                try:
                    target_station_id = int(target.split("_", 1)[1])
                except ValueError:
                    pass
            if isinstance(target, str) and target.startswith("idle_at_"):
                try:
                    target_zone_index = int(target.split("_", 2)[2])
                except ValueError:
                    pass
            if isinstance(target, str) and target.startswith("reloc_to_"):
                try:
                    target_location = int(target.rsplit("_", 1)[1])
                except ValueError:
                    pass
            candidates = [edge for edge in graph.edges if edge.vehicle_id == int(vehicle_id)]
            match = None
            for edge in candidates:
                if target_request_id is not None and edge.request_id == int(target_request_id):
                    match = edge
                    break
                if target_station_id is not None and edge.station_id == target_station_id:
                    match = edge
                    break
                if (
                    target_zone_index is not None
                    and edge.action_type == ActionType.RELOCATE
                    and dict(edge.metadata).get("zone_index")
                    == target_zone_index
                ):
                    match = edge
                    break
                if (
                    target_location is not None
                    and edge.action_type == ActionType.RELOCATE
                    and edge.target_location == target_location
                ):
                    match = edge
                    break
                if target is None and edge.action_type == ActionType.WAIT:
                    match = edge
                    break
                if isinstance(target, str) and target == "waiting" and edge.action_type == ActionType.WAIT:
                    match = edge
                    break
                if isinstance(target, str) and target == "reloc" and edge.action_type == ActionType.RELOCATE:
                    match = edge
                    break
            if match is None:
                raise AssertionError(
                    f"executed assignment for vehicle {vehicle_id} cannot be "
                    f"mapped to its serialized graph edge: {target!r}"
                )
            selected.append(match.edge_id)
        for edge in graph.edges:
            if bool(dict(edge.metadata).get("continuing", False)):
                selected.append(edge.edge_id)
        return tuple(selected)

    @staticmethod
    def _append_continuing_edges(
        env: Any,
        state: SystemSnapshot,
        edges: list[FeasibleEdgeSnapshot],
        *,
        matrix_vehicle_ids: set[int],
        stage_id: int,
    ) -> None:
        mode = str(dict(state.exogenous_context).get("decision_mode", "integrated"))
        if mode == "integrated":
            fleet_types = {1, 2}
        elif mode in {"evfirst", "ev_first"}:
            fleet_types = {1} if stage_id == 1 else {2}
        elif mode in {"aevfirst", "aev_first"}:
            fleet_types = {2} if stage_id == 1 else {1}
        else:
            fleet_types = {1, 2}
        request_map = {request.request_id: request for request in state.requests}
        station_map = {station.station_id: station for station in state.stations}
        for vehicle in state.vehicles:
            if (
                not vehicle.online
                or vehicle.vehicle_type not in fleet_types
                or vehicle.vehicle_id in matrix_vehicle_ids
            ):
                continue
            request_id = vehicle.service_request_id
            station_id = (
                vehicle.charging_station
                if vehicle.charging_station is not None
                else vehicle.charging_target
            )
            target = vehicle.location
            post_location = vehicle.location
            request_value = 0.0
            post_duration = 1.0
            post_distance = 0.0
            resource_type = None
            resource_id = None
            resource_capacity = 1
            queue_features: tuple[float, ...] = ()
            metadata: dict[str, float | int | str | bool | None] = {
                "continuing": True,
                "service_phase": vehicle.service_phase,
                "remaining_pickup_time": vehicle.remaining_pickup_time,
                "remaining_trip_time": vehicle.remaining_trip_time,
                "remaining_service_distance": vehicle.remaining_service_distance,
                "charging_time_left": vehicle.charging_time_left,
                "remaining_relocation_time": vehicle.remaining_relocation_time,
                "stationary_duration_left": vehicle.stationary_duration_left,
            }
            if request_id is not None:
                action_type = ActionType.SERVICE
                action_id = f"assign_{request_id}:continue"
                request = request_map.get(request_id)
                request_value = float(
                    vehicle.service_request_value
                    or getattr(request, "final_value", 0.0)
                    or 0.0
                )
                if vehicle.service_phase == "passenger_onboard":
                    target = int(
                        vehicle.service_dropoff
                        if vehicle.service_dropoff is not None
                        else vehicle.location
                    )
                    post_duration = max(1.0, vehicle.remaining_trip_time)
                else:
                    target = int(
                        vehicle.service_pickup
                        if vehicle.service_pickup is not None
                        else vehicle.location
                    )
                    post_duration = max(
                        1.0,
                        vehicle.remaining_pickup_time
                        + vehicle.remaining_trip_time,
                    )
                post_location = int(
                    vehicle.service_dropoff
                    if vehicle.service_dropoff is not None
                    else target
                )
                post_distance = vehicle.remaining_service_distance
                resource_type = "request"
                resource_id = int(request_id)
            elif station_id is not None:
                action_type = ActionType.CHARGE
                action_id = f"charge_{station_id}:continue"
                station = station_map.get(station_id)
                target = int(
                    station.location if station is not None else vehicle.location
                )
                post_location = target
                travel_duration = (
                    0.0
                    if vehicle.charging_station is not None
                    else _travel_time(env, vehicle.location, target)
                )
                post_duration = max(
                    1.0, travel_duration + vehicle.charging_time_left
                )
                post_distance = _distance(env, vehicle.location, target)
                # An in-progress session does not consume another admission
                # slot, so it has a typed, separate capacity resource.
                resource_type = "station_session"
                resource_id = int(station_id)
                resource_capacity = max(
                    1,
                    sum(
                        other.charging_station == station_id
                        or other.charging_target == station_id
                        for other in state.vehicles
                    ),
                )
                value_function = _value_function_for_vehicle(
                    env, vehicle.vehicle_type
                )
                queue_builder = getattr(value_function, "_queue_features", None)
                if callable(queue_builder):
                    try:
                        queue_features = tuple(
                            float(value)
                            for value in queue_builder(
                                station_id=station_id,
                                target_location=target,
                                vehicle_id=vehicle.vehicle_id,
                                vehicle_location=vehicle.location,
                                current_time=state.current_time,
                                num_requests=float(len(state.requests)),
                                travel_duration=travel_duration,
                            )
                        )
                    except (TypeError, ValueError, RuntimeError):
                        queue_features = ()
            elif vehicle.relocation_target is not None:
                action_type = ActionType.RELOCATE
                target = int(vehicle.relocation_target)
                post_location = target
                action_id = f"reloc_{target}:continue"
                post_duration = max(1.0, vehicle.remaining_relocation_time)
                post_distance = _distance(env, vehicle.location, target)
            else:
                action_type = ActionType.WAIT
                action_id = "continue_wait"
                post_duration = max(1.0, vehicle.stationary_duration_left)
            distance = _distance(env, vehicle.location, target)
            value_function = _value_function_for_vehicle(
                env, vehicle.vehicle_type
            )
            score_builder = getattr(value_function, "_myopic_score", None)
            if callable(score_builder):
                try:
                    structured_score = float(
                        score_builder(
                            int(action_type),
                            request_value,
                            distance,
                            post_distance,
                        )
                    )
                except (TypeError, ValueError, RuntimeError):
                    structured_score = 0.0
            else:
                structured_score = 0.0
            if action_type == ActionType.CHARGE:
                structured_score = -abs(
                    float(getattr(env, "charging_penalty", 0.0) or 0.0)
                )
            elif action_type == ActionType.WAIT:
                wait_penalty = getattr(env, "idle_penalty", None)
                structured_score = (
                    -abs(float(wait_penalty))
                    if wait_penalty is not None
                    else float(getattr(env, "movingpenalty", 0.0) or 0.0)
                )
            edges.append(
                FeasibleEdgeSnapshot(
                    edge_id=(
                        f"{stage_id}:{vehicle.vehicle_id}:continue:{action_id}"
                    ),
                    vehicle_id=vehicle.vehicle_id,
                    vehicle_type=vehicle.vehicle_type,
                    action_type=action_type,
                    action_id=action_id,
                    target_location=int(target),
                    post_action_location=int(post_location),
                    request_id=request_id,
                    station_id=station_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    resource_capacity=resource_capacity,
                    structured_score=structured_score,
                    collection_score=structured_score,
                    request_value=request_value,
                    acceptance_probability=0.0,
                    target_distance=float(distance),
                    post_action_distance=float(post_distance),
                    post_action_duration=float(post_duration),
                    target_zoneid=_zone_for_location(
                        env, int(target), list(state.zone_ids)
                    ),
                    post_action_zoneid=_zone_for_location(
                        env, int(post_location), list(state.zone_ids)
                    ),
                    queue_features=queue_features,
                    metadata=tuple(sorted(metadata.items())),
                )
            )


def _distance(env: Any, origin: int, destination: int) -> float:
    for name in ("get_distance_km", "_manhattan_distance_loc"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                return float(fn(int(origin), int(destination)))
            except Exception:
                pass
    grid_size = max(1, int(getattr(env, "grid_size", 1)))
    ox, oy = int(origin) % grid_size, int(origin) // grid_size
    dx, dy = int(destination) % grid_size, int(destination) // grid_size
    return float(abs(ox - dx) + abs(oy - dy))


def _zone_for_location(
    env: Any,
    location: int,
    zone_candidates: list[int],
) -> int:
    location = int(location)
    if location in zone_candidates:
        return location
    for name in ("get_zone_id", "get_distribution_zone_index"):
        fn = getattr(env, name, None)
        if not callable(fn):
            continue
        try:
            zone = int(fn(location))
        except (TypeError, ValueError, RuntimeError):
            continue
        if zone in zone_candidates:
            return zone
        if 0 <= zone < len(zone_candidates):
            return int(zone_candidates[zone])
    return location


def _value_function_for_vehicle(env: Any, vehicle_type: int) -> Any | None:
    if int(vehicle_type) == 1:
        return getattr(env, "value_function_ev", None)
    return getattr(env, "value_function", None)


def _travel_time(env: Any, origin: int, destination: int) -> float:
    fn = getattr(env, "get_travel_time", None)
    if callable(fn):
        try:
            return float(fn(int(origin), int(destination)))
        except Exception:
            pass
    return _distance(env, origin, destination)
