"""Builders for immutable simulator and feasible-graph snapshots."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping
import uuid

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
        vehicles = tuple(
            VehicleSnapshot.from_vehicle(vehicle_id, vehicle)
            for vehicle_id, vehicle in sorted(getattr(env, "vehicles", {}).items())
        )
        requests = tuple(
            RequestSnapshot.from_request(request)
            for _, request in sorted(
                getattr(env, "active_requests", {}).items(),
                key=lambda item: int(item[0]),
            )
        )
        station_rows = []
        stations = getattr(getattr(env, "charging_manager", None), "stations", {})
        for station_id, station in sorted(stations.items()):
            station_rows.append(
                StationSnapshot(
                    station_id=int(station_id),
                    location=int(getattr(station, "location", 0)),
                    capacity=int(getattr(station, "max_capacity", 0) or 0),
                    occupied=len(getattr(station, "current_vehicles", ()) or ()),
                    inbound=len(
                        getattr(station, "charging_queue_notarrived", ()) or ()
                    ),
                    queued=len(getattr(station, "charging_queue", ()) or ()),
                )
            )
        zones = list(
            getattr(env, "aux_zone_ids", ())
            or getattr(env, "relocation_target_ids", ())
            or sorted(getattr(env, "zone_coords", {}).keys())
        )
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
            requests=requests,
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
        zone_ids = list(getattr(env, "_last_matrix_zone_target_ids", ()))[:num_zones]
        vehicle_map = {vehicle.vehicle_id: vehicle for vehicle in state.vehicles}
        edges: list[FeasibleEdgeSnapshot] = []

        for row, vehicle_id in enumerate(vehicle_ids):
            vehicle = vehicle_map[int(vehicle_id)]
            feasible_columns = np.flatnonzero(np.asarray(action_matrix[row]) > 0)
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
                    post_duration = request.travel_time
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
                    post_duration = float(getattr(env, "charge_duration", 1.0))
                    resource_type = "station"
                    resource_id = station_id
                    resource_capacity = max(
                        0,
                        station.capacity - station.occupied - station.inbound,
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
                                travel_duration=post_duration,
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
                    )
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
                zone_demand = {
                    zone_id: sum(
                        int(request.pickup == zone_id) for request in state.requests
                    )
                    for zone_id in {
                        edges[index].post_action_location for index in indices
                    }
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
                        float(zone_demand[edges[index].post_action_location])
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
        return FeasibleGraphSnapshot(
            graph_id=str(uuid.uuid4()),
            epoch_id=state.epoch_id,
            stage_id=int(stage_id),
            solver_backend=str(solver_backend),
            state=state,
            edges=tuple(edges),
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
            text = str(target)
            if isinstance(target, str) and target.startswith("charge_"):
                try:
                    target_station_id = int(target.split("_", 1)[1])
                except ValueError:
                    pass
            if isinstance(target, str) and target.startswith("idle_at_"):
                try:
                    target_location = int(target.split("_", 2)[2])
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
                    target_location is not None
                    and edge.action_type == ActionType.RELOCATE
                    and edge.target_location == target_location
                ):
                    match = edge
                    break
                if target is None and edge.action_type == ActionType.WAIT:
                    match = edge
                    break
                if isinstance(target, str) and target in {"waiting", "reloc"} and edge.action_type == ActionType.WAIT:
                    match = edge
                    break
            if match is None and candidates:
                # The execution code sometimes turns a generic relocation token
                # into a concrete target after solving.  Its outside-option edge
                # is still the correct selected graph edge.
                match = next(
                    (edge for edge in candidates if edge.action_type == ActionType.WAIT),
                    None,
                )
            if match is not None:
                selected.append(match.edge_id)
        return tuple(selected)


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
