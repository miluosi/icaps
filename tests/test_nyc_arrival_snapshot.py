from dataclasses import replace

import numpy as np
import pytest

from benchmark_conservative_charging import make_case, solve_adapter
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.recourse.target_builder import RecourseTargetBuilder
from src.recourse.types import ActionType


def snapshot_case(*, same_arrival=False, committed=False):
    env, ids, scores, _ = make_case(8, 1, 4, 'synchronized', 0)
    env.active_requests = {}
    env.value_function = env.value_function_ev = None
    env.get_travel_time = lambda origin, destination: (1 if same_arrival or origin <= 4 else 5)
    if committed:
        env.charging_manager.stations[100].current_vehicles = ['99']
        env.vehicles[99] = dict(type=2, location=0, battery=.5, charging_time_left=2,
                               charging_station=100)
    mask = np.column_stack((env.generate_vehicle_chargerange(ids), np.ones(len(ids))))
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env, ids, mask, scores, scores, num_requests=0, num_stations=1,
        num_zones=0, stage_id=2, solver_backend='ortools',
    )
    return env, ids, mask, scores, graph


def test_eight_arrivals_at_two_epochs_are_not_four_slot_static_overflow():
    env, ids, mask, scores, graph = snapshot_case()
    assignments, _ = solve_adapter(env, ids, mask, scores, True)
    assert sum(action == 'charge_100' for action in assignments.values()) == 8
    selected = StateSnapshotBuilder.selected_edge_ids(graph, assignments)
    RecourseTargetBuilder.verify_feasible(graph, selected)
    projected = RecourseTargetBuilder().project(graph)
    RecourseTargetBuilder.verify_feasible(graph, projected)
    assert sum(edge.edge_id in projected and edge.action_type == ActionType.CHARGE
               for edge in graph.edges) == 8
    # Reproduce the old failure without suppressing the verifier.
    old_graph = replace(graph, edges=tuple(
        replace(edge, resource_type='station') if edge.action_type == ActionType.CHARGE else edge
        for edge in graph.edges))
    with pytest.raises(AssertionError, match='count=8, capacity=4'):
        RecourseTargetBuilder.verify_feasible(old_graph, selected)


def test_same_arrival_still_enforces_four_slot_limit():
    env, ids, mask, scores, graph = snapshot_case(same_arrival=True)
    assignments, _ = solve_adapter(env, ids, mask, scores, True)
    assert sum(action == 'charge_100' for action in assignments.values()) == 4
    selected = StateSnapshotBuilder.selected_edge_ids(graph, assignments)
    RecourseTargetBuilder.verify_feasible(graph, selected)
    all_charge = tuple(edge.edge_id for edge in graph.edges if edge.action_type == ActionType.CHARGE)
    with pytest.raises(AssertionError, match='count=8, capacity=4'):
        RecourseTargetBuilder.verify_feasible(graph, all_charge)
    projected = RecourseTargetBuilder().project(graph)
    assert sum(edge.edge_id in projected and edge.action_type == ActionType.CHARGE
               for edge in graph.edges) == 4


def test_arrival_capacity_is_frozen_for_replay_and_includes_completed_background():
    env, ids, mask, scores, graph = snapshot_case(committed=True)
    charges = [edge for edge in graph.edges if edge.action_type == ActionType.CHARGE
               and edge.vehicle_id in ids]
    assert {edge.resource_capacity for edge in charges if edge.resource_type == 'station_arrival:1'} == {3}
    assert {edge.resource_capacity for edge in charges if edge.resource_type == 'station_arrival:5'} == {4}
    projected = RecourseTargetBuilder().project(graph)
    env.charging_manager.stations[100].max_capacity = 0
    env._last_expected_charge_expansion['station_schedules'].clear()
    assert RecourseTargetBuilder().project(graph) == projected
    RecourseTargetBuilder.verify_feasible(graph, projected)
