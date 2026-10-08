"""Live conservative admission and Bellman graphs must use identical resources."""
from dataclasses import replace
from itertools import product
import pickle

import numpy as np
import pytest

from benchmark_conservative_charging import make_case, solve_adapter
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.recourse.target_builder import RecourseTargetBuilder
from src.recourse.types import ActionType


def case(same_arrival=False):
    env, ids, scores, _ = make_case(3, 1, 2, 'synchronized', 0)
    env.conservative_charging = True
    env.active_requests = {}
    env.value_function = env.value_function_ev = None
    env.get_travel_time = lambda origin, destination: 3 if same_arrival or origin <= 2 else 10
    for vehicle in env.vehicles.values():
        vehicle['battery'] = .7
    station = env.charging_manager.stations[100]
    station.current_vehicles = ['99']
    station.available_slots = 1
    env.vehicles[99] = dict(type=2, location=0, battery=.5, charging_time_left=2,
                            charging_station=100)
    mask = np.column_stack((env.generate_vehicle_chargerange(ids), np.ones(len(ids))))
    return env, ids, mask, scores


def snapshot(env, ids, mask, scores):
    return StateSnapshotBuilder.feasible_graph_from_matrix(
        env, ids, mask, scores, scores, num_requests=0, num_stations=1,
        num_zones=0, stage_id=2, solver_backend='ortools')


@pytest.mark.parametrize('ssg', [False, True])
@pytest.mark.parametrize('same_arrival,expected', [(False, 3), (True, 2)])
def test_live_and_replay_optimum_agree_after_committed_vehicle_releases(ssg, same_arrival, expected):
    env, ids, mask, scores = case(same_arrival)
    assignments, stats = solve_adapter(env, ids, mask, scores, ssg)
    assert stats['cut_iterations'] == 0
    assert sum(a == 'charge_100' for a in assignments.values()) == expected
    graph = snapshot(env, ids, mask, scores)
    assert not graph.allow_charging_queue
    selected = StateSnapshotBuilder.selected_edge_ids(graph, assignments)
    RecourseTargetBuilder.verify_feasible(graph, selected)
    if not same_arrival:
        old_graph = replace(graph, edges=tuple(
            replace(e, resource_type='station', resource_capacity=2)
            if e.resource_type == 'station_admitted' else e for e in graph.edges))
        with pytest.raises(AssertionError, match='count=3, capacity=2'):
            RecourseTargetBuilder.verify_feasible(old_graph, selected)
    # Every subset of certified edges is feasible, even when their total
    # number exceeds physical plugs because the intervals reuse those plugs.
    candidates = [[e for e in graph.edges if e.vehicle_id == vid] for vid in ids]
    oracle = max(sum(e.collection_score for e in choice) for choice in product(*candidates))
    assert stats['assignment_score'] == pytest.approx(oracle)
    projected = RecourseTargetBuilder().project(graph)
    assert sum(e.collection_score for e in graph.edges if e.edge_id in projected) == pytest.approx(oracle)
    charges = [e for e in graph.edges if e.edge_id in projected and e.vehicle_id in ids
               and e.action_type == ActionType.CHARGE]
    assert len(charges) == expected
    assert all('conservative_end_offset' in dict(e.metadata) for e in charges)
    # Disk replay and later calendar changes must not alter this feasible set.
    frozen = pickle.loads(pickle.dumps(graph))
    env._last_conservative_charge.clear()
    env._last_expected_charge_expansion.clear()
    env.charging_manager.stations[100].max_capacity = 0
    assert RecourseTargetBuilder().project(frozen) == projected


@pytest.mark.parametrize('corruption', ['stale', 'missing', 'overlap', 'window'])
def test_invalid_admission_fails_closed(corruption):
    env, ids, mask, scores = case()
    if corruption == 'stale':
        env.current_time += 1
    elif corruption == 'missing':
        env._last_conservative_charge['virtual_windows'].pop((ids[0], 100))
    elif corruption == 'overlap':
        env._last_expected_charge_expansion['station_schedules'][100]['capacity'] = 1
    else:
        env._last_conservative_charge['virtual_windows'][ids[0], 100]['end_offset'] += 1
    with pytest.raises(ValueError, match='[Cc]onservative'):
        snapshot(env, ids, mask, scores)
