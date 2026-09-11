"""Equivalent action masks/Q values and shared graph-encoder gradients."""
from dataclasses import replace
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.NYCEnvironment import NYCEnvironment
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction
from test_recourse_must_fix import _graph


def test_visible_station_list_does_not_sort_existing_public_ids():
    from types import SimpleNamespace
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.vehicles = {1: {'type': 1}, 2: {'type': 2}}
    env.public_charging_station_ids = [9, 3]
    env.aev_charging_station_ids = [100]
    env.charging_manager = SimpleNamespace(stations={3: None, 9: None, 100: None})
    with patch('builtins.sorted', side_effect=AssertionError('unnecessary sorting')):
        assert env._charging_station_ids_for_vehicle(1) == [9, 3]
        assert env._charging_station_ids_for_vehicle(2) == [100]
        assert env._charging_station_ids_for_vehicle_record({'type': 1}) == [9, 3]
    env.public_charging_station_ids.append(7)
    assert env._charging_station_ids_for_vehicle(1) == [9, 3, 7]


@pytest.mark.parametrize('method', ['_selected_raw_tensors', '_selected_correction_tensors'])
def test_shared_selected_graph_matches_values_and_all_parameter_gradients(method):
    torch.manual_seed(17)
    vf = PyTorchChargingValueFunction(grid_size=2, num_vehicles=8, device='cpu',
                                     episode_length=10, max_requests=10, neighbour_number=0)
    vf.training_step = 1000
    reference = deepcopy(vf)
    graph = _graph('eight', stage=2, vehicle_id=1, vehicle_type=2)
    aev = graph.state.vehicles[1]
    graph = replace(graph, state=replace(graph.state, vehicles=tuple(
        replace(aev, vehicle_id=i) for i in range(1, 9))), edges=tuple(
        replace(graph.edges[0], edge_id=f'w:{i}', vehicle_id=i) for i in range(1, 9)))
    ids = tuple(edge.edge_id for edge in graph.edges)
    counts = []
    hook = vf.graph_encoder.register_forward_hook(lambda *args: counts.append(1))
    rows1, rows2 = [], []
    edge_method = '_edge_raw_tensors' if method == '_selected_raw_tensors' else '_edge_correction_tensors'
    for edge in graph.edges:
        reference._graph_cache_key = reference._graph_cache = None
        one, two, _ = getattr(reference, edge_method)(graph, edge, target_context=False)
        rows1.append(one.reshape(())); rows2.append(two.reshape(()))
    expected1, expected2 = torch.stack(rows1).sum(), torch.stack(rows2).sum()
    actual1, actual2, _ = getattr(vf, method)(graph, ids)
    torch.testing.assert_close(actual1, expected1)
    torch.testing.assert_close(actual2, expected2)
    (expected1 + expected2).backward()
    (actual1 + actual2).backward()
    for name in ('graph_encoder', 'mixer', 'network', 'critic2'):
        for observed, expected in zip(getattr(vf, name).parameters(), getattr(reference, name).parameters()):
            assert (observed.grad is None) == (expected.grad is None)
            if observed.grad is not None:
                torch.testing.assert_close(observed.grad, expected.grad, atol=2e-5, rtol=2e-5)
    assert len(counts) == 1
    assert vf._graph_cache is None
    # A subsequent prediction must use a fresh differentiable graph.
    one, two, _ = getattr(vf, method)(graph, ids)
    (one + two).backward()
    assert len(counts) == 2
    hook.remove()


@pytest.mark.parametrize('conservative', [False, True])
def test_nyc_prepared_q_matrix_and_ev_shortcut_are_equivalent(conservative):
    from run_recourse_audit import build_pair
    from train_acceptance_model import make_environment, parse_args
    env = make_environment(parse_args(['--environment', 'nyc', '--num-vehicles', '20',
                                      '--num-ev', '10', '--aev-charging-center-count', '3',
                                      '--stop-hour', '8.05']), 901)
    env.evaluatemode, env.adp_value = False, 1.
    env.conservative_charging = conservative
    env.configure_recourse_experiment('r3')
    env.state_variant = 'joint_state_separate_critics'
    env.learner_variant = 'optimization_anchored_residual'
    build_pair(env)
    from src.Request import Request
    pickup = env.vehicles[0]['location']
    env.active_requests = {9000: Request(9000, pickup, pickup, 0., 1., value=25., final_value=25.)}
    for onlyev in (True, False):
        ids = [vid for vid in env.vehicles if env._is_ev(vid) == onlyev]
        prepared = env.generate_whole_matrix(ids, onlyev=onlyev)
        if onlyev:
            assert prepared[2:] == (0, 0)
            request_mask = env.generate_vehicle_requests(ids)
            np.testing.assert_array_equal(prepared[0], np.hstack((request_mask, np.ones((len(ids), 1)))))
            with (patch.object(env, 'generate_vehicle_zone', side_effect=AssertionError('unused EV relocation')),
                  patch.object(env, 'generate_vehicle_chargerange', side_effect=AssertionError('unused EV charge'))):
                np.testing.assert_array_equal(env.generate_whole_matrix(ids)[0], prepared[0])
        reference = env.generate_vehicle_qvalue(ids, onlyev=onlyev)
        prepared = env.generate_whole_matrix(ids, onlyev=onlyev)
        with patch.object(env, 'generate_whole_matrix', side_effect=AssertionError('duplicate matrix')):
            actual = env.generate_vehicle_qvalue(ids, onlyev=onlyev, prepared_action_matrix=prepared)
        np.testing.assert_array_equal(actual, reference)
