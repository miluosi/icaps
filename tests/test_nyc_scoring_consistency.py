"""Compare the actual NYC batch input path against its stored replay graph."""
from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest
import torch

from test_nyc_ev_charge_execution import env
from test_recourse_must_fix import _request
from src.ValueFunction_optimization_anchored_residual import PyTorchChargingValueFunction
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.recourse.target_builder import RecourseTargetBuilder


@pytest.mark.parametrize('residual_bias', [None, 0., 1000.])
@pytest.mark.parametrize('response_anchor', [False, True])
def test_nyc_live_scores_match_replay_all_actions(env, monkeypatch, residual_bias, response_anchor):
    torch.manual_seed(71)
    env.current_time = 3
    env.vehicles[1]['type'] = 2
    for v in env.vehicles.values():
        v['battery'] = .5
    zones = list(env.manhattan_zone_ids)
    env.active_requests = {
        10: _request(10, pickup=zones[1], dropoff=zones[2], value=100.),
        11: _request(11, pickup=zones[2], dropoff=zones[3], value=99.7),
    }
    env.active_requests[10].travel_time = 27.5  # distinct from centroid travel
    env.vehicles[2] = deepcopy(env.vehicles[0])
    env.vehicles[2]['charging_station'] = env.test_station_id
    env.vehicles[2]['charging_time_left'] = 4
    env._last_matrix_request_ids = [10, 11]
    env._last_matrix_charge_station_ids = [env.test_station_id]
    env._last_matrix_zone_target_ids = [zones[4]]
    env._last_matrix_zone_indices = [0]
    env._last_expected_charge_expansion = None
    env.decision_mode = 'integrated'
    env._active_stage_state_snapshot = None
    env._ev_default_relocation_targets = {0: zones[5]}
    monkeypatch.setattr(env, '_sample_ev_default_relocation_target', lambda vid: zones[5])
    # Snapshot/live response probability is controlled identically, no fitted model required.
    monkeypatch.setattr('src.recourse.state_snapshot.predicted_rejection', lambda *a, **k: .3)
    monkeypatch.setattr('src.recourse.state_snapshot.offer_context', lambda *a, **k: None)
    env.ev_acceptance_feature = 'off'
    vf = PyTorchChargingValueFunction(env=env, num_vehicles=2, grid_size=20,
                                     neighbour_number=0, device='cpu')
    env.value_function = env.value_function_ev = vf
    vf.training_step = 10000
    vf.response_anchor_enabled = response_anchor
    if response_anchor:
        env.ev_acceptance_feature = 'predicted'
    vf.rejection_for_live_edges = lambda ids, acts, *a: np.asarray([
        .3 if env.vehicles[int(vid)]['type'] == 1 and act == 2 and response_anchor else 0.
        for vid, act in zip(ids, acts)])
    vf.response_masks_for_live_edges = lambda ids, acts: np.asarray([
        env.vehicles[int(vid)]['type'] == 1 and act == 2 and response_anchor
        for vid, act in zip(ids, acts)])
    vf.queue_predictor_trained = True
    for module in (vf.queue_predictor, vf.target_queue_predictor):
        for param in module.parameters():
            param.data.zero_()
        list(module.parameters())[-1].data.fill_(10000.)
    if residual_bias is not None:
        for critic in (vf.network, vf.critic2):
            critic.base.net[-1].weight.data.zero_()
            critic.base.net[-1].bias.data.fill_(residual_bias)
    mask = np.ones((2, 5), dtype=np.float32)
    mask[0, 2:4] = 0
    # Store actual neural input rows, then compare before relying on scores.
    seen = []
    hook = vf.network.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    live = env.generate_vehicle_qvalue([0, 1], prepared_action_matrix=(mask, 2, 1, 1))
    hook.remove()
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env, [0, 1], mask, live, np.full_like(mask, 999.), num_requests=2,
        num_stations=1, num_zones=1, stage_id=0, solver_backend='ortools')
    # NYC forwards AEV, then EV; in each call service/charge/reloc/wait order.
    for fleet, captured in zip((2, 1), seen):
        edges = [edge for edge in graph.edges if edge.vehicle_type == fleet
                 and not dict(edge.metadata).get('continuing', False)]
        features, _ = vf._graph_edge_tensor_batch(graph, edges, target_context=False)
        torch.testing.assert_close(features, captured, rtol=2e-5, atol=2e-5)
        for edge in edges:
            kind = 2 if edge.request_id is not None else 3 if edge.station_id is not None else 1
            expected = vf.assignment_anchor(kind, edge.request_value, edge.target_distance,
                edge.post_action_distance, edge.rejection_probability, edge.human_response_mask)[0]
            assert edge.structured_score == expected
    scores, _ = vf._graph_edge_scores(graph, target_context=False)
    for edge in graph.edges:
        if dict(edge.metadata).get('continuing', False):
            continue
        row, column = map(int, edge.edge_id.split(':')[1:3])
        assert scores[edge.edge_id] == pytest.approx(float(live[row, column]), abs=2e-3)
    selected = RecourseTargetBuilder().project(graph, scores)
    collected = RecourseTargetBuilder().project(graph, {e.edge_id: e.collection_score for e in graph.edges})
    assert sum(scores[e] for e in selected) == pytest.approx(sum(scores[e] for e in collected), abs=2e-3)
    # R2 must keep the actual environment structured-only matrix, even if the
    # shared value-function object's mode is restored before replay scoring.
    env._structured_only_planning = True
    matrix = np.full_like(mask, 17.)
    r2_graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env, [0, 1], mask, matrix, matrix, num_requests=2, num_stations=1,
        num_zones=1, stage_id=0, solver_backend='ortools')
    r2_scores, _ = vf._graph_edge_scores(r2_graph, target_context=False)
    assert all(r2_scores[e.edge_id] == 17. for e in r2_graph.edges
               if not dict(e.metadata).get('continuing', False))
    env._structured_only_planning = False
    before = deepcopy(scores)
    env.current_time += 100
    env.active_requests.clear()
    for v in env.vehicles.values():
        v.update(battery=.01, location=zones[-1], is_online=False)
    assert vf._graph_edge_scores(graph, target_context=False)[0] == before


def test_continuing_edges_do_not_change_candidate_clipping():
    from test_recourse_must_fix import _vf, _graph
    vf = _vf()
    vf.training_step = 10000
    for critic in (vf.network, vf.critic2):
        critic.net[-1].bias.data.fill_(1000.)
    g = _graph('clip', stage=2, vehicle_id=1, vehicle_type=2)
    baseline = vf._graph_edge_scores(g, target_context=False)[0]
    fixed = replace(g.edges[0], edge_id='continuing', structured_score=100000., metadata=(('continuing', True),))
    extended = replace(g, edges=(*g.edges, fixed))
    assert vf._graph_edge_scores(extended, target_context=False)[0][g.edges[0].edge_id] == baseline[g.edges[0].edge_id]


@pytest.mark.parametrize('stage', [1, 2])
def test_no_available_vehicles_is_not_an_empty_paper_stage(env, stage):
    env.decision_mode = 'ev_first'
    env.vehicles[1]['type'] = 2
    for vehicle in env.vehicles.values():
        vehicle.update(charging_station=env.test_station_id, charging_time_left=5)
    env._last_matrix_request_ids = []
    env._last_matrix_charge_station_ids = []
    env._last_matrix_zone_target_ids = []
    env.value_function = env.value_function_ev = None
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env, [], np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1)),
        num_requests=0, num_stations=0, num_zones=0, stage_id=stage, solver_backend='ortools')
    assert len(graph.edges) == 1
    assert all(dict(edge.metadata)['continuing'] for edge in graph.edges)
    assert graph.edges[0].vehicle_type == stage
