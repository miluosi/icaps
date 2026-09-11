import importlib
from types import MethodType

import numpy as np
import pytest
import torch

from src.qvalue_inference import bounded_edge_inference, inference_batch_size
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction as Base


def mixed_edges(n=29):
    i = np.arange(n)
    return dict(
        vehicle_ids=i % 4, vehicle_locations=i % 4,
        target_locations=(i + 1) % 4, current_times=np.full(n, 2.0),
        other_vehicles=np.full(n, 4), num_requests=np.full(n, 20),
        battery_levels=np.full(n, .6), request_values=np.linspace(0, 150, n),
        target_distances=np.ones(n), target_zoneids=(i + 1) % 4,
        vehicle_idle_times=np.zeros(n), action_type_ids=(i % 3) + 1,
        post_action_distances=np.full(n, 2.0), post_action_durations=np.full(n, 3.0),
        post_action_locations=(i + 2) % 4, post_action_zoneids=(i + 2) % 4,
        target_station_ids=np.full(n, -1),
        vehicle_neighbour_candidates={vid: [dict(node_type='zone', node_id=(vid+1)%4,
            target_location=(vid+1)%4, distance=1.0)] for vid in range(4)},
    )


def legacy_assemble(self, rows, vehicle_types, graph, sources, vids, locations):
    # Scalar reference: the original per-edge tensor transfer/concatenation.
    edges = [torch.cat((torch.tensor(row, dtype=torch.float32, device=self.device),
                       sources[int(vid)], self._graph_embedding_for_location(graph, loc)))
             for row, vid, loc in zip(rows, vids, locations)]
    weights = [graph['w_ev'] if typ == 1 else graph['w_aev'] for typ in vehicle_types]
    return torch.stack(edges), torch.stack(weights).unsqueeze(1), graph['baseline']


@pytest.mark.parametrize('module', [
    'st_masac_gat', 'st_masac_gat_post_demand',
    'st_masac_gat_post_demand_direct', 'optimization_anchored_residual',
    'integrated_directq',
])
def test_chunking_preserves_full_batch_scores_and_subclass_semantics(module):
    cls = importlib.import_module('src.ValueFunction_' + module).PyTorchChargingValueFunction
    torch.manual_seed(14)
    net = cls(grid_size=2, num_vehicles=4, neighbour_number=2, device='cpu')
    net.eta_pi = .2
    net.training_step = 10000
    # Activate the residual and force clipping, so per-chunk std(g) would fail.
    for critic in (net.network, net.critic2):
        getattr(critic, 'base', critic).net[-1].bias.data.fill_(1000.)
    if hasattr(net, 'post_demand_predictor_trained'):
        net.post_demand_predictor_trained = True
    inputs = mixed_edges()
    net.qvalue_inference_batch_size = 1000
    net._assemble_edge_rows = MethodType(legacy_assemble, net)
    reference = net.batch_get_mixed_q_values(**inputs)
    del net._assemble_edge_rows
    net.qvalue_inference_batch_size = 5
    actual = net.batch_get_mixed_q_values(**inputs)
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-5)
    assert net._last_qvalue_inference_stats['max_batch_edges'] == 5
    assert net._last_qvalue_inference_stats['batches'] == 6


def test_bulk_gather_preserves_gradients_with_repeated_vehicles_and_destinations():
    net = Base(grid_size=2, num_vehicles=2, neighbour_number=0)
    rows = np.arange(12).reshape(3, 4).astype(np.float32)
    gradients = []
    for assembly in [legacy_assemble, Base._assemble_edge_rows]:
        embeddings = torch.randn(4, 3, requires_grad=True)
        wev = torch.tensor(2., requires_grad=True)
        waev = torch.tensor(3., requires_grad=True)
        graph = dict(embeddings=embeddings, zone_to_row={i:i for i in range(4)},
                     global_row=0, w_ev=wev, w_aev=waev, baseline=torch.tensor(0.))
        output, weights, _ = assembly(net, rows, [1, 2, 1], graph,
                                     {0:embeddings[0], 1:embeddings[1]}, [0, 1, 0], [2, 3, 2])
        (output.sum() + weights.sum()).backward()
        gradients.append((embeddings.grad, wev.grad, waev.grad))
    for actual, expected in zip(*gradients):
        torch.testing.assert_close(actual, expected)


def test_oom_halves_batch_retries_same_edges_and_never_substitutes_scores(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    attempts = []
    def evaluate(part):
        attempts.append((part.start, part.stop))
        if part.stop - part.start > 2:
            raise torch.cuda.OutOfMemoryError('simulated device allocation failure')
        return np.arange(part.start, part.stop) + 3
    output, stats = bounded_edge_inference(7, 7, evaluate, device='cuda')
    np.testing.assert_array_equal(output, np.arange(7) + 3)
    assert stats['oom_retries'] == 2
    assert attempts[:3] == [(0, 7), (0, 3), (0, 1)]
    with pytest.raises(ValueError, match='real bug'):
        bounded_edge_inference(2, 2, lambda _: (_ for _ in ()).throw(ValueError('real bug')), device='cuda')
    with pytest.raises(torch.cuda.OutOfMemoryError):
        bounded_edge_inference(1, 1, lambda _: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError()), device='cuda')


def test_batch_size_environment_override(monkeypatch):
    monkeypatch.setenv('ICAPS_QVALUE_BATCH_SIZE', '512')
    assert inference_batch_size() == 512
    assert inference_batch_size(16) == 16
    with pytest.raises(ValueError):
        inference_batch_size(0)


def test_trained_queue_predictor_is_also_bounded():
    net = Base(grid_size=2, num_vehicles=4, neighbour_number=0,
               qvalue_inference_batch_size=3)
    net.queue_predictor_trained = True
    net._queue_features = lambda **kwargs: [0.] * net.queue_feature_dim
    waits = net.predict_queue_waits(vehicle_ids=np.arange(11))
    assert waits.shape == (11,)
    assert net._last_queue_inference_stats['max_batch_edges'] == 3


def test_oom_releases_intermediate_tensors_before_cache_cleanup(monkeypatch):
    import weakref
    references = []
    def evaluate(part):
        tensor = torch.ones(2)
        if part.stop - part.start > 1:
            references.append(weakref.ref(tensor))
            raise torch.cuda.OutOfMemoryError('simulated')
        return [9.]
    def check_released():
        assert references[-1]() is None
    monkeypatch.setattr(torch.cuda, 'empty_cache', check_released)
    values, _ = bounded_edge_inference(2, 2, evaluate, device='cuda')
    np.testing.assert_array_equal(values, [9., 9.])
