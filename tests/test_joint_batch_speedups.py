"""Exact lifecycle accounting and batched replay-feature/gradient regression."""
from copy import deepcopy
from dataclasses import replace
import random
import pytest
import torch
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction
from src.ValueFunction_st_masac_gat_post_demand_direct import PyTorchChargingValueFunction as PostDemandValueFunction
from src.recourse.lifecycle import RequestLifecycleTracker, _MutableLifecycle
from src.recourse.types import ActionType, STATE_VARIANTS
from test_recourse_must_fix import _graph


def test_history_summary_preserves_first_rejection_and_filtered_order():
    tracker = RequestLifecycleTracker()
    keys = [(rid, epoch) for rid in range(40) for epoch in (0, 3, 9)]
    random.Random(41).shuffle(keys)
    for rid, epoch in keys:
        tracker._rejection_events[rid, epoch] = _MutableLifecycle(
            request_id=rid, rejected_epoch_id=epoch, residual_category='rejected',
            completed=epoch == 9, eligible=bool(rid % 2))
    tracker._requests[100] = _MutableLifecycle(request_id=100, residual_epoch_id=3,
                                               residual_category='unoffered')
    all_events = tracker.outcome_summary().events
    assert [(e.request_id, e.epoch_id) for e in all_events] == sorted(keys + [(100, 3)])
    assert all(e.first_rejected_epoch == 0 for e in all_events if e.request_id < 40)
    assert all_events[-1].first_rejected_epoch is None
    for epoch in (0, 3, 9, 15):
        assert tracker.outcome_summary(epoch_id=epoch).events == tuple(
            e for e in all_events if e.epoch_id == epoch)
    tracker._rejection_events[1, -1] = _MutableLifecycle(request_id=1, rejected_epoch_id=-1)
    assert next(e for e in tracker.outcome_summary(epoch_id=3).events
                if e.request_id == 1).first_rejected_epoch == -1


@pytest.mark.parametrize('demand_mode', ['none', 'mixed'])
@pytest.mark.parametrize('value_class', [PyTorchChargingValueFunction, PostDemandValueFunction])
@pytest.mark.parametrize('variant', STATE_VARIANTS)
@pytest.mark.parametrize('target', [False, True])
def test_batch_features_scores_and_gradients_match_scalar_reference(variant, target, value_class, demand_mode):
    torch.manual_seed(32)
    vf = value_class(grid_size=3, num_vehicles=2, device='cpu',
                                     episode_length=10, max_requests=10, neighbour_number=0)
    vf.state_variant = variant
    reference = deepcopy(vf)
    g = _graph('mixed', stage=0, vehicle_id=1, vehicle_type=2)
    edges = tuple(replace(g.edges[0], edge_id=f'e{i}', vehicle_id=i % 2, vehicle_type=i % 2 + 1,
                         action_type=ActionType(i % 4), action_id=('wait','reloc','service','charge')[i % 4],
                         target_location=2, post_action_location=2, post_action_duration=float(i + 1),
                         target_distance=.5, post_action_distance=1.,
                         post_demand_feature=.7 if demand_mode == 'mixed' and i % 3 else None,
                         request_value=20., rejection_probability=.2 if i % 4 == 2 else 0., human_response_mask=float(i % 4 == 2))
                  for i in range(12))
    g = replace(g, edges=edges)
    expected, expected_weights = [], []
    for edge in edges:
        exp = reference._edge_experience(g, edge, state_variant=variant)
        row, weight, _ = reference._edge_tensor_from_experience(exp, target_context=target,
                                                               state_snapshot=exp['state_snapshot'])
        expected.append(row); expected_weights.append(weight)
    actual, weights = vf._graph_edge_tensor_batch(g, edges, target_context=target)
    expected = torch.cat(expected); expected_weights = torch.cat(expected_weights)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights, expected_weights)
    if not target:
        a = (vf.network(actual) * weights).sum() + (vf.critic2(actual) * weights).sum()
        b = (reference.network(expected) * expected_weights).sum() + (reference.critic2(expected) * expected_weights).sum()
        torch.testing.assert_close(a, b)
        a.backward(); b.backward()
        for name in ('graph_encoder', 'mixer', 'network', 'critic2'):
            for observed, original in zip(getattr(vf, name).parameters(), getattr(reference, name).parameters()):
                assert (observed.grad is None) == (original.grad is None)
                if observed.grad is not None:
                    torch.testing.assert_close(observed.grad, original.grad, atol=2e-5, rtol=2e-5)


def test_streamed_checkpoint_hash_matches_legacy_bytes_and_round_trip():
    import json
    from hashlib import sha256
    from src.recourse.replay import PrioritizedJointReplayBuffer
    from test_recourse_must_fix import _transition
    replay = PrioritizedJointReplayBuffer(capacity=4)
    for i in range(3):
        replay.add(_transition(transition_id=f'测试:{i}', sequence=i), td_error=i + .125)
    for mode in ('none', 'recent', 'full'):
        saved = replay.state_dict(mode=mode, recent_count=2)
        expected = sha256(json.dumps({'items': [r.to_dict() for r in saved['items']],
            'priorities': [float(p).hex() for p in saved['priorities']]}, ensure_ascii=False,
            sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        assert saved['content_hash'] == expected
        restored = PrioritizedJointReplayBuffer(capacity=4)
        restored.load_state_dict(saved)
        assert tuple(restored) == tuple(saved['items'])
        assert restored.priorities == tuple(saved['priorities'])
