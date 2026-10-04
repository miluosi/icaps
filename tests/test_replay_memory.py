"""RAM fixes must preserve replay population, random samples and learning."""
from copy import deepcopy
import copyreg
from dataclasses import asdict, replace
import pickle
from types import SimpleNamespace

import pytest
import torch

from src.memory_lifecycle import release_training_caches
from src.recourse.disk_replay import DiskPrioritizedJointReplayBuffer, make_joint_replay
from src.recourse.replay import PrioritizedJointReplayBuffer
from src.recourse.types import FeasibleEdgeSnapshot
from test_recourse_must_fix import _graph, _transition, _vf


def row(i, **kwargs):
    return _transition(f'run:episode:0:sequence:{i}', sequence=i,
        ev_graph=_graph(f'ev-{i}', stage=1, vehicle_id=0, vehicle_type=1),
        aev_graph=_graph(f'aev-{i}', stage=2, vehicle_id=1, vehicle_type=2), **kwargs)


def test_disk_population_per_rng_priority_and_checkpoint_equal_ram(tmp_path):
    ram = PrioritizedJointReplayBuffer(capacity=4, seed=79)
    disk = DiskPrioritizedJointReplayBuffer(capacity=4, seed=79, directory=tmp_path)
    for i in range(11):
        for replay in (ram, disk):
            replay.add(row(i), td_error=i + .15)
        assert tuple(ram) == tuple(disk)
        assert ram.priorities == disk.priorities
        a, b = ram.sample(3), disk.sample(3)
        assert a.indices == b.indices
        assert a.weights == b.weights
        assert a.probabilities == b.probabilities
        assert tuple(a.transitions) == tuple(b.transitions)
        for replay in (ram, disk):
            replay.update_priorities(a.indices, [.1, .2, .3][:len(a.indices)])
            replay.advance_beta()
    assert len(list(tmp_path.glob('joint-*/*.pkl.gz'))) == 4
    assert len(disk._items.cache) <= 2
    assert disk.get_by_transition_id('run:episode:0:sequence:0') is None
    for mode in ('none', 'recent', 'full'):
        a = ram.state_dict(mode=mode, recent_count=2)
        b = disk.state_dict(mode=mode, recent_count=2)
        if mode == 'none':
            assert a == b
        else:
            assert b['items'] == []
            assert b['priorities'] == a['priorities']
        restored = DiskPrioritizedJointReplayBuffer(capacity=1, directory=tmp_path)
        restored.load_state_dict(b)
        assert tuple(restored) == tuple(a['items'])
        assert restored.priorities == tuple(a['priorities'])
        assert restored.rng.bit_generator.state == ram.rng.bit_generator.state
        memory_restored = PrioritizedJointReplayBuffer(capacity=4)
        memory_restored.load_state_dict(a)
        for restored_buffer in (restored, memory_restored):
            restored_buffer.add(row(11))
        assert tuple(restored) == tuple(memory_restored)
        if mode == 'full':
            assert restored._next_index == (ram._next_index + 1) % ram.capacity


@pytest.mark.parametrize('mode', ['integrated', 'ev_first', 'aev_first'])
@pytest.mark.parametrize('variant', ['r1', 'r2', 'r3', 'r4', 'recourse_macro'])
def test_header_readiness_equals_full_graph_without_disk_reads(tmp_path, mode, variant):
    vf = _vf(recourse=variant)
    disk = DiskPrioritizedJointReplayBuffer(capacity=8, directory=tmp_path)
    vf.joint_replay_buffer = disk
    rows = [row(0, mode=mode, recourse_variant=variant, done=False,
                next_id='run:episode:0:sequence:1'),
            row(1, mode=mode, recourse_variant=variant),
            row(2, mode=mode, recourse_variant=variant, done=False, next_id='missing')]
    for item in rows:
        disk.add(item)
    for fleet in (False, True):
        expected = [i for i, item in enumerate(rows) if vf._joint_row_ready(item, ifEV=fleet)]
        assert disk.has_ready(ifEV=fleet) == bool(expected)
        sample = disk.sample_ready(8, ready_fleet=fleet)
        assert set(sample.indices) == set(expected)
    assert disk._items.loads == 0


@pytest.mark.parametrize('fleet', [False, True])
def test_disk_and_ram_produce_identical_training_updates(tmp_path, fleet):
    torch.manual_seed(31)
    ram = _vf(recourse='r3')
    disk = deepcopy(ram)
    disk.joint_replay_buffer = DiskPrioritizedJointReplayBuffer(capacity=100000,
        directory=tmp_path, seed=0)
    for i in range(6):
        # Include nonterminal successor scoring/assignment, not just terminal TD.
        transition = row(i, done=i == 5, next_id=None if i == 5 else f'run:episode:0:sequence:{i+1}')
        ram.joint_replay_buffer.add(transition)
        disk.joint_replay_buffer.add(transition)
    for _ in range(2):
        assert ram._train_joint_step(3, ifEV=fleet) == disk._train_joint_step(3, ifEV=fleet)
        assert ram.joint_replay_buffer.priorities == disk.joint_replay_buffer.priorities
        for name in ('network', 'critic2', 'graph_encoder', 'mixer', 'target_network'):
            for left, right in zip(getattr(ram, name).parameters(), getattr(disk, name).parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_sampling_is_lazy_and_hot_cache_does_not_delete_samples(tmp_path):
    disk = DiskPrioritizedJointReplayBuffer(capacity=100, directory=tmp_path, cache_rows=1)
    for i in range(20):
        disk.add(row(i))
    sample = disk.sample(20)
    assert disk._items.loads == 0
    first = next(iter(sample.transitions))
    assert disk._items.loads == 1
    assert len(disk._items.cache) == 1
    disk.clear_hot_cache()
    assert len(disk) == 20 and len(disk._items.cache) == 0
    assert disk.get_by_transition_id(first.transition_id) == first


def test_failed_disk_write_does_not_destroy_existing_replay(tmp_path, monkeypatch):
    disk = DiskPrioritizedJointReplayBuffer(capacity=1, directory=tmp_path)
    original = row(0)
    disk.add(original)
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr('src.recourse.disk_replay.pickle.dump', fail)
    with pytest.raises(OSError, match='disk full'):
        disk.add(row(1))
    assert tuple(disk) == (original,)
    assert disk.has_transition(original.transition_id)
    assert not list(tmp_path.glob('joint-*/*.tmp'))


def test_slotted_edges_load_old_dict_pickles_and_keep_exact_fields():
    edge = _graph('edge', stage=1, vehicle_id=0, vehicle_type=1).edges[0]
    assert not hasattr(edge, '__dict__')
    assert deepcopy(edge) is edge
    assert pickle.loads(pickle.dumps(edge)) == edge
    class LegacyEdge:
        def __reduce__(self):
            return copyreg._reconstructor, (FeasibleEdgeSnapshot, object, None), asdict(edge)
    assert pickle.loads(pickle.dumps(LegacyEdge())) == edge
    assert asdict(replace(edge, structured_score=42.))['structured_score'] == 42.


def test_boundary_cleanup_preserves_training_state_and_scores(tmp_path):
    vf = _vf()
    vf.joint_replay_buffer = DiskPrioritizedJointReplayBuffer(capacity=4, directory=tmp_path)
    vf.joint_replay_buffer.add(row(0))
    graph = row(0).ev_stage_graph
    before = vf._graph_edge_scores(graph, target_context=False)
    priority = vf.joint_replay_buffer.priorities
    random_state = deepcopy(vf.joint_replay_buffer.rng.bit_generator.state)
    vf._target_component_cache['obsolete'] = object()
    release_training_caches(vf, vf)
    assert vf._graph_cache is None and not vf._target_component_cache
    assert len(vf.joint_replay_buffer) == 1
    assert vf.joint_replay_buffer.priorities == priority
    assert vf.joint_replay_buffer.rng.bit_generator.state == random_state
    assert vf._graph_edge_scores(graph, target_context=False) == before


def test_factory_auto_spills_large_fleets_only(tmp_path, monkeypatch):
    monkeypatch.setenv('ICAPS_REPLAY_STORAGE', 'auto')
    monkeypatch.setenv('ICAPS_REPLAY_DIR', str(tmp_path))
    assert make_joint_replay(env=SimpleNamespace(num_vehicles=3000)).storage == 'disk'
    assert make_joint_replay(env=SimpleNamespace(num_vehicles=100)).storage == 'memory'


def test_joint_auxiliary_rows_do_not_pin_full_graphs():
    from src.Action import IdleAction
    vf = _vf(recourse='r3')
    graph = row(0).ev_stage_graph
    action = IdleAction([], 0, 0)
    action.metadata.state_snapshot = graph.state
    action.metadata.feasible_graph_snapshot = graph
    action.metadata.extras['aev_stage_graph'] = graph
    vf.set_replay_collection_context(action)
    vf.store_experience(vehicle_id=0, vehicle_location=0, battery_level=.8)
    assert not any('snapshot' in k or k == 'aev_stage_graph' for k in vf.experience_buffer[-1])


def test_trace_hash_matches_historical_json_bytes():
    import hashlib
    import json
    from src.memory_lifecycle import JsonArrayDigest
    values = ['edge:0', '中文', 'quotes"\\', 42, None, True]
    digest = JsonArrayDigest()
    assert digest.hexdigest() == hashlib.sha256(b'[]').hexdigest()
    digest.extend(values[:3]); digest.extend(values[3:])
    assert digest.hexdigest() == hashlib.sha256(json.dumps(values).encode()).hexdigest()
    assert digest.hexdigest() == digest.hexdigest()


def test_ram_guard_does_not_silently_drop_training_samples(monkeypatch):
    from src.memory_lifecycle import check_training_memory
    monkeypatch.setenv('ICAPS_RAM_LIMIT_GB', '10')
    monkeypatch.setenv('ICAPS_RAM_LOG_EVERY', '100')
    monkeypatch.setattr('src.memory_lifecycle.ram_usage_gib', lambda: 12.)
    released = []
    monkeypatch.setattr('src.memory_lifecycle.release_training_caches', lambda *args: released.append(True))
    assert check_training_memory(1) is None
    with pytest.raises(MemoryError, match='replay was not discarded'):
        check_training_memory(100)
    assert released == [True]


def test_disk_checkpoint_streams_without_hydrating_rows(tmp_path):
    disk = DiskPrioritizedJointReplayBuffer(capacity=10, directory=tmp_path)
    for i in range(6):
        disk.add(row(i))
    before = disk._items.loads
    state = disk.state_dict(mode='recent', recent_count=4)
    assert disk._items.loads == before
    assert not state['items']
    checkpoint = tmp_path / 'resume.pth'
    torch.save({'replay': state}, checkpoint)
    state = torch.load(checkpoint, weights_only=False)['replay']
    restored = DiskPrioritizedJointReplayBuffer(capacity=10, directory=tmp_path)
    restored.load_state_dict(state)
    assert tuple(restored) == tuple(row(i) for i in range(2, 6))
    replay_file = tmp_path / 'replay.pkl'
    disk.save(replay_file)
    assert tuple(PrioritizedJointReplayBuffer.load(replay_file)) == tuple(disk)
    assert tuple(DiskPrioritizedJointReplayBuffer.load(replay_file)) == tuple(disk)
    from pathlib import Path
    Path(state['disk_replay_archive']).write_bytes(b'damaged')
    with pytest.raises(ValueError, match='checksum'):
        restored.load_state_dict(state)
