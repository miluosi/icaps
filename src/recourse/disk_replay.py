"""Lossless disk payloads with the same PER indices, priorities and RNG as RAM replay.

Only headers are scanned for readiness. Sample payloads are read lazily, so a
64-row sample does not hydrate 64 complete fleet graphs before the edge budget
can stop the update. The hot cache is a performance cache, never a sample filter.
"""
from collections import OrderedDict
from dataclasses import dataclass
import gzip
import hashlib
import os
from pathlib import Path
import pickle
import tempfile
import tarfile
import uuid

import numpy as np

from .replay import PrioritizedJointReplayBuffer, ReplaySample
from .types import is_true_same_epoch_recourse


@dataclass(frozen=True, slots=True)
class ReplayHeader:
    transition_id: str
    next_transition_id: str | None
    done: bool
    mode: str
    variant: str
    family: str
    ev_present: bool
    aev_present: bool
    ev_selected: bool
    aev_selected: bool
    rejection: bool
    recourse: bool

    @classmethod
    def from_row(cls, row):
        return cls(row.transition_id, row.next_transition_id, row.done, row.mode,
                   row.recourse_variant, row.recourse_target_family,
                   row.ev_stage_graph is not None, row.aev_stage_graph is not None,
                   bool(row.ev_joint_action and row.ev_joint_action.selected_edge_ids),
                   bool(row.aev_joint_action and row.aev_joint_action.selected_edge_ids),
                   bool(row.rejection_outcome.rejected_request_ids),
                   any(is_true_same_epoch_recourse(e) for e in row.outcome_summary.events))

    def ready(self, if_ev, contains):
        if self.mode in {'integrated', 'integrated_repair'}:
            if if_ev or not (self.ev_present and self.ev_selected):
                return False
        elif self.mode == 'ev_first':
            if not if_ev and self.variant in {'r1_structured', 'r2'}:
                return False
            if not ((self.ev_present and self.ev_selected) if if_ev else
                    (self.aev_present and self.aev_selected)):
                return False
        elif self.mode == 'aev_first':
            if not ((self.ev_present and self.ev_selected) if if_ev else
                    (self.aev_present and self.aev_selected)):
                return False
        else:
            raise ValueError(f'unsupported joint transition mode: {self.mode}')
        if self.done:
            return True
        if self.mode == 'ev_first' and if_ev and self.family == 'nested_follower':
            return self.aev_present
        if self.mode == 'aev_first' and not if_ev:
            return self.ev_present
        return bool(self.next_transition_id and contains(self.next_transition_id))


class LazyReplayRows:
    def __init__(self, store, indices):
        self.store, self.indices = store, tuple(int(i) for i in indices)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for index in self.indices:
            yield self.store[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return LazyReplayRows(self.store, self.indices[index])
        return self.store[self.indices[index]]


class DiskRows:
    def __init__(self, directory, cache_rows):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='joint-', dir=directory)
        self.headers = []
        self.cache = OrderedDict()
        self.cache_rows = max(0, int(cache_rows))
        self.loads = self.writes = 0

    def __len__(self):
        return len(self.headers)

    def _path(self, index):
        return Path(self.directory.name) / f'{index}.pkl.gz'

    def __getitem__(self, index):
        if isinstance(index, slice):
            return LazyReplayRows(self, range(len(self))[index])
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        with gzip.open(self._path(index), 'rb') as handle:
            row = pickle.load(handle)
        self.loads += 1
        if self.cache_rows:
            self.cache[index] = row
            while len(self.cache) > self.cache_rows:
                self.cache.popitem(last=False)
        return row

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __setitem__(self, index, row):
        path = self._path(index)
        temporary = path.with_suffix('.tmp')
        try:
            # Stream pickle directly; never construct pickle.dumps(row) in RAM.
            with gzip.open(temporary, 'wb', compresslevel=1) as handle:
                pickle.dump(row, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        header = ReplayHeader.from_row(row)
        if index == len(self):
            self.headers.append(header)
        else:
            self.headers[index] = header
        self.cache.pop(index, None)
        self.writes += 1

    def append(self, row):
        self[len(self)] = row


class DiskPrioritizedJointReplayBuffer(PrioritizedJointReplayBuffer):
    storage = 'disk'

    def __init__(self, *args, directory, cache_rows=2, **kwargs):
        super().__init__(*args, **kwargs)
        self._items = DiskRows(directory, cache_rows)
        self.cache_rows = self._items.cache_rows

    def __iter__(self):
        return iter(self._items)

    def add(self, transition, *, td_error=None):
        self._validate(transition)
        if self.has_transition(transition.transition_id):
            raise ValueError(f'duplicate transition id: {transition.transition_id}')
        priority = self._initial_priority(transition, td_error)
        index = len(self) if len(self) < self.capacity else self._next_index
        previous_id = self._items.headers[index].transition_id if index < len(self) else None
        self._items[index] = transition  # commit to disk before changing the index
        if previous_id is not None:
            self._transition_index.pop(previous_id, None)
            self._priorities[index] = priority
        else:
            self._priorities.append(priority)
        self._transition_index[transition.transition_id] = index
        self._next_index = (index + 1) % self.capacity
        return index

    append = add

    def has_ready(self, *, ifEV):
        return any(h.ready(ifEV, self.has_transition) for h in self._items.headers)

    def sample_ready(self, batch_size, *, predicate=None, rng=None, ready_fleet=None):
        if not len(self):
            raise ValueError('cannot sample an empty replay buffer')
        if ready_fleet is None:
            eligible = [i for i in range(len(self))
                        if predicate is None or predicate(self._items[i])]
        else:
            eligible = [i for i, h in enumerate(self._items.headers)
                        if h.ready(ready_fleet, self.has_transition)]
        if not eligible:
            return ReplaySample((), (), (), ())
        indices = np.asarray(eligible, dtype=np.int64)
        priorities = np.asarray([self._priorities[i] for i in eligible], dtype=np.float64)
        scaled = np.power(np.maximum(priorities, self.epsilon), self.alpha)
        probabilities = scaled / scaled.sum()
        count = min(max(1, int(batch_size)), len(eligible))
        local = (rng or self.rng).choice(len(eligible), size=count,
                                        replace=count > len(eligible), p=probabilities)
        selected = indices[local]
        weights = np.power(len(eligible) * probabilities[local], -self.beta)
        weights /= max(float(weights.max()), self.epsilon)
        return ReplaySample(LazyReplayRows(self._items, selected),
                            tuple(int(i) for i in selected), tuple(float(w) for w in weights),
                            tuple(float(probabilities[i]) for i in local))

    def update_priorities(self, indices, td_errors):
        if len(indices) != len(td_errors):
            raise ValueError('indices and td_errors must have the same length')
        for index, error in zip(indices, td_errors):
            index = int(index)
            if not 0 <= index < len(self):
                raise IndexError(f'replay index out of range: {index}')
            h = self._items.headers[index]
            self._priorities[index] = (abs(float(error)) + self.epsilon
                + self.rejection_bonus * h.rejection + self.recourse_bonus * h.recourse)

    def clear_hot_cache(self):
        self._items.cache.clear()

    @staticmethod
    def _file_hash(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def state_dict(self, *, mode='full', recent_count=5000):
        state = super().state_dict(mode='none')
        if mode == 'none':
            return state
        if mode not in {'recent', 'full'}:
            raise ValueError('replay checkpoint mode must be none, recent, or full')
        if mode == 'recent' and int(recent_count) < 1:
            raise ValueError('recent_count must be positive')
        indices = ([(self._next_index - offset - 1) % len(self)
                    for offset in range(int(recent_count))][::-1]
                   if mode == 'recent' and len(self) > int(recent_count)
                   else list(range(len(self))))
        # An explicit replay checkpoint gets a persistent sidecar. Copy the
        # already compressed records, not a list of hydrated Python graphs.
        root = Path(self._items.directory.name).parent / 'checkpoints'
        root.mkdir(exist_ok=True)
        path = root / f'{uuid.uuid4().hex}.tar'
        temporary = path.with_suffix('.tmp')
        try:
            with tarfile.open(temporary, 'w') as archive:
                for offset, index in enumerate(indices):
                    archive.add(self._items._path(index), arcname=f'{offset}.pkl.gz')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        checksum = self._file_hash(path)
        state.update(checkpoint_mode=mode, priorities=[self._priorities[i] for i in indices],
            next_index=self._next_index if len(indices) == len(self) else len(indices) % self.capacity,
            content_hash=checksum,
            disk_replay_schema=1, disk_replay_archive=str(path.resolve()),
            disk_replay_count=len(indices), disk_replay_sha256=checksum)
        return state

    def load_state_dict(self, state):
        store = self._items
        if 'disk_replay_schema' in state:
            if state['disk_replay_schema'] != 1:
                raise ValueError('Unsupported disk replay checkpoint schema')
            path = Path(state['disk_replay_archive'])
            if self._file_hash(path) != state['disk_replay_sha256']:
                raise ValueError('Replay sidecar checksum mismatch')
            count = int(state['disk_replay_count'])
            priorities = list(state['priorities'])
            if count != len(priorities) or count > int(state['capacity']):
                raise ValueError('Replay sidecar count/priorities mismatch')
            header_state = dict(state, items=[], priorities=[])
            header_state.pop('content_hash', None)
            header_state.pop('disk_replay_schema', None)
            super().load_state_dict(header_state)
            store.cache.clear()
            store.headers.clear()
            with tarfile.open(path, 'r') as archive:
                for index in range(count):
                    member = archive.next()
                    if member is None or not member.isfile() or member.name != f'{index}.pkl.gz':
                        raise ValueError('Invalid replay sidecar record')
                    with archive.extractfile(member) as source:
                        with gzip.GzipFile(fileobj=source, mode='rb') as zipped:
                            row = pickle.load(zipped)
                    self._validate(row)
                    if row.transition_id in self._transition_index:
                        raise ValueError('Duplicate replay sidecar transition')
                    store.append(row)
                    self._transition_index[row.transition_id] = index
            self._items = store
            self._priorities = [float(p) for p in priorities]
            self._next_index = int(state['next_index']) % self.capacity
            return
        super().load_state_dict(state)
        rows = self._items
        store.cache.clear()
        for path in Path(store.directory.name).glob('*.pkl.gz'):
            path.unlink()
        store.headers.clear()
        for row in rows:
            store.append(row)
        self._items = store


def make_joint_replay(*, env=None, **kwargs):
    """Large production fleets spill replay; small experiments keep the RAM path."""
    mode = os.environ.get('ICAPS_REPLAY_STORAGE', 'auto').lower()
    if mode not in {'auto', 'memory', 'disk'}:
        raise ValueError('ICAPS_REPLAY_STORAGE must be auto, memory, or disk')
    large = int(getattr(env, 'num_vehicles', 0) or 0) >= 500
    if mode == 'disk' or mode == 'auto' and large:
        replay = DiskPrioritizedJointReplayBuffer(**kwargs,
            directory=os.environ.get('ICAPS_REPLAY_DIR', 'results/.replay_cache'),
            cache_rows=int(os.environ.get('ICAPS_REPLAY_CACHE_ROWS', '2')))
        print(f'Joint replay: lossless disk storage at {replay._items.directory.name}; '
              f'{replay.cache_rows} cached rows, capacity={replay.capacity} (unchanged)', flush=True)
        return replay
    return PrioritizedJointReplayBuffer(**kwargs)
