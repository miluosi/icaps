"""Schema-checked TD-priority replay for joint transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence
import json
import pickle

import numpy as np

from .types import REPLAY_SCHEMA_VERSION, RecourseTransition


@dataclass(frozen=True)
class ReplaySample:
    transitions: tuple[RecourseTransition, ...]
    indices: tuple[int, ...]
    weights: tuple[float, ...]
    probabilities: tuple[float, ...]


class PrioritizedJointReplayBuffer:
    """One typed source of truth for integrated and recourse transitions."""

    def __init__(
        self,
        capacity: int = 100_000,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 100_000,
        epsilon: float = 1e-5,
        rejection_bonus: float = 1.0,
        recourse_bonus: float = 1.0,
        seed: int = 0,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta)
        self.beta_end = float(beta_end)
        self.beta_anneal_steps = max(1, int(beta_anneal_steps))
        self.beta_step = 0
        self.beta = self.beta_start
        self.epsilon = float(epsilon)
        self.rejection_bonus = float(rejection_bonus)
        self.recourse_bonus = float(recourse_bonus)
        self._items: list[RecourseTransition] = []
        self._priorities: list[float] = []
        self._next_index = 0
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._transition_index: dict[str, int] = {}
        self._validate_hyperparameters()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[RecourseTransition]:
        return iter(tuple(self._items))

    @property
    def priorities(self) -> tuple[float, ...]:
        return tuple(self._priorities)

    def get_by_transition_id(self, transition_id: str) -> RecourseTransition | None:
        index = self._transition_index.get(str(transition_id))
        if index is None or index >= len(self._items):
            return None
        transition = self._items[index]
        return (
            transition
            if transition.transition_id == str(transition_id)
            else None
        )

    def add(self, transition: RecourseTransition, *, td_error: float | None = None) -> int:
        self._validate(transition)
        if transition.transition_id in self._transition_index:
            raise ValueError(
                f"duplicate transition id: {transition.transition_id}"
            )
        priority = self._initial_priority(transition, td_error)
        if len(self._items) < self.capacity:
            index = len(self._items)
            self._items.append(transition)
            self._priorities.append(priority)
        else:
            index = self._next_index
            previous_id = self._items[index].transition_id
            self._transition_index.pop(previous_id, None)
            self._items[index] = transition
            self._priorities[index] = priority
        self._transition_index[transition.transition_id] = index
        self._next_index = (index + 1) % self.capacity
        return index

    append = add

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> ReplaySample:
        return self.sample_ready(batch_size, rng=rng)

    def sample_ready(
        self,
        batch_size: int,
        *,
        predicate: Callable[[RecourseTransition], bool] | None = None,
        rng: np.random.Generator | None = None,
    ) -> ReplaySample:
        """Sample only trainable rows without advancing the beta schedule."""

        if not self._items:
            raise ValueError("cannot sample an empty replay buffer")
        rng = rng or self.rng
        eligible_indices = np.asarray(
            [
                index
                for index, transition in enumerate(self._items)
                if predicate is None or bool(predicate(transition))
            ],
            dtype=np.int64,
        )
        if eligible_indices.size == 0:
            return ReplaySample((), (), (), ())
        count = min(max(1, int(batch_size)), int(eligible_indices.size))
        priorities = np.asarray(
            [self._priorities[int(index)] for index in eligible_indices],
            dtype=np.float64,
        )
        scaled = np.power(np.maximum(priorities, self.epsilon), self.alpha)
        probabilities = scaled / scaled.sum()
        local_indices = rng.choice(
            len(eligible_indices),
            size=count,
            replace=count > len(eligible_indices),
            p=probabilities,
        )
        indices = eligible_indices[local_indices]
        weights = np.power(
            len(eligible_indices) * probabilities[local_indices], -self.beta
        )
        weights /= max(float(weights.max()), self.epsilon)
        return ReplaySample(
            transitions=tuple(self._items[int(index)] for index in indices),
            indices=tuple(int(index) for index in indices),
            weights=tuple(float(weight) for weight in weights),
            probabilities=tuple(
                float(probabilities[int(index)]) for index in local_indices
            ),
        )

    def advance_beta(self) -> None:
        """Advance PER annealing after one successful optimizer update."""

        self.beta_step += 1
        fraction = min(1.0, self.beta_step / float(self.beta_anneal_steps))
        self.beta = self.beta_start + fraction * (
            self.beta_end - self.beta_start
        )

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        if len(indices) != len(td_errors):
            raise ValueError("indices and td_errors must have the same length")
        for index, error in zip(indices, td_errors):
            index = int(index)
            if index < 0 or index >= len(self._priorities):
                raise IndexError(f"replay index out of range: {index}")
            self._priorities[index] = self.priority_from_td(
                self._items[index], float(error)
            )

    def priority_from_td(
        self,
        transition: RecourseTransition,
        td_error: float,
    ) -> float:
        has_rejection = bool(
            transition.rejection_outcome.rejected_request_ids
        )
        has_recourse = bool(
            transition.outcome_summary.count("assigned")
            or transition.outcome_summary.count("picked_up")
        )
        return float(
            abs(float(td_error))
            + self.epsilon
            + self.rejection_bonus * float(has_rejection)
            + self.recourse_bonus * float(has_recourse)
        )

    def state_dict(
        self,
        *,
        mode: str = "full",
        recent_count: int = 5_000,
    ) -> dict:
        mode = str(mode)
        if mode not in {"none", "recent", "full"}:
            raise ValueError("replay checkpoint mode must be none, recent, or full")
        if mode == "none":
            items: list[RecourseTransition] = []
            priorities: list[float] = []
        elif mode == "recent" and len(self._items) > int(recent_count):
            ordered_indices = [
                (self._next_index - offset - 1) % len(self._items)
                for offset in range(int(recent_count))
            ][::-1]
            items = [self._items[index] for index in ordered_indices]
            priorities = [self._priorities[index] for index in ordered_indices]
        else:
            items = list(self._items)
            priorities = list(self._priorities)
        content_hash = self._content_hash(items, priorities)
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_anneal_steps": self.beta_anneal_steps,
            "beta_step": self.beta_step,
            "epsilon": self.epsilon,
            "rejection_bonus": self.rejection_bonus,
            "recourse_bonus": self.recourse_bonus,
            "checkpoint_mode": mode,
            "items": items,
            "priorities": priorities,
            "next_index": len(items) % self.capacity,
            "seed": self.seed,
            "rng_state": self.rng.bit_generator.state,
            "content_hash": content_hash,
        }

    def load_state_dict(self, state: dict) -> None:
        version = int(state.get("schema_version", -1))
        if version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"incompatible replay schema {version}; expected {REPLAY_SCHEMA_VERSION}"
            )
        items = list(state.get("items", ()))
        priorities = list(state.get("priorities", ()))
        if len(items) != len(priorities):
            raise ValueError("replay items/priorities length mismatch")
        if len(items) > int(state.get("capacity", self.capacity)):
            raise ValueError("serialized replay exceeds its declared capacity")
        expected_hash = state.get("content_hash")
        if expected_hash is not None and str(expected_hash) != self._content_hash(
            items, priorities
        ):
            raise ValueError("serialized replay content hash mismatch")
        for transition in items:
            self._validate(transition)
        transition_ids = [transition.transition_id for transition in items]
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("serialized replay contains duplicate transition ids")
        self.capacity = int(state.get("capacity", self.capacity))
        self.alpha = float(state.get("alpha", self.alpha))
        self.beta_start = float(
            state.get("beta_start", state.get("beta", self.beta_start))
        )
        self.beta_end = float(state.get("beta_end", self.beta_end))
        self.beta_anneal_steps = int(
            state.get("beta_anneal_steps", self.beta_anneal_steps)
        )
        self.beta_step = int(state.get("beta_step", 0))
        fraction = min(1.0, self.beta_step / float(max(1, self.beta_anneal_steps)))
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.rejection_bonus = float(
            state.get("rejection_bonus", self.rejection_bonus)
        )
        self.recourse_bonus = float(state.get("recourse_bonus", self.recourse_bonus))
        self._items = items
        self._priorities = [float(priority) for priority in priorities]
        self._next_index = int(state.get("next_index", len(items))) % self.capacity
        self.seed = int(state.get("seed", self.seed))
        self.rng = np.random.default_rng(self.seed)
        if state.get("rng_state") is not None:
            self.rng.bit_generator.state = state["rng_state"]
        self._transition_index = {
            transition.transition_id: index
            for index, transition in enumerate(self._items)
        }
        self._validate_hyperparameters()

    @staticmethod
    def _content_hash(
        items: Sequence[RecourseTransition], priorities: Sequence[float]
    ) -> str:
        payload = {
            "items": [transition.to_dict() for transition in items],
            "priorities": [float(priority).hex() for priority in priorities],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "PrioritizedJointReplayBuffer":
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        replay = cls(capacity=int(state.get("capacity", 1)))
        replay.load_state_dict(state)
        return replay

    def _initial_priority(
        self,
        transition: RecourseTransition,
        td_error: float | None,
    ) -> float:
        base = (
            max(self._priorities, default=1.0)
            if td_error is None
            else abs(float(td_error)) + self.epsilon
        )
        # ``base`` for a fresh row is the current maximum priority, not a TD
        # error.  Add the persistent event bonuses using the same formula as
        # later updates.
        return self.priority_from_td(transition, base - self.epsilon)

    @staticmethod
    def _validate(transition: RecourseTransition) -> None:
        if not isinstance(transition, RecourseTransition):
            raise TypeError("joint replay accepts only RecourseTransition rows")
        if transition.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"incompatible transition schema {transition.schema_version}"
            )

    def _validate_hyperparameters(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("PER alpha must be in [0, 1]")
        if not 0.0 <= self.beta_start <= 1.0:
            raise ValueError("PER beta_start must be in [0, 1]")
        if not self.beta_start <= self.beta_end <= 1.0:
            raise ValueError("PER beta_end must be in [beta_start, 1]")
        if self.epsilon <= 0.0:
            raise ValueError("PER epsilon must be positive")
