"""Schema-checked TD-priority replay for joint transitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
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
        epsilon: float = 1e-5,
        rejection_bonus: float = 1.0,
        recourse_bonus: float = 1.0,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.rejection_bonus = float(rejection_bonus)
        self.recourse_bonus = float(recourse_bonus)
        self._items: list[RecourseTransition] = []
        self._priorities: list[float] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[RecourseTransition]:
        return iter(tuple(self._items))

    @property
    def priorities(self) -> tuple[float, ...]:
        return tuple(self._priorities)

    def add(self, transition: RecourseTransition, *, td_error: float | None = None) -> int:
        self._validate(transition)
        priority = self._initial_priority(transition, td_error)
        if len(self._items) < self.capacity:
            index = len(self._items)
            self._items.append(transition)
            self._priorities.append(priority)
        else:
            index = self._next_index
            self._items[index] = transition
            self._priorities[index] = priority
        self._next_index = (index + 1) % self.capacity
        return index

    append = add

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> ReplaySample:
        if not self._items:
            raise ValueError("cannot sample an empty replay buffer")
        rng = rng or np.random.default_rng()
        count = min(max(1, int(batch_size)), len(self._items))
        priorities = np.asarray(self._priorities, dtype=np.float64)
        scaled = np.power(np.maximum(priorities, self.epsilon), self.alpha)
        probabilities = scaled / scaled.sum()
        indices = rng.choice(
            len(self._items),
            size=count,
            replace=count > len(self._items),
            p=probabilities,
        )
        weights = np.power(len(self._items) * probabilities[indices], -self.beta)
        weights /= max(float(weights.max()), self.epsilon)
        return ReplaySample(
            transitions=tuple(self._items[int(index)] for index in indices),
            indices=tuple(int(index) for index in indices),
            weights=tuple(float(weight) for weight in weights),
            probabilities=tuple(float(probabilities[int(index)]) for index in indices),
        )

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        if len(indices) != len(td_errors):
            raise ValueError("indices and td_errors must have the same length")
        for index, error in zip(indices, td_errors):
            index = int(index)
            if index < 0 or index >= len(self._priorities):
                raise IndexError(f"replay index out of range: {index}")
            transition = self._items[index]
            self._priorities[index] = self._initial_priority(transition, float(error))

    def state_dict(self) -> dict:
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "rejection_bonus": self.rejection_bonus,
            "recourse_bonus": self.recourse_bonus,
            "items": list(self._items),
            "priorities": list(self._priorities),
            "next_index": self._next_index,
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
        for transition in items:
            self._validate(transition)
        self.capacity = int(state.get("capacity", self.capacity))
        self.alpha = float(state.get("alpha", self.alpha))
        self.beta = float(state.get("beta", self.beta))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.rejection_bonus = float(
            state.get("rejection_bonus", self.rejection_bonus)
        )
        self.recourse_bonus = float(state.get("recourse_bonus", self.recourse_bonus))
        self._items = items
        self._priorities = [float(priority) for priority in priorities]
        self._next_index = int(state.get("next_index", len(items))) % self.capacity

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
        has_rejection = bool(transition.rejection_outcome.rejected_request_ids)
        has_recourse = bool(
            transition.outcome_summary.count("assigned")
            or transition.outcome_summary.count("picked_up")
        )
        return float(
            base
            + self.rejection_bonus * float(has_rejection)
            + self.recourse_bonus * float(has_recourse)
        )

    @staticmethod
    def _validate(transition: RecourseTransition) -> None:
        if not isinstance(transition, RecourseTransition):
            raise TypeError("joint replay accepts only RecourseTransition rows")
        if transition.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"incompatible transition schema {transition.schema_version}"
            )

