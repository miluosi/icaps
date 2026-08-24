"""Collection-time coordinator for linked joint transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

from .lifecycle import RequestLifecycleTracker
from .replay import PrioritizedJointReplayBuffer
from .state_snapshot import StateSnapshotBuilder
from .types import (
    FeasibleGraphSnapshot,
    JointActionSnapshot,
    PlannerMetadata,
    RecourseTransition,
    SystemSnapshot,
)


@dataclass
class PendingTransition:
    transition_id: str
    episode_id: int
    day_id: str
    seed: int
    epoch_id: int
    mode: str
    recourse_variant: str
    state_variant: str
    learner_variant: str
    solver_backend: str
    pre_state: SystemSnapshot
    ev_stage_graph: FeasibleGraphSnapshot | None = None
    ev_joint_action: JointActionSnapshot | None = None
    residual_state: SystemSnapshot | None = None
    aev_stage_graph: FeasibleGraphSnapshot | None = None
    aev_joint_action: JointActionSnapshot | None = None


class RecourseCoordinator:
    def __init__(
        self,
        *,
        lifecycle: RequestLifecycleTracker | None = None,
        replay: PrioritizedJointReplayBuffer | None = None,
    ) -> None:
        self.lifecycle = lifecycle or RequestLifecycleTracker()
        self.replay = replay or PrioritizedJointReplayBuffer()
        self.pending: PendingTransition | None = None

    def begin(
        self,
        env: Any,
        *,
        mode: str,
        recourse_variant: str = "legacy",
        state_variant: str = "joint_state_shared_critic",
        learner_variant: str = "optimization_anchored_residual",
        solver_backend: str = "unknown",
    ) -> PendingTransition:
        if self.pending is not None:
            raise RuntimeError("previous joint transition was not finalized")
        pre_state = StateSnapshotBuilder.build(env)
        day_id = dict(pre_state.exogenous_context).get("day_id", "")
        self.pending = PendingTransition(
            transition_id=str(uuid.uuid4()),
            episode_id=int(getattr(env, "episode_day_index", getattr(env, "episode_index", 0)) or 0),
            day_id=str(day_id),
            seed=int(getattr(env, "initial_random_seed", 0) or 0),
            epoch_id=pre_state.epoch_id,
            mode=str(mode),
            recourse_variant=str(recourse_variant),
            state_variant=str(state_variant),
            learner_variant=str(learner_variant),
            solver_backend=str(solver_backend),
            pre_state=pre_state,
        )
        return self.pending

    def finalize(
        self,
        env: Any,
        *,
        rewards: dict[int, float],
        done: bool,
    ) -> RecourseTransition | None:
        pending = self.pending
        if pending is None:
            return None
        reward_ev = sum(
            float(rewards.get(vehicle_id, 0.0))
            for vehicle_id, vehicle in getattr(env, "vehicles", {}).items()
            if int(vehicle.get("type", 1)) == 1
        )
        reward_aev = sum(
            float(rewards.get(vehicle_id, 0.0))
            for vehicle_id, vehicle in getattr(env, "vehicles", {}).items()
            if int(vehicle.get("type", 1)) == 2
        )
        next_state = StateSnapshotBuilder.build(env)
        transition = RecourseTransition(
            transition_id=pending.transition_id,
            episode_id=pending.episode_id,
            day_id=pending.day_id,
            seed=pending.seed,
            epoch_id=pending.epoch_id,
            mode=pending.mode,
            recourse_variant=pending.recourse_variant,
            state_variant=pending.state_variant,
            learner_variant=pending.learner_variant,
            solver_backend=pending.solver_backend,
            pre_state=pending.pre_state,
            ev_stage_graph=pending.ev_stage_graph,
            ev_joint_action=pending.ev_joint_action,
            rejection_outcome=self.lifecycle.rejection_outcome(
                transition_id=pending.transition_id
            ),
            residual_state=pending.residual_state,
            aev_stage_graph=pending.aev_stage_graph,
            aev_joint_action=pending.aev_joint_action,
            reward_ev=reward_ev,
            reward_aev=reward_aev,
            reward_system=reward_ev + reward_aev,
            next_state=next_state,
            elapsed_epochs=max(0.0, next_state.current_time - pending.pre_state.current_time),
            done=bool(done),
            planner_metadata=PlannerMetadata(backend=pending.solver_backend),
            outcome_summary=self.lifecycle.outcome_summary(epoch_id=pending.epoch_id),
        )
        self.replay.add(transition)
        self.pending = None
        return transition
