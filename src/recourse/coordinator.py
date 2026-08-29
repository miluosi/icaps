"""Collection-time coordinator for linked joint transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lifecycle import RequestLifecycleTracker
from .replay import PrioritizedJointReplayBuffer
from .state_snapshot import StateSnapshotBuilder
from .reward_ledger import build_reward_ledger
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
    run_id: str
    cumulative_episode_id: int
    transition_sequence_index: int
    previous_transition_id: str | None
    next_transition_id: str
    request_generation_seed: int
    vehicle_initialization_seed: int
    ev_stage_graph: FeasibleGraphSnapshot | None = None
    ev_joint_action: JointActionSnapshot | None = None
    residual_state: SystemSnapshot | None = None
    aev_stage_graph: FeasibleGraphSnapshot | None = None
    aev_joint_action: JointActionSnapshot | None = None
    committed_aev_edge_ids: tuple[str, ...] = ()
    repair_hold_aev_ids: tuple[int, ...] = ()
    repair_candidate_request_ids: tuple[int, ...] = ()


class RecourseCoordinator:
    def __init__(
        self,
        *,
        lifecycle: RequestLifecycleTracker | None = None,
        replay: PrioritizedJointReplayBuffer | None = None,
    ) -> None:
        self.lifecycle = lifecycle or RequestLifecycleTracker()
        # Learners own the replay used for optimization.  Keeping another
        # implicit full copy here previously tripled snapshot memory.
        self.replay = replay
        self.pending: PendingTransition | None = None
        self._active_episode_id: int | None = None
        self._sequence_index = 0

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
        cumulative_episode_id = int(
            getattr(
                env,
                "cumulative_episode_index",
                getattr(env, "episode_day_index", getattr(env, "episode_index", 0)),
            )
            or 0
        )
        if self._active_episode_id != cumulative_episode_id:
            self._active_episode_id = cumulative_episode_id
            self._sequence_index = 0
        sequence_index = self._sequence_index
        self._sequence_index += 1
        run_id = str(
            getattr(
                env,
                "recourse_run_id",
                f"seed-{int(getattr(env, 'initial_random_seed', 0) or 0)}",
            )
        )
        transition_id = (
            f"{run_id}:episode:{cumulative_episode_id}:sequence:{sequence_index}"
        )
        next_transition_id = (
            f"{run_id}:episode:{cumulative_episode_id}:sequence:{sequence_index + 1}"
        )
        previous_transition_id = (
            None
            if sequence_index == 0
            else f"{run_id}:episode:{cumulative_episode_id}:sequence:{sequence_index - 1}"
        )
        self.pending = PendingTransition(
            transition_id=transition_id,
            episode_id=cumulative_episode_id,
            day_id=str(day_id),
            seed=int(getattr(env, "initial_random_seed", 0) or 0),
            epoch_id=pre_state.epoch_id,
            mode=str(mode),
            recourse_variant=str(recourse_variant),
            state_variant=str(state_variant),
            learner_variant=str(learner_variant),
            solver_backend=str(solver_backend),
            pre_state=pre_state,
            run_id=run_id,
            cumulative_episode_id=cumulative_episode_id,
            transition_sequence_index=sequence_index,
            previous_transition_id=previous_transition_id,
            next_transition_id=next_transition_id,
            request_generation_seed=int(
                getattr(env, "request_generation_seed", 0) or 0
            ),
            vehicle_initialization_seed=int(
                getattr(env, "vehicle_initialization_seed", 0) or 0
            ),
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
        self._complete_missing_stage_graphs(env, pending)
        selected_edges = []
        for graph, action in (
            (pending.ev_stage_graph, pending.ev_joint_action),
            (pending.aev_stage_graph, pending.aev_joint_action),
        ):
            if graph is None or action is None:
                continue
            if tuple(graph.selected_edge_ids) != tuple(action.selected_edge_ids):
                raise AssertionError(
                    "serialized graph/action selected-edge mismatch: "
                    f"graph={graph.selected_edge_ids}, "
                    f"action={action.selected_edge_ids}"
                )
            selected_ids = set(action.selected_edge_ids)
            selected_edges.extend(
                edge for edge in graph.edges if edge.edge_id in selected_ids
            )
        rewarded_vehicle_ids = tuple(
            sorted({int(edge.vehicle_id) for edge in selected_edges})
        )
        unattributed = [
            int(vehicle_id)
            for vehicle_id, reward in rewards.items()
            if abs(float(reward)) > 1e-12
            and int(vehicle_id) not in rewarded_vehicle_ids
        ]
        if unattributed:
            raise AssertionError(
                "epoch reward contains vehicles without a serialized selected "
                f"action edge: {unattributed}"
            )
        continuing_action_edge_ids = tuple(
            edge.edge_id
            for edge in selected_edges
            if bool(dict(edge.metadata).get("continuing", False))
        )
        reward_ev = sum(
            float(rewards.get(vehicle_id, 0.0))
            for vehicle_id, vehicle in getattr(env, "vehicles", {}).items()
            if int(vehicle_id) in rewarded_vehicle_ids
            if int(vehicle.get("type", 1)) == 1
        )
        reward_aev = sum(
            float(rewards.get(vehicle_id, 0.0))
            for vehicle_id, vehicle in getattr(env, "vehicles", {}).items()
            if int(vehicle_id) in rewarded_vehicle_ids
            if int(vehicle.get("type", 1)) == 2
        )
        next_state = StateSnapshotBuilder.build(env)
        ledger = build_reward_ledger(env, pending, rewards, self.lifecycle)
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
            run_id=pending.run_id,
            cumulative_episode_id=pending.cumulative_episode_id,
            transition_sequence_index=pending.transition_sequence_index,
            previous_transition_id=pending.previous_transition_id,
            next_transition_id=(None if done else pending.next_transition_id),
            request_generation_seed=pending.request_generation_seed,
            vehicle_initialization_seed=pending.vehicle_initialization_seed,
            reward_scope="selected_epoch_actions",
            rewarded_vehicle_ids=rewarded_vehicle_ids,
            continuing_action_edge_ids=continuing_action_edge_ids,
            reward_ledger=ledger,
            committed_aev_edge_ids=pending.committed_aev_edge_ids,
            repair_hold_aev_ids=pending.repair_hold_aev_ids,
            repair_candidate_request_ids=pending.repair_candidate_request_ids,
        )
        if self.replay is not None:
            self.replay.add(transition)
        self.pending = None
        return transition

    @staticmethod
    def _complete_missing_stage_graphs(
        env: Any, pending: PendingTransition
    ) -> None:
        def build(stage_id: int, state: SystemSnapshot):
            matrix = np.zeros((0, 1), dtype=np.float32)
            graph = StateSnapshotBuilder.feasible_graph_from_matrix(
                env,
                [],
                matrix,
                matrix,
                matrix,
                num_requests=0,
                num_stations=0,
                num_zones=0,
                stage_id=stage_id,
                solver_backend=pending.solver_backend,
                state=state,
            )
            selected = StateSnapshotBuilder.selected_edge_ids(graph, {})
            return graph.with_selected(selected, status="continuing_only"), (
                JointActionSnapshot.from_graph(graph, selected)
            )

        if pending.mode in {"integrated", "integrated_repair"}:
            if pending.ev_stage_graph is None or pending.ev_joint_action is None:
                pending.ev_stage_graph, pending.ev_joint_action = build(
                    0, pending.pre_state
                )
            if pending.mode == "integrated_repair" and pending.aev_stage_graph is None:
                graph, _ = build(2, pending.residual_state or pending.pre_state)
                # Committed/continuing vehicles belong only to stage 0.
                from dataclasses import replace
                graph = replace(graph, edges=(), selected_edge_ids=())
                pending.aev_stage_graph = graph
                pending.aev_joint_action = JointActionSnapshot.from_graph(graph)
            return
        if pending.mode in {"ev_first", "evfirst"}:
            if pending.ev_stage_graph is None or pending.ev_joint_action is None:
                pending.ev_stage_graph, pending.ev_joint_action = build(
                    1, pending.pre_state
                )
            if pending.aev_stage_graph is None or pending.aev_joint_action is None:
                pending.aev_stage_graph, pending.aev_joint_action = build(
                    2, pending.residual_state or pending.pre_state
                )
            return
        if pending.mode in {"aev_first", "aevfirst"}:
            if pending.ev_stage_graph is None or pending.ev_joint_action is None:
                pending.ev_stage_graph, pending.ev_joint_action = build(
                    2, pending.residual_state or pending.pre_state
                )
            if pending.aev_stage_graph is None or pending.aev_joint_action is None:
                pending.aev_stage_graph, pending.aev_joint_action = build(
                    1, pending.pre_state
                )
