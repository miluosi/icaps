from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.Action import IdleAction, WaitingAction
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction
from src.recourse.coordinator import RecourseCoordinator
from src.recourse.critics import enforce_critic_identity
from src.recourse.lifecycle import RequestLifecycleTracker
from src.recourse.replay import PrioritizedJointReplayBuffer
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.recourse.target_builder import RecourseTargetBuilder, TargetComponents
from src.recourse.types import (
    ActionType,
    FeasibleEdgeSnapshot,
    FeasibleGraphSnapshot,
    JointActionSnapshot,
    OutcomeSummary,
    PlannerMetadata,
    RecourseTransition,
    RejectionOutcomeSnapshot,
    RequestSnapshot,
    SystemSnapshot,
    VehicleSnapshot,
)


def _vehicle(vehicle_type: int, *, location: int = 0) -> dict:
    return {
        "type": vehicle_type,
        "location": location,
        "battery": 0.8,
        "idle_timer": 0,
        "is_online": True,
        "assigned_request": None,
        "passenger_onboard": None,
        "charging_station": None,
        "charging_target": None,
        "target_location": None,
    }


def _request(request_id: int, pickup: int = 1, dropoff: int = 2):
    return SimpleNamespace(
        request_id=request_id,
        pickup=pickup,
        dropoff=dropoff,
        value=8.0,
        final_value=10.0,
        created_time=0.0,
        pickup_deadline=20.0,
        dropoff_deadline=30.0,
        travel_time=2.0,
    )


def _state(epoch: int = 4) -> SystemSnapshot:
    return SystemSnapshot(
        epoch_id=epoch,
        current_time=float(epoch),
        zone_ids=(0, 1, 2, 3),
        vehicles=(
            VehicleSnapshot.from_vehicle(0, _vehicle(1, location=0)),
            VehicleSnapshot.from_vehicle(1, _vehicle(2, location=1)),
        ),
        requests=(RequestSnapshot.from_request(_request(10)),),
        stations=(),
    )


def _single_edge_graph(
    *,
    graph_id: str,
    stage: int,
    vehicle_id: int,
    vehicle_type: int,
    structured: float,
    action_type: ActionType = ActionType.WAIT,
) -> FeasibleGraphSnapshot:
    edge = FeasibleEdgeSnapshot(
        edge_id=f"{graph_id}:edge",
        vehicle_id=vehicle_id,
        vehicle_type=vehicle_type,
        action_type=action_type,
        action_id="wait" if action_type == ActionType.WAIT else "assign_10",
        target_location=vehicle_id,
        post_action_location=vehicle_id,
        request_id=10 if action_type == ActionType.SERVICE else None,
        resource_type="request" if action_type == ActionType.SERVICE else None,
        resource_id=10 if action_type == ActionType.SERVICE else None,
        structured_score=structured,
        collection_score=structured,
        request_value=10.0 if action_type == ActionType.SERVICE else 0.0,
        post_action_duration=1.0,
    )
    return FeasibleGraphSnapshot(
        graph_id=graph_id,
        epoch_id=4,
        stage_id=stage,
        solver_backend="test",
        state=_state(),
        edges=(edge,),
        selected_edge_ids=(edge.edge_id,),
    )


def _transition(
    transition_id: str = "run:episode:0:sequence:0",
    *,
    mode: str = "ev_first",
    ev_graph: FeasibleGraphSnapshot | None = None,
    aev_graph: FeasibleGraphSnapshot | None = None,
    done: bool = True,
) -> RecourseTransition:
    state = _state()
    ev_action = JointActionSnapshot.from_graph(
        ev_graph, ev_graph.selected_edge_ids
    ) if ev_graph else None
    aev_action = JointActionSnapshot.from_graph(
        aev_graph, aev_graph.selected_edge_ids
    ) if aev_graph else None
    return RecourseTransition(
        transition_id=transition_id,
        episode_id=0,
        day_id="day",
        seed=7,
        epoch_id=4,
        mode=mode,
        recourse_variant="r4",
        state_variant="joint_state_separate_critics",
        learner_variant="optimization_anchored_residual",
        solver_backend="test",
        pre_state=state,
        ev_stage_graph=ev_graph,
        ev_joint_action=ev_action,
        rejection_outcome=RejectionOutcomeSnapshot(),
        residual_state=state,
        aev_stage_graph=aev_graph,
        aev_joint_action=aev_action,
        reward_ev=-2.0,
        reward_aev=7.0,
        reward_system=5.0,
        next_state=state,
        elapsed_epochs=1.0,
        done=done,
        planner_metadata=PlannerMetadata(backend="test"),
        outcome_summary=OutcomeSummary(),
        run_id="run",
        cumulative_episode_id=0,
        transition_sequence_index=0,
        next_transition_id=None if done else "run:episode:0:sequence:1",
    )


def _value_function() -> PyTorchChargingValueFunction:
    vf = PyTorchChargingValueFunction(
        grid_size=2,
        num_vehicles=2,
        device="cpu",
        episode_length=10,
        max_requests=10,
        neighbour_number=0,
    )
    vf.state_variant = "joint_state_separate_critics"
    vf.learner_variant = "optimization_anchored_residual"
    vf.recourse_variant = "r4"
    vf.training_step = vf.beta_warmup_steps
    # Start online corrections inside the deployment clip so the test
    # measures both Huber-loss gradients rather than clamp saturation.
    for module in (vf.network, vf.critic2):
        module.net[-1].weight.data.zero_()
        module.net[-1].bias.data.zero_()
    return vf


def _reject(
    tracker: RequestLifecycleTracker,
    request_id: int,
    *,
    epoch: int = 4,
    eligible: bool,
) -> None:
    tracker.record_offer(
        transition_id=f"t-{request_id}",
        epoch_id=epoch,
        request=_request(request_id),
        ev_id=0,
        vehicle=_vehicle(1),
        acceptance_probability=0.1,
        acceptance_uniform=0.9,
        accepted=False,
    )
    tracker.mark_residual(
        request_id, epoch_id=epoch, category="rejected", eligible=eligible
    )


def test_full_residual_bellman_identity_is_13_point_7():
    components = TargetComponents(
        ("next",),
        online_full_value=13.0,
        target_structured_value=10.0,
        target_correction_value=3.0,
    )
    target = RecourseTargetBuilder.correction_bellman_target(
        reward=6.0,
        discount=0.9,
        next_components=components,
        current_structured_value=4.0,
    )
    assert target == pytest.approx(13.7)
    assert target != pytest.approx(4.7)


def test_r4_leader_uses_explicit_aev_target_provider():
    leader = PyTorchChargingValueFunction.__new__(PyTorchChargingValueFunction)
    leader.device = torch.device("cpu")
    seen = []

    def aev_provider(graph, **kwargs):
        seen.append(graph)
        return TargetComponents(("aev",), 17.0, 11.0, 6.0)

    leader.set_follower_target_provider(aev_provider)
    components = leader._r4_follower_components("aev-graph")
    assert seen == ["aev-graph"]
    assert components.target_full_value == 17.0


def test_integrated_edges_route_to_their_own_fleet_critics():
    class ConstantProvider:
        def __init__(self, q1, q2):
            self.q1 = torch.nn.Parameter(torch.tensor(float(q1)))
            self.q2 = torch.nn.Parameter(torch.tensor(float(q2)))
            self.calls = []
            self._graph_cache_key = None
            self._graph_cache = None

        def _edge_correction_tensors(self, graph, edge, *, target_context):
            self.calls.append((edge.vehicle_type, target_context))
            return self.q1, self.q2, torch.tensor(1.0)

    ev = ConstantProvider(2, 3)
    aev = ConstantProvider(5, 7)
    router = PyTorchChargingValueFunction.__new__(PyTorchChargingValueFunction)
    router.device = torch.device("cpu")
    router._joint_critic_router = {1: ev, 2: aev}
    graph = FeasibleGraphSnapshot(
        graph_id="joint",
        epoch_id=4,
        stage_id=0,
        solver_backend="test",
        state=_state(),
        edges=(
            FeasibleEdgeSnapshot("ev", 0, 1, ActionType.WAIT, "wait", 0, 0),
            FeasibleEdgeSnapshot("aev", 1, 2, ActionType.WAIT, "wait", 1, 1),
        ),
    )
    q1, q2, providers = router._selected_correction_tensors(
        graph, ("ev", "aev")
    )
    assert q1.item() == 7.0
    assert q2.item() == 10.0
    assert ev.calls == [(1, False)]
    assert aev.calls == [(2, False)]
    assert set(map(id, providers)) == {id(ev), id(aev)}


def test_one_joint_update_backpropagates_into_both_critics():
    vf = _value_function()
    graph = _single_edge_graph(
        graph_id="aev", stage=2, vehicle_id=1, vehicle_type=2, structured=1.0
    )
    vf.store_recourse_transition(_transition(aev_graph=graph))
    assert vf._train_joint_step(1, ifEV=False) > 0.0
    grad1 = sum(
        float(parameter.grad.abs().sum())
        for parameter in vf.network.parameters()
        if parameter.grad is not None
    )
    grad2 = sum(
        float(parameter.grad.abs().sum())
        for parameter in vf.critic2.parameters()
        if parameter.grad is not None
    )
    assert grad1 > 0.0
    assert grad2 > 0.0


def test_direct_link_lookup_cannot_cross_episode_boundaries():
    graph0 = _single_edge_graph(
        graph_id="episode-0-next", stage=2, vehicle_id=1, vehicle_type=2,
        structured=1.0,
    )
    graph1 = replace(graph0, graph_id="episode-1-next")
    current = replace(_transition(done=False), next_transition_id="ep0-next")
    correct = replace(
        _transition("ep0-next", aev_graph=graph0),
        run_id="run",
        episode_id=0,
        cumulative_episode_id=0,
        transition_sequence_index=1,
    )
    other_episode = replace(
        _transition("ep1-next", aev_graph=graph1),
        run_id="run",
        episode_id=1,
        cumulative_episode_id=1,
        transition_sequence_index=1,
    )
    vf = PyTorchChargingValueFunction.__new__(PyTorchChargingValueFunction)
    vf.joint_replay_buffer = PrioritizedJointReplayBuffer(capacity=4)
    vf.joint_replay_buffer.add(other_episode)
    vf.joint_replay_buffer.add(correct)
    vf.next_transition_link_lookups = 0
    vf.next_transition_link_misses = 0
    assert vf._next_transition_graph(current, fleet="aev").graph_id == "episode-0-next"


def test_station_admission_and_action_durations_match_execution_formula():
    station = SimpleNamespace(
        location=3,
        max_capacity=4,
        current_vehicles=[9],
        charging_queue=[8],
        charging_queue_notarrived=[7],
    )
    env = SimpleNamespace(
        current_time=4.0,
        vehicles={1: _vehicle(2, location=0)},
        active_requests={10: _request(10, pickup=1, dropoff=2)},
        charging_manager=SimpleNamespace(stations={3: station}),
        NUM_LOCATIONS=4,
        grid_size=2,
        decision_mode="ev_first",
        station_queue_capacity=3,
        reserve_inbound_charging_capacity=True,
        charge_duration=7,
        _last_matrix_request_ids=[10],
        _last_matrix_charge_station_ids=[3],
        _last_matrix_zone_target_ids=[],
        value_function=None,
        value_function_ev=None,
        get_travel_time=lambda origin, destination: abs(destination - origin) + 0.5,
        get_distance_km=lambda origin, destination: abs(destination - origin),
    )
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env,
        [1],
        np.ones((1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        num_requests=1,
        num_stations=1,
        num_zones=0,
        stage_id=2,
        solver_backend="test",
    )
    service = next(edge for edge in graph.edges if edge.action_type == ActionType.SERVICE)
    charge = next(edge for edge in graph.edges if edge.action_type == ActionType.CHARGE)
    station_snapshot = graph.state.stations[0]
    # physical 4 + waiting room 3 - (occupied 1 + queued 1 + inbound 1)
    assert station_snapshot.remaining_admission_capacity == 4
    assert charge.resource_capacity == 4
    assert service.post_action_duration == pytest.approx(3.0)
    assert charge.post_action_duration == pytest.approx(10.5)


def test_rollout_verifier_enforces_the_same_exact_one_contract():
    graph = _single_edge_graph(
        graph_id="exact-one", stage=2, vehicle_id=1, vehicle_type=2,
        structured=-5.0,
    )
    selected = RecourseTargetBuilder().project(graph)
    assert selected == graph.selected_edge_ids
    RecourseTargetBuilder.verify_feasible(graph, selected)
    with pytest.raises(AssertionError, match="exactly one"):
        RecourseTargetBuilder.verify_feasible(graph, ())


def test_multi_epoch_rejection_history_and_eligibility_denominators():
    tracker = RequestLifecycleTracker()
    _reject(tracker, 10, eligible=True)
    tracker.mark_residual(10, epoch_id=5, category="unoffered", eligible=False)
    _reject(tracker, 11, eligible=False)
    assert tracker.record_aev_assignment(10, vehicle_id=1, epoch_id=4)
    event = next(
        event for event in tracker.outcome_summary().events if event.request_id == 10
    )
    assert event.first_rejected_epoch == 4
    assert [(row.epoch_id, row.category) for row in event.residual_observations] == [
        (4, "rejected"),
        (5, "unoffered"),
    ]
    metrics = tracker.metrics()
    assert metrics["rejected_residual_count"] == 2
    assert metrics["recovery_rate_assignment"] == pytest.approx(0.5)
    assert metrics["eligible_rejected_residual_count"] == 1
    assert metrics["conditional_recovery_rate_assignment"] == pytest.approx(1.0)


def test_ev_pickup_is_not_counted_as_aev_recovery():
    tracker = RequestLifecycleTracker()
    _reject(tracker, 10, eligible=True)
    assert not tracker.record_pickup(
        10, vehicle_id=0, vehicle_type=1, epoch_id=5
    )
    assert tracker.metrics()["aev_pickup_after_rejection_count"] == 0


def test_waiting_and_relocation_have_distinct_canonical_actions():
    wait = WaitingAction([], (0, 0))
    relocate = IdleAction([], (0, 0), (1, 0))
    assert wait.metadata.canonical_type == ActionType.WAIT
    assert wait.learning_action_type == "wait"
    assert relocate.metadata.canonical_type == ActionType.RELOCATE
    assert relocate.learning_action_type == "reloc"


def test_replay_restores_rng_and_beta_schedule_exactly():
    replay = PrioritizedJointReplayBuffer(
        capacity=8, seed=123, beta=0.2, beta_end=0.8, beta_anneal_steps=4
    )
    for index in range(4):
        replay.add(
            replace(
                _transition(f"t-{index}"),
                transition_sequence_index=index,
            ),
            td_error=float(index + 1),
        )
    replay.sample(2)
    state = replay.state_dict()
    assert len(state["content_hash"]) == 64
    expected = replay.sample(3)
    resumed = PrioritizedJointReplayBuffer(capacity=1)
    resumed.load_state_dict(state)
    actual = resumed.sample(3)
    assert actual.indices == expected.indices
    assert actual.weights == pytest.approx(expected.weights)
    assert resumed.beta == pytest.approx(replay.beta)
    resumed.update_priorities((0,), (2.5,))
    assert resumed.priorities[0] == pytest.approx(2.5 + resumed.epsilon)
    corrupted = dict(state)
    corrupted["priorities"] = list(state["priorities"])
    corrupted["priorities"][0] += 1.0
    with pytest.raises(ValueError, match="content hash"):
        resumed.load_state_dict(corrupted)


def test_checkpoint_variant_mismatch_fails_instead_of_overriding_request():
    vf = _value_function()
    with pytest.raises(ValueError, match="checkpoint state_variant mismatch"):
        vf.load_extra_checkpoint_state(
            {
                "state_variant": "joint_state_shared_critic",
                "learner_variant": vf.learner_variant,
                "recourse_variant": vf.recourse_variant,
            }
        )


def test_shared_checkpoint_pair_preserves_object_identity():
    loaded = _value_function()
    loaded.state_variant = "joint_state_shared_critic"
    loaded.load_extra_checkpoint_state(
        {
            "state_variant": "joint_state_shared_critic",
            "learner_variant": loaded.learner_variant,
            "recourse_variant": loaded.recourse_variant,
        }
    )
    unused_second_instance = _value_function()
    aev, ev = enforce_critic_identity(
        loaded,
        unused_second_instance,
        state_variant="joint_state_shared_critic",
    )
    env = SimpleNamespace(value_function=aev, value_function_ev=ev)
    assert env.value_function is env.value_function_ev


def test_collection_to_r4_gradient_path_uses_full_aev_target():
    env = SimpleNamespace(
        current_time=4.0,
        current_epoch=4,
        vehicles={0: _vehicle(1, location=0), 1: _vehicle(2, location=1)},
        active_requests={10: _request(10)},
        charging_manager=SimpleNamespace(stations={}),
        NUM_LOCATIONS=4,
        grid_size=2,
        decision_mode="ev_first",
        initial_random_seed=7,
        cumulative_episode_index=0,
        recourse_run_id="deterministic-test",
        request_generation_seed=101,
        vehicle_initialization_seed=202,
    )
    coordinator = RecourseCoordinator()
    pending = coordinator.begin(
        env,
        mode="ev_first",
        recourse_variant="r4",
        state_variant="joint_state_separate_critics",
        learner_variant="optimization_anchored_residual",
        solver_backend="test",
    )
    ev_graph = _single_edge_graph(
        graph_id="ev-stage", stage=1, vehicle_id=0, vehicle_type=1,
        structured=4.0, action_type=ActionType.SERVICE,
    )
    aev_graph = _single_edge_graph(
        graph_id="aev-stage", stage=2, vehicle_id=1, vehicle_type=2,
        structured=10.0, action_type=ActionType.SERVICE,
    )
    pending.ev_stage_graph = ev_graph
    pending.ev_joint_action = JointActionSnapshot.from_graph(
        ev_graph, ev_graph.selected_edge_ids
    )
    coordinator.lifecycle.record_offer(
        transition_id=pending.transition_id,
        epoch_id=4,
        request=env.active_requests[10],
        ev_id=0,
        vehicle=env.vehicles[0],
        acceptance_probability=0.0,
        acceptance_uniform=0.5,
        accepted=False,
    )
    coordinator.lifecycle.mark_residual(
        10, epoch_id=4, category="rejected", eligible=True
    )
    assert coordinator.lifecycle.record_aev_assignment(
        10, vehicle_id=1, vehicle_type=2, epoch_id=4
    )
    pending.residual_state = StateSnapshotBuilder.build(
        env, request_labels={10: "rejected"}
    )
    pending.aev_stage_graph = aev_graph
    pending.aev_joint_action = JointActionSnapshot.from_graph(
        aev_graph, aev_graph.selected_edge_ids
    )
    transition = coordinator.finalize(env, rewards={0: -2.0, 1: 7.0}, done=True)
    assert transition is not None
    assert transition.rewarded_vehicle_ids == (0, 1)
    assert transition.reward_system == transition.reward_ev + transition.reward_aev
    assert transition.rejection_outcome.rejected_request_ids == (10,)

    ev_vf = _value_function()
    aev_vf = _value_function()
    ev_vf.set_joint_critic_router(ev_value_function=ev_vf, aev_value_function=aev_vf)
    aev_vf.set_joint_critic_router(ev_value_function=ev_vf, aev_value_function=aev_vf)
    ev_vf.set_follower_target_provider(aev_vf.target_components_for_graph)
    for module in (aev_vf.target_network, aev_vf.target_critic2):
        for parameter in module.parameters():
            parameter.data.zero_()
    base_follower = aev_vf.target_components_for_graph(aev_graph).target_full_value
    aev_vf.target_network.net[-1].bias.data.fill_(1.0)
    aev_vf.target_critic2.net[-1].bias.data.fill_(1.0)
    perturbed_follower = aev_vf.target_components_for_graph(aev_graph).target_full_value
    assert perturbed_follower > base_follower

    ev_vf.store_recourse_transition(transition)
    before_q1 = [parameter.detach().clone() for parameter in ev_vf.network.parameters()]
    before_q2 = [parameter.detach().clone() for parameter in ev_vf.critic2.parameters()]
    assert ev_vf._train_joint_step(1, ifEV=True) > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_q1, ev_vf.network.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_q2, ev_vf.critic2.parameters())
    )
    diagnostic = ev_vf.joint_training_diagnostics[-1]
    assert diagnostic["r4_follower_full_value"] == pytest.approx(perturbed_follower)
    assert diagnostic["joint_target_full"] == pytest.approx(-2.0 + perturbed_follower)


def test_transition_and_offer_ids_are_reproducible():
    def collect_ids():
        env = SimpleNamespace(
            current_time=4.0,
            vehicles={0: _vehicle(1)},
            active_requests={10: _request(10)},
            charging_manager=SimpleNamespace(stations={}),
            NUM_LOCATIONS=4,
            decision_mode="ev_first",
            initial_random_seed=3,
            cumulative_episode_index=9,
            recourse_run_id="stable-run",
        )
        coordinator = RecourseCoordinator()
        pending = coordinator.begin(env, mode="ev_first")
        offer = coordinator.lifecycle.record_offer(
            transition_id=pending.transition_id,
            epoch_id=4,
            request=env.active_requests[10],
            ev_id=0,
            vehicle=env.vehicles[0],
            acceptance_probability=0.5,
            acceptance_uniform=0.7,
            accepted=False,
        )
        return pending.transition_id, pending.next_transition_id, offer.offer_id

    assert collect_ids() == collect_ids()
