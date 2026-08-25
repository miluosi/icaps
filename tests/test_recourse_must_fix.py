from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.Action import IdleAction
from src.GurobiOptimizer import GurobiOptimizer
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction
from src.recourse.critics import wire_recourse_critics
from src.recourse.lifecycle import RequestLifecycleTracker
from src.recourse.replay import PrioritizedJointReplayBuffer
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.recourse.target_builder import RecourseTargetBuilder
from src.recourse.training import training_readiness
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


def _request(request_id=10, *, pickup=1, dropoff=2, value=10.0):
    return SimpleNamespace(
        request_id=request_id,
        pickup=pickup,
        dropoff=dropoff,
        value=value,
        final_value=value,
        created_time=0.0,
        pickup_deadline=20.0,
        dropoff_deadline=30.0,
        travel_time=float(abs(dropoff - pickup) + 1),
        trip_distance_km=float(abs(dropoff - pickup)),
    )


def _vehicle(vehicle_type, *, location=0):
    return {
        "type": vehicle_type,
        "location": location,
        "battery": 0.9,
        "idle_timer": 0,
        "is_online": True,
        "assigned_request": None,
        "passenger_onboard": None,
        "charging_station": None,
        "charging_target": None,
        "target_location": None,
        "stationary_duration": 0,
    }


def _state(epoch=0):
    return SystemSnapshot(
        epoch_id=epoch,
        current_time=float(epoch),
        zone_ids=(0, 1, 2),
        vehicles=(
            VehicleSnapshot.from_vehicle(0, _vehicle(1)),
            VehicleSnapshot.from_vehicle(1, _vehicle(2, location=1)),
        ),
        requests=(RequestSnapshot.from_request(_request()),),
        stations=(),
    )


def _graph(graph_id, *, stage, vehicle_id, vehicle_type, structured=1.0):
    edge = FeasibleEdgeSnapshot(
        edge_id=f"{graph_id}:edge",
        vehicle_id=vehicle_id,
        vehicle_type=vehicle_type,
        action_type=ActionType.WAIT,
        action_id="wait",
        target_location=vehicle_id,
        post_action_location=vehicle_id,
        structured_score=structured,
        collection_score=structured,
        post_action_duration=1.0,
    )
    return FeasibleGraphSnapshot(
        graph_id=graph_id,
        epoch_id=0,
        stage_id=stage,
        solver_backend="test",
        state=_state(),
        edges=(edge,),
        selected_edge_ids=(edge.edge_id,),
    )


def _transition(
    transition_id="run:episode:0:sequence:0",
    *,
    sequence=0,
    mode="ev_first",
    ev_graph=None,
    aev_graph=None,
    done=True,
    next_id=None,
    reward=8.0,
    recourse_variant="r3",
):
    state = _state(sequence)
    return RecourseTransition(
        transition_id=transition_id,
        episode_id=0,
        day_id="day",
        seed=1,
        epoch_id=sequence,
        mode=mode,
        recourse_variant=recourse_variant,
        state_variant="joint_state_shared_critic",
        learner_variant="optimization_anchored_residual",
        solver_backend="test",
        pre_state=state,
        ev_stage_graph=ev_graph,
        ev_joint_action=JointActionSnapshot.from_graph(ev_graph),
        rejection_outcome=RejectionOutcomeSnapshot(),
        residual_state=state,
        aev_stage_graph=aev_graph,
        aev_joint_action=JointActionSnapshot.from_graph(aev_graph),
        reward_ev=reward,
        reward_aev=reward,
        reward_system=2.0 * reward,
        next_state=state,
        elapsed_epochs=1.0,
        done=done,
        planner_metadata=PlannerMetadata(backend="test"),
        outcome_summary=OutcomeSummary(),
        run_id="run",
        cumulative_episode_id=0,
        transition_sequence_index=sequence,
        next_transition_id=next_id,
    )


def _vf(*, state_variant="joint_state_shared_critic", recourse="r3"):
    value_function = PyTorchChargingValueFunction(
        grid_size=2,
        num_vehicles=2,
        device="cpu",
        episode_length=10,
        max_requests=10,
        neighbour_number=0,
    )
    value_function.state_variant = state_variant
    value_function.learner_variant = "optimization_anchored_residual"
    value_function.recourse_variant = recourse
    for module in (value_function.network, value_function.critic2):
        module.net[-1].weight.data.zero_()
        module.net[-1].bias.data.zero_()
    return value_function


def test_production_wiring_routes_fleets_and_shares_transition_payload():
    aev = _vf(state_variant="joint_state_separate_critics", recourse="r4")
    ev = _vf(state_variant="joint_state_separate_critics", recourse="r4")
    aev, ev = wire_recourse_critics(
        aev, ev, state_variant="joint_state_separate_critics"
    )

    assert aev._joint_critic_router == {1: ev, 2: aev}
    assert ev._joint_critic_router == {1: ev, 2: aev}
    assert ev._follower_target_provider.__self__ is aev
    assert aev.joint_replay_buffer is ev.joint_replay_buffer
    assert ev._owns_joint_replay_payload is False


def test_joint_replay_readiness_is_independent_of_empty_edge_replay():
    vf = _vf()
    graph = _graph("aev", stage=2, vehicle_id=1, vehicle_type=2)
    vf.store_recourse_transition(_transition(aev_graph=graph))

    readiness = training_readiness(
        vf, ifEV=False, edge_warmup=64
    )
    assert not readiness.edge_ready
    assert readiness.joint_ready
    assert vf.train_step(batch_size=1, ifEV=False) > 0.0


def test_missing_successor_is_a_true_noop_until_link_arrives():
    vf = _vf()
    graph = _graph("current-aev", stage=2, vehicle_id=1, vehicle_type=2)
    current = _transition(
        aev_graph=graph,
        done=False,
        next_id="run:episode:0:sequence:1",
    )
    vf.store_recourse_transition(current)
    priorities_before = vf.joint_replay_buffer.priorities
    parameters_before = [
        parameter.detach().clone() for parameter in vf.network.parameters()
    ]

    assert vf._train_joint_step(1, ifEV=False) == 0.0
    assert vf.joint_replay_buffer.beta_step == 0
    assert vf.joint_replay_buffer.priorities == priorities_before
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, vf.network.parameters())
    )

    next_ev = _graph("next-ev", stage=1, vehicle_id=0, vehicle_type=1)
    successor = _transition(
        "run:episode:0:sequence:1",
        sequence=1,
        ev_graph=next_ev,
        done=True,
    )
    vf.store_recourse_transition(successor)
    assert vf.has_trainable_joint_rows(ifEV=False)
    assert vf._train_joint_step(2, ifEV=False) > 0.0
    assert vf.joint_replay_buffer.beta_step == 1


def test_ev_first_follower_bootstraps_to_next_ev_leader_graph():
    vf = _vf()
    current_graph = _graph(
        "current-aev", stage=2, vehicle_id=1, vehicle_type=2
    )
    current = _transition(
        aev_graph=current_graph,
        done=False,
        next_id="run:episode:0:sequence:1",
    )
    next_ev = _graph("next-ev-13", stage=1, vehicle_id=0, vehicle_type=1)
    next_aev = _graph("next-aev-97", stage=2, vehicle_id=1, vehicle_type=2)
    successor = _transition(
        "run:episode:0:sequence:1",
        sequence=1,
        ev_graph=next_ev,
        aev_graph=next_aev,
    )
    vf.store_recourse_transition(current)
    vf.store_recourse_transition(successor)

    selected = vf._temporal_successor_graph(current)
    assert selected is next_ev
    assert selected.graph_id == "next-ev-13"


def test_raw_joint_critic_updates_while_deployment_beta_is_zero():
    vf = _vf()
    assert vf._beta() == 0.0
    graph = _graph("raw-aev", stage=2, vehicle_id=1, vehicle_type=2)
    vf.store_recourse_transition(
        _transition(aev_graph=graph, reward=100.0)
    )
    before = [parameter.detach().clone() for parameter in vf.network.parameters()]

    assert vf._train_joint_step(1, ifEV=False) > 0.0
    assert vf.joint_training_step == 1
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, vf.network.parameters())
    )


def test_aev_first_updates_both_declared_stages_instead_of_silent_skip():
    vf = _vf(recourse="legacy")
    aev_graph = _graph("aev-leader", stage=1, vehicle_id=1, vehicle_type=2)
    ev_graph = _graph("ev-follower", stage=2, vehicle_id=0, vehicle_type=1)
    transition = _transition(
        mode="aev_first",
        aev_graph=aev_graph,
        ev_graph=ev_graph,
        recourse_variant="legacy",
    )
    vf.store_recourse_transition(transition)

    assert vf._train_joint_step(1, ifEV=False) > 0.0
    assert vf._train_joint_step(1, ifEV=True) > 0.0


def test_continuing_service_edges_serialize_value_phase_and_remaining_work():
    first = _vehicle(2, location=0)
    second = _vehicle(2, location=0)
    first["assigned_request"] = 10
    second["passenger_onboard"] = 11
    env = SimpleNamespace(
        current_time=3.0,
        vehicles={0: first, 1: second},
        active_requests={
            10: _request(10, pickup=1, dropoff=2, value=5.0),
            11: _request(11, pickup=0, dropoff=4, value=50.0),
        },
        charging_manager=SimpleNamespace(stations={}),
        NUM_LOCATIONS=5,
        grid_size=3,
        decision_mode="integrated",
        get_travel_time=lambda origin, destination: abs(destination - origin),
        get_distance_km=lambda origin, destination: abs(destination - origin),
        value_function=None,
        value_function_ev=None,
    )
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env,
        [],
        np.zeros((0, 1), dtype=np.float32),
        np.zeros((0, 1), dtype=np.float32),
        np.zeros((0, 1), dtype=np.float32),
        num_requests=0,
        num_stations=0,
        num_zones=0,
        stage_id=0,
        solver_backend="test",
    )
    edges = {edge.vehicle_id: edge for edge in graph.edges}

    assert edges[0].request_value == 5.0
    assert edges[1].request_value == 50.0
    assert dict(edges[0].metadata)["service_phase"] == "to_pickup"
    assert dict(edges[1].metadata)["service_phase"] == "passenger_onboard"
    assert edges[0].post_action_duration != edges[1].post_action_duration


def test_target_projection_uses_rollout_integer_grid_for_near_ties():
    state = _state()
    edges = (
        FeasibleEdgeSnapshot(
            "first", 0, 1, ActionType.RELOCATE, "reloc_0", 0, 0
        ),
        FeasibleEdgeSnapshot(
            "second", 0, 1, ActionType.RELOCATE, "reloc_1", 1, 1
        ),
    )
    graph = FeasibleGraphSnapshot(
        "near-tie",
        0,
        0,
        "exact",
        state,
        edges,
        objective_cost_scale=10_000,
    )
    selected = RecourseTargetBuilder().project(
        graph, {"first": 1.000041, "second": 1.000039}
    )
    assert selected == ("first",)


def test_typed_action_metadata_overrides_legacy_idle_storage_string():
    vf = _vf()
    wait = IdleAction([], (0, 0), (0, 0))
    vf.set_replay_collection_context(wait)
    vf.store_experience(vehicle_id=0, action_type="idle")
    relocate = IdleAction([], (0, 0), (1, 0))
    vf.set_replay_collection_context(relocate)
    vf.store_experience(vehicle_id=0, action_type="idle")

    assert vf.experience_buffer[-2]["action_type_id"] == int(ActionType.WAIT)
    assert vf.experience_buffer[-1]["action_type_id"] == int(ActionType.RELOCATE)


def test_lifecycle_creates_one_event_per_rejection_epoch_and_separates_ev_rescue():
    tracker = RequestLifecycleTracker()
    request = _request(10)
    vehicle = _vehicle(1)
    for epoch in (4, 5):
        tracker.record_offer(
            transition_id=f"t-{epoch}",
            epoch_id=epoch,
            request=request,
            ev_id=0,
            vehicle=vehicle,
            acceptance_probability=0.1,
            acceptance_uniform=0.9,
            accepted=False,
        )
        tracker.mark_residual(
            10, epoch_id=epoch, category="rejected", eligible=True
        )
    assert tracker.record_aev_assignment(10, vehicle_id=1, epoch_id=5)
    assert not tracker.record_completion(
        10, epoch_id=6, vehicle_id=0, vehicle_type=1
    )

    events = tracker.outcome_summary().events
    assert [event.epoch_id for event in events] == [4, 5]
    assert events[1].same_epoch_recourse_link
    assert not events[1].completed
    assert events[1].ultimately_served
    metrics = tracker.metrics()
    assert metrics["rejected_residual_count"] == 2
    assert metrics["same_epoch_aev_assignment_count"] == 1
    assert metrics["completion_after_rejection_count"] == 0
    assert metrics["ultimately_unserved_count"] == 0
    assert metrics["mean_completion_recovery_delay"] == 0.0


def test_per_bonus_persists_and_duplicate_transition_ids_are_rejected():
    tracker = RequestLifecycleTracker()
    tracker.record_offer(
        transition_id="t",
        epoch_id=0,
        request=_request(),
        ev_id=0,
        vehicle=_vehicle(1),
        acceptance_probability=0.1,
        acceptance_uniform=0.9,
        accepted=False,
    )
    tracker.mark_residual(10, epoch_id=0, category="rejected", eligible=True)
    tracker.record_aev_assignment(10, vehicle_id=1, epoch_id=0)
    transition = replace(
        _transition(),
        rejection_outcome=tracker.rejection_outcome(),
        outcome_summary=tracker.outcome_summary(),
    )
    replay = PrioritizedJointReplayBuffer(
        capacity=4, rejection_bonus=2.0, recourse_bonus=3.0
    )
    replay.add(transition, td_error=1.0)
    replay.update_priorities((0,), (2.0,))

    assert replay.priorities[0] == pytest.approx(
        2.0 + replay.epsilon + 2.0 + 3.0
    )
    with pytest.raises(ValueError, match="duplicate transition id"):
        replay.add(transition)


def test_ev_first_heuristic_cannot_select_matrix_infeasible_high_q_request():
    requests = {
        10: _request(10, pickup=1, dropoff=2),
        11: _request(11, pickup=2, dropoff=3),
    }
    env = SimpleNamespace(
        vehicles={0: _vehicle(1)},
        active_requests=requests,
        charging_manager=SimpleNamespace(stations={}),
        battery_consum=0.01,
        min_battery_level=0.1,
        _last_matrix_request_ids=[10, 11],
        _last_matrix_charge_station_ids=[],
        _last_matrix_zone_indices=[],
        _last_matrix_zone_target_ids=[],
        _last_matrix_num_requests=2,
        _last_matrix_num_stations=0,
        _last_matrix_num_zones=0,
        _manhattan_distance_loc=lambda origin, destination: abs(
            destination - origin
        ),
        get_distance_km=lambda origin, destination: abs(destination - origin),
    )
    optimizer = GurobiOptimizer.__new__(GurobiOptimizer)
    optimizer.env = env
    assignments = optimizer._heuristic_assignment_fastqvalue_evfirst(
        [0],
        [],
        np.asarray([[0, 1, 1]], dtype=np.int8),
        np.asarray([[100.0, 1.0, 0.0]], dtype=np.float32),
    )
    assert assignments[0].request_id == 11


def test_filtered_nyc_zone_index_maps_executed_idle_action_to_graph_edge():
    env = SimpleNamespace(
        _last_matrix_request_ids=[],
        _last_matrix_charge_station_ids=[],
        _last_matrix_zone_indices=[25],
        _last_matrix_zone_target_ids=[2],
        mcmf_cost_scale=10_000,
        grid_size=2,
        value_function=None,
        value_function_ev=None,
        get_distance_km=lambda origin, destination: abs(destination - origin),
        get_travel_time=lambda origin, destination: abs(destination - origin),
    )
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env,
        [1],
        np.asarray([[1, 1]], dtype=np.int8),
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        num_requests=0,
        num_stations=0,
        num_zones=1,
        stage_id=2,
        solver_backend="heuristic",
        state=replace(
            _state(), exogenous_context=(("decision_mode", "evfirst"),)
        ),
    )

    selected = StateSnapshotBuilder.selected_edge_ids(
        graph, {1: "idle_at_25"}
    )
    assert len(selected) == 1
    selected_edge = next(edge for edge in graph.edges if edge.edge_id == selected[0])
    assert dict(selected_edge.metadata)["zone_index"] == 25
