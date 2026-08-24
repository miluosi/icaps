from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.Environment import ChargingIntegratedEnvironment
from src.NYCEnvironment import NYCEnvironment
from src.recourse.lifecycle import RequestLifecycleTracker
from src.recourse.metrics import summarize_metric_with_uncertainty
from src.recourse.manifest import write_experiment_manifest
from src.recourse.replay import PrioritizedJointReplayBuffer
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.recourse.target_builder import RecourseTargetBuilder
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
from src.value_function_registry import (
    VALUE_FUNCTION_CHOICES,
    get_value_function_class,
    validate_value_function_registry,
)


def _request(request_id=10):
    return SimpleNamespace(
        request_id=request_id,
        pickup=2,
        dropoff=3,
        value=5.0,
        final_value=7.0,
        created_time=0.0,
        pickup_deadline=20.0,
        dropoff_deadline=30.0,
        travel_time=4.0,
    )


def _vehicle(vehicle_type=1):
    return {
        "type": vehicle_type,
        "location": 1,
        "battery": 0.8,
        "idle_timer": 2,
        "is_online": True,
        "assigned_request": None,
        "passenger_onboard": None,
        "charging_station": None,
        "charging_target": None,
        "target_location": None,
    }


def _state(epoch=4):
    return SystemSnapshot(
        epoch_id=epoch,
        current_time=float(epoch),
        zone_ids=(0, 1),
        vehicles=(
            VehicleSnapshot.from_vehicle(0, _vehicle(1)),
            VehicleSnapshot.from_vehicle(1, _vehicle(2)),
        ),
        requests=(RequestSnapshot.from_request(_request()),),
        stations=(),
    )


def _graph(edges, stage=1):
    return FeasibleGraphSnapshot(
        graph_id=f"g{stage}",
        epoch_id=4,
        stage_id=stage,
        solver_backend="test",
        state=_state(),
        edges=tuple(edges),
    )


def _transition(*, rejected=False, assigned=False):
    tracker = RequestLifecycleTracker()
    if rejected:
        tracker.record_offer(
            transition_id="t1",
            epoch_id=4,
            request=_request(),
            ev_id=0,
            vehicle=_vehicle(1),
            acceptance_probability=0.2,
            acceptance_uniform=0.1,
            accepted=False,
        )
        tracker.mark_residual(10, epoch_id=4, category="rejected", eligible=True)
        if assigned:
            tracker.record_aev_assignment(10, vehicle_id=1, epoch_id=4)
    state = _state()
    return RecourseTransition(
        transition_id="t1",
        episode_id=0,
        day_id="d",
        seed=1,
        epoch_id=4,
        mode="ev_first",
        recourse_variant="r4",
        state_variant="joint_state_separate_critics",
        learner_variant="optimization_anchored_residual",
        solver_backend="test",
        pre_state=state,
        ev_stage_graph=None,
        ev_joint_action=None,
        rejection_outcome=tracker.rejection_outcome(transition_id="t1"),
        residual_state=state,
        aev_stage_graph=None,
        aev_joint_action=None,
        reward_ev=-2.0,
        reward_aev=7.0,
        reward_system=5.0,
        next_state=state,
        elapsed_epochs=1.0,
        done=False,
        planner_metadata=PlannerMetadata(backend="test"),
        outcome_summary=tracker.outcome_summary(epoch_id=4),
    )


def test_lifecycle_tracks_assignment_pickup_completion_without_double_counting():
    tracker = RequestLifecycleTracker()
    for _ in range(2):
        tracker.record_offer(
            transition_id="t",
            epoch_id=5,
            request=_request(),
            ev_id=0,
            vehicle=_vehicle(1),
            acceptance_probability=0.1,
            acceptance_uniform=0.01,
            accepted=False,
        )
    tracker.mark_residual(10, epoch_id=5, category="rejected", eligible=True)
    assert tracker.record_aev_assignment(10, vehicle_id=1, epoch_id=5)
    assert not tracker.record_aev_assignment(10, vehicle_id=1, epoch_id=5)
    assert tracker.record_pickup(10, vehicle_id=1, epoch_id=6)
    assert tracker.record_completion(10, epoch_id=8)
    metrics = tracker.metrics()
    assert metrics["ev_offer_count"] == 2
    assert metrics["rejected_residual_count"] == 1
    assert metrics["same_epoch_aev_assignment_count"] == 1
    assert metrics["aev_pickup_after_rejection_count"] == 1
    assert metrics["completion_after_rejection_count"] == 1
    assert metrics["mean_completion_recovery_delay"] == 3.0
    tracker.assert_reconciled()


def test_lifecycle_excludes_later_assignment_and_unoffered_request_from_recourse():
    tracker = RequestLifecycleTracker()
    tracker.record_offer(
        transition_id="t",
        epoch_id=5,
        request=_request(10),
        ev_id=0,
        vehicle=_vehicle(1),
        acceptance_probability=0.1,
        acceptance_uniform=0.01,
        accepted=False,
    )
    tracker.mark_residual(10, epoch_id=5, category="rejected", eligible=True)
    tracker.mark_residual(11, epoch_id=5, category="unoffered", eligible=True)
    assert not tracker.record_aev_assignment(10, vehicle_id=1, epoch_id=6)
    assert not tracker.record_aev_assignment(11, vehicle_id=1, epoch_id=5)
    tracker.record_expiry(10)
    metrics = tracker.metrics()
    assert metrics["same_epoch_aev_assignment_count"] == 0
    assert metrics["unoffered_residual_count"] == 1
    assert metrics["unrecovered_rejected_count"] == 1


def test_epoch_matching_uses_integer_epoch_not_float_simulator_time():
    tracker = RequestLifecycleTracker()
    tracker.record_offer(
        transition_id="t",
        epoch_id=7,
        request=_request(),
        ev_id=0,
        vehicle=_vehicle(1),
        acceptance_probability=0.1,
        acceptance_uniform=0.01,
        accepted=False,
    )
    tracker.mark_residual(10, epoch_id=7, category="rejected", eligible=True)
    assert tracker.record_aev_assignment(10, vehicle_id=1, epoch_id=7)


def test_snapshots_and_replay_are_immutable_and_round_trip(tmp_path):
    state = _state()
    with pytest.raises(FrozenInstanceError):
        state.current_time = 99.0
    transition = _transition(rejected=True, assigned=True)
    replay = PrioritizedJointReplayBuffer(capacity=4)
    replay.add(transition, td_error=3.0)
    path = tmp_path / "recourse-replay.pkl"
    replay.save(path)
    loaded = PrioritizedJointReplayBuffer.load(path)
    assert tuple(loaded)[0] == transition
    assert loaded.priorities == replay.priorities
    incompatible = replay.state_dict()
    incompatible["schema_version"] = 999
    with pytest.raises(ValueError, match="schema"):
        loaded.load_state_dict(incompatible)


def test_state_snapshot_is_not_changed_by_live_environment_mutation():
    live_vehicle = _vehicle(1)
    env = SimpleNamespace(
        current_time=4.0,
        vehicles={0: live_vehicle},
        active_requests={10: _request()},
        charging_manager=SimpleNamespace(stations={}),
        NUM_LOCATIONS=4,
        decision_mode="ev_first",
    )
    from src.recourse.state_snapshot import StateSnapshotBuilder

    snapshot = StateSnapshotBuilder.build(env)
    live_vehicle["location"] = 99
    env.active_requests[10].pickup = 88
    assert snapshot.vehicles[0].location == 1
    assert snapshot.requests[0].pickup == 2


def test_feasible_graph_persists_queue_and_post_demand_features():
    from src.recourse.state_snapshot import StateSnapshotBuilder

    class SnapshotAwareValueFunction:
        def _queue_features(self, **kwargs):
            return [float(len(env.charging_manager.stations[3].charging_queue))] * 9

        def predict_post_action_demand(self, **kwargs):
            return np.asarray(kwargs["current_zone_demands"], dtype=np.float32) / 10.0

    station = SimpleNamespace(
        location=2,
        max_capacity=2,
        current_vehicles=[],
        charging_queue=[9],
        charging_queue_notarrived=[],
    )
    env = SimpleNamespace(
        current_time=4.0,
        vehicles={1: _vehicle(2)},
        active_requests={10: _request()},
        charging_manager=SimpleNamespace(stations={3: station}),
        NUM_LOCATIONS=4,
        grid_size=2,
        charge_duration=3,
        decision_mode="ev_first",
        _last_matrix_request_ids=[],
        _last_matrix_charge_station_ids=[3],
        _last_matrix_zone_target_ids=[],
    )
    env.value_function = SnapshotAwareValueFunction()
    env.value_function_ev = None
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env,
        [1],
        np.asarray([[1.0]]),
        np.asarray([[2.0]]),
        np.asarray([[1.0]]),
        num_requests=0,
        num_stations=1,
        num_zones=0,
        stage_id=2,
        solver_backend="test",
    )
    edge = graph.edges[0]
    assert edge.queue_features == (1.0,) * 9
    assert edge.post_demand_feature == pytest.approx(0.1)
    station.charging_queue.extend([10, 11])
    env.active_requests[10].pickup = 1
    assert graph.edges[0].queue_features == (1.0,) * 9
    assert graph.edges[0].post_demand_feature == pytest.approx(0.1)


def test_r4_stage_coupling_and_terminal_within_epoch_value():
    builder = RecourseTargetBuilder()
    assert builder.leader_target(
        variant="r4",
        reward_ev=-2,
        follower_value=7,
        temporal_value=100,
        done=False,
    ) == 5
    assert builder.leader_target(
        variant="r4",
        reward_ev=-2,
        follower_value=7,
        temporal_value=100,
        done=True,
    ) == 5
    r3 = builder.leader_target(
        variant="r3",
        reward_ev=-2,
        follower_value=7,
        temporal_value=0,
        done=False,
    )
    assert r3 == -2


def test_r0_r4_policies_are_explicit_and_behaviorally_distinct():
    builder = RecourseTargetBuilder()
    policies = {name: builder.variant_policy(name) for name in ("r0", "r1", "r2", "r3", "r4")}
    assert policies["r0"].rejection_enabled is False
    assert policies["r1"].rejection_enabled is True
    assert policies["r1"].same_epoch_repair is False
    assert policies["r2"].same_epoch_repair is True
    assert policies["r2"].structured_only_follower is True
    assert policies["r3"].learned_follower is True
    assert policies["r4"].learned_follower is True
    assert policies["r3"].stage_coupled_leader is False
    assert policies["r4"].stage_coupled_leader is True


def test_terminal_edge_replay_retains_r4_within_epoch_follower_value():
    from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction

    value_function = PyTorchChargingValueFunction.__new__(PyTorchChargingValueFunction)
    value_function.device = torch.device("cpu")
    value_function.target_network = None
    value_function.target_critic2 = None
    value_function._solver_consistent_residual_value = lambda graph: (
        torch.tensor(7.0),
        ("aev-edge",),
    )
    value_function._last_actor_loss_tensor = torch.tensor(0.0)
    value_function._last_alpha_term_tensor = torch.tensor(0.0)
    leader_action = JointActionSnapshot(
        stage_id=1,
        selected_edge_ids=("ev-edge",),
        structured_value=0.0,
        collection_value=0.0,
    )
    values = value_function._next_soft_values(
        [
            {
                "stage_id": 1,
                "recourse_variant": "r4",
                "is_system_done": True,
                "aev_stage_graph": object(),
                "joint_action_snapshot": leader_action,
            }
        ]
    )
    assert values.item() == 7.0


def test_joint_projection_respects_requests_stations_and_double_q():
    edges = (
        FeasibleEdgeSnapshot("v0_req", 0, 1, ActionType.SERVICE, "assign_10", 2, 3, request_id=10, resource_type="request", resource_id=10, resource_capacity=1, structured_score=9, collection_score=4),
        FeasibleEdgeSnapshot("v1_req", 1, 2, ActionType.SERVICE, "assign_10", 2, 3, request_id=10, resource_type="request", resource_id=10, resource_capacity=1, structured_score=8, collection_score=7),
        FeasibleEdgeSnapshot("v0_station", 0, 1, ActionType.CHARGE, "charge_1", 1, 1, station_id=1, resource_type="station", resource_id=1, resource_capacity=1, structured_score=1, collection_score=6),
        FeasibleEdgeSnapshot("v1_station", 1, 2, ActionType.CHARGE, "charge_1", 1, 1, station_id=1, resource_type="station", resource_id=1, resource_capacity=1, structured_score=2, collection_score=5),
    )
    graph = _graph(edges, stage=2)
    builder = RecourseTargetBuilder()
    selected = builder.project(graph)
    builder.verify_feasible(graph, selected)
    assert len({edge.resource_id for edge in edges if edge.edge_id in selected and edge.resource_type == "request"}) <= 1
    target = builder.double_q_target(
        graph,
        online_scores={"v0_req": 10, "v1_req": 2, "v0_station": 1, "v1_station": 9},
        target_scores={"v0_req": 3, "v1_req": 20, "v0_station": 30, "v1_station": 4},
    )
    assert set(target.selected_edge_ids) == {"v0_req", "v1_station"}
    assert target.target_evaluation_value == 7
    structured = builder.double_q_target(
        graph,
        online_scores={edge.edge_id: -100 for edge in edges},
        target_scores={edge.edge_id: 100 for edge in edges},
        structured_only=True,
    )
    assert structured.target_evaluation_value == structured.online_selection_value


def test_joint_projection_uses_optional_outside_action_for_negative_scores():
    edges = (
        FeasibleEdgeSnapshot(
            "v0_wait", 0, 1, ActionType.WAIT, "wait", 1, 1,
            structured_score=-2, collection_score=-2,
        ),
        FeasibleEdgeSnapshot(
            "v1_wait", 1, 2, ActionType.WAIT, "wait", 1, 1,
            structured_score=-3, collection_score=-3,
        ),
    )
    graph = _graph(edges)
    builder = RecourseTargetBuilder()
    selected = builder.project(graph)
    assert selected == ()
    builder.verify_feasible(graph, selected)


def test_assignment_mapping_does_not_confuse_relocation_with_request_pickup():
    graph = _graph(
        (
            FeasibleEdgeSnapshot(
                "v0_req", 0, 1, ActionType.SERVICE, "assign_10", 3, 4,
                request_id=10, resource_type="request", resource_id=10,
                resource_capacity=1,
            ),
            FeasibleEdgeSnapshot(
                "v0_reloc", 0, 1, ActionType.RELOCATE, "reloc_3", 3, 3,
            ),
        )
    )

    selected = StateSnapshotBuilder.selected_edge_ids(graph, {0: "idle_at_3"})

    assert selected == ("v0_reloc",)


def test_fleet_local_mask_changes_only_other_fleet_fields():
    joint = _state()
    local = joint.masked("fleet_local_separate_critics", vehicle_type=1)
    assert local.epoch_id == joint.epoch_id
    assert local.requests == joint.requests
    assert local.vehicles[0] == joint.vehicles[0]
    assert local.vehicles[1].online is False
    assert local.vehicles[1].location == 0


def test_registry_choices_all_import_and_invalid_key_fails():
    loaded = validate_value_function_registry()
    assert set(loaded) == set(VALUE_FUNCTION_CHOICES)
    assert get_value_function_class("optimization_anchored_residual") is loaded[
        "optimization_anchored_residual"
    ]
    with pytest.raises(ValueError, match="unknown distribution mode"):
        get_value_function_class("missing_model")


@pytest.mark.parametrize(
    "environment_class", [ChargingIntegratedEnvironment, NYCEnvironment]
)
def test_synthetic_and_nyc_expose_the_same_recourse_contract(environment_class):
    env = environment_class.__new__(environment_class)
    env.ifreject = True
    env.current_time = 3.49
    env.initial_random_seed = 4
    env.episode_start_day = 0
    env.episode_day_index = 0
    env.vehicles = {0: _vehicle(1)}
    env._last_offer_realizations = {}
    env.configure_recourse_experiment("r0", common_random_numbers=True)
    env._ensure_recourse_runtime()
    assert isinstance(env.request_lifecycle, RequestLifecycleTracker)
    assert env._epoch_id() == 3
    assert env._should_reject_request(0, _request()) is False
    with pytest.raises(ValueError):
        env.configure_recourse_experiment("not-a-variant")


def test_transition_joint_reward_reconciles():
    transition = _transition(rejected=True, assigned=True)
    assert transition.reward_system == transition.reward_ev + transition.reward_aev


def test_seed_day_summary_reports_uncertainty():
    summary = summarize_metric_with_uncertainty(
        [
            {"seed": 11, "day_id": "d1", "reward": 1.0},
            {"seed": 23, "day_id": "d2", "reward": 3.0},
        ],
        "reward",
    )
    assert summary["count"] == 2
    assert summary["seed_count"] == 2
    assert summary["day_count"] == 2
    assert summary["ci_lower"] < summary["mean"] < summary["ci_upper"]


def test_experiment_manifest_records_config_hashes_and_uncertainty(tmp_path):
    data_path = tmp_path / "demand.bin"
    data_path.write_bytes(b"stable-demand")
    output_path = tmp_path / "run.manifest.json"
    write_experiment_manifest(
        output_path,
        arguments={"random_seed": 11, "recourse_variant": "r4"},
        results={
            "episode_rewards": [1.0, 3.0],
            "episode_detailed_stats": [
                {"day_id": "d1", "completed_orders": 2},
                {"day_id": "d2", "completed_orders": 4},
            ],
        },
        data_paths=[data_path],
    )
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["resolved_config"]["recourse_variant"] == "r4"
    assert manifest["data_hashes"][str(data_path)]
    assert manifest["uncertainty"]["reward"]["mean"] == 2.0
    assert manifest["replay_schema_version"] == 1
