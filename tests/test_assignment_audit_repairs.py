import argparse
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.recourse.config import (
    AssignmentOracleConfig,
    METHOD_ALIASES,
    METHODS,
    PAPER_METHODS,
    add_method_arguments,
    canonical_method,
)
from src.recourse.contracts import (
    assert_method_event_contract,
    evaluate_method_event_contract,
)
from src.recourse.crn import vehicle_normal
from src.recourse.target_builder import RecourseTargetBuilder
from src.recourse.types import FeasibleGraphSnapshot
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction
from test_recourse_must_fix import _graph


def test_canonical_method_registry_contains_controls_but_no_alias_duplicates():
    assert tuple(METHODS) == PAPER_METHODS
    assert 'recourse_aware' not in METHODS
    assert canonical_method('recourse_aware') == 'recourse_macro'
    assert METHODS['evfirst_no_rejection'].variant == 'r0'
    assert METHODS['evfirst_no_repair_structured'].variant == 'r1_structured'
    assert METHODS['evfirst_no_repair_structured'].repair_policy == 'structured'


def test_hold_boolean_flag_can_be_disabled_and_target_policy_is_explicit():
    parser = argparse.ArgumentParser()
    add_method_arguments(parser)
    args = parser.parse_args(['--no-integrated-repair-hold-enabled'])
    assert args.integrated_repair_hold_enabled is False
    assert args.target_solver_policy == 'same_as_rollout_exact'


def test_target_projection_consumes_serialized_solver_configuration():
    graph = replace(
        _graph('solver-config', stage=1, vehicle_id=0, vehicle_type=1),
        solver_backend='primal_dual',
        graph_reduction=False, solver_verify=False,
        target_solver_policy='same_as_rollout_exact', objective_cost_scale=1234,
    )
    builder = RecourseTargetBuilder()
    builder.project(graph)
    diagnostics = builder.last_solver_diagnostics
    assert diagnostics['backend'] == diagnostics['rollout_backend'] == 'primal_dual'
    assert diagnostics['graph_reduction'] is False
    assert diagnostics['verify'] is False
    assert diagnostics['cost_scale'] == 1234
    assert diagnostics['target_solver_family'] == 'exact'
    assert diagnostics['strict'] is True
    assert diagnostics['selected_edge_trace_hash']


def test_unknown_new_solver_metadata_fails_closed_but_old_test_marker_migrates():
    graph = _graph('solver-migration', stage=1, vehicle_id=0, vehicle_type=1)
    builder = RecourseTargetBuilder()
    builder.project(graph)  # historical fixtures serialize backend="test"
    assert builder.last_solver_diagnostics['legacy_solver_metadata_migrated']
    builder.project(replace(graph, solver_backend='mcmf:primal_dual'))
    assert builder.last_solver_diagnostics['backend'] == 'primal_dual'
    assert builder.last_solver_diagnostics['legacy_solver_metadata_migrated']
    with pytest.raises(ValueError, match='unknown exact MCMF backend'):
        builder.project(replace(graph, solver_backend='invented_backend'))


def test_strict_local_graph_view_and_scalar_remove_other_fleet():
    graph = _graph('strict-local', stage=1, vehicle_id=0, vehicle_type=1)
    focal = graph.state.vehicles[0]
    other = replace(focal, vehicle_id=1, vehicle_type=2, online=True, location=7)
    graph = replace(graph, state=replace(graph.state, vehicles=(focal, other)))
    local = graph.state.masked('fleet_local_separate_critics', vehicle_type=1)
    strict = graph.state.masked('strict_fleet_local_separate_critics', vehicle_type=1)
    assert len(local.vehicles) == 2 and local.vehicles[1].online is False
    assert strict.vehicles == (focal,)
    exp = PyTorchChargingValueFunction._edge_experience(
        graph, graph.edges[0], state_variant='strict_fleet_local_separate_critics'
    )
    assert exp['other_vehicles'] == 0
    assert len(exp['state_snapshot'].vehicles) == 1


def test_reward_crn_is_event_keyed_and_does_not_depend_on_call_order():
    env = SimpleNamespace(
        common_random_numbers=True, initial_random_seed=4,
        recourse_run_id='paired', cumulative_episode_index=1,
        episode_day_index=0, current_time=9,
    )
    expected = vehicle_normal(env, 2, 'pickup_reward', .2, request_id=10)
    for vehicle_id in reversed(range(50)):
        vehicle_normal(env, vehicle_id, 'movement_reward', .1, request_id=99)
    assert vehicle_normal(env, 2, 'pickup_reward', .2, request_id=10) == expected
    assert vehicle_normal(env, 2, 'dropoff_reward', .2, request_id=10) != expected


def test_event_contracts_fail_closed_and_report_exact_missing_mechanism():
    stats = dict(
        eligible_rejected_residual_count=1,
        same_epoch_aev_assignment_count=0,
        aev_follower_optimizer_steps=0,
        aev_learned_score_difference_count=0,
    )
    result = evaluate_method_event_contract('repair_only', stats)
    assert not result.passed
    assert result.failures == ('same_epoch_repair_present',)
    with pytest.raises(AssertionError, match='same_epoch_repair_present'):
        assert_method_event_contract('repair_only', stats)


def test_causal_r2_r3_share_frozen_p0_predictor_readiness():
    from src.recourse.training import training_readiness

    class Fake:
        experience_buffer = [1] * 8
        queue_experience_buffer = [1] * 8
        freeze_causal_predictors = True

        def train_queue_predictor(self):
            pass

        def has_trainable_joint_rows(self, *, ifEV):
            return True

    for variant in ('r2', 'r3'):
        value = Fake()
        value.recourse_variant = variant
        readiness = training_readiness(value, ifEV=False, edge_warmup=4)
        assert readiness.predictor_ready is False
    r2 = Fake()
    r2.recourse_variant = 'r2'
    r3 = Fake()
    r3.recourse_variant = 'r3'
    assert not training_readiness(r2, ifEV=False, edge_warmup=4).follower_critic_ready
    assert training_readiness(r3, ifEV=False, edge_warmup=4).follower_critic_ready


def test_production_integrated_selector_uses_shared_stage0_control():
    from src.NYCtrainer import NYCTrainer

    integrated = object()
    legacy = object()
    env = SimpleNamespace(
        simulate_motion_integrated_control=integrated,
        simulate_motion=legacy,
        simulate_motion_evfirst=object(),
        simulate_motion_integrated_repair=object(),
    )
    assert NYCTrainer._select_motion_fn(env, 'integrated') is integrated
    assert NYCTrainer._select_motion_fn(env, 'integrated_repair') is env.simulate_motion_integrated_repair


def test_integrated_and_samitha_hold_disabled_share_solver_and_tie_break():
    graph = replace(
        _graph('shared-stage0', stage=0, vehicle_id=0, vehicle_type=1),
        solver_backend='primal_dual', graph_reduction=True,
        solver_verify=True, target_solver_policy='same_as_rollout_exact',
    )
    # Both production paths call the same builder; two independent adapters
    # must therefore select and quantize identically on an exact tie.
    integrated = RecourseTargetBuilder()
    samitha_control = RecourseTargetBuilder()
    assert integrated.project(graph) == samitha_control.project(graph)
    assert integrated.last_solver_diagnostics['backend'] == samitha_control.last_solver_diagnostics['backend']
    assert integrated.last_solver_diagnostics['selected_edge_trace_hash'] == samitha_control.last_solver_diagnostics['selected_edge_trace_hash']


@pytest.mark.parametrize('environment', ['synthetic', 'nyc'])
def test_shared_integrated_control_executes_without_hold_or_stage2(environment):
    from train_acceptance_model import make_environment, parse_args

    args = parse_args([
        '--environment', environment, '--num-vehicles', '4', '--num-ev', '2',
        '--simulation-period', '10', '--stop-hour', '8.05',
    ])
    env = make_environment(args, 511)
    env.configure_recourse_experiment('legacy', common_random_numbers=True)
    actions, stores, stores_ev = env.simulate_motion_integrated_control()
    initial, repair = env._last_integrated_repair_graphs
    config = AssignmentOracleConfig.from_environment(env)
    assert initial.stage_id == 0
    assert repair is None
    assert not any(dict(edge.metadata).get('repair_reserve') for edge in initial.edges)
    assert env._integrated_repair_metrics['hold_selected_count'] == 0
    assert config.solver_family == env.mcmf_solver
    env.step(actions, stores, stores_ev)


def test_nyc_structured_q_keeps_zero_request_layout_when_all_requests_blocked():
    from src.Request import Request
    from train_acceptance_model import make_environment, parse_args

    env = make_environment(parse_args([
        '--environment', 'nyc', '--num-vehicles', '4', '--num-ev', '2',
        '--simulation-period', '10', '--stop-hour', '8.05',
    ]), 512)
    aev_ids = [vid for vid, vehicle in env.vehicles.items() if vehicle['type'] == 2]
    location = env.vehicles[aev_ids[0]]['location']
    env.active_requests = {
        60001: Request(60001, location, location, env.current_time, 1., value=20., final_value=20.)
    }
    env._same_epoch_blocked_request_ids = {60001}
    matrix, num_requests, _num_stations, _num_zones = env.generate_whole_matrix(
        aev_ids, rebalance_num=len(aev_ids)
    )
    q_values = env.generate_vehicle_qvalue_withoutqnetwork(aev_ids)
    assert num_requests == 0
    assert q_values.shape == matrix.shape
