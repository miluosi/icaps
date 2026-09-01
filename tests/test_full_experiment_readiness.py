from dataclasses import replace
import json
from types import SimpleNamespace

import torch

import run_assignment_state_audit
import run_recourse_multiday_panel
import run_recourse_sensitivity
from run_recourse_spatiotemporal_analysis import aggregate as aggregate_spatiotemporal
from src.NYCEnvironment import NYCEnvironment
from src.recourse.integrated_repair import apply_hold_policy
from src.value_function_registry import VALUE_FUNCTION_CHOICES, get_value_function_class
from test_recourse_must_fix import _graph
from test_recourse_reaudit import _single_edge_graph, _transition


def test_public_registry_contains_only_residual_and_full_q_learners():
    assert VALUE_FUNCTION_CHOICES == (
        'optimization_anchored_residual', 'integrated_directq',
    )
    assert get_value_function_class('optimization_anchored_residual').learner_variant == (
        'optimization_anchored_residual'
    )
    assert get_value_function_class('integrated_directq').learner_variant == (
        'integrated_directq'
    )


def test_nyc_demand_scaling_is_keyed_deterministic_and_assigns_unique_ids():
    def run(scale):
        env = SimpleNamespace(
            demand_scale=scale, request_generation_seed=71, request_counter=2,
        )
        requests = [
            SimpleNamespace(request_id=1, pickup=10, dropoff=20),
            SimpleNamespace(request_id=2, pickup=11, dropoff=21),
        ]
        history = [{'pickup_zone': 10}, {'pickup_zone': 11}]
        return NYCEnvironment._apply_demand_scale(
            env, requests, history, date_label='2025-12-18', epoch_bin=4,
        )

    first, first_history = run(2.0)
    second, second_history = run(2.0)
    assert [row.request_id for row in first] == [1, 3, 2, 4]
    assert [row.request_id for row in first] == [row.request_id for row in second]
    assert first_history == second_history
    assert all(row['demand_scale'] == 2.0 for row in first_history)


def test_charge_duration_scale_changes_dynamic_session_duration():
    env = NYCEnvironment.__new__(NYCEnvironment)
    env.charge_target_soc = 0.8
    env.chargeincrease_per_epoch = 0.1
    env.min_charging_session_epochs = 1
    env.max_charging_session_epochs = 100
    env.charge_duration_scale = 1.0
    baseline = env._charge_duration_for_battery(0.2)
    env.charge_duration_scale = 1.5
    assert env._charge_duration_for_battery(0.2) > baseline


def test_fixed_hold_policy_reserves_the_declared_deterministic_fraction():
    base = _graph('fixed-hold', stage=0, vehicle_id=10, vehicle_type=2)
    edges = []
    for vehicle_id in range(10, 14):
        service = replace(
            base.edges[0], edge_id=f'service-{vehicle_id}',
            vehicle_id=vehicle_id, vehicle_type=2,
        )
        hold = replace(
            service, edge_id=f'hold-{vehicle_id}', action_id='hold_for_repair',
            metadata=(('repair_reserve', True),),
        )
        edges.extend((service, hold))
    graph = replace(base, edges=tuple(edges), selected_edge_ids=())
    env = SimpleNamespace(
        samitha_hold_rule='fixed', samitha_fixed_hold_fraction=0.5,
        current_time=10, recourse_run_id='paired',
    )
    result = apply_hold_policy(env, graph)
    by_vehicle = {vehicle_id: [edge for edge in result.edges if edge.vehicle_id == vehicle_id]
                  for vehicle_id in range(10, 14)}
    held = [vehicle_id for vehicle_id, vehicle_edges in by_vehicle.items()
            if any(dict(edge.metadata).get('repair_reserve') for edge in vehicle_edges)]
    assert len(held) == 2
    assert all(len(by_vehicle[vehicle_id]) == 1 for vehicle_id in held)
    assert all(not any(dict(edge.metadata).get('repair_reserve') for edge in by_vehicle[vehicle_id])
               for vehicle_id in set(by_vehicle) - set(held))


def test_multiday_plan_fits_once_per_method_seed_and_evaluates_many(tmp_path, capsys):
    run_recourse_multiday_panel.main([
        '--train-days', '2025-12-16', '2025-12-17',
        '--test-days', '2025-12-18', '2025-12-19', '2025-12-20',
        '--seeds', '71', '72', '--methods', 'recourse_macro', 'samitha',
        '--parquet-path', str(tmp_path / 'data.parquet'),
        '--energy-model', 'general_charging', '--dry-run',
        '--output-dir', str(tmp_path / 'out'),
    ])
    plan = json.loads(capsys.readouterr().out)
    assert plan['fitted_policy_jobs'] == 4
    assert plan['heldout_evaluations'] == 12
    assert len(plan['commands']) == 4
    assert all(command.count('--worker-method') == 1 for command in plan['commands'])
    assert all(command.count('--test-days') == 1 for command in plan['commands'])


def test_multiday_cumulative_training_counters_are_not_double_counted():
    summary = run_recourse_multiday_panel._aggregate_training_stats([
        {'macro_leader_target_count': 3, 'aev_follower_optimizer_steps': 2},
        {'macro_leader_target_count': 7, 'aev_follower_optimizer_steps': 5},
    ])
    assert summary['macro_leader_target_count'] == 7
    assert summary['aev_follower_optimizer_steps'] == 5


def test_sensitivity_design_includes_duration_and_nominal_robustness(tmp_path):
    args = run_recourse_sensitivity.parse_args([
        '--train-days', '2025-12-17', '--test-days', '2025-12-18',
        '--seeds', '71', '--parquet-path', str(tmp_path / 'data.parquet'),
        '--energy-model', 'general_charging',
        '--charge-duration-scales', '0.5', '1.0', '1.5',
    ])
    configs = run_recourse_sensitivity.configurations(args)
    assert any(row['charge_duration_scale'] == 0.5 for row in configs)
    assert any(row['charge_duration_scale'] == 1.5 for row in configs)
    assert sum(row['configuration_id'] == 'nominal' for row in configs) == 1


def test_state_audit_covers_pre_residual_and_both_stage_graph_states(tmp_path):
    ev_graph = _single_edge_graph(
        graph_id='ev', stage=1, vehicle_id=1, vehicle_type=1, structured=1.0,
    )
    aev_graph = _single_edge_graph(
        graph_id='aev', stage=2, vehicle_id=2, vehicle_type=2, structured=1.0,
    )
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'learners': [{'extra': {'joint_replay_state_dict': {
        'items': [_transition('state-audit', ev_graph=ev_graph, aev_graph=aev_graph)]
    }}}]}, checkpoint)
    report = run_assignment_state_audit.audit_checkpoint(
        checkpoint, ['strict_fleet_local_separate_critics'],
    )
    sources = report['observations']['strict_fleet_local_separate_critics']['state_sources']
    assert set(sources) == {
        'pre_state', 'residual_state', 'ev_stage_graph_state', 'aev_stage_graph_state',
    }
    assert all(source['max_other_fleet_node_count'] == 0 for source in sources.values())


def test_spatiotemporal_aggregation_outputs_recovery_hold_and_zone_mechanisms():
    payload = {'rows': [{
        'method': 'samitha',
        'hourly_recourse_events': [{
            'hour': 8, 'rejected_count': 4, 'eligible_count': 2,
            'assigned_count': 1, 'pickup_count': 1, 'completion_count': 1,
        }],
        'hourly_completed_orders': [{
            'completed_hour': 8, 'completed_orders': 5,
            'completed_ev_orders': 2, 'completed_aev_orders': 3,
        }],
        'samitha_hold_history': [{
            'hour': 8, 'hold_candidate_count': 4, 'hold_selected_count': 2,
            'hold_utilized_count': 1, 'unused_hold_count': 1,
        }],
        'spatial_recourse_events': [{
            'hour': 8, 'pickup_zone_id': 161, 'eligible': True,
            'assigned': True, 'picked_up': True, 'completed': True,
        }],
        'hourly_zone_request_completed_orders': [{
            'request_hour': 8, 'zone_id': 161,
            'generated_requests': 10, 'completed_requests': 5,
        }],
        'hourly_zone_vehicle_counts': [{
            'hour': 8, 'zone_id': 161, 'mean_total_vehicles': 8,
            'mean_ev_vehicles': 3, 'mean_aev_vehicles': 5,
        }],
        'hourly_zone_charge_station_counts': [{
            'hour': 8, 'zone_id': 161, 'mean_station_count': 1,
            'mean_total_capacity': 4, 'mean_queue_vehicle_count': 2,
            'mean_queue_to_capacity_ratio': 0.5,
        }],
    }]}
    selected, hourly, spatial = aggregate_spatiotemporal(payload)
    assert len(selected) == len(hourly) == len(spatial) == 1
    assert hourly[0]['conditional_completion_recovery'] == 0.5
    assert hourly[0]['repair_completion_per_held_aev'] == 0.5
    assert spatial[0]['request_completion_ratio'] == 0.5
    assert spatial[0]['mean_aev_vehicles'] == 5
