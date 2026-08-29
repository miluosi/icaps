from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.Request import Request
from src.recourse import integrated_repair as impl
from src.recourse.replay import PrioritizedJointReplayBuffer
from src.recourse.target_builder import RecourseTargetBuilder
from src.recourse.types import ActionType, FeasibleEdgeSnapshot
from train_acceptance_model import make_environment, parse_args
from test_recourse_must_fix import _graph


@pytest.mark.parametrize('environment', ['synthetic', 'nyc'])
@pytest.mark.parametrize('reject', [False, True])
def test_limited_hold_real_execution_preserves_commits(environment, reject, monkeypatch):
    args = parse_args(['--environment', environment, '--num-vehicles', '3', '--num-ev', '1',
                       '--simulation-period', '10', '--stop-hour', '8.05'])
    env = make_environment(args, 301)
    env.evaluatemode = False
    env.recourse_coordinator.replay = PrioritizedJointReplayBuffer()
    env._should_consider_ev_charging = lambda vid: False
    ev = next(vid for vid, v in env.vehicles.items() if v['type'] == 1)
    commit, held = [vid for vid, v in env.vehicles.items() if v['type'] == 2]
    loc = env.vehicles[ev]['location']
    for v in env.vehicles.values():
        v.update(location=loc, coordinates=impl._coords(env, loc), battery=.95,
                 assigned_request=None, passenger_onboard=None, charging_station=None,
                 charging_target=None, target_location=None, idle_target=None, penalty_timer=0,
                 is_online=True, is_stationary=False)
    env.active_requests = {rid: Request(rid, loc, loc, env.current_time, 1., value=20., final_value=20.) for rid in (10001, 10002, 10003)}
    env._should_reject_request = lambda vid, request: bool(reject and vid == ev)
    original = impl.build_stage_graph

    def fixed_first_plan(env, ids, **kwargs):
        graph = original(env, ids, **kwargs)
        if kwargs['stage'] == 0:
            chosen = [next(e.edge_id for e in graph.edges if e.vehicle_id == ev and e.request_id == 10001),
                      next(e.edge_id for e in graph.edges if e.vehicle_id == commit and e.request_id == 10002),
                      next(e.edge_id for e in graph.edges if e.vehicle_id == held and e.action_id == 'hold_for_repair')]
            graph = graph.with_selected(tuple(chosen))
            RecourseTargetBuilder.verify_feasible(graph, graph.selected_edge_ids)
        return graph

    monkeypatch.setattr(impl, 'build_stage_graph', fixed_first_plan)
    actions, stores, stores_ev = env.simulate_motion_integrated_repair()
    first, second = env._last_integrated_repair_graphs
    assert {e.vehicle_id for e in second.edges} == {held}
    candidates = {e.request_id for e in second.edges if e.request_id is not None}
    assert 10002 not in candidates and 10003 in candidates
    assert (10001 in candidates) is reject
    assert env.vehicles[commit]['assigned_request'] == 10002
    assert env._integrated_repair_metrics['committed_aev_reassignment_count'] == 0
    pending = env.recourse_coordinator.pending
    assert pending.repair_hold_aev_ids == (held,)
    env.step(actions, stores, stores_ev)
    row = list(env.recourse_coordinator.replay)[-1]
    assert row.mode == 'integrated_repair'
    assert row.stage1_graph.stage_id == 0 and row.stage2_graph.stage_id == 2
    assert row.reward_system == pytest.approx(row.reward_ledger.system)
    assert row.reward_system == pytest.approx(row.reward_ev + row.reward_aev)


def test_repair_charging_capacity_subtracts_initial_commit_once():
    first = _graph('initial', stage=0, vehicle_id=1, vehicle_type=2)
    commit = replace(first.edges[0], action_type=ActionType.CHARGE, action_id='charge_0',
                     resource_type='station', resource_id=0, station_id=0, resource_capacity=2)
    hold = replace(first.edges[0], edge_id='hold', vehicle_id=2, action_id='hold_for_repair', metadata=(('repair_reserve', True),))
    first = replace(first, edges=(commit, hold), selected_edge_ids=(commit.edge_id, hold.edge_id))
    repair = replace(first, stage_id=2, edges=(replace(commit, vehicle_id=2, edge_id='repair-charge'), commit), selected_edge_ids=())
    result = impl.residual_graph(repair, first, {2}, set())
    assert len(result.edges) == 1
    assert result.edges[0].vehicle_id == 2 and result.edges[0].resource_capacity == 1


def test_critical_battery_cannot_hold_and_hold_is_not_ordinary_wait():
    graph = _graph('g', stage=0, vehicle_id=1, vehicle_type=2)
    env = SimpleNamespace(vehicles={1: dict(type=2, battery=.05, location=0)}, critical_charging_battery=.1)
    assert impl.add_hold_edges(env, graph, [1]).edges == graph.edges
    env.vehicles[1]['battery'] = .9
    held = impl.add_hold_edges(env, graph, [1]).edges[-1]
    assert held.action_id == 'hold_for_repair'
    assert dict(held.metadata)['repair_reserve']
    assert held.post_action_duration == 0


def test_nyc_relocation_executor_receives_hotspot_index_not_zone_id():
    edge = replace(_graph('g', stage=0, vehicle_id=1, vehicle_type=2).edges[0],
                   action_type=ActionType.RELOCATE, target_location=161)
    assert impl._action_target(SimpleNamespace(hotspot_locations=[43, 161, 237]), edge) == 'idle_at_1'


@pytest.mark.parametrize('environment', ['synthetic', 'nyc'])
def test_no_residual_matches_existing_integrated_with_identical_initial_plan(environment, monkeypatch):
    """Run both real execution paths; pin only the common initial assignment."""
    from src.GurobiOptimizer import GurobiOptimizer
    from src.recourse.state_snapshot import StateSnapshotBuilder

    args = parse_args(['--environment', environment, '--num-vehicles', '2', '--num-ev', '1',
                      '--simulation-period', '10', '--stop-hour', '8.05'])
    original = impl.build_stage_graph
    outcomes = []
    for mode in ('integrated', 'integrated_repair'):
        env = make_environment(args, 309)
        env.configure_recourse_experiment('legacy', common_random_numbers=True)
        env.recourse_run_id = 'same-initial-plan'
        env._should_consider_ev_charging = lambda vid: False
        env._should_reject_request = lambda vid, req: False
        ids = sorted(env.vehicles)
        loc = env.vehicles[ids[0]]['location']
        for v in env.vehicles.values():
            v.update(location=loc, coordinates=impl._coords(env, loc), battery=.95,
                     assigned_request=None, passenger_onboard=None, charging_station=None,
                     charging_target=None, target_location=None, idle_target=None, penalty_timer=0,
                     is_online=True, is_stationary=False)
        env.active_requests = {rid: Request(rid, loc, loc, 0., 1., value=20., final_value=20.)
                               for rid in (10001, 10002)}
        assignments = dict(zip(ids, env.active_requests.values()))

        def first_plan(environment, vehicle_ids, **kwargs):
            graph = original(environment, vehicle_ids, **kwargs)
            if kwargs['stage'] == 0:
                graph = graph.with_selected(tuple(next(e.edge_id for e in graph.edges
                    if e.vehicle_id == vid and e.request_id == req.request_id)
                    for vid, req in assignments.items()))
            return graph

        monkeypatch.setattr(impl, 'build_stage_graph', first_plan)
        if environment == 'nyc':
            def solve(vehicle_ids, requests):
                env._last_feasible_graph_snapshot = first_plan(env, vehicle_ids, stage=0,
                    state=StateSnapshotBuilder.build(env))
                return assignments
            monkeypatch.setattr(env, '_solve_rebalancing', solve)
        else:
            monkeypatch.setattr(GurobiOptimizer, '_np_vehicle_rebalancing_network',
                                lambda *a, **k: assignments)
        motion = env.simulate_motion if mode == 'integrated' else env.simulate_motion_integrated_repair
        actions, stores, stores_ev = motion()
        if mode == 'integrated_repair':
            assert env._last_integrated_repair_graphs[1].edges == ()
            assert env._last_integrated_repair_graphs[1].selected_edge_ids == ()
        _, rewards, _, _, _ = env.step(actions, stores, stores_ev)
        outcomes.append((rewards, [(v['location'], v['battery'], v['assigned_request'], v['passenger_onboard'])
                                   for v in env.vehicles.values()], env.get_episode_stats()['completed_orders']))
    assert outcomes[0] == outcomes[1]


def test_samitha_unoffered_pickup_completion_not_misclassified_as_ev_recourse():
    from src.recourse.lifecycle import RequestLifecycleTracker
    tracker = RequestLifecycleTracker()
    tracker.mark_residual(7, epoch_id=1, category='unoffered', eligible=True,
                          repair_architecture='integrated_repair')
    tracker.record_integrated_repair_assignment(7, vehicle_id=2, epoch_id=1)
    tracker.record_pickup(7, vehicle_id=2, epoch_id=2)
    tracker.record_completion(7, vehicle_id=2, vehicle_type=2, epoch_id=3)
    # A separate initial integrated service must not contribute to repair.
    tracker.record_pickup(8, vehicle_id=3, epoch_id=2)
    tracker.record_completion(8, vehicle_id=3, vehicle_type=2, epoch_id=3)
    metrics = tracker.metrics()
    assert metrics['samitha_repair_pickup_count'] == metrics['samitha_repair_completion_count'] == 1
    assert metrics['same_epoch_aev_assignment_count'] == 0
    assert metrics['rejected_residual_count'] == 0


@pytest.mark.parametrize('charge', [False, True])
def test_synthetic_human_charging_cdf_and_cooldown_match_integrated(charge):
    results = []
    for mode in ('integrated', 'integrated_repair'):
        env = make_environment(parse_args(['--environment', 'synthetic', '--num-vehicles', '2', '--num-ev', '1']), 403)
        vid = next(vid for vid, v in env.vehicles.items() if v['type'] == 1)
        v = env.vehicles[vid]
        v.update(battery=.9, assigned_request=None, passenger_onboard=None, charging_station=None,
                 charging_target=None, idle_target=None, target_location=None, is_online=True)
        stations = list(env.charging_manager.stations)[:2]
        env._should_consider_ev_charging = lambda vehicle_id: True
        env.compute_ev_charge_probability = lambda vehicle_id: (float(charge), {stations[0]: .1, stations[1]: .3})
        env._charge_uniform = lambda vehicle_id, stream, **kwargs: .8
        motion = env.simulate_motion if mode == 'integrated' else env.simulate_motion_integrated_repair
        motion(rebalance=False)
        results.append((v.get('charging_target'), v.get('no_charge_cooldown_until')))
    assert results[0] == results[1]
