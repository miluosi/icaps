"""Real EV-first collection: physical repair versus learned repair."""
import pytest
import torch

from src.Request import Request
from run_recourse_audit import build_pair
from src.recourse.integrated_repair import _coords
from train_acceptance_model import make_environment, parse_args


def collect(variant):
    env = make_environment(parse_args(['--environment', 'nyc', '--num-vehicles', '2', '--num-ev', '1',
                                      '--stop-hour', '8.05']), 901)
    env.evaluatemode, env.adp_value, env.episode_length = False, 1., 1
    env.configure_recourse_experiment(variant, common_random_numbers=True)
    env.state_variant, env.learner_variant = 'joint_state_separate_critics', 'optimization_anchored_residual'
    env._should_consider_ev_charging = lambda vid: False
    ev = next(vid for vid, v in env.vehicles.items() if v['type'] == 1)
    loc = env.vehicles[ev]['location']
    for v in env.vehicles.values():
        v.update(location=loc, coordinates=_coords(env, loc), battery=.95, assigned_request=None,
                 passenger_onboard=None, charging_station=None, charging_target=None,
                 target_location=None, idle_target=None, penalty_timer=0, is_online=True)
    env.active_requests = {rid: Request(rid, loc, loc, 0., 1., value=value, final_value=value)
                           for rid, value in [(10001, 100.), (10002, 60.)]}
    env._should_reject_request = lambda vid, request: env.vehicles[vid]['type'] == 1
    torch.manual_seed(991)
    pair = build_pair(env)
    for value in pair:
        value.training_step = 1000
        for critic in (value.network, value.critic2):
            critic.base.net[-1].weight.data.zero_()
            critic.base.net[-1].bias.data.fill_(5.)
    actions, stores, stores_ev = env.simulate_motion_evfirst()
    pending = env.recourse_coordinator.pending
    graph = pending.aev_stage_graph
    env.step(actions, stores, stores_ev)
    return env, pair, graph, list(pair[0].joint_replay_buffer)[-1]


def test_r2_r3_share_repair_feasibility_but_only_r3_learns_follower():
    _, r2, graph2, row2 = collect('repair_only')
    _, r3, graph3, row3 = collect('repair_learning')
    key = lambda e: (e.vehicle_id, e.action_type, e.request_id, e.station_id, e.target_location, e.resource_capacity)
    assert {key(e) for e in graph2.edges} == {key(e) for e in graph3.edges}
    assert {e.request_id for e in graph2.edges if e.request_id} == {10001, 10002}
    assert all(e.collection_score == pytest.approx(e.structured_score) for e in graph2.edges)
    assert any(abs(e.collection_score - e.structured_score) > 1e-4 for e in graph3.edges)
    for pair, row in ((r2, row2), (r3, row3)):
        pair[1]._r4_follower_components = lambda *args: pytest.fail('uncoupled EV must not query Q2')
        pair[1].train_step(batch_size=1, ifEV=True)
        assert pair[1].joint_training_diagnostics[-1]['joint_target_full'] == pytest.approx(row.reward_ev)
    assert r2[0].train_step(batch_size=1, ifEV=False) == 0.
    assert r2[0].optimizer_steps_joint == r2[0].optimizer_steps_queue == 0
    assert r3[0].train_step(batch_size=1, ifEV=False) > 0.
    assert r3[0].optimizer_steps_joint == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in r3[0].network.parameters())
