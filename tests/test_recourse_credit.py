from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src.recourse.config import METHODS, canonical_variant, method_metadata
from src.recourse.target_builder import RecourseTargetBuilder, TargetComponents
from src.recourse.types import OutcomeSummary, RecourseEvent, RewardLedger, is_true_same_epoch_recourse
from src.recourse.replay import PrioritizedJointReplayBuffer
from src.ValueFunction_optimization_anchored_residual import PyTorchChargingValueFunction
from test_recourse_must_fix import _vf, _transition, _graph


def test_names_preserve_integrated_and_r1_and_separate_target_families():
    assert METHODS['no_repair'].operating_mode == 'integrated'
    assert METHODS['evfirst_no_repair'].variant == 'r1'
    assert canonical_variant('repair_only') == 'r2'
    assert canonical_variant('repair_learning') == 'r3'
    assert canonical_variant('recourse_aware') == 'recourse_macro'
    assert canonical_variant('recourse_nested_q2') == 'r4'
    assert method_metadata('evfirst', 'r2')['follower_learning'] is False


@pytest.mark.parametrize('done, expected', [(False, 14.), (True, 5.)])
@pytest.mark.parametrize('direct_q', [False, True])
def test_macro_joint_target_uses_entire_reward_without_follower_query(done, expected, direct_q):
    value = _vf(recourse='recourse_macro')
    value.direct_q = direct_q
    value.gamma = .9
    graph = _graph('ev-now', stage=1, vehicle_id=0, vehicle_type=1, structured=4.)
    row = _transition(ev_graph=graph, done=done,
                      next_id=None if done else 'run:episode:0:sequence:1', recourse_variant='recourse_macro')
    row = replace(row, reward_ev=-2., reward_aev=7., reward_system=5.)
    value.store_recourse_transition(row)
    if not done:
        value.store_recourse_transition(_transition('run:episode:0:sequence:1', sequence=1,
            ev_graph=_graph('ev-next', stage=1, vehicle_id=0, vehicle_type=1), recourse_variant='recourse_macro'))
    value.target_components_for_graph = lambda graph, **kwargs: TargetComponents((), 10., 10., 0.)
    value._r4_follower_components = lambda *args: pytest.fail('macro must not query Q2')
    value._next_soft_values = lambda *args: pytest.fail('main recourse must not use edge Bellman TD')
    # Directly constrain the sample so only the target under test is updated.
    value._joint_row_ready = lambda transition, **kwargs: transition.transition_id == row.transition_id
    assert value.train_step(batch_size=1, ifEV=True) > 0
    diagnostic = value.joint_training_diagnostics[-1]
    assert diagnostic['joint_target_full'] == pytest.approx(expected)
    assert diagnostic['joint_residual_target'] == pytest.approx(expected - (0. if direct_q else 4.))
    assert diagnostic['leader_recourse_credit'] == 7.
    assert value.optimizer_steps_edge == 0


def test_macro_and_exact_nested_match_but_approximate_follower_differs():
    builder = RecourseTargetBuilder()
    kwargs = dict(reward_ev=-2., reward_aev=7., temporal_value=10., gamma=.9, done=False)
    macro = builder.leader_target(variant='recourse_macro', follower_value=999., **kwargs)
    assert macro == 14.
    assert builder.leader_target(variant='r4', follower_value=16., **kwargs) == macro
    assert builder.leader_target(variant='r4', follower_value=18., **kwargs) != macro
    assert builder.leader_target(variant='r3', follower_value=999., **kwargs) == 7.
    assert builder.leader_target(variant='r2', follower_value=999., **kwargs) == 7.


def test_repair_only_never_updates_any_follower_optimizer():
    value = PyTorchChargingValueFunction(grid_size=2, num_vehicles=2, episode_length=10)
    value.recourse_variant = 'r2'
    for _ in range(8):
        value.queue_experience_buffer.append(dict(features=[.1] * value.queue_feature_dim, observed_wait=2.))
        value.post_demand_experience_buffer.append(([.1] * value.post_demand_feature_dim, .2))
    parameters = [p for module in (value.network, value.queue_predictor, value.post_demand_predictor) for p in module.parameters()]
    before = [p.detach().clone() for p in parameters]
    assert value.train_step(batch_size=4, ifEV=False) == 0.
    assert all(torch.equal(a, b) for a, b in zip(before, parameters))
    assert value.optimizer_steps_queue == value.optimizer_steps_joint == value.optimizer_steps_edge == 0


@pytest.mark.parametrize('changes', [dict(residual_category='unoffered'), dict(eligible=False),
    dict(assigned_vehicle_type=1), dict(assignment_epoch_id=2), dict(same_epoch_recourse_link=False)])
def test_priority_bonus_only_rewards_true_same_epoch_repair(changes):
    true = RecourseEvent(1, 1, 'rejected', True, True, False, False,
        assignment_epoch_id=1, first_rejected_epoch=1, assigned_vehicle_type=2, same_epoch_recourse_link=True)
    other = replace(true, **changes)
    assert is_true_same_epoch_recourse(true) and not is_true_same_epoch_recourse(other)
    buffer = PrioritizedJointReplayBuffer(rejection_bonus=0., recourse_bonus=3.)
    row = _transition()
    a = replace(row, outcome_summary=OutcomeSummary((true,)))
    b = replace(row, outcome_summary=OutcomeSummary((other,)))
    assert buffer.priority_from_td(a, 2.) - buffer.priority_from_td(b, 2.) == pytest.approx(3.)


def test_reward_ledger_requires_exact_return_reconciliation():
    ledger = RewardLedger(ev_accepted_service=3., ev_rejection_penalty=-2., aev_unoffered_service=5., aev_charging=-1.)
    row = replace(_transition(), reward_ev=1., reward_aev=4., reward_system=5., reward_ledger=ledger)
    assert row.reward_ledger.system == row.reward_system == 5.
    assert row.stage1_graph is row.ev_stage_graph
    with pytest.raises(ValueError, match='ledger'):
        replace(row, reward_ledger=RewardLedger())


def test_old_replay_hash_and_r4_meaning_are_migrated_without_fake_rewards():
    old = PrioritizedJointReplayBuffer()
    row = _transition(recourse_variant='r4')
    old.add(row)
    state = old.state_dict()
    state.pop('recourse_credit_schema')
    state['content_hash'] = old._legacy_content_hash(state['items'], state['priorities'])
    restored = PrioritizedJointReplayBuffer()
    restored.load_state_dict(state)
    migrated = list(restored)[0]
    assert migrated.recourse_variant == 'r4'
    assert migrated.recourse_target_family == 'nested_follower'
    assert migrated.reward_ledger is None
    assert restored.state_dict()['recourse_credit_schema'] == 1


def test_charge_crn_is_keyed_and_does_not_advance_global_rng():
    import random
    import numpy as np
    from src.recourse.crn import vehicle_uniform
    env = SimpleNamespace(common_random_numbers=True, initial_random_seed=9, current_time=3.)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    value = vehicle_uniform(env, 2, 'charge_decision')
    for vid in range(20):
        vehicle_uniform(env, vid, 'charge_reward')
    assert vehicle_uniform(env, 2, 'charge_decision') == value
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert vehicle_uniform(env, 2, 'charge_station') != value


def test_legacy_non_joint_learner_is_rejected_for_recourse_but_r1_is_preserved():
    from src.recourse.config import validate_joint_learner
    validate_joint_learner('evfirst', 'r1', object)
    validate_joint_learner('integrated', 'legacy', object)
    validate_joint_learner('evfirst', 'recourse_macro', PyTorchChargingValueFunction)
    with pytest.raises(ValueError, match='joint critic'):
        validate_joint_learner('evfirst', 'r3', object)
    with pytest.raises(ValueError, match='joint critic'):
        validate_joint_learner('integrated_repair', 'legacy', object)


@pytest.mark.parametrize('environment', ['synthetic', 'nyc'])
def test_paired_methods_share_offer_and_charge_crn(environment):
    from run_recourse_audit import build_env, parse_args
    args = parse_args(['--environment', environment, '--num-vehicles', '4', '--num-ev', '2'])
    draws = []
    for method in METHODS:
        env = build_env(args, 411, method, training=False)
        req = SimpleNamespace(request_id=12345)
        vid = next(vid for vid, v in env.vehicles.items() if v['type'] == 1)
        draws.append((env._acceptance_uniform(vid, req), env._charge_uniform(vid, 'charge_decision')))
    assert len(set(draws)) == 1


@pytest.mark.parametrize('key', ['ev_rejected_offer_count', 'human_ev_charging_sessions', 'aev_charging_sessions'])
def test_checkpoint_validation_checks_real_required_metrics(key):
    from run_recourse_audit import INFERENCE_CHECK_KEYS, verify_checkpoint_stats
    stats = dict.fromkeys(INFERENCE_CHECK_KEYS, 0.)
    verify_checkpoint_stats(stats, dict(stats))
    with pytest.raises(AssertionError, match='mismatch'):
        verify_checkpoint_stats(stats, dict(stats, **{key: 1.}))
    missing = dict(stats)
    missing.pop(key)
    with pytest.raises(AssertionError, match='missing'):
        verify_checkpoint_stats(missing, missing)


def test_target_summary_distinguishes_system_leader_from_absent_follower():
    from src.recourse.metrics import summarize_joint_targets
    summary = summarize_joint_targets([dict(phase='system', joint_target_full=5.,
        joint_prediction_abs=2., target_projection_runtime=.03)])
    assert summary['leader_td_target_mean'] == 5.
    assert summary['follower_td_target_mean'] is None
    assert summary['follower_td_target_count'] == 0


def test_ordinary_service_displacement_is_a_fixed_graph_counterfactual():
    from src.recourse.metrics import ordinary_service_displacement
    from src.recourse.types import ActionType
    graph = _graph('g', stage=2, vehicle_id=1, vehicle_type=2)
    rejected = replace(graph.edges[0], edge_id='repair', action_type=ActionType.SERVICE,
        request_id=10, resource_type='request', resource_id=10, resource_capacity=1, collection_score=20.)
    ordinary = replace(rejected, edge_id='ordinary', request_id=11, resource_id=11, collection_score=10.)
    graph = replace(graph, edges=(rejected, ordinary), selected_edge_ids=('repair',),
                    state=replace(graph.state, request_labels=((10, 'rejected'), (11, 'unoffered'))))
    assert ordinary_service_displacement(graph) == 1
    assert graph.selected_edge_ids == ('repair',)
