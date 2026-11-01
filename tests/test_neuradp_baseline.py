"""NeurADP external baseline: CLI dispatch, scoring, per-agent TD and checkpoints."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_nyctrainer
import test_nyc_model
from src.ADPtrainer import ADPTrainer
from src.recourse.config import (
    ICAPS_METHODS,
    METHODS,
    PAPER_METHODS,
    method_checkpoint_suffix,
    method_learner_variant,
    method_spec,
    neuradp_options,
    resolve_method_list_arguments,
)
from src.recourse.types import ActionType
from src.ValueFunction_neuradp import PyTorchChargingValueFunction as NeurADP
from src.value_function_registry import VALUE_FUNCTION_CHOICES, get_value_function_class


def test_neuradp_is_selectable_but_not_a_paper_method():
    assert "neuradp" not in ICAPS_METHODS
    assert "neuradp" not in METHODS and "neuradp" not in PAPER_METHODS
    assert "neuradp" not in resolve_method_list_arguments(["all"])
    assert resolve_method_list_arguments(["neuradp", "macro"]) == ["neuradp", "macro"]
    spec = method_spec("neuradp")
    assert (spec.operating_mode, spec.variant, spec.repair_policy, spec.leader_credit) == (
        "integrated", "legacy", "none", "uncoupled")
    assert method_learner_variant("neuradp", "optimization_anchored_residual") == "neuradp"
    assert method_learner_variant("macro", "integrated_directq") == "integrated_directq"
    # Paper methods cannot be combined with the NeurADP learner by flag.
    assert "neuradp" not in VALUE_FUNCTION_CHOICES
    assert get_value_function_class("neuradp") is NeurADP


@pytest.mark.parametrize("entrypoint", [run_nyctrainer, test_nyc_model])
def test_train_and_test_parsers_accept_neuradp_options(entrypoint):
    args = entrypoint.parse_args([
        "--methods", "neuradp", "--neuradp-gamma", "0.9", "--neuradp-noise-scale", "0",
        "--neuradp-response-aware",
    ])
    assert args.methods == ["neuradp"]
    assert neuradp_options(args) == {
        "neuradp_gamma": 0.9, "neuradp_noise_scale": 0.0, "neuradp_response_aware": True,
    }
    assert neuradp_options(entrypoint.parse_args(["--methods", "r1"])) == {}


def test_train_and_test_share_the_neuradp_checkpoint_namespace():
    suffix = method_checkpoint_suffix(
        "experiment", "neuradp", state_variant="joint_state_shared_critic",
        learner_variant=method_learner_variant("neuradp", "optimization_anchored_residual"),
        rejection_logit_shift=0.0,
    )
    assert "method-neuradp_rec-legacy" in suffix and "learner-neuradp" in suffix
    dirs = [
        test_nyc_model.build_checkpoint_dir(
            "gurobi", "integrated", 1500, True, vtype, "2025-12-08", "2025-12-10",
            "neuradp", True, False, checkpoint_suffix=suffix)
        for vtype in ("aev", "ev")
    ]
    assert all("_integrated_" in path and "_neuradp_" in path for path in dirs)


def test_run_nyctrainer_dispatches_neuradp_with_its_own_learner(monkeypatch):
    calls = []

    def fake_training(**kwargs):
        calls.append(kwargs)
        return {"episode_rewards": []}, SimpleNamespace(parquet_path=())

    monkeypatch.setattr(run_nyctrainer, "run_nyc_training", fake_training)
    run_nyctrainer.main([
        "--methods", "neuradp", "r3", "--episodes", "1", "--start-date", "2025-12-18",
        "--aev-charging-center-count", "3", "--no-ifreject",
        "--neuradp-samples-per-update", "5",
    ])
    neuradp, r3 = calls
    assert (neuradp["transportation_mode"], neuradp["recourse_variant"]) == ("integrated", "legacy")
    assert neuradp["learner_variant"] == neuradp["zone_distribution_mode"] == "neuradp"
    assert "learner-neuradp" in neuradp["checkpoint_suffix"]
    assert neuradp["neuradp_options"] == {"neuradp_samples_per_update": 5}
    assert r3["learner_variant"] == "optimization_anchored_residual"
    assert "learner-optimization_anchored_residual" in r3["checkpoint_suffix"]


@pytest.fixture(scope="module")
def nyc_learning_run():
    """Collect real integrated NYC transitions under the NeurADP policy."""
    from train_acceptance_model import make_environment, parse_args

    args = parse_args([
        "--environment", "nyc", "--num-vehicles", "6", "--num-ev", "3",
        "--mcmf-backend", "primal_dual", "--stop-hour", "8.1",
    ])
    env = make_environment(args, 301)
    env.evaluatemode = False
    env.adp_value = 1.0
    env.learner_variant = "neuradp"
    env.state_variant = "joint_state_shared_critic"
    env.neuradp_noise_scale = 0.0
    torch.manual_seed(0)
    value_function = NeurADP(
        grid_size=env.grid_size, num_vehicles=6, device="cpu",
        episode_length=env.episode_length, env=env,
    )
    value_function.state_variant = "joint_state_shared_critic"
    value_function.recourse_variant = "legacy"
    env.set_value_function(value_function)
    env.set_value_function_ev(value_function)
    scored = []
    original = value_function._graph_edge_scores

    def recording(graph, *, target_context):
        result = original(graph, target_context=target_context)
        scored.append((graph, result[0]))
        return result

    value_function._graph_edge_scores = recording
    for _ in range(6):
        actions, stores, stores_ev = env.simulate_motion_integrated_control()
        env.step(actions, stores, stores_ev)
    value_function._graph_edge_scores = original
    return env, value_function, scored


def test_rollout_scores_are_reward_plus_discounted_post_decision_value(nyc_learning_run):
    _env, value_function, scored = nyc_learning_run
    assert scored, "integrated control never asked the NeurADP learner for scores"
    graph, scores = scored[-1]
    assert set(scores) == {edge.edge_id for edge in graph.edges}
    values = value_function._values(graph, list(graph.edges), target=False).numpy()
    for edge, value in zip(graph.edges, values):
        terminal, outcomes = value_function._outcomes(edge)
        rows = [(edge.vehicle_id, x[1]) for x in outcomes]
        with torch.no_grad():
            arrays = value_function._record_rows(graph, rows, next_epoch=True)
            next_values = value_function.network(*(torch.as_tensor(a) for a in arrays)).numpy()
        expected = sum(x[0]*(x[2]-x[3]+.95*v*(not terminal)) for x, v in zip(outcomes, next_values))
        assert scores[edge.edge_id] == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_per_agent_td_update_uses_target_resolve_and_changes_weights(nyc_learning_run):
    _env, value_function, _scored = nyc_learning_run
    replay = value_function.joint_replay_buffer
    assert len(replay) >= 2
    ready = [row for row in replay if value_function._joint_row_ready(row, ifEV=False)]
    assert ready and all(row.mode == "integrated" and row.learner_variant == "neuradp" for row in ready)
    row = next(item for item in ready if not item.done)
    next_graph = value_function._temporal_successor_graph(row)
    targets = value_function._saved_graph_targets(next_graph)
    online_vehicles = {vehicle.vehicle_id for vehicle in next_graph.state.vehicles if vehicle.online}
    assert set(targets) <= {edge.vehicle_id for edge in next_graph.edges}
    assert set(targets) == online_vehicles & {edge.vehicle_id for edge in next_graph.edges}

    before = [parameter.detach().clone() for parameter in value_function.network.parameters()]
    target_before = [p.detach().clone() for p in value_function.target_network.parameters()]
    assert value_function.train_step(batch_size=64, ifEV=True) == 0.0
    loss = value_function.train_step(batch_size=64, ifEV=False)
    assert loss > 0.0 and np.isfinite(loss)
    assert value_function.training_step == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, value_function.network.parameters()))
    # Soft update: the target moved, but only partly (tau = 0.1).
    moved = [not torch.equal(a, b) for a, b in zip(target_before, value_function.target_network.parameters())]
    assert any(moved)


def test_response_aware_variant_takes_acceptance_expectation(nyc_learning_run):
    from dataclasses import replace

    _env, value_function, scored = nyc_learning_run
    graph = next(g for g, _ in reversed(scored) if any(
        e.vehicle_type == 1 and e.action_type == ActionType.SERVICE
        and not value_function._is_continuing(e) for e in g.edges))
    offer = next(e for e in graph.edges if e.vehicle_type == 1
                 and e.action_type == ActionType.SERVICE and not value_function._is_continuing(e))
    import json
    metadata = dict(offer.metadata)
    terminal, outcomes = value_function._outcomes(offer)
    first = outcomes[0]
    metadata['neuradp_epoch_outcomes'] = json.dumps([terminal, [
        [.7, first[1], first[2], first[3], False],
        [.3, first[1], -2., 0., True]]])
    offer = replace(offer, metadata=tuple(metadata.items()))
    graph = replace(graph, edges=tuple(offer if e.edge_id == offer.edge_id else e for e in graph.edges))
    value_function.neuradp_config['response_aware'] = False
    plain = value_function._neuradp_scores(graph, target=False)[offer.edge_id]
    value_function.neuradp_config['response_aware'] = True
    aware = value_function._neuradp_scores(graph, target=False)[offer.edge_id]
    stay = float(value_function._values(graph, [offer], target=False, stay=True)[0])
    assert aware == pytest.approx(.7*plain+.3*(-2.+.95*stay*(not terminal)), rel=1e-5)


def test_neuradp_checkpoint_round_trip(nyc_learning_run, tmp_path):
    env, value_function, scored = nyc_learning_run
    path = ADPTrainer._save_inference_checkpoint(value_function, 1, tmp_path)["full_state"]
    graph, _ = scored[-1]
    restored = NeurADP(grid_size=env.grid_size, num_vehicles=6, device="cpu",
                       episode_length=env.episode_length, env=env)
    restored.state_variant = "joint_state_shared_critic"
    restored.recourse_variant = "legacy"
    trainer = ADPTrainer.__new__(ADPTrainer)
    assert trainer.load_checkpoint(restored, str(path))
    env.evaluatemode = True  # no exploration noise in the comparison
    try:
        actual, _ = restored._graph_edge_scores(graph, target_context=False)
        online, _ = value_function._graph_edge_scores(graph, target_context=False)
        # Inference files store only the online network; the target is resynced.
        restored_target, _ = restored._graph_edge_scores(graph, target_context=True)
    finally:
        env.evaluatemode = False
    assert actual == pytest.approx(online)
    assert restored_target == pytest.approx(actual)


def test_neuradp_uses_joint_replay_without_unused_legacy_graph_roots(nyc_learning_run):
    _env, value_function, _scored = nyc_learning_run
    assert len(value_function.joint_replay_buffer) >= 2
    assert not value_function.experience_buffer
    assert not value_function.queue_experience_buffer

    # Direct legacy callers must not retain the graph through the bound action
    # either (the simulator skips these calls before constructing any rows).
    from src.Action import IdleAction
    action = IdleAction([], (0., 0.), (0., 0.), 0, .8)
    action.metadata.feasible_graph_snapshot = _scored[-1][0]
    value_function.set_replay_collection_context(action)
    value_function.store_experience(vehicle_id=0, vehicle_location=0, battery_level=.8)
    assert not value_function.experience_buffer
    assert value_function._replay_collection_context is None


@pytest.mark.parametrize('mode', ['current', 'real_conservative'])
@pytest.mark.parametrize('battery', [.2, .8])
def test_neuradp_executes_human_outside_action_with_non_manhattan_target(
    monkeypatch, tmp_path, mode, battery,
):
    import pandas as pd
    from src.NYCEnvironment import NYCEnvironment
    from src.real_conservative_charging import RealConservativeNYCEnvironment
    from src.recourse.state_snapshot import StateSnapshotBuilder

    demand = pd.DataFrame({
        'pickup_date': pd.to_datetime([]),
        'pickup_second_of_day': pd.Series(dtype=float),
        'PULocationID': pd.Series(dtype=int), 'DOLocationID': pd.Series(dtype=int),
    })
    monkeypatch.setattr(NYCEnvironment, '_load_demand_data', lambda self: demand.copy())
    monkeypatch.setenv('ICAPS_REPLAY_STORAGE', 'disk')
    monkeypatch.setenv('ICAPS_REPLAY_DIR', str(tmp_path))
    env_cls = RealConservativeNYCEnvironment if mode == 'real_conservative' else NYCEnvironment
    env = env_cls(num_vehicles=2, ev_num_vehicles=1, random_seed=301,
                  start_date='2025-12-08', end_date='2025-12-08',
                  start_hour=8, stop_hour=8.1, aev_charging_center_count=3,
                  usemcmf=True, mcmf_solver='exact', mcmf_backend='ortools',
                  daily_drop_off=False, ifreject=False)
    env.reset()
    env.evaluatemode = False
    env.adp_value = 1.0
    env.learner_variant = 'neuradp'
    env.state_variant = 'joint_state_shared_critic'
    env.configure_recourse_experiment('legacy')
    value_function = NeurADP(grid_size=env.grid_size, num_vehicles=2, device='cpu',
                             episode_length=env.episode_length, env=env,
                             neuradp_noise_scale=0.)
    value_function.state_variant = 'joint_state_shared_critic'
    env.set_value_function(value_function)
    env.set_value_function_ev(value_function)
    monkeypatch.setattr(env, '_ev_charging_phase', lambda actions, stores: None)
    monkeypatch.setattr(env, 'compute_ev_relocation_probability', lambda vid: (119, {119: 1.}))
    human_id = next(vid for vid in env.vehicles if env._is_ev(vid))
    vehicle = env.vehicles[human_id]
    # At the service floor the outside action stays in its current non-hotspot
    # zone; otherwise its cached sampled destination is that same valid zone.
    origin = 119 if battery == .2 else 161
    vehicle.update(location=origin, zone_id=origin, coordinates=env.zone_coords[origin],
                   battery=battery, target_location=None, idle_target=None)
    assert 119 not in env.hotspot_locations
    for _ in range(4):
        actions, stores, stores_ev = env.simulate_motion_integrated_control()
        graph = env._last_integrated_repair_graphs[0]
        selected = next(edge for edge in graph.edges
                        if edge.vehicle_id == human_id and edge.edge_id in graph.selected_edge_ids)
        assert selected.action_type == ActionType.RELOCATE
        assert selected.target_location == vehicle['target_location'] == 119
        assert actions[human_id].target_coords == tuple(env.zone_coords[119])
        assert env.storeactions[human_id] is None
        assert env.storeactions_ev[human_id] is actions[human_id]
        assert selected.edge_id in StateSnapshotBuilder.selected_edge_ids(graph, {human_id: 'reloc_to_119'})
        env.step(actions, stores, stores_ev)
    assert len(value_function.joint_replay_buffer) == 4
    assert not value_function.experience_buffer
    assert value_function.train_step(batch_size=2) > 0


def test_neuradp_real_conservative_trainer_saves_and_reloads_without_legacy_replay(
    tmp_path, monkeypatch,
):
    import pandas as pd
    from src.NYCEnvironment import NYCEnvironment
    from src.real_conservative_charging import RealConservativeNYCEnvironment
    from test_real_conservative_nyc_cli import base_kwargs

    demand = pd.DataFrame([
        dict(pickup_date=pd.Timestamp('2025-12-15'), pickup_second_of_day=second,
             PULocationID=161, DOLocationID=162, fare_amount=15., trip_distance=.1)
        for second in range(0, 360, 30)
    ])
    monkeypatch.setattr(NYCEnvironment, '_load_demand_data', lambda self: demand.copy())
    monkeypatch.setattr(run_nyctrainer, 'resolve_nyc_parquet_paths', lambda *a, **k: [])
    monkeypatch.setattr(ADPTrainer, '_save_episode_stats_to_excel', lambda *a, **k: (None, None))
    monkeypatch.setenv('ICAPS_REPLAY_STORAGE', 'disk')
    monkeypatch.setenv('ICAPS_REPLAY_DIR', str(tmp_path / 'replay'))
    monkeypatch.chdir(tmp_path)
    kwargs = base_kwargs()
    kwargs.update(transportation_mode='integrated', recourse_variant='legacy',
                  learner_variant='neuradp', zone_distribution_mode='neuradp',
                  state_variant='joint_state_separate_critics', mcmf_backend='ortools',
                  mcmf_graph_reduction=True, neuradp_options={'neuradp_noise_scale': 0.})
    trained, env = run_nyctrainer.run_nyc_training(**kwargs)
    assert isinstance(env, RealConservativeNYCEnvironment)
    assert trained['optimizer_budget']['optimizer_steps_total'] > 0
    assert len(env.value_function.joint_replay_buffer) == 12
    for vf in (env.value_function, env.value_function_ev):
        assert not vf.experience_buffer and not vf.queue_experience_buffer
    paths = list((tmp_path / 'checkpoints' / 'charge-real-conservative').glob('*/best_full_state_episode_1.pth'))
    assert len(paths) == 2
    kwargs.update(trainnetwork=False, ifloadcheckpoint=True, charging_model=None,
                  conservative_charging=None, checkpoint_charging_model='real_conservative',
                  checkpoint_selection='best_reward')
    evaluated, restored = run_nyctrainer.run_nyc_training(**kwargs)
    assert evaluated['charging_model'] == 'real_conservative'
    assert evaluated['optimizer_budget']['optimizer_steps_total'] == 0
    assert not restored.value_function.joint_replay_buffer
    for name, tensor in env.value_function.network.state_dict().items():
        assert torch.equal(tensor, restored.value_function.network.state_dict()[name])
