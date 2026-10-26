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
        continuing = value_function._is_continuing(edge)
        reward = (
            edge.structured_score if (not continuing or edge.action_type == ActionType.WAIT)
            else 0.0
        )
        if not continuing and edge.success_structured_score is not None and (
                edge.expected_response_anchor or edge.human_response_mask):
            reward = edge.success_structured_score
        assert scores[edge.edge_id] == pytest.approx(reward + 0.95 * float(value), rel=1e-5, abs=1e-5)


def test_per_agent_td_update_uses_target_resolve_and_changes_weights(nyc_learning_run):
    _env, value_function, _scored = nyc_learning_run
    replay = value_function.joint_replay_buffer
    assert len(replay) >= 2
    ready = [row for row in replay if value_function._joint_row_ready(row, ifEV=False)]
    assert ready and all(row.mode == "integrated" and row.learner_variant == "neuradp" for row in ready)
    row = next(item for item in ready if not item.done)
    next_graph = value_function._temporal_successor_graph(row)
    targets = value_function._next_epoch_targets(next_graph)
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
    success = float(offer.success_structured_score
                    if offer.success_structured_score is not None else offer.structured_score)
    offer = replace(offer, human_response_mask=True, rejection_probability=0.3,
                    rejection_structured_score=-2.0, success_structured_score=success,
                    expected_response_anchor=True, structured_score=0.7 * success + 0.3 * -2.0)
    graph = replace(graph, edges=tuple(offer if e.edge_id == offer.edge_id else e for e in graph.edges))
    plain = value_function._neuradp_scores(graph, target=False)[offer.edge_id]
    value_function.neuradp_config["response_aware"] = True
    try:
        aware = value_function._neuradp_scores(graph, target=False)[offer.edge_id]
    finally:
        value_function.neuradp_config["response_aware"] = False
    stay = float(value_function._values(graph, [offer], target=False, stay=True)[0])
    assert aware == pytest.approx(0.7 * plain + 0.3 * (-2.0 + 0.95 * stay), rel=1e-5)


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
