from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import run_assignment_solver_audit
import run_assignment_state_experiment
import run_fixed_graph_exact_solver_audit
import run_recourse_panel
from run_acceptance_ablation import save_pair, weight_hash
from run_recourse_day import validate_checkpoint_payload
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction
from src.recourse.cluster_stats import (
    summarize_cluster_metric,
    summarize_paired_cluster_difference,
)
from src.recourse.config import (
    CAUSAL_CONTRASTS,
    CAUSAL_PREDICTOR_VARIANTS,
    METHODS,
    VARIANT_CHOICES,
    method_metadata,
)
from src.recourse.contracts import evaluate_method_event_contract
from src.recourse.crn import vehicle_normal, vehicle_uniform
from test_recourse_must_fix import _graph
from test_recourse_reaudit import _single_edge_graph, _transition, _value_function


def test_causal_contrast_uses_structured_no_repair():
    assert CAUSAL_CONTRASTS[0] == (
        "evfirst_no_repair_structured",
        "repair_only",
    )
    assert {"r1_structured", "r2", "r3", "recourse_macro", "r4"} <= (
        CAUSAL_PREDICTOR_VARIANTS
    )
    assert "samitha" not in VARIANT_CHOICES
    assert "no_repair" not in VARIANT_CHOICES


def test_actual_graph_scorer_receives_strict_masked_state(monkeypatch):
    graph = _graph("strict-score", stage=1, vehicle_id=0, vehicle_type=1)
    focal = graph.state.vehicles[0]
    other = replace(focal, vehicle_id=1, vehicle_type=2, online=True, location=7)
    graph = replace(graph, state=replace(graph.state, vehicles=(focal, other)))
    value = PyTorchChargingValueFunction(
        grid_size=2,
        num_vehicles=2,
        episode_length=10,
        max_requests=10,
        neighbour_number=0,
    )
    value.state_variant = "strict_fleet_local_separate_critics"
    value._joint_critic_router = {1: value, 2: value}
    seen = []
    original = value._edge_tensor_from_experience

    def capture(exp, *args, **kwargs):
        seen.append(kwargs.get("state_snapshot"))
        return original(exp, *args, **kwargs)

    monkeypatch.setattr(value, "_edge_tensor_from_experience", capture)
    value._graph_edge_scores(graph, target_context=False)
    assert seen
    assert all(len(snapshot.vehicles) == 1 for snapshot in seen)
    assert all(snapshot.vehicles[0].vehicle_type == 1 for snapshot in seen)


def test_strong_contracts_require_physical_repair():
    common = dict(
        eligible_rejected_residual_count=1,
        same_epoch_aev_assignment_count=0,
        aev_follower_optimizer_steps=1,
        aev_learned_score_difference_count=1,
        macro_leader_target_count=1,
        nested_leader_target_count=1,
        follower_target_query_count=1,
    )
    for method in ("repair_learning", "recourse_macro", "recourse_nested_q2"):
        result = evaluate_method_event_contract(method, common)
        assert not result.passed
        assert "same_epoch_repair_present" in result.failures
    samitha = evaluate_method_event_contract(
        "samitha",
        dict(
            hold_candidate_count=1,
            hold_selected_count=1,
            repair_candidate_rejected_count=1,
            samitha_repair_assignment_count=0,
            committed_aev_reassignment_count=0,
        ),
    )
    assert not samitha.passed
    assert "repair_assignment_present" in samitha.failures


def test_cluster_summary_counts_independent_trained_models():
    rows = [
        dict(seed=1, train_day="d1", day_id="a", method="base", reward=1),
        dict(seed=1, train_day="d1", day_id="b", method="base", reward=3),
        dict(seed=2, train_day="d1", day_id="a", method="base", reward=5),
        dict(seed=2, train_day="d1", day_id="b", method="base", reward=7),
    ]
    result = summarize_cluster_metric(rows, "reward")
    assert result["count"] == 2
    assert result["observation_count"] == 4
    assert result["mean"] == pytest.approx(4.0)

    paired = rows + [
        {**row, "method": "treatment", "reward": row["reward"] + 2}
        for row in rows
    ]
    difference = summarize_paired_cluster_difference(
        paired,
        "reward",
        baseline="base",
        treatment="treatment",
    )
    assert difference["count"] == 2
    assert difference["paired_day_count"] == 4
    assert difference["mean"] == pytest.approx(2.0)


def test_panel_aggregate_has_one_method_identity_and_cluster_fields(tmp_path):
    cluster = dict(seed=1, train_day="2025-12-18", test_day="2025-12-19")
    output = tmp_path / "panel"
    folder = output / "seed-1-train-2025-12-18-test-2025-12-19"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(
        __import__("json").dumps(
            {
                "runs": [
                    {
                        "method": "repair_only",
                        "trained_weight_hash": "same-policy",
                        "testing": {"method": "must-not-duplicate", "reward": 4.0},
                        "event_contract": {"passed": True},
                    }
                ]
            }
        )
    )
    args = SimpleNamespace(output_dir=output, methods=["repair_only"])
    result = run_recourse_panel.aggregate(args, [cluster])
    assert result["rows"][0]["method"] == "repair_only"
    assert result["cluster_fields"] == ["seed", "train_day"]
    assert result["independent_model_count"] == 1


def test_weight_hash_includes_target_and_auxiliary_parameters():
    learner = SimpleNamespace(
        network=torch.nn.Linear(2, 1),
        critic2=torch.nn.Linear(2, 1),
        target_network=torch.nn.Linear(2, 1),
        target_critic2=torch.nn.Linear(2, 1),
        queue_predictor=torch.nn.Linear(2, 1),
        log_alpha=torch.nn.Parameter(torch.tensor(0.0)),
    )
    before = weight_hash([learner])
    learner.target_network.weight.data.add_(1.0)
    assert weight_hash([learner]) != before
    before = weight_hash([learner])
    learner.queue_predictor.bias.data.add_(1.0)
    assert weight_hash([learner]) != before


def test_crn_records_entire_event_stream_independent_of_call_order():
    def environment():
        return SimpleNamespace(
            common_random_numbers=True,
            initial_random_seed=71,
            recourse_run_id="shared-run",
            cumulative_episode_index=0,
            episode_day_index=0,
            current_time=30.0,
        )

    left, right = environment(), environment()
    left_values = {
        "reward": vehicle_normal(
            left, 4, "service_reward", 2.0, request_id=99, attempt_index=0
        ),
        "offer": vehicle_uniform(left, 4, "ev_offer"),
    }
    right_values = {
        "offer": vehicle_uniform(right, 4, "ev_offer"),
        "reward": vehicle_normal(
            right, 4, "service_reward", 2.0, request_id=99, attempt_index=0
        ),
    }
    assert left_values == right_values
    assert left._recourse_random_events == right._recourse_random_events


def test_checkpoint_identity_is_checked_before_tensor_loading():
    method = "repair_only"
    spec = METHODS[method]
    env = SimpleNamespace(
        state_variant="joint_state_separate_critics",
        learner_variant="optimization_anchored_residual",
        mcmf_solver="exact",
        mcmf_backend="primal_dual",
        mcmf_graph_reduction=True,
        mcmf_verify=True,
        mcmf_strict=True,
        mcmf_cost_scale=10_000,
        target_solver_policy="same_as_rollout_exact",
    )
    metadata = {
        "method": method,
        **method_metadata(spec.operating_mode, spec.variant),
        "state_variant": env.state_variant,
        "learner_variant": env.learner_variant,
        "solver_config": {
            "rollout_solver": "exact",
            "backend": "primal_dual",
            "graph_reduction": True,
            "verify": True,
            "strict": True,
            "cost_scale": 10_000,
            "target_policy": "same_as_rollout_exact",
        },
    }
    payload = {
        "checkpoint_schema_version": 2,
        "metadata": metadata,
        "learners": [
            dict(network={}, target={}, optimizer={}, extra={}) for _ in range(2)
        ],
    }
    assert validate_checkpoint_payload(payload, method, env) == metadata
    incompatible = deepcopy(payload)
    incompatible["metadata"]["state_variant"] = "strict_fleet_local_separate_critics"
    with pytest.raises(ValueError, match="state_variant mismatch"):
        validate_checkpoint_payload(incompatible, method, env)


def test_integrated_and_samitha_auction_labels_fail_closed(tmp_path):
    common = [
        "--checkpoint", str(tmp_path / "unused.pt"),
        "--parquet-path", str(tmp_path / "unused.parquet"),
        "--solvers", "auction",
        "--seeds", "1",
        "--dates", "2025-12-19",
        "--output-dir", str(tmp_path / "out"),
    ]
    for method in ("no_repair", "samitha"):
        with pytest.raises(SystemExit):
            run_assignment_solver_audit.parse_args(
                ["--recourse-method", method, *common]
            )


def test_state_experiment_dry_run_and_fixed_graph_audit(tmp_path, capsys):
    run_assignment_state_experiment.main(
        [
            "--state-variants", "joint_state_separate_critics",
            "--train-days", "2025-12-18",
            "--test-days", "2025-12-19",
            "--seeds", "71",
            "--parquet-path", str(tmp_path / "data.parquet"),
            "--output-dir", str(tmp_path / "state"),
            "--dry-run",
        ]
    )
    plan = __import__("json").loads(capsys.readouterr().out)
    assert "--state-variant" in plan["commands"][0]
    assert "joint_state_separate_critics" in plan["commands"][0]

    graph = _single_edge_graph(
        graph_id="fixed-graph",
        stage=2,
        vehicle_id=1,
        vehicle_type=2,
        structured=3.0,
    )
    checkpoint = tmp_path / "fixed.pt"
    torch.save(
        {
            "checkpoint_schema_version": 2,
            "learners": [
                {
                    "extra": {
                        "joint_replay_state_dict": {
                            "items": [
                                _transition("fixed", aev_graph=graph, done=True)
                            ]
                        }
                    }
                }
            ],
        },
        checkpoint,
    )
    output = tmp_path / "fixed.json"
    run_fixed_graph_exact_solver_audit.main(
        [
            "--checkpoint", str(checkpoint),
            "--backends", "primal_dual",
            "--reductions", "on", "off",
            "--output", str(output),
        ]
    )
    fixed = __import__("json").loads(output.read_text())
    assert fixed["graph_count"] == 1
    assert fixed["comparisons"][0]["objective_gap"] == 0
    assert fixed["comparisons"][0]["selected_edge_agreement"]


def _restore_value_function(saved):
    value = _value_function()
    value.checkpoint_replay = "full"
    value.network.load_state_dict(saved["network"])
    value.target_network.load_state_dict(saved["target"])
    value.optimizer.load_state_dict(saved["optimizer"])
    value.load_extra_checkpoint_state(saved["extra"])
    return value


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_interrupted_resume_matches_uninterrupted_joint_training(tmp_path):
    initial = _value_function()
    initial.checkpoint_replay = "full"
    for index in range(4):
        graph = _single_edge_graph(
            graph_id=f"resume-{index}",
            stage=2,
            vehicle_id=1,
            vehicle_type=2,
            structured=float(index + 1),
        )
        initial.store_recourse_transition(
            replace(
                _transition(
                    f"run:episode:0:sequence:{index}",
                    aev_graph=graph,
                    done=True,
                ),
                transition_sequence_index=index,
            )
        )
    snapshot = dict(
        network=deepcopy(initial.network.state_dict()),
        target=deepcopy(initial.target_network.state_dict()),
        optimizer=deepcopy(initial.optimizer.state_dict()),
        extra=deepcopy(initial.extra_checkpoint_state()),
    )
    uninterrupted = _restore_value_function(snapshot)
    interrupted = _restore_value_function(snapshot)
    for _ in range(4):
        uninterrupted._train_joint_step(1, ifEV=False)
    for _ in range(2):
        interrupted._train_joint_step(1, ifEV=False)

    checkpoint = tmp_path / "resume.pt"
    save_pair([interrupted], checkpoint, {"method": "resume-test"})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["checkpoint_schema_version"] == 2
    assert len(payload["learners"][0]["extra"]["joint_replay_state_dict"]["items"]) == 4
    resumed = _restore_value_function(payload["learners"][0])
    for _ in range(2):
        resumed._train_joint_step(1, ifEV=False)

    assert weight_hash([uninterrupted]) == weight_hash([resumed])
    _assert_nested_equal(
        uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    )
    assert uninterrupted.joint_replay_buffer.priorities == pytest.approx(
        resumed.joint_replay_buffer.priorities
    )
    assert uninterrupted.joint_replay_buffer.beta == resumed.joint_replay_buffer.beta
    assert uninterrupted.joint_training_diagnostics == resumed.joint_training_diagnostics
