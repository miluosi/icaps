"""Test-only checkpoint round trips must preserve deployed Q and exact actions."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.ADPtrainer import ADPTrainer
from src.ValueFunction_st_masac_gat import PyTorchChargingValueFunction as GAT
from src.ValueFunction_st_masac_gat_post_demand import PyTorchChargingValueFunction as Demand
from src.ValueFunction_st_masac_gat_post_demand_direct import PyTorchChargingValueFunction as Direct
from src.recourse.target_builder import RecourseTargetBuilder
from src.recourse.types import ActionType
from test_recourse_must_fix import _graph


def make_value(value_class=Direct):
    return value_class(grid_size=3, num_vehicles=2, device="cpu",
                       episode_length=10, max_requests=10, neighbour_number=0)


@pytest.mark.parametrize("value_class", [GAT, Demand, Direct])
@pytest.mark.parametrize("variant", ["r1", "r3", "recourse_macro"])
def test_inference_round_trip_never_reads_replay_and_preserves_q(tmp_path, monkeypatch, value_class, variant):
    torch.manual_seed(9)
    source = make_value(value_class)
    source.recourse_variant = variant
    source.training_step = 0
    source.joint_training_step = 37
    source.beta_warmup_steps = 100
    source.queue_predictor_trained = True
    source.recent_station_waits = {2: 7.5}
    if hasattr(source, "post_demand_predictor_trained"):
        source.post_demand_predictor_trained = True
    if hasattr(source.network, "action_weights"):
        with torch.no_grad():
            source.network.action_weights.copy_(torch.tensor([.7, -.2, .4]))
            source.critic2.action_weights.copy_(torch.tensor([.3, -.1, .5]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference save accessed training/replay state")

    monkeypatch.setattr(source, "extra_checkpoint_state", forbidden)
    monkeypatch.setattr(source.joint_replay_buffer, "state_dict", forbidden)
    monkeypatch.setattr(source.optimizer, "state_dict", forbidden)
    source.joint_training_diagnostics = [forbidden]
    source.q_values_history = [forbidden]
    graph = _graph("roundtrip", stage=2, vehicle_id=1, vehicle_type=2)
    edges = tuple(replace(graph.edges[0], edge_id=f"edge-{i}",
                          action_type=ActionType.WAIT if i == 0 else ActionType.RELOCATE,
                          action_id="wait" if i == 0 else f"reloc-{i}",
                          target_location=i, post_action_location=i,
                          structured_score=1. + .05*i, post_demand_feature=.8)
                  for i in range(3))
    graph = replace(graph, edges=edges)
    before = source._graph_edge_scores(graph, target_context=False)
    path = ADPTrainer._save_q_network_checkpoint(source, 2, str(tmp_path), inference_only=True)["full_state"]
    saved = torch.load(path, weights_only=False)
    assert saved["checkpoint_format"] == "inference_v1"
    assert "optimizer_state_dict" not in saved
    assert "joint_replay_state_dict" not in saved["extra_value_function_state"]
    assert Path(path).stat().st_size < 2_000_000

    torch.manual_seed(123)
    restored = make_value(value_class)
    restored.recourse_variant = variant
    restored.beta_warmup_steps = 100
    loader = ADPTrainer.__new__(ADPTrainer)
    assert loader.load_checkpoint(restored, path)
    after = restored._graph_edge_scores(graph, target_context=False)
    assert before == after
    assert source._beta() == restored._beta()
    assert restored.queue_predictor_trained
    assert restored.recent_station_waits == {2: 7.5}
    if hasattr(source, "post_demand_predictor_trained"):
        assert restored.post_demand_predictor_trained
    solver = RecourseTargetBuilder(backend="ortools", graph_reduction=True)
    assert solver.project(graph, before[0]) == solver.project(graph, after[0])


def test_episode_retention_best_pair_and_failed_save_preserves_old_best(tmp_path, monkeypatch):
    value = make_value()
    for tag in ("latest", "best"):
        for episode in (1, 2):
            for role in ("ev", "aev"):
                ADPTrainer._save_q_network_checkpoint(
                    value, episode, str(tmp_path / role), checkpoint_tag=tag,
                    checkpoint_metadata={"checkpoint_pair_id": f"run:{tag}:{episode}"},
                    inference_only=True,
                )
    for role in ("ev", "aev"):
        folder = tmp_path / role
        assert len(list(folder.glob("full_state_episode_*.pth"))) == 2
        assert len(list(folder.glob("best_full_state_episode_*.pth"))) == 1
        assert not list(folder.glob("network_episode_*.pth"))
    pair = ADPTrainer.find_checkpoint_pair(str(tmp_path / "ev"), str(tmp_path / "aev"), prefer_best=True)
    assert all(Path(path).name == "best_full_state_episode_2.pth" for path in pair)
    previous_bytes = Path(pair[0]).read_bytes()
    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        ADPTrainer._save_q_network_checkpoint(value, 3, str(tmp_path / "ev"), checkpoint_tag="best", inference_only=True)
    assert Path(pair[0]).read_bytes() == previous_bytes
    assert not list((tmp_path / "ev").glob("*.tmp"))
