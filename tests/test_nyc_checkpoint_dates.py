from pathlib import Path

import pytest
import torch

import test_nyc_model
from src.ADPtrainer import ADPTrainer
from src.NYCtrainer import NYCTrainer


def directories(start="2025-12-08", end="2025-12-10", suffix="method-r3"):
    return NYCTrainer._checkpoint_dirs(
        transportation_mode="evfirst", assignmentgurobi=True, num_ev=1500,
        use_intense_requests=True, start_date=start, end_date=end,
        zone_distribution_mode="optimization_anchored_residual",
        only_manhattan_zones=True, checkpoint_suffix=suffix,
    )


def test_new_saves_normalize_single_digit_dates():
    assert directories("2025-12-8") == directories("2025-12-08")
    assert "_20250108_20250209_" in directories("2025-1-8", "2025-2-9")[0]


@pytest.mark.parametrize("start,end", [
    ("2025-12-08", "2025-12-10"), ("2025-01-08", "2025-02-09"),
])
def test_test_precheck_and_loader_resolve_same_legacy_pair(tmp_path, monkeypatch, start, end):
    monkeypatch.chdir(tmp_path)
    from datetime import datetime
    old_tokens = []
    for value in (start, end):
        date = datetime.strptime(value, "%Y-%m-%d")
        old_tokens.append(f"{date.year}{date.month}{date.day}")
    canonical_token = f"_{start.replace('-', '')}_{end.replace('-', '')}_"
    legacy_token = "_" + "_".join(old_tokens) + "_"
    canonical = directories(start, end)
    legacy = [p.replace(canonical_token, legacy_token) for p in canonical]
    for directory in legacy:
        Path(directory).mkdir(parents=True)
        torch.save({"episode": 2, "checkpoint_tag": "best", "checkpoint_pair_id": "run:best:2"},
                   Path(directory) / "best_full_state_episode_2.pth")
    for role, expected in zip(("ev", "aev"), legacy):
        actual = test_nyc_model.build_checkpoint_dir(
            "gurobi", "evfirst", 1500, True, role, start, end,
            "optimization_anchored_residual", True, checkpoint_suffix="method-r3",
        )
        assert actual == expected
        assert ADPTrainer.find_latest_checkpoint(actual, prefer_best=True)
    loaded_dirs = [NYCTrainer._resolve_checkpoint_dir(p, start, end) for p in canonical]
    pair = ADPTrainer.find_checkpoint_pair(*loaded_dirs, prefer_best=True)
    assert pair == tuple(str(Path(p) / "best_full_state_episode_2.pth") for p in legacy)
    assert not any(Path(p).exists() for p in canonical)  # Lookup never renames data.


def test_canonical_wins_and_legacy_ambiguity_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    canonical = directories("2025-01-08", "2025-02-09")[0]
    for token in ("_2025018_2025029_", "_202518_202529_"):
        Path(canonical.replace("_20250108_20250209_", token)).mkdir(parents=True)
    with pytest.raises(ValueError, match="Ambiguous legacy"):
        NYCTrainer._resolve_checkpoint_dir(canonical, "2025-01-08", "2025-02-09")
    Path(canonical).mkdir()
    assert NYCTrainer._resolve_checkpoint_dir(canonical, "2025-1-8", "2025-2-9") == canonical


def test_date_fallback_does_not_cross_method_namespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = directories(suffix="method-r1")[0].replace("_20251208_", "_2025128_")
    Path(other).mkdir(parents=True)
    requested = directories(suffix="method-r3")[0]
    assert NYCTrainer._resolve_checkpoint_dir(requested, "2025-12-08", "2025-12-10") == requested


def test_missing_all_checkpoints_exits_without_overwriting_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = Path("results/test_model/test_results_4way_optimization_anchored_residual.npy")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"previous valid results")

    def unexpected_training(**kwargs):
        pytest.fail("Missing models should not start a rollout")

    monkeypatch.setattr(test_nyc_model, "run_nyc_training", unexpected_training)
    with pytest.raises(SystemExit, match="No evaluation results"):
        test_nyc_model.main([
            "--methods", "r3", "--strategies", "ADP-MCMF",
            "--load-model-start-date", "2025-12-08",
            "--load-model-end-date", "2025-12-10",
            "--aev-charging-center-count", "3",
        ])
    assert output.read_bytes() == b"previous valid results"
    assert not list(output.parent.glob("*.xlsx"))
