from types import SimpleNamespace

import pytest

import run_nyctrainer
import test_nyc_model
from src.recourse.config import (
    ICAPS_METHODS,
    METHODS,
    canonical_method,
    method_checkpoint_suffix,
    resolve_method_list_arguments,
)


EXPECTED = {
    "r0": ("evfirst", "r0"),
    "r1": ("evfirst", "r1"),
    "r2": ("evfirst", "r2"),
    "r3": ("evfirst", "r3"),
    "r4": ("evfirst", "r4"),
    "macro": ("evfirst", "recourse_macro"),
    "samitha": ("integrated_repair", "legacy"),
}


def test_public_icaps_method_list_and_internal_mapping():
    assert ICAPS_METHODS == ("r0", "r1", "r2", "r3", "r4", "macro", "samitha")
    assert run_nyctrainer.NYC_TRAIN_METHODS == ICAPS_METHODS
    assert test_nyc_model.NYC_TEST_METHODS == ICAPS_METHODS
    assert resolve_method_list_arguments(["all"]) == list(ICAPS_METHODS)
    for public_name, expected in EXPECTED.items():
        spec = METHODS[canonical_method(public_name)]
        assert (spec.operating_mode, spec.variant) == expected


@pytest.mark.parametrize("entrypoint", [run_nyctrainer, test_nyc_model])
def test_train_and_test_parsers_accept_methods_and_reject_legacy_mode_cli(entrypoint):
    args = entrypoint.parse_args(["--methods", "r0", "r4", "macro", "samitha"])
    assert args.methods == ["r0", "r4", "macro", "samitha"]
    assert not hasattr(args, "transportation_mode")
    assert not hasattr(args, "transportation_modes")
    assert not hasattr(args, "recourse_variant")
    assert not hasattr(args, "all_modes")
    with pytest.raises(SystemExit):
        entrypoint.parse_args(["--transportation-mode", "evfirst"])
    with pytest.raises(SystemExit):
        entrypoint.parse_args(["--transportation-modes", "aevfirst"])


def test_train_and_test_share_checkpoint_namespace():
    suffixes = {
        method: method_checkpoint_suffix(
            "experiment",
            method,
            state_variant="joint_state_separate_critics",
            learner_variant="optimization_anchored_residual",
            rejection_logit_shift=0.0,
        )
        for method in ICAPS_METHODS
    }
    assert len(set(suffixes.values())) == len(ICAPS_METHODS)
    assert "method-r4_rec-r4" in suffixes["r4"]
    assert "method-macro_rec-recourse_macro" in suffixes["macro"]


def test_run_nyctrainer_all_dispatches_the_seven_registered_methods(monkeypatch):
    calls = []

    def fake_training(**kwargs):
        calls.append(kwargs)
        return {"episode_rewards": []}, SimpleNamespace(parquet_path=())

    monkeypatch.setattr(run_nyctrainer, "run_nyc_training", fake_training)
    run_nyctrainer.main([
        "--methods", "all",
        "--episodes", "1",
        "--start-date", "2025-12-18",
        "--aev-charging-center-count", "3",
        "--no-ifreject",
    ])
    assert [
        (call["transportation_mode"], call["recourse_variant"])
        for call in calls
    ] == [EXPECTED[method] for method in ICAPS_METHODS]
    assert all(
        f"method-{method}_" in call["checkpoint_suffix"]
        for method, call in zip(ICAPS_METHODS, calls)
    )
    assert all(call["aev_charging_center_count"] == 3 for call in calls)
    assert all("aev-centers-3" in call["checkpoint_suffix"] for call in calls)
