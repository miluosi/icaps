"""Default-off rejection prediction, including the historical Bayes path."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from src.ADPtrainer import ADPTrainer
from src.Environment import ChargingIntegratedEnvironment
from src.acceptance_features import configure_acceptance_feature, predicted_rejection
from src.value_function_registry import VALUE_FUNCTION_CHOICES, get_value_function_class


@pytest.mark.parametrize("entrypoint", ["run_trainer", "run_nyctrainer"])
def test_training_cli_defaults_leave_rejection_prediction_off(entrypoint, monkeypatch):
    monkeypatch.setattr("sys.argv", [entrypoint + ".py"])
    args = importlib.import_module(entrypoint).parse_args()
    assert args.ev_acceptance_feature == "off"
    assert args.ev_acceptance_model is None


def test_default_configuration_does_not_load_model_or_disable_real_rejection():
    env = SimpleNamespace(ifreject=True, reject_uniform=True)
    configure_acceptance_feature(env, model_path="nonexistent-model-must-not-be-opened.json")
    assert env.ev_acceptance_model is None
    assert env.ev_response_model_hash is None
    assert predicted_rejection(env, 0, object()) == 0.0
    assert env.ifreject and env.reject_uniform


@pytest.mark.parametrize("mode", VALUE_FUNCTION_CHOICES)
def test_every_learner_defaults_to_no_response_model_or_probability_input(mode, tmp_path):
    value = get_value_function_class(mode)(
        grid_size=2, num_vehicles=2, episode_length=10,
        zone_distribution_mode=mode, log_dir=str(tmp_path / mode),
    )
    try:
        assert not value.response_enabled
        assert not value.acceptance_input_enabled
        assert not value.response_anchor_enabled
        assert value.response_model is None
        assert getattr(value, "rejection_predictor", None) is None
        if hasattr(value, "train_rejection_predictor"):
            # Even sufficient, deliberately malformed labels cannot trigger
            # feature extraction or training while this predictor is disabled.
            value.rejection_buffer.extend({"was_rejected": bool(i % 2)} for i in range(64))
            assert value.train_rejection_predictor(batch_size=8) is None
            assert not value.rejection_predictor_trained
            assert value.rejection_training_losses == []
            assert value.rejection_optimizer is None
            assert value.predict_rejection_probability(0, 1, 0, 1, 0) == 0.0
            assert value._calculate_rejection_risk_penalty(0, 25.0) == 0.0
    finally:
        writer = getattr(value, "writer", None)
        if writer is not None:
            writer.close()


def test_environment_keeps_outcome_export_without_starting_disabled_predictor():
    exported = []

    def unexpected_training(**kwargs):
        pytest.fail("disabled legacy predictor must not receive training calls")

    outcomes = [{"was_rejected": True}, {"was_rejected": False}]
    value = SimpleNamespace(
        rejection_buffer=outcomes, rejection_predictor=None,
        train_rejection_predictor=unexpected_training,
    )
    env = SimpleNamespace(
        value_function_ev=value, _rejection_save_counter=9,
        _save_rejection_acceptance_data=exported.append,
    )
    ChargingIntegratedEnvironment._save_and_train_rejection_predictor(env)
    assert exported == [outcomes]


def test_loading_legacy_predictor_weights_does_not_reenable_disabled_model(tmp_path):
    value_class = get_value_function_class("none")
    legacy = value_class(
        grid_size=2, num_vehicles=2, zone_distribution_mode="none",
        log_dir=str(tmp_path / "legacy"), enable_legacy_rejection_predictor=True,
    )
    disabled = value_class(
        grid_size=2, num_vehicles=2, zone_distribution_mode="none",
        log_dir=str(tmp_path / "disabled"),
    )
    try:
        assert legacy.rejection_predictor is not None
        checkpoint = tmp_path / "legacy_full_state.pth"
        torch.save({
            "network_state_dict": legacy.network.state_dict(),
            "target_network_state_dict": legacy.target_network.state_dict(),
            "rejection_predictor_state_dict": legacy.rejection_predictor.state_dict(),
            "rejection_optimizer_state_dict": legacy.rejection_optimizer.state_dict(),
            "rejection_predictor_trained": True,
        }, checkpoint)
        assert ADPTrainer().load_checkpoint(disabled, str(checkpoint))
        assert disabled.rejection_predictor is None
        assert disabled.rejection_optimizer is None
        assert not disabled.rejection_predictor_trained
    finally:
        legacy.writer.close()
        disabled.writer.close()
