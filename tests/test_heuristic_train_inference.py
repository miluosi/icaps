import inspect

import pytest

from src.ADPtrainer import ADPTrainer
from src.NYCtrainer import NYCTrainer
from test_model import STRATEGIES as SYNTHETIC_STRATEGIES
from test_nyc_model import STRATEGIES as NYC_STRATEGIES


def _strategy(strategies, name):
    if isinstance(strategies, dict):
        return strategies[name]
    return next(strategy for strategy in strategies if strategy["name"] == name)


def test_heuristic_assignment_can_train_value_function():
    assert ADPTrainer._should_train_value_function(1.0, True)
    assert not ADPTrainer._should_train_value_function(0.0, True)
    assert not ADPTrainer._should_train_value_function(1.0, False)


def test_legacy_adp_heu_still_defaults_to_exact_trained_checkpoint():
    parameter = inspect.signature(
        ADPTrainer.run_charging_integration_test
    ).parameters["load_checkpoint_assign_tag"]
    assert parameter.default == "gurobi"


@pytest.mark.parametrize("strategies", (SYNTHETIC_STRATEGIES, NYC_STRATEGIES))
def test_exact_trained_and_heuristic_trained_modes_are_distinct(strategies):
    legacy = _strategy(strategies, "ADP-HEU")
    heuristic_trained = _strategy(strategies, "ADP-HEU-HEU")

    assert legacy["checkpoint_assign_tag"] == "gurobi"
    assert heuristic_trained["checkpoint_assign_tag"] == "heu"
    assert legacy.get("gurobi", False) is False
    assert heuristic_trained.get("gurobi", False) is False
    assert legacy.get("usemcmf", False) is False
    assert heuristic_trained.get("usemcmf", False) is False


def test_nyc_training_checkpoint_names_are_separate():
    exact_dirs = NYCTrainer._checkpoint_dirs(
        transportation_mode="integrated",
        assignmentgurobi=True,
        num_ev=25,
        use_intense_requests=True,
        start_date="2025-12-18",
        end_date="2025-12-18",
    )
    heuristic_dirs = NYCTrainer._checkpoint_dirs(
        transportation_mode="integrated",
        assignmentgurobi=False,
        num_ev=25,
        use_intense_requests=True,
        start_date="2025-12-18",
        end_date="2025-12-18",
    )
    assert exact_dirs != heuristic_dirs
    assert all("_gurobi_" in path for path in exact_dirs)
    assert all("_heu_" in path for path in heuristic_dirs)


def test_checkpoint_assignment_tag_validation():
    assert ADPTrainer._resolve_checkpoint_assign_tag(False, "gurobi") == "gurobi"
    assert ADPTrainer._resolve_checkpoint_assign_tag(False, "heu") == "heu"
    with pytest.raises(ValueError, match="gurobi, heu"):
        ADPTrainer._resolve_checkpoint_assign_tag(False, "mcmf")

