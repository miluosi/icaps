import numpy as np
import pytest
from scipy.special import expit

from inspect_nyc_rejection_outputs import inspect_outputs


def parameters():
    return dict(ride_acceptance_asc=1.810, ride_acceptance_beta_idle_min=-0.017,
                ride_acceptance_beta_pickup_min=-0.050, ride_acceptance_beta_surge=0.101,
                average_velocity_kmph=11.21 * 1.609344, assignmentrange=5.0)


def offers():
    rows = []
    for i, (idle, pickup, bonus) in enumerate([(0.5, 16, 0), (60, 16.6, 0), (60, 16.6, 10)]):
        p = float(expit(1.810 - .017 * idle - .050 * pickup + .101 * bonus))
        rows.append(dict(idle_time=idle, pickup_time=pickup, surge_bonus=bonus,
                         latent_acceptance_score=p, accepted=int(p >= .5), seed=1,
                         offer_index=i, vehicle_id=1, request_id=i))
    return rows


def test_parameter_diagnostics_reproduce_scores_and_separate_distance_and_surge():
    result, rows = inspect_outputs(offers(), parameters())
    assert result['accepted'] == 2 and result['rejected'] == 1
    assert result['probability_reconstruction_max_error'] < 1e-14
    assert result['pickup_limit_violations'] == 0
    assert result['same_offers_no_surge']['rejections'] == 2
    assert result['threshold_examples']['pickup_minutes_at_idle_half_minute_bonus_0'] == pytest.approx(36.03)
    assert result['threshold_examples']['idle_minutes_at_max_range_bonus_0'] == pytest.approx(57.56169447678015)
    assert [r['deterministic_reject'] for r in rows] == [False, True, False]
    assert [r['actual_acceptance_probability'] for r in rows] == [1.0, 0.0, 1.0]


def test_parameter_diagnostics_fail_if_recorded_output_does_not_match_parameters():
    rows = offers()
    rows[0]['latent_acceptance_score'] = .51
    with pytest.raises(AssertionError, match='behavior parameters'):
        inspect_outputs(rows, parameters())


def test_parameter_diagnostics_fail_if_actual_answer_is_inconsistent():
    rows = offers()
    rows[0]['accepted'] = 0
    with pytest.raises(AssertionError, match='deterministic rule'):
        inspect_outputs(rows, parameters())
