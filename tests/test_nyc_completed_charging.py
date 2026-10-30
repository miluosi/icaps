"""Audit the reconstruction against independent counts and saved NYC results."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.nyc_completed_charging import completed_aev_charging_metrics


def example():
    # Two completed HEV sessions of 10 min, three AEV sessions of 20 min.
    return dict(continuous_evaluation=True, episodes=3, finished_charge=5 / 3,
                avg_charging_session_duration_minutes_human_ev=10.,
                avg_charging_session_duration_minutes_aev=20.,
                avg_charging_session_duration_minutes_all=16.)


def test_counts_and_duration_are_not_started_session_count():
    row = example()
    row['avg_daily_charging_sessions_per_aev'] = 100.  # irrelevant start rate
    actual = completed_aev_charging_metrics(row, slots=1)
    assert actual['aev_completed_charging_sessions'] == 3
    assert actual['aev_completed_charging_minutes'] == 60.
    assert actual['aev_completed_session_slot_time_pct'] == pytest.approx(100 * 60 / 4320)


@pytest.mark.parametrize('missing', ['omit', 'nan'])
def test_omitted_zero_aev_duration_is_supported_after_dataframe_concat(missing):
    row = example()
    row['finished_charge'] = 2 / 3
    row['avg_charging_session_duration_minutes_all'] = 10.
    if missing == 'omit':
        del row['avg_charging_session_duration_minutes_aev']
    else:
        row['avg_charging_session_duration_minutes_aev'] = np.nan
    assert completed_aev_charging_metrics(row)['aev_completed_session_slot_time_pct'] == 0


def test_equal_durations_do_not_invent_fleet_counts():
    row = example()
    row['avg_charging_session_duration_minutes_aev'] = 10.
    row['avg_charging_session_duration_minutes_all'] = 10.
    with pytest.raises(ValueError, match='cannot identify'):
        completed_aev_charging_metrics(row)


@pytest.mark.parametrize('key,value', [('episodes', 0), ('finished_charge', 1.1),
                                      ('avg_charging_session_duration_minutes_all', 17.)])
def test_invalid_totals_and_durations_fail(key, value):
    row = example()
    row[key] = value
    with pytest.raises(ValueError):
        completed_aev_charging_metrics(row)


def test_latest_24_workbooks_match_independently_audited_values():
    data = Path(__file__).resolve().parents[1] / 'results/nyc_result_1013'
    prior = pd.read_csv(data / 'analysis/charging_mode_mechanism_20261028/runs.csv')
    for row in prior.itertuples():
        raw = pd.read_excel(data / row.source_file, 'detail').iloc[0]
        actual = completed_aev_charging_metrics(raw)
        assert actual['aev_completed_charging_sessions'] == row.aev_completed_charging_sessions
        assert actual['aev_completed_session_slot_time_pct'] == pytest.approx(row.aev_completed_session_slot_time_pct)
