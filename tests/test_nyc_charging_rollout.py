from argparse import Namespace
import json

import pytest

from benchmark_nyc_charging_rollout import run_case, summarize


def test_paired_real_rollout_dispatches_to_aev_centers_and_records_queue(tmp_path):
    args = Namespace(vehicles=1000, hev=500, centers=3, hours=.1)
    results = []
    for conservative in (False, True):
        out = tmp_path / str(conservative)
        out.mkdir()
        rows = run_case(args, 0, 'burst', conservative, out)
        results.append(rows)
        records = [json.loads(line) for line in (out / 'sessions.jsonl').read_text().splitlines()]
        stations = {row['station_id'] for row in map(json.loads, (out / 'stations.jsonl').read_text().splitlines())}
        aev = [r for r in records if r['fleet'] == 'AEV']
        assert len(aev) == 500
        assert all(r['station_id'] in stations for r in aev if r['assignment_epoch'] is not None)
        assert rows[0]['arrival_count'] > rows[0]['initial_committed_count']
        assert sum(r['completion_epoch'] is not None for r in aev) == rows[0]['completion_count']
        assert len(records) == 1000
    assert results[0][0]['initial_state_sha256'] == results[1][0]['initial_state_sha256']
    assert results[0][0]['queued_arrival_count'] > 0
    # This short window alone must not be interpreted as the full-horizon proof.
    assert results[1][0]['queued_arrival_count'] == 0
    for field in ('utilization', 'completion_count', 'mean_queue_length'):
        assert results[0][1][field] == results[1][1][field]


def test_wait_denominators_include_zero_wait_and_censored_queue():
    records = {i: dict(fleet='AEV', arrival_epoch=0, start_epoch=start, completion_epoch=None,
                      initial_charging=False, initial_committed=False, ever_queued=queued,
                      queued_steps=steps, release_epoch=0, assignment_epoch=0)
               for i, start, queued, steps in [(0, 0, False, 0), (1, 2, True, 2), (2, None, True, 4)]}
    rows = [dict(fleet='AEV', occupied=1, queue=2)] * 4
    summary = summarize(records, rows, 'AEV', 3, 4)
    assert summary['mean_wait_minutes_all_arrivals'] == pytest.approx(1.)
    assert summary['mean_wait_minutes_queued_arrivals'] == pytest.approx(1.5)
    assert summary['pending_queue_count'] == 1
    assert summary['completion_count'] == 0
    assert not summary['no_arrival_queue_observed']
