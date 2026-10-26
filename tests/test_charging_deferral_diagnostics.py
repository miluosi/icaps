from argparse import Namespace
import gzip
import json

from benchmark_nyc_charging_rollout import run_case
from src.charging_deferral_diagnostics import DeferralObserver, classify, interval_fits


def test_classification_requires_all_options_removed_and_physical_wait():
    assert classify(True, True, 2, 0, 0, 'wait', 30) == (True, True, True)
    assert classify(True, True, 2, 1, 0, 'wait', 30) == (False, False, False)
    assert classify(True, True, 0, 0, 0, 'wait', 30) == (False, False, False)
    assert classify(True, False, 2, 0, 0, 'wait', 30) == (False, False, False)
    assert classify(True, True, 2, 0, 1, 'wait', 30) == (True, True, False)
    assert classify(True, True, 2, 0, 1, 'serve', 0) == (True, False, False)
    assert classify(True, True, 1, 1, 0, 'charge', 0) == (False, False, False)


def test_calendar_diagnostic_checks_complete_half_open_interval():
    c = [[(0, 5), (8, 10)]]
    assert interval_fits(c, 5, 3)
    assert not interval_fits(c, 5, 4)
    assert not interval_fits(c, 4, 1)
    assert interval_fits(c, 10, 2)
    assert not interval_fits([], 10, 2)


def test_twenty_consecutive_waits_form_one_spell(tmp_path):
    o = DeferralObserver(tmp_path)
    o.open_spells[1] = dict(start_epoch=0, duration_seconds=20*30, vehicle_epochs=20)
    o._close(1, 20)
    assert len(o.spells) == 1 and o.spells[0]['duration_seconds']/60 == 10
    assert not o.spells[0]['right_censored']


def test_observation_preserves_physical_rollout_and_wait_counts(tmp_path):
    args = Namespace(vehicles=1000, hev=500, centers=3, hours=.05)
    outputs = []
    for observed in [False, True]:
        out = tmp_path/str(observed)
        out.mkdir()
        summary = run_case(args, 0, 'burst', 'real_conservative', out,
                           observer=DeferralObserver(out) if observed else None)
        outputs.append((out, summary))
    for filename in ['sessions.jsonl', 'timeseries.jsonl', 'stations.jsonl', 'events.jsonl', 'admissions.jsonl']:
        assert (outputs[0][0]/filename).read_bytes() == (outputs[1][0]/filename).read_bytes()
    out = outputs[1][0]
    rows = [json.loads(x) for x in gzip.open(out/'deferral_decisions.jsonl.gz', 'rt')]
    d = json.loads((out/'deferral_summary.json').read_text())
    assert d['released_new_requests'] == 455
    assert d['blocked_vehicle_epochs'] == sum(x['blocked_wait'] for x in rows)
    assert d['blocked_vehicle_minutes'] == .5*d['blocked_vehicle_epochs']
    assert d['affected_requests'] == len({x['vehicle_id'] for x in rows if x['blocked_wait']})
    assert d['not_started_by_H_count'] > 0
    assert d['affected_requests'] > 0
    spells = [json.loads(x) for x in (out/'deferral_spells.jsonl').read_text().splitlines()]
    assert sum(s['duration_seconds'] for s in spells)/60 == d['blocked_vehicle_minutes']
