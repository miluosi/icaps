"""Summarize the additional isolated charging telemetry, preserving old figures."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'results/nyc_charging_deferral_20261022'
LABELS = {'current': 'Reactive', 'real_conservative': 'Reservation-based'}


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def load_results(folder=DEFAULT):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    config = manifest['config']
    summaries, requests, spells, examples = [], [], [], []
    validations = []
    cohort = {}
    for scenario in config['scenarios']:
        for seed in config['seeds']:
            fingerprints = []
            for policy in config['policies']:
                out = folder/f'{scenario}_seed{seed}_{policy}'
                context = dict(scenario=scenario, seed=seed, policy=policy, mode=LABELS[policy])
                summary = json.loads((out/'deferral_summary.json').read_text())
                physical = json.loads((out/'summary.json').read_text())[0]
                fingerprints.append(physical['initial_state_sha256'])
                reqs = pd.DataFrame(read_rows(out/'deferral_requests.jsonl'))
                sp = pd.DataFrame(read_rows(out/'deferral_spells.jsonl'))
                exc = read_rows(out/'deferral_examples.jsonl')
                unique, last, epochs, spell_count, hasbase, no_reloc = set(), {}, 0, 0, set(), set()
                # Stream verification: don't materialize all vehicle x epoch rows.
                with gzip.open(out/'deferral_decisions.jsonl.gz', 'rt') as f:
                    for line in f:
                        r = json.loads(line)
                        vid, epoch = r['vehicle_id'], r['epoch']
                        if r['n_charge_pre_mask']:
                            hasbase.add(vid)
                        if r['blocked_wait']:
                            assert r['physical_action_after_unfolding'] == 'wait'
                            assert r['n_charge_pre_mask'] > 0 and r['n_charge_post_mask'] == 0
                            unique.add(vid)
                            epochs += 1
                            spell_count += int(last.get(vid) != (epoch-1, True))
                            if r['n_full_relocation_actions'] == 0:
                                no_reloc.add(vid)
                        last[vid] = epoch, r['blocked_wait']
                assert len(unique) == summary['affected_requests']
                assert epochs == summary['blocked_vehicle_epochs']
                assert spell_count == summary['deferral_spells']
                assert .5*epochs == summary['blocked_vehicle_minutes']
                if len(sp):
                    assert sp.duration_seconds.sum()/60 == summary['blocked_vehicle_minutes']
                assert not reqs.vehicle_id.duplicated().any()
                assert len(reqs) == summary['released_new_requests']
                assert len(reqs)+physical['initial_charging_count']+physical['initial_committed_count'] == physical['vehicle_count']
                for key in ['predeparture_wait_minutes', 'station_wait_minutes', 'time_to_start_truncated_minutes']:
                    np.testing.assert_allclose(reqs[key].mean(), summary['mean_'+key])
                assert len(reqs.query('start_epoch.isna()')) == summary['not_started_by_H_count']
                if policy == 'real_conservative':
                    assert physical['queued_arrival_count'] == 0
                    assert reqs.station_wait_minutes.eq(0).all()
                summary.update(context, no_relocation_affected_requests=len(no_reloc),
                               ever_base_candidate_requests=len(hasbase), never_base_candidate_requests=len(reqs)-len(hasbase),
                               affected_not_started_by_H=int(((reqs.blocked_wait_epochs > 0)&reqs.start_epoch.isna()).sum()),
                               new_completed_requests=int(reqs.finish_epoch.notna().sum()),
                               all_aev_completed=physical['completion_count'])
                summaries.append(summary)
                requests.append(reqs.assign(**context))
                if len(sp):
                    spells.append(sp.assign(**context))
                examples += [{**e, **context} for e in exc]
                cohort[policy] = reqs[['vehicle_id', 'release_epoch', 'initial_soc', 'initial_location']].sort_values('vehicle_id').reset_index(drop=True)
                validations.append(dict(**context, released=len(reqs), affected=len(unique),
                                        vehicle_epochs=epochs, spells=spell_count, queue_violations=physical['queued_arrival_count']))
            assert len(set(fingerprints)) == 1
            pd.testing.assert_frame_equal(cohort['current'], cohort['real_conservative'])
    return dict(folder=folder, manifest=manifest, summary=pd.DataFrame(summaries),
                requests=pd.concat(requests, ignore_index=True), spells=pd.concat(spells, ignore_index=True),
                examples=examples, validation=pd.DataFrame(validations))


def summarize(result):
    folder = result['folder']
    out = folder/'analysis'
    out.mkdir(exist_ok=True)
    metrics = ['released_new_requests', 'affected_requests', 'affected_pct', 'deferral_spells',
               'blocked_vehicle_minutes', 'blocked_minutes_per_new_request', 'not_started_by_H_pct',
               'post_selection_slot_feasible_requests', 'no_relocation_affected_requests',
               'mean_predeparture_wait_minutes', 'mean_station_wait_minutes',
               'mean_time_to_start_truncated_minutes', 'never_base_candidate_requests',
               'affected_not_started_by_H', 'new_completed_requests', 'all_aev_completed']
    stats = []
    for (scenario, policy), group in result['summary'].groupby(['scenario', 'policy'], sort=False):
        for metric in metrics:
            stats.append(dict(scenario=scenario, policy=policy, mode=LABELS[policy], metric=metric,
                              n=len(group), mean=group[metric].mean(), sd=group[metric].std(ddof=1)))
    result['statistics'] = pd.DataFrame(stats)
    paired = []
    for scenario, group in result['summary'].groupby('scenario', sort=False):
        a = group[group.policy == 'real_conservative'].set_index('seed')
        b = group[group.policy == 'current'].set_index('seed')
        assert set(a.index) == set(b.index)
        for metric in ['mean_predeparture_wait_minutes', 'mean_station_wait_minutes', 'mean_time_to_start_truncated_minutes', 'new_completed_requests']:
            d = a[metric]-b[metric]
            half = t.ppf(.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
            paired.append(dict(scenario=scenario, metric=metric, comparison='RB minus Reactive', n=len(d),
                               mean=d.mean(), sd=d.std(ddof=1), ci95_low=d.mean()-half, ci95_high=d.mean()+half))
    result['paired'] = pd.DataFrame(paired)
    for name in ['summary', 'requests', 'spells', 'statistics', 'paired', 'validation']:
        result[name].to_csv(out/f'{name}.csv', index=False)
    # Data-driven selection: keep all examples in JSON, rank low-energy/no-reloc
    # and post-selection insertion witnesses for illustrative detailed plots.
    examples = sorted(result['examples'], key=lambda r: (r['n_full_relocation_actions'] != 0,
                      not bool(r['post_selection_slot_feasible_stations']), r['soc']))
    # Read actual environment parameters instead of copying paper defaults.
    from argparse import Namespace
    from contextlib import redirect_stdout
    from benchmark_nyc_charging_rollout import build_environment
    config = result['manifest']['config']
    with (out/'parameter_probe.log').open('w') as f, redirect_stdout(f):
        env = build_environment(Namespace(vehicles=config['vehicles'], hev=config['hev'],
                                         centers=config['centers'], hours=config['hours']), 0)
    parameters = dict(min_battery_level=env.min_battery_level, battery_consum_per_km=env.battery_consum,
                      scope='Service budget is an upper bound using the 20% reserve; the destination charging reserve may make it smaller. No passenger requests are executed in this isolated experiment.')
    # Independently reconstruct selected calendars from persisted reservations,
    # not from the in-memory telemetry that originally reported the opportunity.
    replay_cache = {}
    for example in examples:
        example['max_service_total_distance_km_at_20pct_reserve'] = max(0, (example['soc']-env.min_battery_level)/env.battery_consum)
        example['cannot_serve_even_0p5km_at_20pct_reserve'] = example['soc']-.5*env.battery_consum < env.min_battery_level
        key = example['scenario'], example['seed']
        if key not in replay_cache:
            run = folder/f'{key[0]}_seed{key[1]}_real_conservative'
            audit = json.loads((run/'reservation_audit.json').read_text())
            sessions = {r['vehicle_id']:r for r in read_rows(run/'sessions.jsonl')}
            replay_cache[key] = audit, sessions
        audit, sessions = replay_cache[key]
        epoch = example['epoch']
        matching, witnesses = [], []
        for sid_text, window in example['candidate_windows'].items():
            sid = int(sid_text)
            plugs = [[] for _ in range(env.charging_manager.stations[sid].max_capacity)]
            for reservation in audit['reservations']:
                if reservation['station_id'] != sid:
                    continue
                assigned = sessions[reservation['vehicle_id']]['assignment_epoch']
                if reservation['initial'] or (assigned is not None and assigned <= epoch):
                    plugs[reservation['plug_index']].append((reservation['start'], reservation['end']))
            start = epoch + int(window['travel_epochs'])
            end = start + int(window['charging_duration'])
            for plug, intervals in enumerate(plugs):
                if all(end <= a or start >= b for a,b in intervals):
                    matching.append(sid)
                    witnesses.append(dict(station_id=sid, plug_index=plug, start_epoch=start, end_epoch=end,
                                          fixed_intervals=sorted(intervals)))
                    break
        assert sorted(matching) == sorted(example['post_selection_slot_feasible_stations'])
        example['independent_insertion_witnesses'] = witnesses
    (out/'battery_parameters.json').write_text(json.dumps(parameters, indent=2))
    (out/'examples.json').write_text(json.dumps(examples, indent=2))
    case_table = []
    for e in examples:
        if e['n_full_relocation_actions'] == 0:
            q = e['request_summary']
            case_table.append(dict(scenario=e['scenario'], seed=e['seed'], vehicle_id=e['vehicle_id'],
                                   soc_pct=100*e['soc'], epoch=e['epoch'], pre_candidates=e['n_charge_pre_mask'],
                                   admitted_candidates=e['n_charge_post_mask'],
                                   service_distance_budget_m=1000*e['max_service_total_distance_km_at_20pct_reserve'],
                                   wait_minutes=q['predeparture_wait_minutes'],
                                   blocked_minutes=q['blocked_wait_seconds']/60,
                                   post_selection_fitting_stations=str(e['post_selection_slot_feasible_stations'])))
    result['case_table'] = pd.DataFrame(case_table)
    result['case_table'].to_csv(out/'low_soc_cases.csv', index=False)
    s = result['statistics'].set_index(['scenario', 'policy', 'metric'])
    columns = ['released_new_requests', 'affected_pct', 'deferral_spells', 'blocked_vehicle_minutes',
               'blocked_minutes_per_new_request', 'not_started_by_H_pct']
    lines = [r'\begin{table*}[!t]', r'\centering\small',
             r'\caption{Admission-blocked pre-departure waiting under reservation-based charging. Mean $\pm$ sample SD across ten complete runs per scenario. The cohort excludes 30 initially charging and 15 initially inbound AEVs. Unfinished waits contribute their observed duration.}',
             r'\label{tab:charging-admission-deferral}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Scenario & New requests & Deferred (\%) & Spells/run & Wait (veh-min/run) & Wait/request (min) & Not started by $H$ (\%) \\', r'\midrule']
    for scenario in result['manifest']['config']['scenarios']:
        vals = [s.loc[(scenario, 'real_conservative', m)] for m in columns]
        lines.append(scenario.capitalize()+' & '+' & '.join(f'${r["mean"]:.2f} \\pm {r["sd"]:.2f}$' for r in vals)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    result['latex'] = '\n'.join(lines)
    (out/'admission_deferral.tex').write_text(result['latex']+'\n')
    result['output'] = out
    return result


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = result['folder']/'figures'
    figures.mkdir(exist_ok=True)
    colors = {'current':'#ed4c62', 'real_conservative':'#219b89'}
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    scenarios = result['manifest']['config']['scenarios']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    panels = [('mean_predeparture_wait_minutes', 'Pre-departure waiting', 'Minutes / new request'),
              ('mean_station_wait_minutes', 'Station waiting', 'Minutes / new request'),
              ('mean_time_to_start_truncated_minutes', 'Release to charging start', 'Horizon-truncated minutes / new request')]
    stats = result['statistics'].set_index(['scenario', 'policy', 'metric'])
    x = np.arange(len(scenarios))
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for offset, policy in [(-.18, 'current'), (.18, 'real_conservative')]:
            means = [stats.loc[(s, policy, metric), 'mean'] for s in scenarios]
            sd = [stats.loc[(s, policy, metric), 'sd'] for s in scenarios]
            bars = ax.bar(x+offset, means, width=.34, color=colors[policy], label=LABELS[policy], yerr=sd, capsize=3)
            ax.bar_label(bars, labels=[f'{m:.2f}' for m in means], padding=6, fontsize=9)
        ax.set(xticks=x, xticklabels=[s.capitalize() for s in scenarios], title=title, ylabel=ylabel)
        ax.set_ylim(0, ax.get_ylim()[1]*1.17)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    # Cohort and error-bar definitions belong in the paper caption.
    for ext in ['png', 'pdf']:
        fig.savefig(figures/f'01_wait_location_and_delay.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    # One concrete near-threshold vehicle, with actual execution trace.
    examples = json.loads((result['output']/'examples.json').read_text())
    case = examples[0]
    out = result['folder']/f'{case["scenario"]}_seed{case["seed"]}_real_conservative'
    trace = []
    with gzip.open(out/'deferral_decisions.jsonl.gz', 'rt') as f:
        for l in f:
            r = json.loads(l)
            if r['vehicle_id'] == case['vehicle_id']:
                trace.append(r)
    trace = pd.DataFrame(trace)
    trace.to_csv(result['output']/'illustrative_case_trace.csv', index=False)
    case_req = case['request_summary']
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    minutes = trace.epoch*.5
    axes[0].step(minutes, trace.n_charge_pre_mask, where='post', label='Before reservation screening', color='#426c96')
    axes[0].step(minutes, trace.n_charge_post_mask, where='post', label='After reservation screening', color='#219b89')
    axes[0].set(ylabel='Charging candidates', title=f'{case["scenario"].capitalize()}, seed {case["seed"]}, AEV {case["vehicle_id"]}; initial SoC {case["soc"]*100:.4f}%')
    axes[0].legend(fontsize=9)
    axes[1].step(minutes, trace.n_full_relocation_actions, where='post', color='#e48144', label='Feasible positive-distance relocation actions')
    axes[1].scatter(minutes[trace.post_selection_slot_feasible], np.ones(trace.post_selection_slot_feasible.sum())*.45,
                    s=7, color='#9b6cb3', label='One individually insertable interval after selection')
    axes[1].set(ylim=(-.1, 1.2), ylabel='Read-only diagnostics', xlabel='Simulation time (min)')
    axes[1].legend(fontsize=9, loc='upper right')
    for ax in axes:
        ax.axvspan(case_req['release_epoch']*.5, case_req['depart_epoch']*.5, color='#ed4c62', alpha=.09)
        ax.grid(alpha=.2)
    fig.suptitle(f'Observed pre-departure waiting: {case_req["predeparture_wait_minutes"]:.1f} min; no relocation in the full mask', fontsize=12)
    for ext in ['png', 'pdf']:
        fig.savefig(figures/f'02_low_soc_deferral_case.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=DEFAULT)
    args = p.parse_args()
    result = plot(summarize(load_results(args.results_dir)))
    print(result['statistics'].query('metric in ["affected_pct", "blocked_minutes_per_new_request", "not_started_by_H_pct", "no_relocation_affected_requests"]').to_string(index=False))
    print(result['paired'].to_string(index=False))
