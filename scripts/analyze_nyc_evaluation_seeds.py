"""Read NYC evaluation workbooks without changing them; one seed is one rollout.

Run from the repository root or import from plot_nye_data.ipynb.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import t

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'results/nyc_result_1013'
METHODS = {'r1': 'Learning r1', 'r3': 'Decoupled r3', 'macro': 'Recourse'}
MODES = ('current', 'conservative')
SEEDS = (64, 128, 256)
METRICS = {
    'reward_per_day': '$/day',
    'completed_per_day': 'requests/day',
    'service_daily_mean_pct': '% (difference: percentage points)',
    'hev_completed_per_day': 'requests/day',
    'aev_completed_per_day': 'requests/day',
    'hev_reward_per_day': '$/day',
    'aev_reward_per_day': '$/day',
    'recourse_assigned_per_day': 'requests/day',
    'recourse_completed_per_day': 'requests/day',
    'recourse_coverage_pct': '% (difference: percentage points)',
    'recourse_success_pct': '% (difference: percentage points)',
}
PRIMARY = tuple(METRICS)[:3]


def read_method_results_seed(method, data_dir=DEFAULT_DATA, seeds=SEEDS):
    """Return records, daily records, sources and demand for both charging modes.

    Use file metadata as the seed authority, checking any numeric filename suffix.
    Missing/duplicate seeds fail instead of silently shrinking the comparison.
    """
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    data_dir = Path(data_dir)
    pattern = re.compile(rf'test_model_{method}(_conservative)?(?:_(\d+))?\.xlsx')
    rows, daily_rows, sources, demands = [], [], [], {}
    seen = set()
    for path in sorted(data_dir.glob(f'test_model_{method}*.xlsx')):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        mode = 'conservative' if match[1] else 'current'
        with pd.ExcelFile(path) as book:
            detail = pd.read_excel(book, 'detail')
            if len(detail) != 1:
                raise ValueError(f'{path.name}: expected one full rollout per workbook')
            r = detail.iloc[0]
            seed = int(r.seed)
            if match[2] and int(match[2]) != seed:
                raise ValueError(f'{path.name}: filename seed disagrees with detail.seed={seed}')
            if seed not in seeds:
                continue
            key = (mode, seed)
            if key in seen:
                raise ValueError(f'{method}: duplicate mode/seed {key}')
            seen.add(key)
            expected_mode = 'real_conservative' if mode == 'conservative' else 'current'
            assert r.strategy == 'ADP-MCMF' and r.method == method, path.name
            assert r.continuous_evaluation == True and int(r.rollouts) == 1, path.name
            assert r.charging_model == expected_mode, path.name
            assert r.checkpoint_charging_model == expected_mode, path.name
            daily = pd.read_excel(book, 'daily_req_completion_detail')
            hourly = pd.read_excel(book, 'hourly_req_completion_detail')
        for frame in (daily, hourly):
            assert set(frame.seed) == {seed}, path.name
            assert set(frame.method) == {method} and set(frame.strategy) == {'ADP-MCMF'}, path.name
            frame['request_date'] = pd.to_datetime(frame.request_date).dt.strftime('%Y-%m-%d')
        assert not daily.duplicated(['request_date', 'zone_id']).any(), path.name
        assert not hourly.duplicated(['request_date', 'request_hour', 'zone_id']).any(), path.name
        d = daily.groupby('request_date')[['generated_requests', 'completed_requests']].sum()
        assert list(d.index) == ['2025-12-15', '2025-12-16', '2025-12-17'], path.name
        days = len(d)
        assert int(r.episodes) == days, path.name
        assert (d.generated_requests > 0).all(), path.name
        assert ((d.completed_requests >= 0) & (d.completed_requests <= d.generated_requests)).all(), path.name
        d['service_pct'] = 100 * d.completed_requests / d.generated_requests
        rewards = pd.DataFrame(json.loads(r.daily_evaluation_json)).set_index('date')
        assert not rewards.index.has_duplicates and set(rewards.index) == set(d.index), path.name
        d = d.join(rewards[['reward', 'reward_ev', 'reward_aev']])
        checks = [
            (d.generated_requests.sum(), r.total_orders),
            (d.completed_requests.sum(), r.complete),
            (hourly.generated_requests.sum(), r.total_orders),
            (hourly.completed_requests.sum(), r.complete),
            (d.reward.mean(), r.avg_reward),
            (d.reward.sum(), r.total_reward),
            (r.avg_reward * days, r.total_reward),
            (d.reward_ev.mean(), r.episode_reward_ev),
            (d.reward_aev.mean(), r.episode_reward_aev),
            (r.episode_reward_ev + r.episode_reward_aev, r.avg_reward),
            ((r.mean_ev_completed_orders + r.mean_aev_completed_orders) * days, r.complete),
            (r.recourse_assigned_requests, r.recourse_requests),
            (r.recourse_completed_requests + r.recourse_uncompleted_requests, r.recourse_requests),
        ]
        for actual, expected in checks:
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-6, err_msg=path.name)
        rows.append(dict(
            mode=mode, method=method, label=METHODS[method], seed=seed, days=days,
            reward_per_day=float(r.avg_reward), completed_per_day=float(r.complete / days),
            service_daily_mean_pct=float(d.service_pct.mean()),
            service_pooled_pct=float(100 * r.complete / r.total_orders),
            hev_completed_per_day=float(r.mean_ev_completed_orders),
            aev_completed_per_day=float(r.mean_aev_completed_orders),
            hev_reward_per_day=float(r.episode_reward_ev), aev_reward_per_day=float(r.episode_reward_aev),
            recourse_assigned_per_day=float(r.recourse_requests / days),
            recourse_completed_per_day=float(r.recourse_completed_requests / days),
            recourse_coverage_pct=float(100 * r.recourse_requests / r.rejected_requests) if r.rejected_requests else np.nan,
            recourse_success_pct=float(100 * r.recourse_completed_requests / r.recourse_requests) if r.recourse_requests else np.nan,
            total_orders=int(r.total_orders), completed_total=int(r.complete), source=path.name,
        ))
        daily_rows.append(d.rename_axis('date').reset_index().assign(mode=mode, method=method, seed=seed))
        sources.append(dict(file=path.name, method=method, mode=mode, seed=seed,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            charging_model=r.charging_model, checkpoint_charging_model=r.checkpoint_charging_model))
        demands[(mode, method, seed)] = hourly.set_index(
            ['request_date', 'request_hour', 'zone_id']).generated_requests.sort_index()
    expected = {(mode, seed) for mode in MODES for seed in seeds}
    if seen != expected:
        raise ValueError(f'{method}: missing mode/seed results: {sorted(expected - seen)}')
    return dict(records=pd.DataFrame(rows).sort_values(['mode', 'seed']),
                daily=pd.concat(daily_rows, ignore_index=True), sources=sources, demands=demands)


def read_r1_results_seed(data_dir=DEFAULT_DATA, seeds=SEEDS):
    return read_method_results_seed('r1', data_dir, seeds)


def read_r3_results_seed(data_dir=DEFAULT_DATA, seeds=SEEDS):
    return read_method_results_seed('r3', data_dir, seeds)


def read_macro_results_seed(data_dir=DEFAULT_DATA, seeds=SEEDS):
    return read_method_results_seed('macro', data_dir, seeds)


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    sd = float(values.std(ddof=1)) if n > 1 else np.nan
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return dict(n=n, mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half)


def analyze(data_dir=DEFAULT_DATA, seeds=SEEDS, batches=None):
    if batches is None:
        batches = [read_method_results_seed(method, data_dir, seeds) for method in METHODS]
    records = pd.concat([b['records'] for b in batches], ignore_index=True)
    expected = {(mode, method, seed) for mode in MODES for method in METHODS for seed in seeds}
    assert not records.duplicated(['mode', 'method', 'seed']).any()
    assert set(records[['mode', 'method', 'seed']].itertuples(index=False, name=None)) == expected
    daily = pd.concat([b['daily'] for b in batches], ignore_index=True)
    sources = [s for b in batches for s in b['sources']]
    demands = {key: val for b in batches for key, val in b['demands'].items()}
    ref = next(iter(demands.values()))
    for key, demand in demands.items():
        idx = ref.index.union(demand.index)
        np.testing.assert_array_equal(ref.reindex(idx, fill_value=0), demand.reindex(idx, fill_value=0), err_msg=str(key))
    summary = []
    for (mode, method), group in records.groupby(['mode', 'method'], sort=False):
        for metric, unit in METRICS.items():
            summary.append(dict(mode=mode, method=method, metric=metric, unit=unit, **describe(group[metric])))
    differences = []

    def pair(a, b, comparison, mode):
        a, b = a.set_index('seed').sort_index(), b.set_index('seed').sort_index()
        assert a.index.equals(b.index) and not a.index.has_duplicates
        for seed in a.index:
            for metric in METRICS:
                av, bv = float(a.loc[seed, metric]), float(b.loc[seed, metric])
                differences.append(dict(mode=mode, comparison=comparison, seed=seed, metric=metric,
                                        value_a=av, value_b=bv, difference=av-bv,
                                        relative_difference_pct=100*(av-bv)/bv if bv else np.nan))
    for mode in MODES:
        frame = records[records['mode'] == mode]
        for a, b in [('macro', 'r3'), ('macro', 'r1'), ('r3', 'r1')]:
            pair(frame[frame.method == a], frame[frame.method == b], f'{a}_minus_{b}', mode)
    for method in METHODS:
        frame = records[records.method == method]
        pair(frame[frame['mode'] == 'conservative'], frame[frame['mode'] == 'current'],
             f'{method}:conservative_minus_current', 'cross_mode')
    differences = pd.DataFrame(differences)
    paired = []
    for (mode, comparison, metric), group in differences.groupby(['mode', 'comparison', 'metric'], sort=False):
        stats = describe(group.difference)
        valid = group.dropna(subset=['difference'])
        baseline = valid.value_b.mean()
        paired.append(dict(mode=mode, comparison=comparison, metric=metric, unit=METRICS[metric], **stats,
                           positive_seeds=int((valid.difference > 0).sum()),
                           negative_seeds=int((valid.difference < 0).sum()),
                           zero_seeds=int((valid.difference == 0).sum()),
                           relative_mean_difference_pct=100*stats['mean']/baseline if baseline else np.nan))
    return dict(records=records, daily=daily, summary=pd.DataFrame(summary),
                paired_raw=differences, paired=pd.DataFrame(paired), sources=sources)


def make_full_policy_latex(result, data_dir=DEFAULT_DATA):
    """Combine repeated core policies and single-seed references in one table."""
    seeds = sorted(result['records'].seed.unique())
    stats = result['summary'].set_index(['mode', 'method', 'metric'])
    columns = [('reward_per_day', 1e6, 3), ('hev_completed_per_day', 1e3, 2),
               ('aev_completed_per_day', 1e3, 2), ('hev_reward_per_day', 1e6, 3),
               ('aev_reward_per_day', 1e6, 3), ('service_daily_mean_pct', 1, 2)]
    lines = [r'\begin{table*}[!t]', r'\centering', r'\footnotesize',
             rf'\caption{{NYC continuous-day evaluation, December 15--17, 2025. Learning r1, Decoupled r3, and Recourse report mean $\pm$ sample standard deviation across evaluation seeds {", ".join(map(str, seeds))} ($n={len(seeds)}$). Myopic r1/r2 and Learning r2 report single values from seed 256 ($n=1$). Within each rollout, rewards and completed orders are daily means, and service is the mean of daily rates. HEV denotes the human-driven fleet.}}',
             r'\label{tab:nyc-combined-policy-comparison-seeds}',
             r'\setlength{\tabcolsep}{2pt}', r'\begin{tabular}{@{}rllrrrrrr@{}}', r'\toprule',
             r' & & & Reward & HEV completed & AEV completed & HEV reward & AEV reward & Service \\',
             r'ID & Charging mode & Method & ($10^6$ \$/day) & ($10^3$/day) & ($10^3$/day) & ($10^6$ \$/day) & ($10^6$ \$/day) & (\%) \\',
             r'\midrule']
    num = 0
    all_methods = [('myopic_r1', 'Myopic r1'), ('myopic_r2', 'Myopic r2'),
                   ('r1', 'Learning r1'), ('r2', 'Learning r2'),
                   ('r3', 'Decoupled r3'), ('macro', 'Recourse')]
    single_sources = []
    for mode in MODES:
        for method, label in all_methods:
            num += 1
            values = []
            single = None
            if method not in METHODS:
                suffix = '_conservative' if mode == 'conservative' else ''
                path = Path(data_dir) / f'test_model_{method}{suffix}.xlsx'
                with pd.ExcelFile(path) as book:
                    detail = pd.read_excel(book, 'detail')
                    daily = pd.read_excel(book, 'daily_req_completion_detail')
                assert len(detail) == 1, path.name
                r = detail.iloc[0]
                assert int(r.seed) == 256 and set(daily.seed) == {256}, path.name
                assert r.method == method.removeprefix('myopic_'), path.name
                assert r.strategy == ('MCMF' if method.startswith('myopic_') else 'ADP-MCMF'), path.name
                expected = 'real_conservative' if mode == 'conservative' else 'current'
                assert r.charging_model == expected, path.name
                if method == 'r2':
                    assert r.checkpoint_charging_model == expected, path.name
                assert r.continuous_evaluation == True and int(r.episodes) == 3, path.name
                assert not daily.duplicated(['request_date', 'zone_id']).any(), path.name
                d = daily.groupby('request_date')[['generated_requests', 'completed_requests']].sum()
                assert list(pd.to_datetime(d.index).strftime('%Y-%m-%d')) == ['2025-12-15', '2025-12-16', '2025-12-17'], path.name
                np.testing.assert_allclose([d.generated_requests.sum(), d.completed_requests.sum()], [r.total_orders, r.complete])
                single = dict(reward_per_day=r.avg_reward, hev_completed_per_day=r.mean_ev_completed_orders,
                              aev_completed_per_day=r.mean_aev_completed_orders, hev_reward_per_day=r.episode_reward_ev,
                              aev_reward_per_day=r.episode_reward_aev,
                              service_daily_mean_pct=(100*d.completed_requests/d.generated_requests).mean())
                single_sources.append(dict(file=path.name, seed=256, method=method, mode=mode,
                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for metric, scale, digits in columns:
                if single is None:
                    row = stats.loc[(mode, method, metric)]
                    assert int(row['n']) == len(seeds)
                    values.append(f'${row["mean"]/scale:.{digits}f} \\pm {row["sd"]/scale:.{digits}f}$')
                else:
                    values.append(f'${single[metric]/scale:.{digits}f}$')
            lines.append(f'{num} & {mode} & {label} & ' + ' & '.join(values) + r' \\')
        if mode == 'current':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
              r'\begin{minipage}{\textwidth}', r'\footnotesize',
              r'\textit{Notes.} Each evaluation seed contributes one continuous three-day rollout; days are not independent replicates. The reported dispersion is sample standard deviation, not a confidence interval. Entries without $\pm$ are single-seed reference results; their cross-seed variability is not estimated. Vehicle and station states carry over across days. These are evaluation-seed repeats, not independent training runs. Learned policies are trained and evaluated with the charging mode indicated in each row. Policy definitions follow Table~\ref{tab:nyc-combined-policy-comparison}.',
              r'\end{minipage}', r'\end{table*}']
    result['fleet_sources'] = result['sources'] + single_sources
    return '\n'.join(lines)


def save_analysis(result, data_dir=DEFAULT_DATA):
    output = Path(data_dir) / 'analysis/evaluation_seeds'
    output.mkdir(parents=True, exist_ok=True)
    for name in ('records', 'daily', 'summary', 'paired_raw', 'paired'):
        result[name].to_csv(output / f'{name}.csv', index=False)
    for source in result['sources']:
        assert hashlib.sha256((Path(data_dir) / source['file']).read_bytes()).hexdigest() == source['sha256']
    notes = (
        'Each evaluation seed is one continuous three-day rollout, not three independent observations.\n'
        'Reward and completions: daily means within each rollout. Service: arithmetic mean of the three daily request-cohort completion rates.\n'
        'Across-seed SD uses ddof=1. Paired differences match seed IDs. 95% t intervals use n-1 degrees of freedom.\n'
        'These are unadjusted, normal-model intervals with a small seed sample, not distribution-free evidence or independent training replications.\n'
        'Same hourly/zone demand is verified across all workbooks. Matching seed IDs does not prove event-keyed common random numbers.\n'
        'The workbooks verify training/test charging namespaces but do not record checkpoint hashes, code commits or CRN settings; identical checkpoints/code across seeds cannot be certified from these files alone.\n'
        'Cross-mode differences compare whole trained-and-evaluated configurations, not an isolated fixed-checkpoint charging intervention.\n'
        'r1 recourse success is undefined (no recourse assignments), not zero.\n'
    )
    (output / 'sources.json').write_text(json.dumps(result['sources'], indent=2), encoding='utf-8')
    report = [notes, 'PER-SEED VALUES', result['records'][['mode', 'method', 'seed', *PRIMARY]].to_string(index=False),
              '\nMEAN / SAMPLE SD / 95% t CI', result['summary'].query('metric in @PRIMARY').to_string(index=False),
              '\nPAIRED DIFFERENCES (A minus B)', result['paired'].query('metric in @PRIMARY').to_string(index=False)]
    (output / 'report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    seed_ids = sorted(result['records'].seed.unique())
    seed_text = ', '.join(str(s) for s in seed_ids)
    lines = [r'\begin{table*}[!t]', r'\centering', r'\small',
             rf'\caption{{NYC continuous three-day evaluation, December 15--17, 2025. Mean $\pm$ sample standard deviation across evaluation seeds {seed_text} ($n={len(seed_ids)}$ rollouts per configuration). Service is the mean of daily rates within each rollout.}}',
             r'\label{tab:nyc-evaluation-seeds}', r'\begin{tabular}{llrrr}', r'\toprule',
             r'Charging mode & Method & Reward (million \$/day) & Completed (/day) & Service (\%) \\', r'\midrule']
    stats = result['summary'].set_index(['mode', 'method', 'metric'])
    for mode in MODES:
        for method, label in METHODS.items():
            cells = []
            for metric in PRIMARY:
                row = stats.loc[(mode, method, metric)]
                scale = 1e6 if metric == 'reward_per_day' else 1
                digits = 3 if metric == 'reward_per_day' else 2
                cells.append(f'${row["mean"]/scale:.{digits}f} \\pm {row["sd"]/scale:.{digits}f}$')
            lines.append(f'{mode.capitalize()} & {label} & ' + ' & '.join(cells) + r' \\')
        if mode == 'current':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (output / 'policy_mean_sd.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    paired_lines = [r'\begin{table*}[!t]', r'\centering', r'\small',
                    rf'\caption{{Same-seed differences, $n={len(seed_ids)}$ continuous rollouts. Entries are mean differences [unadjusted 95\% Student-$t$ confidence intervals]. Positive values favor the first method. Intervals rely on a normal model for independent rollout differences; days are not independent replicates.}}',
                    r'\label{tab:nyc-paired-evaluation-seeds}', r'\begin{tabular}{llrrr}', r'\toprule',
                    r'Charging mode & Comparison & Reward (thousand \$/day) & Completed (/day) & Service (pp) \\', r'\midrule']
    for mode in MODES:
        for comparison, label in [('macro_minus_r3', 'Recourse $-$ R3'), ('macro_minus_r1', 'Recourse $-$ R1'), ('r3_minus_r1', 'R3 $-$ R1')]:
            part = result['paired'].query('mode == @mode and comparison == @comparison').set_index('metric')
            cells = []
            for metric in PRIMARY:
                scale = 1000 if metric == 'reward_per_day' else 1
                r = part.loc[metric]
                cells.append(f'{r["mean"]/scale:.2f} [{r.ci95_low/scale:.2f}, {r.ci95_high/scale:.2f}]')
            paired_lines.append(f'{mode.capitalize()} & {label} & ' + ' & '.join(cells) + r' \\')
    paired_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (output / 'paired_comparison.tex').write_text('\n'.join(paired_lines)+'\n', encoding='utf-8')
    full_latex = make_full_policy_latex(result, data_dir)
    (output / 'policy_fleet_mean_sd.tex').write_text(full_latex+'\n', encoding='utf-8')
    (output / 'fleet_sources.json').write_text(json.dumps(result['fleet_sources'], indent=2), encoding='utf-8')
    result.update(output_dir=output, notes=notes, latex='\n'.join(lines), paired_latex='\n'.join(paired_lines),
                  fleet_latex=full_latex)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    result = save_analysis(analyze(args.data_dir, args.seeds), args.data_dir)
    print((result['output_dir'] / 'report.txt').read_text())
