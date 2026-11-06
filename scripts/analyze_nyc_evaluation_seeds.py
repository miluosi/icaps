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
if __package__:
    from .nyc_completed_charging import completed_aev_charging_metrics
else:
    from nyc_completed_charging import completed_aev_charging_metrics

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'results/nyc_result_1013'
METHODS = {
    'myopic_r1': 'Myopic r1', 'myopic_r2': 'Myopic r2',
    'neuradp': 'NeurADP (2020)',
    'r1': 'Learning r1', 'r2': 'Learning r2',
    'r3': 'Decoupled r3', 'macro': 'Recourse',
}
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
    'rejected_without_same_epoch_recourse_per_day': 'requests/day',
    'aev_completed_session_slot_time_pct': '% of dedicated plug-time (completed sessions only)',
}
PRIMARY = tuple(METRICS)[:3]


def parse_result_filename(name):
    """Normalize old and new export names without merging different policies.

    The workbook metadata is checked separately: ``neuadp`` is a filename
    spelling of ``neuradp``; ``r1_myopic`` and ``myopic_r1`` identify MCMF r1.
    An omitted charging suffix means current, but an omitted seed is read
    from detail.seed rather than inferred from the filename.
    """
    match = re.fullmatch(
        r'test_model_(myopic_r[12]|r[12]_myopic|r[123]|macro|neur?adp)'
        r'(?:_(current|conservative))?(?:_(\d+))?\.xlsx', name)
    if not match:
        return None
    method, mode, seed = match.groups()
    if method.endswith('_myopic'):
        method = 'myopic_' + method.removesuffix('_myopic')
    if method == 'neuadp':
        method = 'neuradp'
    return method, mode or 'current', int(seed) if seed is not None else None


def read_method_results_seed(method, data_dir=DEFAULT_DATA, seeds=SEEDS):
    """Return records, daily records, sources and demand for both charging modes.

    Use file metadata as the seed authority, checking any numeric filename suffix.
    Missing/duplicate seeds fail instead of silently shrinking the comparison.
    """
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    data_dir = Path(data_dir)
    expected_method = method.removeprefix('myopic_')
    expected_strategy = 'MCMF' if method.startswith('myopic_') else 'ADP-MCMF'
    rows, daily_rows, completion_rows, sources, demands = [], [], [], [], {}
    seen = set()
    for path in sorted(data_dir.glob('test_model_*.xlsx')):
        parsed = parse_result_filename(path.name)
        if parsed is None or parsed[0] != method:
            continue
        _, mode, filename_seed = parsed
        with pd.ExcelFile(path) as book:
            detail = pd.read_excel(book, 'detail')
            if len(detail) != 1:
                raise ValueError(f'{path.name}: expected one full rollout per workbook')
            r = detail.iloc[0]
            seed = int(r.seed)
            if filename_seed is not None and filename_seed != seed:
                raise ValueError(f'{path.name}: filename seed disagrees with detail.seed={seed}')
            if seed not in seeds:
                continue
            key = (mode, seed)
            if key in seen:
                raise ValueError(f'{method}: duplicate mode/seed {key}')
            seen.add(key)
            expected_mode = 'real_conservative' if mode == 'conservative' else 'current'
            assert r.strategy == expected_strategy and r.method == expected_method, path.name
            assert r.continuous_evaluation == True and int(r.rollouts) == 1, path.name
            assert r.charging_model == expected_mode, path.name
            if expected_strategy == 'ADP-MCMF':
                assert r.checkpoint_charging_model == expected_mode, path.name
            daily = pd.read_excel(book, 'daily_req_completion_detail')
            hourly = pd.read_excel(book, 'hourly_req_completion_detail')
            completed = pd.read_excel(book, 'hourly_completed_detail')
        for frame in (daily, hourly):
            assert set(frame.seed) == {seed}, path.name
            assert set(frame.method) == {expected_method} and set(frame.strategy) == {expected_strategy}, path.name
            frame['request_date'] = pd.to_datetime(frame.request_date).dt.strftime('%Y-%m-%d')
        assert not daily.duplicated(['request_date', 'zone_id']).any(), path.name
        assert not hourly.duplicated(['request_date', 'request_hour', 'zone_id']).any(), path.name
        assert set(completed.seed) == {seed}, path.name
        assert set(completed.method) == {expected_method}, path.name
        assert set(completed.strategy) == {expected_strategy}, path.name
        completed['completed_date'] = pd.to_datetime(completed.completed_date).dt.strftime('%Y-%m-%d')
        d = daily.groupby('request_date')[['generated_requests', 'completed_requests']].sum()
        assert list(d.index) == ['2025-12-15', '2025-12-16', '2025-12-17'], path.name
        days = len(d)
        expected_hours = pd.MultiIndex.from_product(
            [d.index, range(24)], names=['completed_date', 'completed_hour'])
        actual_hours = completed.set_index(['completed_date', 'completed_hour']).sort_index()
        assert actual_hours.index.equals(expected_hours), f'{path.name}: missing/duplicate completion hours'
        assert (completed[['completed_orders', 'completed_ev_orders', 'completed_aev_orders']] >= 0).all().all(), path.name
        np.testing.assert_array_equal(completed.completed_ev_orders + completed.completed_aev_orders,
                                      completed.completed_orders, err_msg=path.name)
        assert int(r.episodes) == days, path.name
        assert (d.generated_requests > 0).all(), path.name
        assert (d.completed_requests >= 0).all(), path.name
        # The exporter buckets completions by completion timestamp, even though
        # the daily sheet calls its date column request_date. This is a daily
        # completion-to-release ratio, not a release-cohort completion fraction.
        finish_counts = completed.groupby('completed_date').completed_orders.sum().reindex(d.index)
        np.testing.assert_array_equal(d.completed_requests, finish_counts, err_msg=path.name)
        d['service_pct'] = 100 * d.completed_requests / d.generated_requests
        rewards = pd.DataFrame(json.loads(r.daily_evaluation_json)).set_index('date')
        assert not rewards.index.has_duplicates and set(rewards.index) == set(d.index), path.name
        d = d.join(rewards[['reward', 'reward_ev', 'reward_aev']])
        checks = [
            (d.generated_requests.sum(), r.total_orders),
            (d.completed_requests.sum(), r.complete),
            (hourly.generated_requests.sum(), r.total_orders),
            (hourly.completed_requests.sum(), r.complete),
            (completed.completed_orders.sum(), r.complete),
            (completed.completed_ev_orders.sum() / days, r.mean_ev_completed_orders),
            (completed.completed_aev_orders.sum() / days, r.mean_aev_completed_orders),
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
            completed_order_mean_final_value=float(r.mean_completed_order_value),
            gross_completed_fares_per_day=float(r.mean_completed_order_value * r.complete / days),
            gross_minus_reward_per_day=float(r.mean_completed_order_value * r.complete / days - r.avg_reward),
            service_daily_mean_pct=float(d.service_pct.mean()),
            service_pooled_pct=float(100 * r.complete / r.total_orders),
            hev_completed_per_day=float(r.mean_ev_completed_orders),
            aev_completed_per_day=float(r.mean_aev_completed_orders),
            hev_reward_per_day=float(r.episode_reward_ev), aev_reward_per_day=float(r.episode_reward_aev),
            recourse_assigned_per_day=float(r.recourse_requests / days),
            recourse_completed_per_day=float(r.recourse_completed_requests / days),
            recourse_coverage_pct=float(100 * r.recourse_requests / r.rejected_requests) if r.rejected_requests else np.nan,
            recourse_success_pct=float(100 * r.recourse_completed_requests / r.recourse_requests) if r.recourse_requests else np.nan,
            rejected_without_same_epoch_recourse_per_day=float((r.rejected_requests - r.recourse_requests) / days),
            total_orders=int(r.total_orders), completed_total=int(r.complete), source=path.name,
            **completed_aev_charging_metrics(r, days=days),
        ))
        daily_rows.append(d.rename_axis('date').reset_index().assign(mode=mode, method=method, seed=seed))
        completion_rows.append(completed[['completed_date', 'completed_hour', 'completed_orders',
                                          'completed_ev_orders', 'completed_aev_orders']].rename(
            columns={'completed_date': 'date', 'completed_hour': 'hour'}).assign(
                mode=mode, method=method, seed=seed))
        sources.append(dict(file=path.name, method=method, mode=mode, seed=seed,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            strategy=expected_strategy, charging_model=r.charging_model,
                            checkpoint_charging_model=r.checkpoint_charging_model
                            if expected_strategy == 'ADP-MCMF' else None))
        demands[(mode, method, seed)] = hourly.set_index(
            ['request_date', 'request_hour', 'zone_id']).generated_requests.sort_index()
    expected = {(mode, seed) for mode in MODES for seed in seeds}
    if seen != expected:
        raise ValueError(f'{method}: missing mode/seed results: {sorted(expected - seen)}')
    return dict(records=pd.DataFrame(rows).sort_values(['mode', 'seed']),
                daily=pd.concat(daily_rows, ignore_index=True),
                hourly_completed=pd.concat(completion_rows, ignore_index=True),
                sources=sources, demands=demands)


def read_r1_results_seed(data_dir=DEFAULT_DATA, seeds=SEEDS):
    return read_method_results_seed('r1', data_dir, seeds)


def read_myopic_r1_results_seed(data_dir=DEFAULT_DATA, seeds=SEEDS):
    return read_method_results_seed('myopic_r1', data_dir, seeds)


def read_myopic_r2_results_seed(data_dir=DEFAULT_DATA, seeds=SEEDS):
    return read_method_results_seed('myopic_r2', data_dir, seeds)


def read_r2_results_seed(data_dir=DEFAULT_DATA, seeds=SEEDS):
    return read_method_results_seed('r2', data_dir, seeds)


def read_neuradp_results_seed(data_dir=DEFAULT_DATA, seeds=SEEDS):
    return read_method_results_seed('neuradp', data_dir, seeds)


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


def analyze(data_dir=DEFAULT_DATA, seeds=SEEDS, batches=None, methods=None):
    methods = tuple(METHODS) if methods is None else tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError(f'Invalid evaluation methods: {methods}')
    if batches is None:
        batches = [read_method_results_seed(method, data_dir, seeds) for method in methods]
    records = pd.concat([b['records'] for b in batches], ignore_index=True)
    expected = {(mode, method, seed) for mode in MODES for method in methods for seed in seeds}
    assert not records.duplicated(['mode', 'method', 'seed']).any()
    assert set(records[['mode', 'method', 'seed']].itertuples(index=False, name=None)) == expected
    daily = pd.concat([b['daily'] for b in batches], ignore_index=True)
    hourly_completed = pd.concat([b['hourly_completed'] for b in batches], ignore_index=True)
    sources = [s for b in batches for s in b['sources']]
    demands = {key: val for b in batches for key, val in b['demands'].items()}
    ref = next(iter(demands.values()))
    for key, demand in demands.items():
        idx = ref.index.union(demand.index)
        np.testing.assert_array_equal(ref.reindex(idx, fill_value=0), demand.reindex(idx, fill_value=0), err_msg=str(key))
    hourly_demand = ref.rename('generated_requests').reset_index().groupby(
        ['request_date', 'request_hour'], as_index=False).generated_requests.sum().rename(
            columns={'request_date': 'date', 'request_hour': 'hour'})
    hourly_summary = hourly_completed.groupby(['mode', 'method', 'date', 'hour']).completed_orders.agg(
        n='count', mean='mean', sd='std').reset_index()
    assert (hourly_summary.n == len(seeds)).all()
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
            if a in methods and b in methods:
                pair(frame[frame.method == a], frame[frame.method == b], f'{a}_minus_{b}', mode)
    for method in methods:
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
    return dict(methods=methods, records=records, daily=daily, hourly_completed=hourly_completed,
                hourly_demand=hourly_demand, hourly_summary=hourly_summary, summary=pd.DataFrame(summary),
                paired_raw=differences, paired=pd.DataFrame(paired), sources=sources)


def make_full_policy_latex(result, data_dir=DEFAULT_DATA):
    """Combine all policies using each policy's validated rollout statistics."""
    seeds = sorted(result['records'].seed.unique())
    methods = result['methods']
    stats = result['summary'].set_index(['mode', 'method', 'metric'])
    columns = [('reward_per_day', 1e6, 3), ('hev_completed_per_day', 1e3, 2),
               ('aev_completed_per_day', 1e3, 2), ('hev_reward_per_day', 1e6, 3),
               ('aev_reward_per_day', 1e6, 3), ('service_daily_mean_pct', 1, 2),
               ('aev_completed_session_slot_time_pct', 1, 2)]
    lines = [r'\begin{table*}[!t]', r'\centering', r'\footnotesize',
             rf'\caption{{NYC continuous-day evaluation, December 15--17, 2025. All {len(methods)} methods report mean $\pm$ sample standard deviation across evaluation seeds {", ".join(map(str, seeds))} ($n={len(seeds)}$ rollouts per configuration). Within each rollout, rewards and completed orders are daily means, and service is the mean daily ratio of completed trips to newly released requests. RB denotes reservation-based charging; HEV denotes the human-driven fleet.}}',
             r'\label{tab:nyc-combined-policy-comparison-seeds}',
             r'\setlength{\tabcolsep}{1.9pt}', r'\begin{tabular}{@{}rllrrrrrrr@{}}', r'\toprule',
             r' & & & Reward & HEV completed & AEV completed & HEV reward & AEV reward & Service & $U_A^{\mathrm{cmp}}$ \\',
             r'ID & Charging mode & Method & ($10^6$ \$/day) & ($10^3$/day) & ($10^3$/day) & ($10^6$ \$/day) & ($10^6$ \$/day) & (\%) & (\%) \\',
             r'\midrule']
    num = 0
    table_rows = []
    for mode in MODES:
        for method in methods:
            label = METHODS[method]
            num += 1
            values = []
            saved = dict(ID=num, mode=mode, method=method, label=label,
                         seeds=','.join(map(str, seeds)), n=len(seeds))
            for metric, scale, digits in columns:
                row = stats.loc[(mode, method, metric)]
                assert int(row['n']) == len(seeds)
                if int(row['n']) > 1:
                    values.append(f'${row["mean"]/scale:.{digits}f} \\pm {row["sd"]/scale:.{digits}f}$')
                else:
                    values.append(f'${row["mean"]/scale:.{digits}f}$')
                saved.update({f'{metric}_mean': row['mean'], f'{metric}_sd': row['sd']})
            table_rows.append(saved)
            # Keep the existing ten-column paper layout at its original width.
            tex_label = r'\shortstack[l]{NeurADP\\(2020)}' if method == 'neuradp' else label
            display_mode = 'Reactive' if mode == 'current' else 'RB'
            lines.append(f'{num} & {display_mode} & {tex_label} & ' + ' & '.join(values) + r' \\')
        if mode == 'current':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
              r'\begin{minipage}{\textwidth}', r'\footnotesize',
              r'\textit{Notes.} Each evaluation seed contributes one continuous three-day rollout; days are not independent replicates. The reported dispersion is sample standard deviation, not a confidence interval. Vehicle and station states carry over across days. These are evaluation-seed repeats, not independent training runs. Learned policies are trained and evaluated with the charging mode indicated in each row. Myopic r1/r2 use structured scores; Learning r1 learns both fleets, and Learning r2 learns HEV scores with structured AEV scores. R1 excludes same-epoch HEV-rejection repair; r2, r3 and Recourse allow it. Decoupled r3 and Recourse learn both fleets; only Recourse credits HEV offers with the shaped HEV--AEV return.'
              + (' NeurADP (2020) is the joint value-based baseline, without the same-epoch recourse module.' if 'neuradp' in methods else ''),
              r'$U_A^{\mathrm{cmp}}=100\,T_A^{\mathrm{cmp}}/(C_A H)$, where $T_A^{\mathrm{cmp}}$ sums durations of completed AEV charging sessions, $C_A=150$ plugs, and $H=4320$ minutes. Sessions unfinished at the horizon are excluded; this is a lower bound on charging utilization, not energy delivered. Charging starts empty. Ratios are computed per rollout before averaging across seeds.',
              r'\end{minipage}', r'\end{table*}']
    result['fleet_sources'] = result['sources']
    result['fleet_table'] = pd.DataFrame(table_rows)
    return '\n'.join(lines)


def save_analysis(result, data_dir=DEFAULT_DATA):
    output = Path(data_dir) / 'analysis/evaluation_seeds'
    output.mkdir(parents=True, exist_ok=True)
    for name in ('records', 'daily', 'summary', 'paired_raw', 'paired',
                 'hourly_completed', 'hourly_demand', 'hourly_summary'):
        result[name].to_csv(output / f'{name}.csv', index=False)
    for source in result['sources']:
        assert hashlib.sha256((Path(data_dir) / source['file']).read_bytes()).hexdigest() == source['sha256']
    notes = (
        'Each evaluation seed is one continuous three-day rollout, not three independent observations.\n'
        'Reward and completions: daily means within each rollout. Service: arithmetic mean of the three daily completion-to-release ratios (completion date / release date).\n'
        'Across-seed SD uses ddof=1. Paired differences match seed IDs. 95% t intervals use n-1 degrees of freedom.\n'
        'These are unadjusted, normal-model intervals with a small seed sample, not distribution-free evidence or independent training replications.\n'
        'Same hourly/zone demand is verified across all workbooks. Matching seed IDs does not prove event-keyed common random numbers.\n'
        'The workbooks verify training/test charging namespaces but do not record checkpoint hashes, code commits or CRN settings; identical checkpoints/code across seeds cannot be certified from these files alone.\n'
        'Cross-mode differences compare whole trained-and-evaluated configurations, not an isolated fixed-checkpoint charging intervention.\n'
        'For policies without recourse assignments (including Myopic r1, Learning r1 and NeurADP), recourse success is undefined, not zero.\n'
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
        for method in result['methods']:
            label = METHODS[method]
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
            if part.empty:
                continue
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
    result['fleet_table'].to_csv(output / 'policy_fleet_mean_sd.csv', index=False)
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
