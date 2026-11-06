"""Figures from validated NYC rollouts; uncertainty is across seeds, not days."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde

from .analyze_nyc_evaluation_seeds import METHODS, MODES


COLORS = {
    'myopic_r1': '#697786', 'myopic_r2': '#b28c57', 'neuradp': '#cf526a',
    'r1': '#4c78a8', 'r2': '#9b6cb3', 'r3': '#e48144', 'macro': '#219b89',
}


def plot_policy_overview(result):
    """Selected policies, paired charging-mode bars, mean +/- sample SD."""
    methods = result['methods']
    stats = result['summary'].set_index(['mode', 'method', 'metric'])
    seeds = sorted(result['records'].seed.unique())
    panels = [
        ('reward_per_day', 1e6, 'Daily system reward', 'Reward ($ million/day)'),
        ('rejected_without_same_epoch_recourse_per_day', 1e3,
         'HEV-rejected requests without\nsame-epoch recourse', 'Requests (thousand/day)'),
        ('service_daily_mean_pct', 1, 'Mean daily service rate', 'Completed / newly released (%)'),
        ('recourse_coverage_pct', 1, 'Same-epoch recourse coverage',
         'Reassigned / HEV-rejected (%)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.subplots_adjust(left=.065, right=.99, bottom=.09, top=.87, hspace=.54, wspace=.20)
    x, width = np.arange(len(methods)), .37
    for ax, (metric, scale, title, ylabel) in zip(axes.flat, panels):
        tops = []
        for mode, offset, hatch in [('current', -width/2, None),
                                    ('conservative', width/2, '///')]:
            rows = stats.loc[[(mode, method, metric) for method in methods]]
            assert (rows.n == len(seeds)).all()
            means, sd = rows['mean'].to_numpy()/scale, rows.sd.to_numpy()/scale
            assert np.isfinite(means).all() and np.isfinite(sd).all()
            bars = ax.bar(x+offset, means, width, color=[COLORS[m] for m in methods],
                          hatch=hatch, edgecolor='white', linewidth=.7,
                          yerr=sd, capsize=2.5,
                          error_kw={'elinewidth': .9, 'capthick': .9, 'ecolor': '#303030'})
            for bar, mean, spread in zip(bars, means, sd):
                ax.annotate(f'{mean:.2f}', (bar.get_x()+bar.get_width()/2, mean+spread),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8.5)
            tops.extend(means+sd)
        labels = ['NeurADP\n(2020)' if m == 'neuradp' else METHODS[m].replace(' ', '\n', 1)
                  for m in methods]
        ax.set_xticks(x, labels, fontsize=10.5)
        ax.set_title(title, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(0, 104 if metric == 'recourse_coverage_pct' else max(max(tops)*1.20, 1))
        ax.grid(axis='x', visible=False)
    # Statistical definitions and dates appear in the paper caption.
    # Offset close paired labels without moving bars or error bars.
    for ax in axes.flat:
        n = len(methods)
        for first, second in zip(ax.texts[:n], ax.texts[n:]):
            if abs(first.xy[1] - second.xy[1]) < .08 * ax.get_ylim()[1]:
                second.set_position((0, 17))
    fig.legend(handles=[Patch(facecolor='#697786', edgecolor='white', label='Reactive (left)'),
                        Patch(facecolor='#697786', edgecolor='white', hatch='///',
                              label='RB (right)')],
               loc='upper center', bbox_to_anchor=(.52, .99), ncol=2,
               frameon=False, fontsize=14)
    return fig


def plot_daily_completion(result):
    """Three dates, selected learned policies, both charging modes, mean +/- SD."""
    methods = [m for m in ['neuradp', 'r1', 'r2', 'r3', 'macro'] if m in result['methods']]
    seeds = sorted(result['records'].seed.unique())
    dates = sorted(result['hourly_demand'].date.unique())
    assert len(dates) == 3, 'The paper figure uses three calendar-day panels'
    summary = result['hourly_summary'].set_index(['mode', 'method', 'date', 'hour'])
    demand = result['hourly_demand'].set_index(['date', 'hour']).generated_requests
    hours, centers = np.arange(24), np.arange(24)+.5
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.05, right=.99, bottom=.14, top=.73, wspace=.08)
    for ax, date in zip(axes, dates):
        counts = demand.loc[date].reindex(hours)
        assert counts.notna().all()
        kde = gaussian_kde(centers, weights=counts.to_numpy(), bw_method=.7)
        kde_x = np.linspace(0, 24, 481)
        kde_rate = kde(kde_x)/kde.integrate_box_1d(0, 24)*counts.sum()
        np.testing.assert_allclose(trapezoid(kde_rate, kde_x), counts.sum(), rtol=1e-4)
        ax.fill_between(kde_x, kde_rate/1000, color='#aeb8c2', alpha=.20, zorder=1)
        ax.plot(kde_x, kde_rate/1000, color='#788896', linewidth=1.4, linestyle=':', zorder=2)
        for mode in MODES:
            for method in methods:
                row = summary.loc[(mode, method, date)].reindex(hours)
                assert (row.n == len(seeds)).all()
                mean, sd = row['mean'].to_numpy()/1000, row.sd.to_numpy()/1000
                assert np.isfinite(mean).all() and np.isfinite(sd).all()
                conservative = mode == 'conservative'
                ax.fill_between(centers, np.maximum(0, mean-sd), mean+sd,
                                color=COLORS[method], alpha=.10, linewidth=0, zorder=2)
                ax.plot(centers, mean, color=COLORS[method],
                        linestyle=(0, (4, 2)) if conservative else '-',
                        marker='s' if conservative else 'o', markersize=3.7,
                        markerfacecolor='white' if conservative else COLORS[method],
                        markeredgewidth=.9, markevery=3,
                        linewidth=2.4 if method == 'macro' else 1.7, zorder=3)
        ax.set_title(date, fontsize=15)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.tick_params(labelsize=14)
        ax.set_xlim(0, 24)
        ax.set_xlabel('Hour of day', fontsize=15)
    axes[0].set_ylabel('Completed trips / demand (thousand/hour)', fontsize=14)
    axes[0].set_ylim(bottom=0)
    fig.legend(handles=[Line2D([0], [0], color=COLORS[m], linewidth=2.5, label=METHODS[m])
                        for m in methods], loc='upper center', bbox_to_anchor=(.52, .99),
               ncol=len(methods), frameon=False, fontsize=15)
    fig.legend(handles=[
        Line2D([0], [0], color='#444444', linestyle='-', marker='o', markersize=4,
               label='Reactive'),
        Line2D([0], [0], color='#444444', linestyle=(0, (4, 2)), marker='s',
               markerfacecolor='white', markersize=4, label='RB'),
        Patch(facecolor='#aeb8c2', alpha=.3, label='Shared demand KDE'),
    ], loc='upper center', bbox_to_anchor=(.52, .90), ncol=3, frameon=False, fontsize=14)
    return fig
