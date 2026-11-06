"""Audit RB service-output evidence using existing continuous NYC exports.

No simulation is run. Availability is not inferred from aggregate vehicle counts.
"""
from pathlib import Path
import json
import pandas as pd
from .analyze_nyc_evaluation_seeds import DEFAULT_DATA


def export_service_evidence(result, out_dir=None):
    out = Path(out_dir or DEFAULT_DATA / 'analysis/reservation_service_20261106')
    out.mkdir(parents=True, exist_ok=True)
    records = result['records']
    daily = (result['hourly_completed']
             .groupby(['method', 'mode', 'seed', 'date'])
             [['completed_orders', 'completed_ev_orders', 'completed_aev_orders']]
             .sum().reset_index())
    long = daily.melt(id_vars=['method', 'mode', 'seed', 'date'],
                      var_name='metric', value_name='value')
    summary = long.groupby(['method', 'mode', 'date', 'metric']).value.agg(
        mean='mean', sample_sd='std', n='count').reset_index()
    metrics = ['reward_per_day', 'aev_completed_per_day', 'service_daily_mean_pct',
               'recourse_coverage_pct', 'aev_completed_session_slot_time_pct']
    paired = []
    for method in result['methods']:
        f = records.loc[records.method.eq(method)].set_index(['mode', 'seed'])
        for metric in metrics:
            dif = f.loc['conservative', metric] - f.loc['current', metric]
            paired.append({'method': method, 'metric': metric,
                           'reactive_mean': f.loc['current', metric].mean(),
                           'rb_mean': f.loc['conservative', metric].mean(),
                           'paired_rb_minus_reactive_mean': dif.mean(),
                           'paired_difference_sample_sd': dif.std(ddof=1), 'n': len(dif)})
    inventory = []
    for source in result['sources']:
        path = DEFAULT_DATA / source['file']
        with pd.ExcelFile(path) as book:
            schemas = {name: pd.read_excel(book, name, nrows=0).columns.tolist()
                       for name in book.sheet_names}
        inventory.append({'file': str(path), 'sha256': source['sha256'], 'sheets': schemas})
    daily.to_csv(out / 'daily_fleet_completions_per_seed.csv', index=False)
    summary.to_csv(out / 'daily_fleet_completions_mean_sd.csv', index=False)
    records.to_csv(out / 'rollout_metrics.csv', index=False)
    pd.DataFrame(paired).to_csv(out / 'rb_reactive_comparison.csv', index=False)
    (out / 'export_inventory.json').write_text(json.dumps(inventory, indent=2))
    (out / 'metric_audit.json').write_text(json.dumps({
        'rollouts': len(records), 'evaluation_seeds': sorted(map(int, records.seed.unique())),
        'daily_service': '100/3 * sum_d(completed_on_day_d / new_requests_released_on_day_d)',
        'recourse_coverage': '100 * unique same-epoch AEV-reassigned HEV-rejected IDs / unique HEV-rejected IDs',
        'charging_utilization': 'Completed AEV charging-session minutes / (150 plugs * 4320 minutes) * 100',
        'availability_status': 'Not recoverable from these exports: hourly_zone_vehicle_detail records fleet totals only; per-epoch action commitments, passenger status, SoC and feasible service edges are not exported.',
        'scope': 'Same-seed comparisons of mode-specific trained and executed policies; no isolated causal effect is estimated.',
        'waiting_comparison': 'Not exported or restored, following the author instruction.',
    }, indent=2))
    return out
