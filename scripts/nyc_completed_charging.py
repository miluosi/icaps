"""Completed-session charging statistics for continuous NYC result exports.

The legacy exporter saves completed session counts per day and three pooled
mean durations, but omits fleet-specific completed counts. Recover them from
the weighted-mean identity. All-zero AEV duration columns may be omitted or
become NaN after concatenating workbooks. Never use started-session counts as
completed counts, or treat a missing positive duration as zero.
"""
import math


def completed_aev_charging_metrics(row, *, slots=150, days=None):
    """Return a completed-session slot-time percentage, not full utilization.

    ``slots`` is the actual total dedicated AEV plug count. The current NYC
    batch uses 3 * 50. The observation must begin with no charging sessions
    already in progress; unfinished sessions at the horizon are excluded.
    """
    if row.get('continuous_evaluation') != True:
        raise ValueError('Completed charging reconstruction requires continuous evaluation')
    days = float(row['episodes'] if days is None else days)
    slots = float(slots)
    if not math.isfinite(days) or days <= 0 or not math.isfinite(slots) or slots <= 0:
        raise ValueError('Charging days and AEV slots must be positive and finite')
    n_float = float(row['finished_charge']) * days
    if not math.isfinite(n_float) or n_float < 0 or abs(n_float - round(n_float)) > 1e-6:
        raise ValueError('finished_charge * days must be a nonnegative integer')
    total = int(round(n_float))
    hev = float(row['avg_charging_session_duration_minutes_human_ev'])
    overall = float(row['avg_charging_session_duration_minutes_all'])
    aev = float(row.get('avg_charging_session_duration_minutes_aev', float('nan')))
    aev_omitted = math.isnan(aev)
    if aev_omitted:
        aev = 0.
    if not all(math.isfinite(x) and x >= 0 for x in (hev, aev, overall)):
        raise ValueError('Completed-session durations must be nonnegative and finite')
    if total == 0:
        if any((hev, aev, overall)):
            raise ValueError('Nonzero mean duration with zero completed sessions')
        completed = 0
    else:
        if math.isclose(hev, aev, abs_tol=1e-12, rel_tol=0):
            raise ValueError('Equal fleet durations cannot identify completed AEV counts')
        count = total * (hev - overall) / (hev - aev)
        completed = int(round(count))
        if abs(count - completed) > 1e-6 or not 0 <= completed <= total:
            raise ValueError('Fleet completed counts do not reconcile with pooled durations')
        if aev_omitted and completed:
            raise ValueError('AEV duration is missing for positive completed AEV counts')
    minutes = completed * aev
    percent = 100 * minutes / (slots * days * 1440)
    if percent > 100 + 1e-6:
        raise ValueError('Completed charging time exceeds the configured plug-time budget')
    return dict(aev_completed_charging_sessions=completed,
                aev_completed_charging_minutes=minutes,
                aev_completed_charging_minutes_per_day=minutes / days,
                aev_completed_session_slot_time_pct=percent)
