"""Aggregate hourly and TLC-zone mechanism tables from a formal panel.

The input is a ``panel_summary.json`` produced by the multiday, learner,
state, sensitivity, or Samitha runners.  No simulation is rerun: this script
turns the event-level episode outputs into plotting-ready CSV and JSON tables.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-summary', type=Path, required=True)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    args.panel_summary = args.panel_summary.resolve()
    if not args.panel_summary.is_file():
        parser.error(f'panel summary does not exist: {args.panel_summary}')
    args.output_dir = (args.output_dir or args.panel_summary.parent /
                       f"spatiotemporal-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    return args


def _rows(payload):
    rows = payload.get('rows')
    if rows is None:
        raise ValueError('input does not contain top-level panel rows')
    if not isinstance(rows, list):
        raise TypeError('panel rows must be a list')
    return rows


def _method(row):
    return str(row.get('method', row.get('recourse_variant', 'unknown')))


def _number(row, key):
    value = row.get(key, 0)
    return float(value) if value is not None else 0.0


def aggregate(payload, methods=None):
    selected = [row for row in _rows(payload)
                if methods is None or _method(row) in methods]
    hourly = defaultdict(lambda: defaultdict(float))
    spatial = defaultdict(lambda: defaultdict(float))

    for episode in selected:
        method = _method(episode)
        for event in episode.get('hourly_recourse_events', ()):
            key = (method, int(event['hour']))
            for field in ('rejected_count', 'eligible_count', 'assigned_count',
                          'pickup_count', 'completion_count'):
                hourly[key][field] += _number(event, field)
        for event in episode.get('hourly_completed_orders', ()):
            key = (method, int(event.get('completed_hour', event.get('hour', 0))))
            for field in ('completed_orders', 'completed_ev_orders', 'completed_aev_orders'):
                hourly[key][field] += _number(event, field)
        for event in episode.get('samitha_hold_history', ()):
            if event.get('hour') is None:
                continue
            key = (method, int(event['hour']))
            for field in ('hold_candidate_count', 'hold_selected_count',
                          'hold_utilized_count', 'unused_hold_count',
                          'rejected_repair_candidate_count',
                          'unassigned_repair_candidate_count'):
                hourly[key][field] += _number(event, field)

        for event in episode.get('spatial_recourse_events', ()):
            zone = event.get('pickup_zone_id')
            if zone is None:
                continue
            key = (method, int(event['hour']), int(zone))
            spatial[key]['rejected_count'] += 1
            for source, target in (
                ('eligible', 'eligible_count'), ('assigned', 'assigned_count'),
                ('picked_up', 'pickup_count'), ('completed', 'completion_count'),
            ):
                spatial[key][target] += float(bool(event.get(source, False)))
        for event in episode.get('hourly_zone_request_completed_orders', ()):
            key = (method, int(event.get('request_hour', 0)), int(event['zone_id']))
            spatial[key]['generated_requests'] += _number(event, 'generated_requests')
            spatial[key]['completed_requests'] += _number(event, 'completed_requests')
        for event in episode.get('hourly_zone_vehicle_counts', ()):
            key = (method, int(event.get('hour', 0)), int(event['zone_id']))
            for field in ('mean_total_vehicles', 'mean_ev_vehicles', 'mean_aev_vehicles'):
                spatial[key][field + '_sum'] += _number(event, field)
            spatial[key]['vehicle_episode_count'] += 1
        for event in episode.get('hourly_zone_charge_station_counts', ()):
            key = (method, int(event.get('hour', 0)), int(event['zone_id']))
            for field in ('mean_station_count', 'mean_total_capacity',
                          'mean_queue_vehicle_count', 'mean_queue_to_capacity_ratio'):
                spatial[key][field + '_sum'] += _number(event, field)
            spatial[key]['charging_episode_count'] += 1

    hourly_rows = []
    for (method, hour), values in sorted(hourly.items()):
        eligible = values['eligible_count']
        held = values['hold_selected_count']
        hourly_rows.append({
            'method': method, 'hour': hour, **dict(values),
            'conditional_assignment_recovery': values['assigned_count'] / eligible if eligible else 0.0,
            'conditional_pickup_recovery': values['pickup_count'] / eligible if eligible else 0.0,
            'conditional_completion_recovery': values['completion_count'] / eligible if eligible else 0.0,
            'repair_completion_per_held_aev': values['completion_count'] / held if held else 0.0,
        })

    spatial_rows = []
    for (method, hour, zone), values in sorted(spatial.items()):
        eligible = values['eligible_count']
        generated = values['generated_requests']
        vehicle_n = values['vehicle_episode_count']
        charging_n = values['charging_episode_count']
        row = {
            'method': method, 'hour': hour, 'zone_id': zone,
            **{key: value for key, value in values.items() if not key.endswith('_sum')},
            'conditional_assignment_recovery': values['assigned_count'] / eligible if eligible else 0.0,
            'conditional_pickup_recovery': values['pickup_count'] / eligible if eligible else 0.0,
            'conditional_completion_recovery': values['completion_count'] / eligible if eligible else 0.0,
            'request_completion_ratio': values['completed_requests'] / generated if generated else 0.0,
        }
        for field in ('mean_total_vehicles', 'mean_ev_vehicles', 'mean_aev_vehicles'):
            row[field] = values[field + '_sum'] / vehicle_n if vehicle_n else 0.0
        for field in ('mean_station_count', 'mean_total_capacity',
                      'mean_queue_vehicle_count', 'mean_queue_to_capacity_ratio'):
            row[field] = values[field + '_sum'] / charging_n if charging_n else 0.0
        spatial_rows.append(row)
    return selected, hourly_rows, spatial_rows


def _write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    payload = json.loads(args.panel_summary.read_text())
    selected, hourly, spatial = aggregate(payload, set(args.methods) if args.methods else None)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.output_dir / 'hourly_mechanism.csv', hourly)
    _write_csv(args.output_dir / 'zone_hour_mechanism.csv', spatial)
    summary = {
        'source_panel': str(args.panel_summary),
        'methods': sorted({_method(row) for row in selected}),
        'episode_row_count': len(selected),
        'hourly_rows': hourly,
        'spatial_rows': spatial,
        'interpretation': {
            'recovery_denominator': 'eligible rejected residual requests',
            'vehicle_values': 'mean across held-out episode rows',
            'spatial_unit': 'TLC pickup zone and hour',
        },
    }
    result = args.output_dir / 'spatiotemporal_summary.json'
    result.write_text(json.dumps(summary, indent=2))
    print(result)


if __name__ == '__main__':
    main()
