"""Read-only telemetry for the existing isolated charge/wait rollout.

Counts come from the unreduced action mask; waits are verified after physical
execution. No random draws, scheduling changes, or hypothetical waiting times.
"""
import gzip
import json
from bisect import bisect_right
from collections import Counter

import numpy as np

from .Action import IdleAction
from .conservative_charging import _hard_calendars
from .expected_charging import has_complete_charging_window, epoch_offset


def classify(pending, available, before, after, other, action, wait_seconds):
    if min(before, after, other) < 0 or after > before:
        raise ValueError('Invalid unreduced candidate counts')
    blocked = pending and available and before > 0 and after == 0
    waited = blocked and action == 'wait' and wait_seconds > 0
    return bool(blocked), bool(waited), bool(waited and other == 0)


def interval_fits(calendars, arrival, duration):
    """Can ONE candidate be inserted at its arrival without moving any job?"""
    for calendar in calendars:
        idx = bisect_right(calendar, (arrival, float('inf')))
        if idx and calendar[idx-1][1] > arrival:
            continue
        if idx < len(calendar) and arrival + duration > calendar[idx][0]:
            continue
        return True
    return False


class DeferralObserver:
    def __init__(self, output):
        self.output = output
        self.current = {}
        self.spells = []
        self.open_spells = {}
        self.counts = Counter()
        self.examples = {}

    def initialize(self, env, records, horizon, scenario, seed, policy):
        self.context = dict(scenario=scenario, seed=seed, policy=policy)
        self.horizon = horizon
        self.step_seconds = float(env.EPOCH_LENGTH)
        self.requests = {}
        self.trace = gzip.open(self.output / 'deferral_decisions.jsonl.gz', 'wt')
        ids = [vid for vid, r in records.items() if r['fleet'] == 'AEV'
               and not r['initial_charging'] and not r['initial_committed']]
        zones = list(env.relocation_target_ids) if env.relocation_target_ids else sorted(env.zone_to_locs)
        if ids:
            # Production relocation mask, including reserve/range/top-k rules.
            reloc = env.generate_vehicle_zone(ids)
            locations = np.array([env.vehicles[vid]['location'] for vid in ids], dtype=int)
            distances = env.distance_matrix[locations[:, None], np.asarray(zones)[None, :]]
            counts = ((reloc > 0) & (distances > 1e-9)).sum(axis=1)
        # Supplemental energy-only probe, NEVER added to the experiment's graph.
        # All inter-zone pickup/dropoff pairs on the loaded valid zone grid,
        # geographical trip distances, no pickup range/top-k restrictions.
        service_zones = sorted(env.zone_to_locs)
        trip = env.distance_matrix[np.ix_(service_zones, service_zones)].astype(float)
        trip = np.where(np.eye(len(service_zones), dtype=bool), np.inf, trip)
        required_cache = {}
        for j, vid in enumerate(ids):
            r = records[vid]
            location = env.vehicles[vid]['location']
            stations = tuple(env._charging_station_ids_for_vehicle(vid))
            key = location, stations
            if key not in required_cache:
                reserve = np.array([env._post_action_battery_reserve(z, vehicle_id=vid) for z in service_zones])
                pickup = env.distance_matrix[location, service_zones]
                required_cache[key] = float(np.min((pickup[:, None] + trip) * env.battery_consum + reserve[None, :]))
            self.requests[vid] = dict(
                charge_request_id=f'{scenario}:{seed}:aev:{vid}', vehicle_id=vid,
                release_epoch=r['release_epoch'], initial_soc=r['initial_soc'],
                initial_location=r['initial_location'], n_full_relocation_actions=int(counts[j]),
                interzone_service_min_required_soc=required_cache[key],
                blocked_wait_epochs=0, blocked_wait_seconds=0., slot_feasible_wait_epochs=0,
                low_energy_wait_epochs=0, spells=0)

    def before_assignment(self, env, ids, sids, charge, epoch):
        self.current = {}
        expansion = env._last_expected_charge_expansion
        self.windows = expansion['candidate_windows']
        self.schedules = expansion['station_schedules']
        virtual = (env._last_conservative_charge or {}).get('virtual_windows', {})
        self.virtual = virtual
        for row, vid in enumerate(ids):
            v = env.vehicles[vid]
            windows = {sid: self.windows[vid, sid] for sid in sids if (vid, sid) in self.windows
                       and self.windows[vid, sid].get('charging_duration', 0) > 0}
            # Windows are produced after reachability/permission checks, BEFORE
            # both committed-full-window and all-potential reservation filters.
            pre = len(windows)
            post = int(np.count_nonzero(charge[row]))
            reactive = sum(has_complete_charging_window(
                self.schedules[sid], arrival_offset=w['travel_epochs'], charging_duration=1)
                for sid, w in windows.items())
            background = sum(bool(w['feasible']) for w in windows.values())
            self.current[vid] = dict(
                vehicle_id=vid, charge_request_id=self.requests[vid]['charge_request_id'],
                epoch=epoch, soc=float(v['battery']), location=int(v['location']),
                coordinates=tuple(v['coordinates']), n_charge_pre_mask=pre,
                n_charge_post_mask=post, n_committed_window_options=background,
                n_reactive_same_state_options=int(reactive), n_other_nonwait_actions=0,
                n_full_relocation_actions=self.requests[vid]['n_full_relocation_actions'],
                interzone_service_min_required_soc=self.requests[vid]['interzone_service_min_required_soc'],
                below_min_battery=bool(v['battery'] < env.min_battery_level),
                no_interzone_service_energy=bool(v['battery'] < self.requests[vid]['interzone_service_min_required_soc']))

    def _close(self, vid, end_epoch, censored=False):
        spell = self.open_spells.pop(vid, None)
        if spell is not None:
            spell.update(end_epoch=end_epoch, right_censored=censored)
            self.spells.append(spell)

    def after_step(self, env, records, actions, epoch):
        calendars = {sid: _hard_calendars(s, int(s['capacity'])) for sid, s in self.schedules.items()} if self.current else {}
        # Freeze pre-dispatch commitments + only actual selected new reservations.
        # Excluded virtual candidates do not occupy this diagnostic calendar.
        for vid, before in self.current.items():
            r = records[vid]
            if r['assignment_epoch'] == epoch:
                sid = r['station_id']
                vw = self.virtual.get((vid, sid))
                if vw and vw['kept']:
                    calendars[sid][vw['plug_index']].append((vw['arrival_offset'], vw['end_offset']))
        for plugs in calendars.values():
            for plug in plugs:
                plug.sort()
        for vid, before in self.current.items():
            v, r = env.vehicles[vid], records[vid]
            action = actions[vid]
            stationary = (isinstance(action, IdleAction) and r['assignment_epoch'] is None
                          and tuple(v['coordinates']) == before['coordinates']
                          and v.get('charging_station') is None and v.get('charging_target') is None)
            physical = 'wait' if stationary else 'charge' if r['assignment_epoch'] == epoch else 'other'
            wait_seconds = self.step_seconds if stationary else 0.
            blocked, waited, strict = classify(True, True, before['n_charge_pre_mask'],
                                                before['n_charge_post_mask'], 0, physical, wait_seconds)
            # 'Admission blocked' applies only to RB, not reactive scarcity.
            is_rb = self.context['policy'] == 'real_conservative'
            blocked, waited, strict = (x and is_rb for x in (blocked, waited, strict))
            post_fits = []
            if waited:
                for sid in self.schedules:
                    w = self.windows.get((vid, sid))
                    if w and interval_fits(calendars[sid], epoch_offset(w['travel_epochs']), epoch_offset(w['charging_duration'])):
                        post_fits.append(sid)
            low_energy = waited and before['n_full_relocation_actions'] == 0 and before['no_interzone_service_energy']
            reason = ('charged' if physical == 'charge' else 'no-base-charge-option' if before['n_charge_pre_mask'] == 0
                      else 'admission-blocked/wait-only' if waited
                      else 'admitted-but-waited' if stationary and before['n_charge_post_mask'] > 0
                      else 'reactive-capacity-blocked' if stationary else 'other')
            self.counts[reason] += 1
            self.counts['stationary_wait_vehicle_epochs'] += int(stationary)
            self.counts['no_relocation_wait_vehicle_epochs'] += int(waited and before['n_full_relocation_actions'] == 0)
            if waited:
                req = self.requests[vid]
                req['blocked_wait_epochs'] += 1
                req['blocked_wait_seconds'] += wait_seconds
                req['slot_feasible_wait_epochs'] += int(bool(post_fits))
                req['low_energy_wait_epochs'] += int(low_energy)
                if vid not in self.open_spells:
                    req['spells'] += 1
                    self.open_spells[vid] = dict(vehicle_id=vid, charge_request_id=req['charge_request_id'],
                                                 start_epoch=epoch, duration_seconds=0., vehicle_epochs=0)
                self.open_spells[vid]['duration_seconds'] += wait_seconds
                self.open_spells[vid]['vehicle_epochs'] += 1
                # Bounded illustrative evidence; aggregate counts include ALL rows.
                score = (int(low_energy), int(bool(post_fits)), -before['soc'])
                if vid not in self.examples or score > self.examples[vid][0]:
                    example = {**before, 'post_selection_slot_feasible_stations': post_fits,
                               'physical_action_after_unfolding': physical,
                               'actual_wait_seconds': wait_seconds, 'low_energy_probe_case': bool(low_energy),
                               'candidate_windows': {str(sid): {k: w.get(k) for k in ('travel_epochs', 'charging_duration', 'feasible')}
                                                     for (k, sid), w in self.windows.items() if k == vid},
                               'virtual_windows': {str(sid): w for (k, sid), w in self.virtual.items() if k == vid}}
                    example.pop('coordinates')
                    self.examples[vid] = score, example
            else:
                self._close(vid, epoch)
            row = {k: val for k, val in before.items() if k != 'coordinates'}
            row.update(physical_action_after_unfolding=physical, actual_predeparture_wait_s=wait_seconds,
                       admission_blocked=bool(blocked), blocked_wait=bool(waited), strict_wait_only=bool(strict),
                       post_selection_slot_feasible=bool(post_fits), low_energy_probe_case=bool(low_energy), reason=reason)
            self.trace.write(json.dumps(row)+'\n')
        self.current = {}

    def finish(self, env, records, horizon):
        self.trace.close()
        for vid in list(self.open_spells):
            self._close(vid, horizon, True)
        rows = []
        for vid, req in self.requests.items():
            r = records[vid]
            if r['release_epoch'] >= horizon:
                continue
            depart = r['assignment_epoch']
            arrive, start = r['arrival_epoch'], r['start_epoch']
            pre = (depart if depart is not None else horizon)-r['release_epoch']
            station = ((start if start is not None else horizon)-arrive) if arrive is not None else 0.
            time_to_start = (start if start is not None else horizon)-r['release_epoch']
            travel = ((arrive if arrive is not None else horizon)-depart) if depart is not None else 0.
            assert abs(time_to_start-pre-travel-station) < 1e-8
            rows.append({**req, 'depart_epoch': depart, 'arrive_epoch': arrive, 'start_epoch': start,
                         'finish_epoch': r['completion_epoch'], 'not_started_by_H': start is None,
                         'predeparture_wait_minutes': pre*self.step_seconds/60,
                         'station_wait_minutes': station*self.step_seconds/60,
                         'travel_minutes': travel*self.step_seconds/60,
                         'time_to_start_truncated_minutes': time_to_start*self.step_seconds/60})
        n = len(rows)
        affected = [r for r in rows if r['blocked_wait_epochs']]
        summary = {**self.context, 'released_new_requests': n,
                   'affected_requests': len(affected), 'affected_pct': 100*len(affected)/n,
                   'deferral_spells': len(self.spells), 'censored_spells': sum(s['right_censored'] for s in self.spells),
                   'blocked_vehicle_epochs': sum(r['blocked_wait_epochs'] for r in rows),
                   'blocked_vehicle_minutes': sum(r['blocked_wait_seconds'] for r in rows)/60,
                   'blocked_minutes_per_new_request': sum(r['blocked_wait_seconds'] for r in rows)/60/n,
                   'not_started_by_H_count': sum(r['not_started_by_H'] for r in rows),
                   'not_started_by_H_pct': 100*sum(r['not_started_by_H'] for r in rows)/n,
                   'post_selection_slot_feasible_epochs': sum(r['slot_feasible_wait_epochs'] for r in rows),
                   'post_selection_slot_feasible_requests': sum(r['slot_feasible_wait_epochs'] > 0 for r in rows),
                   'low_energy_probe_wait_epochs': sum(r['low_energy_wait_epochs'] for r in rows),
                   'low_energy_probe_requests': sum(r['low_energy_wait_epochs'] > 0 for r in rows),
                   'counts_by_reason': dict(self.counts),
                   'service_probe_scope': 'Energy-only all distinct pickup/dropoff zone pairs on loaded grid, geographical trip distances; NOT live requests and NOT a claim about arbitrary same-zone trips.'}
        for metric in ('predeparture_wait_minutes', 'station_wait_minutes', 'time_to_start_truncated_minutes', 'travel_minutes'):
            summary['mean_'+metric] = sum(r[metric] for r in rows)/n
        assert n + sum(r['initial_charging'] or r['initial_committed'] for r in records.values() if r['fleet']=='AEV') == sum(r['fleet']=='AEV' for r in records.values())
        examples = [x[1] for x in sorted(self.examples.values(), key=lambda x: x[0], reverse=True)[:12]]
        for example in examples:
            example['request_summary'] = next(r for r in rows if r['vehicle_id'] == example['vehicle_id'])
        for name, data in [('deferral_requests', rows), ('deferral_spells', self.spells), ('deferral_examples', examples)]:
            with (self.output / f'{name}.jsonl').open('w') as f:
                for row in data:
                    f.write(json.dumps(row)+'\n')
        (self.output/'deferral_summary.json').write_text(json.dumps(summary, indent=2))
        return summary
