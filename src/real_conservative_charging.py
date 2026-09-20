"""Executable conservative reservations for the controlled NYC AEV stations.

This opt-in environment leaves NYCEnvironment's current/conservative behavior
unchanged. It uses the same all-potential admission filter, but retains selected
absolute per-plug intervals. AEV charging travel ends at the station coordinate,
not at the zone boundary. Prediction and execution share the same motion step.
No vehicle is held outside a station to manufacture a zero waiting statistic.
"""
from dataclasses import asdict, dataclass
import math

from .Action import ChargingAction
from .NYCEnvironment import NYCEnvironment, _haversine_km


@dataclass(frozen=True)
class Reservation:
    vehicle_id: int
    station_id: int
    plug_index: int
    arrival: int
    start: int
    end: int
    initial: bool = False


def charging_motion_step(coordinates, target, step_km):
    """One deterministic physical step, shared by prediction and execution."""
    remaining = _haversine_km(*coordinates, *target)
    if remaining <= 1e-9:
        return tuple(target), 0.0, True
    if step_km <= 0:
        raise ValueError('real_conservative requires positive vehicle speed')
    moved = min(step_km, remaining)
    if moved >= remaining - 1e-6:
        return tuple(target), moved, True
    ratio = moved / remaining
    return tuple(a + (b-a)*ratio for a, b in zip(coordinates, target)), moved, False


class RealConservativeNYCEnvironment(NYCEnvironment):
    charging_model = 'real_conservative'

    def __init__(self, *args, real_conservative_active=True, **kwargs):
        self.real_conservative_active = False
        self._reset_reservations()
        super().__init__(*args, **kwargs)
        self.real_conservative_active = real_conservative_active
        if real_conservative_active:
            self.conservative_charging = True

    def _reset_reservations(self):
        self.real_reservations = {}
        self.real_reservation_history = []
        self.real_execution_events = []
        self.real_plug_occupants = {}
        self._real_route_cache = {}
        self._real_initialized = False
        self.real_invariant_checks = 0

    def reset(self):
        self._reset_reservations()
        return super().reset()

    def _controlled(self, vehicle_id):
        return self.real_conservative_active and not self._is_ev(vehicle_id)

    def _service_epochs(self, battery, timer=None, target=None):
        # Match NYC's actual completion condition: timer OR target SOC reached.
        timer = self._charge_duration_for_battery(battery) if timer is None else timer
        target = self._charge_target_for_battery(battery) if target is None else target
        rate = float(self.chargeincrease_per_epoch)
        if rate <= 0:
            raise ValueError('real_conservative requires a positive charging rate')
        to_target = max(1, math.ceil((target - battery - 1e-6) / rate))
        return max(1, min(int(timer), to_target))

    def _route_plan(self, vehicle_id, station_id):
        vehicle = self.vehicles[vehicle_id]
        target = tuple(self.zone_coords[self.charging_manager.stations[station_id].location])
        coordinates = tuple(vehicle['coordinates'])
        step_km = self.average_velocity_kmph * self.EPOCH_LENGTH / 3600.0
        key = (coordinates, target, float(vehicle['battery']), step_km,
               self.battery_consum, self.chargeincrease_per_epoch,
               self.charge_duration_scale, self.charge_target_soc,
               self.charge_topup_soc, self.min_charging_session_epochs,
               self.max_charging_session_epochs)
        cached = self._real_route_cache.get((vehicle_id, station_id))
        if cached is not None and cached[0] == key:
            return cached[1]
        battery = float(vehicle['battery'])
        distance = 0.0
        steps = 0
        while True:
            coordinates, moved, arrived = charging_motion_step(coordinates, target, step_km)
            battery -= moved * self.battery_consum
            distance += moved
            steps += 1
            if arrived:
                break
            if steps > 100_000:
                raise RuntimeError('Charging route failed to terminate')
        plan = dict(expected_battery_after_travel=battery, distance_km=distance,
                    travel_time=max(0, steps-1), travel_epochs=max(0, steps-1),
                    charging_duration=self._service_epochs(battery))
        # Dispatch and the first motion step occur at t, hence n steps arrive at t+n-1.
        self._real_route_cache[vehicle_id, station_id] = (key, plan)
        return plan

    def calculate_expected_entercharge_station_time_battery(self, vehicle_id, station_id):
        if not self._controlled(vehicle_id):
            return super().calculate_expected_entercharge_station_time_battery(vehicle_id, station_id)
        plan = dict(self._route_plan(vehicle_id, station_id))
        if plan['expected_battery_after_travel'] < 0:
            return None
        plan['expected_entercharge_station_time'] = self.current_time + plan['travel_epochs']
        plan['expected_charge_completion_time'] = (plan['expected_entercharge_station_time']
                                                   + plan['charging_duration'])
        return plan

    def enable_real_conservative(self):
        """Import a queue-free initial state once; reject incompatible commitments."""
        self.real_conservative_active = True
        self.conservative_charging = True
        if self._real_initialized:
            return
        if not self.aev_charging_station_ids:
            raise ValueError('real_conservative requires dedicated AEV stations')
        now = int(self.current_time)
        for sid in self.aev_charging_station_ids:
            station = self.charging_manager.stations[sid]
            if station.charging_queue:
                raise ValueError('real_conservative requires a queue-free initial state')
            for plug, raw_vid in enumerate(station.current_vehicles):
                vid = int(raw_vid)
                v = self.vehicles[vid]
                end = now + self._service_epochs(v['battery'], v['charging_time_left'],
                                                v.get('charge_target_soc'))
                self._retain(Reservation(vid, sid, plug, now, now, end, True))
                self.real_plug_occupants[sid, plug] = vid
            for raw_vid in station.charging_queue_notarrived:
                vid = int(raw_vid)
                window = self.calculate_expected_entercharge_station_time_battery(vid, sid)
                if window is None:
                    raise ValueError('Initial inbound vehicle cannot reach its station')
                start = int(window['expected_entercharge_station_time'])
                end = int(window['expected_charge_completion_time'])
                plug = next((p for p in range(station.max_capacity)
                             if self._plug_free(sid, p, start, end)), None)
                if plug is None:
                    raise ValueError('Initial inbound commitments are not queue-free')
                self._retain(Reservation(vid, sid, plug, start, start, end, True))
        self._real_initialized = True
        self._check_real_state()

    def _plug_free(self, sid, plug, start, end):
        return all(r.station_id != sid or r.plug_index != plug or
                   end <= r.start or r.end <= start for r in self.real_reservations.values())

    def _retain(self, reservation):
        r = reservation
        if r.vehicle_id in self.real_reservations:
            raise RuntimeError('Vehicle already has a binding charging commitment')
        capacity = self.charging_manager.stations[r.station_id].max_capacity
        if not (0 <= r.plug_index < capacity and r.arrival == r.start < r.end):
            raise RuntimeError('Invalid immediate-service reservation')
        if not self._plug_free(r.station_id, r.plug_index, r.start, r.end):
            raise RuntimeError('Selected reservation overlaps an existing commitment')
        self.real_reservations[r.vehicle_id] = r
        self.real_reservation_history.append(asdict(r))

    def _build_expected_charging_occupancy(self, station_id, exclude_vehicle_id=None):
        if not self.real_conservative_active or station_id not in self.aev_charging_station_ids:
            return super()._build_expected_charging_occupancy(station_id, exclude_vehicle_id)
        self.enable_real_conservative()
        now = int(self.current_time)
        intervals = []
        for r in self.real_reservations.values():
            if r.station_id != station_id or r.vehicle_id == exclude_vehicle_id:
                continue
            intervals.append(dict(vehicle_id=r.vehicle_id, plug_index=r.plug_index,
                                  start_offset=max(0, r.start-now), end_offset=r.end-now,
                                  source='binding_reservation'))
        horizon = max((r['end_offset'] for r in intervals), default=0)
        occupancy = [0] * horizon
        for r in intervals:
            for offset in range(r['start_offset'], r['end_offset']):
                occupancy[offset] += 1
        return dict(capacity=self.charging_manager.stations[station_id].max_capacity,
                    occupancy=tuple(occupancy), intervals=tuple(intervals))

    def generate_vehicle_chargerange(self, vehicle_ids):
        if self.real_conservative_active:
            self.enable_real_conservative()
            if any(vid in self.real_reservations for vid in vehicle_ids):
                raise RuntimeError('Committed vehicle cannot re-enter charging candidates')
        mask = super().generate_vehicle_chargerange(vehicle_ids)
        self._real_admission_epoch = int(self.current_time)
        return mask

    def _commit_selected(self, vehicle_id, station_id):
        self.enable_real_conservative()
        existing = self.real_reservations.get(vehicle_id)
        if existing:
            if existing.station_id != station_id:
                raise RuntimeError('Cannot redirect a binding charging reservation')
            return
        if getattr(self, '_real_admission_epoch', None) != int(self.current_time):
            raise RuntimeError('Charging assignment has no current admission certificate')
        virtual = (self._last_conservative_charge or {}).get('virtual_windows', {}).get((vehicle_id, station_id))
        if not virtual or not virtual['kept'] or virtual['wait_offset'] != 0:
            raise RuntimeError('Charging assignment bypassed conservative admission')
        now = int(self.current_time)
        self._retain(Reservation(vehicle_id, station_id, int(virtual['plug_index']),
                                 now + virtual['arrival_offset'], now + virtual['start_offset'],
                                 now + virtual['end_offset']))

    def _move_vehicle_to_charging_station(self, vehicle_id, station_id):
        if self._controlled(vehicle_id):
            self._commit_selected(vehicle_id, station_id)
        return super()._move_vehicle_to_charging_station(vehicle_id, station_id)

    def _execute_action(self, vehicle_id, action):
        if not self._controlled(vehicle_id):
            return super()._execute_action(vehicle_id, action)
        existing = self.real_reservations.get(vehicle_id)
        if existing and not isinstance(action, ChargingAction):
            raise RuntimeError('An outstanding charging commitment was not executed')
        if not isinstance(action, ChargingAction):
            return super()._execute_action(vehicle_id, action)
        if existing is None:
            self._move_vehicle_to_charging_station(vehicle_id, action.charging_station_id)
        else:
            self._commit_selected(vehicle_id, action.charging_station_id)
        r = self.real_reservations[vehicle_id]
        v = self.vehicles[vehicle_id]
        if not v.get('is_online', True):
            raise RuntimeError('Committed AEV went offline')
        if self.storeactions.get(vehicle_id) is None:
            self.storeactions[vehicle_id] = action
            action.dur_reward = 0
            action.current_time = self.current_time
        reward = -self.charging_penalty
        if v['charging_station'] is None:
            station = self.charging_manager.stations[r.station_id]
            target = tuple(self.zone_coords[station.location])
            coords, moved, arrived = charging_motion_step(
                tuple(v['coordinates']), target, self.average_velocity_kmph*self.EPOCH_LENGTH/3600.)
            v['coordinates'] = coords
            v['battery'] -= moved * self.battery_consum
            if v['battery'] < -1e-9:
                raise RuntimeError('Committed charging route exhausted its battery')
            v['total_distance'] += moved
            v['location'] = station.location if arrived else self.map_zone(coords)
            if v['location'] < 0:
                v['location'] = self._nearest_zone(*coords, prefer_polygon=False)
            v['zone_id'] = v['location']
            self.vehicle_position_history.setdefault(vehicle_id, []).append(
                dict(zone=v['location'], coordinates=coords, time=self.current_time))
            if arrived:
                if self.current_time != r.arrival:
                    raise RuntimeError(f'Physical arrival disagrees with reservation: {vehicle_id}, {self.current_time}, {r}')
                if (r.station_id, r.plug_index) in self.real_plug_occupants:
                    raise RuntimeError('Reserved plug was not released before arrival')
                self._mark_charging_queue_arrival(vehicle_id, r.station_id)
                if station.charging_queue or not station.is_available():
                    raise RuntimeError('real_conservative arrival would queue')
                if not station.start_charging(str(vehicle_id)):
                    raise RuntimeError('Reserved charging start failed')
                self.real_plug_occupants[r.station_id, r.plug_index] = vehicle_id
                self._clear_aev_notarrived_if_arrived(vehicle_id, r.station_id)
                self._mark_charging_started(vehicle_id, r.station_id)
                v.update(charging_station=r.station_id, charging_target=None, target_location=None)
                self._set_vehicle_charging_session(vehicle_id)
                if self.current_time + self._service_epochs(v['battery']) != r.end:
                    raise RuntimeError('Actual charging duration disagrees with reservation')
                v['charging_count'] += 1
                self.real_execution_events.append(dict(event='arrival_start', vehicle_id=vehicle_id,
                                                       station_id=r.station_id, plug_index=r.plug_index,
                                                       epoch=self.current_time, reserved_arrival=r.arrival,
                                                       reserved_end=r.end, wait=0))
            else:
                if self.current_time >= r.arrival:
                    raise RuntimeError('Committed arrival missed its reserved time')
                reward = self._movement_cost(moved)
        reward -= self.aev_charging_extra_penalty
        self.storeactions[vehicle_id].dur_reward += reward
        return reward, getattr(action, 'dur_reward', 0)

    def _update_environment(self):
        if not self.real_conservative_active:
            return super()._update_environment()
        self.enable_real_conservative()
        super()._update_environment()
        # Base update advances to t+1 and releases completions; the next step's
        # arrivals at t+1 occur only after this release. The same plug may be reused.
        now = int(self.current_time)
        for vid, r in list(self.real_reservations.items()):
            v = self.vehicles[vid]
            if (r.station_id, r.plug_index) in self.real_plug_occupants and self.real_plug_occupants[r.station_id, r.plug_index] == vid:
                if v['charging_station'] is None:
                    if now != r.end:
                        raise RuntimeError('Charging completed outside its binding interval')
                    del self.real_plug_occupants[r.station_id, r.plug_index]
                    del self.real_reservations[vid]
                    self.real_execution_events.append(dict(event='complete', vehicle_id=vid,
                                                           station_id=r.station_id, epoch=now,
                                                           reserved_end=r.end))
                elif now >= r.end:
                    raise RuntimeError('Charging overran its binding reservation')
            elif now > r.arrival:
                raise RuntimeError('An inbound reservation was not executed')
        self._check_real_state()

    def _check_real_state(self):
        for sid in self.aev_charging_station_ids:
            station = self.charging_manager.stations[sid]
            if station.charging_queue:
                raise RuntimeError('Physical AEV queue is nonempty')
            occupants = {str(vid) for (station_id, _), vid in self.real_plug_occupants.items() if station_id == sid}
            if occupants != set(station.current_vehicles):
                raise RuntimeError('Physical plugs and binding plug registry disagree')
            if station.available_slots != station.max_capacity-len(occupants):
                raise RuntimeError('Physical slot count disagrees with reservations')
            inbound = {str(r.vehicle_id) for r in self.real_reservations.values()
                       if r.station_id == sid and str(r.vehicle_id) not in occupants}
            if inbound != set(station.charging_queue_notarrived):
                raise RuntimeError('Unrecorded or missing inbound charging commitment')
        self.real_invariant_checks += 1
