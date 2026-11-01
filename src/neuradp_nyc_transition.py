"""Side-effect-free, one-epoch NYC outcomes for NeurADP's model-based backup.

Only a car, its request and its station are copied. Motion, pickup/dropoff,
charging and costs execute the simulator's own methods. Random reward noise
is integrated by its mean; no simulation or global random stream is consumed.
Other agents retain their committed charging progress, not hypothetical new
actions. This is the per-agent transition approximation used by the baseline.
"""
from copy import copy
import json
import math
from types import SimpleNamespace

import numpy as np

from .Action import ChargingAction, IdleAction, ServiceAction
from .recourse.types import ActionType, VehicleSnapshot

_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(20)

def vehicle_record(vehicle):
    """Small immutable state representation, shared by current/next inputs."""
    return [vehicle.location, vehicle.battery,
            vehicle.service_pickup if vehicle.service_phase == 'to_pickup' else None,
            vehicle.service_dropoff, vehicle.remaining_pickup_time,
            vehicle.remaining_trip_time, vehicle.charging_time_left,
            vehicle.relocation_target, vehicle.remaining_relocation_time,
            vehicle.stationary_duration_left, vehicle.online,
            vehicle.charging_station, vehicle.charging_target,
            vehicle.target_location, vehicle.remaining_service_distance,
            vehicle.service_deadline_remaining, vehicle.service_request_value]


def rejection_probability(env, vehicle_id, request):
    """Integrate NYC's noisy logit and uniform/threshold acceptance rule."""
    if not env.ifreject or not env._is_ev(vehicle_id):
        return 0.
    from .recourse.target_builder import RecourseTargetBuilder
    if not RecourseTargetBuilder.variant_policy(env.recourse_variant).rejection_enabled:
        return 0.
    p = env._calculate_known_rejection_probability(vehicle_id, request, sample_noise=False)
    sigma = float(env.ride_acceptance_noise_std)
    # Recover the unclipped utility, including the simulator's .99 cap.
    v = env.vehicles[vehicle_id]
    utility = (env.ride_acceptance_asc
               + env.ride_acceptance_beta_idle_min * float(v.get('idle_timer', 0))*env.EPOCH_LENGTH/60
               + env.ride_acceptance_beta_pickup_min * env.get_travel_time_minutes(v['location'], request.pickup)
               + env.ride_acceptance_beta_surge * env._request_surge_bonus(request)
               + float(getattr(env, 'rejection_logit_shift', 0.)))
    if not env.reject_uniform:
        return float(utility < 0) if sigma <= 0 else .5*math.erfc(utility/(sigma*math.sqrt(2)))
    if sigma <= 0:
        return float(p)
    u = np.clip(utility + math.sqrt(2)*sigma*_HERMITE_NODES, -60, 60)
    return float(np.dot(_HERMITE_WEIGHTS, np.clip(1-1/(1+np.exp(-u)), 0, .99))/math.sqrt(math.pi))


def _sandbox(env, vid, request_id, sid):
    preview = copy(env)
    v = dict(env.vehicles[vid])
    v['completed_charging_durations_minutes'] = list(v.get('completed_charging_durations_minutes', ()))
    preview.vehicles = {vid: v}
    request_ids = {request_id, v.get('assigned_request'), v.get('passenger_onboard')}
    preview.active_requests = {rid: copy(env.active_requests[rid]) for rid in request_ids
                               if rid is not None and rid in env.active_requests}
    stations = {}
    if sid is not None:
        station = copy(env.charging_manager.stations[sid])
        for name in ('current_vehicles', 'charging_queue', 'charging_queue_notarrived'):
            setattr(station, name, list(getattr(station, name)))
        station.log_queue_arrivals = False
        stations[sid] = station
        for raw in station.current_vehicles:
            other = int(raw)
            if other != vid:
                preview.vehicles[other] = dict(env.vehicles[other])
                preview.vehicles[other]['completed_charging_durations_minutes'] = []
    preview.charging_manager = SimpleNamespace(stations=stations)
    preview.storeactions = {vid: None}
    preview.storeactions_ev = {vid: None}
    preview._recourse_random_events = {}
    preview.common_random_numbers = False
    preview.service_event_reward_noise_std = 0.
    preview.movement_reward_noise_std = 0.
    preview._charge_uniform = lambda *args, **kwargs: .5
    preview._record_vehicle_position = lambda *args, **kwargs: None
    preview._mark_charging_queue_arrival = lambda *args, **kwargs: None
    preview._mark_charging_started = lambda *args, **kwargs: None
    preview._record_ev_completion = lambda *args: None
    preview.ev_rejected_request_ids = set()
    preview.completed_requests = []
    preview.completed_requests_ev = []
    preview.completed_request_time_records = []
    preview.charge_stats = {}
    preview.idle_charging_num = {}
    return preview, v


def preview_epoch(env, *, vehicle_id, action_type, target_location,
                  request_id=None, station_id=None, continuing=False,
                  rejected_target=None, return_rejoin_probability=False):
    """Return (next vehicle record, physical reward, learning-only shaping)."""
    from .real_conservative_charging import Reservation
    vid = vehicle_id
    preview, v = _sandbox(env, vid, request_id, station_id)
    loc, soc = v['location'], v['battery']
    if action_type == ActionType.SERVICE:
        if not continuing:
            v.update(assigned_request=request_id, idle_target=None,
                     is_stationary=False, continual_reject=0, penalty_timer=0, idle_timer=0)
        action = ServiceAction([], request_id or v.get('passenger_onboard') or v.get('assigned_request'), loc, soc)
        if rejected_target is not None:
            # The executor uses a ServiceAction for a rejected offer's relocation.
            v.update(assigned_request=None, passenger_onboard=None, idle_target=rejected_target)
            v['continual_reject'] = int(env.vehicles[vid].get('continual_reject', 0))+1
            if v['continual_reject'] >= env.penalty_reject_requestnum:
                v.update(penalty_timer=env.ev_penalty_duration, continual_reject=0)
            action.was_rejected = True
            action.rejection_reward = preview._calculate_rejection_reward(vid, preview.active_requests[request_id])
    elif action_type == ActionType.CHARGE:
        action = ChargingAction([], station_id, preview._charge_duration_for_vehicle(vid), loc, soc)
        if not continuing:
            v.update(assigned_request=None, passenger_onboard=None, idle_target=None,
                     is_stationary=False, charging_target=station_id,
                     target_location=preview.charging_manager.stations[station_id].location)
        if getattr(env, 'real_conservative_active', False) and not env._is_ev(vid):
            reservation = env.real_reservations.get(vid)
            if reservation is None:
                window = env._last_conservative_charge['virtual_windows'][vid, station_id]
                if not window['kept'] or window['wait_offset'] != 0:
                    raise ValueError('NeurADP charge preview lacks an admission certificate')
                now = int(env.current_time)
                reservation = Reservation(vid, station_id, int(window['plug_index']),
                    now+window['arrival_offset'], now+window['start_offset'], now+window['end_offset'])
            preview.real_reservations = {vid: reservation}
            preview.real_plug_occupants = {key: car for key, car in env.real_plug_occupants.items() if key[0] == station_id}
            preview.real_execution_events = []
            preview._commit_selected = lambda *args: None
    else:
        target = loc if action_type == ActionType.WAIT else target_location
        if not continuing:
            if action_type == ActionType.WAIT:
                v['is_stationary'] = True
            else:
                v.update(idle_target=target, target_location=target, is_stationary=False,
                         assigned_request=None, passenger_onboard=None, charging_station=None)
        action = IdleAction([], preview.zone_coords[loc], preview.zone_coords[target], loc, soc)
        action.learning_action_type = 'wait' if action_type == ActionType.WAIT else 'reloc'
    shaping = preview._action_soc_wait_learning_penalty(vid, action)
    reward, _ = preview._execute_action(vid, action)
    previous_date = preview._current_date_label()
    preview.current_time += 1
    rejoin_probability = None
    if preview._current_date_label() != previous_date:
        # Same daily-driver update order as NYC._update_environment. Rejoin is
        # the only random branch here; freeze both outcomes instead of drawing.
        if preview._is_ev(vid):
            preview.update_satisfaction(vid)
        if not getattr(preview, 'continuous_evaluation', False):
            v['idle_timer'] = 0
        was_online = v.get('is_online', True)
        if preview.daily_drop_off and preview._can_daily_dropout(v):
            preview._handle_vehicle_dropout_event(vid)
        elif not was_online:
            rejoin_probability = float(preview.calculate_rejoin_probability(vid))
        v['daily_salary'] = 0.
        v['salary_ratio'] = 0. if preview._is_ev(vid) else 1e4
    if v['assigned_request'] is None and v['passenger_onboard'] is None:
        v['idle_timer'] += 1
    v['penalty_timer'] = max(0., v['penalty_timer']-1)
    preview._advance_charging_progress()
    if v['battery'] <= 0:
        preview._clear_vehicle_assignments(vid)
    rid = v.get('passenger_onboard')
    if rid is None:
        rid = v.get('assigned_request')
    state = VehicleSnapshot.from_vehicle(vid, v, request=preview.active_requests.get(rid),
                                         env=preview, current_time=preview.current_time)
    result = vehicle_record(state), float(reward), float(shaping)
    return (*result, rejoin_probability) if return_rejoin_probability else result


def freeze_epoch_outcomes(env, *, vehicle, action_type, target_location,
                          request_id=None, station_id=None, continuing=False,
                          relocation_cache=None):
    """JSON scalars only: replay never holds an env, car, action or RNG."""
    kwargs = dict(vehicle_id=vehicle.vehicle_id, action_type=action_type,
                  target_location=target_location, request_id=request_id,
                  station_id=station_id, continuing=continuing)
    def branches(probability, rejected=False, **extra):
        record, reward, shaping, rejoin = preview_epoch(
            env, **kwargs, **extra, return_rejoin_probability=True)
        if rejoin is None:
            return [[probability, record, reward, shaping, rejected]]
        online = list(record)
        online[10] = True
        return [[probability*(1-rejoin), record, reward, shaping, rejected],
                [probability*rejoin, online, reward, shaping, rejected]]
    outcomes = branches(1.)
    if action_type == ActionType.SERVICE and vehicle.vehicle_type == 1 and not continuing:
        q = rejection_probability(env, vehicle.vehicle_id, env.active_requests[request_id])
        if q:
            for outcome in outcomes:
                outcome[0] *= 1-q
            if vehicle.battery <= env.min_battery_level+2*env.battery_consum:
                distribution = {vehicle.location: 1.}
            else:
                if relocation_cache is not None and vehicle.vehicle_id in relocation_cache:
                    distribution = relocation_cache[vehicle.vehicle_id]
                else:
                    distribution = env.compute_ev_relocation_distribution(vehicle.vehicle_id)
                    if relocation_cache is not None:
                        relocation_cache[vehicle.vehicle_id] = distribution
            for target, p in distribution.items():
                if p <= 0:
                    continue
                outcomes.extend(branches(q*float(p), True, rejected_target=target))
    terminal = env.current_time+1 >= env.episode_length
    return (("neuradp_epoch_outcomes", json.dumps([terminal, outcomes], separators=(',', ':'))),)
