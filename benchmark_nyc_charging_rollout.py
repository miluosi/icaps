"""Paired non-learning charging-demand experiment using NYC's physical step.

This isolates charging admission, not passenger dispatch or learned reward.
Every vehicle requests one charge; AEVs use identical max-admitted/min-travel
OR-Tools scores in both policies. HEV station choices are fixed before rollout.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from src.NYCEnvironment import NYCEnvironment
from src.Action import ChargingAction, IdleAction
from benchmark_conservative_charging import solve_adapter

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vehicles', type=int, default=1000)
    p.add_argument('--hev', type=int, default=500)
    p.add_argument('--centers', type=int, choices=(3,4,5), default=3)
    p.add_argument('--hours', type=float, default=4.)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--scenarios', nargs='+', choices=('burst','staggered'), default=['burst','staggered'])
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/nyc_charging_rollout_1000')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if not 0 <= args.hev < args.vehicles or args.hours <= 0:
        p.error('Require 0 <= HEV < vehicles, hours > 0')
    return args


def build_environment(args, seed):
    env = NYCEnvironment(
        num_vehicles=args.vehicles, ev_num_vehicles=args.hev, random_seed=seed,
        parquet_path=str(ROOT/'nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet'),
        station_csv=str(ROOT/'nyedata/nyc_all_charging_stations.csv'),
        aev_charging_center_count=args.centers, start_date='2025-12-18', end_date='2025-12-18',
        start_hour=0., stop_hour=args.hours, epoch_length_sec=30.,
        assignmentgurobi=True, usemcmf=True, mcmf_solver='exact', mcmf_backend='ortools',
        mcmf_strict=True, daily_drop_off=False, ifreject=False,
    )
    env.adp_value = 0.; env.evaluatemode = True
    env.value_function = env.value_function_ev = None
    env.reset()
    env.active_requests = {}
    env.generate_requests = lambda: []  # controlled charge requests replace passenger demand
    env._should_log_timing = lambda *a, **k: False
    env.current_time = 0.
    for station in env.charging_manager.stations.values():
        station.current_vehicles.clear(); station.charging_queue.clear()
        station.charging_queue_notarrived.clear(); station.available_slots = station.max_capacity
        station.log_queue_arrivals = False
    return env


def install_workload(env, seed, scenario, horizon):
    rng = np.random.default_rng(seed)
    records = {}
    aev_ids = [vid for vid,v in env.vehicles.items() if v['type'] == 2]
    for vid, v in env.vehicles.items():
        v.update(battery=float(rng.uniform(.2,.4)), is_online=True, is_stationary=True,
                 assigned_request=None, passenger_onboard=None, charging_station=None,
                 charging_target=None, target_charging_station=None, target_location=None,
                 idle_target=None, penalty_timer=0, needs_emergency_charging=False,
                 charging_count=0, charging_time_left=0, completed_charging_durations_minutes=[])
        release = 0 if scenario == 'burst' else int(rng.integers(0, max(1,horizon//2)))
        records[vid] = dict(vehicle_id=vid, fleet='HEV' if v['type']==1 else 'AEV',
                            initial_location=int(v['location']), initial_soc=v['battery'],
                            release_epoch=release, assignment_epoch=None, station_id=None,
                            arrival_epoch=None, start_epoch=None, completion_epoch=None,
                            queued_steps=0, ever_queued=False, initial_charging=False,
                            initial_committed=False, fixed_station=None)
        if v['type']==1:
            sids = env._charging_station_ids_for_vehicle(vid)
            distances = np.array([env.get_distance_km(v['location'], env.charging_manager.stations[sid].location) for sid in sids])
            tied = np.flatnonzero(np.isclose(distances, distances.min()))
            records[vid]['fixed_station'] = int(sids[int(rng.choice(tied))])
    # Both policies start with the same occupied plugs and committed arrivals.
    cursor = 0
    for sid in env.aev_charging_station_ids:
        station = env.charging_manager.stations[sid]
        busy = min(10, station.max_capacity, max(0, len(aev_ids)//(4*len(env.aev_charging_station_ids))))
        inbound = min(5, busy)
        for _ in range(busy):
            vid = aev_ids[cursor]; cursor += 1
            v=env.vehicles[vid]; v.update(location=station.location, coordinates=env.zone_coords[station.location],
                                        battery=float(rng.uniform(.70,.78)), is_stationary=False, charging_station=sid)
            station.start_charging(str(vid)); env._set_vehicle_charging_session(vid)
            records[vid].update(release_epoch=0,assignment_epoch=0,station_id=int(sid),
                                arrival_epoch=0,start_epoch=0,initial_charging=True,
                                initial_soc=v['battery'],initial_location=int(v['location']))
        for _ in range(inbound):
            vid=aev_ids[cursor]; cursor+=1
            env.vehicles[vid]['is_stationary']=False
            env._move_vehicle_to_charging_station(vid,sid)
            records[vid].update(release_epoch=0,assignment_epoch=0,station_id=int(sid),initial_committed=True)
    return records


def summarize(records, rows, fleet, capacity, horizon, epoch_minutes=.5):
    rs=[r for r in records.values() if fleet=='ALL' or r['fleet']==fleet]
    ts=[r for r in rows if r['fleet']==fleet]
    arrivals=[r for r in rs if r['arrival_epoch'] is not None and not r['initial_charging']]
    started=[r for r in arrivals if r['start_epoch'] is not None]
    waited=[r for r in arrivals if r['ever_queued']]
    finished=[r for r in rs if r['completion_epoch'] is not None]
    ready=[r for r in rs if r['release_epoch']<horizon]
    native_wait=[max(0,r['start_epoch']-r['arrival_epoch'])*epoch_minutes for r in started]
    return dict(fleet=fleet,vehicle_count=len(rs),capacity=capacity,
        mean_occupied_slots=float(np.mean([r['occupied'] for r in ts])),
        utilization=float(np.mean([r['occupied'] for r in ts]))/capacity if capacity else 0.,
        mean_queue_length=float(np.mean([r['queue'] for r in ts])),
        max_queue_length=max((r['queue'] for r in ts),default=0),
        mean_wait_minutes_all_arrivals=sum(r['queued_steps'] for r in arrivals)*epoch_minutes/len(arrivals) if arrivals else None,
        mean_wait_minutes_queued_arrivals=sum(r['queued_steps'] for r in waited)*epoch_minutes/len(waited) if waited else 0.,
        native_mean_wait_minutes_started=float(np.mean(native_wait)) if native_wait else None,
        arrival_count=len(arrivals),started_arrival_count=len(started),queued_arrival_count=len(waited),
        newly_assigned_queued_count=sum(r['ever_queued'] and not r['initial_committed'] for r in arrivals),
        initial_committed_queued_count=sum(r['ever_queued'] and r['initial_committed'] for r in arrivals),
        same_epoch_queue_then_start_count=sum(r['ever_queued'] and r['arrival_epoch']==r['start_epoch'] for r in started),
        completion_count=len(finished),initial_charging_count=sum(r['initial_charging'] for r in rs),
        initial_committed_count=sum(r['initial_committed'] for r in rs),
        pending_dispatch_count=sum(r['assignment_epoch'] is None and r['release_epoch']<horizon for r in rs),
        pending_inbound_count=sum(r['assignment_epoch'] is not None and r['arrival_epoch'] is None for r in rs),
        pending_queue_count=sum(r['arrival_epoch'] is not None and r['start_epoch'] is None for r in rs),
        pending_charging_count=sum(r['start_epoch'] is not None and r['completion_epoch'] is None for r in rs),
        dispatch_delay_lower_bound_minutes=sum((min(horizon,r['assignment_epoch']) if r['assignment_epoch'] is not None else horizon)-r['release_epoch'] for r in ready)*epoch_minutes/len(ready) if ready else 0.,
        no_arrival_queue_observed=not bool(waited))


def run_case(args, seed, scenario, conservative, output):
    horizon=int(round(args.hours*120)); env=build_environment(args,seed)
    records=install_workload(env,seed,scenario,horizon)
    initial=json.dumps(list(records.values()),sort_keys=True)
    fingerprint=hashlib.sha256(initial.encode()).hexdigest()
    env.conservative_charging=conservative
    events=[]; rows=[]; station_rows=[]; admissions=[]
    original_arrival=env._mark_charging_queue_arrival
    def arrival(vid,sid):
        r=records[vid]
        if r['arrival_epoch'] is None:
            r['arrival_epoch']=float(env.current_time)
            events.append(dict(event='arrival',epoch=env.current_time,vehicle_id=vid,station_id=sid))
        return original_arrival(vid,sid)
    env._mark_charging_queue_arrival=arrival
    original_start=env._mark_charging_started
    def start(vid,sid):
        if records[vid]['start_epoch'] is None:
            records[vid]['start_epoch']=float(env.current_time)
            events.append(dict(event='start',epoch=env.current_time,vehicle_id=vid,station_id=sid))
        return original_start(vid,sid)
    env._mark_charging_started=start
    for sid,station in env.charging_manager.stations.items():
        original_add=station.add_to_queue; original_stop=station.stop_charging
        def add(vid,_sid=sid,_station=station,_old=original_add):
            added=_old(vid)
            if added:
                r=records[int(vid)]; r['ever_queued']=True
                events.append(dict(event='queue_enter',epoch=env.current_time,vehicle_id=int(vid),station_id=_sid,
                                   queue_length=len(_station.charging_queue),occupied=len(_station.current_vehicles)))
            return added
        def stop(vid,_sid=sid,_old=original_stop):
            stopped=_old(vid)
            if stopped:
                records[int(vid)]['completion_epoch']=float(env.current_time+1)
                events.append(dict(event='complete',epoch=env.current_time+1,vehicle_id=int(vid),station_id=_sid))
            return stopped
        station.add_to_queue=add; station.stop_charging=stop
    aev_sids=set(env.aev_charging_station_ids)
    capacities={'AEV':sum(s.max_capacity for sid,s in env.charging_manager.stations.items() if sid in aev_sids),
                'HEV':sum(s.max_capacity for sid,s in env.charging_manager.stations.items() if sid not in aev_sids)}
    capacities['ALL']=sum(capacities.values())
    original_update=env._update_environment
    def update():
        counts={'AEV':[0,0], 'HEV':[0,0]}
        for sid,s in env.charging_manager.stations.items():
            fleet='AEV' if sid in aev_sids else 'HEV'
            counts[fleet][0]+=len(s.current_vehicles); counts[fleet][1]+=len(s.charging_queue)
            assert len(s.current_vehicles)<=s.max_capacity
            assert s.available_slots==s.max_capacity-len(s.current_vehicles)
            for vid in s.charging_queue: records[int(vid)]['queued_steps']+=1
            if sid in aev_sids:
                station_rows.append(dict(epoch=int(env.current_time),station_id=int(sid),occupied=len(s.current_vehicles),
                                         queue=len(s.charging_queue),inbound=len(s.charging_queue_notarrived),capacity=s.max_capacity))
        counts['ALL']=[sum(x[i] for x in counts.values()) for i in range(2)]
        for fleet,(occupied,queue) in counts.items():
            relevant=[r for r in records.values() if fleet=='ALL' or r['fleet']==fleet]
            rows.append(dict(epoch=int(env.current_time),minutes=env.current_time*.5,fleet=fleet,
                             occupied=occupied,queue=queue,capacity=capacities[fleet],
                             completed=sum(r['completion_epoch'] is not None for r in relevant),
                             pending_dispatch=sum(r['assignment_epoch'] is None and r['release_epoch']<=env.current_time for r in relevant)))
        return original_update()
    env._update_environment=update
    wall=time.perf_counter(); solve_seconds=0.; charge_mask_seconds=0.
    for epoch in range(horizon):
        pending=[vid for vid,r in records.items() if r['assignment_epoch'] is None and r['release_epoch']<=epoch]
        # Fixed, pre-sampled HEV choices are an identical control in both arms.
        assignments={vid:f"charge_{records[vid]['fixed_station']}" for vid in pending if records[vid]['fleet']=='HEV'}
        ids=[vid for vid in pending if records[vid]['fleet']=='AEV']
        if ids:
            t=time.perf_counter();charge=env.generate_vehicle_chargerange(ids);charge_mask_seconds+=time.perf_counter()-t
            sids=sorted({sid for vid in ids for sid in env._charging_station_ids_for_vehicle(vid)})
            env._last_matrix_charge_station_ids=sids;env._last_matrix_request_ids=[]
            env._last_matrix_zone_indices=[];env._last_matrix_zone_target_ids=[]
            env._last_matrix_num_requests=0;env._last_matrix_num_stations=len(sids)
            env._last_matrix_num_zones=0;env._last_matrix_layout_ready=True
            mask=np.column_stack((charge,np.ones(len(ids),dtype=np.float32)))
            if np.any(charge):
                travel=np.array([[env.get_travel_time(env.vehicles[vid]['location'],env.charging_manager.stations[sid].location) for sid in sids] for vid in ids])
                benefit=(len(ids)+1)*(float(travel.max())+1)
                scores=np.column_stack((benefit-travel,np.zeros(len(ids))))
                selected,stats=solve_adapter(env,ids,mask,scores,True)
                if selected is None: raise RuntimeError(stats)
                solve_seconds+=stats['assignment_wall_seconds'];assignments.update(selected)
            else: assignments.update({vid:'waiting' for vid in ids})
        for vid,action_id in assignments.items():
            if not action_id.startswith('charge_'):continue
            sid=int(action_id.split('_')[1]);r=records[vid]
            assert sid in env._charging_station_ids_for_vehicle(vid), 'Solver station layout mismatch'
            r.update(assignment_epoch=epoch,station_id=sid)
            env.vehicles[vid]['is_stationary']=False
            if r['fleet']=='AEV':
                w=env._last_expected_charge_expansion['candidate_windows'].get((vid,sid),{})
                admissions.append(dict(vehicle_id=vid,station_id=sid,epoch=epoch,
                                       expected_arrival=w.get('expected_entercharge_station_time'),
                                       expected_completion=w.get('expected_charge_completion_time')))
            env._move_vehicle_to_charging_station(vid,sid)
            assert env.vehicles[vid]['charging_target']==sid, 'Dispatch was not accepted by NYC'
        actions={}
        for vid,r in records.items():
            v=env.vehicles[vid]
            if r['assignment_epoch'] is not None and r['completion_epoch'] is None:
                actions[vid]=ChargingAction([],r['station_id'],env._charge_duration_for_vehicle(vid),v['location'],v['battery'])
            else:
                actions[vid]=IdleAction([],v['coordinates'],v['coordinates'],v['location'],v['battery'])
        env.step(actions,{}, {})
        if epoch%120==0: print(f'epoch={epoch}/{horizon} completed={sum(r["completion_epoch"] is not None for r in records.values())}',flush=True)
    summary=[dict(scenario=scenario,seed=seed,policy='conservative' if conservative else 'current',
                  initial_state_sha256=fingerprint,wall_seconds=time.perf_counter()-wall,
                  charge_mask_seconds=charge_mask_seconds,assignment_seconds=solve_seconds,
                  **summarize(records,rows,fleet,capacities[fleet],horizon)) for fleet in ('AEV','HEV','ALL')]
    for name,data in [('sessions',list(records.values())),('timeseries',rows),('stations',station_rows),('events',events),('admissions',admissions)]:
        with (output/f'{name}.jsonl').open('w') as f:
            for row in data:f.write(json.dumps(row)+'\n')
    assert len({r['vehicle_id'] for r in records.values()})==args.vehicles
    assert sum(r['completion_epoch'] is not None for r in records.values())==env.charge_finished
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main(argv=None):
    args=parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items() if k!='resume'}
    manifest=args.output_dir/'manifest.json'
    metadata=dict(config=config,python=platform.python_version(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('benchmark_nyc_charging_rollout.py','src/NYCEnvironment.py','src/conservative_charging.py','src/GurobiOptimizer.py')},
                  policy='Controlled one-charge-per-vehicle workload; no passenger requests, no value network; AEV max-admitted then min-travel on existing adapter final graph; fixed HEV nearest-station tie choices.',
                  wait_definition='30-second queue exposure sampled after action execution and before charging update; includes same-epoch queue/release. Native arrival-to-start timestamps reported separately. Unfinished waits/delays are censored lower bounds.',
                  occupancy_definition='Time mean of occupied physical plugs sampled after action execution, before update; station records cover AEV centers; fleet aggregates include public stations.')
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old!=metadata:raise ValueError('Resume requires identical configuration and source hashes')
    else:manifest.write_text(json.dumps(metadata,indent=2))
    all_rows=[]
    for scenario in args.scenarios:
        for seed in args.seeds:
            paired=[]
            for conservative in (False,True):
                policy='conservative' if conservative else 'current';out=args.output_dir/f'{scenario}_seed{seed}_{policy}'
                out.mkdir(exist_ok=args.resume)
                if args.resume and (out/'summary.json').exists():rows=json.loads((out/'summary.json').read_text())
                else:
                    with (out/'run.log').open('w') as log, contextlib.redirect_stdout(log): rows=run_case(args,seed,scenario,conservative,out)
                paired.append(rows[0]['initial_state_sha256']);all_rows.extend(rows)
                (args.output_dir/'summary.json').write_text(json.dumps(all_rows,indent=2))
                print(scenario,seed,policy,'AEV:',{k:rows[0][k] for k in ('utilization','mean_wait_minutes_all_arrivals','max_queue_length','completion_count','queued_arrival_count','wall_seconds')},flush=True)
            assert paired[0]==paired[1], 'Unpaired initial states'
    print('Saved:',args.output_dir,flush=True)

if __name__=='__main__':main()
