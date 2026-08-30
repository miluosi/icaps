"""Shared integrated-then-limited-hold repair for NYC and synthetic simulators.

Only explicit stage-0 hold edges release AEV capacity to stage 2. Committed
actions are applied once, before constructing the residual feasible graph.
The repair solver is myopic; its outcome is part of one integrated macro TD.
"""
from collections import Counter
from dataclasses import replace
import time

import numpy as np

from src.Action import ChargingAction, IdleAction, ServiceAction
from .state_snapshot import StateSnapshotBuilder
from .target_builder import RecourseTargetBuilder
from .types import ActionType, FeasibleEdgeSnapshot, JointActionSnapshot, RequestSnapshot


def _coords(env, location):
    return getattr(env, 'zone_coords', {}).get(location, (location % env.grid_size, location // env.grid_size))


def _available_requests(env):
    busy = {v.get(k) for v in env.vehicles.values() for k in ('assigned_request', 'passenger_onboard')}
    return {rid for rid, req in env.active_requests.items()
            if rid not in busy and env.current_time <= req.pickup_deadline}


def add_hold_edges(env, graph, vehicle_ids):
    edges = list(graph.edges)
    for vid in vehicle_ids:
        v = env.vehicles[vid]
        if int(v['type']) != 2 or v['battery'] <= getattr(env, 'critical_charging_battery', .1):
            continue
        # Hold is a within-epoch reservation, not the restricted ordinary wait.
        edges.append(FeasibleEdgeSnapshot(
            edge_id=f'{graph.graph_id}:hold:{vid}', vehicle_id=vid, vehicle_type=2,
            action_type=ActionType.WAIT, action_id='hold_for_repair',
            target_location=v['location'], post_action_location=v['location'],
            post_action_duration=0., response_model_hash=getattr(env, 'ev_response_model_hash', None),
            metadata=(('repair_reserve', True),)))
    return replace(graph, edges=tuple(edges))


def residual_graph(graph, initial_graph, held_ids, candidate_ids):
    """Filter continuing commits and enforce resources left by the first plan."""
    committed = [e for e in initial_graph.edges if e.edge_id in initial_graph.selected_edge_ids
                 and not dict(e.metadata).get('repair_reserve', False)
                 and not dict(e.metadata).get('continuing', False)]
    # A rejected EV offer releases its request capacity. It is not a committed
    # service, even though its edge remains selected in the stage-0 replay.
    used = Counter((e.resource_type, e.resource_id) for e in committed if e.resource_type
                   and not (e.resource_type == 'request' and e.request_id in candidate_ids))
    capacities = {(e.resource_type, e.resource_id): e.resource_capacity for e in initial_graph.edges if e.resource_type}
    edges = []
    for edge in graph.edges:
        if edge.vehicle_id not in held_ids or dict(edge.metadata).get('continuing', False):
            continue
        if edge.request_id is not None and edge.request_id not in candidate_ids:
            continue
        key = (edge.resource_type, edge.resource_id)
        if edge.resource_type and key in capacities:
            capacity = min(edge.resource_capacity, capacities[key] - used[key])
            if capacity <= 0:
                continue
            edge = replace(edge, resource_capacity=capacity)
        edges.append(edge)
    return replace(graph, edges=tuple(edges), selected_edge_ids=())


def build_stage_graph(
    env,
    vehicle_ids,
    *,
    stage,
    state,
    initial=None,
    candidates=None,
    include_hold_edges=None,
):
    """Build and solve a stage through the shared Integrated/Samitha oracle."""
    env._active_recourse_stage = stage
    env._active_stage_state_snapshot = state
    if vehicle_ids:
        matrix, nr, ns, nz = env.generate_whole_matrix(vehicle_ids, rebalance_num=len(vehicle_ids))
        structured = env.generate_vehicle_qvalue_withoutqnetwork(vehicle_ids)
    else:
        matrix = structured = np.zeros((0, 1))
        nr = ns = nz = 0
    graph = StateSnapshotBuilder.feasible_graph_from_matrix(
        env, vehicle_ids, np.asarray(matrix), np.asarray(structured), np.asarray(structured),
        num_requests=nr, num_stations=ns, num_zones=nz, stage_id=stage,
        solver_backend=str(getattr(env, 'mcmf_backend', 'primal_dual')), state=state)
    counters = getattr(env, '_integrated_repair_metrics', {})
    if stage == 0:
        counters['integrated_stage0_graph_count'] = (
            counters.get('integrated_stage0_graph_count', 0) + 1
        )
        counters['integrated_service_candidate_count'] = (
            counters.get('integrated_service_candidate_count', 0)
            + sum(edge.action_type == ActionType.SERVICE for edge in graph.edges)
        )
    env._integrated_repair_metrics = counters
    if include_hold_edges is None:
        include_hold_edges = bool(
            getattr(env, 'integrated_repair_hold_enabled', True)
        )
    if stage == 0:
        if include_hold_edges:
            graph = add_hold_edges(env, graph, vehicle_ids)
    else:
        graph = residual_graph(graph, initial, set(vehicle_ids), set(candidates))
        if not candidates:
            graph = replace(graph, edges=())
    if stage == 0 and getattr(env, 'adp_value', 0) > 0 and env.value_function is not None:
        scorer = getattr(env.value_function, '_graph_edge_scores', None)
        if scorer is None:
            raise ValueError('integrated_repair requires a solver-consistent joint value function')
        scores, _ = scorer(graph, target_context=False)
        graph = replace(graph, edges=tuple(replace(e, collection_score=scores[e.edge_id]) for e in graph.edges))
    else:
        graph = replace(graph, edges=tuple(replace(e, collection_score=e.structured_score) for e in graph.edges))
    if (
        str(getattr(env, 'mcmf_backend', 'primal_dual')) == 'gurobi_network'
        and not hasattr(env, 'gurobi_optimizer')
    ):
        from src.GurobiOptimizer import GurobiOptimizer
        env.gurobi_optimizer = GurobiOptimizer(env)
    builder = RecourseTargetBuilder.from_environment(env)
    selection = builder.project(graph, {e.edge_id: e.collection_score for e in graph.edges})
    if graph.edges:
        counters['exact_assignment_call_count'] = (
            counters.get('exact_assignment_call_count', 0) + 1
        )
    env._integrated_repair_solver_seconds = (
        getattr(env, '_integrated_repair_solver_seconds', 0.) + builder.last_solver_runtime_seconds)
    builder.verify_feasible(graph, selection)
    return graph.with_selected(selection, status='optimal')


def build_integrated_stage0(
    env,
    vehicle_ids,
    *,
    state,
    include_hold_edges=False,
):
    """Canonical stage-0 graph/oracle shared by Integrated and Samitha.

    ``include_hold_edges`` is false for Integrated and true only for the
    Samitha limited-repair treatment (or an explicit controlled test).
    """
    return build_stage_graph(
        env,
        vehicle_ids,
        stage=0,
        state=state,
        include_hold_edges=bool(include_hold_edges),
    )


def _selected(graph):
    if graph is None:
        return []
    return [e for e in graph.edges if e.edge_id in graph.selected_edge_ids]


def _action_target(env, edge):
    if edge.action_type == ActionType.SERVICE:
        return env.active_requests[edge.request_id]
    if edge.action_type == ActionType.CHARGE:
        return f'charge_{edge.station_id}'
    if edge.action_type == ActionType.WAIT:
        return 'waiting'
    # The NYC executor's legacy string contains an index in hotspot_locations,
    # not a TLC zone ID. The serialized graph always uses the actual location.
    return f'idle_at_{list(env.hotspot_locations).index(edge.target_location)}'


def execute_commits(env, graph, actions, stores, stores_ev):
    edges = [e for e in _selected(graph) if not dict(e.metadata).get('continuing', False)
             and not dict(e.metadata).get('repair_reserve', False)]
    if hasattr(env, '_process_integrated_assignments'):
        return env._process_integrated_assignments(
            {e.vehicle_id: _action_target(env, e) for e in edges}, actions, stores, stores_ev)
    assigned = charging = 0
    for e in edges:
        vid, v = e.vehicle_id, env.vehicles[e.vehicle_id]
        loc, battery = v['location'], v['battery']
        if e.action_type == ActionType.SERVICE:
            req = env.active_requests[e.request_id]
            accepted = env._assign_request_to_vehicle(vid, e.request_id)
            action = ServiceAction([], e.request_id, loc, battery, req_num=len(env.active_requests))
            action.metadata.request_snapshot = RequestSnapshot.from_request(req)
            action.was_rejected = not accepted
            if accepted:
                assigned += 1
            else:
                v['assigned_request'] = None
                v['idle_target'] = _coords(env, loc)
                v['target_location'] = loc
        elif e.action_type == ActionType.CHARGE:
            env._register_aev_notarrived_reservation(vid, e.station_id)
            env._move_vehicle_to_charging_station(vid, e.station_id)
            action = ChargingAction([], e.station_id, env.charge_duration, loc, battery)
            charging += 1
        else:
            target = e.target_location if e.action_type == ActionType.RELOCATE else loc
            v['target_location'] = target
            v['idle_target'] = _coords(env, target)
            v['is_stationary'] = e.action_type == ActionType.WAIT
            action = IdleAction([], _coords(env, loc), _coords(env, target), loc, battery)
            action.learning_action_type = 'wait' if e.action_type == ActionType.WAIT else 'reloc'
        actions[vid] = action
        env._update_storeaction(vid, action, stores_ev if v['type'] == 1 else stores, is_ev=v['type'] == 1)
    return assigned, charging


def _fallback(env, actions):
    if hasattr(env, '_build_fallback_actions'):
        env._build_fallback_actions(actions)
        return
    for vid, v in env.vehicles.items():
        if vid in actions:
            continue
        loc, battery = v['location'], v['battery']
        rid = v.get('passenger_onboard') if v.get('passenger_onboard') is not None else v.get('assigned_request')
        sid = v.get('charging_station') if v.get('charging_station') is not None else v.get('charging_target')
        if rid is not None:
            actions[vid] = ServiceAction([], rid, loc, battery)
        elif sid is not None:
            actions[vid] = ChargingAction([], sid, env.charge_duration, loc, battery)
        else:
            target = v.get('target_location')
            if not isinstance(target, int):
                target = loc
            actions[vid] = IdleAction([], _coords(env, loc), _coords(env, target), loc, battery)


def simulate_integrated_repair(
    env,
    agents=None,
    current_requests=None,
    rebalance=True,
    *,
    include_hold_edges=None,
    decision_mode='integrated_repair',
):
    started = time.perf_counter()
    RecourseTargetBuilder.validate_variant(
        getattr(env, 'recourse_variant', 'legacy'), decision_mode
    )
    env.decision_mode = decision_mode
    env._leader_is_ev = None
    env._integrated_repair_solver_seconds = 0.
    pending = env._begin_joint_collection(decision_mode)
    initial_state = pending.pre_state if pending else StateSnapshotBuilder.build(env)
    actions = {}
    stores = {vid: env.storeactions.get(vid) for vid in env.vehicles}
    stores_ev = {vid: env.storeactions_ev.get(vid) for vid in env.vehicles}
    if hasattr(env, '_ev_charging_phase'):
        env._ev_charging_phase(actions, stores_ev)
    else:
        for vid, v in env.vehicles.items():
            if (v['type'] == 1 and v.get('is_online', True) and all(v.get(k) is None for k in
                ('charging_station', 'assigned_request', 'passenger_onboard', 'target_location', 'idle_target'))
                and env._should_consider_ev_charging(vid)):
                p, stations = env.compute_ev_charge_probability(vid)
                if stations and (env._charge_uniform(vid, 'charge_decision') < p or v['battery'] <= .2):
                    draw = env._charge_uniform(vid, 'charge_station')
                    sid = next(iter(stations))
                    cumulative = 0.
                    for candidate, weight in stations.items():
                        cumulative += float(weight)
                        if draw <= cumulative:
                            sid = int(candidate)
                            break
                    location, battery = v['location'], v['battery']
                    env._move_vehicle_to_charging_station(vid, sid)
                    actions[vid] = ChargingAction([], sid, env.charge_duration, location, battery)
                    env._update_storeaction(vid, actions[vid], stores_ev, is_ev=True)
                else:
                    v['no_charge_cooldown_until'] = env.current_time + 5
    ids = [vid for vid, v in env.vehicles.items() if vid not in actions and v.get('is_online', True)
           and all(v.get(k) is None for k in ('assigned_request', 'passenger_onboard', 'charging_station', 'charging_target'))
           and v.get('penalty_timer', 0) <= 0 and (v['type'] == 1 or v.get('target_location') is None)] if rebalance else []
    if hasattr(env, '_is_vehicle_committed_to_charging'):
        ids = [vid for vid in ids if not env._is_vehicle_committed_to_charging(vid)]
    # Human charging has been realized before the integrated dispatch. Capture
    # it so continuing charge edges, reservations and the pre-plan state agree.
    initial_state = StateSnapshotBuilder.build(env)
    if pending:
        pending.pre_state = initial_state
    initial_requests = _available_requests(env)
    initial = build_integrated_stage0(
        env,
        ids,
        state=initial_state,
        include_hold_edges=(
            bool(getattr(env, 'integrated_repair_hold_enabled', True))
            if include_hold_edges is None else bool(include_hold_edges)
        ),
    )
    env._last_feasible_graph_snapshot = initial
    held = {e.vehicle_id for e in _selected(initial) if dict(e.metadata).get('repair_reserve', False)}
    committed = tuple(e.edge_id for e in _selected(initial) if e.vehicle_type == 2 and e.vehicle_id not in held)
    execute_commits(env, initial, actions, stores, stores_ev)
    rejected = set(env.request_lifecycle.rejection_outcome(epoch_id=initial_state.epoch_id).rejected_request_ids)
    candidates = initial_requests & _available_requests(env)
    stage2_enabled = decision_mode == 'integrated_repair'
    repair_candidates = candidates if stage2_enabled else set()
    labels = (
        {rid: 'rejected' if rid in rejected else 'unoffered' for rid in repair_candidates}
        if stage2_enabled else {}
    )
    residual_state = StateSnapshotBuilder.build(env, request_labels=labels)
    repair = None
    if stage2_enabled:
        repair = build_stage_graph(
            env, sorted(held), stage=2, state=residual_state,
            initial=initial, candidates=repair_candidates,
        )
        env._last_feasible_graph_snapshot = repair
        feasible_requests = {
            e.request_id for e in repair.edges if e.request_id is not None
        }
        for rid, category in labels.items():
            env.request_lifecycle.mark_residual(
                rid, epoch_id=initial_state.epoch_id, category=category,
                eligible=rid in feasible_requests,
                repair_architecture='integrated_repair',
            )
        execute_commits(env, repair, actions, stores, stores_ev)
        for edge in _selected(repair):
            if (edge.action_type == ActionType.SERVICE
                    and env.vehicles[edge.vehicle_id].get('assigned_request') == edge.request_id):
                env.request_lifecycle.record_integrated_repair_assignment(
                    edge.request_id, vehicle_id=edge.vehicle_id,
                    epoch_id=initial_state.epoch_id,
                )
    _fallback(env, actions)
    if pending:
        pending.ev_stage_graph, pending.aev_stage_graph = initial, repair
        pending.ev_joint_action = JointActionSnapshot.from_graph(initial)
        pending.aev_joint_action = (
            JointActionSnapshot.from_graph(repair) if repair is not None else None
        )
        pending.residual_state = residual_state
        pending.committed_aev_edge_ids = committed
        pending.repair_hold_aev_ids = tuple(sorted(held))
        pending.repair_candidate_request_ids = tuple(sorted(repair_candidates))
        for graph in (initial, repair):
            if graph is None:
                continue
            for edge in _selected(graph):
                if edge.vehicle_id not in actions or dict(edge.metadata).get('repair_reserve', False):
                    continue
                action = actions[edge.vehicle_id]
                action.metadata.transition_id = pending.transition_id
                action.metadata.stage_id = graph.stage_id
                action.metadata.state_snapshot = graph.state
                action.metadata.feasible_graph_snapshot = graph
                action.metadata.joint_action_snapshot = JointActionSnapshot.from_graph(graph)
                action.metadata.residual_state_snapshot = residual_state
                env._pending_recourse_actions[edge.vehicle_id] = action
    stats = getattr(env, '_integrated_repair_metrics', {})
    hold_candidates = sum(
        bool(dict(edge.metadata).get('repair_reserve', False))
        for edge in initial.edges
    )
    repair_assignments = sum(
        e.action_type == ActionType.SERVICE for e in _selected(repair)
    )
    additions = dict(initial_integrated_ev_offer_count=sum(e.vehicle_type == 1 and e.action_type == ActionType.SERVICE and not dict(e.metadata).get('continuing') for e in _selected(initial)),
                     initial_integrated_aev_commit_count=len(committed), aev_hold_for_repair_count=len(held),
                     hold_candidate_count=hold_candidates,
                     hold_selected_count=len(held),
                     hold_marginal_score=sum(
                         e.collection_score for e in _selected(initial)
                         if dict(e.metadata).get('repair_reserve', False)
                     ),
                     repair_usage_per_hold=(repair_assignments / max(1, len(held))),
                     unused_hold_count=max(0, len(held) - repair_assignments),
                     repair_candidate_rejected_count=len(repair_candidates & rejected),
                     repair_candidate_unassigned_count=len(repair_candidates - rejected),
                     samitha_repair_assignment_count=repair_assignments,
                     committed_aev_reassignment_count=0)
    for key, count in additions.items():
        stats[key] = stats.get(key, 0) + count
    env._integrated_repair_metrics = stats
    stats['hold_selection_rate'] = (
        stats.get('hold_selected_count', 0) /
        max(1, stats.get('hold_candidate_count', 0))
    )
    env._last_integrated_repair_graphs = (initial, repair)
    env._active_recourse_stage, env._active_stage_state_snapshot = 0, None
    env._last_simulation_profile = {'total_time_sec': time.perf_counter() - started}
    env._last_rebalancing_profile = {'solver_name': 'integrated_limited_hold',
                                     'solver_time_sec': env._integrated_repair_solver_seconds}
    if current_requests:
        env.update_recent_requests(current_requests)
    return actions, stores, stores_ev


def simulate_integrated_control(env, agents=None, current_requests=None, rebalance=True):
    """Integrated no-repair control using the exact Samitha stage-0 pipeline."""
    return simulate_integrated_repair(
        env,
        agents,
        current_requests,
        rebalance,
        include_hold_edges=False,
        decision_mode='integrated',
    )
