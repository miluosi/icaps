from typing import List, Dict
from .Request import Request
import random
import numpy as np
import gurobipy as gp
import networkx as nx
from src.ValueFunction_pytorch_bayes import PyTorchChargingValueFunction
import time
import os
import math

from src.exact_mcmf import build_reduced_problem, solve_exact


_GPU_MCMF_KERNELS = None


def _get_gpu_mcmf_kernels():
    global _GPU_MCMF_KERNELS
    if _GPU_MCMF_KERNELS is not None:
        return _GPU_MCMF_KERNELS

    from numba import cuda

    @cuda.jit
    def copy_float_kernel(src, dst, num_items):
        idx = cuda.grid(1)
        if idx < num_items:
            dst[idx] = src[idx]

    @cuda.jit
    def relax_edges_kernel(
        dist,
        next_dist,
        edge_src,
        edge_dst,
        edge_cap,
        edge_cost,
        num_edges,
        inf_value,
        tol,
        changed,
    ):
        idx = cuda.grid(1)
        if idx >= num_edges or edge_cap[idx] <= 0:
            return

        u = edge_src[idx]
        v = edge_dst[idx]
        dist_u = dist[u]
        if dist_u >= inf_value:
            return

        candidate = dist_u + edge_cost[idx]
        old_value = cuda.atomic.min(next_dist, v, candidate)
        if candidate + tol < old_value:
            changed[0] = 1

    @cuda.jit
    def write_parent_kernel(
        dist,
        next_dist,
        edge_src,
        edge_dst,
        edge_cap,
        edge_cost,
        edge_graph_idx,
        num_edges,
        inf_value,
        tol,
        parent_node,
        parent_edge,
    ):
        idx = cuda.grid(1)
        if idx >= num_edges or edge_cap[idx] <= 0:
            return

        u = edge_src[idx]
        v = edge_dst[idx]
        dist_u = dist[u]
        if dist_u >= inf_value:
            return

        candidate = dist_u + edge_cost[idx]
        if abs(candidate - next_dist[v]) <= tol:
            parent_node[v] = u
            parent_edge[v] = edge_graph_idx[idx]

    _GPU_MCMF_KERNELS = (
        cuda,
        copy_float_kernel,
        relax_edges_kernel,
        write_parent_kernel,
    )
    return _GPU_MCMF_KERNELS


class GurobiOptimizer:
    """Gurobi-based optimization for vehicle assignment and rebalancing"""
    
    def __init__(self, env, num_threads=16):
        self.env = env
        self.num_threads = num_threads  # Global thread configuration
        self.network_time_limit = float(getattr(env, 'gurobi_network_time_limit', 10.0))
        self._auction_solver_cache = {}
        self._gurobi_runtime_failed = False
        self.mcmf_backend = getattr(env, 'mcmf_backend', 'auto')
        self.mcmf_strict = bool(getattr(env, 'mcmf_strict', True))
        self.mcmf_cost_scale = int(getattr(env, 'mcmf_cost_scale', 10_000))
        self.mcmf_graph_reduction = bool(
            getattr(env, 'mcmf_graph_reduction', True)
        )
        self.mcmf_verify = bool(getattr(env, 'mcmf_verify', False))
        # Only import Gurobi if it's available
        try:
            import gurobipy as gp
            from gurobipy import GRB
            self.gp = gp
            self.GRB = GRB
            self.available = True
            print(f"✓ Gurobi optimizer available (Threads: {self.num_threads})")
        except ImportError:
            print("⚠ Gurobi not available, using heuristic methods")
            self.available = False

    def _movement_cost(self, distance: float) -> float:
        """Return the environment's common distance operating cost."""
        operating_cost = getattr(self.env, 'operating_cost_per_km', None)
        if operating_cost is not None:
            return -abs(float(operating_cost)) * max(0.0, float(distance or 0.0))
        return float(getattr(self.env, 'movingpenalty', -0.1)) * max(
            0.0,
            float(distance or 0.0),
        )

    def _fallback_after_gurobi_failure(
        self,
        vehicle_ids,
        available_requests,
        vehicle_action_matrix,
        batch_q_value,
        *,
        iflp=True,
        ev_only=False,
        error=None,
    ):
        first_failure = not self._gurobi_runtime_failed
        self._gurobi_runtime_failed = True
        if first_failure:
            print(
                f"⚠ Gurobi runtime unavailable ({error}); "
                "switching this environment to CPU MCMF",
                flush=True,
            )
        if ev_only:
            return self._np_vehicle_rebalancing_network_ev(
                vehicle_ids,
                available_requests,
                vehicle_action_matrix,
                batch_q_value,
                iflp=iflp,
            )
        return self._np_vehicle_rebalancing_network(
            vehicle_ids,
            available_requests,
            vehicle_action_matrix,
            batch_q_value,
            iflp=iflp,
        )

    def _qvalue_heuristic_fallback(
        self,
        vehicle_ids,
        vehicle_action_matrix,
        batch_q_value,
    ):
        stations = []
        if hasattr(self.env, 'charging_manager'):
            stations = [
                station
                for station in self.env.charging_manager.stations.values()
                if station.available_slots > 0
            ]
        return self._heuristic_assignment_fastqvalue(
            vehicle_ids,
            stations,
            vehicle_action_matrix,
            batch_q_value,
        )

    def _get_relocation_targets(self):
        targets = getattr(self.env, 'relocation_target_ids', None)
        if targets is not None:
            return targets
        return getattr(self.env, 'hotspot_locations', [])

    def _get_relocation_target_count(self):
        targets = self._get_relocation_targets()
        if targets:
            return len(targets)
        return getattr(self.env, 'hotspot_locations_num', getattr(self.env, 'num_zones', 4))

    def _should_extract_solution(self, model):
        return model.SolCount > 0 and model.status in {
            self.GRB.OPTIMAL,
            self.GRB.SUBOPTIMAL,
            self.GRB.TIME_LIMIT,
            self.GRB.INTERRUPTED,
        }

    def _get_matrix_action_layout(self, available_requests, num_action):
        """Return the exact action-column layout recorded by generate_whole_matrix."""
        action_cols = max(int(num_action) - 1, 0)

        def _int_attr(name, default):
            try:
                return int(getattr(self.env, name, default))
            except (TypeError, ValueError):
                return default

        stored_nr = _int_attr('_last_matrix_num_requests', -1)
        stored_ns = _int_attr('_last_matrix_num_stations', -1)
        stored_nz = _int_attr('_last_matrix_num_zones', -1)
        metadata_matches = (
            stored_nr >= 0
            and stored_ns >= 0
            and stored_nz >= 0
            and stored_nr + stored_ns + stored_nz == action_cols
        )

        if metadata_matches:
            num_requests = stored_nr
            charge_station_ids = list(getattr(self.env, '_last_matrix_charge_station_ids', []))
            zone_indices = list(getattr(self.env, '_last_matrix_zone_indices', []))
            zone_targets = list(getattr(self.env, '_last_matrix_zone_target_ids', []))
            num_charging = min(stored_ns, len(charge_station_ids))
            num_zones = min(stored_nz, len(zone_indices))
            charge_station_ids = charge_station_ids[:num_charging]
            zone_indices = zone_indices[:num_zones]
            zone_targets = zone_targets[:num_zones]
        else:
            num_requests = min(len(available_requests), action_cols)
            remaining_cols = max(action_cols - num_requests, 0)
            charge_station_ids = (
                sorted(self.env.charging_manager.stations.keys())
                if hasattr(self.env, 'charging_manager') else []
            )
            num_charging = min(len(charge_station_ids), remaining_cols)
            charge_station_ids = charge_station_ids[:num_charging]
            remaining_cols -= num_charging
            zone_indices = list(range(self._get_relocation_target_count()))
            zone_targets = list(self._get_relocation_targets())
            num_zones = min(len(zone_indices), remaining_cols)
            zone_indices = zone_indices[:num_zones]
            zone_targets = zone_targets[:num_zones]

        request_by_id = {
            getattr(req, 'request_id', None): req
            for req in available_requests
        }
        matrix_request_ids = list(getattr(self.env, '_last_matrix_request_ids', []))
        if metadata_matches and matrix_request_ids:
            matrix_requests = [
                request_by_id.get(request_id)
                for request_id in matrix_request_ids[:num_requests]
            ]
        else:
            matrix_requests = list(available_requests[:num_requests])
        if len(matrix_requests) < num_requests:
            matrix_requests.extend([None] * (num_requests - len(matrix_requests)))

        return {
            'num_requests': num_requests,
            'num_charging': num_charging,
            'num_zones': num_zones,
            'requests': matrix_requests,
            'charge_station_ids': charge_station_ids,
            'zone_indices': zone_indices,
            'zone_targets': zone_targets,
        }

    def _charging_station_vacancy(self, station):
        reserved = len(getattr(station, 'current_vehicles', []) or [])
        if getattr(self.env, 'reserve_inbound_charging_capacity', False):
            reserved += len(getattr(station, 'charging_queue', []) or [])
            reserved += len(
                getattr(station, 'charging_queue_notarrived', []) or []
            )
        queue_capacity = max(
            0,
            int(getattr(self.env, 'station_queue_capacity', 0) or 0),
        )
        physical_capacity = int(getattr(
            station,
            'max_capacity',
            int(getattr(station, 'available_slots', 0) or 0) + reserved,
        ))
        admission_capacity = physical_capacity + queue_capacity
        return max(0, admission_capacity - reserved)

    def _exact_mcmf_selection(self):
        """Return the exact backend requested through the legacy solver knob."""
        solver = getattr(self.env, 'mcmf_solver', None)
        if solver is None:
            return None
        normalized = str(solver).strip().lower().replace('-', '_')
        if normalized == 'exact':
            return str(getattr(self.env, 'mcmf_backend', self.mcmf_backend))
        if normalized in {
            'auto', 'ortools', 'primal_dual', 'primaldual', 'python',
            'gurobi_network', 'exact_gurobi',
        }:
            return 'gurobi_network' if normalized == 'exact_gurobi' else normalized
        return None

    @staticmethod
    def _mcmf_numpy(value, *, dtype=None):
        """Move a tensor/array to CPU once instead of synchronizing per edge."""
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        return np.asarray(value, dtype=dtype)

    def _exact_mcmf_fallback_values(self, feasibility, q_values):
        configured = getattr(self.env, 'mcmf_fallback_value', None)
        num_vehicles = feasibility.shape[0]
        if configured is None:
            # EAGR requires a real, capacity-nonbinding outside action in the
            # feasibility matrix.  Do not silently add a synthetic action,
            # because that would change the full assignment problem stated in
            # the manuscript.  A caller that truly needs a private fallback
            # can still provide mcmf_fallback_value explicitly.
            return None
        values = np.asarray(configured, dtype=np.float64)
        if values.ndim == 0:
            return np.full(num_vehicles, float(values), dtype=np.float64)
        values = values.reshape(-1)
        if values.shape != (num_vehicles,):
            raise ValueError(
                'mcmf_fallback_value must be scalar or one value per vehicle'
            )
        return values

    def _build_exact_mcmf_inputs(
        self,
        vehicle_ids,
        available_requests,
        vehicle_action_matrix,
        batch_q_value,
        *,
        ev_only=False,
    ):
        feasibility = self._mcmf_numpy(vehicle_action_matrix) != 0
        q_values = self._mcmf_numpy(batch_q_value, dtype=np.float64)
        if feasibility.ndim != 2 or q_values.shape != feasibility.shape:
            raise ValueError(
                'vehicle_action_matrix and batch_q_value must have the same 2-D shape'
            )
        if feasibility.shape[0] != len(vehicle_ids):
            raise ValueError('MCMF matrix row count does not match vehicle_ids')
        if not np.all(np.isfinite(q_values)):
            raise ValueError('MCMF Q-value matrix contains NaN or infinity')

        num_vehicles, num_actions = feasibility.shape
        layout = self._get_matrix_action_layout(available_requests, num_actions)
        num_requests = layout['num_requests']
        num_charging = layout['num_charging']

        if not ev_only:
            # Keep the last reloc/wait column as the EV outside action, matching
            # the existing MCMF interface.  Other charge/zone columns are AEV-only.
            for row, vehicle_id in enumerate(vehicle_ids):
                if self.env.vehicles[vehicle_id]['type'] == 1:
                    feasibility[row, num_requests:max(num_actions - 1, num_requests)] = False

        action_capacities = np.full(num_actions, num_vehicles, dtype=np.int64)
        action_capacities[:num_requests] = 1
        if not ev_only:
            station_ids = layout['charge_station_ids']
            for local_index in range(num_charging):
                capacity = 0
                if local_index < len(station_ids) and hasattr(self.env, 'charging_manager'):
                    station = self.env.charging_manager.stations.get(
                        station_ids[local_index]
                    )
                    if station is not None:
                        capacity = self._charging_station_vacancy(station)
                action_capacities[num_requests + local_index] = capacity

        fallback_values = self._exact_mcmf_fallback_values(feasibility, q_values)
        return feasibility, q_values, action_capacities, fallback_values, layout

    def _decode_exact_mcmf_assignments(
        self,
        action_by_vehicle,
        vehicle_ids,
        layout,
        *,
        ev_only=False,
    ):
        assignments = {}
        num_requests = layout['num_requests']
        num_charging = layout['num_charging']
        num_zones = layout['num_zones']
        matrix_requests = layout['requests']
        station_ids = layout['charge_station_ids']
        zone_indices = layout['zone_indices']

        for row, vehicle_id in enumerate(vehicle_ids):
            action = int(action_by_vehicle[row])
            is_ev = self.env.vehicles[vehicle_id]['type'] == 1
            if action < 0:
                assignments[vehicle_id] = 'reloc' if is_ev or ev_only else 'waiting'
            elif action < num_requests:
                request = matrix_requests[action] if action < len(matrix_requests) else None
                assignments[vehicle_id] = (
                    request if request is not None else ('reloc' if is_ev else 'waiting')
                )
            elif ev_only:
                assignments[vehicle_id] = 'reloc'
            elif action < num_requests + num_charging:
                station_index = action - num_requests
                if station_index < len(station_ids):
                    assignments[vehicle_id] = f'charge_{station_ids[station_index]}'
                else:
                    assignments[vehicle_id] = 'waiting'
            elif action < num_requests + num_charging + num_zones:
                zone_index = action - num_requests - num_charging
                if is_ev:
                    assignments[vehicle_id] = 'reloc'
                else:
                    mapped = (
                        zone_indices[zone_index]
                        if zone_index < len(zone_indices) else zone_index
                    )
                    assignments[vehicle_id] = f'idle_at_{mapped}'
            else:
                assignments[vehicle_id] = 'reloc' if is_ev else 'waiting'
        return assignments

    def _record_exact_mcmf_result(self, result, *, build_time, solve_time):
        total_time = build_time + solve_time
        scale = int(getattr(self.env, 'mcmf_cost_scale', self.mcmf_cost_scale))
        generator_stats = getattr(self.env, 'qvalue_precision_last', {}) or {}
        generator_matches = (
            int(generator_stats.get('qvalue_scale', -1)) == result.qvalue_scale
            and int(generator_stats.get('qvalue_entries', -1)) == result.qvalue_entries
        )
        generator_rounded_entries = (
            int(generator_stats.get('qvalue_rounded_entries', 0))
            if generator_matches else 0
        )
        generator_rounding_max_abs = (
            float(generator_stats.get('qvalue_rounding_max_abs', 0.0))
            if generator_matches else 0.0
        )
        rounded_entries = generator_rounded_entries + result.qvalue_rounded_entries
        rounding_max_abs = (
            generator_rounding_max_abs + result.qvalue_rounding_max_abs
        )
        # The solver is exact on the canonical Q grid.  This is only the
        # worst-case difference between the caller's unrounded matrix and the
        # canonical assignment objective, not a solver optimality gap.
        input_rounding_bound = result.flow * rounding_max_abs
        grid_rounding_bound = result.flow * 0.5 / scale
        record = {
            'optimal': result.optimal,
            'status': result.status,
            'backend': result.backend,
            'fallback_used': result.fallback_used,
            'solver_fallback_used': result.solver_fallback_used,
            'objective_int': result.objective_int,
            'objective_q': result.objective_q,
            'objective_mode': result.objective_mode,
            'quantization_bound': input_rounding_bound,
            'input_rounding_bound': input_rounding_bound,
            'solver_objective_gap': 0.0,
            'qvalue_scale': result.qvalue_scale,
            'qvalue_entries': result.qvalue_entries,
            'qvalue_rounded_entries': rounded_entries,
            'qvalue_rounding_max_abs': rounding_max_abs,
            'qvalue_grid_rounding_bound': grid_rounding_bound,
            'qvalue_generator_rounded_entries': generator_rounded_entries,
            'qvalue_solver_rounded_entries': result.qvalue_rounded_entries,
            'flow': result.flow,
            'build_time': build_time,
            'solve_time': solve_time,
            'total_assignment_time': total_time,
            'original_edges': result.original_edges,
            'reduced_edges': result.reduced_edges,
            'edge_reduction_ratio': result.edge_reduction_ratio,
            'reduction_rounds': result.reduction_rounds,
        }
        self.env.mcmf_last_result = record
        if hasattr(self.env, 'record_time') and self.env.record_time:
            for key in (
                'mcmf_build', 'mcmf_solve', 'mcmf_total',
                'mcmf_original_edges', 'mcmf_reduced_edges',
            ):
                self.env.time_stats.setdefault(key, [])
            self.env.time_stats['mcmf_build'].append(build_time)
            self.env.time_stats['mcmf_solve'].append(solve_time)
            self.env.time_stats['mcmf_total'].append(total_time)
            self.env.time_stats['mcmf_original_edges'].append(result.original_edges)
            self.env.time_stats['mcmf_reduced_edges'].append(result.reduced_edges)
        if not getattr(self, '_reported_exact_mcmf_backend', False):
            print(
                f"✓ Exact MCMF backend={result.backend}, flow={result.flow}, "
                f"edges={result.original_edges}->{result.reduced_edges}, "
                f"objective={result.objective_mode}, "
                f"q_scale={result.qvalue_scale}, "
                f"rounded={rounded_entries}, "
                f"max_delta={rounding_max_abs:.6g}"
            )
            self._reported_exact_mcmf_backend = True

    def _record_online_mcmf_fallback(self, error):
        """Make it explicit that an online fallback has no optimality proof."""
        self.env.mcmf_last_result = {
            'optimal': False,
            'status': 'FALLBACK_LEGACY',
            'backend': 'legacy',
            'fallback_used': True,
            'solver_fallback_used': True,
            'objective_int': None,
            'objective_q': None,
            'objective_mode': None,
            'qvalue_scale': None,
            'qvalue_entries': None,
            'qvalue_rounded_entries': None,
            'qvalue_rounding_max_abs': None,
            'flow': None,
            'error': f'{type(error).__name__}: {error}',
        }

    def _exact_vehicle_rebalancing_network(
        self,
        vehicle_ids,
        available_requests,
        vehicle_action_matrix,
        batch_q_value,
        *,
        ev_only=False,
    ):
        if not vehicle_ids:
            return {}
        build_start = time.perf_counter()
        feasibility, q_values, capacities, fallback, layout = (
            self._build_exact_mcmf_inputs(
                vehicle_ids,
                available_requests,
                vehicle_action_matrix,
                batch_q_value,
                ev_only=ev_only,
            )
        )
        problem = build_reduced_problem(
            feasibility,
            q_values,
            capacities,
            cost_scale=int(getattr(
                self.env, 'mcmf_cost_scale', self.mcmf_cost_scale
            )),
            fallback_values=fallback,
            graph_reduction=bool(getattr(
                self.env, 'mcmf_graph_reduction', self.mcmf_graph_reduction
            )),
        )
        build_time = time.perf_counter() - build_start

        requested_backend = self._exact_mcmf_selection() or self.mcmf_backend
        solve_start = time.perf_counter()
        result = solve_exact(
            problem,
            backend=requested_backend,
            verify=bool(getattr(self.env, 'mcmf_verify', self.mcmf_verify)),
            gp=self.gp if self.available and not self._gurobi_runtime_failed else None,
            grb=self.GRB if self.available and not self._gurobi_runtime_failed else None,
            num_threads=self.num_threads,
        )
        solve_time = time.perf_counter() - solve_start
        self._record_exact_mcmf_result(
            result, build_time=build_time, solve_time=solve_time
        )
        return self._decode_exact_mcmf_assignments(
            result.action_by_vehicle,
            vehicle_ids,
            layout,
            ev_only=ev_only,
        )
    
    
    
    
    
    
    def optimize_vehicle_rebalancing(self, vehicle_ids):
        """Optimize vehicle rebalancing using Gurobi or heuristic methods"""
        if not self.available:
            return self._heuristic_rebalancing_assignment(vehicle_ids)
        
        try:
            return self._gurobi_vehicle_rebalancing(vehicle_ids)
        except Exception as e:
            print(f"Gurobi rebalancing failed: {e}, using heuristic")
            return self._heuristic_rebalancing_assignment(vehicle_ids)
        
        
        
    def optimize_vehicle_rebalancing_reject(self, vehicle_ids):
        """Optimize vehicle rebalancing using Gurobi or heuristic methods with reject consideration"""
        if not self.available:
            return self._heuristic_rebalancing_assignment(vehicle_ids)
        
        # Get available requests from environment
        available_requests = []
        if hasattr(self.env, 'active_requests') and self.env.active_requests:
            available_requests = list(self.env.active_requests.values())
        
        # Get available charging stations
        charging_stations = []
        if hasattr(self.env, 'charging_manager') and self.env.charging_manager.stations:
            charging_stations = [station for station in self.env.charging_manager.stations.values() 
                               if station.available_slots > 0]
        
        try:
            return self._gurobi_vehicle_rebalancing_knownreject(vehicle_ids, available_requests, charging_stations)
        except Exception as e:
            print(f"Gurobi rebalancing with reject failed: {e}, using heuristic")
            return self._heuristic_rebalancing_assignment(vehicle_ids)
        
    
    
    def optimize_vehicle_rebalancing_integrated(self, vehicle_ids):
        """Optimize vehicle rebalancing using Gurobi or heuristic methods with reject consideration"""
        if not self.available:
            return self._heuristic_rebalancing_assignment(vehicle_ids)
        
        assigned_requests = set()
        for vehicle in self.env.vehicles.values():
            if vehicle['assigned_request'] is not None:
                assigned_requests.add(vehicle['assigned_request'])
            if vehicle['passenger_onboard'] is not None:
                assigned_requests.add(vehicle['passenger_onboard'])
        # vehicle_id_ev = []
        # for vehicle_id in vehicle_ids:
        #     if self.env.vehicles[vehicle_id]['type'] == 1:
        #         vehicle_id_ev.append(vehicle_id)              
        # assigned_requests_ev = {}
        # for vehicle_id in vehicle_id_ev:
        #     vehicle = self.env.vehicles[vehicle_id]
        #     if self.env.vehicles[vehicle_id]['assigned_request'] is not None:
        #         assigned_requests_ev[self.env.vehicles[vehicle_id]['assigned_request']] = vehicle_id
        #     if self.env.vehicles[vehicle_id]['passenger_onboard'] is not None:
        #         assigned_requests_ev[self.env.vehicles[vehicle_id]['passenger_onboard']] = vehicle_id
        # for assigned_request in assigned_requests_ev.keys():
        #     if ifaccept (self.env.value_function_ev.update_accept_assignment(assigned_request)):
        #         continue
        #     else:
        #         assigned_requests_ev.pop(assigned_request)


        available_requests = list(self.env.active_requests.values())
        available_requests = [req for req in available_requests if req.request_id not in assigned_requests]
        # Get available charging stations
        charging_stations = []
        if hasattr(self.env, 'charging_manager') and self.env.charging_manager.stations:
            charging_stations = [station for station in self.env.charging_manager.stations.values() 
                               if station.available_slots > 0]
        
        try:
            return self._gurobi_vehicle_rebalancing_integrated(vehicle_ids, available_requests, charging_stations)
        except Exception as e:
            print(f"Gurobi rebalancing with reject failed: {e}, using heuristic")
            return self._heuristic_rebalancing_assignment(vehicle_ids)
        
        

    def _gurobi_vehicle_rebalancing(self, vehicle_ids):
        """Use Gurobi optimization to assign vehicles to available requests"""
        assignments = {}

        if not hasattr(self.env, 'active_requests') or not self.env.active_requests:
            return assignments

        # Convert active requests to list
        available_requests = list(self.env.active_requests.values())
        request_count = len(available_requests)
        # Get available charging stations
        charging_stations = []
        if hasattr(self.env, 'charging_manager') and self.env.charging_manager.stations:
            charging_stations = [station for station in self.env.charging_manager.stations.values() 
                               if station.available_slots > 0]
        
        # Return empty if no requests and no need for charging rebalancing
        if not available_requests and not charging_stations:
            return assignments

        # Create Gurobi model for vehicle-to-request assignment
        model = self.gp.Model("vehicle_rebalancing")
        model.setParam('OutputFlag', 0)  # Suppress output
        model.setParam('TimeLimit', 30)  # Set time limit
        model.setParam('Threads', self.num_threads)  # Set thread count

        # Decision variables: x[i,j] = 1 if vehicle i is assigned to request j
        request_decision = {}

        charge_decision = {}
        idle_vehicle = {}
        waiting_vehicle = {}
        for i, vehicle_id in enumerate(vehicle_ids):
            for j, request in enumerate(available_requests):
                request_decision[i, j] = model.addVar(vtype=self.GRB.BINARY,
                                     name=f'vehicle_{vehicle_id}_request_{request.request_id}')

        for i, vehicle_id in enumerate(vehicle_ids):
            for j, station in enumerate(charging_stations):
                charge_decision[i, j] = model.addVar(vtype=self.GRB.BINARY,
                                     name=f'vehicle_{vehicle_id}_charge_{station.id}')
        
        # Constraint 1: Each vehicle can be assigned to at most one request

        for i in range(len(vehicle_ids)):
            idle_vehicle[i] = model.addVar(vtype=self.GRB.BINARY,
                                     name=f'vehicle_{vehicle_ids[i]}_idle')
        for i in range(len(vehicle_ids)):
            waiting_vehicle[i] = model.addVar(vtype=self.GRB.BINARY,
                                     name=f'vehicle_{vehicle_ids[i]}_waiting')
        for i in range(len(vehicle_ids)):
            actionv = self.gp.LinExpr()
            for j in range(len(available_requests)):
                actionv += request_decision[i, j]
            for j in range(len(charging_stations)):
                actionv += charge_decision[i, j]
            model.addConstr(actionv <= 1)
            model.addConstr(idle_vehicle[i] + actionv + waiting_vehicle[i] == 1) 
        idlevehicle = self.gp.LinExpr()
        for i in range(len(vehicle_ids)):
            idlevehicle += idle_vehicle[i]
        model.addConstr(idlevehicle >= getattr(self.env, 'idle_vehicle_requirement', 0)) 
        servedrequest = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                servedrequest += request_decision[i, j]     
        # Constraint 2: Each request can be assigned to at most one vehicle
        for j in range(len(available_requests)):
            model.addConstr(self.gp.quicksum(request_decision[i, j] for i in range(len(vehicle_ids))) <= 1)

        # Objective: Maximize blended objective
        # - If adp_weight == 0: use immediate rewards (movement cost + request value / charging penalty)
        # - Else: use option-completion Q-values scaled by adp_weight (to avoid double-counting immediate rewards)
        objective_terms  = self.gp.LinExpr()
        adp_weight = getattr(self.env, 'adp_value', 1.0)
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]

            # Process request assignments using option value
            for j, request in enumerate(available_requests):
                if adp_weight <= 0:
                    # Immediate reward fallback
                    req_val = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    cur_loc = vehicle['location']
                    d1 = self._manhattan_loc(cur_loc, request.pickup)
                    d2 = self._request_trip_distance(request)
                    moving_cost = self._movement_cost(d1 + d2)
                    immediate = req_val + moving_cost
                    objective_terms += immediate * request_decision[i, j]
                else:
                    option_q = 0.0
                    if hasattr(self.env, 'evaluate_service_option'):
                        try:
                            option_q = self.env.evaluate_service_option(vehicle_id, request)
                        except Exception:
                            option_q = 0.0
                    objective_terms += option_q * adp_weight * request_decision[i, j]
            
            # Process charging assignments using option value
            for j, station in enumerate(charging_stations):
                if adp_weight <= 0:
                    # Immediate charging cost fallback
                    cur_loc = vehicle['location']
                    d_travel = self._manhattan_loc(cur_loc, station.location)
                    moving_cost = self._movement_cost(d_travel)
                    charge_steps = getattr(self.env, 'charge_duration', 2)
                    charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * charge_steps
                    immediate = moving_cost + charging_penalty
                    objective_terms += immediate * charge_decision[i, j]
                else:
                    charging_q = 0.0
                    if hasattr(self.env, 'evaluate_charging_option'):
                        try:
                            charging_q = self.env.evaluate_charging_option(vehicle_id, station)
                        except Exception:
                            charging_q = 0.0
                    objective_terms += charging_q * adp_weight * charge_decision[i, j]
        objective_terms -= getattr(self.env, 'unserved_penalty', 1.5) * (request_count - servedrequest)
        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)

        # Solve the optimization problem
        model.optimize()

        # Extract assignments
        if model.status == self.GRB.OPTIMAL:
            for i, vehicle_id in enumerate(vehicle_ids):
                # Check request assignments
                for j, request in enumerate(available_requests):
                    if request_decision[i, j].x > 0.5:  # Binary variable threshold
                        assignments[vehicle_id] = request
                        break
                
                # Check charging assignments if no request assigned
                if vehicle_id not in assignments:
                    for j, station in enumerate(charging_stations):
                        if charge_decision[i, j].x > 0.5:  # Binary variable threshold
                            assignments[vehicle_id] = f"charge_{station.id}"
                            break
                
        return assignments
    
    
    
    
    def optimize_vehicle_assignment(self, requests, vehicles):
        """Optimize assignment of vehicles to requests using Gurobi"""
        if not self.available or not requests:
            return self._heuristic_order_assignment(requests, vehicles)
        
        try:
            # Create optimization model
            model = self.gp.Model("vehicle_assignment")
            model.setParam('OutputFlag', 0)  # Suppress output
            model.setParam('Threads', self.num_threads)  # Set thread count
            
            # Decision variables: x[i,j] = 1 if vehicle i is assigned to request j
            x = {}
            for i, vehicle_id in enumerate(vehicles):
                for j, request in enumerate(requests):
                    x[i, j] = model.addVar(vtype=self.GRB.BINARY, 
                                         name=f'assign_{vehicle_id}_{request.request_id}')
            
            # Constraints: Each request can be assigned to at most one vehicle
            for j in range(len(requests)):
                model.addConstr(self.gp.quicksum(x[i, j] for i in range(len(vehicles))) <= 1)
            
            # Constraints: Each vehicle can be assigned to at most one request
            for i in range(len(vehicles)):
                model.addConstr(self.gp.quicksum(x[i, j] for j in range(len(requests))) <= 1)
            
            # Objective: Minimize total distance + maximize total value
            obj = self.gp.quicksum(
                x[i, j] * (requests[j].value - self._calculate_distance(vehicles[i], requests[j]))
                for i in range(len(vehicles))
                for j in range(len(requests))
            )
            model.setObjective(obj, self.GRB.MAXIMIZE)
            
            # Solve
            model.optimize()
            
            # Extract solution
            assignments = {}
            if model.status == self.GRB.OPTIMAL:
                for i, vehicle_id in enumerate(vehicles):
                    for j, request in enumerate(requests):
                        if x[i, j].x > 0.5:  # Binary variable is 1
                            assignments[vehicle_id] = request.request_id
            
            return assignments
            
        except Exception as e:
            print(f"Gurobi optimization failed: {e}, using heuristic")
            return self._heuristic_order_assignment(requests, vehicles)
    
    
    
    
    def _calculate_distance(self, vehicle_id, request):
        """Calculate distance from vehicle to request pickup"""
        vehicle = self.env.vehicles[vehicle_id]
        vehicle_coords = vehicle['coordinates']
        pickup_coords = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)
        return abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
    
    def _manhattan_loc(self, a_loc: int, b_loc: int) -> float:
        """Return NYC kilometres, or synthetic-grid Manhattan distance."""
        if hasattr(self.env, 'get_distance_km'):
            return float(self.env.get_distance_km(int(a_loc), int(b_loc)))
        gx = self.env.grid_size
        ax, ay = a_loc % gx, a_loc // gx
        bx, by = b_loc % gx, b_loc // gx
        return float(abs(ax - bx) + abs(ay - by))

    def _request_trip_distance(self, request) -> float:
        """Use a request's TLC route distance when one is available."""
        if hasattr(self.env, '_request_trip_distance_km'):
            return float(self.env._request_trip_distance_km(request))
        route_distance = getattr(request, 'trip_distance_km', None)
        if route_distance is not None:
            try:
                route_distance = float(route_distance)
                if math.isfinite(route_distance) and route_distance >= 0.0:
                    return route_distance
            except (TypeError, ValueError):
                pass
        return self._manhattan_loc(request.pickup, request.dropoff)

    
    
    
    def _heuristic_rebalancing_assignment(self, vehicle_ids):
        """Advanced heuristic assignment for vehicle rebalancing when Gurobi is not available"""
        assignments = {}
        
        if not hasattr(self.env, 'active_requests') or not self.env.active_requests:
            return assignments

        available_requests = list(self.env.active_requests.values())
        if not available_requests:
            return assignments

        available_vehicles = set(vehicle_ids)

        # Strategy 1: Prioritize high-value requests with deadline urgency
        # Calculate request priorities based on value, urgency, and vehicle compatibility
        request_priorities = []
        for request in available_requests:
            # Calculate urgency factor (requests closer to deadline are more urgent)
            time_remaining = max(1, request.pickup_deadline - self.env.current_time)
            urgency_factor = 1.0 / time_remaining
            
            # Combined priority: value + urgency
            priority = request.value * 0.7 + urgency_factor * 0.3
            request_priorities.append((priority, request))

        # Sort requests by priority (highest first)
        request_priorities.sort(key=lambda x: x[0], reverse=True)

        # Strategy 2: Match vehicles optimally considering multiple factors
        for priority, request in request_priorities:
            if not available_vehicles:
                break

            best_vehicle = None
            best_score = float('-inf')

            pickup_pos = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)

            for vehicle_id in available_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_pos = vehicle['coordinates']
                
                # Calculate distance
                distance = abs(vehicle_pos[0] - pickup_pos[0]) + abs(vehicle_pos[1] - pickup_pos[1])
                
                # Distance score (closer is better)
                distance_score = 1.0 / (1.0 + distance)
                
                # Battery level score (higher battery is better for service)
                battery_score = vehicle['battery']
                
                # Vehicle type compatibility score
                type_score = 1.0
                if vehicle['type'] == 2:
                    type_score = 1.2  # AEV vehicles are preferred for service
                
                # Combined score: distance + battery + type
                total_score = (distance_score * 0.5 + 
                             battery_score * 0.3 + 
                             type_score * 0.2)
                
                if total_score > best_score:
                    best_score = total_score
                    best_vehicle = vehicle_id

            if best_vehicle:
                assignments[best_vehicle] = request
                available_vehicles.remove(best_vehicle)

        return assignments

    def _heuristic_order_assignment(self, requests, vehicles):
        """Enhanced heuristic assignment for order processing"""
        assignments = {}
        available_vehicles = set(vehicles)
        
        # Strategy: Multi-criteria optimization for order assignment
        # Calculate vehicle capabilities and request requirements
        vehicle_scores = {}
        for vehicle_id in vehicles:
            vehicle = self.env.vehicles[vehicle_id]
            # Base score considers battery level and vehicle type
            base_score = vehicle['battery'] * 0.6
            if vehicle['type'] == 2:
                base_score += 0.3  # AEV bonus for reliability
            vehicle_scores[vehicle_id] = base_score

        # Sort requests by combined value and urgency
        enhanced_requests = []
        for request in requests:
            # Calculate time pressure
            if hasattr(request, 'pickup_deadline'):
                time_remaining = max(1, request.pickup_deadline - self.env.current_time)
                urgency = 1.0 / time_remaining
            else:
                urgency = 0.5  # Default urgency
            
            # Combined priority
            priority = request.value * 0.8 + urgency * 0.2
            enhanced_requests.append((priority, request))

        enhanced_requests.sort(key=lambda x: x[0], reverse=True)

        # Assign vehicles to requests using enhanced scoring
        for priority, request in enhanced_requests:
            if not available_vehicles:
                break

            best_vehicle = None
            best_combined_score = float('-inf')

            pickup_coords = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)

            for vehicle_id in available_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                
                # Distance factor
                distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                distance_factor = 1.0 / (1.0 + distance * 0.1)
                
                # Combine vehicle capability with distance efficiency
                combined_score = vehicle_scores[vehicle_id] * 0.6 + distance_factor * 0.4
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_vehicle = vehicle_id

            if best_vehicle:
                assignments[best_vehicle] = request.request_id
                available_vehicles.remove(best_vehicle)

        return assignments

    def _heuristic_charge_assignment(self, requests, vehicles):
        """Enhanced heuristic assignment for charging optimization"""
        assignments = {}
        
        # Get vehicle data and sort by charging priority
        vehicle_data = {}
        for vehicle_id in vehicles:
            vehicle_data[vehicle_id] = self.env.vehicles[vehicle_id]
        
        # Strategy: Prioritize vehicles by charging need and strategic positioning
        charging_priorities = []
        for vehicle_id, vehicle in vehicle_data.items():
            # Calculate charging urgency (lower battery = higher urgency)
            battery_urgency = 1.0 - vehicle['battery']
            
            # Calculate strategic value (position relative to high-demand areas)
            strategic_value = 0.5  # Base strategic value
            
            # If we have active requests, consider proximity to demand
            if hasattr(self.env, 'active_requests') and self.env.active_requests:
                total_distance_to_requests = 0
                num_requests = len(self.env.active_requests)
                
                for request in self.env.active_requests.values():
                    pickup_pos = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)
                    vehicle_pos = vehicle['coordinates']
                    distance = abs(vehicle_pos[0] - pickup_pos[0]) + abs(vehicle_pos[1] - pickup_pos[1])
                    total_distance_to_requests += distance
                
                # Lower average distance = higher strategic value
                avg_distance = total_distance_to_requests / num_requests if num_requests > 0 else 10
                strategic_value = 1.0 / (1.0 + avg_distance * 0.1)
            
            # Combined priority: urgency + strategic positioning
            priority = battery_urgency * 0.7 + strategic_value * 0.3
            charging_priorities.append((priority, vehicle_id))
        
        # Sort by priority (highest first)
        charging_priorities.sort(key=lambda x: x[0], reverse=True)
        
        available_vehicles = set(vehicles)
        
        # Sort requests by value and accessibility
        enhanced_requests = []
        for request in requests:
            # For charging requests, prioritize based on value and station availability
            station_accessibility = 1.0  # Default accessibility
            
            # If request is related to charging stations, consider station load
            if hasattr(self.env, 'charging_manager'):
                # Calculate average distance to available charging stations
                available_stations = [s for s in self.env.charging_manager.stations.values() 
                                    if s.current_capacity < s.max_capacity]
                if available_stations:
                    min_station_distance = float('inf')
                    request_pos = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)
                    
                    for station in available_stations:
                        station_distance = abs(station.location[0] - request_pos[0]) + abs(station.location[1] - request_pos[1])
                        min_station_distance = min(min_station_distance, station_distance)
                    
                    station_accessibility = 1.0 / (1.0 + min_station_distance * 0.1)
            
            enhanced_value = request.value * station_accessibility
            enhanced_requests.append((enhanced_value, request))
        
        enhanced_requests.sort(key=lambda x: x[0], reverse=True)
        
        # Assign prioritized vehicles to enhanced requests
        for enhanced_value, request in enhanced_requests:
            if not available_vehicles:
                break
            
            # Find the best vehicle from our priority list
            best_vehicle = None
            
            for priority, vehicle_id in charging_priorities:
                if vehicle_id in available_vehicles:
                    # Check if this vehicle can handle the request efficiently
                    vehicle = vehicle_data[vehicle_id]
                    distance = self._calculate_distance(vehicle_id, request)
                    
                    # Only assign if vehicle has sufficient battery or is close
                    if vehicle['battery'] > 0.2 or distance <= 3:
                        best_vehicle = vehicle_id
                        break
            
            if best_vehicle:
                assignments[best_vehicle] = request.request_id
                available_vehicles.remove(best_vehicle)
                # Remove assigned vehicle from priority list
                charging_priorities = [(p, v) for p, v in charging_priorities if v != best_vehicle]
        
        return assignments
    
    
    
    
    def _gurobi_vehicle_rebalancing_ev(self, vehicle_ids, available_requests, charging_stations=None):
  
  
        if not self.available:
            return {}
        
        assignments = {}
        
        # Create optimization model
        model = self.gp.Model("vehicle_assignment_with_reject_and_charging")
        model.setParam('OutputFlag', 0)  # Suppress output
        model.setParam('TimeLimit', self.network_time_limit)
        model.setParam('Threads', self.num_threads)

        # Aggregate stats for opportunity costs (optional)
        active_requests_count = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
        active_requests_value = sum(getattr(req, 'final_value', getattr(req, 'value', 0.0)) for req in (self.env.active_requests.values() if hasattr(self.env, 'active_requests') else []))
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 0.0

        # Parameters
        min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

        battery_consum = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05 # Battery consumption per travel step
        service_consumption = 0.05 # Battery consumption per service
        request_decision =[[model.addVar(vtype=self.GRB.BINARY,
                     name=f'request_{vehicle_id}_{request.request_id}') for request in available_requests] for i, vehicle_id in enumerate(vehicle_ids)]
            
        # Battery level variables (t-1 and t)
        battery_t_minus_1 = {}  # Battery level at t-1 (current)
        battery_t = {}          # Battery level at t (after actions)
        idle_vehicle = {}
        for i in range(len(vehicle_ids)):
            idle_vehicle[i] = model.addVar(vtype=self.GRB.BINARY,
                                     name=f'vehicle_{vehicle_ids[i]}_idle')
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # t-1 battery level (current battery level)
            battery_t_minus_1[i] = vehicle['battery']
            
            # t battery level (decision variable)
            battery_t[i] = model.addVar(
                vtype=self.GRB.CONTINUOUS,

                name=f'battery_t_{vehicle_id}'
            )
        

        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Initialize battery expressions as Gurobi LinExpr
            battery_loss = self.gp.LinExpr()

            # Battery consumption from service requests (travel to pickup + pickup to dropoff)
            if available_requests:
                for j, request in enumerate(available_requests):
                    # Travel from vehicle current position to pickup
                    pickup_x = request.pickup % self.env.grid_size
                    pickup_y = request.pickup // self.env.grid_size
                    travel_distance_to_pickup = abs(vehicle['coordinates'][0] - pickup_x) + abs(vehicle['coordinates'][1] - pickup_y)
                    
                    # Travel from pickup to dropoff
                    dropoff_x = request.dropoff % self.env.grid_size
                    dropoff_y = request.dropoff // self.env.grid_size
                    travel_distance_pickup_to_dropoff = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
                    
                    # Total battery consumption for this request
                    total_travel_distance = travel_distance_to_pickup + travel_distance_pickup_to_dropoff
                    battery_loss += total_travel_distance * battery_consum * request_decision[i][j]

            # Battery transition constraint (simplified to avoid infeasibility)
            model.addConstr(battery_t[i] == battery_t_minus_1[i] - battery_loss)
            # Ensure vehicle has enough battery for actions (but allow some flexibility)
            model.addConstr(battery_loss <= battery_t_minus_1[i] )  # Allow small battery deficit to avoid infeasibility
            # Ensure battery doesn't go below minimum (but allow some flexibility)
            model.addConstr(battery_t[i] >=min_battery_level)  # If not idle, must meet min battery


            
            # Constraint 1: Each vehicle can only take one action
        for i in range(len(vehicle_ids)):
            actionv = self.gp.LinExpr()
            # Add valid request assignments
            for j in range(len(available_requests)):
                actionv += request_decision[i][j]
            model.addConstr(actionv + idle_vehicle[i] == 1)
        

        
        # Constraint 2: Each request can be assigned to at most one vehicle
        for j in range(len(available_requests)):
            valid_vehicles = self.gp.LinExpr()
            for i in range(len(vehicle_ids)):
                valid_vehicles += request_decision[i][j]
            model.addConstr(valid_vehicles <= 1)
            
        objective_terms = self.gp.LinExpr()
        adp_weight = getattr(self.env, 'adp_value', 1.0)
        
        # 批量预计算所有vehicle-request对的Q值以提高性能
        option_q_cache = {}
        rejection_adjusted_values = {}  # 存储拒绝感知调整后的价值
        
        if adp_weight > 0:
            # 收集所有需要计算的vehicle-request对
            vehicle_request_pairs = []
            for i, vehicle_id in enumerate(vehicle_ids):
                for j, request in enumerate(available_requests):
                    vehicle_request_pairs.append((vehicle_id, request))
            
            vehicle_request_pairs_ev = [(vid, req) for vid, req in vehicle_request_pairs if self.env.vehicles[vid]['type'] == 1]
            if hasattr(self.env, 'batch_evaluate_service_options'):
                try:
                    
                    batch_q_values_ev = self.env.batch_evaluate_service_options(vehicle_request_pairs_ev,True)
                    
                    # 批量计算拒绝概率（只对EV）
                    batch_rejection_probs = self._batch_calculate_reject_pro_network(vehicle_request_pairs_ev)
                    
                    for i, (vehicle_id, request) in enumerate(vehicle_request_pairs_ev):
                        q_value = batch_q_values_ev[i] if i < len(batch_q_values_ev) else 0.0
                        rejection_prob = batch_rejection_probs[i] if i < len(batch_rejection_probs) else 0.0
                        
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        
                        # 计算拒绝感知调整价值
                        adjusted_value = self._calculate_rejection_aware_value(
                            vehicle_id, request, q_value, rejection_prob
                        )
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = adjusted_value
                    for i, (vehicle_id, request) in enumerate(vehicle_request_pairs_ev):
                        q_value = batch_q_values_ev[i] if i < len(batch_q_values_ev) else 0.0
                        rejection_prob = 0.0
                        
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        

                except Exception as e:
                    print(f"Batch evaluation failed: {e}, falling back to individual calculations")
            
            
            # 如果批量计算失败，使用单独计算
            if not option_q_cache:
                # 批量计算拒绝概率（只对EV）
                batch_rejection_probs = self._batch_calculate_reject_pro_network(vehicle_request_pairs)
                
                for i, (vehicle_id, request) in enumerate(vehicle_request_pairs):
                    try:
                        q_value = self.env.evaluate_service_option(vehicle_id, request, True)
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        
                        # 使用批量计算的拒绝概率
                        rejection_prob = batch_rejection_probs[i] if i < len(batch_rejection_probs) else 0.0
                        
                        adjusted_value = self._calculate_rejection_aware_value(
                            vehicle_id, request, q_value, rejection_prob
                        )
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = adjusted_value
                    except Exception:
                        option_q_cache[(vehicle_id, request.request_id)] = 0.0
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = 0.0
            
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]

            for j, request in enumerate(available_requests):
                if adp_weight <= 0:
                    # 回退到基础计算
                    req_val = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    cur_loc = vehicle['location']
                    d1 = self._manhattan_loc(cur_loc, request.pickup)
                    d2 = self._request_trip_distance(request)
                    moving_cost = self._movement_cost(d1 + d2)
                    immediate = req_val + moving_cost
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    objective_terms += immediate* request_decision[i][j]*(1 - rejection_prob)
                else:
                    # 使用批量计算的Q值和拒绝感知的调整价值
                    base_q_value = option_q_cache.get((vehicle_id, request.request_id), 0.0)
                    #adjusted_value = rejection_adjusted_values.get((vehicle_id, request.request_id), base_q_value)
                    objective_terms += base_q_value * adp_weight * request_decision[i][j]
                


        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)
        try:
            model.optimize()
            
            # Extract assignments
            if model.status == self.GRB.OPTIMAL:
                # Print battery level optimization results for debugging

                
                for i, vehicle_id in enumerate(vehicle_ids):
                    # Check request assignments
                    for j, request in enumerate(available_requests):
                        if request_decision[i][j].x > 0.5:
                            assignments[vehicle_id] = request
                            break
                    if idle_vehicle[i].x > 0.5:
                        assignments[vehicle_id] = "idle"
                    
                # Update vehicle battery levels based on optimization results
                for i, vehicle_id in enumerate(vehicle_ids):
                    if hasattr(self.env.vehicles[vehicle_id], 'predicted_battery_t'):
                        self.env.vehicles[vehicle_id]['predicted_battery_t'] = battery_t[i].x
                        
            else:
                print(f"Optimization status: {model.status}")
                for i, vehicle_id in enumerate(vehicle_ids):
                    assignments[vehicle_id] = f"waiting"
                if model.status == self.GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS...")
                    model.computeIIS()
                    print("Infeasible constraints:")
                    print("ev infeasible constraints:")
                    for c in model.getConstrs():
                        if c.IISConstr:
                            print(f"  {c.constrName}")
                            
            selected_request = []
            for vehicle_id, assignment in assignments.items():
                if isinstance(assignment, str) and assignment.startswith("charge_"):
                    continue
                elif isinstance(assignment, str) and assignment in ["waiting", "idle"]:
                    continue
                else:
                    selected_request.append(assignment.request_id)
            remaining_requests = [req.request_id for req in available_requests if req.request_id not in selected_request]
        except Exception as e:
            print(f"Gurobi optimization with reject and charging levels failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to heuristic with reject consideration
            assignments = self._heuristic_assignment_with_reject(vehicle_ids, available_requests, charging_stations)
        
        return assignments,remaining_requests
    
    
    def _networkxsolve_vehicle_rebalancing(self, vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value):
        """
        NetworkX-based min-cost max-flow optimization using batch_q_value for costs
        batch_q_value shape: [num_vehicles, num_actions]
        where num_actions = num_requests + num_charging_stations + num_zones + 1 (wait)
        """
        import time
        starttime = time.time()
        G = nx.DiGraph()

        num_vehicles = len(vehicle_ids)
        num_requests = len(available_requests)
        num_charging = self.env.num_stations
        num_zones = self._get_relocation_target_count()
        
        layers = {
            'source': ['s'],
            'vehicles': [f'v{i}' for i in range(num_vehicles)],
            'requests': [f'r{j}' for j in range(num_requests)],
            'charging': [f'c{k}' for k in range(num_charging)],
            'zones': [f'z{m}' for m in range(num_zones)],
            'wait': ['w'],
            'sink': ['t']
        }
        #print("node number:", sum(len(v) for v in layers.values()))
        
        # Add all nodes with demands
        G.add_node('s', demand=-num_vehicles)  # Source supplies all vehicles
        for v in layers['vehicles']:
            G.add_node(v, demand=0)
        for r in layers['requests']:
            G.add_node(r, demand=0)
        for c in layers['charging']:
            G.add_node(c, demand=0)
        for z in layers['zones']:
            G.add_node(z, demand=0)
        G.add_node('w', demand=0)
        G.add_node('t', demand=num_vehicles)  # Sink demands all vehicles

        # Add edges: source → vehicles
        for v in layers['vehicles']:
            G.add_edge('s', v, capacity=1, weight=0)

        # Vehicles → requests (use -Q_value as cost for minimization, skip if Q-value is 0)
        for i in range(num_vehicles):
            for j in range(num_requests):
                if vehicle_action_matrix[i, j] == 0:
                    continue  # Skip edges with zero Q-value
                q_value = float(batch_q_value[i, j])
                cost = -q_value  # Minimize negative Q-value = Maximize Q-value
                G.add_edge(f'v{i}', f'r{j}', capacity=1, weight=cost)
        
        # Vehicles → charging stations (use -Q_value as cost, skip if Q-value is 0)
        for i in range(num_vehicles):
            for k in range(num_charging):
                col_idx = num_requests + k
                if vehicle_action_matrix[i, col_idx] == 0 or self.env.vehicles[vehicle_ids[i]]['type'] == 1:
                    continue  # Skip edges with zero Q-value
                q_value = float(batch_q_value[i, col_idx])
                cost = -q_value
                G.add_edge(f'v{i}', f'c{k}', capacity=1, weight=cost)
        
        # Vehicles → zone relocation (use -Q_value as cost, skip if Q-value is 0)
        for i in range(num_vehicles):
            for m in range(num_zones):
                col_idx = num_requests + num_charging + m
                if vehicle_action_matrix[i, col_idx] == 0 or self.env.vehicles[vehicle_ids[i]]['type'] == 1:
                    continue  # Skip edges with zero Q-value
                q_value = float(batch_q_value[i, col_idx])
                cost = -q_value
                G.add_edge(f'v{i}', f'z{m}', capacity=1, weight=cost)
        
        # Vehicles → wait (use -Q_value as cost, skip if Q-value is 0)
        for i in range(num_vehicles):
            col_idx = num_requests + num_charging + num_zones
            if vehicle_action_matrix[i, col_idx] == 0:
                continue  # Skip edges with zero Q-value
            q_value = float(batch_q_value[i, col_idx])
            cost = -q_value
            G.add_edge(f'v{i}', 'w', capacity=1, weight=cost)
        
        # Requests → sink (each request can be assigned to at most one vehicle)
        for j in range(num_requests):
            if vehicle_action_matrix[:, j].sum() > 0:
                G.add_edge(f'r{j}', 't', capacity=1, weight=0)
        
        # Charging stations → sink (capacity = vacancy)
        charging_stations_list = list(self.env.charging_manager.stations.values()) if hasattr(self.env, 'charging_manager') else []
        for k in range(num_charging):
            if k < len(charging_stations_list):
                station = charging_stations_list[k]
                vacancy = station.max_capacity - len(station.current_vehicles)
            else:
                vacancy = 1  # Fallback
            # 只为有入边的充电站节点添加到sink的边
            if vehicle_action_matrix[:, num_requests + k].sum() > 0:
                G.add_edge(f'c{k}', 't', capacity=vacancy, weight=0)
        
        # Zones → sink (只为有入边的zone添加)
        for m in range(num_zones):
            if vehicle_action_matrix[:, num_requests + num_charging + m].sum() > 0:
                G.add_edge(f'z{m}', 't', capacity=num_vehicles, weight=0)
        
        # Wait → sink (只有当wait节点有入边时才添加)
        if vehicle_action_matrix[:, num_requests + num_charging + num_zones].sum() > 0:
            G.add_edge('w', 't', capacity=num_vehicles, weight=0)
        
        # 验证图的可行性
        total_supply = sum(data.get('demand', 0) for node, data in G.nodes(data=True) if data.get('demand', 0) < 0)
        total_demand = sum(data.get('demand', 0) for node, data in G.nodes(data=True) if data.get('demand', 0) > 0)
        
        # 打印图的基本信息
        # print(f"📊 Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
        # print(f"   Supply: {-total_supply}, Demand: {total_demand}", flush=True)
        

        
        # 正确检查：supply 是负数，demand 是正数，绝对值应该相等
        if abs(total_supply) != total_demand:
            print(f"❌ ERROR: Supply-demand imbalance! Supply={-total_supply}, Demand={total_demand}")
            print(f"  This will cause nx.min_cost_flow() to hang!")
            # 返回空分配
            return {vehicle_id: f"waiting" for vehicle_id in vehicle_ids}
        
        # 检查是否所有车辆都有至少一条出边
        vehicles_without_edges = []
        for i in range(num_vehicles):
            vehicle_node = f'v{i}'
            out_edges = list(G.successors(vehicle_node))
            if len(out_edges) == 0 or (len(out_edges) == 1 and out_edges[0] == 's'):
                vehicles_without_edges.append(i)
        for vehicle_idx in vehicles_without_edges:
            vehicle_id = vehicle_ids[vehicle_idx]
            battery = self.env.vehicles[vehicle_id]['battery']
            print(f"   Vehicle {vehicle_id} has no valid outgoing edges! Battery level: {battery}", flush=True)
        if vehicles_without_edges:
            print(f"❌ ERROR: {len(vehicles_without_edges)} vehicles have no valid actions!")
            print(f"  Vehicles: {vehicles_without_edges[:10]}...")
            print(f"  This will cause infeasibility!")
            # 为这些车辆添加到 wait 节点的边
            for i in vehicles_without_edges:
                G.add_edge(f'v{i}', 'w', capacity=1, weight=50.0)  # 高成本的等待
            print(f"  Added emergency wait edges for {len(vehicles_without_edges)} vehicles")
        
        # Compute min cost flow
        # print(f"🔵 Starting nx.min_cost_flow() with {num_vehicles} vehicles, {G.number_of_edges()} edges...", flush=True)
        # print(f"   (Requests: {num_requests}, Charging: {num_charging}, Zones: {num_zones})", flush=True)
        
        import sys
        sys.stdout.flush()
        t_flow_start = time.time()
        
        try:
            # 使用线程+超时机制（Windows兼容）
            import threading
            flow_result = [None]
            error_result = [None]
            
            def compute_flow():
                try:
                    # 尝试不同的算法
                    # 1. 先尝试 capacity_scaling（通常最快）
                    try:
                        from networkx.algorithms.flow import capacity_scaling
                        flow_cost, flow_result_tmp = capacity_scaling(G)
                        flow_result[0] = flow_result_tmp
                        # print(f"  (使用 capacity_scaling 算法)", flush=True)
                    except Exception as e1:
                        # 2. 回退到 shortest_augmenting_path
                        try:
                            from networkx.algorithms.flow import shortest_augmenting_path
                            flow_result[0] = shortest_augmenting_path(G)
                            print(f"  (使用 shortest_augmenting_path 算法)", flush=True)
                        except Exception as e2:
                    #         # 3. 最后使用默认的 network_simplex
                            flow_cost, flow_dict_tmp = nx.network_simplex(G)
                            flow_result[0] = flow_dict_tmp
                    # print(f"  (使用 network_simplex 默认算法)", flush=True)
                except Exception as e:
                    error_result[0] = e
            
            thread = threading.Thread(target=compute_flow)
            thread.daemon = True
            thread.start()
            thread.join(timeout=10.0)  # 10秒超时
            
            if thread.is_alive():
                print(f"❌ NetworkX min_cost_flow() 超时（10秒）！", flush=True)
                print(f"   建议使用 Gurobi（设置 usenetworkx=False）", flush=True)
                # 返回等待动作
                return {vehicle_id: f"waiting" for vehicle_id in vehicle_ids}
            
            if error_result[0]:
                raise error_result[0]
            
            flow_dict = flow_result[0]
            t_flow_end = time.time()
            elapsed = t_flow_end - t_flow_start
            #print(f"✅ nx.min_cost_flow() 完成，耗时 {elapsed:.4f} 秒", flush=True)
            
            if elapsed > 5.0:
                print(f"⚠️  NetworkX 较慢（{elapsed:.2f}秒），建议使用 Gurobi", flush=True)
        except nx.NetworkXUnfeasible:
            print("❌ ERROR: Flow problem is infeasible!")
            return {vehicle_id: f"waiting" for vehicle_id in vehicle_ids}
        except Exception as e:
            print(f"❌ ERROR in nx.min_cost_flow(): {e}")
            return {vehicle_id: f"waiting" for vehicle_id in vehicle_ids}
        
        total_cost = nx.cost_of_flow(G, flow_dict)
        
        # Extract assignments from flow
        assignments = {}
        unassigned_vehicles = []
        
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle_node = f'v{i}'
            assigned = False
            
            if vehicle_node in flow_dict:
                for target_node, flow in flow_dict[vehicle_node].items():
                    if flow > 0 and target_node != 's':  # Flow exists and not back to source
                        # Decode target node
                        if target_node.startswith('r'):
                            # Request assignment
                            request_idx = int(target_node[1:])
                            request = available_requests[request_idx]
                            assignments[vehicle_id] = request
                            assigned = True
                        elif target_node.startswith('c'):
                            # Charging station assignment
                            charging_idx = int(target_node[1:])
                            if charging_idx < len(charging_stations_list):
                                station = charging_stations_list[charging_idx]
                            else:
                                station = None
                            if station:
                                assignments[vehicle_id] = f"charge_{station.id}"
                            else:
                                if self.env.vehicles[vehicle_id]['type'] == 1:
                                    # EV without valid station, force wait
                                    assignments[vehicle_id] = f"reloc"
                                else:    
                                    assignments[vehicle_id] = f"waiting"
                            assigned = True
                        elif target_node.startswith('z'):
                            if self.env.vehicles[vehicle_id]['type'] == 1:
                                # EV cannot relocate to zones, force wait
                                assignments[vehicle_id] = f"reloc"
                            else:
                                # Zone relocation assignment
                                zone_idx = int(target_node[1:])
                                assignments[vehicle_id] = f"idle_at_{zone_idx}"
                            assigned = True
                        elif target_node == 'w':
                            if self.env.vehicles[vehicle_id]['type'] == 1:
                                # EV cannot wait, force relocate
                                assignments[vehicle_id] = f"reloc"
                            else:
                                assignments[vehicle_id] = f"waiting"
                            assigned = True
                        
                        if assigned:
                            break  # Found the assignment, move to next vehicle
            
            if not assigned:
                unassigned_vehicles.append((i, vehicle_id))
                if self.env.vehicles[vehicle_id]['type'] == 1:
                    assignments[vehicle_id] = f"reloc"  # EV强制重新定位
                else:
                    assignments[vehicle_id] = f"waiting"  # 默认等待
        
        if unassigned_vehicles:
            print(f"⚠️  Warning: {len(unassigned_vehicles)} vehicles have no assignment in flow solution!")
            print(f"   Unassigned: {unassigned_vehicles[:5]}...")  # 只打印前5个
            print(f"   Assigning them to 'reloc' by default")
        
        if len(assignments) != len(vehicle_ids):
            print(f"⚠️  Warning: Mismatch in assignments - vehicles: {len(vehicle_ids)}, assignments: {len(assignments)} at step {self.env.step_num-1}")
        
        end_time = time.time()
        #print(f"NetworkX optimization time: {end_time - starttime:.4f} seconds")
        return assignments
    
    
    def _networkxsolve_ev_rebalancing(self, vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value):
        """
        NetworkX-based min-cost max-flow optimization using batch_q_value for costs
        batch_q_value shape: [num_vehicles, num_actions]
        where num_actions = num_requests + num_charging_stations + num_zones + 1 (wait)
        """
        import time
        starttime = time.time()
        G = nx.DiGraph()

        num_vehicles = len(vehicle_ids)
        num_requests = len(available_requests)
        
        layers = {
            'source': ['s'],
            'vehicles': [f'v{i}' for i in range(num_vehicles)],
            'requests': [f'r{j}' for j in range(num_requests)],
            'wait': ['w'],
            'sink': ['t']
        }
        #print("node number:", sum(len(v) for v in layers.values()))
        
        # Add all nodes with demands
        G.add_node('s', demand=-num_vehicles)  # Source supplies all vehicles
        for v in layers['vehicles']:
            G.add_node(v, demand=0)
        for r in layers['requests']:
            G.add_node(r, demand=0)
        G.add_node('w', demand=0)
        G.add_node('t', demand=num_vehicles)  # Sink demands all vehicles

        # Add edges: source → vehicles
        for v in layers['vehicles']:
            G.add_edge('s', v, capacity=1, weight=0)

        # Vehicles → requests (use -Q_value as cost for minimization, skip if Q-value is 0)
        for i in range(num_vehicles):
            for j in range(num_requests):
                if vehicle_action_matrix[i, j] == 0:
                    continue  # Skip edges with zero Q-value
                q_value = float(batch_q_value[i, j])
                cost = -q_value  # Minimize negative Q-value = Maximize Q-value
                G.add_edge(f'v{i}', f'r{j}', capacity=1, weight=cost)
        
        
        # Vehicles → wait (use -Q_value as cost, skip if Q-value is 0)
        for i in range(num_vehicles):
            col_idx = num_requests
            if vehicle_action_matrix[i, col_idx] == 0:
                continue  # Skip edges with zero Q-value
            cost = 0
            G.add_edge(f'v{i}', 'w', capacity=1, weight=cost)
        
        # Requests → sink (each request can be assigned to at most one vehicle)
        for j in range(num_requests):
            if vehicle_action_matrix[:, j].sum() > 0:
                G.add_edge(f'r{j}', 't', capacity=1, weight=0)
        
        
        # Wait → sink (只有当wait节点有入边时才添加)
        if vehicle_action_matrix[:, num_requests].sum() > 0:
            G.add_edge('w', 't', capacity=num_vehicles, weight=0)
        
        # 验证图的可行性
        total_supply = sum(data.get('demand', 0) for node, data in G.nodes(data=True) if data.get('demand', 0) < 0)
        total_demand = sum(data.get('demand', 0) for node, data in G.nodes(data=True) if data.get('demand', 0) > 0)
        
        # 打印图的基本信息
        # print(f"📊 Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
        # print(f"   Supply: {-total_supply}, Demand: {total_demand}", flush=True)
        

        
        # 正确检查：supply 是负数，demand 是正数，绝对值应该相等
        if abs(total_supply) != total_demand:
            print(f"❌ ERROR: Supply-demand imbalance! Supply={-total_supply}, Demand={total_demand}")
            print(f"  This will cause nx.min_cost_flow() to hang!")
            # 返回空分配
            return {vehicle_id: f"waiting" for vehicle_id in vehicle_ids}
        
        # 检查是否所有车辆都有至少一条出边
        vehicles_without_edges = []
        for i in range(num_vehicles):
            vehicle_node = f'v{i}'
            out_edges = list(G.successors(vehicle_node))
            if len(out_edges) == 0 or (len(out_edges) == 1 and out_edges[0] == 's'):
                vehicles_without_edges.append(i)
        
        if vehicles_without_edges:
            print(f"❌ ERROR: {len(vehicles_without_edges)} vehicles have no valid actions!")
            print(f"  Vehicles: {vehicles_without_edges[:10]}...")
            print(f"  This will cause infeasibility!")
            # 为这些车辆添加到 wait 节点的边
            for i in vehicles_without_edges:
                G.add_edge(f'v{i}', 'w', capacity=1, weight=50.0)  # 高成本的等待
            print(f"  Added emergency wait edges for {len(vehicles_without_edges)} vehicles")
        
        # Compute min cost flow
        # print(f"🔵 Starting nx.min_cost_flow() with {num_vehicles} vehicles, {G.number_of_edges()} edges...", flush=True)
        # print(f"   (Requests: {num_requests}, Charging: {num_charging}, Zones: {num_zones})", flush=True)
        
        import sys
        sys.stdout.flush()
        t_flow_start = time.time()
        
        try:
            # 使用线程+超时机制（Windows兼容）
            import threading
            flow_result = [None]
            error_result = [None]
            
            def compute_flow():
                try:
                    # 尝试不同的算法
                    # 1. 先尝试 capacity_scaling（通常最快）
                    try:
                        from networkx.algorithms.flow import capacity_scaling
                        flow_cost, flow_result_tmp = capacity_scaling(G)
                        flow_result[0] = flow_result_tmp
                        # print(f"  (使用 capacity_scaling 算法)", flush=True)
                    except Exception as e1:
                        # 2. 回退到 shortest_augmenting_path
                        try:
                            from networkx.algorithms.flow import shortest_augmenting_path
                            flow_result[0] = shortest_augmenting_path(G)
                            print(f"  (使用 shortest_augmenting_path 算法)", flush=True)
                        except Exception as e2:
                            flow_cost, flow_dict_tmp = nx.network_simplex(G)
                            flow_result[0] = flow_dict_tmp
                    # print(f"  (使用 network_simplex 默认算法)", flush=True)
                except Exception as e:
                    error_result[0] = e
            
            thread = threading.Thread(target=compute_flow)
            thread.daemon = True
            thread.start()
            thread.join(timeout=10.0)  # 10秒超时
            
            if thread.is_alive():
                print(f"❌ NetworkX min_cost_flow() 超时（10秒）！", flush=True)
                print(f"   建议使用 Gurobi（设置 usenetworkx=False）", flush=True)
                # 返回等待动作
                return {vehicle_id: f"waiting" for vehicle_id in vehicle_ids}
            
            if error_result[0]:
                raise error_result[0]
            
            flow_dict = flow_result[0]
            t_flow_end = time.time()
            elapsed = t_flow_end - t_flow_start
            #print(f"✅ nx.min_cost_flow() 完成，耗时 {elapsed:.4f} 秒", flush=True)
            
            if elapsed > 5.0:
                print(f"⚠️  NetworkX 较慢（{elapsed:.2f}秒），建议使用 Gurobi", flush=True)
        except nx.NetworkXUnfeasible:
            print("❌ ERROR: Flow problem is infeasible!")
            return {vehicle_id: f"waiting" for vehicle_id in vehicle_ids}
        except Exception as e:
            print(f"❌ ERROR in nx.min_cost_flow(): {e}")
            return {vehicle_id: f"waiting" for vehicle_id in vehicle_ids}
        
        total_cost = nx.cost_of_flow(G, flow_dict)
        
        # Extract assignments from flow
        assignments = {}
        unassigned_vehicles = []
        
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle_node = f'v{i}'
            assigned = False
            
            if vehicle_node in flow_dict:
                for target_node, flow in flow_dict[vehicle_node].items():
                    if flow > 0 and target_node != 's':  # Flow exists and not back to source
                        # Decode target node
                        if target_node.startswith('r'):
                            # Request assignment
                            request_idx = int(target_node[1:])
                            request = available_requests[request_idx]
                            assignments[vehicle_id] = request
                            assigned = True
                        elif target_node == 'w':
                            assignments[vehicle_id] = f"reloc"
                            assigned = True
                        if assigned:
                            break  # Found the assignment, move to next vehicle
            
            if not assigned:
                unassigned_vehicles.append((i, vehicle_id))
                assignments[vehicle_id] = f"reloc"  # 默认等待
        
        if unassigned_vehicles:
            print(f"⚠️  Warning: {len(unassigned_vehicles)} vehicles have no assignment in flow solution!")
            print(f"   Unassigned: {unassigned_vehicles[:5]}...")  # 只打印前5个
            print(f"   Assigning them to 'reloc' by default")
        
        if len(assignments) != len(vehicle_ids):
            print(f"⚠️  Warning: Mismatch in assignments - vehicles: {len(vehicle_ids)}, assignments: {len(assignments)} at step {self.env.step_num-1}")
        
        end_time = time.time()
        #print(f"NetworkX optimization time: {end_time - starttime:.4f} seconds")
        return assignments
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    def _gurobi_vehicle_rebalancing_network(self, vehicle_ids, available_requests,  vehicle_action_matrix, batch_q_value,iflp = True):
        """
        Gurobi optimization using vehicle_action_matrix for Q-values
        vehicle_action_matrix shape: [num_vehicles, num_actions]
        where num_actions = num_requests + num_charging_stations + num_zones + 1 (wait)
        """
        #print("current time:,", self.env.current_time, "rebalance vehicle:", len(vehicle_ids), "available requests:", len(available_requests))
        if not self.available or self._gurobi_runtime_failed:
            return self._fallback_after_gurobi_failure(
                vehicle_ids,
                available_requests,
                vehicle_action_matrix,
                batch_q_value,
                iflp=iflp,
                error="Gurobi is not available",
            )
        
        assignments = {}
        
        # Create optimization model
        try:
            model = self.gp.Model("vehicle_assignment_network")
        except Exception as e:
            return self._fallback_after_gurobi_failure(
                vehicle_ids,
                available_requests,
                vehicle_action_matrix,
                batch_q_value,
                iflp=iflp,
                error=e,
            )
        model.setParam('OutputFlag', 0)  # Suppress output
        model.setParam('Threads', self.num_threads)
        model.setParam('TimeLimit', self.network_time_limit)
        if iflp:
            model.setParam('Method', 1)
        
        num_action = vehicle_action_matrix.shape[1]
        layout = self._get_matrix_action_layout(available_requests, num_action)
        num_requests = layout['num_requests']
        scale_charge_station_ids = layout['charge_station_ids']
        scale_zone_indices = layout['zone_indices']
        matrix_requests = layout['requests']
        num_charging = layout['num_charging']
        num_zones = layout['num_zones']
        
        # Get charging stations list
        charging_stations_list = [
            self.env.charging_manager.stations[sid]
            for sid in scale_charge_station_ids
            if sid in self.env.charging_manager.stations
        ] if hasattr(self.env, 'charging_manager') else []
        
        # Create decision variables: assign_vehicle[i][a] = 1 if vehicle i takes action a
        if iflp:
            assign_vehicle = [[model.addVar(vtype=self.GRB.CONTINUOUS,
                         name=f'vehicle_{vehicle_id}_action_{a}', lb=0.0, ub=1.0) 
                         for a in range(num_action)] 
                         for i, vehicle_id in enumerate(vehicle_ids)]
        else:
            assign_vehicle = [[model.addVar(vtype=self.GRB.BINARY,
                     name=f'vehicle_{vehicle_id}_action_{a}') 
                     for a in range(num_action)] 
                     for i, vehicle_id in enumerate(vehicle_ids)]
        
        # Constraint 1: Each vehicle must choose exactly one action.
        #
        # In the queue-constrained synthetic environment, several critically
        # low-battery AEVs can simultaneously have only capacity-limited
        # charging actions.  Requiring every such row to take one of those
        # actions makes the global assignment infeasible once the shared
        # station admissions are exhausted.  Add a heavily penalized
        # synthetic-only fallback variable so Gurobi can leave only the
        # conflicting vehicles unassigned; extraction maps that fallback to
        # waiting.  NYC keeps the original formulation unchanged.
        synthetic_fallback_enabled = hasattr(
            self.env,
            'synthetic_demand_profile',
        )
        fallback_vehicle = [None] * len(vehicle_ids)
        for i in range(len(vehicle_ids)):
            has_feasible_action = bool(np.any(vehicle_action_matrix[i, :] != 0))
            is_ev = self.env.vehicles[vehicle_ids[i]]['type'] == 1
            if synthetic_fallback_enabled and not is_ev:
                fallback_vehicle[i] = model.addVar(
                    vtype=(self.GRB.CONTINUOUS if iflp else self.GRB.BINARY),
                    name=f'vehicle_{vehicle_ids[i]}_synthetic_fallback',
                    lb=0.0,
                    ub=1.0,
                )
                model.addConstr(
                    self.gp.quicksum(
                        assign_vehicle[i][a] for a in range(num_action)
                    ) + fallback_vehicle[i] == 1,
                    name=f'vehicle_{vehicle_ids[i]}_one_action',
                )
            elif is_ev or not has_feasible_action:
                model.addConstr(self.gp.quicksum(assign_vehicle[i][a] for a in range(num_action)) <= 1,
                            name=f'vehicle_{vehicle_ids[i]}_one_action')
            else:
                model.addConstr(self.gp.quicksum(assign_vehicle[i][a] for a in range(num_action)) == 1,
                            name=f'vehicle_{vehicle_ids[i]}_one_action')
        
        # EV only assign
        
        for i in range(len(vehicle_ids)):
            if self.env.vehicles[vehicle_ids[i]]['type'] == 1:  # EV vehicle
                for a in range(num_action):
                    if a >= num_requests:
                        model.addConstr(assign_vehicle[i][a] == 0,
                                      name=f'ev_vehicle_{vehicle_ids[i]}_no_non_request_action_{a}')

        
        
        # Constraint 2: Each request can be assigned to at most one vehicle
        for j in range(num_requests):
            model.addConstr(self.gp.quicksum(assign_vehicle[i][j] for i in range(len(vehicle_ids))) <= 1,
                          name=f'request_{j}_max_one_vehicle')
        
        
        for j in range(num_action):
            for i in range(len(vehicle_ids)):
                if vehicle_action_matrix[i, j] == 0:
                    model.addConstr(assign_vehicle[i][j] == 0,
                                  name=f'no_action_vehicle_{vehicle_ids[i]}_action_{j}')
        
        
        
        # Constraint 3: Charging station capacity constraints
        for k in range(num_charging):
            if k < len(charging_stations_list):
                station = charging_stations_list[k]
                vacancy = self._charging_station_vacancy(station)
            else:
                # Fallback if charging station list is shorter than expected
                vacancy = 0
            station_idx = k + num_requests
            model.addConstr(self.gp.quicksum(assign_vehicle[i][station_idx] for i in range(len(vehicle_ids))) <= vacancy,
                          name=f'charging_station_{k}_capacity')
        
        # Objective: Maximize total Q-value
        # vehicle_action_matrix[i, a] contains the Q-value for vehicle i taking action a
        objective_terms = self.gp.LinExpr()
        for i in range(len(vehicle_ids)):
            for a in range(num_action):
                q_value = float(batch_q_value[i, a])
                objective_terms += q_value * assign_vehicle[i][a]
        if synthetic_fallback_enabled:
            finite_q = np.asarray(batch_q_value, dtype=float)
            finite_q = finite_q[np.isfinite(finite_q)]
            q_scale = float(np.max(np.abs(finite_q))) if finite_q.size else 1.0
            fallback_penalty = -max(100.0, 10.0 * q_scale)
            for fallback_var in fallback_vehicle:
                if fallback_var is not None:
                    objective_terms += fallback_penalty * fallback_var
        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)
        
        # Update model to ensure all variables and constraints are built
        model.update()
        
        # Record model statistics before optimization
        if hasattr(self.env, 'record_time') and self.env.record_time:
            num_vars = model.NumVars
            num_constrs = model.NumConstrs
            # print(f"Gurobi model stats: {num_vars} variables, {num_constrs} constraints")
            self.env.time_stats['gurobi_variables'].append(num_vars)
            self.env.time_stats['gurobi_constraints'].append(num_constrs)
        
        # Optimize
        try:
            import time
            t_solve_start = time.time()
            model.optimize()
            t_solve_end = time.time()
            
            # Record solve time
            if hasattr(self.env, 'record_time') and self.env.record_time:
                self.env.time_stats['gurobi_solve'].append(t_solve_end - t_solve_start)
            
            #print(len(model.getVars()), "variables, ", len(model.getConstrs()), "constraints in Gurobi model.")
            # Extract assignments
            if self._should_extract_solution(model):
                synthetic_fallback_count = 0
                for i, vehicle_id in enumerate(vehicle_ids):
                    assigned = False
                    for a in range(num_action):
                        if assign_vehicle[i][a].x > 0.5:
                            # Decode action
                            if a < num_requests:
                                # Request assignment
                                request = matrix_requests[a] if a < len(matrix_requests) else None
                                if request is None:
                                    assignments[vehicle_id] = "reloc" if self.env.vehicles[vehicle_id]['type'] == 1 else "waiting"
                                else:
                                    assignments[vehicle_id] = request
                                assigned = True
                            elif a < num_requests + num_charging:
                                # Charging station assignment
                                station_idx = a - num_requests
                                if station_idx < len(charging_stations_list):
                                    station = charging_stations_list[station_idx]
                                    assignments[vehicle_id] = f"charge_{station.id}"
                                else:
                                    assignments[vehicle_id] = f"waiting"
                                assigned = True
                            elif a < num_requests + num_charging + num_zones:
                                # Zone relocation assignment
                                zone_idx = a - num_requests - num_charging
                                if self.env.vehicles[vehicle_id]['type'] == 1:
                                    # EV cannot relocate to zones, force wait
                                    assignments[vehicle_id] = f"reloc"
                                else:
                                    mapped_zone_idx = scale_zone_indices[zone_idx] if zone_idx < len(scale_zone_indices) else zone_idx
                                    assignments[vehicle_id] = f"idle_at_{mapped_zone_idx}"
                                assigned = True
                            else:
                                if self.env.vehicles[vehicle_id]['type'] == 1:
                                    # EV cannot wait, force relocate
                                    assignments[vehicle_id] = f"reloc"
                                else:
                                    assignments[vehicle_id] = f"waiting"
                                assigned = True
                            break
                    if not assigned:
                        fallback_var = fallback_vehicle[i]
                        if (
                            fallback_var is not None
                            and fallback_var.x > 0.5
                        ):
                            synthetic_fallback_count += 1
                        if self.env.vehicles[vehicle_id]['type'] == 1:
                            # EV cannot wait, force relocate
                            assignments[vehicle_id] = f"reloc"
                        else:
                            assignments[vehicle_id] = f"waiting"
                if synthetic_fallback_count:
                    current_count = int(getattr(
                        self.env,
                        'synthetic_solver_fallback_count',
                        0,
                    ))
                    self.env.synthetic_solver_fallback_count = (
                        current_count + synthetic_fallback_count
                    )
            else:
                print(f"Optimization status: {model.status}")
                # Fallback: all vehicles wait
                for vehicle_id in vehicle_ids:
                    assignments[vehicle_id] = f"waiting"
                    
                if model.status == self.GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS...")
                    model.computeIIS()
                    model.write("infeasible_model.ilp")
                    
        except Exception as e:
            return self._fallback_after_gurobi_failure(
                vehicle_ids,
                available_requests,
                vehicle_action_matrix,
                batch_q_value,
                iflp=iflp,
                error=e,
            )
        
        return assignments
        
        
    
    
    
    def _gurobi_vehicle_rebalancing_network_ev(self, vehicle_ids, available_requests,  vehicle_action_matrix, batch_q_value,iflp = True):
        """
        Gurobi optimization using vehicle_action_matrix for Q-values
        vehicle_action_matrix shape: [num_vehicles, num_actions]
        where num_actions = num_requests + num_charging_stations + num_zones + 1 (wait)
        """
        #print("current time:,", self.env.current_time, "rebalance vehicle:", len(vehicle_ids), "available requests:", len(available_requests))
        if not self.available or self._gurobi_runtime_failed:
            return self._fallback_after_gurobi_failure(
                vehicle_ids,
                available_requests,
                vehicle_action_matrix,
                batch_q_value,
                iflp=iflp,
                ev_only=True,
                error="Gurobi is not available",
            )
        
        assignments = {}
        
        # Create optimization model
        try:
            model = self.gp.Model("vehicle_assignment_network")
        except Exception as e:
            return self._fallback_after_gurobi_failure(
                vehicle_ids,
                available_requests,
                vehicle_action_matrix,
                batch_q_value,
                iflp=iflp,
                ev_only=True,
                error=e,
            )
        model.setParam('OutputFlag', 0)  # Suppress output
        model.setParam('Threads', self.num_threads)
        model.setParam('TimeLimit', self.network_time_limit)
        if iflp:
            model.setParam('Method', 1)
        
        num_action = vehicle_action_matrix.shape[1]
        layout = self._get_matrix_action_layout(available_requests, num_action)
        num_requests = layout['num_requests']
        matrix_requests = layout['requests']
        
        if iflp:
            assign_vehicle = [[model.addVar(vtype=self.GRB.CONTINUOUS,
                         name=f'vehicle_{vehicle_id}_action_{a}', lb=0.0, ub=1.0) 
                         for a in range(num_action)] 
                         for i, vehicle_id in enumerate(vehicle_ids)]
        else:
            assign_vehicle = [[model.addVar(vtype=self.GRB.BINARY,
                        name=f'vehicle_{vehicle_id}_action_{a}') 
                        for a in range(num_action)] 
                        for i, vehicle_id in enumerate(vehicle_ids)]
        
        # Constraint 1: Each vehicle must choose exactly one action
        for i in range(len(vehicle_ids)):
            if bool(np.any(vehicle_action_matrix[i, :] != 0)):
                model.addConstr(self.gp.quicksum(assign_vehicle[i][a] for a in range(num_action)) == 1,
                            name=f'vehicle_{vehicle_ids[i]}_one_action')
            else:
                model.addConstr(self.gp.quicksum(assign_vehicle[i][a] for a in range(num_action)) <= 1,
                            name=f'vehicle_{vehicle_ids[i]}_one_action')
        
    
        
        # Constraint 2: Each request can be assigned to at most one vehicle
        for j in range(num_requests):
            model.addConstr(self.gp.quicksum(assign_vehicle[i][j] for i in range(len(vehicle_ids))) <= 1,
                          name=f'request_{j}_max_one_vehicle')
        
        
        for j in range(num_action):
            for i in range(len(vehicle_ids)):
                if vehicle_action_matrix[i, j] == 0:
                    model.addConstr(assign_vehicle[i][j] == 0,
                                  name=f'no_action_vehicle_{vehicle_ids[i]}_action_{j}')
        

        
        # Objective: Maximize total Q-value
        # vehicle_action_matrix[i, a] contains the Q-value for vehicle i taking action a
        objective_terms = self.gp.LinExpr()
        for i in range(len(vehicle_ids)):
            for a in range(num_action):
                q_value = float(batch_q_value[i, a])
                objective_terms += q_value * assign_vehicle[i][a]
        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)
        
        # Optimize
        try:
            model.optimize()
            #print(len(model.getVars()), "variables, ", len(model.getConstrs()), "constraints in Gurobi model.")
            # Extract assignments
            if self._should_extract_solution(model):
                for i, vehicle_id in enumerate(vehicle_ids):
                    assigned = False
                    for a in range(num_action):
                        if assign_vehicle[i][a].x > 0.5:
                            # Decode action
                            if a < num_requests:
                                # Request assignment
                                request = matrix_requests[a] if a < len(matrix_requests) else None
                                assignments[vehicle_id] = request if request is not None else "reloc"
                                assigned = True
                            else:
                                assignments[vehicle_id] = f"reloc"
                                assigned = True
                            break
                    if not assigned:
                        if self.env.vehicles[vehicle_id]['type'] == 1:
                            # EV cannot wait, force relocate
                            assignments[vehicle_id] = f"reloc"
                        else:
                            assignments[vehicle_id] = f"waiting"
            else:
                print(f"Optimization status: {model.status}")
                # Fallback: all vehicles wait
                for vehicle_id in vehicle_ids:
                    assignments[vehicle_id] = f"waiting"
                    
                if model.status == self.GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS...")
                    model.computeIIS()
                    model.write("infeasible_model.ilp")
                    
        except Exception as e:
            return self._fallback_after_gurobi_failure(
                vehicle_ids,
                available_requests,
                vehicle_action_matrix,
                batch_q_value,
                iflp=iflp,
                ev_only=True,
                error=e,
            )
        
        return assignments


    # ═══════════════════════════════════════════════════════════════════
    #  Min-Cost Max-Flow (SPFA) based vehicle rebalancing
    #  Polynomial O(V·E·F) complexity, no Gurobi dependency
    #  Based on minflow.py — each action gets its own node for full
    #  global optimality (no pre-computed fallback).
    # ═══════════════════════════════════════════════════════════════════

    class _MCMFsolver_gpu:
        def __init__(self, num_nodes: int):
            self.num_nodes = num_nodes
            self.graph = [[] for _ in range(num_nodes)]
            self.available = False
            self._cuda = None
            self._copy_float_kernel = None
            self._relax_edges_kernel = None
            self._write_parent_kernel = None
            self._init_cuda()

        def _init_cuda(self):
            try:
                (
                    self._cuda,
                    self._copy_float_kernel,
                    self._relax_edges_kernel,
                    self._write_parent_kernel,
                ) = _get_gpu_mcmf_kernels()
                self.available = bool(self._cuda.is_available())
            except Exception:
                self.available = False

        def add_edge(self, u: int, v: int, capacity: int, cost: float):
            self.graph[u].append([v, capacity, cost, len(self.graph[v])])
            self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

        def _build_residual_arrays(self):
            edge_src = []
            edge_dst = []
            edge_cap = []
            edge_cost = []
            edge_graph_idx = []

            for u, edges in enumerate(self.graph):
                for idx, edge in enumerate(edges):
                    v, cap, cost, _ = edge
                    if cap <= 0:
                        continue
                    edge_src.append(u)
                    edge_dst.append(v)
                    edge_cap.append(int(cap))
                    edge_cost.append(float(cost))
                    edge_graph_idx.append(idx)

            return (
                np.asarray(edge_src, dtype=np.int32),
                np.asarray(edge_dst, dtype=np.int32),
                np.asarray(edge_cap, dtype=np.int32),
                np.asarray(edge_cost, dtype=np.float32),
                np.asarray(edge_graph_idx, dtype=np.int32),
            )

        def sssp(self, s: int, t: int, deadline: float = None):
            if not self.available:
                raise RuntimeError("CUDA MCMF solver is not available")

            (
                edge_src,
                edge_dst,
                edge_cap,
                edge_cost,
                edge_graph_idx,
            ) = self._build_residual_arrays()
            num_edges = len(edge_src)
            if num_edges == 0:
                return False, [-1] * self.num_nodes, [-1] * self.num_nodes

            inf_value = np.float32(1e20)
            tol = np.float32(1e-5)
            threads_per_block = 256
            edge_blocks = max(1, math.ceil(num_edges / threads_per_block))
            node_blocks = max(1, math.ceil(self.num_nodes / threads_per_block))

            d_edge_src = self._cuda.to_device(edge_src)
            d_edge_dst = self._cuda.to_device(edge_dst)
            d_edge_cap = self._cuda.to_device(edge_cap)
            d_edge_cost = self._cuda.to_device(edge_cost)
            d_edge_graph_idx = self._cuda.to_device(edge_graph_idx)

            dist = np.full(self.num_nodes, inf_value, dtype=np.float32)
            dist[s] = 0.0
            parent_node = np.full(self.num_nodes, -1, dtype=np.int32)
            parent_edge = np.full(self.num_nodes, -1, dtype=np.int32)

            d_dist = self._cuda.to_device(dist)
            d_next_dist = self._cuda.device_array_like(d_dist)
            d_parent_node = self._cuda.to_device(parent_node)
            d_parent_edge = self._cuda.to_device(parent_edge)
            d_changed = self._cuda.to_device(np.zeros(1, dtype=np.int32))

            for _ in range(self.num_nodes - 1):
                if deadline is not None and time.time() > deadline:
                    raise TimeoutError("MCMF GPU SSSP exceeded time limit")

                self._copy_float_kernel[node_blocks, threads_per_block](
                    d_dist, d_next_dist, self.num_nodes
                )
                d_changed.copy_to_device(np.zeros(1, dtype=np.int32))
                self._relax_edges_kernel[edge_blocks, threads_per_block](
                    d_dist,
                    d_next_dist,
                    d_edge_src,
                    d_edge_dst,
                    d_edge_cap,
                    d_edge_cost,
                    num_edges,
                    inf_value,
                    tol,
                    d_changed,
                )
                self._cuda.synchronize()

                changed_host = d_changed.copy_to_host()
                if not changed_host[0]:
                    break

                self._write_parent_kernel[edge_blocks, threads_per_block](
                    d_dist,
                    d_next_dist,
                    d_edge_src,
                    d_edge_dst,
                    d_edge_cap,
                    d_edge_cost,
                    d_edge_graph_idx,
                    num_edges,
                    inf_value,
                    tol,
                    d_parent_node,
                    d_parent_edge,
                )
                self._cuda.synchronize()
                d_dist, d_next_dist = d_next_dist, d_dist

            dist_host = d_dist.copy_to_host()
            if not np.isfinite(dist_host[t]) or dist_host[t] >= inf_value / 2:
                return False, [-1] * self.num_nodes, [-1] * self.num_nodes

            return (
                True,
                d_parent_node.copy_to_host().tolist(),
                d_parent_edge.copy_to_host().tolist(),
            )

        def solve(self, s: int, t: int, time_limit: float = None):
            deadline = time.time() + time_limit if time_limit is not None else None
            max_flow = 0
            min_cost = 0.0

            while True:
                if deadline is not None and time.time() > deadline:
                    raise TimeoutError("MCMF GPU solve exceeded time limit")

                has_path, parent_node, parent_edge = self.sssp(s, t, deadline=deadline)
                if not has_path:
                    break

                push = float('inf')
                curr = t
                hop_count = 0
                while curr != s:
                    p = parent_node[curr]
                    idx = parent_edge[curr]
                    if p < 0 or idx < 0:
                        raise RuntimeError("GPU SSSP returned an incomplete augmenting path")
                    push = min(push, self.graph[p][idx][1])
                    curr = p
                    hop_count += 1
                    if hop_count > self.num_nodes:
                        raise RuntimeError("GPU SSSP produced a cyclic augmenting path")

                curr = t
                while curr != s:
                    p = parent_node[curr]
                    idx = parent_edge[curr]
                    rev_idx = self.graph[p][idx][3]
                    self.graph[p][idx][1] -= push
                    self.graph[curr][rev_idx][1] += push
                    min_cost += push * self.graph[p][idx][2]
                    curr = p

                max_flow += push

            return max_flow, min_cost

    class Auction_solver:
        def __init__(
            self, vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value,
            action_capacities=None, epsilon=1e-3, max_iterations=100,
            add_dummy_slots=True, dummy_values=None, top_k=None,
        ):
            self.vehicle_ids = list(vehicle_ids)
            self.available_requests = available_requests
            self.vehicle_action_matrix = np.asarray(vehicle_action_matrix)
            self.batch_q_value = np.asarray(batch_q_value, dtype=np.float64)
            self.num_vehicles = len(vehicle_ids)
            self.num_actions = self.vehicle_action_matrix.shape[1]
            self.epsilon = float(epsilon)
            self.max_iterations = max_iterations or max(1000, self.num_vehicles * max(self.num_actions, 1) * 20)
            self.add_dummy_slots = add_dummy_slots
            self.dummy_values = None if dummy_values is None else np.asarray(dummy_values, dtype=np.float64)
            self.top_k = top_k
            self.action_capacities = self._normalize_capacities(action_capacities)
            self.expanded_actions = None
            self.slot_is_unlimited = None
            self.expanded_q_value = None
            self.expanded_feasibility = None
            self.dummy_value = None
            self.prices = None
            self.owner = None
            self.assigned_slot = None
            self.iterations = 0
            self.converged = False

        def _normalize_capacities(self, action_capacities):
            if action_capacities is None:
                capacities = np.full(self.num_actions, self.num_vehicles, dtype=np.int64)
                capacities[:len(self.available_requests)] = 1
                return capacities
            capacities = np.asarray(action_capacities, dtype=np.int64)
            if capacities.shape[0] != self.num_actions:
                raise ValueError("action_capacities length must match number of actions")
            return np.maximum(capacities, 0)

        def _fallback_dummy_value(self):
            if self.dummy_values is not None:
                if self.dummy_values.shape[0] != self.num_vehicles:
                    raise ValueError("dummy_values length must match number of vehicles")
                return self.dummy_values.astype(np.float64, copy=False)

            feasible = self.vehicle_action_matrix != 0
            if np.any(feasible):
                values = self.batch_q_value[feasible]
                min_value = float(np.min(values))
                span = float(np.max(values) - min_value)
                value = min_value - max(span, abs(min_value), 1.0) - 1.0
            else:
                value = 0.0
            return np.full(self.num_vehicles, value, dtype=np.float64)

        def _build_expanded_problem(self):
            expanded_actions = []
            slot_is_unlimited = []
            for action_idx, capacity in enumerate(self.action_capacities):
                if capacity <= 0:
                    continue
                if capacity >= self.num_vehicles:
                    expanded_actions.append(action_idx)
                    slot_is_unlimited.append(True)
                else:
                    expanded_actions.extend([action_idx] * int(capacity))
                    slot_is_unlimited.extend([False] * int(capacity))

            if self.add_dummy_slots:
                expanded_actions.append(-1)
                slot_is_unlimited.append(True)

            if not expanded_actions:
                expanded_actions = [-1]
                slot_is_unlimited = [True]

            self.expanded_actions = np.asarray(expanded_actions, dtype=np.int64)
            self.slot_is_unlimited = np.asarray(slot_is_unlimited, dtype=bool)
            num_slots = self.expanded_actions.shape[0]
            self.expanded_q_value = np.empty((self.num_vehicles, num_slots), dtype=np.float64)
            self.expanded_feasibility = np.ones((self.num_vehicles, num_slots), dtype=bool)
            self.dummy_value = self._fallback_dummy_value()

            for slot_idx, action_idx in enumerate(self.expanded_actions):
                if action_idx < 0:
                    self.expanded_q_value[:, slot_idx] = self.dummy_value
                    self.expanded_feasibility[:, slot_idx] = True
                else:
                    self.expanded_q_value[:, slot_idx] = self.batch_q_value[:, action_idx]
                    self.expanded_feasibility[:, slot_idx] = self.vehicle_action_matrix[:, action_idx] != 0

            if self.top_k is not None and self.top_k > 0 and self.top_k < num_slots:
                for row_idx in range(self.num_vehicles):
                    feasible_slots = np.flatnonzero(self.expanded_feasibility[row_idx])
                    if feasible_slots.size <= self.top_k:
                        continue
                    row_values = self.expanded_q_value[row_idx, feasible_slots]
                    keep_local = np.argpartition(row_values, -self.top_k)[-self.top_k:]
                    keep_slots = feasible_slots[keep_local]
                    self.expanded_feasibility[row_idx, feasible_slots] = False
                    self.expanded_feasibility[row_idx, keep_slots] = True

            self.prices = np.zeros(num_slots, dtype=np.float64)
            self.owner = np.full(num_slots, -1, dtype=np.int64)
            self.assigned_slot = np.full(self.num_vehicles, -1, dtype=np.int64)

        def _find_best_and_second_match_by_row(self, row_idx):
            if self.expanded_q_value is None:
                self._build_expanded_problem()
            net_values = self.expanded_q_value[row_idx] - self.prices
            net_values = np.where(self.expanded_feasibility[row_idx], net_values, -np.inf)
            best_slot = int(np.argmax(net_values))
            best_value = float(net_values[best_slot])
            if not np.isfinite(best_value):
                return -1, -np.inf, -np.inf
            second_values = net_values.copy()
            second_values[best_slot] = -np.inf
            second_value = float(np.max(second_values))
            return best_slot, best_value, second_value

        def find_best_and_second_match(self, vehicle_id):
            row_idx = self.vehicle_ids.index(vehicle_id)
            return self._find_best_and_second_match_by_row(row_idx)

        def _calculate_bid_price_by_row(self, row_idx):
            best_slot, best_value, second_value = self._find_best_and_second_match_by_row(row_idx)
            if best_slot < 0:
                return -1, -np.inf
            if not np.isfinite(second_value):
                second_value = best_value - self.epsilon
            bid_increment = best_value - second_value + self.epsilon
            return best_slot, float(self.prices[best_slot] + bid_increment)

        def calculate_bid_price(self, vehicle_id):
            row_idx = self.vehicle_ids.index(vehicle_id)
            return self._calculate_bid_price_by_row(row_idx)

        def solve(self):
            self._build_expanded_problem()
            if self.num_vehicles == 0:
                self.converged = True
                return {}

            active = np.ones(self.num_vehicles, dtype=bool)
            for iteration in range(self.max_iterations):
                active_indices = np.flatnonzero(active)
                if active_indices.size == 0:
                    self.iterations = iteration
                    self.converged = True
                    break

                vehicle_idx = int(active_indices[0])
                best_slot, bid = self._calculate_bid_price_by_row(vehicle_idx)
                if best_slot < 0:
                    active[vehicle_idx] = False
                    continue

                previous_slot = int(self.assigned_slot[vehicle_idx])
                if previous_slot >= 0 and previous_slot != best_slot and self.owner[previous_slot] == vehicle_idx:
                    self.owner[previous_slot] = -1

                if self.slot_is_unlimited[best_slot]:
                    self.assigned_slot[vehicle_idx] = best_slot
                    active[vehicle_idx] = False
                    continue

                old_owner = int(self.owner[best_slot])
                self.owner[best_slot] = vehicle_idx
                self.assigned_slot[vehicle_idx] = best_slot
                self.prices[best_slot] = bid
                active[vehicle_idx] = False

                if old_owner >= 0 and old_owner != vehicle_idx:
                    self.assigned_slot[old_owner] = -1
                    active[old_owner] = True
            else:
                self.iterations = self.max_iterations

            return self.get_assigned_actions()

        def get_assigned_actions(self):
            assigned_actions = {}
            if self.expanded_actions is None:
                return assigned_actions
            for vehicle_idx, vehicle_id in enumerate(self.vehicle_ids):
                slot_idx = int(self.assigned_slot[vehicle_idx])
                if slot_idx >= 0:
                    action_idx = int(self.expanded_actions[slot_idx])
                    if action_idx >= 0:
                        assigned_actions[vehicle_id] = action_idx
            return assigned_actions



    class Auction_gpu_solver_cuda(Auction_solver):
        def __init__(
            self, vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value,
            action_capacities=None, epsilon=1e-3, max_rounds=100,
            add_dummy_slots=True, dummy_values=None, top_k=None,
            sync_interval=32,
        ):
            self.add_dummy_slots = add_dummy_slots
            self.iterations = 0
            self.converged = False
            self.expanded_actions = None
            self.expanded_actions_t = None
            self.expanded_action_mask = None
            self.expanded_action_indices = None
            self.all_slot_indices = None
            self.slot_is_unlimited = None
            self.expanded_feasibility = None
            self.candidate_slots = None
            self.candidate_q = None
            self.candidate_is_unlimited = None
            self._cached_candidate_width = None
            self._structure_signature = None
            self.prices = None
            self.owner = None
            self.assigned_slot = None
            self.sync_interval = 1
            self.available = False
            self.unavailable_reason = ""
            self._torch = None
            self._device = None

            try:
                import torch

                self._torch = torch
                device = None
                if isinstance(batch_q_value, torch.Tensor) and batch_q_value.is_cuda:
                    device = batch_q_value.device
                elif isinstance(vehicle_action_matrix, torch.Tensor) and vehicle_action_matrix.is_cuda:
                    device = vehicle_action_matrix.device

                if device is None:
                    self.unavailable_reason = "Torch CUDA auction requires a CUDA tensor input"
                    return

                self._device = device
                self.refresh_inputs(
                    vehicle_ids,
                    available_requests,
                    vehicle_action_matrix,
                    batch_q_value,
                    action_capacities=action_capacities,
                    dummy_values=dummy_values,
                    top_k=top_k,
                    epsilon=epsilon,
                    max_rounds=max_rounds,
                    sync_interval=sync_interval,
                )
                self.available = True
            except Exception as exc:
                self.available = False
                self.unavailable_reason = f"{type(exc).__name__}: {exc}"

        def _as_torch_tensor(self, value, dtype=None):
            torch = self._torch
            if isinstance(value, torch.Tensor):
                if value.device != self._device:
                    value = value.to(device=self._device)
                if dtype is not None and value.dtype != dtype:
                    value = value.to(dtype=dtype)
                return value
            return torch.as_tensor(value, device=self._device, dtype=dtype)

        def _normalize_capacities(self, action_capacities):
            torch = self._torch
            if action_capacities is None:
                capacities = torch.full(
                    (self.num_actions,),
                    self.num_vehicles,
                    dtype=torch.long,
                    device=self._device,
                )
                capacities[:len(self.available_requests)] = 1
                return capacities
            capacities = self._as_torch_tensor(action_capacities, dtype=torch.long).reshape(-1)
            if int(capacities.shape[0]) != self.num_actions:
                raise ValueError("action_capacities length must match number of actions")
            return torch.clamp(capacities, min=0)

        def refresh_inputs(
            self,
            vehicle_ids,
            available_requests,
            vehicle_action_matrix,
            batch_q_value,
            action_capacities=None,
            dummy_values=None,
            top_k=None,
            epsilon=None,
            max_rounds=None,
            sync_interval=None,
        ):
            self.vehicle_ids = list(vehicle_ids)
            self.available_requests = available_requests
            if epsilon is not None:
                self.epsilon = float(epsilon)
            if top_k is not None or self.expanded_actions is None:
                self.top_k = top_k
            if sync_interval is not None:
                self.sync_interval = max(1, int(sync_interval))

            self.batch_q_value = self._as_torch_tensor(batch_q_value, dtype=self._torch.float32).detach()
            self.vehicle_action_matrix = self._as_torch_tensor(vehicle_action_matrix).detach()
            if self.vehicle_action_matrix.ndim != 2 or self.batch_q_value.ndim != 2:
                raise ValueError("auction inputs must be rank-2 tensors")

            self.num_vehicles = len(vehicle_ids)
            self.num_actions = int(self.vehicle_action_matrix.shape[1])
            if int(self.vehicle_action_matrix.shape[0]) != self.num_vehicles:
                raise ValueError("vehicle_action_matrix row count must match vehicle_ids")
            if int(self.batch_q_value.shape[0]) != self.num_vehicles:
                raise ValueError("batch_q_value row count must match vehicle_ids")
            if int(self.batch_q_value.shape[1]) != self.num_actions:
                raise ValueError("batch_q_value column count must match vehicle_action_matrix")

            self.max_iterations = max_rounds or max(1000, self.num_vehicles * max(self.num_actions, 1) * 20)
            self.max_rounds = self.max_iterations
            self.dummy_values = None if dummy_values is None else self._as_torch_tensor(dummy_values, dtype=self.batch_q_value.dtype).reshape(-1).detach()
            self.action_capacities = self._normalize_capacities(action_capacities)
            self._ensure_structure_cache()

        def _ensure_structure_cache(self):
            torch = self._torch
            capacities_host = tuple(int(x) for x in self.action_capacities.detach().cpu().tolist())
            signature = (
                self.num_vehicles,
                self.num_actions,
                capacities_host,
                bool(self.add_dummy_slots),
                self.top_k,
                str(self._device),
            )
            if signature == self._structure_signature:
                return

            expanded_actions = []
            slot_is_unlimited = []
            for action_idx, capacity in enumerate(capacities_host):
                if capacity <= 0:
                    continue
                if capacity >= self.num_vehicles:
                    expanded_actions.append(action_idx)
                    slot_is_unlimited.append(True)
                else:
                    expanded_actions.extend([action_idx] * int(capacity))
                    slot_is_unlimited.extend([False] * int(capacity))

            if self.add_dummy_slots:
                expanded_actions.append(-1)
                slot_is_unlimited.append(True)

            if not expanded_actions:
                expanded_actions = [-1]
                slot_is_unlimited = [True]

            self.expanded_actions = np.asarray(expanded_actions, dtype=np.int64)
            self.expanded_actions_t = torch.as_tensor(self.expanded_actions, dtype=torch.long, device=self._device)
            self.expanded_action_mask = self.expanded_actions_t >= 0
            self.expanded_action_indices = self.expanded_actions_t[self.expanded_action_mask]
            self.slot_is_unlimited = torch.as_tensor(slot_is_unlimited, dtype=torch.bool, device=self._device)
            self.all_slot_indices = torch.arange(int(self.expanded_actions_t.shape[0]), dtype=torch.long, device=self._device)
            self.expanded_feasibility = torch.empty(
                (self.num_vehicles, int(self.expanded_actions_t.shape[0])),
                dtype=torch.bool,
                device=self._device,
            )
            self.prices = torch.zeros(int(self.expanded_actions_t.shape[0]), dtype=self.batch_q_value.dtype, device=self._device)
            self.owner = torch.full((int(self.expanded_actions_t.shape[0]),), -1, dtype=torch.long, device=self._device)
            self.assigned_slot = torch.full((self.num_vehicles,), -1, dtype=torch.long, device=self._device)
            candidate_width = self._candidate_width(int(self.expanded_actions_t.shape[0]))
            self.candidate_slots = torch.empty((self.num_vehicles, candidate_width), dtype=torch.long, device=self._device)
            self.candidate_q = torch.empty((self.num_vehicles, candidate_width), dtype=self.batch_q_value.dtype, device=self._device)
            self.candidate_is_unlimited = torch.empty((self.num_vehicles, candidate_width), dtype=torch.bool, device=self._device)
            self._cached_candidate_width = candidate_width
            self._structure_signature = signature

        def _candidate_width(self, num_slots):
            if self.top_k is None or self.top_k <= 0 or self.top_k >= num_slots:
                return num_slots
            return max(2, min(int(self.top_k), num_slots))

        def _fallback_dummy_value(self):
            torch = self._torch
            if self.dummy_values is not None:
                if int(self.dummy_values.shape[0]) != self.num_vehicles:
                    raise ValueError("dummy_values length must match number of vehicles")
                return self.dummy_values.to(dtype=self.batch_q_value.dtype)

            feasible = self.vehicle_action_matrix != 0
            values = self.batch_q_value.masked_select(feasible)
            if values.numel() > 0:
                min_value = values.min()
                span = values.max() - min_value
                low_value = min_value - torch.maximum(
                    torch.maximum(span, min_value.abs()),
                    torch.tensor(1.0, device=self._device, dtype=self.batch_q_value.dtype),
                ) - 1.0
            else:
                low_value = torch.tensor(0.0, device=self._device, dtype=self.batch_q_value.dtype)
            return low_value.expand(self.num_vehicles).clone()

        def _prepare_candidate_buffers(self):
            torch = self._torch
            num_slots = int(self.expanded_actions_t.shape[0])
            self.expanded_feasibility.fill_(True)
            if self.expanded_action_mask.any():
                self.expanded_feasibility[:, self.expanded_action_mask] = self.vehicle_action_matrix.index_select(1, self.expanded_action_indices) != 0

            dummy_value = self._fallback_dummy_value()
            if self.expanded_action_mask.any():
                expanded_q = self.batch_q_value.index_select(1, self.expanded_action_indices)
            else:
                expanded_q = self.batch_q_value.new_empty((self.num_vehicles, 0))

            candidate_width = self._candidate_width(num_slots)
            if candidate_width != self._cached_candidate_width:
                self.candidate_slots = torch.empty((self.num_vehicles, candidate_width), dtype=torch.long, device=self._device)
                self.candidate_q = torch.empty((self.num_vehicles, candidate_width), dtype=self.batch_q_value.dtype, device=self._device)
                self.candidate_is_unlimited = torch.empty((self.num_vehicles, candidate_width), dtype=torch.bool, device=self._device)
                self._cached_candidate_width = candidate_width

            if candidate_width == num_slots:
                self.candidate_slots.copy_(self.all_slot_indices.unsqueeze(0).expand(self.num_vehicles, -1))
                if self.expanded_action_mask.any():
                    self.candidate_q[:, self.expanded_action_mask] = expanded_q
                if (~self.expanded_action_mask).any():
                    self.candidate_q[:, ~self.expanded_action_mask] = dummy_value.unsqueeze(1)
            else:
                neg_inf = torch.full((), -torch.inf, dtype=self.batch_q_value.dtype, device=self._device)
                full_q = torch.empty((self.num_vehicles, num_slots), dtype=self.batch_q_value.dtype, device=self._device)
                if self.expanded_action_mask.any():
                    full_q[:, self.expanded_action_mask] = expanded_q
                if (~self.expanded_action_mask).any():
                    full_q[:, ~self.expanded_action_mask] = dummy_value.unsqueeze(1)
                masked_values = torch.where(self.expanded_feasibility, full_q, neg_inf)
                candidate_slots = torch.topk(masked_values, k=candidate_width, dim=1).indices
                self.candidate_slots.copy_(candidate_slots)
                self.candidate_q.copy_(full_q.gather(1, candidate_slots))

            self.candidate_is_unlimited.copy_(self.slot_is_unlimited[self.candidate_slots])

        def _reset_auction_state(self):
            self.prices.zero_()
            self.owner.fill_(-1)
            self.assigned_slot.fill_(-1)

        def _resolve_sparse_winners(self, bid_slots, bid_values, bid_vehicles, num_slots):
            torch = self._torch
            if bid_slots.numel() == 0:
                empty_slots = torch.empty(0, dtype=torch.long, device=self._device)
                empty_values = torch.empty(0, dtype=bid_values.dtype, device=self._device)
                return empty_slots, empty_slots, empty_values

            winner_bid = torch.full((num_slots,), -torch.inf, dtype=bid_values.dtype, device=self._device)
            winner_bid.scatter_reduce_(0, bid_slots, bid_values, reduce="amax", include_self=True)
            matching_bids = bid_values == winner_bid[bid_slots]
            encoded_vehicles = torch.where(matching_bids, bid_vehicles + 1, torch.zeros_like(bid_vehicles))
            winner_vehicle_plus = torch.zeros((num_slots,), dtype=torch.long, device=self._device)
            winner_vehicle_plus.scatter_reduce_(0, bid_slots, encoded_vehicles, reduce="amax", include_self=True)
            winning_slots = torch.nonzero(winner_vehicle_plus > 0, as_tuple=False).reshape(-1)
            return winning_slots, winner_vehicle_plus[winning_slots] - 1, winner_bid[winning_slots]

        def solve(self):
            if not self.available:
                raise RuntimeError(self.unavailable_reason or "Torch CUDA auction is unavailable")

            if self.num_vehicles == 0:
                self.converged = True
                return {}

            torch = self._torch
            with torch.no_grad():
                self.batch_q_value = self.batch_q_value.detach()
                self.vehicle_action_matrix = self.vehicle_action_matrix.detach()
                self._prepare_candidate_buffers()
                self._reset_auction_state()
                self.converged = False
                prices = self.prices
                owner = self.owner
                assigned_slot = self.assigned_slot
                active = torch.ones(self.num_vehicles, dtype=torch.bool, device=self._device)
                vehicle_indices = torch.arange(self.num_vehicles, dtype=torch.long, device=self._device)
                epsilon = torch.tensor(self.epsilon, dtype=self.batch_q_value.dtype, device=self._device)
                num_slots = int(self.prices.shape[0])

                for round_idx in range(self.max_rounds):
                    if round_idx % self.sync_interval == 0 and not bool(active.any().item()):
                        self.iterations = round_idx
                        self.converged = True
                        break

                    candidate_prices = prices[self.candidate_slots]
                    net_values = self.candidate_q - candidate_prices
                    net_values = torch.where(active.unsqueeze(1), net_values, -torch.inf)
                    candidate_width = int(net_values.shape[1])
                    if candidate_width > 1:
                        top2_values, top2_indices = torch.topk(net_values, k=2, dim=1)
                        best_values = top2_values[:, 0]
                        best_choice = top2_indices[:, 0]
                        second_values = top2_values[:, 1]
                    else:
                        best_values = net_values[:, 0]
                        best_choice = torch.zeros(self.num_vehicles, dtype=torch.long, device=self._device)
                        second_values = torch.full_like(best_values, -torch.inf)

                    bid_slots = self.candidate_slots.gather(1, best_choice.unsqueeze(1)).squeeze(1)
                    bid_unlimited = self.candidate_is_unlimited.gather(1, best_choice.unsqueeze(1)).squeeze(1)
                    second_values = torch.where(torch.isfinite(second_values), second_values, best_values - epsilon)
                    bid_values = prices[bid_slots] + best_values - second_values + epsilon

                    valid_bid = active & torch.isfinite(best_values)
                    active[active & ~valid_bid] = False

                    unlimited_winner = valid_bid & bid_unlimited
                    unlimited_vehicles = vehicle_indices[unlimited_winner]
                    if unlimited_vehicles.numel() > 0:
                        previous_slots = assigned_slot[unlimited_vehicles]
                        previous_valid = previous_slots >= 0
                        previous_slots_valid = previous_slots[previous_valid]
                        if previous_slots_valid.numel() > 0:
                            finite_previous = previous_slots_valid[~self.slot_is_unlimited[previous_slots_valid]]
                            if finite_previous.numel() > 0:
                                owner[finite_previous] = -1
                        assigned_slot[unlimited_vehicles] = bid_slots[unlimited_vehicles]
                        active[unlimited_vehicles] = False

                    finite_bid_mask = valid_bid & ~bid_unlimited
                    winning_slots, winning_vehicles, winning_bids = self._resolve_sparse_winners(
                        bid_slots[finite_bid_mask],
                        bid_values[finite_bid_mask],
                        vehicle_indices[finite_bid_mask],
                        num_slots,
                    )
                    if winning_slots.numel() > 0:
                        previous_slots = assigned_slot[winning_vehicles]
                        previous_valid = previous_slots >= 0
                        previous_slots_valid = previous_slots[previous_valid]
                        if previous_slots_valid.numel() > 0:
                            finite_previous = previous_slots_valid[~self.slot_is_unlimited[previous_slots_valid]]
                            if finite_previous.numel() > 0:
                                owner[finite_previous] = -1

                        old_owners = owner[winning_slots].clone()
                        owner[winning_slots] = winning_vehicles
                        assigned_slot[winning_vehicles] = winning_slots
                        prices[winning_slots] = winning_bids
                        active[winning_vehicles] = False

                        displaced = (old_owners >= 0) & (old_owners != winning_vehicles)
                        displaced_vehicles = old_owners[displaced]
                        if displaced_vehicles.numel() > 0:
                            assigned_slot[displaced_vehicles] = -1
                            active[displaced_vehicles] = True
                else:
                    self.iterations = self.max_rounds

                self._gpu_prices = prices
                self._gpu_owner = owner
                self._gpu_active = active
                self.assigned_slot = assigned_slot.detach().cpu().numpy().astype(np.int64)
                if not self.converged:
                    self.iterations = self.max_rounds
                    self.converged = not bool(active.any().item())
                return self.get_assigned_actions()





    class Auction_gpu_parallel_solver(Auction_solver):
        def __init__(
            self, vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value,
            action_capacities=None, epsilon=1e-3, max_rounds=None,
            add_dummy_slots=True, dummy_values=None, top_k=None,
            sync_interval=32,
        ):
            super().__init__(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value,
                action_capacities=action_capacities, epsilon=epsilon,
                max_iterations=max_rounds, add_dummy_slots=add_dummy_slots,
                dummy_values=dummy_values, top_k=top_k,
            )
            self.max_rounds = self.max_iterations
            self.sync_interval = max(1, int(sync_interval))
            self.available = False
            self._cp = None
            self.unavailable_reason = ""
            try:
                import cupy as cp  # type: ignore[import-not-found]
                self._cp = cp
                device_count = cp.cuda.runtime.getDeviceCount()
                self.available = device_count > 0
                if not self.available:
                    self.unavailable_reason = "CuPy sees 0 CUDA devices"
            except Exception as exc:
                self.available = False
                self.unavailable_reason = f"{type(exc).__name__}: {exc}"

        def _resolve_sparse_winners(self, bid_slots, bid_values, bid_vehicles):
            cp = self._cp
            if bid_slots.size == 0:
                empty_slots = cp.empty(0, dtype=cp.int32)
                empty_vehicles = cp.empty(0, dtype=cp.int32)
                empty_values = cp.empty(0, dtype=cp.float32)
                return empty_slots, empty_vehicles, empty_values

            order = cp.lexsort((bid_vehicles, bid_values, bid_slots))
            sorted_slots = bid_slots[order]
            sorted_values = bid_values[order]
            sorted_vehicles = bid_vehicles[order]
            unique_slots, first_idx, counts = cp.unique(
                sorted_slots,
                return_index=True,
                return_counts=True,
            )
            winner_pos = first_idx + counts - 1
            return unique_slots, sorted_vehicles[winner_pos], sorted_values[winner_pos]

        def solve(self):
            if not self.available:
                return super().solve()

            self._build_expanded_problem()
            if self.num_vehicles == 0:
                self.converged = True
                return {}

            cp = self._cp
            q_value = cp.asarray(self.expanded_q_value, dtype=cp.float32)
            feasibility = cp.asarray(self.expanded_feasibility, dtype=cp.bool_)
            slot_is_unlimited = cp.asarray(self.slot_is_unlimited, dtype=cp.bool_)
            prices = cp.zeros(q_value.shape[1], dtype=cp.float32)
            owner = cp.full(q_value.shape[1], -1, dtype=cp.int32)
            assigned_slot = cp.full(self.num_vehicles, -1, dtype=cp.int32)
            active = cp.ones(self.num_vehicles, dtype=cp.bool_)
            vehicle_indices = cp.arange(self.num_vehicles, dtype=cp.int32)
            slot_indices = cp.arange(q_value.shape[1], dtype=cp.int32)
            neg_inf = cp.float32(-cp.inf)
            epsilon = cp.float32(self.epsilon)
            num_slots = q_value.shape[1]

            for round_idx in range(self.max_rounds):
                if round_idx % self.sync_interval == 0 and not bool(cp.any(active).get()):
                    self.iterations = round_idx
                    self.converged = True
                    break

                # Kernel 1: each active vehicle finds best and second-best action on GPU.
                net_values = q_value - prices[cp.newaxis, :]
                net_values = cp.where(feasibility & active[:, cp.newaxis], net_values, neg_inf)
                if num_slots > 1:
                    top2_idx = cp.argpartition(net_values, num_slots - 2, axis=1)[:, -2:]
                    top2_values = cp.take_along_axis(net_values, top2_idx, axis=1)
                    best_local = cp.argmax(top2_values, axis=1).astype(cp.int32)
                    second_local = 1 - best_local
                    bid_slots = top2_idx[vehicle_indices, best_local].astype(cp.int32)
                    best_values = top2_values[vehicle_indices, best_local]
                    second_values = top2_values[vehicle_indices, second_local]
                else:
                    bid_slots = cp.zeros(self.num_vehicles, dtype=cp.int32)
                    best_values = net_values[:, 0]
                    second_values = cp.full(self.num_vehicles, neg_inf, dtype=cp.float32)
                second_values = cp.where(cp.isfinite(second_values), second_values, best_values - epsilon)
                bid_values = prices[bid_slots] + best_values - second_values + epsilon

                valid_bid = active & cp.isfinite(best_values)
                inactive_without_bid = active & ~valid_bid
                active[inactive_without_bid] = False

                # Kernel 3a: unlimited slots (zone/wait/dummy) accept every active bidder without owner contention.
                unlimited_winner = valid_bid & slot_is_unlimited[bid_slots]
                unlimited_vehicles = vehicle_indices[unlimited_winner]
                if unlimited_vehicles.size:
                    previous_slots = assigned_slot[unlimited_vehicles]
                    previous_valid = previous_slots >= 0
                    previous_slots_valid = previous_slots[previous_valid]
                    finite_previous = previous_slots_valid[~slot_is_unlimited[previous_slots_valid]]
                    if finite_previous.size:
                        owner[finite_previous] = -1
                    assigned_slot[unlimited_vehicles] = bid_slots[unlimited_vehicles]
                    active[unlimited_vehicles] = False

                # Kernel 3b: finite-capacity slots resolve only the submitted bids instead of building a dense matrix.
                finite_bid_mask = valid_bid & ~slot_is_unlimited[bid_slots]
                winning_slots, winning_vehicles, winning_bids = self._resolve_sparse_winners(
                    bid_slots[finite_bid_mask],
                    bid_values[finite_bid_mask],
                    vehicle_indices[finite_bid_mask],
                )
                if winning_slots.size:
                    previous_slots = assigned_slot[winning_vehicles]
                    previous_valid = previous_slots >= 0
                    previous_slots_valid = previous_slots[previous_valid]
                    finite_previous = previous_slots_valid[~slot_is_unlimited[previous_slots_valid]]
                    if finite_previous.size:
                        owner[finite_previous] = -1

                    old_owners = owner[winning_slots]
                    owner[winning_slots] = winning_vehicles
                    assigned_slot[winning_vehicles] = winning_slots
                    prices[winning_slots] = winning_bids
                    active[winning_vehicles] = False

                    displaced = (old_owners >= 0) & (old_owners != winning_vehicles)
                    displaced_vehicles = old_owners[displaced]
                    if displaced_vehicles.size:
                        assigned_slot[displaced_vehicles] = -1
                        active[displaced_vehicles] = True
            else:
                self.iterations = self.max_rounds

            self._gpu_prices = prices
            self._gpu_owner = owner
            self._gpu_active = active
            self.assigned_slot = cp.asnumpy(assigned_slot).astype(np.int64)
            if not self.converged:
                self.iterations = self.max_rounds
                self.converged = not bool(cp.any(active).get())
            return self.get_assigned_actions()

    def _build_auction_action_capacities(self, vehicle_ids, available_requests, vehicle_action_matrix, ev_only=False):
        num_vehicles = len(vehicle_ids)
        num_action = vehicle_action_matrix.shape[1]
        num_requests = len(available_requests)
        num_charging = 0 if ev_only else getattr(self.env, 'num_stations', 0)
        capacities = np.full(num_action, num_vehicles, dtype=np.int64)
        capacities[:num_requests] = 1

        if not ev_only:
            charging_stations_list = (
                list(self.env.charging_manager.stations.values())
                if hasattr(self.env, 'charging_manager') else []
            )
            for k in range(num_charging):
                action_idx = num_requests + k
                if action_idx >= num_action:
                    break
                vacancy = 0
                if k < len(charging_stations_list):
                    station = charging_stations_list[k]
                    vacancy = max(0, station.max_capacity - len(station.current_vehicles))
                capacities[action_idx] = vacancy

        return capacities

    def _decode_auction_assignments(self, action_by_vehicle, vehicle_ids, available_requests, num_charging=None, num_zones=None):
        assignments = {}
        num_requests = len(available_requests)
        num_charging = getattr(self.env, 'num_stations', 0) if num_charging is None else num_charging
        num_zones = len(getattr(self.env, 'hotspot_locations', [])) if num_zones is None else num_zones
        charging_stations_list = (
            list(self.env.charging_manager.stations.values())
            if hasattr(self.env, 'charging_manager') else []
        )

        for vehicle_id in vehicle_ids:
            action_idx = action_by_vehicle.get(vehicle_id)
            if action_idx is None:
                assignments[vehicle_id] = "reloc" if self.env.vehicles[vehicle_id]['type'] == 1 else "waiting"
            elif action_idx < num_requests:
                assignments[vehicle_id] = available_requests[action_idx]
            elif action_idx < num_requests + num_charging:
                station_idx = action_idx - num_requests
                if station_idx < len(charging_stations_list):
                    assignments[vehicle_id] = f"charge_{charging_stations_list[station_idx].id}"
                else:
                    assignments[vehicle_id] = "waiting"
            elif action_idx < num_requests + num_charging + num_zones:
                zone_idx = action_idx - num_requests - num_charging
                assignments[vehicle_id] = "reloc" if self.env.vehicles[vehicle_id]['type'] == 1 else f"idle_at_{zone_idx}"
            else:
                assignments[vehicle_id] = "reloc" if self.env.vehicles[vehicle_id]['type'] == 1 else "waiting"

        return assignments

    def _should_use_gpu_auction(self):
        explicit = getattr(self.env, 'auction_use_gpu', None)
        if explicit is None:
            explicit = getattr(self.env, 'mcmf_use_gpu', None)
        return bool(explicit)

    def _should_use_torch_cuda_auction(self, vehicle_action_matrix, batch_q_value):
        if not bool(getattr(self.env, 'ifsolveauctioncuda', False)):
            return False
        try:
            import torch
        except Exception:
            return False
        if not torch.cuda.is_available():
            return False
        # The torch CUDA auction path can accept numpy inputs; callers move them
        # onto CUDA before constructing Auction_gpu_solver_cuda.
        return True

    def _prepare_torch_cuda_auction_inputs(self, vehicle_action_matrix, batch_q_value):
        import torch

        device = None
        if isinstance(batch_q_value, torch.Tensor) and batch_q_value.is_cuda:
            device = batch_q_value.device
        elif isinstance(vehicle_action_matrix, torch.Tensor) and vehicle_action_matrix.is_cuda:
            device = vehicle_action_matrix.device
        else:
            device = torch.device("cuda")

        if isinstance(vehicle_action_matrix, torch.Tensor):
            auction_action_matrix = vehicle_action_matrix.to(device=device).clone()
        else:
            auction_action_matrix = torch.as_tensor(vehicle_action_matrix, device=device)

        if isinstance(batch_q_value, torch.Tensor):
            auction_q_value = batch_q_value.to(device=device, dtype=torch.float32)
        else:
            auction_q_value = torch.as_tensor(batch_q_value, device=device, dtype=torch.float32)

        return auction_action_matrix, auction_q_value

    def _to_cpu_auction_array(self, value):
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    def _build_auction_dummy_values(self, vehicle_ids, batch_q_value, ev_only=False):
        if self._should_use_torch_cuda_auction(None, batch_q_value):
            import torch

            tensor = batch_q_value if isinstance(batch_q_value, torch.Tensor) else torch.as_tensor(batch_q_value)
            tensor = tensor.to(dtype=torch.float32)
            if tensor.numel() > 0:
                min_value = tensor.min()
                span = tensor.max() - min_value
                low_value = min_value - torch.maximum(
                    torch.maximum(span, min_value.abs()),
                    torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype),
                ) - 1.0
            else:
                low_value = torch.tensor(-1.0, device=tensor.device, dtype=tensor.dtype)

            dummy_values = low_value.expand(len(vehicle_ids)).clone()
            for i, vehicle_id in enumerate(vehicle_ids):
                if ev_only or self.env.vehicles[vehicle_id]['type'] == 1:
                    dummy_values[i] = 0.0
            return dummy_values

        feasible_values = np.asarray(batch_q_value, dtype=np.float64)
        if feasible_values.size:
            min_value = float(np.min(feasible_values))
            span = float(np.max(feasible_values) - min_value)
            low_value = min_value - max(span, abs(min_value), 1.0) - 1.0
        else:
            low_value = -1.0

        dummy_values = np.full(len(vehicle_ids), low_value, dtype=np.float64)
        for i, vehicle_id in enumerate(vehicle_ids):
            if ev_only or self.env.vehicles[vehicle_id]['type'] == 1:
                dummy_values[i] = 0.0
        return dummy_values

    def _create_auction_solver(self, vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, action_capacities):
        epsilon = float(getattr(self.env, 'auction_epsilon', 1e-3))
        max_rounds = getattr(self.env, 'auction_max_rounds', None)
        top_k = getattr(self.env, 'auction_top_k', None)
        dummy_values = getattr(self.env, '_auction_dummy_values', None)
        sync_interval = getattr(self.env, 'auction_gpu_sync_interval', 32)
        if self._should_use_torch_cuda_auction(vehicle_action_matrix, batch_q_value):
            solver = self._auction_solver_cache.get('torch_cuda')
            if not isinstance(solver, GurobiOptimizer.Auction_gpu_solver_cuda):
                solver = GurobiOptimizer.Auction_gpu_solver_cuda(
                    vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value,
                    action_capacities=action_capacities, epsilon=epsilon, max_rounds=max_rounds,
                    dummy_values=dummy_values, top_k=top_k, sync_interval=sync_interval,
                )
                self._auction_solver_cache['torch_cuda'] = solver
            else:
                solver.refresh_inputs(
                    vehicle_ids,
                    available_requests,
                    vehicle_action_matrix,
                    batch_q_value,
                    action_capacities=action_capacities,
                    dummy_values=dummy_values,
                    top_k=top_k,
                    epsilon=epsilon,
                    max_rounds=max_rounds,
                    sync_interval=sync_interval,
                )
        elif self._should_use_gpu_auction():
            solver = GurobiOptimizer.Auction_gpu_parallel_solver(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value,
                action_capacities=action_capacities, epsilon=epsilon, max_rounds=max_rounds,
                dummy_values=dummy_values, top_k=top_k, sync_interval=sync_interval,
            )
        else:
            solver = GurobiOptimizer.Auction_solver(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value,
                action_capacities=action_capacities, epsilon=epsilon, max_iterations=max_rounds,
                dummy_values=dummy_values, top_k=top_k,
            )
        if isinstance(solver, (GurobiOptimizer.Auction_gpu_parallel_solver, GurobiOptimizer.Auction_gpu_solver_cuda)) and not solver.available:
            reason = getattr(solver, 'unavailable_reason', '')
            suffix = f": {reason}" if reason else ""
            print(f"⚠ CUDA auction requested but unavailable{suffix}, falling back to CPU auction")
            solver = GurobiOptimizer.Auction_solver(
                vehicle_ids,
                available_requests,
                self._to_cpu_auction_array(vehicle_action_matrix),
                self._to_cpu_auction_array(batch_q_value),
                action_capacities=action_capacities,
                epsilon=epsilon,
                max_iterations=max_rounds,
                dummy_values=self._to_cpu_auction_array(dummy_values),
                top_k=top_k,
            )
        return solver

    def _fallback_to_mcmf_after_auction(self, vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=True, ev_only=False):
        missing = object()
        old_useauction = getattr(self.env, 'useauction', missing)
        old_mcmf_solver = getattr(self.env, 'mcmf_solver', missing)

        try:
            self.env.useauction = False
            if old_mcmf_solver is not missing:
                self.env.mcmf_solver = None

            if ev_only:
                return self._np_vehicle_rebalancing_network_ev(
                    vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=iflp)
            return self._np_vehicle_rebalancing_network(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=iflp)
        finally:
            if old_useauction is missing:
                try:
                    delattr(self.env, 'useauction')
                except AttributeError:
                    pass
            else:
                self.env.useauction = old_useauction

            if old_mcmf_solver is missing:
                try:
                    delattr(self.env, 'mcmf_solver')
                except AttributeError:
                    pass
            else:
                self.env.mcmf_solver = old_mcmf_solver

    def _set_auction_dummy_values(self, vehicle_ids, batch_q_value, ev_only=False):
        self.env._auction_dummy_values = self._build_auction_dummy_values(vehicle_ids, batch_q_value, ev_only=ev_only)

    def _clear_auction_dummy_values(self):
        try:
            delattr(self.env, '_auction_dummy_values')
        except AttributeError:
            pass

    def _auction_vehicle_rebalancing_network(self, vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=True):
        if len(vehicle_ids) == 0:
            return {}

        use_torch_cuda = self._should_use_torch_cuda_auction(vehicle_action_matrix, batch_q_value)
        try:
            import torch
        except Exception:
            torch = None

        if use_torch_cuda and torch is not None:
            auction_action_matrix, auction_q_value = self._prepare_torch_cuda_auction_inputs(
                vehicle_action_matrix,
                batch_q_value,
            )
        else:
            auction_action_matrix = np.array(vehicle_action_matrix, copy=True)
            auction_q_value = batch_q_value
        num_requests = len(available_requests)
        for i, vehicle_id in enumerate(vehicle_ids):
            if self.env.vehicles[vehicle_id]['type'] == 1:
                auction_action_matrix[i, num_requests:] = 0

        capacities = self._build_auction_action_capacities(
            vehicle_ids, available_requests, auction_action_matrix, ev_only=False,
        )
        if use_torch_cuda:
            self.env._auction_dummy_values = self._build_auction_dummy_values(vehicle_ids, auction_q_value, ev_only=False)
        else:
            self._set_auction_dummy_values(vehicle_ids, auction_q_value, ev_only=False)
        solver = self._create_auction_solver(
            vehicle_ids, available_requests, auction_action_matrix, auction_q_value, capacities,
        )
        self._clear_auction_dummy_values()

        import time as _time
        t0 = _time.time()
        action_by_vehicle = solver.solve()
        elapsed = _time.time() - t0
        if hasattr(self.env, 'record_time') and self.env.record_time:
            self.env.time_stats.setdefault('auction_solve', []).append(elapsed)

        if not solver.converged:
            if getattr(self.env, 'auction_no_fallback', False):
                raise RuntimeError(f"Auction did not converge after {solver.iterations} iterations")
            print(f"⚠ Auction did not converge after {solver.iterations} iterations, falling back to MCMF")
            return self._fallback_to_mcmf_after_auction(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=iflp, ev_only=False)

        num_zones = len(getattr(self.env, 'hotspot_locations', []))
        return self._decode_auction_assignments(
            action_by_vehicle, vehicle_ids, available_requests,
            num_charging=getattr(self.env, 'num_stations', 0), num_zones=num_zones,
        )

    def _auction_vehicle_rebalancing_network_ev(self, vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=True):
        if len(vehicle_ids) == 0:
            return {}

        use_torch_cuda = self._should_use_torch_cuda_auction(vehicle_action_matrix, batch_q_value)
        if use_torch_cuda:
            auction_action_matrix, auction_q_value = self._prepare_torch_cuda_auction_inputs(
                vehicle_action_matrix,
                batch_q_value,
            )
        else:
            auction_action_matrix = vehicle_action_matrix
            auction_q_value = batch_q_value

        capacities = self._build_auction_action_capacities(
            vehicle_ids, available_requests, auction_action_matrix, ev_only=True,
        )
        if use_torch_cuda:
            self.env._auction_dummy_values = self._build_auction_dummy_values(vehicle_ids, auction_q_value, ev_only=True)
        else:
            self._set_auction_dummy_values(vehicle_ids, auction_q_value, ev_only=True)
        solver = self._create_auction_solver(
            vehicle_ids, available_requests, auction_action_matrix, auction_q_value, capacities,
        )
        self._clear_auction_dummy_values()

        import time as _time
        t0 = _time.time()
        action_by_vehicle = solver.solve()
        elapsed = _time.time() - t0
        if hasattr(self.env, 'record_time') and self.env.record_time:
            self.env.time_stats.setdefault('auction_solve', []).append(elapsed)

        if not solver.converged:
            if getattr(self.env, 'auction_no_fallback', False):
                raise RuntimeError(f"Auction EV did not converge after {solver.iterations} iterations")
            print(f"⚠ Auction EV did not converge after {solver.iterations} iterations, falling back to MCMF")
            return self._fallback_to_mcmf_after_auction(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=iflp, ev_only=True)

        assignments = {}
        num_requests = len(available_requests)
        for vehicle_id in vehicle_ids:
            action_idx = action_by_vehicle.get(vehicle_id)
            if action_idx is not None and action_idx < num_requests:
                assignments[vehicle_id] = available_requests[action_idx]
            else:
                assignments[vehicle_id] = "reloc"
        return assignments





    class _MCMFSolver:
        """SPFA-based min-cost max-flow solver (Successive Shortest Paths).

        Graph representation uses adjacency list with explicit reverse-edge
        index, following the minflow.py convention.

        Complexity: O(V * E * F) where F = max_flow ≤ N (num vehicles).
        """

        def __init__(self, num_nodes: int):
            self.num_nodes = num_nodes
            self.graph = [[] for _ in range(num_nodes)]

        def add_edge(self, u: int, v: int, capacity: int, cost: float):
            self.graph[u].append([v, capacity, cost, len(self.graph[v])])
            self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

        def spfa(self, s: int, t: int, deadline: float = None):
            from collections import deque
            dist = np.full(self.num_nodes, np.inf)
            parent_node = [-1] * self.num_nodes
            parent_edge = [-1] * self.num_nodes
            in_queue = np.zeros(self.num_nodes, dtype=bool)

            q = deque([s])
            dist[s] = 0.0
            in_queue[s] = True

            _iters = 0
            while q:
                _iters += 1
                if deadline is not None and _iters % 500 == 0:
                    import time as _t
                    if _t.time() > deadline:
                        raise TimeoutError("MCMF SPFA exceeded time limit")
                u = q.popleft()
                in_queue[u] = False
                for idx, edge in enumerate(self.graph[u]):
                    v, cap, cost, _ = edge
                    if cap > 0 and dist[u] + cost < dist[v] - 1e-6:
                        dist[v] = dist[u] + cost
                        parent_node[v] = u
                        parent_edge[v] = idx
                        if not in_queue[v]:
                            q.append(v)
                            in_queue[v] = True

            return dist[t] != np.inf, parent_node, parent_edge

        def solve(self, s: int, t: int, time_limit: float = None):
            import time as _t
            deadline = _t.time() + time_limit if time_limit is not None else None
            max_flow = 0
            min_cost = 0.0

            while True:
                if deadline is not None and _t.time() > deadline:
                    raise TimeoutError("MCMF solve exceeded time limit")
                has_path, parent_node, parent_edge = self.spfa(s, t, deadline=deadline)
                if not has_path:
                    break

                # bottleneck
                push = float('inf')
                curr = t
                while curr != s:
                    p = parent_node[curr]
                    idx = parent_edge[curr]
                    push = min(push, self.graph[p][idx][1])
                    curr = p

                # augment
                curr = t
                while curr != s:
                    p = parent_node[curr]
                    idx = parent_edge[curr]
                    rev_idx = self.graph[p][idx][3]
                    self.graph[p][idx][1] -= push
                    self.graph[curr][rev_idx][1] += push
                    min_cost += push * self.graph[p][idx][2]
                    curr = p

                max_flow += push

            return max_flow, min_cost

    def _should_use_gpu_mcmf(self):
        explicit = getattr(self.env, 'mcmf_use_gpu', None)
        if explicit is None:
            explicit = getattr(self.env, 'use_cuda_ssp', None)
        return bool(explicit)

    def _create_mcmf_solver(self, num_nodes: int):
        if self._should_use_gpu_mcmf():
            gpu_solver = GurobiOptimizer._MCMFsolver_gpu(num_nodes)
            if gpu_solver.available:
                return gpu_solver
            print("⚠ CUDA SSP requested but unavailable, falling back to CPU MCMF")
        return GurobiOptimizer._MCMFSolver(num_nodes)

    # ── AEV+EV mixed rebalancing (mirrors _gurobi_vehicle_rebalancing_network) ──

    def _np_vehicle_rebalancing_network(
        self, vehicle_ids, available_requests,
        vehicle_action_matrix, batch_q_value, iflp=True,
    ):
        """MCMF-based vehicle rebalancing — drop-in replacement for the Gurobi
        ILP/LP version.  Each action (request/charging/zone/wait) is a
        separate node so the solver can globally optimise all assignments.

        Node layout:
            Source(0) → Vehicles(1..N) → Actions(N+1..N+A) → Sink(N+A+1)

        Constraints encoded via edge capacities:
          • request→sink cap=1       (each request served once)
          • charging→sink cap=vacancy (station capacity limit)
          • zone/wait→sink cap=N     (unlimited)
          • EV vehicles: only request edges (a < num_requests)
        """
        assignments = {}
        if getattr(self.env, 'useauction', False) or getattr(self.env, 'mcmf_solver', None) == 'auction':
            return self._auction_vehicle_rebalancing_network(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=iflp)

        if self._exact_mcmf_selection() is not None:
            try:
                return self._exact_vehicle_rebalancing_network(
                    vehicle_ids,
                    available_requests,
                    vehicle_action_matrix,
                    batch_q_value,
                    ev_only=False,
                )
            except Exception as error:
                if bool(getattr(self.env, 'mcmf_strict', self.mcmf_strict)):
                    raise
                self._record_online_mcmf_fallback(error)
                print('⚠ Exact MCMF failed in online mode; falling back to legacy MCMF')
                previous_solver = getattr(self.env, 'mcmf_solver', None)
                try:
                    self.env.mcmf_solver = 'legacy'
                    return self._np_vehicle_rebalancing_network(
                        vehicle_ids,
                        available_requests,
                        vehicle_action_matrix,
                        batch_q_value,
                        iflp=iflp,
                    )
                finally:
                    self.env.mcmf_solver = previous_solver

        if len(vehicle_ids) == 0:
            return assignments

        num_vehicles = len(vehicle_ids)
        num_action = vehicle_action_matrix.shape[1]
        layout = self._get_matrix_action_layout(available_requests, num_action)
        num_requests = layout['num_requests']
        scale_charge_station_ids = layout['charge_station_ids']
        scale_zone_indices = layout['zone_indices']
        matrix_requests = layout['requests']
        num_charging = layout['num_charging']
        num_zones = layout['num_zones']

        charging_stations_list = (
            [self.env.charging_manager.stations[sid] for sid in scale_charge_station_ids if sid in self.env.charging_manager.stations]
            if hasattr(self.env, 'charging_manager') else []
        )

        source = 0
        sink = num_vehicles + num_action + 1
        mcmf = self._create_mcmf_solver(sink + 1)

        # Source → Vehicles
        for i in range(num_vehicles):
            mcmf.add_edge(source, i + 1, capacity=1, cost=0.0)

        # Vehicles → Actions
        for i, v_id in enumerate(vehicle_ids):
            is_ev = self.env.vehicles[v_id]['type'] == 1
            for a in range(num_action):
                # EVs cannot take AEV charge/zone actions in the mixed graph,
                # but they still need their wait/reloc column as an opt-out
                # action. Otherwise max-flow forces EVs to consume request
                # capacity while AEVs are pushed to non-request actions.
                if is_ev and a >= num_requests and a != num_action - 1:
                    continue
                if vehicle_action_matrix[i, a] != 0:
                    q_value = float(batch_q_value[i, a])
                    mcmf.add_edge(i + 1, num_vehicles + 1 + a,
                                  capacity=1, cost=-q_value)

        # Actions → Sink
        for a in range(num_action):
            a_node = num_vehicles + 1 + a
            if a < num_requests:
                mcmf.add_edge(a_node, sink, capacity=1, cost=0.0)
            elif a < num_requests + num_charging:
                k = a - num_requests
                vacancy = 0
                if k < len(charging_stations_list):
                    station = charging_stations_list[k]
                    vacancy = self._charging_station_vacancy(station)
                if vacancy > 0:
                    mcmf.add_edge(a_node, sink, capacity=vacancy, cost=0.0)
            else:
                mcmf.add_edge(a_node, sink, capacity=num_vehicles, cost=0.0)

        # Solve
        import time as _time
        t0 = _time.time()
        try:
            mcmf.solve(source, sink, time_limit=60.0)
        except TimeoutError:
            t1 = _time.time()
            print(f"⚠️  MCMF solve timed out after {t1 - t0:.1f}s")
            if hasattr(self.env, 'record_time') and self.env.record_time:
                self.env.time_stats.setdefault('mcmf_solve', []).append(t1 - t0)
            if self._gurobi_runtime_failed or not self.available:
                print("⚠ Gurobi unavailable; falling back to Q-value heuristic")
                return self._qvalue_heuristic_fallback(
                    vehicle_ids,
                    vehicle_action_matrix,
                    batch_q_value,
                )
            print("⚠ Falling back to Gurobi")
            return self._gurobi_vehicle_rebalancing_network(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=iflp)
        t1 = _time.time()
        if hasattr(self.env, 'record_time') and self.env.record_time:
            self.env.time_stats.setdefault('mcmf_solve', []).append(t1 - t0)

        # Decode assignments
        for i, vehicle_id in enumerate(vehicle_ids):
            v_node = i + 1
            assigned = False
            for edge in mcmf.graph[v_node]:
                v, cap, cost, _ = edge
                if cap == 0 and v > num_vehicles and v != sink:
                    a = v - num_vehicles - 1
                    if a < num_requests:
                        request = matrix_requests[a] if a < len(matrix_requests) else None
                        if request is None:
                            assignments[vehicle_id] = "reloc" if self.env.vehicles[vehicle_id]['type'] == 1 else "waiting"
                        else:
                            assignments[vehicle_id] = request
                    elif a < num_requests + num_charging:
                        station_idx = a - num_requests
                        if station_idx < len(charging_stations_list):
                            assignments[vehicle_id] = f"charge_{charging_stations_list[station_idx].id}"
                        else:
                            assignments[vehicle_id] = "waiting"
                    elif a < num_requests + num_charging + num_zones:
                        zone_idx = a - num_requests - num_charging
                        if self.env.vehicles[vehicle_id]['type'] == 1:
                            assignments[vehicle_id] = "reloc"
                        else:
                            mapped_zone_idx = scale_zone_indices[zone_idx] if zone_idx < len(scale_zone_indices) else zone_idx
                            assignments[vehicle_id] = f"idle_at_{mapped_zone_idx}"
                    else:
                        if self.env.vehicles[vehicle_id]['type'] == 1:
                            assignments[vehicle_id] = "reloc"
                        else:
                            assignments[vehicle_id] = "waiting"
                    assigned = True
                    break

            if not assigned:
                if self.env.vehicles[vehicle_id]['type'] == 1:
                    assignments[vehicle_id] = "reloc"
                else:
                    assignments[vehicle_id] = "waiting"

        return assignments

    # ── EV-only rebalancing (mirrors _gurobi_vehicle_rebalancing_network_ev) ──

    def _np_vehicle_rebalancing_network_ev(
        self, vehicle_ids, available_requests,
        vehicle_action_matrix, batch_q_value, iflp=True,
    ):
        """MCMF-based EV-only rebalancing.

        Node layout:
            Source(0) → Vehicles(1..N) → Actions(N+1..N+A) → Sink(N+A+1)
        EV vehicles can only be assigned to requests or fall back to reloc.
        """
        assignments = {}
        if getattr(self.env, 'useauction', False) or getattr(self.env, 'mcmf_solver', None) == 'auction':
            return self._auction_vehicle_rebalancing_network_ev(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=iflp)

        if self._exact_mcmf_selection() is not None:
            try:
                return self._exact_vehicle_rebalancing_network(
                    vehicle_ids,
                    available_requests,
                    vehicle_action_matrix,
                    batch_q_value,
                    ev_only=True,
                )
            except Exception as error:
                if bool(getattr(self.env, 'mcmf_strict', self.mcmf_strict)):
                    raise
                self._record_online_mcmf_fallback(error)
                print('⚠ Exact EV MCMF failed in online mode; falling back to legacy MCMF')
                previous_solver = getattr(self.env, 'mcmf_solver', None)
                try:
                    self.env.mcmf_solver = 'legacy'
                    return self._np_vehicle_rebalancing_network_ev(
                        vehicle_ids,
                        available_requests,
                        vehicle_action_matrix,
                        batch_q_value,
                        iflp=iflp,
                    )
                finally:
                    self.env.mcmf_solver = previous_solver

        if len(vehicle_ids) == 0:
            return assignments

        num_vehicles = len(vehicle_ids)
        num_action = vehicle_action_matrix.shape[1]
        layout = self._get_matrix_action_layout(available_requests, num_action)
        num_requests = layout['num_requests']
        matrix_requests = layout['requests']

        source = 0
        sink = num_vehicles + num_action + 1
        mcmf = self._create_mcmf_solver(sink + 1)

        # Source → Vehicles
        for i in range(num_vehicles):
            mcmf.add_edge(source, i + 1, capacity=1, cost=0.0)

        # Vehicles → Actions
        for i in range(num_vehicles):
            for a in range(num_action):
                if vehicle_action_matrix[i, a] != 0:
                    q_value = float(batch_q_value[i, a])
                    mcmf.add_edge(i + 1, num_vehicles + 1 + a,
                                  capacity=1, cost=-q_value)

        # Actions → Sink
        for a in range(num_action):
            a_node = num_vehicles + 1 + a
            if a < num_requests:
                mcmf.add_edge(a_node, sink, capacity=1, cost=0.0)
            else:
                mcmf.add_edge(a_node, sink, capacity=num_vehicles, cost=0.0)

        # Solve
        import time as _time
        t0 = _time.time()
        try:
            mcmf.solve(source, sink, time_limit=60.0)
        except TimeoutError:
            t1 = _time.time()
            print(f"⚠️  MCMF EV solve timed out after {t1 - t0:.1f}s")
            if hasattr(self.env, 'record_time') and self.env.record_time:
                self.env.time_stats.setdefault('mcmf_solve', []).append(t1 - t0)
            if self._gurobi_runtime_failed or not self.available:
                print("⚠ Gurobi unavailable; falling back to Q-value heuristic")
                return self._qvalue_heuristic_fallback(
                    vehicle_ids,
                    vehicle_action_matrix,
                    batch_q_value,
                )
            print("⚠ Falling back to Gurobi")
            return self._gurobi_vehicle_rebalancing_network_ev(
                vehicle_ids, available_requests, vehicle_action_matrix, batch_q_value, iflp=iflp)
        t1 = _time.time()
        if hasattr(self.env, 'record_time') and self.env.record_time:
            self.env.time_stats.setdefault('mcmf_solve', []).append(t1 - t0)

        # Decode
        for i, vehicle_id in enumerate(vehicle_ids):
            v_node = i + 1
            assigned = False
            for edge in mcmf.graph[v_node]:
                v, cap, cost, _ = edge
                if cap == 0 and v > num_vehicles and v != sink:
                    a = v - num_vehicles - 1
                    if a < num_requests:
                        request = matrix_requests[a] if a < len(matrix_requests) else None
                        assignments[vehicle_id] = request if request is not None else "reloc"
                    else:
                        assignments[vehicle_id] = "reloc"
                    assigned = True
                    break

            if not assigned:
                if self.env.vehicles[vehicle_id]['type'] == 1:
                    assignments[vehicle_id] = "reloc"
                else:
                    assignments[vehicle_id] = "waiting"

        return assignments


    
    
    def _gurobi_vehicle_rebalancing_integrated(self, vehicle_ids, available_requests, charging_stations=None):
        """
        Gurobi optimization with known reject behavior for EVs and charging level constraints
        EVs won't be assigned to requests they would reject
        Includes t-1 to t charging level progression with minimum battery requirements
        """
        #print("current time:,", self.env.current_time, "rebalance vehicle:", len(vehicle_ids), "available requests:", len(available_requests))
        if not self.available:
            return {}
        
        assignments = {}
        charging_stations = charging_stations or []

        if not vehicle_ids:
            return assignments

        vehicle_action_matrix, num_requests, num_stations, num_zones = self.env.generate_whole_matrix(
            vehicle_ids,
            rebalance_num=len(vehicle_ids),
        )
        request_feasibility = vehicle_action_matrix[:, :num_requests]
        charging_feasibility = vehicle_action_matrix[:, num_requests:num_requests + num_stations]
        zone_feasibility = vehicle_action_matrix[:, num_requests + num_stations:num_requests + num_stations + num_zones]
        wait_feasibility = vehicle_action_matrix[:, -1]
        scale_charge_station_ids = list(getattr(self.env, '_last_matrix_charge_station_ids', []))[:num_stations]
        scale_zone_indices = list(getattr(self.env, '_last_matrix_zone_indices', []))[:num_zones]
        scale_zone_targets = list(getattr(self.env, '_last_matrix_zone_target_ids', []))[:num_zones]
        charging_stations = [self.env.charging_manager.stations[sid] for sid in scale_charge_station_ids if sid in self.env.charging_manager.stations]
        
        # Create optimization model
        model = self.gp.Model("vehicle_assignment_with_reject_and_charging")
        model.setParam('OutputFlag', 0)  # Suppress output
        model.setParam('TimeLimit', self.network_time_limit)
        model.setParam('Threads', self.num_threads)

        # Aggregate stats for opportunity costs (optional)
        active_requests_count = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
        active_requests_value = sum(getattr(req, 'final_value', getattr(req, 'value', 0.0)) for req in (self.env.active_requests.values() if hasattr(self.env, 'active_requests') else []))
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 0.0

        # Parameters
        min_battery_level = self.env.min_battery_level
        battery_consum = self.env.battery_consum
        service_consumption = 0.05 # Battery consumption per service
        request_decision =[[model.addVar(vtype=self.GRB.BINARY,
                     name=f'request_{vehicle_id}_{request.request_id}') for request in available_requests] for i, vehicle_id in enumerate(vehicle_ids)]

        reloc_decision  = [[model.addVar(vtype=self.GRB.BINARY,
                 name=f'reloc_{vehicle_id}_{scale_zone_indices[j]}') for j in range(num_zones)] for i, vehicle_id in enumerate(vehicle_ids)]
        
        
        for i in range(len(vehicle_ids)):
            if self.env.vehicles[vehicle_ids[i]]['type'] == 1: # EV
                for j in range(num_zones):
                    model.addConstr(reloc_decision[i][j] == 0)
        





        charge_decision = {}
        if charging_stations:
            for i, vehicle_id in enumerate(vehicle_ids):
                for j, station in enumerate(charging_stations):
                    charge_decision[i, j] = model.addVar(
                        vtype=self.GRB.BINARY,
                        name=f'charge_{vehicle_id}_{station.id}'
                    )
            
        # Battery level variables (t-1 and t)
        battery_t_minus_1 = {}  # Battery level at t-1 (current)
        battery_t = {}          # Battery level at t (after actions)
        
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # t-1 battery level (current battery level)
            battery_t_minus_1[i] = vehicle['battery']
            
            # t battery level (decision variable)
            battery_t[i] = model.addVar(
                vtype=self.GRB.CONTINUOUS,

                name=f'battery_t_{vehicle_id}'
            )
        


        waiting_vehicle = {}
        for i in range(len(vehicle_ids)):
            waiting_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_waiting'
            )
            # Battery level transition constraints (t-1 to t relationship)
            
            
        for i, vehicle_id in enumerate(vehicle_ids):
            if self.env.vehicles[vehicle_id]['type'] == 1: # EV
                model.addConstr(waiting_vehicle[i]== 0)
                for j in range(len(charging_stations)):
                    model.addConstr(charge_decision[i, j] == 0)

        for i, vehicle_id in enumerate(vehicle_ids):
            for j in range(len(available_requests)):
                if request_feasibility[i, j] == 0:
                    model.addConstr(request_decision[i][j] == 0)

            for j, station in enumerate(charging_stations):
                if j >= charging_feasibility.shape[1] or charging_feasibility[i, j] == 0:
                    model.addConstr(charge_decision[i, j] == 0)

            for j in range(num_zones):
                if zone_feasibility[i, j] == 0:
                    model.addConstr(reloc_decision[i][j] == 0)

            if wait_feasibility[i] == 0:
                model.addConstr(waiting_vehicle[i] == 0)
                
        rebalance_num = len(vehicle_ids)
        carindex =  self.env.findchargerange_c(rebalance_num = rebalance_num)
        positivenum = 500
        
        carindex_aev = []
        for i, vehicle_id in enumerate(vehicle_ids):
            carindex_aev.append(carindex[vehicle_id])
        #print("carindex_aev:", carindex_aev)
        for i in range(len(vehicle_ids)):
            model.addConstr(carindex_aev[i]<=positivenum*(1 - waiting_vehicle[i]))



        for i, vehicle_id in enumerate(vehicle_ids):
            reloc = self.gp.LinExpr()
            for j in range(num_zones):
                reloc += reloc_decision[i][j]
            model.addConstr(reloc <= 1 )  # If waiting,


        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Initialize battery expressions as Gurobi LinExpr
            battery_loss = self.gp.LinExpr()
            battery_increase = self.gp.LinExpr()
            veh_loc = vehicle['location']
            # Battery consumption from charging (travel to station)
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    # Convert station location index to coordinates
                    travel_distance = self.env._manhattan_distance_loc(veh_loc, station.location)
                    battery_loss += travel_distance * battery_consum * charge_decision[i, j]
                    battery_increase +=  self.env.chargeincrease_whole*charge_decision[i, j]
            
            # Battery consumption from service requests (travel to pickup + pickup to dropoff)
            if available_requests:
                for j, request in enumerate(available_requests):

                    travel_distance_to_pickup =self.env._manhattan_distance_loc(veh_loc, request.pickup)
                    travel_distance_pickup_to_dropoff = self._request_trip_distance(request)
                    
                    # Total battery consumption for this request
                    total_travel_distance = travel_distance_to_pickup + travel_distance_pickup_to_dropoff
                    battery_loss += total_travel_distance * battery_consum * request_decision[i][j]
            for j, zone_location in enumerate(scale_zone_targets):
                distance1 = self.env._manhattan_distance_loc(veh_loc, zone_location)
                # 检查距离是否在阈值内，且电池足够
                battery_loss += distance1 * self.env.battery_consum* reloc_decision[i][j]
                

            # battery_loss+=idle_vehicle[i]*2*battery_consum # idle consumption
            # Battery transition constraint (simplified to avoid infeasibility)
            model.addConstr(battery_t[i] == battery_t_minus_1[i] - battery_loss + battery_increase)
            # Ensure vehicle has enough battery for actions (but allow some flexibility)
            model.addConstr(battery_loss <= battery_t_minus_1[i] )  # Allow small battery deficit to avoid infeasibility
            # Ensure battery doesn't go below minimum (but allow some flexibility)
            model.addConstr(battery_t[i] >=min_battery_level-1e-3)  # If not idle, must meet min battery

        idle_carnum = self.gp.LinExpr()
        for i, vehicle_id in enumerate(vehicle_ids):
            for j in range(num_zones):
                idle_carnum += reloc_decision[i][j]

        current_online_vehicles = int(getattr(self.env, 'current_online', 0))
        idle_requirement = int(getattr(self.env, 'idle_vehicle_requirement', 0))

            
            # Constraint 1: Each vehicle can only take one action
        for i, vehicle_id in enumerate(vehicle_ids):
            if self.env.vehicles[vehicle_id]['type'] == 1: # EV
                actionv = self.gp.LinExpr()
                for j in range(len(available_requests)):
                    actionv += request_decision[i][j]
                model.addConstr(actionv <= 1)
            else:
                actionv = self.gp.LinExpr()
                # Add valid request assignments
                for j in range(len(available_requests)):
                    actionv += request_decision[i][j]
                # Add charging assignments
                if charging_stations:
                    for j in range(len(charging_stations)):
                        actionv += charge_decision[i, j]
                model.addConstr(actionv <= 1)
                idle_carnum = self.gp.LinExpr()
                for j in range(num_zones):
                    idle_carnum += reloc_decision[i][j]

                model.addConstr(idle_carnum + actionv + waiting_vehicle[i] == 1)
        

        
        # Constraint 2: Each request can be assigned to at most one vehicle
        for j in range(len(available_requests)):
            valid_vehicles = self.gp.LinExpr()
            for i in range(len(vehicle_ids)):
                valid_vehicles += request_decision[i][j]
            model.addConstr(valid_vehicles <= 1)
            
            # Constraint 3: Each charging station capacity
        if charging_stations:
            for j, station in enumerate(charging_stations):
                model.addConstr(
                    self.gp.quicksum(charge_decision[i, j] for i in range(len(vehicle_ids))) 
                    <= max(
                        0,
                        station.max_capacity - len(station.current_vehicles)
                ))
        objective_terms = self.gp.LinExpr()
        adp_weight = getattr(self.env, 'adp_value', 1.0)
        if getattr(self.env, 'value_function', None) is None and getattr(self.env, 'value_function_ev', None) is None:
            adp_weight = 0.0
        
        # 批量预计算所有vehicle-request对的Q值以提高性能
        option_q_cache = {}
        rejection_adjusted_values = {}  # 存储拒绝感知调整后的价值
        
        if adp_weight > 0:
            # 收集所有需要计算的vehicle-request对
            vehicle_request_pairs_ev = []
            vehicle_request_pairs_aev = []
            vehicle_request_pairs = []
            for i , vehicle_id in enumerate(vehicle_ids):
                for j, request in enumerate(available_requests):
                    vehicle_request_pairs.append((vehicle_id, request))
            for i, vehicle_id in enumerate(vehicle_ids):
                if self.env.vehicles[vehicle_id]['type'] == 1: # EV
                    for j, request in enumerate(available_requests):
                        vehicle_request_pairs_ev.append((vehicle_id, request))
                else:
                    for j, request in enumerate(available_requests):
                        vehicle_request_pairs_aev.append((vehicle_id, request))
            
            
            
            
            # 批量计算Q值和拒绝感知价值
            if hasattr(self.env, 'batch_evaluate_service_options'):
                try:
                    
                    batch_q_values_ev = self.env.batch_evaluate_service_options(vehicle_request_pairs_ev,True)
                    batch_q_values_aev = self.env.batch_evaluate_service_options(vehicle_request_pairs_aev,False)
                    # 批量计算拒绝概率（只对EV）
                    batch_rejection_probs = self._batch_calculate_reject_pro_network(vehicle_request_pairs_ev)
                    
                    for i, (vehicle_id, request) in enumerate(vehicle_request_pairs_ev):
                        q_value = batch_q_values_ev[i] if i < len(batch_q_values_ev) else 0.0
                        rejection_prob = batch_rejection_probs[i] if i < len(batch_rejection_probs) else 0.0
                        
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        
                        # 计算拒绝感知调整价值
                        adjusted_value = self._calculate_rejection_aware_value(
                            vehicle_id, request, q_value, rejection_prob
                        )
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = adjusted_value
                    for i, (vehicle_id, request) in enumerate(vehicle_request_pairs_aev):
                        q_value = batch_q_values_aev[i] if i < len(batch_q_values_aev) else 0.0
                        rejection_prob = 0.0
                        
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        

                except Exception as e:
                    print(f"Batch evaluation failed: {e}, falling back to individual calculations")
            
            # 如果批量计算失败，使用单独计算
            if not option_q_cache:
                # 批量计算拒绝概率（只对EV）
                print("individual calculation fallback")
        
        

        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            request_list = {}
            for j, request in enumerate(available_requests):
                if adp_weight <= 0:
                    vehicle_location = vehicle['location']
                    req_val = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    cur_loc = vehicle['location']
                    d1 = self.env._manhattan_distance_loc(vehicle_location, request.pickup)
                    d2 = self._request_trip_distance(request)
                    moving_cost = self._movement_cost(d1 + d2)
                    immediate = req_val + moving_cost  # 加上移动成本（负值）
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    objective_terms += immediate * request_decision[i][j]
                else:
                    # 使用批量计算的Q值和拒绝感知的调整价值
                    base_q_value = option_q_cache.get((vehicle_id, request.request_id), 0.0)
                    pickupdistance = abs((request.pickup // self.env.grid_size) - vehicle['coordinates'][1]) + abs((request.pickup % self.env.grid_size) - vehicle['coordinates'][0])
                    
                    if base_q_value > 0:
                        request_list[request.request_id] = (request, base_q_value, pickupdistance)
                    
                    objective_terms += base_q_value * adp_weight * request_decision[i][j]
            request_list = dict(sorted(request_list.items(), key=lambda x: x[1][1], reverse=True))
            # if len(request_list) > 0 and vehicle['type'] == 1:
            #     print("sorted request list for vehicle",vehicle_id,":",[(req_id, data[0].final_value, data[1], data[2]) for req_id, data in list(request_list.items())[:25]])
                    #adjusted_value = rejection_adjusted_values.get((vehicle_id, request.request_id), base_q_value)
                # Process charging assignments
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    cur_loc = vehicle['location']
                    d_travel = self.env._manhattan_distance_loc(cur_loc, station.location)
                    moving_cost = self._movement_cost(d_travel)
                    charge_steps = getattr(self.env, 'charge_duration', 2)
                    charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * charge_steps
                    immediate = moving_cost + charging_penalty
                    if adp_weight <= 0:
                        # Immediate charging cost fallback
                        objective_terms += immediate * charge_decision[i, j]
                    else:
                        # Use option-completion Q-value for charging
                        charging_q = 0.0
                        if hasattr(self.env, 'evaluate_charging_option'):
                            try:
                                charging_q = self.env.evaluate_charging_option(vehicle_id, station)
                            except Exception:
                                charging_q = 0.0
                        # print(f"Charging Q-value for vehicle {vehicle_id} at station {station.id}: {charging_q}")
                        objective_terms +=charging_q* adp_weight * charge_decision[i, j]
            
        served_requests = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                served_requests += request_decision[i][j]
        wait_q_penalty = -5e+3
        idle_q_penalty = -5e+3
        for i in range(len(vehicle_ids)):
            # 使用神经网络预测的idle Q值替代固定的idle_vehicle_reward
            vehicle_id = vehicle_ids[i]
            vehicle = self.env.vehicles[vehicle_id]
            wait_q_value = 0    
            if hasattr(self.env, 'evaluate_waiting_option'):
                try:
                    wait_q_value = self.env.evaluate_waiting_option(
                        vehicle_id=vehicle_id,
                    )

                except Exception as e:
                    print(f"Warning: Failed to get waiting Q-value for vehicle {vehicle_id}: {e}")
                    # 使用默认的waiting奖励作为后备
                    wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            else:
                # 如果没有神经网络方法，使用默认奖励
                wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            
            if adp_weight <= 0:
                objective_terms += (-50 ) * waiting_vehicle[i]  # Additional opportunity cost penalty
            else:
                objective_terms += wait_q_value * waiting_vehicle[i]  # Use neural network predicted waiting Q-value
        for i in range(len(vehicle_ids)):
            for j in range(num_zones):
                idle_q_value = 0    
                idle_loc = scale_zone_targets[j]
                vehicle_location = self.env.vehicles[vehicle_ids[i]]['location']
                distance = self.env._manhattan_distance_loc(vehicle_location, idle_loc)
                movingcost = self._movement_cost(distance)
                if hasattr(self.env, 'evaluate_idle_option'):
                    try:
                        idle_q_value = self.env.evaluate_idle_option(
                            vehicle_id=vehicle_ids[i],
                            target_loc=idle_loc
                        )
                    except Exception as e:
                        print(f"Warning: Failed to get idle Q-value for vehicle {vehicle_ids[i]} at location {idle_loc}: {e}")
                        # 使用默认的idle奖励作为后备
                        idle_q_value = getattr(self.env, 'idle_vehicle_reward', -0.1)
                else:
                    # 如果没有神经网络方法，使用默认奖励
                    idle_q_value = getattr(self.env, 'idle_vehicle_reward', -0.1)
                
                if adp_weight <= 0:
                    objective_terms += movingcost * reloc_decision[i][j]  # Immediate moving cost for idling at this location
                else:
                    objective_terms += idle_q_value * adp_weight * reloc_decision[i][j]  # Neural network predicted idle Q-value

        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)

        try:
            model.optimize()
            #print(len(model.getVars()), "variables, ", len(model.getConstrs()), "constraints in Gurobi model.")
            # Extract assignments
            if self._should_extract_solution(model):
                # Print battery level optimization results for debugging

                
                for i, vehicle_id in enumerate(vehicle_ids):
                    # Check request assignments
                    for j, request in enumerate(available_requests):
                        if request_decision[i][j].x > 0.5:
                            assignments[vehicle_id] = request
                            break
                    
                    # Check charging assignments if no request assigned
                    if vehicle_id not in assignments and charging_stations:
                        for j, station in enumerate(charging_stations):
                            if charge_decision[i, j].x > 0.5:
                                assignments[vehicle_id] = f"charge_{station.id}"
                                break
                    if waiting_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"waiting"

                    for j in range(num_zones):
                        if reloc_decision[i][j].x > 0.5:
                            assignments[vehicle_id] = f"idle_at_{scale_zone_indices[j]}"

                    # if idle_vehicle[i].x > 0.1:
                    #     assignments[vehicle_id] = f"idle"
                
                for i, vehicle_id in enumerate(vehicle_ids):
                    # Check request assignments
                    if vehicle_id not in assignments:
                        assignments[vehicle_id] = f"reloc"
                
                
                
                
                # Update vehicle battery levels based on optimization results
                for i, vehicle_id in enumerate(vehicle_ids):
                    self.env.vehicles[vehicle_id]['predicted_battery_t'] = battery_t[i].x
                        
            else:
                print(f"Optimization status: {model.status}")
                for i, vehicle_id in enumerate(vehicle_ids):
                    assignments[vehicle_id] = f"waiting"
                if model.status == self.GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS...")
                    model.computeIIS()
                    print("Infeasible constraints (_gurobi_vehicle_rebalancing_integrated):")
                    for c in model.getConstrs():
                        if c.IISConstr:
                            print(f"  {c.constrName}:")
                            print(f"    Sense: {c.sense}, RHS: {c.RHS}")
                            row = model.getRow(c)
                            terms = []
                            for idx in range(row.size()):
                                var = row.getVar(idx)
                                coeff = row.getCoeff(idx)
                                # 提取vehicle_id（如果变量名包含）
                                var_name_parts = var.VarName.split('_')
                                if 'request' in var.VarName and len(var_name_parts) >= 3:
                                    vehicle_id = var_name_parts[-2]
                                    battery_value = self.env.vehicles[int(vehicle_id)]['battery'] if int(vehicle_id) in self.env.vehicles else 'N/A'
                                    terms.append(f"{coeff:+.2f}*{var.VarName}[battery={battery_value}]")
                                else:
                                    terms.append(f"{coeff:+.2f}*{var.VarName}")
                            expr = " ".join(terms[:15])  # 显示前15项
                            if row.size() > 15:
                                expr += f" ... (+{row.size()-15} more)"
                            print(f"    Expr: {expr}")
        except Exception as e:
            print(f"Gurobi optimization with reject and charging levels failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to heuristic with reject consideration
            assignments = self._heuristic_assignment_with_reject(vehicle_ids, available_requests, charging_stations)
        
        return assignments
    
    
    
    
    
    def _gurobi_vehicle_rebalancing_aev(self, vehicle_ids, available_requests, charging_stations=None):
        """
        Gurobi optimization with known reject behavior for EVs and charging level constraints
        EVs won't be assigned to requests they would reject
        Includes t-1 to t charging level progression with minimum battery requirements
        """
        if not self.available:
            return {}
        
        assignments = {}
        
        # Create optimization model
        model = self.gp.Model("vehicle_assignment_with_reject_and_charging")
        model.setParam('OutputFlag', 0)  # Suppress output

        # Aggregate stats for opportunity costs (optional)
        active_requests_count = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
        active_requests_value = sum(getattr(req, 'final_value', getattr(req, 'value', 0.0)) for req in (self.env.active_requests.values() if hasattr(self.env, 'active_requests') else []))
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 0.0

        # Parameters
        min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

        battery_consum = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05 # Battery consumption per travel step
        service_consumption = 0.05 # Battery consumption per service

            
        vehicle_ids_aev = []
        for vid in vehicle_ids:
            if self.env.vehicles[vid]['type'] == 2:
                vehicle_ids_aev.append(vid)
                
        request_decision =[[model.addVar(vtype=self.GRB.BINARY,
                name=f'request_{vehicle_id}_{request.request_id}') for request in available_requests] for i, vehicle_id in enumerate(vehicle_ids_aev)]
        charge_decision = {}
        if charging_stations:
            for i, vehicle_id in enumerate(vehicle_ids_aev):
                for j, station in enumerate(charging_stations):
                    charge_decision[i, j] = model.addVar(
                        vtype=self.GRB.BINARY,
                        name=f'charge_{vehicle_id}_{station.id}'
                    )
            
        # Battery level variables (t-1 and t)
        battery_t_minus_1 = {}  # Battery level at t-1 (current)
        battery_t = {}          # Battery level at t (after actions)
        
        for i, vehicle_id in enumerate(vehicle_ids_aev):
            vehicle = self.env.vehicles[vehicle_id]
            
            # t-1 battery level (current battery level)
            battery_t_minus_1[i] = vehicle['battery']
            
            # t battery level (decision variable)
            battery_t[i] = model.addVar(
                vtype=self.GRB.CONTINUOUS,

                name=f'battery_t_{vehicle_id}'
            )
        
        idle_vehicle = {}
        for i in range(len(vehicle_ids_aev)):
            idle_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_idle'
            )

        idle_vehicle_assign = {}
        for i in range(len(vehicle_ids_aev)):
            for j in range(self.env.hotspot_locations_num):
                idle_vehicle_assign[i,j] = model.addVar(
                    vtype=self.GRB.BINARY,
                    name=f'vehicle_{vehicle_ids[i]}_idle_assign_{j}'
                )
        for i in range(len(vehicle_ids_aev)):
            model.addConstr(self.gp.quicksum(idle_vehicle_assign[i,j] for j in range(self.env.hotspot_locations_num)) == idle_vehicle[i])
        
        for i in range(len(vehicle_ids_aev)):
            vehicle = self.env.vehicles[vehicle_ids_aev[i]]
            if vehicle['type'] == 1: # EV
                nearest_hotspot_index = self.env.return_nearest_hotspot_index(vehicle_ids[i])
                if nearest_hotspot_index is not None:
                    for j in range(self.env.hotspot_locations_num):
                        if j != nearest_hotspot_index:
                            model.addConstr(idle_vehicle_assign[i,j] == 0)

        waiting_vehicle = {}
        for i in range(len(vehicle_ids_aev)):
            waiting_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_waiting'
            )
            # Battery level transition constraints (t-1 to t relationship)




        rebalance_num = len(vehicle_ids)
        carindex =  self.env.findchargerange_c(rebalance_num = rebalance_num)
        positivenum = 500
        
        carindex_aev = []
        for i, vehicle_id in enumerate(vehicle_ids_aev):
            carindex_aev.append(carindex[vehicle_id])

        for i in range(len(vehicle_ids_aev)):
            model.addConstr(carindex_aev[i]<=positivenum*(1 - waiting_vehicle[i]))

    
        for i, vehicle_id in enumerate(vehicle_ids_aev):
            vehicle = self.env.vehicles[vehicle_id]
            battery_loss = self.gp.LinExpr()
            battery_increase = self.gp.LinExpr()
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    # Convert station location index to coordinates
                    station_x = station.location % self.env.grid_size
                    station_y = station.location // self.env.grid_size
                    travel_distance = abs(vehicle['coordinates'][0] - station_x) + abs(vehicle['coordinates'][1] - station_y)
                    battery_loss += travel_distance * battery_consum * charge_decision[i, j]
                    battery_increase +=  self.env.chargeincrease_whole*charge_decision[i, j]
            if available_requests:
                for j, request in enumerate(available_requests):
                    # Travel from vehicle current position to pickup
                    pickup_x = request.pickup % self.env.grid_size
                    pickup_y = request.pickup // self.env.grid_size
                    travel_distance_to_pickup = abs(vehicle['coordinates'][0] - pickup_x) + abs(vehicle['coordinates'][1] - pickup_y)
                    
                    # Travel from pickup to dropoff
                    dropoff_x = request.dropoff % self.env.grid_size
                    dropoff_y = request.dropoff // self.env.grid_size
                    travel_distance_pickup_to_dropoff = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
                    total_travel_distance = travel_distance_to_pickup + travel_distance_pickup_to_dropoff
                    battery_loss += total_travel_distance * battery_consum * request_decision[i][j]
            veh_loc = vehicle['location']
            if vehicle['type'] == 2:
                for j in range(self.env.hotspot_locations_num):
                    hotspot_loc_x , hotspot_loc_y = self.env.hotspot_locations[j]
                    hotspot_loc = hotspot_loc_x*self.env.grid_size + hotspot_loc_y
                    maximam_idle_distance = self._manhattan_loc(veh_loc, hotspot_loc)
                    battery_loss += maximam_idle_distance * battery_consum * idle_vehicle_assign[i,j]
                    
                    # 添加额外约束：如果电池不足以到达该hotspot，禁止分配
                    required_battery = maximam_idle_distance * battery_consum + min_battery_level
                    if battery_t_minus_1[i] < required_battery:
                        model.addConstr(idle_vehicle_assign[i,j] == 0)
            model.addConstr(battery_t[i] == battery_t_minus_1[i] - battery_loss + battery_increase)
            model.addConstr(battery_t[i] >=min_battery_level)  



        idle_carnum = self.gp.LinExpr()
        for i, vehicle_id in enumerate(vehicle_ids_aev):
            idle_carnum += idle_vehicle[i]

        current_online_vehicles = int(getattr(self.env, 'current_online', 0))
        idle_requirement = int(getattr(self.env, 'idle_vehicle_requirement', 0))

            
            # Constraint 1: Each vehicle can only take one action
        for i in range(len(vehicle_ids_aev)):
            actionv = self.gp.LinExpr()
            # Add valid request assignments
            for j in range(len(available_requests)):
                actionv += request_decision[i][j]
            # Add charging assignments
            if charging_stations:
                for j in range(len(charging_stations)):
                    actionv += charge_decision[i, j]
            model.addConstr(actionv <= 1)
            model.addConstr(idle_vehicle[i] + actionv + waiting_vehicle[i] == 1)
        

        
        # Constraint 2: Each request can be assigned to at most one vehicle
        for j in range(len(available_requests)):
            valid_vehicles = self.gp.LinExpr()
            for i in range(len(vehicle_ids_aev)):
                valid_vehicles += request_decision[i][j]
            model.addConstr(valid_vehicles <= 1)
            
            # Constraint 3: Each charging station capacity
        if charging_stations:
            for j, station in enumerate(charging_stations):
                model.addConstr(
                    self.gp.quicksum(charge_decision[i, j] for i in range(len(vehicle_ids))) 
                    <= max(
                        0,
                        station.max_capacity - len(station.current_vehicles)
                ))
        objective_terms = self.gp.LinExpr()
        adp_weight = getattr(self.env, 'adp_value', 1.0)
        
        # 批量预计算所有vehicle-request对的Q值以提高性能
        option_q_cache = {}
        rejection_adjusted_values = {}  # 存储拒绝感知调整后的价值
        
        if adp_weight > 0:
            # 收集所有需要计算的vehicle-request对
            vehicle_request_pairs = []
            for i, vehicle_id in enumerate(vehicle_ids_aev):
                for j, request in enumerate(available_requests):
                    vehicle_request_pairs.append((vehicle_id, request))
            
            # 批量计算Q值和拒绝感知价值
            if hasattr(self.env, 'batch_evaluate_service_options'):
                try:
                    batch_q_values = self.env.batch_evaluate_service_options(vehicle_request_pairs)
                    
                    # 批量计算拒绝概率（只对EV）
                    batch_rejection_probs = self._batch_calculate_reject_pro_network(vehicle_request_pairs)
                    
                    for i, (vehicle_id, request) in enumerate(vehicle_request_pairs):
                        q_value = batch_q_values[i] if i < len(batch_q_values) else 0.0
                        rejection_prob = batch_rejection_probs[i] if i < len(batch_rejection_probs) else 0.0
                        
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        
                        # 计算拒绝感知调整价值
                        adjusted_value = self._calculate_rejection_aware_value(
                            vehicle_id, request, q_value, rejection_prob
                        )
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = adjusted_value
                except Exception as e:
                    print(f"Batch evaluation failed: {e}, falling back to individual calculations")
            
            # 如果批量计算失败，使用单独计算
            if not option_q_cache:
                # 批量计算拒绝概率（只对EV）
                batch_rejection_probs = self._batch_calculate_reject_pro_network(vehicle_request_pairs)
                
                for i, (vehicle_id, request) in enumerate(vehicle_request_pairs):
                    try:
                        q_value = self.env.evaluate_service_option(vehicle_id, request, False)
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        
                        # 使用批量计算的拒绝概率
                        rejection_prob = batch_rejection_probs[i] if i < len(batch_rejection_probs) else 0.0
                        
                        adjusted_value = self._calculate_rejection_aware_value(
                            vehicle_id, request, q_value, rejection_prob
                        )
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = adjusted_value
                    except Exception:
                        option_q_cache[(vehicle_id, request.request_id)] = 0.0
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = 0.0
            
        for i, vehicle_id in enumerate(vehicle_ids_aev):
            vehicle = self.env.vehicles[vehicle_id]

            for j, request in enumerate(available_requests):
                if adp_weight <= 0:
                    # 回退到基础计算
                    req_val = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    cur_loc = vehicle['location']
                    d1 = self._manhattan_loc(cur_loc, request.pickup)
                    d2 = self._request_trip_distance(request)
                    moving_cost = self._movement_cost(d1 + d2)
                    immediate = req_val + moving_cost
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    objective_terms += immediate* request_decision[i][j]*(1 - rejection_prob)
                else:
                    # 使用批量计算的Q值和拒绝感知的调整价值
                    base_q_value = option_q_cache.get((vehicle_id, request.request_id), 0.0)
                    #adjusted_value = rejection_adjusted_values.get((vehicle_id, request.request_id), base_q_value)
                    objective_terms += base_q_value * adp_weight * request_decision[i][j]
                
                # Process charging assignments
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    cur_loc = vehicle['location']
                    d_travel = self._manhattan_loc(cur_loc, station.location)
                    moving_cost = self._movement_cost(d_travel)
                    charge_steps = getattr(self.env, 'charge_duration', 2)
                    charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * charge_steps
                    immediate = moving_cost + charging_penalty
                    if adp_weight <= 0:
                        # Immediate charging cost fallback
                        objective_terms += immediate * charge_decision[i, j]
                    else:
                        # Use option-completion Q-value for charging
                        charging_q = 0.0
                        if hasattr(self.env, 'evaluate_charging_option'):
                            try:
                                charging_q = self.env.evaluate_charging_option(vehicle_id, station)
                            except Exception:
                                charging_q = 0.0
                        objective_terms += charging_q * adp_weight * charge_decision[i, j]
            
        served_requests = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                served_requests += request_decision[i][j]
        wait_q_penalty = -5e+3
        idle_q_penalty = -5e+3
        for i in range(len(vehicle_ids_aev)):
            # 使用神经网络预测的idle Q值替代固定的idle_vehicle_reward
            vehicle_id = vehicle_ids_aev[i]
            vehicle = self.env.vehicles[vehicle_id]
            wait_q_value = 0    
            if hasattr(self.env, 'evaluate_waiting_option'):
                try:
                    wait_q_value = self.env.evaluate_waiting_option(
                        vehicle_id=vehicle_id,
                    )
                    wait_q_value =wait_q_value*self.env.adp_value
                except Exception as e:
                    print(f"Warning: Failed to get waiting Q-value for vehicle {vehicle_id}: {e}")
                    # 使用默认的waiting奖励作为后备
                    wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            else:
                # 如果没有神经网络方法，使用默认奖励
                wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            
            if adp_weight <= 0:
                objective_terms += (-avg_request_value ) * waiting_vehicle[i]  # Additional opportunity cost penalty
            else:
                objective_terms += (wait_q_value) * waiting_vehicle[i]  # Use neural network predicted waiting Q-value
        for i in range(len(vehicle_ids_aev)):
            for j in range(self.env.hotspot_locations_num):
                idle_q_value = 0    
                idle_loc_x , idle_loc_y = self.env.hotspot_locations[j]
                idle_loc = idle_loc_y * self.env.grid_size + idle_loc_x
                if hasattr(self.env, 'evaluate_idle_option'):
                    try:
                        idle_q_value = self.env.evaluate_idle_option(
                            vehicle_id=vehicle_ids[i],
                            target_loc=idle_loc
                        )
                        idle_q_value =idle_q_value*self.env.adp_value
                    except Exception as e:
                        print(f"Warning: Failed to get idle Q-value for vehicle {vehicle_ids[i]} at location {j}: {e}")
                        # 使用默认的idle奖励作为后备
                        idle_q_value = getattr(self.env, 'idle_vehicle_reward', -0.1)
                else:
                    # 如果没有神经网络方法，使用默认奖励
                    idle_q_value = getattr(self.env, 'idle_vehicle_reward', -0.1)
                
                if adp_weight <= 0:
                    objective_terms += (-avg_request_value ) * idle_vehicle_assign[i,j]  # Additional opportunity cost penalty
                else:
                    objective_terms += (idle_q_value) * idle_vehicle_assign[i,j]  # Use neural network predicted idle Q-value

        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)

        try:
            model.optimize()
            
            # Extract assignments
            if model.status == self.GRB.OPTIMAL:
                # Print battery level optimization results for debugging

                
                for i, vehicle_id in enumerate(vehicle_ids_aev):
                    # Check request assignments
                    for j, request in enumerate(available_requests):
                        if request_decision[i][j].x > 0.5:
                            assignments[vehicle_id] = request
                            break
                    
                    # Check charging assignments if no request assigned
                    if vehicle_id not in assignments and charging_stations:
                        for j, station in enumerate(charging_stations):
                            if charge_decision[i, j].x > 0.5:
                                assignments[vehicle_id] = f"charge_{station.id}"
                                break
                    if waiting_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"waiting"

                    if idle_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"idle"
                
                # Update vehicle battery levels based on optimization results
                for i, vehicle_id in enumerate(vehicle_ids_aev):
                    if hasattr(self.env.vehicles[vehicle_id], 'predicted_battery_t'):
                        self.env.vehicles[vehicle_id]['predicted_battery_t'] = battery_t[i].x
                        
            else:
                print(f"Optimization status: {model.status}")
                for i, vehicle_id in enumerate(vehicle_ids_aev):
                    assignments[vehicle_id] = f"waiting"
                if model.status == self.GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS...")
                    model.computeIIS()
                    print("Infeasible constraints:")
                    print("aev infeasible constraints:")
                    for c in model.getConstrs():
                        if c.IISConstr:
                            # Get constraint details
                            constr_name = c.constrName
                            constr_sense = c.sense
                            constr_rhs = c.RHS
                            
                            # Get the linear expression
                            row = model.getRow(c)
                            expr_str = ""
                            for i in range(row.size()):
                                var = row.getVar(i)
                                coeff = row.getCoeff(i)
                                if i > 0 and coeff >= 0:
                                    expr_str += " + "
                                elif coeff < 0:
                                    expr_str += " - "
                                    coeff = -coeff
                                else:
                                    pass  # First term, no operator needed
                                
                                if coeff != 1.0:
                                    expr_str += f"{coeff}*"
                                expr_str += var.varName
                            
                            print(f"  {constr_name}: {expr_str} {constr_sense} {constr_rhs}")
        except Exception as e:
            print(f"Gurobi optimization with reject and charging levels failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to heuristic with reject consideration
            assignments = self._heuristic_assignment_with_reject(vehicle_ids, available_requests, charging_stations)
        
        return assignments

    
    
    
    
    
    
    
    
    
    
    
    
    
    def _gurobi_vehicle_rebalancing_knownreject(self, vehicle_ids, available_requests, charging_stations=None):
        """
        Gurobi optimization with known reject behavior for EVs and charging level constraints
        EVs won't be assigned to requests they would reject
        Includes t-1 to t charging level progression with minimum battery requirements
        """
        if not self.available:
            return {}
        
        assignments = {}
        
        # Create optimization model
        model = self.gp.Model("vehicle_assignment_with_reject_and_charging")
        model.setParam('OutputFlag', 0)  # Suppress output

        # Aggregate stats for opportunity costs (optional)
        active_requests_count = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
        active_requests_value = sum(getattr(req, 'final_value', getattr(req, 'value', 0.0)) for req in (self.env.active_requests.values() if hasattr(self.env, 'active_requests') else []))
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 0.0

        # Parameters
        min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

        battery_consum = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05 # Battery consumption per travel step
        service_consumption = 0.05 # Battery consumption per service
        request_decision =[[model.addVar(vtype=self.GRB.BINARY,
                     name=f'request_{vehicle_id}_{request.request_id}') for request in available_requests] for i, vehicle_id in enumerate(vehicle_ids)]
            
        # Decision variables for charging assignments
        charge_decision = {}
        if charging_stations:
            for i, vehicle_id in enumerate(vehicle_ids):
                for j, station in enumerate(charging_stations):
                    charge_decision[i, j] = model.addVar(
                        vtype=self.GRB.BINARY,
                        name=f'charge_{vehicle_id}_{station.id}'
                    )
            
        # Battery level variables (t-1 and t)
        battery_t_minus_1 = {}  # Battery level at t-1 (current)
        battery_t = {}          # Battery level at t (after actions)
        
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # t-1 battery level (current battery level)
            battery_t_minus_1[i] = vehicle['battery']
            
            # t battery level (decision variable)
            battery_t[i] = model.addVar(
                vtype=self.GRB.CONTINUOUS,

                name=f'battery_t_{vehicle_id}'
            )
        
        idle_vehicle = {}
        for i in range(len(vehicle_ids)):
            idle_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_idle'
            )

        idle_vehicle_assign = {}
        for i in range(len(vehicle_ids)):
            for j in range(self.env.hotspot_locations_num):
                idle_vehicle_assign[i,j] = model.addVar(
                    vtype=self.GRB.BINARY,
                    name=f'vehicle_{vehicle_ids[i]}_idle_assign_{j}'
                )
        for i in range(len(vehicle_ids)):
            model.addConstr(self.gp.quicksum(idle_vehicle_assign[i,j] for j in range(self.env.hotspot_locations_num)) == idle_vehicle[i])
        
        for i in range(len(vehicle_ids)):
            vehicle = self.env.vehicles[vehicle_ids[i]]
            if vehicle['type'] == 1: # EV
                nearest_hotspot_index = self.env.return_nearest_hotspot_index(vehicle_ids[i])
                if nearest_hotspot_index is not None:
                    for j in range(self.env.hotspot_locations_num):
                        if j != nearest_hotspot_index:
                            model.addConstr(idle_vehicle_assign[i,j] == 0)

        waiting_vehicle = {}
        for i in range(len(vehicle_ids)):
            waiting_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_waiting'
            )
            # Battery level transition constraints (t-1 to t relationship)
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Initialize battery expressions as Gurobi LinExpr
            battery_loss = self.gp.LinExpr()
            battery_increase = self.gp.LinExpr()
            
            # Battery consumption from charging (travel to station)
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    # Convert station location index to coordinates
                    station_x = station.location % self.env.grid_size
                    station_y = station.location // self.env.grid_size
                    travel_distance = abs(vehicle['coordinates'][0] - station_x) + abs(vehicle['coordinates'][1] - station_y)
                    battery_loss += travel_distance * battery_consum * charge_decision[i, j]
                    battery_increase +=  self.env.chargeincrease_whole*charge_decision[i, j]
            
            # Battery consumption from service requests (travel to pickup + pickup to dropoff)
            if available_requests:
                for j, request in enumerate(available_requests):
                    # Travel from vehicle current position to pickup
                    pickup_x = request.pickup % self.env.grid_size
                    pickup_y = request.pickup // self.env.grid_size
                    travel_distance_to_pickup = abs(vehicle['coordinates'][0] - pickup_x) + abs(vehicle['coordinates'][1] - pickup_y)
                    
                    # Travel from pickup to dropoff
                    dropoff_x = request.dropoff % self.env.grid_size
                    dropoff_y = request.dropoff // self.env.grid_size
                    travel_distance_pickup_to_dropoff = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
                    
                    # Total battery consumption for this request
                    total_travel_distance = travel_distance_to_pickup + travel_distance_pickup_to_dropoff
                    battery_loss += total_travel_distance * battery_consum * request_decision[i][j]
            veh_loc = vehicle['location']
            if vehicle['type'] == 1: # EV
                nearest_hotspot_index = self.env.return_nearest_hotspot_index(vehicle_id)
                if nearest_hotspot_index is not None:
                    hotspot_loc_x , hotspot_loc_y = self.env.hotspot_locations[nearest_hotspot_index]
                    hotspot_loc = hotspot_loc_x*self.env.grid_size + hotspot_loc_y
                    travel_distance_to_hotspot = self._manhattan_loc(veh_loc, hotspot_loc)
                    battery_loss += travel_distance_to_hotspot * battery_consum * idle_vehicle_assign[i,nearest_hotspot_index]
                else:
                    for j in range(self.env.hotspot_locations_num):
                        hotspot_loc_x , hotspot_loc_y = self.env.hotspot_locations[j]
                        hotspot_loc = hotspot_loc_x*self.env.grid_size + hotspot_loc_y
                        maximam_idle_distance = self._manhattan_loc(veh_loc, hotspot_loc)
                        battery_loss += maximam_idle_distance * battery_consum * idle_vehicle_assign[i,j]
            # battery_loss+=idle_vehicle[i]*2*battery_consum # idle consumption
            # Battery transition constraint (simplified to avoid infeasibility)
            model.addConstr(battery_t[i] == battery_t_minus_1[i] - battery_loss + battery_increase)
            # Ensure vehicle has enough battery for actions (but allow some flexibility)
            model.addConstr(battery_loss <= battery_t_minus_1[i] )  # Allow small battery deficit to avoid infeasibility
            # Ensure battery doesn't go below minimum (but allow some flexibility)
            model.addConstr(battery_t[i] >=min_battery_level)  # If not idle, must meet min battery

        idle_carnum = self.gp.LinExpr()
        for i, vehicle_id in enumerate(vehicle_ids):
            idle_carnum += idle_vehicle[i]

        current_online_vehicles = int(getattr(self.env, 'current_online', 0))
        idle_requirement = int(getattr(self.env, 'idle_vehicle_requirement', 0))

            
            # Constraint 1: Each vehicle can only take one action
        for i in range(len(vehicle_ids)):
            actionv = self.gp.LinExpr()
            # Add valid request assignments
            for j in range(len(available_requests)):
                actionv += request_decision[i][j]
            # Add charging assignments
            if charging_stations:
                for j in range(len(charging_stations)):
                    actionv += charge_decision[i, j]
            model.addConstr(actionv <= 1)
            model.addConstr(idle_vehicle[i] + actionv + waiting_vehicle[i] == 1)
        

        
        # Constraint 2: Each request can be assigned to at most one vehicle
        for j in range(len(available_requests)):
            valid_vehicles = self.gp.LinExpr()
            for i in range(len(vehicle_ids)):
                valid_vehicles += request_decision[i][j]
            model.addConstr(valid_vehicles <= 1)
            
            # Constraint 3: Each charging station capacity
        if charging_stations:
            for j, station in enumerate(charging_stations):
                model.addConstr(
                    self.gp.quicksum(charge_decision[i, j] for i in range(len(vehicle_ids))) 
                    <= max(
                        0,
                        station.max_capacity - len(station.current_vehicles)
                ))
        objective_terms = self.gp.LinExpr()
        adp_weight = getattr(self.env, 'adp_value', 1.0)
        
        # 批量预计算所有vehicle-request对的Q值以提高性能
        option_q_cache = {}
        rejection_adjusted_values = {}  # 存储拒绝感知调整后的价值
        
        if adp_weight > 0:
            # 收集所有需要计算的vehicle-request对
            vehicle_request_pairs = []
            for i, vehicle_id in enumerate(vehicle_ids):
                for j, request in enumerate(available_requests):
                    vehicle_request_pairs.append((vehicle_id, request))
            
            # 批量计算Q值和拒绝感知价值
            if hasattr(self.env, 'batch_evaluate_service_options'):
                try:
                    batch_q_values = self.env.batch_evaluate_service_options(vehicle_request_pairs)
                    
                    # 批量计算拒绝概率（只对EV）
                    batch_rejection_probs = self._batch_calculate_reject_pro_network(vehicle_request_pairs)
                    
                    for i, (vehicle_id, request) in enumerate(vehicle_request_pairs):
                        q_value = batch_q_values[i] if i < len(batch_q_values) else 0.0
                        rejection_prob = batch_rejection_probs[i] if i < len(batch_rejection_probs) else 0.0
                        
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        
                        # 计算拒绝感知调整价值
                        adjusted_value = self._calculate_rejection_aware_value(
                            vehicle_id, request, q_value, rejection_prob
                        )
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = adjusted_value
                except Exception as e:
                    print(f"Batch evaluation failed: {e}, falling back to individual calculations")
            
            # 如果批量计算失败，使用单独计算
            if not option_q_cache:
                # 批量计算拒绝概率（只对EV）
                batch_rejection_probs = self._batch_calculate_reject_pro_network(vehicle_request_pairs)
                
                for i, (vehicle_id, request) in enumerate(vehicle_request_pairs):
                    try:
                        q_value = self.env.evaluate_service_option(vehicle_id, request)
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        
                        # 使用批量计算的拒绝概率
                        rejection_prob = batch_rejection_probs[i] if i < len(batch_rejection_probs) else 0.0
                        
                        adjusted_value = self._calculate_rejection_aware_value(
                            vehicle_id, request, q_value, rejection_prob
                        )
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = adjusted_value
                    except Exception:
                        option_q_cache[(vehicle_id, request.request_id)] = 0.0
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = 0.0
            
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]

            for j, request in enumerate(available_requests):
                if adp_weight <= 0:
                    # 回退到基础计算
                    req_val = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    cur_loc = vehicle['location']
                    d1 = self._manhattan_loc(cur_loc, request.pickup)
                    d2 = self._request_trip_distance(request)
                    moving_cost = self._movement_cost(d1 + d2)
                    immediate = req_val + moving_cost
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    objective_terms += immediate* request_decision[i][j]*(1 - rejection_prob)
                else:
                    # 使用批量计算的Q值和拒绝感知的调整价值
                    base_q_value = option_q_cache.get((vehicle_id, request.request_id), 0.0)
                    #adjusted_value = rejection_adjusted_values.get((vehicle_id, request.request_id), base_q_value)
                    objective_terms += base_q_value * adp_weight * request_decision[i][j]
                
                # Process charging assignments
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    cur_loc = vehicle['location']
                    d_travel = self._manhattan_loc(cur_loc, station.location)
                    moving_cost = self._movement_cost(d_travel)
                    charge_steps = getattr(self.env, 'charge_duration', 2)
                    charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * charge_steps
                    immediate = moving_cost + charging_penalty
                    if adp_weight <= 0:
                        # Immediate charging cost fallback
                        objective_terms += immediate * charge_decision[i, j]
                    else:
                        # Use option-completion Q-value for charging
                        charging_q = 0.0
                        if hasattr(self.env, 'evaluate_charging_option'):
                            try:
                                charging_q = self.env.evaluate_charging_option(vehicle_id, station)
                            except Exception:
                                charging_q = 0.0
                        objective_terms += charging_q * adp_weight * charge_decision[i, j]
            
        served_requests = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                served_requests += request_decision[i][j]
        wait_q_penalty = -5e+3
        idle_q_penalty = -5e+3
        for i in range(len(vehicle_ids)):
            # 使用神经网络预测的idle Q值替代固定的idle_vehicle_reward
            vehicle_id = vehicle_ids[i]
            vehicle = self.env.vehicles[vehicle_id]
            wait_q_value = 0    
            if hasattr(self.env, 'evaluate_waiting_option'):
                try:
                    wait_q_value = self.env.evaluate_waiting_option(
                        vehicle_id=vehicle_id,
                    )
                    wait_q_value =wait_q_value*self.env.adp_value
                except Exception as e:
                    print(f"Warning: Failed to get waiting Q-value for vehicle {vehicle_id}: {e}")
                    # 使用默认的waiting奖励作为后备
                    wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            else:
                # 如果没有神经网络方法，使用默认奖励
                wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            
            if adp_weight <= 0:
                objective_terms += (-avg_request_value ) * waiting_vehicle[i]  # Additional opportunity cost penalty
            else:
                objective_terms += (wait_q_value) * waiting_vehicle[i]  # Use neural network predicted waiting Q-value
        for i in range(len(vehicle_ids)):
            for j in range(self.env.hotspot_locations_num):
                idle_q_value = 0    
                idle_loc_x , idle_loc_y = self.env.hotspot_locations[j]
                idle_loc = idle_loc_y * self.env.grid_size + idle_loc_x
                if hasattr(self.env, 'evaluate_idle_option'):
                    try:
                        idle_q_value = self.env.evaluate_idle_option(
                            vehicle_id=vehicle_ids[i],
                            target_loc=idle_loc
                        )
                        idle_q_value =idle_q_value*self.env.adp_value
                    except Exception as e:
                        print(f"Warning: Failed to get idle Q-value for vehicle {vehicle_ids[i]} at location {j}: {e}")
                        # 使用默认的idle奖励作为后备
                        idle_q_value = getattr(self.env, 'idle_vehicle_reward', -0.1)
                else:
                    # 如果没有神经网络方法，使用默认奖励
                    idle_q_value = getattr(self.env, 'idle_vehicle_reward', -0.1)
                
                if adp_weight <= 0:
                    objective_terms += (-avg_request_value ) * idle_vehicle_assign[i,j]  # Additional opportunity cost penalty
                else:
                    objective_terms += (idle_q_value) * idle_vehicle_assign[i,j]  # Use neural network predicted idle Q-value

        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)

        try:
            model.optimize()
            
            # Extract assignments
            if model.status == self.GRB.OPTIMAL:
                # Print battery level optimization results for debugging

                
                for i, vehicle_id in enumerate(vehicle_ids):
                    # Check request assignments
                    for j, request in enumerate(available_requests):
                        if request_decision[i][j].x > 0.5:
                            assignments[vehicle_id] = request
                            break
                    
                    # Check charging assignments if no request assigned
                    if vehicle_id not in assignments and charging_stations:
                        for j, station in enumerate(charging_stations):
                            if charge_decision[i, j].x > 0.5:
                                assignments[vehicle_id] = f"charge_{station.id}"
                                break
                    if waiting_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"waiting"

                    if idle_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"idle"
                
                # Update vehicle battery levels based on optimization results
                for i, vehicle_id in enumerate(vehicle_ids):
                    if hasattr(self.env.vehicles[vehicle_id], 'predicted_battery_t'):
                        self.env.vehicles[vehicle_id]['predicted_battery_t'] = battery_t[i].x
                        
            else:
                print(f"Optimization status: {model.status}")
                for i, vehicle_id in enumerate(vehicle_ids):
                    assignments[vehicle_id] = f"waiting"
                if model.status == self.GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS...")
                    model.computeIIS()
                    print("Infeasible constraints:")
                    for c in model.getConstrs():
                        if c.IISConstr:
                            print(f"  {c.constrName}")
        except Exception as e:
            print(f"Gurobi optimization with reject and charging levels failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to heuristic with reject consideration
            assignments = self._heuristic_assignment_with_reject(vehicle_ids, available_requests, charging_stations)
        
        return assignments





    def _gurobi_vehicle_rebalancing_knownreject_state(self, vehicle_ids, available_requests, charging_stations=None):
        """
        Gurobi optimization with known reject behavior for EVs and charging level constraints
        EVs won't be assigned to requests they would reject
        Includes t-1 to t charging level progression with minimum battery requirements
        """
        if not self.available:
            return {}
        
        assignments = {}
        
        # Create optimization model
        model = self.gp.Model("vehicle_assignment_with_reject_and_charging")
        model.setParam('OutputFlag', 0)  # Suppress output

        # Aggregate stats for opportunity costs (optional)
        active_requests_count = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
        active_requests_value = sum(getattr(req, 'final_value', getattr(req, 'value', 0.0)) for req in (self.env.active_requests.values() if hasattr(self.env, 'active_requests') else []))
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 0.0

        # Parameters
        min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

        battery_consum = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05 # Battery consumption per travel step
        service_consumption = 0.05 # Battery consumption per service
        
        # Filter out rejected requests for each EV
        valid_assignments = {}  # (vehicle_id, request_idx) -> is_valid

        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            for j, request in enumerate(available_requests):
                # Check if EV would reject this request
                if vehicle['type'] == 1:
                    # Calculate rejection probability
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    # If rejection probability is high (>50%), don't allow assignment
                    valid_assignments[(i, j)] = rejection_prob < 0.5
                else:
                    # AEV never rejects
                    valid_assignments[(i, j)] = True
            
    # Decision variables for request assignments
        request_decision =[[model.addVar(vtype=self.GRB.BINARY,
                     name=f'request_{vehicle_id}_{request.request_id}') for request in available_requests] for i, vehicle_id in enumerate(vehicle_ids)]
            
        # Constraint invalid assignments to 0
        for i in range(len(vehicle_ids)):
            for j in range(len(available_requests)):
                if not valid_assignments.get((i, j), False):
                    model.addConstr(request_decision[i][j] == 0)
            
            
        # Decision variables for charging assignments
        charge_decision = {}
        if charging_stations:
            for i, vehicle_id in enumerate(vehicle_ids):
                for j, station in enumerate(charging_stations):
                    charge_decision[i, j] = model.addVar(
                        vtype=self.GRB.BINARY,
                        name=f'charge_{vehicle_id}_{station.id}'
                    )
            
        # Battery level variables (t-1 and t)
        battery_t_minus_1 = {}  # Battery level at t-1 (current)
        battery_t = {}          # Battery level at t (after actions)
        
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # t-1 battery level (current battery level)
            battery_t_minus_1[i] = vehicle['battery']
            
            # t battery level (decision variable)
            battery_t[i] = model.addVar(
                vtype=self.GRB.CONTINUOUS,

                name=f'battery_t_{vehicle_id}'
            )
        
        

        idle_vehicle = {}
        for i in range(len(vehicle_ids)):
            idle_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_idle'
            )
        waiting_vehicle = {}
        for i in range(len(vehicle_ids)):
            waiting_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_waiting'
            )
            # Battery level transition constraints (t-1 to t relationship)
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Initialize battery expressions as Gurobi LinExpr
            battery_loss = self.gp.LinExpr()
            battery_increase = self.gp.LinExpr()
            
            # Battery consumption from charging (travel to station)
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    # Convert station location index to coordinates
                    station_x = station.location % self.env.grid_size
                    station_y = station.location // self.env.grid_size
                    travel_distance = abs(vehicle['coordinates'][0] - station_x) + abs(vehicle['coordinates'][1] - station_y)
                    battery_loss += travel_distance * battery_consum * charge_decision[i, j]
                    battery_increase +=  self.env.chargeincrease_whole*charge_decision[i, j]
            
            # Battery consumption from service requests (travel to pickup + pickup to dropoff)
            if available_requests:
                for j, request in enumerate(available_requests):
                    # Travel from vehicle current position to pickup
                    pickup_x = request.pickup % self.env.grid_size
                    pickup_y = request.pickup // self.env.grid_size
                    travel_distance_to_pickup = abs(vehicle['coordinates'][0] - pickup_x) + abs(vehicle['coordinates'][1] - pickup_y)
                    
                    # Travel from pickup to dropoff
                    dropoff_x = request.dropoff % self.env.grid_size
                    dropoff_y = request.dropoff // self.env.grid_size
                    travel_distance_pickup_to_dropoff = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
                    
                    # Total battery consumption for this request
                    total_travel_distance = travel_distance_to_pickup + travel_distance_pickup_to_dropoff
                    battery_loss += total_travel_distance * battery_consum * request_decision[i][j]
            battery_loss+=idle_vehicle[i]*2*battery_consum # idle consumption
            # Battery transition constraint (simplified to avoid infeasibility)
            model.addConstr(battery_t[i] == battery_t_minus_1[i] - battery_loss + battery_increase)
            # Ensure vehicle has enough battery for actions (but allow some flexibility)
            model.addConstr(battery_loss <= battery_t_minus_1[i] )  # Allow small battery deficit to avoid infeasibility
            # Ensure battery doesn't go below minimum (but allow some flexibility)
            model.addConstr(battery_t[i] >=min_battery_level*(1 - waiting_vehicle[i]))  # If not idle, must meet min battery

            
            
            # Constraint 1: Each vehicle can only take one action
        for i in range(len(vehicle_ids)):
            actionv = self.gp.LinExpr()
            # Add valid request assignments
            for j in range(len(available_requests)):
                actionv += request_decision[i][j]
            # Add charging assignments
            if charging_stations:
                for j in range(len(charging_stations)):
                    actionv += charge_decision[i, j]
            model.addConstr(actionv <= 1)
            model.addConstr(idle_vehicle[i] + actionv + waiting_vehicle[i] == 1)
        
        # Minimum idle vehicles constraint
        idle_vehicles = self.gp.LinExpr()
        for i in range(len(vehicle_ids)):
            idle_vehicles += idle_vehicle[i]
        #model.addConstr(idle_vehicles >= self.env.idle_vehicle_requirement)
        
        # Constraint 2: Each request can be assigned to at most one vehicle
        for j in range(len(available_requests)):
            valid_vehicles = self.gp.LinExpr()
            for i in range(len(vehicle_ids)):
                valid_vehicles += request_decision[i][j]
            model.addConstr(valid_vehicles <= 1)
            
            # Constraint 3: Each charging station capacity
        if charging_stations:
            for j, station in enumerate(charging_stations):
                model.addConstr(
                    self.gp.quicksum(charge_decision[i, j] for i in range(len(vehicle_ids))) 
                    <= max(
                        0,
                        station.max_capacity - len(station.current_vehicles)
                ))
        
        # Objective: Maximize total value considering Q-values
        objective_terms = self.gp.LinExpr()
        adp_weight = getattr(self.env, 'adp_value', 1.0)
            
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Process valid request assignments
            for j, request in enumerate(available_requests):
                if (i, j) in request_decision:
                    req_val = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    cur_loc = vehicle['location']
                    d1 = self._manhattan_loc(cur_loc, request.pickup)
                    d2 = self._request_trip_distance(request)
                    moving_cost = self._movement_cost(d1 + d2)
                    immediate = req_val + moving_cost
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    if adp_weight <= 0:
                        objective_terms += immediate *(1 - rejection_prob)* request_decision[i, j]
                    else:
                        # Use option-completion Q-value for request assignment
                        print(f"Evaluating service option for vehicle {vehicle_id} and request {request}")
                        option_q = self.env.evaluate_service_option(vehicle_id, request)
                        objective_terms += option_q * adp_weight * request_decision[i, j]
                
                # Process charging assignments
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    cur_loc = vehicle['location']
                    d_travel = self._manhattan_loc(cur_loc, station.location)
                    moving_cost = self._movement_cost(d_travel)
                    charge_steps = getattr(self.env, 'charge_duration', 2)
                    charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * charge_steps
                    immediate = moving_cost + charging_penalty
                    if adp_weight <= 0:
                        # Immediate charging cost fallback
                        objective_terms += immediate * charge_decision[i, j]
                    else:
                        # Use option-completion Q-value for charging
                        charging_q = 0.0
                        if hasattr(self.env, 'evaluate_charging_option'):
                            try:
                                charging_q = self.env.evaluate_charging_option(vehicle_id, station)
                            except Exception:
                                charging_q = 0.0
                        objective_terms += charging_q * adp_weight * charge_decision[i, j]
            

            
            # Add penalty for unserved requests (considering reject behavior)
        served_requests = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                served_requests += request_decision[i][j]
        wait_q_penalty = -5e+3
        idle_q_penalty = -5e+3
        for i in range(len(vehicle_ids)):
            # 使用神经网络预测的idle Q值替代固定的idle_vehicle_reward
            vehicle_id = vehicle_ids[i]
            vehicle = self.env.vehicles[vehicle_id]
            
            # 获取神经网络预测的idle Q值
            idle_q_value = 0
            wait_q_value = 0
            current_coords = vehicle['coordinates']
            target_x = max(0, min(self.env.grid_size-1, 
                                current_coords[0] + random.randint(-1, 1)))
            target_y = max(0, min(self.env.grid_size-1, 
                                current_coords[1] + random.randint(-1, 1)))
            target_loc = target_y * self.env.grid_size + target_x            
            if hasattr(self.env, 'evaluate_idle_option'):
                try:
                    idle_q_value = self.env.evaluate_idle_option(
                        vehicle_id=vehicle_id,
                        target_loc = target_loc,
                    )
                    idle_q_value = idle_q_value*self.env.adp_value
                except Exception as e:
                    print(f"Warning: Failed to get idle Q-value for vehicle {vehicle_id}: {e}")
                    # 使用默认的idle奖励作为后备
                    idle_q_value = getattr(self.env, 'idle_vehicle_reward', 0.0)
            else:
                # 如果没有神经网络方法，使用默认奖励
                idle_q_value = getattr(self.env, 'idle_vehicle_reward', 0.0)
            
            # 获取神经网络预测的waiting Q值
            if hasattr(self.env, 'evaluate_waiting_option'):
                try:
                    wait_q_value = self.env.evaluate_waiting_option(
                        vehicle_id=vehicle_id,
                    )
                    wait_q_value = wait_q_value*self.env.adp_value
                except Exception as e:
                    print(f"Warning: Failed to get waiting Q-value for vehicle {vehicle_id}: {e}")
                    # 使用默认的waiting奖励作为后备
                    wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            else:
                # 如果没有神经网络方法，使用默认奖励
                wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            
            if adp_weight <= 0:
                objective_terms += -avg_request_value * idle_vehicle[i]
                objective_terms += -avg_request_value * waiting_vehicle[i]  # Additional opportunity cost penalty
            else:
                objective_terms += idle_q_value * idle_vehicle[i]
                objective_terms += wait_q_value * waiting_vehicle[i]  # Use neural network predicted waiting Q-value

            # Penalty for unserved requests
        unserved_penalty = getattr(self.env, 'unserved_penalty', 1.5)
        # objective_terms -= avg_request_value * (len(available_requests) - served_requests)
        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)
        
        objvalue = []
        try:
            model.optimize()
            
            # Extract assignments
            if model.status == self.GRB.OPTIMAL:
                
                for i in range(len(vehicle_ids)):
                    vehicle_id = vehicle_ids[i]  # Add this line to define vehicle_id
                    vehicle_obj = 0
                    for j in range(len(available_requests)):
                        if request_decision[i][j].x > 0.5:
                            request = available_requests[j]
                            option_q = self.env.evaluate_service_option_state(vehicle_id, request)
                            vehicle_obj += getattr(available_requests[j], 'final_value', getattr(available_requests[j], 'value', 0.0)) + option_q
                        if charging_stations:
                            for k, station in enumerate(charging_stations):
                                if charge_decision[i, k].x > 0.5:
                                    charging_q = self.env.evaluate_charging_option_state(vehicle_id, station)
                                    vehicle_obj += -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)+ charging_q
                        if idle_vehicle[i].x > 0.1:
                            idle_q = self.env.evaluate_idle_option_state(vehicle_id)
                            vehicle_obj += getattr(self.env, 'idle_vehicle_reward', -0.1)+  idle_q
                        if waiting_vehicle[i].x > 0.1:
                            wait_q = self.env.evaluate_waiting_option_state(vehicle_id)
                            vehicle_obj += getattr(self.env, 'waiting_vehicle_reward', -0.1)+ wait_q
                    objvalue.append(vehicle_obj)

                for i, vehicle_id in enumerate(vehicle_ids):
                    # Check request assignments
                    for j, request in enumerate(available_requests):
                        if request_decision[i][j].x > 0.5:
                            assignments[vehicle_id] = request
                            break
                    
                    # Check charging assignments if no request assigned
                    if vehicle_id not in assignments and charging_stations:
                        for j, station in enumerate(charging_stations):
                            if charge_decision[i, j].x > 0.5:
                                assignments[vehicle_id] = f"charge_{station.id}"
                                break
                    if waiting_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"waiting"

                    if idle_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"idle"
                
                # Update vehicle battery levels based on optimization results
                for i, vehicle_id in enumerate(vehicle_ids):
                    if hasattr(self.env.vehicles[vehicle_id], 'predicted_battery_t'):
                        self.env.vehicles[vehicle_id]['predicted_battery_t'] = battery_t[i].x
                        
            else:
                print(f"Optimization status: {model.status}")
                objvalue = [0 for _ in range(len(vehicle_ids))]
                for i, vehicle_id in enumerate(vehicle_ids):
                    assignments[vehicle_id] = f"waiting"
                if model.status == self.GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS...")
                    model.computeIIS()
                    print("Infeasible constraints:")
                    for c in model.getConstrs():
                        if c.IISConstr:
                            print(f"  {c.constrName}")
        except Exception as e:
            print(f"Gurobi optimization with reject and charging levels failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to heuristic with reject consideration
            objvalue = [0 for _ in range(len(vehicle_ids))]
            assignments = self._heuristic_assignment_with_reject(vehicle_ids, available_requests, charging_stations)
        
        return objvalue, assignments





    def _heuristic_assignment_with_reject_previous(self, vehicle_ids, available_requests, charging_stations=None):

        assignments = {}
        battery_threshold = self.env.heuristic_battery_threshold if hasattr(self.env, 'heuristic_battery_threshold') else 0.5
        if not vehicle_ids:
            return assignments
        heuevfirst = self.env.heuevfirst if hasattr(self.env, 'heuevfirst') else False


        assigned_requests = []
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            for vehicle_id in self.env.vehicles.keys():
                if self.env.vehicles[vehicle_id]['assigned_request'] is not None:
                    assigned_requests.append(self.env.vehicles[vehicle_id]['assigned_request'])
                if self.env.vehicles[vehicle_id]['passenger_onboard'] is not None:
                    assigned_requests.append(self.env.vehicles[vehicle_id]['passenger_onboard'])
                

        available_requests = list(self.env.active_requests.values())
        available_requests = [req for req in available_requests if req.request_id not in assigned_requests]


        # 第一步：识别低电量车辆（电池 < 0.5）
        low_battery_vehicles = []
        high_battery_vehicles = []
        
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            if vehicle['battery'] < battery_threshold:
                low_battery_vehicles.append(vehicle_id)
            else:
                high_battery_vehicles.append(vehicle_id)
        
        
        low_battery_vehicles.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery']
        )
        if charging_stations and low_battery_vehicles:
            for vehicle_id in low_battery_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                
                best_station = None
                best_distance = float('inf')
                
                # 找到最近的有容量的充电站
                for station in charging_stations:
                    if len(station.current_vehicles) < station.max_capacity:
                        station_coords = (
                            station.location // self.env.grid_size,
                            station.location % self.env.grid_size
                        )
                        distance = abs(vehicle_coords[0] - station_coords[0]) + \
                                  abs(vehicle_coords[1] - station_coords[1])
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_station = station
                
                if best_station:
                    assignments[vehicle_id] = f"charge_{best_station.id}"
        
        
        high_battery_vehicles.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery'], 
            reverse=True
        )
        

        if available_requests and high_battery_vehicles:
            
            high_battery_vehicles_list = {}
            for vehicle_id in high_battery_vehicles:
                high_battery_vehicles_list[vehicle_id] = self.env.vehicles[vehicle_id]

            remaining_requests = available_requests.copy()
            remaining_requests.sort(
                key=lambda req: getattr(req, 'final_value', getattr(req, 'value', 0.0)),
                reverse=True
            )
            for vehicle_id in high_battery_vehicles:
                if vehicle_id in assignments:  # 已被分配充电
                    continue
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                battery_level = vehicle['battery']
                if vehicle['type'] != 1:
                    for request in remaining_requests:
                        pickup_coords = (
                            request.pickup // self.env.grid_size,
                            request.pickup % self.env.grid_size
                        )

                        distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                                abs(vehicle_coords[1] - pickup_coords[1])
                        dropoff_coords = (
                            request.dropoff // self.env.grid_size,
                            request.dropoff % self.env.grid_size
                        )
                        whole_distance = distance + \
                                        abs(request.pickup // self.env.grid_size - dropoff_coords[0]) + \
                                            abs(request.pickup % self.env.grid_size - dropoff_coords[1])
                        battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                        estimated_consumption = whole_distance * battery_consumption
                        if battery_level - estimated_consumption >= self.env.min_battery_level:
                            assignments[vehicle_id] = request
                            remaining_requests.remove(request)
                            break
                else:
                    
                    # 计算该车辆到所有订单的距离
                    distance_list = {}
                    for request in remaining_requests:
                        pickup_coords = (
                            request.pickup // self.env.grid_size,
                            request.pickup % self.env.grid_size
                        )

                        distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                                abs(vehicle_coords[1] - pickup_coords[1])
                        distance_list[request.request_id] = distance
                    
                    battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                    min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2
                    
                    # 为该车辆寻找最佳订单
                    best_request = None
                    
                    distance_sorted = sorted(distance_list.items(), key=lambda x: x[1])

                    for req_id, distance in distance_sorted:
                        request = next(r for r in remaining_requests if r.request_id == req_id)
                        dropoff_coords = (
                            request.dropoff // self.env.grid_size,
                            request.dropoff % self.env.grid_size
                        )
                        whole_distance = distance + \
                                        abs(request.pickup // self.env.grid_size - dropoff_coords[0]) + \
                                            abs(request.pickup % self.env.grid_size - dropoff_coords[1])
                        estimated_consumption = whole_distance * battery_consumption 
                        if battery_level - estimated_consumption >= min_battery_level:
                            # 检查拒绝概率
                            request = next(r for r in remaining_requests if r.request_id == req_id)
                            best_request = request
                            break
                    if best_request:
                        assignments[vehicle_id] = best_request
                        remaining_requests.remove(best_request)
                        
        for vehicle_id in vehicle_ids:
            if vehicle_id not in assignments:
                if self.env.vehicles[vehicle_id]['type'] == 2:
                    assignments[vehicle_id] = "wait"
                else:
                    assignments[vehicle_id] = "reloc"
        return assignments

    


    def _heuristic_assignment_with_reject(
        self, vehicle_ids, available_requests, charging_stations=None,
        vehicle_action_matrix=None,
    ):
        assignments = {}
        if not vehicle_ids:
            return assignments

        battery_threshold = getattr(self.env, 'heuristic_battery_threshold', 0.5)
        battery_consumption = getattr(self.env, 'battery_consum', 0.05)
        min_battery_level = getattr(self.env, 'min_battery_level', 0.2)
        no_reloc_battery_threshold = float(getattr(self.env, 'no_reloc_battery_threshold', 0.15))
        heuevfirst = getattr(self.env, 'heuevfirst', False)
        battery_first = getattr(self.env, 'battery_first', False)
        charging_stations = list(charging_stations or [])

        def distance_between(source, target):
            if hasattr(self.env, 'get_distance_km'):
                return float(self.env.get_distance_km(source, target))
            if hasattr(self.env, '_manhattan_distance_loc'):
                return float(self.env._manhattan_distance_loc(source, target))
            grid_size = getattr(self.env, 'grid_size', 1)
            sx, sy = source % grid_size, source // grid_size
            tx, ty = target % grid_size, target // grid_size
            return float(abs(sx - tx) + abs(sy - ty))

        assigned_request_ids = set()
        for vehicle in self.env.vehicles.values():
            if vehicle.get('assigned_request') is not None:
                assigned_request_ids.add(vehicle['assigned_request'])
            if vehicle.get('passenger_onboard') is not None:
                assigned_request_ids.add(vehicle['passenger_onboard'])

        active_requests = getattr(self.env, 'active_requests', {})
        if active_requests:
            available_requests = list(active_requests.values())
        else:
            available_requests = list(available_requests or [])
        available_requests = [
            request for request in available_requests
            if request.request_id not in assigned_request_ids
        ]

        if vehicle_action_matrix is not None:
            num_requests = max(0, int(getattr(
                self.env, '_last_matrix_num_requests', len(available_requests))))
            matrix_request_ids = list(getattr(self.env, '_last_matrix_request_ids', []))
            matrix_station_ids = list(getattr(self.env, '_last_matrix_charge_station_ids', []))
        else:
            num_requests = len(available_requests)
            matrix_request_ids = [request.request_id for request in available_requests]
            matrix_station_ids = [station.id for station in charging_stations]
        request_index = {
            request_id: idx
            for idx, request_id in enumerate(matrix_request_ids[:num_requests])
        }
        if not request_index:
            request_index = {
                request.request_id: idx
                for idx, request in enumerate(available_requests[:num_requests])
            }
        station_index = {
            station_id: num_requests + idx
            for idx, station_id in enumerate(matrix_station_ids)
        }
        vehicle_index = {vehicle_id: idx for idx, vehicle_id in enumerate(vehicle_ids)}

        def action_row(vehicle_id):
            if vehicle_action_matrix is None:
                return None
            row_idx = vehicle_index.get(vehicle_id)
            if row_idx is None or row_idx >= vehicle_action_matrix.shape[0]:
                return None
            return vehicle_action_matrix[row_idx]

        def feasible_requests(vehicle_id, requests):
            row = action_row(vehicle_id)
            if row is None:
                return list(requests)
            return [
                request for request in requests
                if request_index.get(request.request_id) is not None
                and request_index[request.request_id] < num_requests
                and row[request_index[request.request_id]] == 1
            ]

        def feasible_stations(vehicle_id):
            row = action_row(vehicle_id)
            if row is None:
                return list(charging_stations)
            return [
                station for station in charging_stations
                if station_index.get(station.id) is not None
                and station_index[station.id] < len(row)
                and row[station_index[station.id]] == 1
            ]

        def request_score(vehicle_id, request):
            score = float(getattr(request, 'final_value', getattr(request, 'value', 0.0)))
            vehicle = self.env.vehicles[vehicle_id]
            if getattr(self.env, 'knownreject', False) and vehicle['type'] == 1:
                reject_probability = self.env._calculate_known_rejection_probability(
                    vehicle_id, request)
                score *= min(1.0, max(0.0, 1.0 - float(reject_probability)))
            return score

        def post_request_reserve(dropoff):
            if hasattr(self.env, '_post_action_battery_reserve'):
                return max(min_battery_level, float(
                    self.env._post_action_battery_reserve(dropoff)))
            return min_battery_level

        def can_serve(vehicle_id, request):
            vehicle = self.env.vehicles[vehicle_id]
            total_distance = (
                distance_between(vehicle['location'], request.pickup)
                + self._request_trip_distance(request)
            )
            remaining_battery = vehicle['battery'] - total_distance * battery_consumption
            return remaining_battery >= post_request_reserve(request.dropoff)

        low_battery_vehicles = []
        service_vehicles = []
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            if vehicle['type'] != 1 and vehicle['battery'] < battery_threshold:
                low_battery_vehicles.append(vehicle_id)
            else:
                service_vehicles.append(vehicle_id)

        station_capacity = {}
        for station in charging_stations:
            available_slots = getattr(station, 'available_slots', None)
            if available_slots is None:
                available_slots = max(
                    0, station.max_capacity - len(station.current_vehicles))
            station_capacity[station.id] = max(0, int(available_slots))

        for vehicle_id in sorted(
            low_battery_vehicles,
            key=lambda vid: self.env.vehicles[vid]['battery'],
        ):
            vehicle_location = self.env.vehicles[vehicle_id]['location']
            candidates = [
                station for station in feasible_stations(vehicle_id)
                if station_capacity.get(station.id, 0) > 0
                and (self.env.vehicles[vehicle_id]['battery']
                     - distance_between(vehicle_location, station.location)
                     * battery_consumption > 0)
            ]
            if not candidates:
                continue
            best_station = min(
                candidates,
                key=lambda station: distance_between(vehicle_location, station.location),
            )
            assignments[vehicle_id] = f"charge_{best_station.id}"
            station_capacity[best_station.id] -= 1

        if battery_first:
            service_vehicles.sort(
                key=lambda vid: self.env.vehicles[vid]['battery'], reverse=True)
        else:
            ev_ids = sorted(
                (vid for vid in service_vehicles if self.env.vehicles[vid]['type'] == 1),
                key=lambda vid: self.env.vehicles[vid].get('salary_ratio', 0.0),
            )
            aev_ids = sorted(
                (vid for vid in service_vehicles if self.env.vehicles[vid]['type'] != 1),
                key=lambda vid: self.env.vehicles[vid]['battery'], reverse=True,
            )
            if heuevfirst:
                service_vehicles = ev_ids + aev_ids
            else:
                # Integrated HEU should not starve one fleet merely because EV and
                # AEV salary_ratio values use different scales.
                service_vehicles = []
                for idx in range(max(len(ev_ids), len(aev_ids))):
                    if idx < len(ev_ids):
                        service_vehicles.append(ev_ids[idx])
                    if idx < len(aev_ids):
                        service_vehicles.append(aev_ids[idx])

        remaining_requests = list(available_requests)
        for vehicle_id in service_vehicles:
            if vehicle_id in assignments:
                continue
            vehicle = self.env.vehicles[vehicle_id]
            candidates = feasible_requests(vehicle_id, remaining_requests)
            if not candidates:
                continue
            if vehicle['type'] == 1 and not getattr(self.env, 'knownreject', False):
                candidates.sort(key=lambda request: distance_between(
                    vehicle['location'], request.pickup))
            else:
                candidates.sort(
                    key=lambda request: (
                        request_score(vehicle_id, request),
                        -distance_between(vehicle['location'], request.pickup),
                    ),
                    reverse=True,
                )
            best_request = next(
                (request for request in candidates if can_serve(vehicle_id, request)),
                None,
            )
            if best_request is not None:
                assignments[vehicle_id] = best_request
                remaining_requests.remove(best_request)

        relocation_targets = list(self._get_relocation_targets() or [])
        target_index = {target: idx for idx, target in enumerate(relocation_targets)}
        request_count = {target: 0 for target in relocation_targets}
        vehicle_count = {target: 0 for target in relocation_targets}
        for request in remaining_requests:
            if request.pickup in request_count:
                request_count[request.pickup] += 1
        for vehicle_id in vehicle_ids:
            location = self.env.vehicles[vehicle_id]['location']
            if location in vehicle_count:
                vehicle_count[location] += 1
        service_rate = {
            target: request_count[target] / max(1, vehicle_count[target])
            for target in relocation_targets
        }
        ranked_targets = sorted(
            relocation_targets,
            key=lambda target: service_rate[target],
            reverse=True,
        )

        for vehicle_id in vehicle_ids:
            if vehicle_id in assignments:
                continue
            vehicle = self.env.vehicles[vehicle_id]
            if vehicle['type'] == 1:
                assignments[vehicle_id] = "reloc"
                continue
            if float(vehicle.get('battery', 1.0)) <= no_reloc_battery_threshold:
                assignments[vehicle_id] = "waiting"
                continue
            best_target = None
            for target in ranked_targets:
                if service_rate[target] <= 0:
                    break
                relocation_distance = distance_between(vehicle['location'], target)
                if (vehicle['battery'] - relocation_distance * battery_consumption
                        >= post_request_reserve(target)):
                    best_target = target
                    break
            if best_target is None or best_target == vehicle['location']:
                assignments[vehicle_id] = "waiting"
            else:
                assignments[vehicle_id] = f"idle_at_{target_index[best_target]}"
        return assignments

    def _heuristic_assignment_with_rejectevfirst(self, vehicle_ids, available_requests, charging_stations=None,vehicle_action_matrix = None):

        assignments = {}
        battery_threshold = self.env.heuristic_battery_threshold if hasattr(self.env, 'heuristic_battery_threshold') else 0.5
        if not vehicle_ids:
            return assignments
        heuevfirst = self.env.heuevfirst if hasattr(self.env, 'heuevfirst') else False

        # 收集已分配的请求（只遍历一次所有车辆）
        assigned_requests = set()
        for vid in self.env.vehicles.keys():
            if self.env.vehicles[vid]['assigned_request'] is not None:
                assigned_requests.add(self.env.vehicles[vid]['assigned_request'])
            if self.env.vehicles[vid]['passenger_onboard'] is not None:
                assigned_requests.add(self.env.vehicles[vid]['passenger_onboard'])

        available_requests = list(self.env.active_requests.values())
        available_requests = [req for req in available_requests if req.request_id not in assigned_requests]

        if vehicle_action_matrix is not None:
            num_requests = int(getattr(self.env, '_last_matrix_num_requests', len(available_requests)))
            if num_requests < 0:
                num_requests = len(available_requests)
            matrix_request_ids = list(getattr(self.env, '_last_matrix_request_ids', []))
            matrix_charge_station_ids = list(getattr(self.env, '_last_matrix_charge_station_ids', []))
        else:
            num_requests = len(available_requests)
            matrix_request_ids = [req.request_id for req in available_requests]
            matrix_charge_station_ids = [station.id for station in charging_stations] if charging_stations else []
        request_index_map = {
            request_id: idx
            for idx, request_id in enumerate(matrix_request_ids[:num_requests])
        }
        if not request_index_map:
            request_index_map = {req.request_id: idx for idx, req in enumerate(available_requests[:num_requests])}
        station_index_map = {
            station_id: num_requests + idx
            for idx, station_id in enumerate(matrix_charge_station_ids)
        }
        vehicle_index_map = {vehicle_id: idx for idx, vehicle_id in enumerate(vehicle_ids)}

        def get_vehicle_action_row(vehicle_id):
            if vehicle_action_matrix is None:
                return None
            row_idx = vehicle_index_map.get(vehicle_id)
            if row_idx is None or row_idx >= vehicle_action_matrix.shape[0]:
                return None
            return vehicle_action_matrix[row_idx]

        def get_feasible_requests_for_vehicle(vehicle_id, candidate_requests):
            action_row = get_vehicle_action_row(vehicle_id)
            if action_row is None:
                return list(candidate_requests)
            feasible_requests = []
            for request in candidate_requests:
                req_idx = request_index_map.get(request.request_id)
                if req_idx is not None and req_idx < num_requests and action_row[req_idx] == 1:
                    feasible_requests.append(request)
            return feasible_requests

        def get_feasible_stations_for_vehicle(vehicle_id):
            if charging_stations is None:
                return []
            action_row = get_vehicle_action_row(vehicle_id)
            if action_row is None:
                return list(charging_stations)
            feasible_stations = []
            for station in charging_stations:
                matrix_idx = station_index_map.get(station.id)
                if matrix_idx is not None and matrix_idx < len(action_row) and action_row[matrix_idx] == 1:
                    feasible_stations.append(station)
            return feasible_stations

        # 动态计算每个区域的订单数量和车辆数量比率
        zone_to_locs = getattr(self.env, 'zone_to_locs', {})
        loc_to_zone = getattr(self.env, 'loc_to_zone', {})

        zone_request_count = {}
        zone_vehicle_count = {}
        zone_center_coords = {}

        for zid, locs in zone_to_locs.items():
            zone_request_count[zid] = 0
            zone_vehicle_count[zid] = 0
            # 计算区域中心坐标 (x, y)
            if locs:
                xs = [loc % self.env.grid_size for loc in locs]
                ys = [loc // self.env.grid_size for loc in locs]
                cx = int(round(sum(xs) / len(xs)))
                cy = int(round(sum(ys) / len(ys)))
                zone_center_coords[zid] = (cx, cy)

        # 统计每个区域的可用订单数
        for req in available_requests:
            pickup_zone = loc_to_zone.get(req.pickup, None)
            if pickup_zone is not None:
                zone_request_count[pickup_zone] = zone_request_count.get(pickup_zone, 0) + 1

        # 统计每个区域的空闲车辆数
        for vid in vehicle_ids:
            veh_loc = self.env.vehicles[vid]['location']
            veh_zone = loc_to_zone.get(veh_loc, None)
            if veh_zone is not None:
                zone_vehicle_count[veh_zone] = zone_vehicle_count.get(veh_zone, 0) + 1

        # 计算每个区域的服务比率（订单/车辆），车辆为0时设为订单数（最高优先）
        zone_service_rate = {}
        for zid in zone_to_locs.keys():
            req_cnt = zone_request_count.get(zid, 0)
            veh_cnt = zone_vehicle_count.get(zid, 0)
            zone_service_rate[zid] = req_cnt / veh_cnt if veh_cnt > 0 else float(req_cnt)

        # 按服务比率降序排列区域
        zone_id_sort = sorted(zone_to_locs.keys(), key=lambda z: zone_service_rate.get(z, 0), reverse=True)


        # 第一步：识别低电量车辆（电池 < 0.5）
        low_battery_vehicles = []
        high_battery_vehicles = []
        
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            if self.env.vehicles[vehicle_id]['type'] == 1:
                high_battery_vehicles.append(vehicle_id)
            else:
                if vehicle['battery'] < battery_threshold:
                    low_battery_vehicles.append(vehicle_id)
                else:
                    high_battery_vehicles.append(vehicle_id)
        
        low_battery_vehicles.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery']
        )
        if charging_stations and low_battery_vehicles:
            for vehicle_id in low_battery_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                feasible_stations = get_feasible_stations_for_vehicle(vehicle_id)
                if not feasible_stations:
                    continue
                
                best_station = None
                best_distance = float('inf')
                
                # 找到最近的有容量的充电站
                for station in feasible_stations:
                    if len(station.current_vehicles) < station.max_capacity:
                        station_coords = (
                            station.location // self.env.grid_size,
                            station.location % self.env.grid_size
                        )
                        distance = abs(vehicle_coords[0] - station_coords[0]) + \
                                  abs(vehicle_coords[1] - station_coords[1])
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_station = station
                
                if best_station:
                    assignments[vehicle_id] = f"charge_{best_station.id}"
        
        
        high_battery_vehicles_ev = [vid for vid in high_battery_vehicles if self.env.vehicles[vid]['type'] == 1]
        high_battery_vehicles_aev = [vid for vid in high_battery_vehicles if self.env.vehicles[vid]['type'] == 2]

        high_battery_vehicles_ev.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery'], 
            reverse=True
        )
        high_battery_vehicles_aev.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery'], 
            reverse=True
        )

        if available_requests:
            remaining_requests = available_requests.copy()
        else:
            remaining_requests = []
        if available_requests and high_battery_vehicles_ev:
            for n , vehicle_id in enumerate(high_battery_vehicles_ev):
                if vehicle_id in assignments:  # 已被分配充电
                    continue
                
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                battery_level = vehicle['battery']
                feasible_requests = get_feasible_requests_for_vehicle(vehicle_id, remaining_requests)
                battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

                best_request = None
                distance_list = {}
                for request in feasible_requests:
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )

                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                            abs(vehicle_coords[1] - pickup_coords[1])
                    distance_list[request.request_id] = distance
                feasible_requests_sort_distance = sorted(feasible_requests, key=lambda r: distance_list.get(r.request_id, float('inf')),reverse=False)
                for request in feasible_requests_sort_distance:
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )

                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                            abs(vehicle_coords[1] - pickup_coords[1])
                    dropoff_coords = (
                        request.dropoff // self.env.grid_size,
                        request.dropoff % self.env.grid_size
                    )
                    whole_distance = distance + \
                                    abs(request.pickup // self.env.grid_size - dropoff_coords[0]) + \
                                        abs(request.pickup % self.env.grid_size - dropoff_coords[1])
                    if battery_level - whole_distance * battery_consumption >= min_battery_level:
                        # 检查拒绝概率
                        request = next(r for r in remaining_requests if r.request_id == request.request_id)
                        best_request = request
                        break
                if best_request:
                    assignments[vehicle_id] = best_request
                    remaining_requests.remove(best_request)

        if available_requests and high_battery_vehicles_aev:
            for n , vehicle_id in enumerate(high_battery_vehicles_aev):
                if vehicle_id in assignments:  # 已被分配充电
                    continue
                
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                battery_level = vehicle['battery']
                feasible_requests = get_feasible_requests_for_vehicle(vehicle_id, remaining_requests)
                battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2
                feasible_requests_sort_value = sorted(feasible_requests, key=lambda r: getattr(r, 'final_value', getattr(r, 'value', 0.0)), reverse=True)
                best_request = None
                for request in feasible_requests_sort_value:
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )

                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                            abs(vehicle_coords[1] - pickup_coords[1])
                    
                    dropoff_coords = (
                        request.dropoff // self.env.grid_size,
                        request.dropoff % self.env.grid_size
                    )
                    whole_distance = distance + \
                                    abs(request.pickup // self.env.grid_size - dropoff_coords[0]) + \
                                        abs(request.pickup % self.env.grid_size - dropoff_coords[1])
                    estimated_consumption = whole_distance * battery_consumption
                    if battery_level - estimated_consumption >= min_battery_level:
                        # 检查拒绝概率
                        request = next(r for r in remaining_requests if r.request_id == request.request_id)
                        best_request = request
                        break
                if best_request:
                    assignments[vehicle_id] = best_request
                    remaining_requests.remove(best_request)

        for vehicle_id in vehicle_ids:
            if vehicle_id not in assignments:
                if self.env.vehicles[vehicle_id]['type'] == 2 and zone_id_sort:
                    # AEV：往订单多车少的区域移动
                    vehicle = self.env.vehicles[vehicle_id]
                    vehicle_coords = vehicle['coordinates']
                    battery_level = vehicle['battery']
                    battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                    min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

                    relocated = False
                    for zone_id in zone_id_sort:
                        if zone_service_rate.get(zone_id, 0) <= 0:
                            break  # 没有订单的区域不用去
                        if zone_id not in zone_center_coords:
                            continue
                        center = zone_center_coords[zone_id]
                        distance = abs(vehicle_coords[0] - center[0]) + abs(vehicle_coords[1] - center[1])
                        if distance == 0:
                            # 已在该区域，原地等待
                            assignments[vehicle_id] = "wait"
                            relocated = True
                            break
                        if distance * battery_consumption < battery_level - min_battery_level:
                            vehicle['idle_target'] = center
                            assignments[vehicle_id] = "wait"
                            relocated = True
                            break

                    if not relocated:
                        assignments[vehicle_id] = "wait"
                elif self.env.vehicles[vehicle_id]['type'] == 2:
                    assignments[vehicle_id] = "wait"
                else:
                    assignments[vehicle_id] = "reloc"
        return assignments

    
    def _heuristic_assignment_with_rejectaevfirst(self, vehicle_ids, available_requests, charging_stations=None,vehicle_action_matrix = None):

        assignments = {}
        battery_threshold = self.env.heuristic_battery_threshold if hasattr(self.env, 'heuristic_battery_threshold') else 0.5
        if not vehicle_ids:
            return assignments
        heuevfirst = self.env.heuevfirst if hasattr(self.env, 'heuevfirst') else False

        # 收集已分配的请求（只遍历一次所有车辆）
        assigned_requests = set()
        for vid in self.env.vehicles.keys():
            if self.env.vehicles[vid]['assigned_request'] is not None:
                assigned_requests.add(self.env.vehicles[vid]['assigned_request'])
            if self.env.vehicles[vid]['passenger_onboard'] is not None:
                assigned_requests.add(self.env.vehicles[vid]['passenger_onboard'])

        available_requests = list(self.env.active_requests.values())
        available_requests = [req for req in available_requests if req.request_id not in assigned_requests]

        if vehicle_action_matrix is not None:
            num_requests = int(getattr(self.env, '_last_matrix_num_requests', len(available_requests)))
            if num_requests < 0:
                num_requests = len(available_requests)
            matrix_request_ids = list(getattr(self.env, '_last_matrix_request_ids', []))
            matrix_charge_station_ids = list(getattr(self.env, '_last_matrix_charge_station_ids', []))
        else:
            num_requests = len(available_requests)
            matrix_request_ids = [req.request_id for req in available_requests]
            matrix_charge_station_ids = [station.id for station in charging_stations] if charging_stations else []
        request_index_map = {
            request_id: idx
            for idx, request_id in enumerate(matrix_request_ids[:num_requests])
        }
        if not request_index_map:
            request_index_map = {req.request_id: idx for idx, req in enumerate(available_requests[:num_requests])}
        station_index_map = {
            station_id: num_requests + idx
            for idx, station_id in enumerate(matrix_charge_station_ids)
        }
        vehicle_index_map = {vehicle_id: idx for idx, vehicle_id in enumerate(vehicle_ids)}

        def get_vehicle_action_row(vehicle_id):
            if vehicle_action_matrix is None:
                return None
            row_idx = vehicle_index_map.get(vehicle_id)
            if row_idx is None or row_idx >= vehicle_action_matrix.shape[0]:
                return None
            return vehicle_action_matrix[row_idx]

        def get_feasible_requests_for_vehicle(vehicle_id, candidate_requests):
            action_row = get_vehicle_action_row(vehicle_id)
            if action_row is None:
                return list(candidate_requests)
            feasible_requests = []
            for request in candidate_requests:
                req_idx = request_index_map.get(request.request_id)
                if req_idx is not None and req_idx < num_requests and action_row[req_idx] == 1:
                    feasible_requests.append(request)
            return feasible_requests

        def get_feasible_stations_for_vehicle(vehicle_id):
            if charging_stations is None:
                return []
            action_row = get_vehicle_action_row(vehicle_id)
            if action_row is None:
                return list(charging_stations)
            feasible_stations = []
            for station in charging_stations:
                matrix_idx = station_index_map.get(station.id)
                if matrix_idx is not None and matrix_idx < len(action_row) and action_row[matrix_idx] == 1:
                    feasible_stations.append(station)
            return feasible_stations

        # 动态计算每个区域的订单数量和车辆数量比率
        zone_to_locs = getattr(self.env, 'zone_to_locs', {})
        loc_to_zone = getattr(self.env, 'loc_to_zone', {})

        zone_request_count = {}
        zone_vehicle_count = {}
        zone_center_coords = {}

        for zid, locs in zone_to_locs.items():
            zone_request_count[zid] = 0
            zone_vehicle_count[zid] = 0
            # 计算区域中心坐标 (x, y)
            if locs:
                xs = [loc % self.env.grid_size for loc in locs]
                ys = [loc // self.env.grid_size for loc in locs]
                cx = int(round(sum(xs) / len(xs)))
                cy = int(round(sum(ys) / len(ys)))
                zone_center_coords[zid] = (cx, cy)

        # 统计每个区域的可用订单数
        for req in available_requests:
            pickup_zone = loc_to_zone.get(req.pickup, None)
            if pickup_zone is not None:
                zone_request_count[pickup_zone] = zone_request_count.get(pickup_zone, 0) + 1

        # 统计每个区域的空闲车辆数
        for vid in vehicle_ids:
            veh_loc = self.env.vehicles[vid]['location']
            veh_zone = loc_to_zone.get(veh_loc, None)
            if veh_zone is not None:
                zone_vehicle_count[veh_zone] = zone_vehicle_count.get(veh_zone, 0) + 1

        # 计算每个区域的服务比率（订单/车辆），车辆为0时设为订单数（最高优先）
        zone_service_rate = {}
        for zid in zone_to_locs.keys():
            req_cnt = zone_request_count.get(zid, 0)
            veh_cnt = zone_vehicle_count.get(zid, 0)
            zone_service_rate[zid] = req_cnt / veh_cnt if veh_cnt > 0 else float(req_cnt)

        # 按服务比率降序排列区域
        zone_id_sort = sorted(zone_to_locs.keys(), key=lambda z: zone_service_rate.get(z, 0), reverse=True)


        # 第一步：识别低电量车辆（电池 < 0.5）
        low_battery_vehicles = []
        high_battery_vehicles = []
        
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            if self.env.vehicles[vehicle_id]['type'] == 1:
                high_battery_vehicles.append(vehicle_id)
            else:
                if vehicle['battery'] < battery_threshold:
                    low_battery_vehicles.append(vehicle_id)
                else:
                    high_battery_vehicles.append(vehicle_id)
        
        low_battery_vehicles.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery']
        )
        if charging_stations and low_battery_vehicles:
            for vehicle_id in low_battery_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                feasible_stations = get_feasible_stations_for_vehicle(vehicle_id)
                if not feasible_stations:
                    continue
                
                best_station = None
                best_distance = float('inf')
                
                # 找到最近的有容量的充电站
                for station in feasible_stations:
                    if len(station.current_vehicles) < station.max_capacity:
                        station_coords = (
                            station.location // self.env.grid_size,
                            station.location % self.env.grid_size
                        )
                        distance = abs(vehicle_coords[0] - station_coords[0]) + \
                                  abs(vehicle_coords[1] - station_coords[1])
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_station = station
                
                if best_station:
                    assignments[vehicle_id] = f"charge_{best_station.id}"
        
        
        high_battery_vehicles_ev = [vid for vid in high_battery_vehicles if self.env.vehicles[vid]['type'] == 1]
        high_battery_vehicles_aev = [vid for vid in high_battery_vehicles if self.env.vehicles[vid]['type'] == 2]

        high_battery_vehicles_ev.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery'], 
            reverse=True
        )
        high_battery_vehicles_aev.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery'], 
            reverse=True
        )

        if available_requests:
            remaining_requests = available_requests.copy()
        else:
            remaining_requests = []
    
        if available_requests and high_battery_vehicles_aev:
            for n , vehicle_id in enumerate(high_battery_vehicles_aev):
                if vehicle_id in assignments:  # 已被分配充电
                    continue
                
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                battery_level = vehicle['battery']
                feasible_requests = get_feasible_requests_for_vehicle(vehicle_id, remaining_requests)
                battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2
                feasible_requests_sort_value = sorted(feasible_requests, key=lambda r: getattr(r, 'final_value', getattr(r, 'value', 0.0)), reverse=True)
                best_request = None
                for request in feasible_requests_sort_value:
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )

                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                            abs(vehicle_coords[1] - pickup_coords[1])
                    
                    dropoff_coords = (
                        request.dropoff // self.env.grid_size,
                        request.dropoff % self.env.grid_size
                    )
                    whole_distance = distance + \
                                    abs(request.pickup // self.env.grid_size - dropoff_coords[0]) + \
                                        abs(request.pickup % self.env.grid_size - dropoff_coords[1])
                    estimated_consumption = whole_distance * battery_consumption
                    if battery_level - estimated_consumption >= min_battery_level:
                        # 检查拒绝概率
                        request = next(r for r in remaining_requests if r.request_id == request.request_id)
                        best_request = request
                        break
                if best_request:
                    assignments[vehicle_id] = best_request
                    remaining_requests.remove(best_request)
        if available_requests and high_battery_vehicles_ev:
            for n , vehicle_id in enumerate(high_battery_vehicles_ev):
                if vehicle_id in assignments:  # 已被分配充电
                    continue
                
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                battery_level = vehicle['battery']
                feasible_requests = get_feasible_requests_for_vehicle(vehicle_id, remaining_requests)
                battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

                best_request = None
                distance_list = {}
                for request in feasible_requests:
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )

                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                            abs(vehicle_coords[1] - pickup_coords[1])
                    distance_list[request.request_id] = distance
                feasible_requests_sort_distance = sorted(feasible_requests, key=lambda r: distance_list.get(r.request_id, float('inf')),reverse=False)
                for request in feasible_requests_sort_distance:
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )

                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                            abs(vehicle_coords[1] - pickup_coords[1])
                    dropoff_coords = (
                        request.dropoff // self.env.grid_size,
                        request.dropoff % self.env.grid_size
                    )
                    whole_distance = distance + \
                                    abs(request.pickup // self.env.grid_size - dropoff_coords[0]) + \
                                        abs(request.pickup % self.env.grid_size - dropoff_coords[1])
                    if battery_level - whole_distance * battery_consumption >= min_battery_level:
                        # 检查拒绝概率
                        request = next(r for r in remaining_requests if r.request_id == request.request_id)
                        best_request = request
                        break
                if best_request:
                    assignments[vehicle_id] = best_request
                    remaining_requests.remove(best_request)
        for vehicle_id in vehicle_ids:
            if vehicle_id not in assignments:
                if self.env.vehicles[vehicle_id]['type'] == 2 and zone_id_sort:
                    # AEV：往订单多车少的区域移动
                    vehicle = self.env.vehicles[vehicle_id]
                    vehicle_coords = vehicle['coordinates']
                    battery_level = vehicle['battery']
                    battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                    min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

                    relocated = False
                    for zone_id in zone_id_sort:
                        if zone_service_rate.get(zone_id, 0) <= 0:
                            break  # 没有订单的区域不用去
                        if zone_id not in zone_center_coords:
                            continue
                        center = zone_center_coords[zone_id]
                        distance = abs(vehicle_coords[0] - center[0]) + abs(vehicle_coords[1] - center[1])
                        if distance == 0:
                            # 已在该区域，原地等待
                            assignments[vehicle_id] = "wait"
                            relocated = True
                            break
                        if distance * battery_consumption < battery_level - min_battery_level:
                            vehicle['idle_target'] = center
                            assignments[vehicle_id] = "wait"
                            relocated = True
                            break

                    if not relocated:
                        assignments[vehicle_id] = "wait"
                elif self.env.vehicles[vehicle_id]['type'] == 2:
                    assignments[vehicle_id] = "wait"
                else:
                    assignments[vehicle_id] = "reloc"
        return assignments

    
    
    
    def _heuristic_assignment_fast(self, vehicle_ids, charging_stations=None):

        assignments = {}
        battery_threshold = self.env.heuristic_battery_threshold if hasattr(self.env, 'heuristic_battery_threshold') else 0.5
        if not vehicle_ids:
            return assignments
        heuevfirst = self.env.heuevfirst if hasattr(self.env, 'heuevfirst') else False


        assigned_requests = []
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            for vehicle_id in self.env.vehicles.keys():
                if self.env.vehicles[vehicle_id]['assigned_request'] is not None:
                    assigned_requests.append(self.env.vehicles[vehicle_id]['assigned_request'])
                if self.env.vehicles[vehicle_id]['passenger_onboard'] is not None:
                    assigned_requests.append(self.env.vehicles[vehicle_id]['passenger_onboard'])
                

        available_requests = list(self.env.active_requests.values())
        available_requests = [req for req in available_requests if req.request_id not in assigned_requests]


        # 第一步：识别低电量车辆（电池 < 0.5）
        low_battery_vehicles = []
        high_battery_vehicles = []
        
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            if vehicle['battery'] < battery_threshold and self.env.vehicles[vehicle_id]['type'] == 2:
                low_battery_vehicles.append(vehicle_id)
            else:
                high_battery_vehicles.append(vehicle_id)
        
        
        low_battery_vehicles.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery']
        )
        charging_stations_list_sort = []
        if charging_stations:
            charging_stations_list_sort = sorted(
                charging_stations, 
                key=lambda station: len(station.current_vehicles) / station.max_capacity
            )
        charging_capacity_list = {}
        for station in charging_stations_list_sort:
            charging_capacity_list[station.id] = max(station.max_capacity - len(station.current_vehicles), 0)
        if charging_stations and low_battery_vehicles:
            for vehicle_id in low_battery_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                veh_loc = vehicle['location']
                battery_level = vehicle['battery']
                best_station = None
                for station in charging_stations_list_sort:
                    if charging_capacity_list[station.id] > 0:
                        distance =self.env._manhattan_distance_loc(veh_loc, station.location)
                        if battery_level - distance * self.env.battery_consum >0:
                            best_station = station
                            assignments[vehicle_id] = f"charge_{best_station.id}"
                            charging_capacity_list[station.id]-=1
                            break
        
        
        high_battery_vehicles.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery'], 
            reverse=True
        )
        remaining_requests = available_requests.copy()
        remaining_requests.sort(
            key=lambda req: getattr(req, 'final_value', getattr(req, 'value', 0.0)),
            reverse=True
        )

        if available_requests and high_battery_vehicles:
            high_battery_vehicles_list = {}
            for vehicle_id in high_battery_vehicles:
                high_battery_vehicles_list[vehicle_id] = self.env.vehicles[vehicle_id]

            remaining_requests = available_requests.copy()
            for vehicle_id in high_battery_vehicles:
                if vehicle_id in assignments:  # 已被分配充电
                    continue
                vehicle = self.env.vehicles[vehicle_id]
                vehloc = vehicle['location']
                vehicle_coords = vehicle['coordinates']
                battery_level = vehicle['battery']
                for request in remaining_requests:
                    pickup_distance = self.env._manhattan_distance_loc(vehloc, request.pickup)
                    
                    dropoff_distance = self._request_trip_distance(request)
                    whole_distance = pickup_distance + dropoff_distance
                    battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                    estimated_consumption = whole_distance * battery_consumption
                    if battery_level - estimated_consumption >= self.env.min_battery_level:
                        assignments[vehicle_id] = request
                        remaining_requests.remove(request)
                        break
        for vehicle_id in vehicle_ids:
            if vehicle_id not in assignments:
                if self.env.vehicles[vehicle_id]['type'] == 2:
                    assignments[vehicle_id] = "wait"
                else:
                    assignments[vehicle_id] = "reloc"
        return assignments



    def _heuristic_assignment_fastqvalue(self, vehicle_ids, charging_stations=None, vehicle_action_matrix=None, batch_q_value=None):
        """基于Q值的启发式分配：按电量排序，AEV比较订单/充电/reloc最大Q值决策，EV选最大订单Q值"""
        

        
        assignments = {}
        if not vehicle_ids:
            return assignments
        
        # 充电站容量跟踪
        chargingcapacitylist = {}
        if charging_stations:
            for station in charging_stations:
                chargingcapacitylist[station.id] = max(station.max_capacity - len(station.current_vehicles), 0)
        

        # 若调用方已经构造好 batch_q_value（来自 generate_vehicle_qvalue），
        # 必须沿用同一份 vehicle_action_matrix 和列布局，否则索引会与 q-value 矩阵错位。
        if batch_q_value is not None and vehicle_action_matrix is not None:
            scale_charge_station_ids = list(getattr(self.env, '_last_matrix_charge_station_ids', []))
            scale_zone_indices = list(getattr(self.env, '_last_matrix_zone_indices', []))
            scale_zone_targets = list(getattr(self.env, '_last_matrix_zone_target_ids', []))
            num_requests = int(getattr(self.env, '_last_matrix_num_requests', -1))
            num_stations = int(getattr(self.env, '_last_matrix_num_stations', -1))
            num_zones = int(getattr(self.env, '_last_matrix_num_zones', -1))
            if num_requests < 0 or num_stations < 0 or num_zones < 0:
                scale_charge_station_ids = sorted(self.env.charging_manager.stations.keys())
                scale_zone_indices = list(range(getattr(self.env, 'hotspot_locations_num', 0)))
                scale_zone_targets = list(getattr(self.env, 'hotspot_locations', []))
                num_stations = len(scale_charge_station_ids)
                num_zones = len(scale_zone_indices)
                action_width = int(batch_q_value.shape[1]) if hasattr(batch_q_value, 'shape') else int(vehicle_action_matrix.shape[1])
                num_requests = max(action_width - num_stations - num_zones - 1, 0)
            scale_charge_station_ids = scale_charge_station_ids[:num_stations]
            scale_zone_indices = scale_zone_indices[:num_zones]
            scale_zone_targets = scale_zone_targets[:num_zones]
        else:
            vehicle_action_matrix, num_requests, num_stations, num_zones = self.env.generate_whole_matrix(
                vehicle_ids,
                rebalance_num=len(vehicle_ids),
            )
            scale_charge_station_ids = list(getattr(self.env, '_last_matrix_charge_station_ids', []))[:num_stations]
            scale_zone_indices = list(getattr(self.env, '_last_matrix_zone_indices', []))[:num_zones]
            scale_zone_targets = list(getattr(self.env, '_last_matrix_zone_target_ids', []))[:num_zones]

        request_feasibility = vehicle_action_matrix[:, :num_requests]
        charging_feasibility = vehicle_action_matrix[:, num_requests:num_requests + num_stations]
        zone_feasibility = vehicle_action_matrix[:, num_requests + num_stations:num_requests + num_stations + num_zones]
        wait_feasibility = vehicle_action_matrix[:, -1]
        charging_stations = [self.env.charging_manager.stations[sid] for sid in scale_charge_station_ids if sid in self.env.charging_manager.stations]
        



        # 获取已分配的订单
        assigned_requests = []
        for vid in self.env.vehicles.keys():
            if self.env.vehicles[vid]['assigned_request'] is not None:
                assigned_requests.append(self.env.vehicles[vid]['assigned_request'])
            if self.env.vehicles[vid]['passenger_onboard'] is not None:
                assigned_requests.append(self.env.vehicles[vid]['passenger_onboard'])
        
        # 获取可用订单
        available_requests = list(self.env.active_requests.values())
        available_requests = [req for req in available_requests if req.request_id not in assigned_requests]

        active_request_by_id = {req.request_id: req for req in self.env.active_requests.values()}
        matrix_request_ids = list(getattr(self.env, '_last_matrix_request_ids', []))
        if matrix_request_ids and len(matrix_request_ids) >= num_requests:
            request_candidates = [
                (col_idx, active_request_by_id[request_id])
                for col_idx, request_id in enumerate(matrix_request_ids[:num_requests])
                if request_id in active_request_by_id and request_id not in assigned_requests
            ]
        else:
            request_candidates = list(enumerate(available_requests[:num_requests]))

        q_cols = int(batch_q_value.shape[1]) if batch_q_value is not None and hasattr(batch_q_value, 'shape') else 0
        # 判断是否onlyev模式(没有充电和reloc列)
        has_charge_reloc = (
            batch_q_value is not None
            and q_cols > num_requests + 1
            and (num_stations > 0 or num_zones > 0)
        )
        
        battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
        
        # 按电量降序排序所有车辆（不区分类型）
        sorted_vehicle_ids = sorted(vehicle_ids, key=lambda vid: self.env.vehicles[vid]['battery'], reverse=True)
        
        # 跟踪已分配的订单
        remaining_requests = request_candidates.copy()
        
        for vehicle_id in sorted_vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            vehicle_idx = vehicle_ids.index(vehicle_id)  # batch_q_value中的行索引
            is_aev = (vehicle['type'] == 2)
            vehloc = vehicle['location']
            battery_level = vehicle['battery']
            
            if batch_q_value is None:
                continue
            
            # penalty期间不分配订单，只允许充电/reloc/wait
            in_penalty = vehicle.get('penalty_timer', 0) > 0
            
            # ---- 1. 找最佳可行订单 ----
            best_request = None
            best_request_q = float('-inf')
            if not in_penalty:
                for request_col, request in remaining_requests:
                    if request_col >= q_cols:
                        continue
                    if request_col >= request_feasibility.shape[1] or request_feasibility[vehicle_idx, request_col] == 0:
                        continue
                    q_val = float(batch_q_value[vehicle_idx, request_col])
                    if q_val <= -1000:
                        continue
                    dist = self.env._manhattan_distance_loc(vehloc, request.pickup) + \
                           self._request_trip_distance(request)
                    if battery_level - dist * battery_consumption < self.env.min_battery_level:
                        continue
                    if q_val > best_request_q:
                        best_request_q = q_val
                        best_request = request
            
            if is_aev and has_charge_reloc:
                # ---- AEV: 比较订单/充电/reloc/wait Q值，选最大 ----
                
                # 2. 最佳充电Q值
                best_charge_station = None
                best_charge_q = float('-inf')
                for k, station in enumerate(charging_stations):
                    if k >= charging_feasibility.shape[1] or charging_feasibility[vehicle_idx, k] == 0:
                        continue
                    q_col = num_requests + k
                    if q_col >= q_cols - 1:
                        continue
                    q_val = float(batch_q_value[vehicle_idx, q_col])
                    if q_val <= -1000:
                        continue
                    if chargingcapacitylist.get(station.id, 0) <= 0:
                        continue
                    if battery_level - self.env._manhattan_distance_loc(vehloc, station.location) * battery_consumption <= 0:
                        continue
                    if q_val > best_charge_q:
                        best_charge_q = q_val
                        best_charge_station = station
                
                # 3. 最佳reloc Q值
                best_reloc_zone = None
                best_reloc_q = float('-inf')
                for m in range(num_zones):
                    if m >= zone_feasibility.shape[1] or zone_feasibility[vehicle_idx, m] == 0:
                        continue
                    q_col = num_requests + num_stations + m
                    if q_col >= q_cols - 1:
                        continue
                    q_val = float(batch_q_value[vehicle_idx, q_col])
                    if q_val <= -1000:
                        continue
                    if q_val > best_reloc_q:
                        best_reloc_q = q_val
                        best_reloc_zone = scale_zone_indices[m] if m < len(scale_zone_indices) else None
                
                # 4. 等待Q值
                wait_q = float(batch_q_value[vehicle_idx, -1])
                
                # 比较所有选项，选Q值最大的
                options = []
                if best_request is not None:
                    options.append(('request', best_request_q, best_request))
                if best_charge_station is not None:
                    options.append(('charge', best_charge_q, best_charge_station))
                if best_reloc_zone is not None:
                    options.append(('reloc', best_reloc_q, best_reloc_zone))
                if wait_q > -1000:
                    options.append(('wait', wait_q, None))
                
                if options:
                    best_type, best_q, best_obj = max(options, key=lambda x: x[1])
                    if best_type == 'request':
                        assignments[vehicle_id] = best_obj
                        remaining_requests = [
                            (col, req) for col, req in remaining_requests
                            if req.request_id != best_obj.request_id
                        ]
                    elif best_type == 'charge':
                        assignments[vehicle_id] = f"charge_{best_obj.id}"
                        chargingcapacitylist[best_obj.id] -= 1
                    elif best_type == 'reloc':
                        assignments[vehicle_id] = f"idle_at_{best_obj}"
                    elif best_type == 'wait':
                        assignments[vehicle_id] = "waiting"
            else:
                # ---- EV: 选最大订单Q值 ----
                if best_request is not None:
                    assignments[vehicle_id] = best_request
                    remaining_requests = [
                        (col, req) for col, req in remaining_requests
                        if req.request_id != best_request.request_id
                    ]
        
        # 未分配的车辆设置为等待或重定位
        for vehicle_id in vehicle_ids:
            if vehicle_id not in assignments:
                if self.env.vehicles[vehicle_id]['type'] == 2:  # AEV
                    assignments[vehicle_id] = "waiting"
                else:  # EV
                    assignments[vehicle_id] = "reloc"
        
        return assignments


 

    def _heuristic_assignment_fastqvalue_evfirst(self, vehicle_ids, charging_stations=None, batch_q_value=None,):
        """基于Q值的启发式分配：按电量排序，AEV比较订单/充电/reloc最大Q值决策，EV选最大订单Q值"""
        

        
        assignments = {}
        if not vehicle_ids:
            return assignments
        
        # 充电站容量跟踪
        chargingcapacitylist = {}
        if charging_stations:
            for station in charging_stations:
                chargingcapacitylist[station.id] = max(station.max_capacity - len(station.current_vehicles), 0)
        
        # 获取已分配的订单
        assigned_requests = []
        for vid in self.env.vehicles.keys():
            if self.env.vehicles[vid]['assigned_request'] is not None:
                assigned_requests.append(self.env.vehicles[vid]['assigned_request'])
            if self.env.vehicles[vid]['passenger_onboard'] is not None:
                assigned_requests.append(self.env.vehicles[vid]['passenger_onboard'])
        
        # 获取可用订单
        available_requests = list(self.env.active_requests.values())
        available_requests = [req for req in available_requests if req.request_id not in assigned_requests]
        
        # 确定batch_q_value的列布局
        num_requests = len(available_requests)
        num_stations = len(self.env.charging_manager.stations) if hasattr(self.env, 'charging_manager') else 0
        num_zones = self._get_relocation_target_count()
        station_list = list(self.env.charging_manager.stations.values()) if hasattr(self.env, 'charging_manager') else []
        
        # 判断是否onlyev模式(没有充电和reloc列)
        has_charge_reloc = False
        if batch_q_value is not None and batch_q_value.shape[1] > num_requests + 1:
            has_charge_reloc = True
        
        battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
        


        ev_vehicles = [vid for vid in vehicle_ids if self.env.vehicles[vid]['type'] != 2]
        aev_vehicles = [vid for vid in vehicle_ids if self.env.vehicles[vid]['type'] == 2]

        sorted_ev_ids = sorted(ev_vehicles, key=lambda vid: self.env.vehicles[vid]['battery'], reverse=True)
        sorted_aev_ids = sorted(aev_vehicles, key=lambda vid: self.env.vehicles[vid]['battery'], reverse=True)


        # 跟踪已分配的订单
        remaining_requests = available_requests.copy()
        


        for vehicle_id in sorted_ev_ids:
            vehicle = self.env.vehicles[vehicle_id]
            vehicle_idx = vehicle_ids.index(vehicle_id)  # batch_q_value中的行索引
            vehloc = vehicle['location']
            battery_level = vehicle['battery']
            
            if batch_q_value is None:
                continue
            
            # penalty期间不分配订单，只允许充电/reloc/wait
            in_penalty = vehicle.get('penalty_timer', 0) > 0
            
            # ---- 1. 找最佳可行订单 ----
            best_request = None
            best_request_q = float('-inf')
            if not in_penalty:
                for request in remaining_requests:
                    orig_idx = available_requests.index(request)
                    q_val = float(batch_q_value[vehicle_idx, orig_idx])
                    if q_val <= -1000:
                        continue
                    dist = self.env._manhattan_distance_loc(vehloc, request.pickup) + \
                           self._request_trip_distance(request)
                    if battery_level - dist * battery_consumption < self.env.min_battery_level:
                        continue
                    if q_val > best_request_q:
                        best_request_q = q_val
                        best_request = request
            if best_request is not None:
                assignments[vehicle_id] = best_request
                remaining_requests.remove(best_request)

        for vehicle_id in sorted_aev_ids:
            vehicle = self.env.vehicles[vehicle_id]
            vehicle_idx = vehicle_ids.index(vehicle_id)  # batch_q_value中的行索引
            is_aev = (vehicle['type'] == 2)
            vehloc = vehicle['location']
            battery_level = vehicle['battery']
            
            if batch_q_value is None:
                continue
            
            # penalty期间不分配订单，只允许充电/reloc/wait
            in_penalty = vehicle.get('penalty_timer', 0) > 0
            
            # ---- 1. 找最佳可行订单 ----
            best_request = None
            best_request_q = float('-inf')
            if not in_penalty:
                for request in remaining_requests:
                    orig_idx = available_requests.index(request)
                    q_val = float(batch_q_value[vehicle_idx, orig_idx])
                    if q_val <= -1000:
                        continue
                    dist = self.env._manhattan_distance_loc(vehloc, request.pickup) + \
                           self._request_trip_distance(request)
                    if battery_level - dist * battery_consumption < self.env.min_battery_level:
                        continue
                    if q_val > best_request_q:
                        best_request_q = q_val
                        best_request = request
            
            if is_aev and has_charge_reloc:
                # ---- AEV: 比较订单/充电/reloc/wait Q值，选最大 ----
                
                # 2. 最佳充电Q值
                best_charge_station = None
                best_charge_q = float('-inf')
                for k, station in enumerate(station_list):
                    q_val = float(batch_q_value[vehicle_idx, num_requests + k])
                    if q_val <= -1000:
                        continue
                    if chargingcapacitylist.get(station.id, 0) <= 0:
                        continue
                    if battery_level - self.env._manhattan_distance_loc(vehloc, station.location) * battery_consumption <= 0:
                        continue
                    if q_val > best_charge_q:
                        best_charge_q = q_val
                        best_charge_station = station
                
                # 3. 最佳reloc Q值
                best_reloc_zone = None
                best_reloc_q = float('-inf')
                for m in range(num_zones):
                    q_val = float(batch_q_value[vehicle_idx, num_requests + num_stations + m])
                    if q_val <= -1000:
                        continue
                    if q_val > best_reloc_q:
                        best_reloc_q = q_val
                        best_reloc_zone = m
                
                # 4. 等待Q值
                wait_q = float(batch_q_value[vehicle_idx, -1])
                
                # 比较所有选项，选Q值最大的
                options = []
                if best_request is not None:
                    options.append(('request', best_request_q, best_request))
                if best_charge_station is not None:
                    options.append(('charge', best_charge_q, best_charge_station))
                if best_reloc_zone is not None:
                    options.append(('reloc', best_reloc_q, best_reloc_zone))
                if wait_q > -1000:
                    options.append(('wait', wait_q, None))
                
                if options:
                    best_type, best_q, best_obj = max(options, key=lambda x: x[1])
                    if best_type == 'request':
                        assignments[vehicle_id] = best_obj
                        remaining_requests.remove(best_obj)
                    elif best_type == 'charge':
                        assignments[vehicle_id] = f"charge_{best_obj.id}"
                        chargingcapacitylist[best_obj.id] -= 1
                    elif best_type == 'reloc':
                        assignments[vehicle_id] = f"idle_at_{best_obj}"
                    elif best_type == 'wait':
                        assignments[vehicle_id] = "waiting"
            else:
                # ---- EV: 选最大订单Q值 ----
                if best_request is not None:
                    assignments[vehicle_id] = best_request
                    remaining_requests.remove(best_request)
        
        # 未分配的车辆设置为等待或重定位
        for vehicle_id in vehicle_ids:
            if vehicle_id not in assignments:
                if self.env.vehicles[vehicle_id]['type'] == 2:  # AEV
                    assignments[vehicle_id] = "waiting"
                else:  # EV
                    assignments[vehicle_id] = "reloc"
        
        return assignments




    def _heuristic_assignment_fastqvalue_aevfirst(self, vehicle_ids, charging_stations=None, batch_q_value=None,):
        """基于Q值的启发式分配：按电量排序，AEV比较订单/充电/reloc最大Q值决策，EV选最大订单Q值"""
        

        
        assignments = {}
        if not vehicle_ids:
            return assignments
        
        # 充电站容量跟踪
        chargingcapacitylist = {}
        if charging_stations:
            for station in charging_stations:
                chargingcapacitylist[station.id] = max(station.max_capacity - len(station.current_vehicles), 0)
        
        # 获取已分配的订单
        assigned_requests = []
        for vid in self.env.vehicles.keys():
            if self.env.vehicles[vid]['assigned_request'] is not None:
                assigned_requests.append(self.env.vehicles[vid]['assigned_request'])
            if self.env.vehicles[vid]['passenger_onboard'] is not None:
                assigned_requests.append(self.env.vehicles[vid]['passenger_onboard'])
        
        # 获取可用订单
        available_requests = list(self.env.active_requests.values())
        available_requests = [req for req in available_requests if req.request_id not in assigned_requests]
        
        # 确定batch_q_value的列布局
        num_requests = len(available_requests)
        num_stations = len(self.env.charging_manager.stations) if hasattr(self.env, 'charging_manager') else 0
        num_zones = self._get_relocation_target_count()
        station_list = list(self.env.charging_manager.stations.values()) if hasattr(self.env, 'charging_manager') else []
        
        # 判断是否onlyev模式(没有充电和reloc列)
        has_charge_reloc = False
        if batch_q_value is not None and batch_q_value.shape[1] > num_requests + 1:
            has_charge_reloc = True
        
        battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
        


        ev_vehicles = [vid for vid in vehicle_ids if self.env.vehicles[vid]['type'] != 2]
        aev_vehicles = [vid for vid in vehicle_ids if self.env.vehicles[vid]['type'] == 2]

        sorted_ev_ids = sorted(ev_vehicles, key=lambda vid: self.env.vehicles[vid]['battery'], reverse=True)
        sorted_aev_ids = sorted(aev_vehicles, key=lambda vid: self.env.vehicles[vid]['battery'], reverse=True)



        # 跟踪已分配的订单
        remaining_requests = available_requests.copy()
        
        for vehicle_id in sorted_aev_ids:
            vehicle = self.env.vehicles[vehicle_id]
            vehicle_idx = vehicle_ids.index(vehicle_id)  # batch_q_value中的行索引
            is_aev = (vehicle['type'] == 2)
            vehloc = vehicle['location']
            battery_level = vehicle['battery']
            
            if batch_q_value is None:
                continue
            
            # penalty期间不分配订单，只允许充电/reloc/wait
            in_penalty = vehicle.get('penalty_timer', 0) > 0
            
            # ---- 1. 找最佳可行订单 ----
            best_request = None
            best_request_q = float('-inf')
            if not in_penalty:
                for request in remaining_requests:
                    orig_idx = available_requests.index(request)
                    q_val = float(batch_q_value[vehicle_idx, orig_idx])
                    if q_val <= -1000:
                        continue
                    dist = self.env._manhattan_distance_loc(vehloc, request.pickup) + \
                           self._request_trip_distance(request)
                    if battery_level - dist * battery_consumption < self.env.min_battery_level:
                        continue
                    if q_val > best_request_q:
                        best_request_q = q_val
                        best_request = request
            
            if is_aev and has_charge_reloc:
                # ---- AEV: 比较订单/充电/reloc/wait Q值，选最大 ----
                
                # 2. 最佳充电Q值
                best_charge_station = None
                best_charge_q = float('-inf')
                for k, station in enumerate(station_list):
                    q_val = float(batch_q_value[vehicle_idx, num_requests + k])
                    if q_val <= -1000:
                        continue
                    if chargingcapacitylist.get(station.id, 0) <= 0:
                        continue
                    if battery_level - self.env._manhattan_distance_loc(vehloc, station.location) * battery_consumption <= 0:
                        continue
                    if q_val > best_charge_q:
                        best_charge_q = q_val
                        best_charge_station = station
                
                # 3. 最佳reloc Q值
                best_reloc_zone = None
                best_reloc_q = float('-inf')
                for m in range(num_zones):
                    q_val = float(batch_q_value[vehicle_idx, num_requests + num_stations + m])
                    if q_val <= -1000:
                        continue
                    if q_val > best_reloc_q:
                        best_reloc_q = q_val
                        best_reloc_zone = m
                
                # 4. 等待Q值
                wait_q = float(batch_q_value[vehicle_idx, -1])
                
                # 比较所有选项，选Q值最大的
                options = []
                if best_request is not None:
                    options.append(('request', best_request_q, best_request))
                if best_charge_station is not None:
                    options.append(('charge', best_charge_q, best_charge_station))
                if best_reloc_zone is not None:
                    options.append(('reloc', best_reloc_q, best_reloc_zone))
                if wait_q > -1000:
                    options.append(('wait', wait_q, None))
                
                if options:
                    best_type, best_q, best_obj = max(options, key=lambda x: x[1])
                    if best_type == 'request':
                        assignments[vehicle_id] = best_obj
                        remaining_requests.remove(best_obj)
                    elif best_type == 'charge':
                        assignments[vehicle_id] = f"charge_{best_obj.id}"
                        chargingcapacitylist[best_obj.id] -= 1
                    elif best_type == 'reloc':
                        assignments[vehicle_id] = f"idle_at_{best_obj}"
                    elif best_type == 'wait':
                        assignments[vehicle_id] = "waiting"
            else:
                # ---- EV: 选最大订单Q值 ----
                if best_request is not None:
                    assignments[vehicle_id] = best_request
                    remaining_requests.remove(best_request)
        

        for vehicle_id in sorted_ev_ids:
            vehicle = self.env.vehicles[vehicle_id]
            vehicle_idx = vehicle_ids.index(vehicle_id)  # batch_q_value中的行索引
            vehloc = vehicle['location']
            battery_level = vehicle['battery']
            
            if batch_q_value is None:
                continue
            
            # penalty期间不分配订单，只允许充电/reloc/wait
            in_penalty = vehicle.get('penalty_timer', 0) > 0
            
            # ---- 1. 找最佳可行订单 ----
            best_request = None
            best_request_q = float('-inf')
            if not in_penalty:
                for request in remaining_requests:
                    orig_idx = available_requests.index(request)
                    q_val = float(batch_q_value[vehicle_idx, orig_idx])
                    if q_val <= -1000:
                        continue
                    dist = self.env._manhattan_distance_loc(vehloc, request.pickup) + \
                           self._request_trip_distance(request)
                    if battery_level - dist * battery_consumption < self.env.min_battery_level:
                        continue
                    if q_val > best_request_q:
                        best_request_q = q_val
                        best_request = request
            if best_request is not None:
                assignments[vehicle_id] = best_request
                remaining_requests.remove(best_request)


        # 未分配的车辆设置为等待或重定位
        for vehicle_id in vehicle_ids:
            if vehicle_id not in assignments:
                if self.env.vehicles[vehicle_id]['type'] == 2:  # AEV
                    assignments[vehicle_id] = "waiting"
                else:  # EV
                    assignments[vehicle_id] = "reloc"
        
        return assignments










    def _heuristic_assignmentevfirst_with_reject(self, vehicle_ids, available_requests, charging_stations=None):

        assignments = {}
        battery_threshold = self.env.heuristic_battery_threshold if hasattr(self.env, 'heuristic_battery_threshold') else 0.5
        if not vehicle_ids:
            return assignments
        heuevfirst = self.env.heuevfirst if hasattr(self.env, 'heuevfirst') else False


        assigned_requests = []
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            for vehicle_id in self.env.vehicles.keys():
                if self.env.vehicles[vehicle_id]['assigned_request'] is not None:
                    assigned_requests.append(self.env.vehicles[vehicle_id]['assigned_request'])
                if self.env.vehicles[vehicle_id]['passenger_onboard'] is not None:
                    assigned_requests.append(self.env.vehicles[vehicle_id]['passenger_onboard'])
                

        available_requests = list(self.env.active_requests.values())
        available_requests = [req for req in available_requests if req.request_id not in assigned_requests]


        # 第一步：识别低电量车辆（电池 < 0.5）
        low_battery_vehicles = []
        high_battery_vehicles = []
        
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            if vehicle['battery'] < battery_threshold:
                low_battery_vehicles.append(vehicle_id)
            else:
                high_battery_vehicles.append(vehicle_id)
        
        
        low_battery_vehicles.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery']
        )
        if charging_stations and low_battery_vehicles:
            for vehicle_id in low_battery_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                
                best_station = None
                best_distance = float('inf')
                
                # 找到最近的有容量的充电站
                for station in charging_stations:
                    if len(station.current_vehicles) < station.max_capacity:
                        station_coords = (
                            station.location // self.env.grid_size,
                            station.location % self.env.grid_size
                        )
                        distance = abs(vehicle_coords[0] - station_coords[0]) + \
                                  abs(vehicle_coords[1] - station_coords[1])
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_station = station
                
                if best_station:
                    assignments[vehicle_id] = f"charge_{best_station.id}"
        
        
        high_battery_vehicles.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery'], 
            reverse=True
        )
        

        if available_requests and high_battery_vehicles:
            
            high_battery_vehicles_list = {}
            for vehicle_id in high_battery_vehicles:
                high_battery_vehicles_list[vehicle_id] = self.env.vehicles[vehicle_id]

            remaining_requests = available_requests.copy()
            evfleet = [vid for vid in high_battery_vehicles if self.env.vehicles[vid]['type'] == 1]
            aevfleet = [vid for vid in high_battery_vehicles if self.env.vehicles[vid]['type'] != 1]
            for vehicle_id in evfleet:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                battery_level = vehicle['battery']
                distance_list = {}
                for request in remaining_requests:
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )

                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                            abs(vehicle_coords[1] - pickup_coords[1])
                    distance_list[request.request_id] = distance
                
                battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2
                
                # 为该车辆寻找最佳订单
                best_request = None
                
                distance_sorted = sorted(distance_list.items(), key=lambda x: x[1])

                for req_id, distance in distance_sorted:
                    request = next(r for r in remaining_requests if r.request_id == req_id)
                    dropoff_coords = (
                        request.dropoff // self.env.grid_size,
                        request.dropoff % self.env.grid_size
                    )
                    whole_distance = distance + \
                                    abs(request.pickup // self.env.grid_size - dropoff_coords[0]) + \
                                        abs(request.pickup % self.env.grid_size - dropoff_coords[1])
                    estimated_consumption = whole_distance * battery_consumption 
                    if battery_level - estimated_consumption >= min_battery_level:
                        # 检查拒绝概率
                        request = next(r for r in remaining_requests if r.request_id == req_id)
                        best_request = request
                        break
                if best_request:
                    assignments[vehicle_id] = best_request
                    remaining_requests.remove(best_request)
            remaining_requests.sort(
                key=lambda req: getattr(req, 'final_value', getattr(req, 'value', 0.0)),
                reverse=True
            )
            for vehicle_id in aevfleet:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                battery_level = vehicle['battery']
                for request in remaining_requests:
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )

                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                            abs(vehicle_coords[1] - pickup_coords[1])
                    dropoff_coords = (
                        request.dropoff // self.env.grid_size,
                        request.dropoff % self.env.grid_size
                    )
                    whole_distance = distance + \
                                    abs(request.pickup // self.env.grid_size - dropoff_coords[0]) + \
                                        abs(request.pickup % self.env.grid_size - dropoff_coords[1])
                    battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                    estimated_consumption = whole_distance * battery_consumption
                    if battery_level - estimated_consumption >= self.env.min_battery_level:
                        assignments[vehicle_id] = request
                        remaining_requests.remove(request)
                        break
                
                        
        for vehicle_id in vehicle_ids:
            if vehicle_id not in assignments:
                if self.env.vehicles[vehicle_id]['type'] == 2:
                    assignments[vehicle_id] = "wait"
                else:
                    assignments[vehicle_id] = "reloc"
        return assignments


    def optimize_vehicle_rebalancing_state(self, vehicle_ids):
        """Optimize vehicle rebalancing using state-based value function (src2-style approach)"""
        if not self.available:
            return self._heuristic_rebalancing_assignment(vehicle_ids), []
        
        # Get available requests from environment
        available_requests = []
        if hasattr(self.env, 'active_requests') and self.env.active_requests:
            available_requests = list(self.env.active_requests.values())
        
        # Get available charging stations
        charging_stations = []
        if hasattr(self.env, 'charging_manager') and self.env.charging_manager.stations:
            charging_stations = [station for station in self.env.charging_manager.stations.values() 
                               if station.available_slots > 0]
        
        try:
            return self._gurobi_vehicle_rebalancing_knownreject_state_enhanced(vehicle_ids, available_requests, charging_stations)
        except Exception as e:
            print(f"Enhanced Gurobi rebalancing failed: {e}, using fallback")
            return self._heuristic_rebalancing_assignment(vehicle_ids), []

    def _gurobi_vehicle_rebalancing_knownreject_state_enhanced(self, vehicle_ids, available_requests, charging_stations=None):
        """
        Enhanced Gurobi optimization using state-based value function (src2-style approach)
        Returns both assignments and individual vehicle rewards (y_ei values)
        """
        if not self.available:
            return {}, []
        
        assignments = {}
        vehicle_rewards = []  # Store y_ei values for each vehicle
        
        # Create optimization model
        model = self.gp.Model("vehicle_assignment_state_based")
        model.setParam('OutputFlag', 0)  # Suppress output
        model.setParam('TimeLimit', 5)  # Set time limit
        model.setParam('Threads', self.num_threads)  # Set thread count

        # Parameters
        min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2
        
        # Filter out rejected requests for each EV
        valid_assignments = {}  # (vehicle_id, request_idx) -> is_valid
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            for j, request in enumerate(available_requests):
                # Check if EV would reject this request
                if vehicle['type'] == 1:
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    valid_assignments[(i, j)] = rejection_prob < 0.5
                else:
                    # AEV never rejects
                    valid_assignments[(i, j)] = True
        
        # Decision variables for request assignments
        request_decision = {}
        for i, vehicle_id in enumerate(vehicle_ids):
            for j, request in enumerate(available_requests):
                request_decision[i, j] = model.addVar(
                    vtype=self.GRB.BINARY,
                    name=f'request_{vehicle_id}_{request.request_id}'
                )
        
        # Decision variables for charging assignments
        charge_decision = {}
        if charging_stations:
            for i, vehicle_id in enumerate(vehicle_ids):
                for j, station in enumerate(charging_stations):
                    charge_decision[i, j] = model.addVar(
                        vtype=self.GRB.BINARY,
                        name=f'charge_{vehicle_id}_{station.id}'
                    )
        
        # Decision variables for idle/waiting
        idle_vehicle = {}
        waiting_vehicle = {}
        for i in range(len(vehicle_ids)):
            idle_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_idle'
            )
            waiting_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_wait'
            )
        
        # Constraints
        # Each vehicle must choose exactly one action
        for i in range(len(vehicle_ids)):
            action_sum = idle_vehicle[i] + waiting_vehicle[i]
            for j in range(len(available_requests)):
                if valid_assignments.get((i, j), False):
                    action_sum += request_decision[i, j]
            if charging_stations:
                for j in range(len(charging_stations)):
                    action_sum += charge_decision[i, j]
            model.addConstr(action_sum == 1)
        
        # Each request can be assigned to at most one vehicle
        for j in range(len(available_requests)):
            request_sum = self.gp.LinExpr()
            for i in range(len(vehicle_ids)):
                if valid_assignments.get((i, j), False):
                    request_sum += request_decision[i, j]
            model.addConstr(request_sum <= 1)
        
        # Constraint invalid assignments to 0
        for i in range(len(vehicle_ids)):
            for j in range(len(available_requests)):
                if not valid_assignments.get((i, j), False):
                    model.addConstr(request_decision[i, j] == 0)
        
        # Objective function using state-based value function
        objective_terms = self.gp.LinExpr()
        
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Process request assignments using state-based value function
            for j, request in enumerate(available_requests):
                if valid_assignments.get((i, j), False):
                    # Use state-based evaluation for service option
                    if hasattr(self.env, 'evaluate_service_option_state'):
                        try:
                            state_value = self.env.evaluate_service_option_state(vehicle_id, request)
                            immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            objective_terms += (immediate_reward + state_value) * request_decision[i, j]
                        except Exception as e:
                            print(f"Warning: Failed to get state value for vehicle {vehicle_id}, request {getattr(request, 'id', 'unknown')}: {e}")
                            # Fallback to immediate reward only
                            immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            objective_terms += immediate_reward * request_decision[i, j]
                    else:
                        # Fallback if state value function not available
                        immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                        objective_terms += immediate_reward * request_decision[i, j]
            
            # Process charging assignments using state-based value function
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    if hasattr(self.env, 'evaluate_charging_option_state'):
                        try:
                            state_value = self.env.evaluate_charging_option_state(vehicle_id, station)
                            charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)
                            objective_terms += (charging_penalty + state_value) * charge_decision[i, j]
                        except Exception as e:
                            print(f"Warning: Failed to get charging state value for vehicle {vehicle_id}: {e}")
                            charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)
                            objective_terms += charging_penalty * charge_decision[i, j]
                    else:
                        charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)
                        objective_terms += charging_penalty * charge_decision[i, j]
            
            # Process idle option using state-based value function
            if hasattr(self.env, 'evaluate_idle_option_state'):
                try:
                    idle_state_value = self.env.evaluate_idle_option_state(vehicle_id)
                    idle_penalty = getattr(self.env, 'idle_vehicle_reward', -0.1)
                    objective_terms += (idle_penalty + idle_state_value) * idle_vehicle[i]
                except Exception as e:
                    print(f"Warning: Failed to get idle state value for vehicle {vehicle_id}: {e}")
                    idle_penalty = getattr(self.env, 'idle_vehicle_reward', -0.1)
                    objective_terms += idle_penalty * idle_vehicle[i]
            else:
                idle_penalty = getattr(self.env, 'idle_vehicle_reward', -0.1)
                objective_terms += idle_penalty * idle_vehicle[i]
            
            # Process waiting option
            wait_penalty = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            objective_terms += wait_penalty * waiting_vehicle[i]
        
        # Penalty for unserved requests
        served_requests = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                if valid_assignments.get((i, j), False):
                    served_requests += request_decision[i, j]
        
        unserved_penalty = getattr(self.env, 'unserved_penalty', 1.5)
        objective_terms -= unserved_penalty * (len(available_requests) - served_requests)
        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)
        
        # Solve the optimization problem
        try:
            model.optimize()
            
            # Extract assignments and calculate individual vehicle rewards (y_ei)
            if model.status == self.GRB.OPTIMAL:
                for i, vehicle_id in enumerate(vehicle_ids):
                    vehicle_obj = 0.0  # This will be the y_ei value for this vehicle
                    
                    # Check request assignments
                    for j, request in enumerate(available_requests):
                        if request_decision[i, j].x > 0.5:
                            assignments[vehicle_id] = request
                            # Calculate contribution to objective (immediate reward + state value)
                            immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                            if hasattr(self.env, 'evaluate_service_option_state'):
                                try:
                                    state_value = self.env.evaluate_service_option_state(vehicle_id, request)
                                    vehicle_obj = immediate_reward + state_value
                                except Exception:
                                    vehicle_obj = immediate_reward
                            else:
                                vehicle_obj = immediate_reward
                            break
                    
                    # Check charging assignments if no request assigned
                    if vehicle_id not in assignments and charging_stations:
                        for j, station in enumerate(charging_stations):
                            if charge_decision[i, j].x > 0.5:
                                assignments[vehicle_id] = f"charge_{station.id}"
                                # Calculate charging contribution
                                charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)
                                if hasattr(self.env, 'evaluate_charging_option_state'):
                                    try:
                                        state_value = self.env.evaluate_charging_option_state(vehicle_id, station)
                                        vehicle_obj = charging_penalty + state_value
                                    except Exception:
                                        vehicle_obj = charging_penalty
                                else:
                                    vehicle_obj = charging_penalty
                                break
                    
                    # Check idle/waiting assignments
                    if vehicle_id not in assignments:
                        if idle_vehicle[i].x > 0.5:
                            assignments[vehicle_id] = "idle"
                            idle_penalty = getattr(self.env, 'idle_vehicle_reward', -0.1)
                            if hasattr(self.env, 'evaluate_idle_option_state'):
                                try:
                                    state_value = self.env.evaluate_idle_option_state(vehicle_id)
                                    vehicle_obj = idle_penalty + state_value
                                except Exception:
                                    vehicle_obj = idle_penalty
                            else:
                                vehicle_obj = idle_penalty
                        elif waiting_vehicle[i].x > 0.5:
                            assignments[vehicle_id] = "waiting"
                            vehicle_obj = getattr(self.env, 'waiting_vehicle_reward', -0.1)
                    
                    vehicle_rewards.append(vehicle_obj)
                
                return assignments, vehicle_rewards
            else:
                print(f"Optimization failed with status: {model.status}")
                # Return fallback assignments and zero rewards
                fallback_assignments = self._heuristic_rebalancing_assignment(vehicle_ids)
                fallback_rewards = [0.0] * len(vehicle_ids)
                return fallback_assignments, fallback_rewards
                
        except Exception as e:
            print(f"Gurobi optimization failed: {e}")
            # Return fallback assignments and zero rewards
            fallback_assignments = self._heuristic_rebalancing_assignment(vehicle_ids)
            fallback_rewards = [0.0] * len(vehicle_ids)
            return fallback_assignments, fallback_rewards

    def store_and_train_state_experiences(self, vehicle_ids, vehicle_rewards, batch_size=32):
        """
        Store state experiences and perform training using src2-style approach
        
        Args:
            vehicle_ids: List of vehicle IDs that were optimized
            vehicle_rewards: List of y_ei values (target values from Gurobi optimization)
            batch_size: Batch size for training
        """
        if not hasattr(self.env, 'value_function_state') or self.env.value_function_state is None:
            print("Warning: State-based value function not available for training")
            return 0.0
        
        # Store experiences for each vehicle
        current_time = getattr(self.env, 'current_time', 0.0)
        num_requests = len(getattr(self.env, 'active_requests', {}))
        
        for i, (vehicle_id, y_ei) in enumerate(zip(vehicle_ids, vehicle_rewards)):
            vehicle = self.env.vehicles.get(vehicle_id)
            if vehicle is None:
                continue
            
            # Get current vehicle state
            vehicle_location = vehicle['location']
            battery_level = vehicle['battery']
            
            # Calculate other vehicles (excluding current vehicle)
            other_vehicles = len([v for vid, v in self.env.vehicles.items() 
                                 if vid != vehicle_id and v['assigned_request'] is None 
                                 and v['passenger_onboard'] is None and v['charging_station'] is None])
            
            # Get request value if vehicle is assigned to a request
            request_value = 0.0
            if vehicle.get('assigned_request') in getattr(self.env, 'active_requests', {}):
                assigned_request = self.env.active_requests[vehicle['assigned_request']]
                request_value = getattr(assigned_request, 'final_value', getattr(assigned_request, 'value', 0.0))
            
            # Store state experience
            self.env.value_function_state.store_experience_state(
                vehicle_id=vehicle_id,
                vehicle_location=vehicle_location,
                battery_level=battery_level,
                current_time=current_time,
                other_vehicles=max(0, other_vehicles),
                num_requests=num_requests,
                request_value=request_value,
                y_ei=y_ei  # Target value from Gurobi optimization
            )
        
        # Perform training step
        if hasattr(self.env.value_function_state, 'experience_buffer_state'):
            buffer_size = len(self.env.value_function_state.experience_buffer_state)
            if buffer_size >= batch_size:
                training_loss = self.env.value_function_state.train_step_state(batch_size=batch_size)
                if buffer_size % 100 == 0:  # Log every 100 experiences
                    print(f"State-based training: Buffer size={buffer_size}, Loss={training_loss:.4f}")
                return training_loss
        
        return 0.0

    def _batch_calculate_reject_pro_network(self, vehicle_request_pairs):
        """
        批量计算多个vehicle-request对的拒绝概率，提高计算效率
        只对EV车辆计算，AEV返回0
        
        Args:
            vehicle_request_pairs: List of (vehicle_id, request) tuples
            
        Returns:
            List of rejection probabilities corresponding to each vehicle-request pair
        """
        if not vehicle_request_pairs:
            return []
        
        # 检查ValueFunction是否有拒绝预测器
        value_function = getattr(self.env, 'value_function', None)
        if value_function is None or not hasattr(value_function, 'rejection_predictor'):
            # 回退到单独计算
            return [self._calculate_reject_pro_network(vehicle_id, request) 
                   for vehicle_id, request in vehicle_request_pairs]
        
        try:
            # 准备批量输入特征
            batch_features = []
            valid_pairs = []
            
            for vehicle_id, request in vehicle_request_pairs:
                vehicle = self.env.vehicles.get(vehicle_id)
                if vehicle is None:
                    continue
                
                # AEV永远不拒绝，EV才需要计算
                if vehicle.get('type') == 2:  # AEV
                    continue
                elif vehicle.get('type') != 1:  # 不是EV
                    continue
                
                # 计算到pickup的距离
                vehicle_coords = vehicle['coordinates']
                pickup_coords = (request.pickup % self.env.grid_size, request.pickup // self.env.grid_size)
                distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                
                # 准备神经网络输入特征
                features = [
                    distance,                           # 距离
                    vehicle.get('battery', 1.0),       # 电池电量
                    self.env.current_time,             # 当前时间
                    len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0,  # 订单数量
                    vehicle.get('type', 1)             # 车辆类型
                ]
                
                batch_features.append(features)
                valid_pairs.append((vehicle_id, request))
            
            if not batch_features:
                # 没有需要计算的EV，返回全0
                return [0.0] * len(vehicle_request_pairs)
            
            # 批量神经网络推理
            import torch
            features_tensor = torch.tensor(batch_features, dtype=torch.float32).to(value_function.device)
            
            with torch.no_grad():
                batch_rejection_probs = value_function.rejection_predictor(features_tensor).squeeze()
                
                # 确保输出是一维的
                if batch_rejection_probs.dim() == 0:
                    batch_rejection_probs = batch_rejection_probs.unsqueeze(0)
                
                rejection_probs_list = batch_rejection_probs.cpu().numpy().tolist()
            
            # 将结果映射回原始的vehicle_request_pairs顺序
            result_probs = []
            valid_idx = 0
            
            for vehicle_id, request in vehicle_request_pairs:
                vehicle = self.env.vehicles.get(vehicle_id)
                if vehicle is None:
                    result_probs.append(0.0)
                elif vehicle.get('type') == 2:  # AEV
                    result_probs.append(0.0)
                elif vehicle.get('type') != 1:  # 不是EV
                    result_probs.append(0.0)
                else:  # EV车辆
                    if valid_idx < len(rejection_probs_list):
                        prob = rejection_probs_list[valid_idx]
                        # 确保概率在合理范围内
                        prob = max(0.0, min(0.95, prob))
                        result_probs.append(prob)
                        valid_idx += 1
                    else:
                        result_probs.append(0.0)
            
            return result_probs
            
        except Exception as e:
            print(f"Batch rejection probability calculation failed: {e}")
            # 回退到单独计算
            return [self._calculate_reject_pro_network(vehicle_id, request) 
                   for vehicle_id, request in vehicle_request_pairs]

    def _calculate_reject_pro_network(self, vehicle_id, request):
        """
        使用ValueFunction的神经网络预测器计算拒绝概率
        只对EV车辆计算，AEV返回0
        
        Args:
            vehicle_id: 车辆ID
            request: 请求对象
            
        Returns:
            float: 拒绝概率 (0-1之间)
        """
        vehicle = self.env.vehicles.get(vehicle_id)
        if vehicle is None:
            return 0.0
        
        # AEV永远不拒绝
        if vehicle.get('type') == 2:  # AEV
            return 0.0
        
        # 只对EV计算拒绝概率
        if vehicle.get('type') != 1:  # 不是EV
            return 0.0
        
        # 检查ValueFunction是否有拒绝预测器
        value_function = getattr(self.env, 'value_function', None)
        if value_function is None or not hasattr(value_function, 'rejection_predictor'):
            # 回退到简单的距离基础计算
            return self._fallback_rejection_probability(vehicle_id, request)
        
        try:
            # 计算到pickup的距离
            vehicle_coords = vehicle['coordinates']
            pickup_coords = (request.pickup % self.env.grid_size, request.pickup // self.env.grid_size)
            distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
            
            # 准备神经网络输入特征
            import torch
            features = torch.tensor([
                distance,                           # 距离
                vehicle.get('battery', 1.0),       # 电池电量
                self.env.current_time,             # 当前时间
                len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0,  # 订单数量
                vehicle.get('type', 1)             # 车辆类型
            ], dtype=torch.float32).unsqueeze(0).to(value_function.device)
            
            # 使用神经网络预测拒绝概率
            with torch.no_grad():
                rejection_prob = value_function.rejection_predictor(features).item()
            
            # 确保概率在合理范围内
            rejection_prob = max(0.0, min(0.95, rejection_prob))
            
            return rejection_prob
            
        except Exception as e:
            print(f"Neural network rejection prediction failed for vehicle {vehicle_id}: {e}")
            # 回退到简单计算
            return self._fallback_rejection_probability(vehicle_id, request)
    
    def _fallback_rejection_probability(self, vehicle_id, request):
        """
        当神经网络不可用时的回退拒绝概率计算
        基于距离的简单模型
        """
        vehicle = self.env.vehicles.get(vehicle_id)
        if vehicle is None:
            return 0.0
        
        # 计算距离
        vehicle_coords = vehicle['coordinates']
        pickup_coords = (request.pickup % self.env.grid_size, request.pickup // self.env.grid_size)
        distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
        
        # 基于距离的简单拒绝概率模型
        distance_factor = 0.2
        rejection_prob = 1 - np.exp(-distance * distance_factor)
        
        # 限制最大拒绝概率为90%
        return min(0.9, rejection_prob)

    def _calculate_rejection_aware_value(self, vehicle_id, request, base_q_value, rejection_prob=None):
        """
        计算拒绝感知的调整价值: Q_value - immediate_reward * rejection_probability
        
        Args:
            vehicle_id: 车辆ID
            request: 请求对象
            base_q_value: 基础Q值（接受订单的正向价值）
            rejection_prob: 拒绝概率（如果为None则重新计算）
            
        Returns:
            float: 调整后的价值
        """
        # 计算立即收益
        vehicle = self.env.vehicles.get(vehicle_id)
        if vehicle is None:
            return base_q_value
            
        # 订单价值（立即收益）
        immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
        
        # 计算移动成本
        cur_loc = vehicle['location']
        d1 = self._manhattan_loc(cur_loc, request.pickup)
        d2 = self._request_trip_distance(request)
        moving_cost = self._movement_cost(d1 + d2)
        
        # 净立即收益（考虑移动成本）
        net_immediate_reward = immediate_reward + moving_cost  # moving_cost通常是负数
        
        # 如果没有提供拒绝概率，则计算
        if rejection_prob is None:
            rejection_prob = self._calculate_reject_pro_network(vehicle_id, request)
        
        # 计算调整后的价值: Q值 - 立即收益 * 拒绝概率
        # 逻辑：如果拒绝概率高，则减去更多的立即收益价值
        adjusted_value = base_q_value - (net_immediate_reward * rejection_prob)
        
        return adjusted_value


    # ==================== cuOpt Integration ====================
    
    def use_cuopt_solver(self):
        """
        Enable cuOpt solver for GPU-accelerated optimization
        
        Returns:
            bool: True if cuOpt is available and enabled
        """
        try:
            from .GurobiOptimizer_cuopt import add_cuopt_methods_to_gurobi_optimizer
            add_cuopt_methods_to_gurobi_optimizer(self)
            self.cuopt_available = True
            return True
        except ImportError as e:
            print(f"⚠ cuOpt not available: {e}")
            self.cuopt_available = False
            return False
    
    
    def optimize_vehicle_rebalancing_cuopt(self, vehicle_ids, available_requests,
                                          vehicle_action_matrix, batch_q_value,
                                          use_relaxation=False, ev_only=False):
        """
        Optimize vehicle rebalancing using cuOpt (GPU-accelerated)
        
        Args:
            vehicle_ids: List of vehicle IDs to rebalance
            available_requests: List of available requests
            vehicle_action_matrix: Binary feasibility matrix [num_vehicles, num_actions]
            batch_q_value: Q-value matrix [num_vehicles, num_actions]
            use_relaxation: If True, use LP relaxation (faster, approximate)
            ev_only: If True, only consider request assignments (for EV vehicles)
            
        Returns:
            assignments: Dict mapping vehicle_id to action
        """
        # Ensure cuOpt is loaded
        if not hasattr(self, 'cuopt_available'):
            self.use_cuopt_solver()
        
        if not self.cuopt_available:
            print("⚠ cuOpt not available, falling back to Gurobi")
            if ev_only:
                return self._gurobi_vehicle_rebalancing_network_ev(
                    vehicle_ids, available_requests, vehicle_action_matrix,
                    batch_q_value, iflp=use_relaxation
                )
            else:
                return self._gurobi_vehicle_rebalancing_network(
                    vehicle_ids, available_requests, vehicle_action_matrix,
                    batch_q_value, iflp=use_relaxation
                )
        
        # Use cuOpt solver
        if ev_only:
            return self._cuopt_vehicle_rebalancing_network_ev(
                vehicle_ids, available_requests, vehicle_action_matrix,
                batch_q_value, use_relaxation=use_relaxation
            )
        else:
            return self._cuopt_vehicle_rebalancing_network(
                vehicle_ids, available_requests, vehicle_action_matrix,
                batch_q_value, use_relaxation=use_relaxation
            )
