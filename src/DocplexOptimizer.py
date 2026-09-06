"""Explicit DOcplex optimizer entry point.

``GurobiOptimizer`` remains import-compatible, but is backend-neutral and uses
DOcplex by default.  This subclass is useful when callers want an unambiguous
DOcplex type without relying on environment defaults.
"""

from src.GurobiOptimizer import GurobiOptimizer


class DocplexOptimizer(GurobiOptimizer):
    """Explicit CPLEX implementation of every legacy ILP model family."""

    _docplex_vehicle_rebalancing = GurobiOptimizer._gurobi_vehicle_rebalancing
    _docplex_vehicle_rebalancing_ev = GurobiOptimizer._gurobi_vehicle_rebalancing_ev
    _docplex_vehicle_rebalancing_network = GurobiOptimizer._gurobi_vehicle_rebalancing_network
    _docplex_vehicle_rebalancing_network_ev = GurobiOptimizer._gurobi_vehicle_rebalancing_network_ev
    _docplex_vehicle_rebalancing_integrated = GurobiOptimizer._gurobi_vehicle_rebalancing_integrated
    _docplex_vehicle_rebalancing_aev = GurobiOptimizer._gurobi_vehicle_rebalancing_aev
    _docplex_vehicle_rebalancing_knownreject = GurobiOptimizer._gurobi_vehicle_rebalancing_knownreject
    _docplex_vehicle_rebalancing_knownreject_state = GurobiOptimizer._gurobi_vehicle_rebalancing_knownreject_state
    _docplex_vehicle_rebalancing_knownreject_state_enhanced = GurobiOptimizer._gurobi_vehicle_rebalancing_knownreject_state_enhanced

    def __init__(self, env, num_threads=16):
        super().__init__(env, num_threads=num_threads, mip_backend="docplex")
