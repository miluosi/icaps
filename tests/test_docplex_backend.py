from types import SimpleNamespace

import numpy as np

from src.DocplexOptimizer import DocplexOptimizer
from src.GurobiOptimizer import GurobiOptimizer
from src.CentralAgent import CentralAgent
from src.exact_mcmf import (
    build_reduced_problem,
    solve_docplex_network,
    solve_primal_dual,
)
from src.mip_backend import get_mip_api


def test_docplex_gurobi_compatible_api_solves_binary_assignment():
    gp, grb, backend = get_mip_api("docplex")
    model = gp.Model("docplex_adapter_assignment")
    x = model.addVar(vtype=grb.BINARY, name="x")
    y = model.addVar(vtype=grb.BINARY, name="y")
    model.addConstr(x + y <= 1)
    model.setObjective(3 * x + 2 * y, grb.MAXIMIZE)
    model.optimize()

    assert backend == "docplex"
    assert model.status == grb.OPTIMAL
    assert model.SolCount == 1
    assert x.X == 1.0
    assert y.X == 0.0
    assert model.objVal == 3.0


def test_docplex_network_matches_project_exact_solver():
    problem = build_reduced_problem(
        np.ones((3, 3), dtype=bool),
        np.array([[5.0, 1.0, 0.0], [4.0, 6.0, 0.0], [3.0, 2.0, 7.0]]),
        np.array([1, 1, 1]),
        cost_scale=100,
        fallback_values=None,
    )
    docplex_result = solve_docplex_network(problem, num_threads=1)
    reference_result = solve_primal_dual(problem, verify=True)

    assert docplex_result.backend == "docplex_network"
    assert docplex_result.status == "OPTIMAL"
    assert docplex_result.objective_int == reference_result.objective_int
    assert docplex_result.action_by_vehicle == reference_result.action_by_vehicle


def test_optimizer_defaults_to_docplex_and_keeps_gurobi_selectable():
    env = SimpleNamespace(mcmf_backend="docplex_network")
    default_optimizer = GurobiOptimizer(env, num_threads=1)
    docplex_optimizer = DocplexOptimizer(env, num_threads=1)
    gurobi_optimizer = GurobiOptimizer(env, num_threads=1, mip_backend="gurobi")

    assert default_optimizer.available
    assert default_optimizer.mip_backend == "docplex"
    assert docplex_optimizer.available
    assert docplex_optimizer.mip_backend == "docplex"
    assert gurobi_optimizer.available
    assert gurobi_optimizer.mip_backend == "gurobi"


def test_explicit_gurobi_backend_still_solves_small_models():
    gp, grb, backend = get_mip_api("gurobi")
    model = gp.Model("retained_gurobi_interface")
    model.setParam("OutputFlag", 0)
    x = model.addVar(vtype=grb.BINARY, name="x")
    model.setObjective(x, grb.MAXIMIZE)
    model.optimize()

    assert backend == "gurobi"
    assert model.status == grb.OPTIMAL
    assert x.X == 1.0


def test_existing_network_assignment_model_runs_through_docplex():
    request_0 = SimpleNamespace(request_id=100)
    request_1 = SimpleNamespace(request_id=101)
    env = SimpleNamespace(
        mip_backend="docplex",
        mcmf_backend="docplex_network",
        vehicles={10: {"type": 1}, 20: {"type": 2}},
        num_zones=0,
        hotspot_locations=[],
        charging_manager=SimpleNamespace(stations={}),
        station_queue_capacity=0,
        reserve_inbound_charging_capacity=False,
        record_time=False,
        synthetic_demand_profile="predictive",
    )
    optimizer = GurobiOptimizer(env, num_threads=1)
    assignments = optimizer._gurobi_vehicle_rebalancing_network(
        [10, 20],
        [request_0, request_1],
        np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int8),
        np.array([[5.0, 1.0, 0.0], [4.0, 6.0, 1.0]]),
        iflp=False,
    )

    assert assignments == {10: request_0, 20: request_1}


def test_central_agent_ilp_uses_configured_docplex_backend():
    class Candidate:
        def __init__(self, requests):
            self.requests = requests

    shared_request = object()
    first = Candidate([shared_request])
    first_idle = Candidate([])
    conflicting = Candidate([shared_request])
    second = Candidate([object()])
    central = CentralAgent(SimpleNamespace(mip_backend="docplex", NUM_AGENTS=2))

    chosen = central._choose_actions_ILP(
        [[(first, 5.0), (first_idle, 0.0)], [(conflicting, 4.0), (second, 3.0)]]
    )

    assert central.mip_backend == "docplex"
    assert [action for action, _ in chosen] == [first, second]
