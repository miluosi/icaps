"""MILP backend compatibility for DOcplex (default) and Gurobi.

The project historically built every assignment model with the small subset of
the gurobipy API implemented below.  The adapter lets those model builders run
unchanged on IBM DOcplex/CPLEX, while an explicit ``gurobi`` backend still
returns the real gurobipy module and constants.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable


class _MIPConstants:
    BINARY = "B"
    Binary = BINARY
    CONTINUOUS = "C"
    INTEGER = "I"
    MAXIMIZE = -1
    MINIMIZE = 1
    OPTIMAL = 2
    INFEASIBLE = 3
    INTERRUPTED = 11
    SUBOPTIMAL = 13
    TIME_LIMIT = 9


def normalize_mip_backend(value: str | None) -> str:
    normalized = str(
        value or os.environ.get("MIXED_FLEET_MIP_BACKEND", "docplex")
    ).strip().lower().replace("-", "_")
    aliases = {
        "cplex": "docplex",
        "docplex_cplex": "docplex",
        "ibm_cplex": "docplex",
        "grb": "gurobi",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"docplex", "gurobi"}:
        raise ValueError(
            f"unknown MILP backend {value!r}; expected 'docplex' or 'gurobi'"
        )
    return normalized


def _unwrap(value: Any) -> Any:
    return value._var if isinstance(value, _DocplexVar) else value


class _DocplexVar:
    __slots__ = ("_var", "_model")

    def __init__(self, variable, model: "_DocplexModel"):
        self._var = variable
        self._model = model

    @property
    def x(self) -> float:
        return self._model._value(self._var)

    X = x

    @property
    def VarName(self) -> str:
        return str(self._var.name)

    def to_linear_expr(self):
        """Allow native DOcplex expressions to consume a wrapped variable."""
        return self._var.to_linear_expr()

    def __add__(self, other):
        return self._var + _unwrap(other)

    def __radd__(self, other):
        return _unwrap(other) + self._var

    def __sub__(self, other):
        return self._var - _unwrap(other)

    def __rsub__(self, other):
        return _unwrap(other) - self._var

    def __mul__(self, other):
        return self._var * _unwrap(other)

    def __rmul__(self, other):
        return _unwrap(other) * self._var

    def __truediv__(self, other):
        return self._var / _unwrap(other)

    def __neg__(self):
        return -self._var

    def __le__(self, other):
        return self._var <= _unwrap(other)

    def __ge__(self, other):
        return self._var >= _unwrap(other)

    def __eq__(self, other):
        return self._var == _unwrap(other)

    __hash__ = object.__hash__


class _ParameterProxy:
    def __init__(self, model: "_DocplexModel"):
        object.__setattr__(self, "_model", model)

    def __setattr__(self, name: str, value: Any) -> None:
        self._model.setParam(name, value)

    def __getattr__(self, name: str) -> Any:
        return self._model._parameters.get(name)


class _ConstraintView:
    def __init__(self, constraint, conflicted: bool = False):
        self._constraint = constraint
        self.IISConstr = bool(conflicted)

    @property
    def constrName(self):
        return str(self._constraint.name or "")

    @property
    def sense(self):
        return str(getattr(self._constraint, "sense", ""))

    @property
    def RHS(self):
        right = getattr(self._constraint, "right_expr", 0.0)
        return float(right) if isinstance(right, (int, float)) else right


class _RowView:
    def __init__(self, model: "_DocplexModel", constraint):
        expression = constraint.left_expr - constraint.right_expr
        self._terms = [
            (_DocplexVar(variable, model), float(coefficient))
            for variable, coefficient in expression.iter_terms()
        ]

    def size(self):
        return len(self._terms)

    def getVar(self, index):
        return self._terms[index][0]

    def getCoeff(self, index):
        return self._terms[index][1]


class _DocplexModel:
    def __init__(self, name: str | None = None):
        from docplex.mp.model import Model

        self._model = Model(name=name or "mixed_fleet_milp")
        self._variables: list[_DocplexVar] = []
        self._objective_terms: list[tuple[float, _DocplexVar]] = []
        self._explicit_objective = False
        self._model_sense = _MIPConstants.MINIMIZE
        self._solution = None
        self._status = _MIPConstants.INTERRUPTED
        self._parameters: dict[str, Any] = {}
        self._conflicted_constraints: set[Any] = set()
        self.Params = _ParameterProxy(self)

    def addVar(
        self,
        *,
        lb: float = 0.0,
        ub: float | None = None,
        obj: float = 0.0,
        vtype: str = _MIPConstants.CONTINUOUS,
        name: str | None = None,
    ):
        normalized = str(vtype).upper()
        if normalized in {_MIPConstants.BINARY, "BINARY"}:
            variable = self._model.binary_var(name=name)
        elif normalized in {_MIPConstants.INTEGER, "INTEGER"}:
            variable = self._model.integer_var(
                lb=lb,
                ub=ub if ub is not None and math.isfinite(float(ub)) else None,
                name=name,
            )
        else:
            variable = self._model.continuous_var(
                lb=lb,
                ub=ub if ub is not None and math.isfinite(float(ub)) else None,
                name=name,
            )
        wrapped = _DocplexVar(variable, self)
        self._variables.append(wrapped)
        if float(obj or 0.0) != 0.0:
            self._objective_terms.append((float(obj), wrapped))
        return wrapped

    def addConstr(self, constraint, name: str | None = None):
        return self._model.add_constraint(_unwrap(constraint), ctname=name)

    def setObjective(self, expression, sense=_MIPConstants.MINIMIZE):
        self._explicit_objective = True
        self._model_sense = sense
        expression = _unwrap(expression)
        if sense == _MIPConstants.MAXIMIZE:
            self._model.maximize(expression)
        else:
            self._model.minimize(expression)

    @property
    def ModelSense(self):
        return self._model_sense

    @ModelSense.setter
    def ModelSense(self, value):
        self._model_sense = value

    def setParam(self, name: str, value: Any):
        key = str(name)
        self._parameters[key] = value
        normalized = key.lower()
        try:
            if normalized == "outputflag":
                self._model.context.solver.log_output = bool(value)
            elif normalized == "timelimit":
                self._model.set_time_limit(float(value))
            elif normalized == "threads":
                self._model.parameters.threads = max(1, int(value))
            elif normalized == "feasibilitytol":
                self._model.parameters.simplex.tolerances.feasibility = float(value)
            elif normalized == "optimalitytol":
                self._model.parameters.simplex.tolerances.optimality = float(value)
            elif normalized == "method" and int(value) == 1:
                self._model.parameters.lpmethod = 2
        except Exception:
            # Parameters unsupported by a specific CPLEX build are only
            # performance hints; the mathematical model remains unchanged.
            pass

    def optimize(self):
        if not self._explicit_objective and self._objective_terms:
            objective = _DocplexModule.quicksum(
                coefficient * variable
                for coefficient, variable in self._objective_terms
            )
            self.setObjective(objective, self._model_sense)
        log_output = bool(self._parameters.get("OutputFlag", 0))
        self._solution = self._model.solve(log_output=log_output)
        details = self._model.solve_details
        status = str(getattr(details, "status", "")).lower()
        if self._solution is not None and "optimal" in status:
            self._status = _MIPConstants.OPTIMAL
        elif "infeasible" in status:
            self._status = _MIPConstants.INFEASIBLE
        elif "time" in status and "limit" in status:
            self._status = _MIPConstants.TIME_LIMIT
        elif self._solution is not None:
            self._status = _MIPConstants.SUBOPTIMAL
        else:
            self._status = _MIPConstants.INTERRUPTED
        return self._solution

    def _value(self, variable) -> float:
        if self._solution is None:
            return 0.0
        return float(self._solution.get_value(variable))

    @property
    def status(self):
        return self._status

    Status = status

    @property
    def SolCount(self):
        return int(self._solution is not None)

    @property
    def objVal(self):
        if self._solution is None:
            raise AttributeError("objective value is unavailable before a solution exists")
        return float(self._solution.objective_value)

    ObjVal = objVal

    @property
    def NumVars(self):
        return int(self._model.number_of_variables)

    @property
    def NumConstrs(self):
        return int(self._model.number_of_constraints)

    def getVars(self):
        return list(self._variables)

    def getConstrs(self):
        return [
            _ConstraintView(ct, ct in self._conflicted_constraints)
            for ct in self._model.iter_constraints()
        ]

    def getRow(self, constraint):
        raw = getattr(constraint, "_constraint", constraint)
        return _RowView(self, raw)

    def update(self):
        return None

    def computeIIS(self):
        try:
            from docplex.mp.conflict_refiner import ConflictRefiner

            conflicts = ConflictRefiner().refine_conflict(self._model)
            self._conflicted_constraints = {
                conflict.element for conflict in conflicts.iter_conflicts()
            }
        except Exception:
            self._conflicted_constraints = set()

    def write(self, path: str):
        return self._model.export_as_lp(path)


class _DocplexModule:
    Model = _DocplexModel

    @staticmethod
    def LinExpr():
        return 0

    @staticmethod
    def quicksum(values: Iterable[Any]):
        total = 0
        for value in values:
            total = total + _unwrap(value)
        return total


def get_mip_api(backend: str | None = None):
    """Return ``(model_api, constants, normalized_backend)``."""
    normalized = normalize_mip_backend(backend)
    if normalized == "docplex":
        import cplex  # noqa: F401 - verifies that the CPLEX runtime is present
        import docplex  # noqa: F401

        return _DocplexModule, _MIPConstants, normalized
    import gurobipy as gp
    from gurobipy import GRB

    return gp, GRB, normalized


def mip_backend_available(backend: str | None = None) -> bool:
    try:
        get_mip_api(backend)
        return True
    except (ImportError, ModuleNotFoundError):
        return False
