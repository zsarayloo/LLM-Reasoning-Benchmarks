# src/experiment/gurobi_verifier.py

from typing import Optional, Dict, Any
import gurobipy as gp
from gurobipy import GRB

from lp_schema import LinearProgram, Constraint


def solve_lp_with_gurobi(lp: LinearProgram) -> Dict[str, Any]:
    """
    Build and solve a LP/MIP in Gurobi from a LinearProgram schema.

    Returns a dict with:
        - status: "OPTIMAL" / "INFEASIBLE" / "UNBOUNDED" / ...
        - obj_value: float or None
        - solution: dict(var_name -> value) or None
    """
    model = gp.Model()
    model.Params.OutputFlag = 0  # silent

    # 1) Create variables
    vars_ = {}
    var_names = sorted(set(
        list(lp.objective.keys()) +
        [v for c in lp.constraints for v in c.coeffs.keys()]
    ))

    for name in var_names:
        lb = lp.var_lower_bounds.get(name, 0.0)
        ub = lp.var_upper_bounds.get(name, GRB.INFINITY)
        vtype = GRB.INTEGER if name in lp.integer_vars else GRB.CONTINUOUS
        vars_[name] = model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)

    model.update()

    # 2) Objective
    expr = gp.LinExpr()
    for name, coef in lp.objective.items():
        expr.add(vars_[name], coef)

    if lp.sense == "max":
        model.setObjective(expr, GRB.MAXIMIZE)
    else:
        model.setObjective(expr, GRB.MINIMIZE)

    # 3) Constraints
    for c in lp.constraints:
        lhs = gp.LinExpr()
        for name, coef in c.coeffs.items():
            lhs.add(vars_[name], coef)

        if c.sense == "<=":
            model.addConstr(lhs <= c.rhs, name=c.name)
        elif c.sense == ">=":
            model.addConstr(lhs >= c.rhs, name=c.name)
        else:  # "=="
            model.addConstr(lhs == c.rhs, name=c.name)

    # 4) Solve
    model.optimize()

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }
    status_str = status_map.get(model.Status, f"STATUS_{model.Status}")

    if model.Status == GRB.OPTIMAL:
        obj = float(model.ObjVal)
        sol = {name: float(v.X) for name, v in vars_.items()}
    else:
        obj = None
        sol = None

    return {
        "status": status_str,
        "obj_value": obj,
        "solution": sol,
    }
