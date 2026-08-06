"""
LP-based rerouting solver via Google OR-Tools (open-source alternative to
Gurobi).

Formulates RerouteProblem as an explicit linear program - decision
variables, constraints, and an objective - rather than min_cost_flow.py's
graph/flow-network modeling. Both are exact solvers for the same underlying
transportation problem and should reach the same (or equally optimal)
objective value; this module demonstrates the constraint-based formulation
pattern gurobi.py will reuse next, and serves as a free, license-free
baseline before comparing against Gurobi's performance/features.

Unlike min_cost_flow.py, no integer scaling is needed here - OR-Tools' LP
solver (GLOP) works directly with continuous floats. There's no combinatorial
decision in this version of the problem (no fixed costs or binary
candidate-selection), so a pure LP - not a MILP - already solves it exactly;
integer/binary variables would only be needed if a future extension added
discrete choices (e.g. "use at most N new suppliers").

GLOP does need decision-variable magnitudes to stay reasonably close to the
objective coefficients, though. demand/supply are raw USD (can run into the
hundreds of billions for a high-value commodity), while unit costs are
$/kg (single digits to low tens) - that spread leaves the constraint matrix
badly conditioned and GLOP returns ABNORMAL instead of solving. VALUE_SCALE
rescales demand/supply/capacity into millions of USD before building the LP
so variable and coefficient magnitudes stay in the same neighborhood; the
solution is scaled back to raw USD afterward.
"""

from ortools.linear_solver import pywraplp

from .compare import SolverResult
from .problem import RerouteProblem

VALUE_SCALE = 1e-6  # USD -> millions of USD, keeps GLOP well-conditioned


def solve(problem: RerouteProblem) -> SolverResult:
    """Solve a RerouteProblem exactly via linear programming (OR-Tools/GLOP).

    Decision variables: x[importer, candidate] >= 0, the USD value of trade
    routed from candidate to importer. unmet[importer] >= 0 is a slack
    variable absorbing any demand no real candidate can cover, penalized at
    a very high per-dollar cost in the objective - the LP equivalent of
    min_cost_flow.py's dummy "unmet" path, guaranteeing the problem is
    always feasible even when total supply < total demand.

    Constraints:
        - for each importer: sum of allocations + unmet == demand[importer]
        - for each candidate: sum of allocations <= supply[candidate]

    Objective: minimize sum(x[i,j] * unit_cost[i,j]) + big_m * sum(unmet[i])
    """

    if not problem.demand:
        return SolverResult(
            solver_name="or_tools",
            success=True,
            allocations=[],
            objective_value=0.0,
            total_unmet_value_usd=0.0,
            pct_covered=None,
            n_new_relationships=0,
        )

    total_demand = sum(problem.demand.values())

    if problem.arcs.empty:
        return SolverResult(
            solver_name="or_tools",
            success=True,
            allocations=[],
            objective_value=0.0,
            total_unmet_value_usd=round(total_demand, 2),
            pct_covered=0.0,
            n_new_relationships=0,
        )

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        return SolverResult(
            solver_name="or_tools", success=False, error="Could not create OR-Tools GLOP solver."
        )

    # Dominates any real routing cost, but not by an arbitrarily large factor:
    # a few real arcs in this dataset are extreme price outliers (low-quantity,
    # high-value edges), and multiplying an already-large outlier by too big a
    # factor (e.g. 1000x) creates a huge objective-coefficient spread that
    # causes GLOP to fail numerically (status ABNORMAL) rather than solve.
    # 100x keeps the penalty dominant while staying numerically well-behaved.
    big_m = problem.arcs["unit_cost_usd_per_kg"].max() * 100

    arc_rows = list(problem.arcs.itertuples(index=False))

    # One decision variable per feasible (importer, candidate) arc.
    x = {
        (row.importer, row.candidate): solver.NumVar(0, solver.infinity(), f"x_{row.importer}_{row.candidate}")
        for row in arc_rows
    }
    unmet = {
        importer: solver.NumVar(0, solver.infinity(), f"unmet_{importer}") for importer in problem.demand
    }

    arcs_by_importer: dict[str, list[tuple[str, str]]] = {}
    arcs_by_candidate: dict[str, list[tuple[str, str]]] = {}
    for key in x:
        arcs_by_importer.setdefault(key[0], []).append(key)
        arcs_by_candidate.setdefault(key[1], []).append(key)

    # Demand constraints: allocations + unmet must exactly cover each importer's need.
    # Values scaled by VALUE_SCALE (see module docstring) to keep GLOP well-conditioned.
    for importer, need in problem.demand.items():
        terms = [x[key] for key in arcs_by_importer.get(importer, [])]
        solver.Add(solver.Sum(terms) + unmet[importer] == need * VALUE_SCALE)

    # Supply constraints: allocations out of each candidate can't exceed its capacity.
    for candidate, capacity in problem.supply.items():
        terms = [x[key] for key in arcs_by_candidate.get(candidate, [])]
        if terms:
            solver.Add(solver.Sum(terms) <= capacity * VALUE_SCALE)

    unit_cost = {(row.importer, row.candidate): row.unit_cost_usd_per_kg for row in arc_rows}
    objective_terms = [unit_cost[key] * var for key, var in x.items()]
    objective_terms += [big_m * var for var in unmet.values()]
    solver.Minimize(solver.Sum(objective_terms))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return SolverResult(
            solver_name="or_tools",
            success=False,
            error=f"Solver did not find an optimal solution (status={status}).",
        )

    arc_lookup = {(row.importer, row.candidate): row for row in arc_rows}

    allocations = []
    for (importer, candidate), var in x.items():
        value = var.solution_value() / VALUE_SCALE
        if value <= 1e-6:
            continue
        row = arc_lookup[(importer, candidate)]
        allocations.append({
            "importer": importer,
            "new_supplier": candidate,
            "allocated_value_usd": round(value, 2),
            "landed_unit_cost_usd_per_kg": round(row.unit_cost_usd_per_kg, 4),
            "freight_cost_usd_per_kg": row.freight_cost_usd_per_kg,
            "tariff_pct": row.tariff_pct,
            "tariff_methodology": row.tariff_methodology,
            "is_new_trade_relationship": bool(row.is_new_trade_relationship),
            "distance_km": round(row.distance_km, 1) if row.distance_km is not None else None,
            "est_supplier_lead_time_days": row.est_supplier_lead_time_days,
        })

    total_unmet_value = sum(var.solution_value() for var in unmet.values()) / VALUE_SCALE
    objective_value = sum(
        a["allocated_value_usd"] * a["landed_unit_cost_usd_per_kg"] for a in allocations
    )
    pct_covered = (
        round((total_demand - total_unmet_value) / total_demand * 100, 2) if total_demand else None
    )
    new_relationships = {
        (a["importer"], a["new_supplier"]) for a in allocations if a["is_new_trade_relationship"]
    }

    return SolverResult(
        solver_name="or_tools",
        success=True,
        allocations=allocations,
        objective_value=round(objective_value, 2),
        total_unmet_value_usd=round(total_unmet_value, 2),
        pct_covered=pct_covered,
        n_new_relationships=len(new_relationships),
    )
