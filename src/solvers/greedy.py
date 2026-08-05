"""
Generic greedy solver for the rerouting problem, operating directly on a
RerouteProblem's arcs/demand/supply - unlike compare.py's `_greedy_adapter`,
which wraps SupplyChainNetwork.find_rerouting_options and is therefore
specific to full-supplier-removal scenarios. This version has no such
assumption, so it can serve as a baseline for any RerouteProblem (removal or
diversification alike), replicating find_rerouting_options' heuristic:
largest-demand-first, cheapest-arc-first, capacity-constrained, splitting a
single importer's demand across multiple candidates if needed.
"""

from .compare import SolverResult
from .problem import RerouteProblem


def solve(problem: RerouteProblem) -> SolverResult:
    """Solve a RerouteProblem with a greedy largest-demand/cheapest-arc heuristic."""

    if not problem.demand:
        return SolverResult(
            solver_name="greedy",
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
            solver_name="greedy",
            success=True,
            allocations=[],
            objective_value=0.0,
            total_unmet_value_usd=round(total_demand, 2),
            pct_covered=0.0,
            n_new_relationships=0,
        )

    remaining_capacity = dict(problem.supply)
    arcs_by_importer: dict[str, list] = {}
    for row in problem.arcs.itertuples(index=False):
        arcs_by_importer.setdefault(row.importer, []).append(row)
    for importer, arcs in arcs_by_importer.items():
        arcs.sort(key=lambda r: r.unit_cost_usd_per_kg)

    allocations = []
    total_unmet = 0.0

    for importer, need in sorted(problem.demand.items(), key=lambda kv: kv[1], reverse=True):
        remaining_to_allocate = need
        for row in arcs_by_importer.get(importer, []):
            if remaining_to_allocate <= 0:
                break
            available = remaining_capacity.get(row.candidate, 0)
            take = min(remaining_to_allocate, available, row.max_alloc_usd)
            if take <= 0:
                continue

            remaining_capacity[row.candidate] -= take
            remaining_to_allocate -= take
            allocations.append({
                "importer": importer,
                "new_supplier": row.candidate,
                "allocated_value_usd": round(take, 2),
                "landed_unit_cost_usd_per_kg": round(row.unit_cost_usd_per_kg, 4),
                "tariff_pct": row.tariff_pct,
                "tariff_methodology": row.tariff_methodology,
                "is_new_trade_relationship": bool(row.is_new_trade_relationship),
                "distance_km": round(row.distance_km, 1) if row.distance_km is not None else None,
                "est_supplier_lead_time_days": row.est_supplier_lead_time_days,
            })

        total_unmet += remaining_to_allocate

    objective_value = sum(
        a["allocated_value_usd"] * a["landed_unit_cost_usd_per_kg"] for a in allocations
    )
    pct_covered = (
        round((total_demand - total_unmet) / total_demand * 100, 2) if total_demand else None
    )
    new_relationships = {
        (a["importer"], a["new_supplier"]) for a in allocations if a["is_new_trade_relationship"]
    }

    return SolverResult(
        solver_name="greedy",
        success=True,
        allocations=allocations,
        objective_value=round(objective_value, 2),
        total_unmet_value_usd=round(total_unmet, 2),
        pct_covered=pct_covered,
        n_new_relationships=len(new_relationships),
    )
