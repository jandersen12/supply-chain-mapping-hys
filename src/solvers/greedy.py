"""
Greedy rerouting heuristic: cheapest available supplier assignment via sequential capacity depletion,
largest displaced relationships being first.
"""

from .compare import SolverResult
from .problem import RerouteProblem


def solve(problem: RerouteProblem) -> SolverResult:
    """
    Solve RerouteProblem with a greedy algorithm.
    Processed the displaced relationships with the largest displaced value first.
    Assign each displacement to its cheapest available candidate(s) until covered or capacity runs out.
    """

    if not problem.demand:
        return SolverResult(
            solver_name="greedy",
            success=True,
            allocations=[],
            objective_value=0.0,
            total_unmet_value_usd=0.0,
            pct_covered=None,
            n_new_relationships=0
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
    arcs_by_importer = {
        importer: group.sort_values("unit_cost_usd_per_kg") for importer, group in problem.arcs.groupby("importer")
    }

    displaced_sorted = sorted(
        problem.displaced, key=lambda d: d["displaced_value_usd"], reverse=True
    )

    allocations = []
    new_relationships: set[tuple[str, str]] = set()
    total_unmet_value = 0.0

    for entry in displaced_sorted:
        importer = entry["importer"]
        remaining_to_allocate = entry["displaced_value_usd"]

        candidate_arcs = arcs_by_importer.get(importer)
        if candidate_arcs is not None:
            for row in candidate_arcs.itertuples(index=False):
                if remaining_to_allocate <= 0:
                    break
                available = remaining_capacity.get(row.candidate, 0)
                take = min(remaining_to_allocate, available)
                if take <= 0:
                    continue

                remaining_capacity[row.candidate] -= take
                remaining_to_allocate -= take
                if row.is_new_trade_relationship:
                    new_relationships.add((importer, row.candidate))

                allocations.append({
                    "importer": importer,
                    "new_supplier": row.candidate,
                    "allocated_value_usd": round(take, 2),
                    "landed_unit_cost_usd_per_kg": round(row.unit_cost_usd_per_kg, 4),
                    "freight_cost_usd_per_kg": row.freight_cost_usd_per_kg,
                    "tariff_pct": row.tariff_pct,
                    "tariff_methodology": row.tariff_methodology,
                    "is_new_trade_relationship": bool(row.is_new_trade_relationship),
                    "distance_km": round(row.distance_km, 1) if row.distance_km is not None else None,
                    "est_supplier_lead_time_days": row.est_supplier_lead_time_days,
                })

        total_unmet_value += round(remaining_to_allocate, 2)

    objective_value = sum(
        a["allocated_value_usd"] * a["landed_unit_cost_usd_per_kg"] for a in allocations
    )
    pct_covered = (
        round((total_demand - total_unmet_value) / total_demand * 100, 2) if total_demand else None
    )

    return SolverResult(
        solver_name="greedy",
        success=True,
        allocations=allocations,
        objective_value=round(objective_value, 2),
        total_unmet_value_usd=round(total_unmet_value, 2),
        pct_covered=pct_covered,
        n_new_relationships=len(new_relationships),
    )
