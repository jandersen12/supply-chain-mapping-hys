"""
Exact min-cost-flow solver for the rerouting problem, via networkx's
network_simplex.

Formulates RerouteProblem as a classic unbalanced transportation problem:
    - a super-source distributes each candidate's supply capacity
    - candidates forward flow to importers at their landed unit cost
    - a dummy "unmet demand" path (very high cost) guarantees the flow
      problem is always feasible even when total supply < total demand,
      mirroring find_rerouting_options' willingness to leave some displaced
      value uncovered rather than failing outright

networkx's network_simplex requires integer demands/capacities/weights, so
USD amounts are scaled to integer cents and $/kg costs are scaled to
preserve 4 decimal places, then unscaled again in the result.
"""

import math

import networkx as nx

from .compare import SolverResult
from .problem import RerouteProblem

VALUE_SCALE = 100  # USD -> integer cents
COST_SCALE = 10_000  # unit_cost $/kg -> integer, preserves 4 decimal places

SOURCE = "__source__"
UNMET = "__unmet__"


def solve(problem: RerouteProblem) -> SolverResult:
    """Solve a RerouteProblem exactly via min-cost flow. See module docstring."""

    if not problem.demand:
        return SolverResult(
            solver_name="min_cost_flow",
            success=True,
            allocations=[],
            objective_value=0.0,
            total_unmet_value_usd=0.0,
            pct_covered=None,
            n_new_relationships=0,
        )

    total_demand = sum(problem.demand.values())

    if problem.arcs.empty:
        # No real candidates at all - everything is unmet.
        return SolverResult(
            solver_name="min_cost_flow",
            success=True,
            allocations=[],
            objective_value=0.0,
            total_unmet_value_usd=round(total_demand, 2),
            pct_covered=0.0,
            n_new_relationships=0,
        )

    max_unit_cost = problem.arcs["unit_cost_usd_per_kg"].max()
    big_m = int(math.ceil(max_unit_cost * COST_SCALE)) * 1000  # dominates any real routing

    # Round each importer's demand individually first, then derive the
    # source's total from that same sum - round(a) + round(b) can differ
    # from round(a + b), and network_simplex requires exact zero balance.
    scaled_demand = {importer: round(value * VALUE_SCALE) for importer, value in problem.demand.items()}

    G = nx.DiGraph()
    G.add_node(SOURCE, demand=-sum(scaled_demand.values()))

    for candidate, capacity in problem.supply.items():
        G.add_edge(SOURCE, f"supplier::{candidate}", capacity=round(capacity * VALUE_SCALE), weight=0)

    # Dummy path absorbs any shortfall (uncapped) so the problem is always feasible.
    G.add_edge(SOURCE, UNMET, weight=big_m)

    for importer, scaled_value in scaled_demand.items():
        G.add_node(f"importer::{importer}", demand=scaled_value)
        G.add_edge(UNMET, f"importer::{importer}", weight=0)

    for row in problem.arcs.itertuples(index=False):
        supplier_node = f"supplier::{row.candidate}"
        importer_node = f"importer::{row.importer}"
        if supplier_node not in G:
            continue  # candidate had no supply capacity (shouldn't happen, but stay safe)
        cost = round(row.unit_cost_usd_per_kg * COST_SCALE)
        edge_kwargs = {"weight": cost}
        if math.isfinite(row.max_alloc_usd):
            # Diversification's per-candidate share cap - default (no attr) is
            # unbounded capacity, which is what build_reroute_problem needs.
            edge_kwargs["capacity"] = round(row.max_alloc_usd * VALUE_SCALE)
        G.add_edge(supplier_node, importer_node, **edge_kwargs)

    try:
        _, flow_dict = nx.network_simplex(G)
    except nx.NetworkXUnfeasible as e:
        return SolverResult(solver_name="min_cost_flow", success=False, error=str(e))

    allocations = []
    for u, flows in flow_dict.items():
        if not u.startswith("supplier::"):
            continue
        candidate = u.removeprefix("supplier::")
        for v, flow in flows.items():
            if flow <= 0 or not v.startswith("importer::"):
                continue
            importer = v.removeprefix("importer::")
            arc_match = problem.arcs[
                (problem.arcs["importer"] == importer) & (problem.arcs["candidate"] == candidate)
            ]
            if arc_match.empty:
                continue
            arc = arc_match.iloc[0]
            allocated_value_usd = round(flow / VALUE_SCALE, 2)
            allocations.append({
                "importer": importer,
                "new_supplier": candidate,
                "allocated_value_usd": allocated_value_usd,
                "landed_unit_cost_usd_per_kg": round(arc["unit_cost_usd_per_kg"], 4),
                "tariff_pct": arc["tariff_pct"],
                "tariff_methodology": arc["tariff_methodology"],
                "is_new_trade_relationship": bool(arc["is_new_trade_relationship"]),
                "distance_km": round(arc["distance_km"], 1) if arc["distance_km"] is not None else None,
                "est_supplier_lead_time_days": arc["est_supplier_lead_time_days"],
            })

    unmet_by_importer = {
        v.removeprefix("importer::"): flow / VALUE_SCALE
        for v, flow in flow_dict.get(UNMET, {}).items()
        if flow > 0 and v.startswith("importer::")
    }

    total_unmet_value = sum(unmet_by_importer.values())
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
        solver_name="min_cost_flow",
        success=True,
        allocations=allocations,
        objective_value=round(objective_value, 2),
        total_unmet_value_usd=round(total_unmet_value, 2),
        pct_covered=pct_covered,
        n_new_relationships=len(new_relationships),
    )
