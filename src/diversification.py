"""
Proactive supplier-concentration diagnostics and rebalancing plans.

Complements the reactive tools in supply_chain_network.py (simulate_removal,
find_rerouting_options), which answer "if a supplier disappears, how bad is
it?" This module answers a different question: "are we over-concentrated on
any single supplier right now, and how do we voluntarily rebalance before
anything breaks?"

Like solvers/problem.py, this is plain functions taking a SupplyChainNetwork
in, rather than growing that class further.
"""

from typing import TYPE_CHECKING, Any

from .solvers import greedy as greedy_solver
from .solvers import min_cost_flow, or_tools
from .solvers.compare import run_all_solvers
from .solvers.problem import build_diversification_problem

if TYPE_CHECKING:
    from .supply_chain_network import SupplyChainNetwork

SOLVER_REGISTRY = {
    "greedy": greedy_solver.solve,
    "min_cost_flow": min_cost_flow.solve,
    "or_tools": or_tools.solve,
}


def concentration_report(
    network: "SupplyChainNetwork",
    graph=None,
    top_n: int | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Report each importer's supplier concentration.

    For every importer with outgoing trade edges in `graph` (defaults to
    network.graph), computes each supplier's share of that importer's total
    trade value dynamically (value / sum of that importer's out-edges) -
    deliberately not read from the precomputed share_of_reporter_total column
    in cleaned_edges.csv, so this same function also works on a hypothetical
    post-rebalance graph, not just the baseline.

    Args:
        network: a loaded SupplyChainNetwork (only used as a graph source
            when `graph` isn't given).
        graph: a networkx DiGraph to compute over; defaults to network.graph.
        top_n: if given, return only the top_n most concentrated importers.
        threshold: if given, populate each row's "oversupplied" list with
            every supplier exceeding this share (otherwise "oversupplied" is
            always empty - this report is diagnostic-only unless a threshold
            is applied).

    Returns:
        List of {importer, total_import_value_usd, top_supplier, top_share,
        oversupplied: [{supplier, share}, ...]}, sorted by top_share
        descending.
    """

    g = graph if graph is not None else network.graph

    rows = []
    for importer in g.nodes():
        out_edges = list(g.edges(importer, data=True))
        total = sum(d.get("trade_value_usd", 0) or 0 for _, _, d in out_edges)
        if total <= 0:
            continue

        shares = sorted(
            (
                (supplier, (d.get("trade_value_usd", 0) or 0) / total)
                for _, supplier, d in out_edges
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        top_supplier, top_share = shares[0]

        oversupplied = []
        if threshold is not None:
            oversupplied = [
                {"supplier": supplier, "share": round(share, 4)}
                for supplier, share in shares
                if share > threshold
            ]

        rows.append({
            "importer": importer,
            "total_import_value_usd": round(total, 2),
            "top_supplier": top_supplier,
            "top_share": round(top_share, 4),
            "oversupplied": oversupplied,
        })

    rows.sort(key=lambda r: r["top_share"], reverse=True)
    return rows[:top_n] if top_n is not None else rows


def plan_diversification(
    network: "SupplyChainNetwork",
    importers: list[str] | None = None,
    max_share_target: float = 0.2,
    capacity_multiplier: float = 0.3,
    onboarding_cost_multiplier: float = 0.0,
    onboarding_lead_time_days: float = 45.0,
    solvers: tuple[str, ...] = ("greedy", "min_cost_flow", "or_tools"),
) -> dict[str, Any]:
    """Find a plan to bring every over-concentrated importer's suppliers under
    max_share_target, by shifting the excess value to alternate suppliers.

    Runs every requested solver (see SOLVER_REGISTRY) against the same
    RerouteProblem and recommends whichever succeeded with the highest
    pct_covered (ties broken by lowest objective_value) - deliberately
    comparing all three rather than picking one up front, since this is
    exactly the kind of exact-optimization-friendly assignment problem the
    solvers/ comparison work already exists for.

    Args:
        network: a loaded SupplyChainNetwork.
        importers: importers to rebalance. If None, every importer whose top
            supplier share exceeds max_share_target is auto-selected via
            concentration_report.
        max_share_target: no supplier should exceed this fraction of an
            importer's total trade value.
        capacity_multiplier, onboarding_cost_multiplier,
            onboarding_lead_time_days: same semantics as
            find_rerouting_options / build_reroute_problem.
        solvers: which registered solvers (see SOLVER_REGISTRY) to run and
            compare.

    Returns:
        A JSON-serializable dict. If a failure occurs, 'success' = False and
        'error' explains what went wrong. If nothing exceeds the target,
        'already_compliant' = True. Otherwise: concentration_before/after,
        the full solver_comparison table, which solver was recommended, and
        the recommended plan (reroutes/summary, shaped like
        find_rerouting_options' output so existing viz code can be reused).
    """

    if not (0 < max_share_target < 1):
        return {"success": False, "error": "max_share_target must be between 0 and 1."}

    if importers is not None:
        resolved: list[str] = []
        unresolved: list[dict[str, Any]] = []
        for name in importers:
            match, suggestions = network._resolve_country(name)
            if match:
                resolved.append(match)
            else:
                unresolved.append({"input": name, "suggestions": suggestions})
        if unresolved:
            return {
                "success": False,
                "error": "One or more importer names were not found in the network.",
                "unresolved": unresolved,
            }
    else:
        resolved = [
            r["importer"]
            for r in concentration_report(network, threshold=max_share_target)
            if r["oversupplied"]
        ]

    if not resolved:
        return {
            "success": True,
            "already_compliant": True,
            "concentration": concentration_report(network, top_n=10),
        }

    built = build_diversification_problem(
        network,
        resolved,
        max_share_target=max_share_target,
        capacity_multiplier=capacity_multiplier,
        onboarding_cost_multiplier=onboarding_cost_multiplier,
        onboarding_lead_time_days=onboarding_lead_time_days,
    )
    if not built["success"]:
        return {"success": False, "error": built["error"]}

    problem = built["problem"]
    solver_fns = {name: SOLVER_REGISTRY[name] for name in solvers if name in SOLVER_REGISTRY}
    results = run_all_solvers(problem, solver_fns)

    solver_comparison = [
        {
            "solver": name,
            "success": r.success,
            "objective_value": r.objective_value,
            "total_unmet_value_usd": r.total_unmet_value_usd,
            "pct_covered": r.pct_covered,
            "n_new_relationships": r.n_new_relationships,
            "runtime_seconds": r.runtime_seconds,
            "error": r.error,
        }
        for name, r in results.items()
    ]

    successful = [(name, r) for name, r in results.items() if r.success]
    if not successful:
        return {
            "success": False,
            "error": "No solver found a feasible diversification plan.",
            "solver_comparison": solver_comparison,
        }

    recommended_solver, winner = max(
        successful,
        key=lambda item: (item[1].pct_covered or -1, -(item[1].objective_value or 0)),
    )

    scenario = {
        "importers": resolved,
        "max_share_target": max_share_target,
        "capacity_multiplier": capacity_multiplier,
        "onboarding_cost_multiplier": onboarding_cost_multiplier,
        "onboarding_lead_time_days": onboarding_lead_time_days,
    }

    concentration_before = [
        r for r in concentration_report(network) if r["importer"] in set(resolved)
    ]

    g_after, plan = _apply_plan(network.graph, problem, winner)

    concentration_after = [
        r for r in concentration_report(network, graph=g_after) if r["importer"] in set(resolved)
    ]

    return {
        "success": True,
        "error": None,
        "scenario": scenario,
        "concentration_before": concentration_before,
        "concentration_after": concentration_after,
        "solver_comparison": solver_comparison,
        "recommended_solver": recommended_solver,
        "plan": plan,
    }


def seek_diversification_plan(
    network: "SupplyChainNetwork",
    importers: list[str] | None = None,
    max_share_target: float = 0.2,
    capacity_multiplier_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    onboarding_cost_multiplier: float = 0.0,
    onboarding_lead_time_days: float = 45.0,
    solvers: tuple[str, ...] = ("greedy", "min_cost_flow", "or_tools"),
) -> dict[str, Any]:
    """Search for a diversification plan that fully clears max_share_target,
    by trying increasingly aggressive capacity_multiplier assumptions rather
    than committing to one up front.

    plan_diversification takes capacity_multiplier as a given and reports how
    much of the required shift it could cover; this wraps it in a search over
    capacity_multiplier_grid (ascending - each value is a more aggressive
    assumption about how much a supplier could realistically absorb, kept in
    the 0.1-0.4 range since a supplier absorbing anywhere close to double its
    current export volume isn't realistic), stopping at the first attempt
    that reaches ~100% coverage. If nothing in the grid gets there, the best
    attempt by pct_covered is returned instead, so the caller always gets a
    usable plan plus an honest trace of what was tried and why it fell short.

    This loop is deliberately plain, deterministic Python, not an LLM
    reasoning loop - cheaper, debuggable, and reproducible. See
    resilience_agent.py for where an LLM is actually used (parsing a
    free-text goal into the arguments this function takes, and narrating the
    result), kept out of the search itself.

    Args:
        network, importers, max_share_target, onboarding_cost_multiplier,
            onboarding_lead_time_days, solvers: same semantics as
            plan_diversification.
        capacity_multiplier_grid: capacity_multiplier values to try, in
            ascending order.

    Returns:
        Same shape as plan_diversification's response, plus:
        goal_met (bool), chosen_capacity_multiplier (float | None - None if
        already_compliant, since compliance doesn't depend on capacity), and
        trace (list of {capacity_multiplier, success, pct_covered,
        recommended_solver} or {capacity_multiplier, success: False, error}
        per attempt).
    """

    trace: list[dict[str, Any]] = []
    attempts: list[tuple[float, dict[str, Any]]] = []

    for capacity_multiplier in capacity_multiplier_grid:
        result = plan_diversification(
            network,
            importers=importers,
            max_share_target=max_share_target,
            capacity_multiplier=capacity_multiplier,
            onboarding_cost_multiplier=onboarding_cost_multiplier,
            onboarding_lead_time_days=onboarding_lead_time_days,
            solvers=solvers,
        )

        if not result["success"]:
            trace.append({
                "capacity_multiplier": capacity_multiplier,
                "success": False,
                "error": result.get("error"),
            })
            continue

        if result.get("already_compliant"):
            # Compliance doesn't depend on capacity_multiplier - nothing to search for.
            return {**result, "goal_met": True, "chosen_capacity_multiplier": None, "trace": []}

        pct_covered = result["plan"]["summary"]["pct_covered"] or 0.0
        trace.append({
            "capacity_multiplier": capacity_multiplier,
            "success": True,
            "pct_covered": pct_covered,
            "recommended_solver": result["recommended_solver"],
        })
        attempts.append((capacity_multiplier, result))

        if pct_covered >= 100 - 1e-6:
            return {**result, "goal_met": True, "chosen_capacity_multiplier": capacity_multiplier, "trace": trace}

    if not attempts:
        return {
            "success": False,
            "error": "No feasible diversification plan found across the capacity_multiplier grid.",
            "trace": trace,
        }

    best_capacity_multiplier, best_result = max(
        attempts, key=lambda item: item[1]["plan"]["summary"]["pct_covered"] or 0.0
    )
    return {
        **best_result,
        "goal_met": False,
        "chosen_capacity_multiplier": best_capacity_multiplier,
        "trace": trace,
    }


def _apply_plan(graph, problem, winner):
    """Apply a winning SolverResult's allocations to a copy of `graph`, and
    build the reroutes/summary output shape (mirroring find_rerouting_options)
    from problem.displaced + winner.allocations.

    Each oversupplied (importer, supplier) edge is reduced by its share of
    however much of that importer's demand was actually fulfilled (not the
    full requested shift, if the plan fell short) - pro-rated across an
    importer's oversupplied suppliers by their contribution to that demand,
    since a solver only tracks aggregate demand per importer.
    """

    g_after = graph.copy()

    allocations_by_importer: dict[str, list[dict]] = {}
    for alloc in winner.allocations:
        allocations_by_importer.setdefault(alloc["importer"], []).append(alloc)

    displaced_by_importer: dict[str, list[dict]] = {}
    for entry in problem.displaced:
        displaced_by_importer.setdefault(entry["importer"], []).append(entry)

    reroutes = []
    new_relationships: set[tuple[str, str]] = set()

    for importer, requested_total in problem.demand.items():
        allocations = allocations_by_importer.get(importer, [])
        allocated_total = sum(a["allocated_value_usd"] for a in allocations)
        fulfilled_ratio = allocated_total / requested_total if requested_total else 0.0

        reduced_suppliers = []
        for entry in displaced_by_importer.get(importer, []):
            actual_shift = round(entry["shift_value_usd"] * fulfilled_ratio, 2)
            if g_after.has_edge(importer, entry["oversupplied_supplier"]):
                g_after[importer][entry["oversupplied_supplier"]]["trade_value_usd"] -= actual_shift
            reduced_suppliers.append({
                "supplier": entry["oversupplied_supplier"],
                "current_share": entry["current_share"],
                "requested_shift_usd": entry["shift_value_usd"],
                "actual_shift_usd": actual_shift,
            })

        for alloc in allocations:
            candidate = alloc["new_supplier"]
            if g_after.has_edge(importer, candidate):
                g_after[importer][candidate]["trade_value_usd"] += alloc["allocated_value_usd"]
            else:
                g_after.add_edge(importer, candidate, trade_value_usd=alloc["allocated_value_usd"])
            if alloc["is_new_trade_relationship"]:
                new_relationships.add((importer, candidate))

        unmet_value = round(requested_total - allocated_total, 2)
        reroutes.append({
            "importer": importer,
            "reduced_suppliers": reduced_suppliers,
            "allocations": allocations,
            "unmet_value_usd": unmet_value,
            "pct_covered": round(allocated_total / requested_total * 100, 2) if requested_total else None,
        })

    total_demand = sum(problem.demand.values())
    summary = {
        "n_importers_rebalanced": len(reroutes),
        "total_shift_value_usd": round(total_demand, 2),
        "total_unmet_value_usd": winner.total_unmet_value_usd,
        "pct_covered": winner.pct_covered,
        "new_trade_relationships_formed": sorted(f"{u} -> {v}" for u, v in new_relationships),
    }

    return g_after, {"reroutes": reroutes, "summary": summary}
