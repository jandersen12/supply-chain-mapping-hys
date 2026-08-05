"""
Solver-agnostic input contract for rerouting-optimization experiments.

build_reroute_problem() extracts the same displaced-relationship / candidate-
cost setup used by SupplyChainNetwork.find_rerouting_options (the greedy
baseline) into a plain, framework-agnostic structure - arcs, demand, supply -
that any solver (networkx min_cost_flow, OR-Tools, Gurobi, stochastic) can
consume identically. This is what makes solver comparisons fair: every solver
sees the exact same costs, capacities, and constraints; they just search over
them differently.

Flow unit: every quantity here (demand, supply, allocations) is trade value
in USD, not physical quantity - the same convention find_rerouting_options
already uses. unit_cost_usd_per_kg is a $/kg price, used as a per-unit-of-flow
cost weight for ranking/optimization even though the flow itself is measured
in dollars, not kg. That's an inherited simplification from the greedy
baseline (which never tracks physical quantity through its allocation loop,
only value), kept here rather than "fixed" so solver comparisons are
apples-to-apples against the existing baseline rather than against a
different problem.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable

import pandas as pd

from ..estimate_lead_times import estimate_lead_time_days
from ..estimate_tariffs import estimate_tariff_pct

if TYPE_CHECKING:
    from ..supply_chain_network import SupplyChainNetwork

ARC_COLUMNS = [
    "importer",
    "candidate",
    "unit_cost_usd_per_kg",
    "is_new_trade_relationship",
    "tariff_pct",
    "tariff_methodology",
    "distance_km",
    "est_supplier_lead_time_days",
    "max_alloc_usd",
]


@dataclass
class RerouteProblem:
    """Solver-agnostic definition of a rerouting-optimization problem.

    Attributes:
        scenario: echoes every input parameter (removed_countries and each
            cost/capacity knob), for traceability in solver output.
        displaced: one entry per (importer, removed_supplier) relationship
            that needs replacement supply - kept separate from `demand`
            (which aggregates by importer) purely for reporting, so results
            can still be attributed back to which removed supplier each
            importer lost.
        demand: importer -> total trade value (USD) needing replacement,
            summed across every removed supplier that importer lost. A
            solver doesn't need the per-removed-supplier split to allocate -
            any remaining candidate can cover any of an importer's lost
            volume - so this is aggregated, unlike `displaced`.
        supply: candidate -> capacity (USD) it can absorb in *additional*
            rerouted trade (current export value x capacity_multiplier).
        arcs: every feasible (importer, candidate) pair with its cost and
            descriptive attributes - one row per pair, not per unit of flow.
            Capacity/demand constraints live in `supply`/`demand`, not on
            individual arcs.
    """

    scenario: dict[str, Any]
    displaced: list[dict[str, Any]]
    demand: dict[str, float]
    supply: dict[str, float]
    arcs: pd.DataFrame


def build_reroute_problem(
    network: "SupplyChainNetwork",
    removed_countries: list[str],
    capacity_multiplier: float = 0.3,
    onboarding_cost_multiplier: float = 0.0,
    onboarding_lead_time_days: float = 45.0,
) -> dict[str, Any]:
    """Build a RerouteProblem from the network for the given removal scenario.

    Mirrors find_rerouting_options' setup logic (displaced relationships,
    candidate cost/capacity, tariff lookup) but stops short of actually
    solving - it hands back the plain arcs/demand/supply structure every
    solver should consume, so they're all optimizing over identical inputs.

    Args:
        network: a loaded SupplyChainNetwork.
        removed_countries: countries whose export capacity is gone - same
            semantics as find_rerouting_options.
        capacity_multiplier, onboarding_cost_multiplier,
            onboarding_lead_time_days: same semantics and defaults as
            find_rerouting_options.

    Returns:
        {"success": True, "problem": RerouteProblem} on success, or
        {"success": False, "error": str, ...} on the same validation
        failures find_rerouting_options can hit (unknown countries, bad
        capacity_multiplier).
    """

    if not removed_countries:
        return {"success": False, "error": "No countries provided.", "suggestions": []}

    if capacity_multiplier <= 0:
        return {"success": False, "error": "capacity_multiplier must be > 0.", "suggestions": []}

    resolved: list[str] = []
    unresolved: list[dict[str, Any]] = []
    for name in removed_countries:
        match, suggestions = network._resolve_country(name)
        if match:
            resolved.append(match)
        else:
            unresolved.append({"input": name, "suggestions": suggestions})

    if unresolved:
        return {
            "success": False,
            "error": "One or more country names were not found in the network.",
            "unresolved": unresolved,
        }

    scenario = {
        "removed_countries": resolved,
        "capacity_multiplier": capacity_multiplier,
        "onboarding_cost_multiplier": onboarding_cost_multiplier,
        "onboarding_lead_time_days": onboarding_lead_time_days,
    }

    all_edges = list(network.graph.edges(data=True))
    displaced_edges = [(u, v, d) for u, v, d in all_edges if v in resolved]

    if not displaced_edges:
        return {
            "success": True,
            "problem": RerouteProblem(
                scenario=scenario,
                displaced=[],
                demand={},
                supply={},
                arcs=pd.DataFrame(columns=ARC_COLUMNS),
            ),
        }

    candidate_stats = _build_candidate_stats(all_edges, exclude_suppliers=set(resolved), capacity_multiplier=capacity_multiplier)
    supply = {supplier: stats["capacity"] for supplier, stats in candidate_stats.items()}

    # Demand: aggregate displaced value per importer. Per-removed-supplier
    # detail is kept in `displaced` for reporting only.
    demand: dict[str, float] = defaultdict(float)
    displaced: list[dict[str, Any]] = []
    for importer, removed_supplier, data in displaced_edges:
        displaced_value = data.get("trade_value_usd", 0) or 0
        original_qty = data.get("trade_qty_kg")
        original_unit_price = (
            displaced_value / original_qty if original_qty == original_qty and original_qty else None
        )
        demand[importer] += displaced_value
        displaced.append({
            "importer": importer,
            "removed_supplier": removed_supplier,
            "displaced_value_usd": round(displaced_value, 2),
            "original_unit_price_usd_per_kg": round(original_unit_price, 4)
            if original_unit_price is not None else None,
        })
    demand = dict(demand)

    arcs = _build_arcs(
        network,
        all_edges,
        demand_importers=demand.keys(),
        candidate_stats=candidate_stats,
        exclude_pairs=set(),
        onboarding_cost_multiplier=onboarding_cost_multiplier,
        onboarding_lead_time_days=onboarding_lead_time_days,
    )

    return {
        "success": True,
        "problem": RerouteProblem(
            scenario=scenario,
            displaced=displaced,
            demand=demand,
            supply=supply,
            arcs=arcs,
        ),
    }


def _build_candidate_stats(
    all_edges: list[tuple[str, str, dict[str, Any]]],
    exclude_suppliers: set[str],
    capacity_multiplier: float,
) -> dict[str, dict[str, float]]:
    """Aggregate each remaining supplier's current export value/quantity into
    an average unit price and a rerouting capacity (current export value x
    capacity_multiplier). Suppliers in exclude_suppliers (e.g. the ones being
    removed) and suppliers with no quantity basis to price them are skipped.

    Returns: supplier -> {"avg_unit_price_usd_per_kg": float, "capacity": float}
    """

    export_value_by_supplier: dict[str, float] = defaultdict(float)
    export_qty_by_supplier: dict[str, float] = defaultdict(float)
    for _, target, data in all_edges:
        if target in exclude_suppliers:
            continue
        export_value_by_supplier[target] += data.get("trade_value_usd", 0) or 0
        qty = data.get("trade_qty_kg")
        if qty == qty:  # filters NaN
            export_qty_by_supplier[target] += qty

    candidate_stats: dict[str, dict[str, float]] = {}
    for supplier, total_value in export_value_by_supplier.items():
        total_qty = export_qty_by_supplier.get(supplier, 0)
        if total_qty <= 0:
            continue  # no price basis to rank this candidate
        candidate_stats[supplier] = {
            "avg_unit_price_usd_per_kg": total_value / total_qty,
            "capacity": total_value * capacity_multiplier,
        }

    return candidate_stats


def _build_arcs(
    network: "SupplyChainNetwork",
    all_edges: list[tuple[str, str, dict[str, Any]]],
    demand_importers: "Iterable[str]",
    candidate_stats: dict[str, dict[str, float]],
    exclude_pairs: set[tuple[str, str]],
    onboarding_cost_multiplier: float,
    onboarding_lead_time_days: float,
    max_alloc_fn: "Callable[[str, str], float] | None" = None,
) -> pd.DataFrame:
    """Build every feasible (importer, candidate) arc: cost, tariff, distance,
    lead time. Skips self-loops and any (importer, candidate) pair in
    exclude_pairs (used by diversification to stop a shift from being
    reallocated straight back to the supplier it's being shifted away from).

    max_alloc_fn(importer, candidate), if given, caps how much of an
    importer's demand that single arc may carry (used by diversification so
    a solver can't fix one over-concentrated supplier by creating another -
    every candidate's post-allocation share of the importer's trade must
    itself stay under max_share_target). Defaults to unbounded (math.inf),
    which is what build_reroute_problem uses - a full-removal scenario has no
    such share ceiling.
    """

    existing_tariffs = {
        (u, v): d.get("estimated_tariff_pct")
        for u, v, d in all_edges
        if d.get("estimated_tariff_pct") == d.get("estimated_tariff_pct")
    }

    arc_rows = []
    for importer in demand_importers:
        for candidate, stats in candidate_stats.items():
            if candidate == importer or (importer, candidate) in exclude_pairs:
                continue

            if (importer, candidate) in existing_tariffs:
                tariff_pct = existing_tariffs[(importer, candidate)]
                tariff_methodology = "existing_trade_relationship"
                is_new_relationship = False
            else:
                est = estimate_tariff_pct(importer, candidate, network._importer_default_rates)
                tariff_pct = est["estimated_tariff_pct"]
                tariff_methodology = est["tariff_methodology"]
                is_new_relationship = True

            unit_cost = stats["avg_unit_price_usd_per_kg"] * (1 + tariff_pct)
            if is_new_relationship:
                unit_cost *= (1 + onboarding_cost_multiplier)

            distance_km = network._distance_km(importer, candidate)
            lead_time = estimate_lead_time_days(
                distance_km, is_new_relationship, onboarding_lead_time_days
            )

            arc_rows.append({
                "importer": importer,
                "candidate": candidate,
                "unit_cost_usd_per_kg": unit_cost,
                "is_new_trade_relationship": is_new_relationship,
                "tariff_pct": tariff_pct,
                "tariff_methodology": tariff_methodology,
                "distance_km": distance_km,
                "est_supplier_lead_time_days": lead_time["est_supplier_lead_time_days"],
                "max_alloc_usd": max_alloc_fn(importer, candidate) if max_alloc_fn else math.inf,
            })

    return pd.DataFrame(arc_rows, columns=ARC_COLUMNS)


def build_diversification_problem(
    network: "SupplyChainNetwork",
    importers: list[str],
    max_share_target: float = 0.2,
    capacity_multiplier: float = 0.3,
    onboarding_cost_multiplier: float = 0.0,
    onboarding_lead_time_days: float = 45.0,
) -> dict[str, Any]:
    """Build a RerouteProblem for voluntarily rebalancing supplier concentration,
    rather than reacting to a supplier's removal.

    For each importer, any supplier whose share of that importer's total trade
    value exceeds max_share_target contributes its excess value
    (value - max_share_target * total) to that importer's demand - summed
    across every over-threshold supplier, so an importer with more than one
    concentrated supplier is handled in one shot rather than needing
    iteration. Those over-threshold (importer, supplier) pairs are excluded
    from the arcs so a solver can't just hand the shifted volume straight
    back to the supplier it's being shifted away from.

    Args:
        network: a loaded SupplyChainNetwork.
        importers: importers to rebalance (already-resolved country names).
        max_share_target: no supplier should exceed this fraction of an
            importer's total trade value.
        capacity_multiplier, onboarding_cost_multiplier,
            onboarding_lead_time_days: same semantics as build_reroute_problem.

    Returns:
        {"success": True, "problem": RerouteProblem} on success, or
        {"success": False, "error": str} if no importer needs rebalancing.
    """

    scenario = {
        "importers": importers,
        "max_share_target": max_share_target,
        "capacity_multiplier": capacity_multiplier,
        "onboarding_cost_multiplier": onboarding_cost_multiplier,
        "onboarding_lead_time_days": onboarding_lead_time_days,
    }

    all_edges = list(network.graph.edges(data=True))

    demand: dict[str, float] = {}
    displaced: list[dict[str, Any]] = []
    exclude_pairs: set[tuple[str, str]] = set()
    importer_totals: dict[str, float] = {}
    existing_value: dict[tuple[str, str], float] = {}

    for importer in importers:
        out_edges = [(u, v, d) for u, v, d in all_edges if u == importer]
        total = sum(d.get("trade_value_usd", 0) or 0 for _, _, d in out_edges)
        if total <= 0:
            continue
        importer_totals[importer] = total

        shift = 0.0
        for _, supplier, d in out_edges:
            value = d.get("trade_value_usd", 0) or 0
            existing_value[(importer, supplier)] = value
            share = value / total
            if share > max_share_target:
                excess = value - max_share_target * total
                shift += excess
                exclude_pairs.add((importer, supplier))
                displaced.append({
                    "importer": importer,
                    "oversupplied_supplier": supplier,
                    "current_share": round(share, 4),
                    "shift_value_usd": round(excess, 2),
                })

        if shift > 0:
            demand[importer] = shift

    if not demand:
        return {"success": False, "error": "No importer in the given set exceeds max_share_target."}

    candidate_stats = _build_candidate_stats(all_edges, exclude_suppliers=set(), capacity_multiplier=capacity_multiplier)
    supply = {supplier: stats["capacity"] for supplier, stats in candidate_stats.items()}

    def max_alloc(importer: str, candidate: str) -> float:
        """Cap how much of importer's demand a single candidate can take, so
        allocating the shift can't just make `candidate` the new
        over-concentrated supplier: candidate's existing + newly-allocated
        value must stay under max_share_target of the importer's total."""
        ceiling = max_share_target * importer_totals[importer]
        return max(0.0, ceiling - existing_value.get((importer, candidate), 0.0))

    arcs = _build_arcs(
        network,
        all_edges,
        demand_importers=demand.keys(),
        candidate_stats=candidate_stats,
        exclude_pairs=exclude_pairs,
        onboarding_cost_multiplier=onboarding_cost_multiplier,
        onboarding_lead_time_days=onboarding_lead_time_days,
        max_alloc_fn=max_alloc,
    )

    return {
        "success": True,
        "problem": RerouteProblem(
            scenario=scenario,
            displaced=displaced,
            demand=demand,
            supply=supply,
            arcs=arcs,
        ),
    }
