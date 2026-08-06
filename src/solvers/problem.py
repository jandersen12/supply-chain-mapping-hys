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

unit_cost_usd_per_kg = candidate's avg export price * (1 + tariff_pct),
optionally inflated by onboarding_cost_multiplier for a new relationship,
plus a distance-based freight cost add-on (see estimate_shipping_cost.py) so
a farther candidate isn't priced the same as an equally-cheap closer one.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..estimate_lead_times import estimate_lead_time_days
from ..estimate_shipping_cost import estimate_freight_cost_usd_per_kg
from ..estimate_tariffs import estimate_tariff_pct

if TYPE_CHECKING:
    from ..supply_chain_network import SupplyChainNetwork

ARC_COLUMNS = [
    "importer",
    "candidate",
    "unit_cost_usd_per_kg",
    "freight_cost_usd_per_kg",
    "is_new_trade_relationship",
    "tariff_pct",
    "tariff_methodology",
    "distance_km",
    "est_supplier_lead_time_days",
]


@dataclass
class RerouteProblem:
    """Solver-agnostic definition of a rerouting-optimization problem.

    Attributes:
        scenario: echoes every input parameter (shocks and each
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
    shocks: dict[str, float],
    capacity_multiplier: float = 0.3,
    onboarding_cost_multiplier: float = 0.0,
    onboarding_lead_time_days: float = 45.0,
) -> dict[str, Any]:
    """Build a RerouteProblem from the network for the given shock scenario.

    Mirrors find_rerouting_options' setup logic (displaced relationships,
    candidate cost/capacity, tariff lookup) but stops short of actually
    solving - it hands back the plain arcs/demand/supply structure every
    solver should consume, so they're all optimizing over identical inputs.

    Args:
        network: a loaded SupplyChainNetwork.
        shocks: country name -> severity (fraction of export capacity lost,
            in (0, 1]) - same semantics as find_rerouting_options. Only the
            displaced fraction of each affected relationship becomes demand;
            a shocked country stays in the candidate pool at its reduced
            remaining export value.
        capacity_multiplier, onboarding_cost_multiplier,
            onboarding_lead_time_days: same semantics and defaults as
            find_rerouting_options.

    Returns:
        {"success": True, "problem": RerouteProblem} on success, or
        {"success": False, "error": str, ...} on the same validation
        failures find_rerouting_options can hit (unknown countries, bad
        severity, bad capacity_multiplier).
    """

    if not shocks:
        return {"success": False, "error": "No shocks provided.", "suggestions": []}

    if capacity_multiplier <= 0:
        return {"success": False, "error": "capacity_multiplier must be > 0.", "suggestions": []}

    resolved, unresolved = network._resolve_shocks(shocks)

    if unresolved:
        return {
            "success": False,
            "error": "One or more country names were not found in the network, or had an invalid severity.",
            "unresolved": unresolved,
        }

    scenario = {
        "shocks": [{"country": c, "severity": s} for c, s in resolved.items()],
        "capacity_multiplier": capacity_multiplier,
        "onboarding_cost_multiplier": onboarding_cost_multiplier,
        "onboarding_lead_time_days": onboarding_lead_time_days,
    }

    all_edges = list(network.graph.edges(data=True))
    displaced_edges = [(u, v, d, resolved[v]) for u, v, d in all_edges if v in resolved]

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

    # Candidate supply: same aggregation as find_rerouting_options - a
    # shocked supplier keeps its retained (1 - severity) share rather than
    # being excluded outright.
    export_value_by_supplier: dict[str, float] = defaultdict(float)
    export_qty_by_supplier: dict[str, float] = defaultdict(float)
    for _, target, data in all_edges:
        retained = 1 - resolved.get(target, 0.0)
        if retained <= 0:
            continue
        export_value_by_supplier[target] += (data.get("trade_value_usd", 0) or 0) * retained
        qty = data.get("trade_qty_kg")
        if qty == qty:  # filters NaN
            export_qty_by_supplier[target] += qty * retained

    avg_unit_price: dict[str, float] = {}
    supply: dict[str, float] = {}
    for supplier, total_value in export_value_by_supplier.items():
        total_qty = export_qty_by_supplier.get(supplier, 0)
        if total_qty <= 0:
            continue  # no price basis to rank this candidate
        avg_unit_price[supplier] = total_value / total_qty
        supply[supplier] = total_value * capacity_multiplier

    existing_tariffs = {
        (u, v): d.get("estimated_tariff_pct")
        for u, v, d in all_edges
        if d.get("estimated_tariff_pct") == d.get("estimated_tariff_pct")
    }

    # Demand: aggregate displaced value per importer. Per-removed-supplier
    # detail is kept in `displaced` for reporting only.
    demand: dict[str, float] = defaultdict(float)
    displaced: list[dict[str, Any]] = []
    self_backfill_exclusions: set[tuple[str, str]] = set()
    for importer, removed_supplier, data, severity in displaced_edges:
        full_value = data.get("trade_value_usd", 0) or 0
        displaced_value = full_value * severity
        original_qty = data.get("trade_qty_kg")
        original_unit_price = (
            full_value / original_qty if original_qty == original_qty and original_qty else None
        )
        demand[importer] += displaced_value
        self_backfill_exclusions.add((importer, removed_supplier))
        displaced.append({
            "importer": importer,
            "removed_supplier": removed_supplier,
            "severity": severity,
            "full_trade_value_usd": round(full_value, 2),
            "displaced_value_usd": round(displaced_value, 2),
            "original_unit_price_usd_per_kg": round(original_unit_price, 4)
            if original_unit_price is not None else None,
        })

    # Arcs: every (importer, candidate) pair with a price basis, excluding
    # self-loops and a shocked supplier covering the exact relationship its
    # own shock displaced (it can still serve *other* importers - see
    # supply_chain_network.py's find_rerouting_options docstring).
    arc_rows = []
    for importer in demand:
        for candidate, unit_price in avg_unit_price.items():
            if candidate == importer or (importer, candidate) in self_backfill_exclusions:
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

            unit_cost = unit_price * (1 + tariff_pct)
            if is_new_relationship:
                unit_cost *= (1 + onboarding_cost_multiplier)

            distance_km = network._distance_km(importer, candidate)
            freight = estimate_freight_cost_usd_per_kg(distance_km)
            unit_cost += freight["freight_cost_usd_per_kg"]

            lead_time = estimate_lead_time_days(
                distance_km, is_new_relationship, onboarding_lead_time_days
            )

            arc_rows.append({
                "importer": importer,
                "candidate": candidate,
                "unit_cost_usd_per_kg": unit_cost,
                "freight_cost_usd_per_kg": freight["freight_cost_usd_per_kg"],
                "is_new_trade_relationship": is_new_relationship,
                "tariff_pct": tariff_pct,
                "tariff_methodology": tariff_methodology,
                "distance_km": distance_km,
                "est_supplier_lead_time_days": lead_time["est_supplier_lead_time_days"],
            })

    arcs = pd.DataFrame(arc_rows, columns=ARC_COLUMNS)

    return {
        "success": True,
        "problem": RerouteProblem(
            scenario=scenario,
            displaced=displaced,
            demand=dict(demand),
            supply=supply,
            arcs=arcs,
        ),
    }
