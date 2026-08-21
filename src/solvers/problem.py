"""
Solver-agnostic input contract for rerouting-optimization experiments.

build_reroute_problem() extracts the same arcs, demand, and supply features that any solver can use as input.

unit_cost_usd_per_kg = candidate's avg export price * (1 + tariff_pct) 
plus a distance-based freight cost add-on (see estimate_shipping_cost.py).
onboarding_cost_multiplier is applied for any new relationships. 
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
        scenario: input shock and capacity parameters.
        displaced: detailed one entry per (importer, removed_supplier) relationship that needs replacement supply.
        demand: importer -> total trade value (USD) needing replacement, summed across every removed supplier that importer lost. 
        supply: candidate -> capacity (USD) it can absorb in additional rerouted trade (current export value x capacity_multiplier).
        arcs: one row per pair every feasible (importer, candidate) pair with its cost and descriptive attributes.
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

    Args:
        network: a loaded SupplyChainNetwork.
        shocks: country name -> severity (fraction of export capacity lost in (0, 1])
        capacity_multiplier, onboarding_cost_multiplier, onboarding_lead_time_days

    Returns:
        {"success": True, "problem": RerouteProblem} on success, or
        {"success": False, "error": str} if shocks is empty.
    """

    if not shocks:
        return {"success": False, "error": "No shocks provided."}

    scenario = {
        "shocks": [{"country": c, "severity": s} for c, s in shocks.items()],
        "capacity_multiplier": capacity_multiplier,
        "onboarding_cost_multiplier": onboarding_cost_multiplier,
        "onboarding_lead_time_days": onboarding_lead_time_days,
    }

    all_edges = list(network.graph.edges(data=True))
    displaced_edges = [(u, v, d, shocks[v]) for u, v, d in all_edges if v in shocks]

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

    ## Find candidate suppliers from the remaining countries
    # From the countries in all_edges (excluding shocked country) save the total export value and qty
    export_value_by_supplier: dict[str, float] = defaultdict(float)
    export_qty_by_supplier: dict[str, float] = defaultdict(float)
    for _, target, data in all_edges:
        if target in shocks:
            continue
        export_value_by_supplier[target] += data.get("trade_value_usd", 0) or 0
        qty = data.get("trade_qty_kg")
        if qty == qty:  # filters NaN
            export_qty_by_supplier[target] += qty

    # Calculate candidate country's avg_unit_price and excess supply available
    avg_unit_price: dict[str, float] = {}
    supply: dict[str, float] = {}
    for supplier, total_value in export_value_by_supplier.items():
        total_qty = export_qty_by_supplier.get(supplier, 0)
        if total_qty <= 0:
            continue  # no price basis to rank this candidate
        avg_unit_price[supplier] = total_value / total_qty
        supply[supplier] = total_value * capacity_multiplier

    # Pull tariff data for existing relationships
    existing_tariffs = {
        (u, v): d.get("estimated_tariff_pct")
        for u, v, d in all_edges
        if d.get("estimated_tariff_pct") == d.get("estimated_tariff_pct")
    }

    ## Demand: aggregate displaced value per importer from displaced_edges
    demand: dict[str, float] = defaultdict(float)
    displaced: list[dict[str, Any]] = []
    for importer, removed_supplier, data, severity in displaced_edges:
        full_value = data.get("trade_value_usd", 0) or 0
        displaced_value = full_value * severity
        original_qty = data.get("trade_qty_kg")
        original_unit_price = (
            full_value / original_qty if original_qty == original_qty and original_qty else None
        )
        demand[importer] += displaced_value
        displaced.append({
            "importer": importer,
            "removed_supplier": removed_supplier,
            "severity": severity,
            "full_trade_value_usd": round(full_value, 2),
            "displaced_value_usd": round(displaced_value, 2),
            "original_unit_price_usd_per_kg": round(original_unit_price, 4)
            if original_unit_price is not None else None,
        })

    ## Arcs: every (importer, candidate) pair with a price basis
    # Factors tariffs, new supplier relationships, and shipping distances into cost of rerouting 
    arc_rows = []
    for importer in demand:
        for candidate, unit_price in avg_unit_price.items():
            if candidate == importer:
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
