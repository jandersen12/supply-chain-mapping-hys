"""
Greedy rerouting heuristic: cheapest-available-supplier assignment.

Unlike min_cost_flow.py/or_tools.py, this doesn't consume a pre-built
RerouteProblem's flat arc list - it owns its own sequential
capacity-depletion loop, computing tariff/freight/distance for each
candidate on the fly against the live network. See _greedy_adapter in
compare.py for how its output gets reshaped into a comparable SolverResult.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from ..estimate_lead_times import estimate_lead_time_days
from ..estimate_shipping_cost import estimate_freight_cost_usd_per_kg
from ..estimate_tariffs import estimate_tariff_pct

if TYPE_CHECKING:
    from ..supply_chain_network import SupplyChainNetwork


def solve(
    network: "SupplyChainNetwork",
    shocks: dict[str, float],
    capacity_multiplier: float = 0.3,
    onboarding_cost_multiplier: float = 0.0,
    onboarding_lead_time_days: float = 45.0,
) -> dict[str, Any]:
    """
    For each importer that sourced this commodity from a shocked country,
    find the best replacement supplier(s) among the remaining network for
    the portion of that trade the shock displaces.

    Rerouting cost is modeled as landed unit cost: a candidate's average
    export unit price (trade_value_usd / trade_qty_kg across its existing
    trade) times (1 + the tariff the importer would pay that candidate),
    plus a distance-based freight cost add-on (see
    estimate_shipping_cost.py) so a farther candidate isn't ranked the
    same as an equally-priced closer one. If the importer already trades
    with the candidate, the existing estimated_tariff_pct is used;
    otherwise a tariff is estimated with the same rule engine that
    produced estimated_tariffs.csv (see estimate_tariffs.py) and flagged
    as such.

    Forming a brand-new supplier relationship carries real-world friction
    (vendor qualification, contracting, first-shipment setup) that an
    existing relationship doesn't. Two separate levers model this rather
    than one, since cost and time are different currencies and shouldn't
    be silently combined:
      - onboarding_cost_multiplier inflates a new candidate's landed unit
        cost by this fraction, same toggle pattern as capacity_multiplier.
        0.0 (default) disables it.
      - onboarding_lead_time_days is added to a new candidate's estimated
        lead time (see estimate_lead_times.py). It's reported per
        allocation but does not affect ranking/cost - lead time isn't
        folded into the dollar objective here to avoid stacking an
        invented $/day conversion on top of the tariff and onboarding-cost
        placeholders already in play.

    Candidates are capacity-constrained: each can absorb up to
    capacity_multiplier times its current total export value in
    *additional* rerouted trade, so a single cheap supplier can't be
    assigned everyone's lost volume. Displaced trade is processed
    largest-value-lost first and split across the cheapest available
    candidates (by landed unit cost) until covered or capacity runs out.
    This is a greedy heuristic, not a globally optimal assignment - it can
    be locally suboptimal when multiple importers compete for the same
    cheap candidate - but is fast and transparent for this dataset's size.

    Args:
        network: a loaded SupplyChainNetwork.
        shocks: country name -> severity (fraction of export capacity
            lost, in (0, 1]) - same semantics as simulate_shock. Only
            the displaced fraction of each affected trade relationship
            needs rerouting; the retained (1 - severity) portion stays
            with the existing supplier. A shocked country remains in the
            candidate pool as a supplier for *other* importers, at its
            reduced remaining export value, but can't be chosen to cover
            the specific relationship its own shock displaced - it
            shouldn't be able to backfill the exact volume it just lost.
        capacity_multiplier: how much additional trade value, as a
            multiple of a candidate's current total exports, it can
            absorb. Default 0.3 means a candidate can at most absorb 30%
            of its current export value in additional rerouted trade (short stock assumption).
        onboarding_cost_multiplier: fractional landed-cost penalty applied
            to candidates that aren't an existing supplier for the
            importer. Default 0.0 (no penalty).
        onboarding_lead_time_days: additional estimated lead-time days for
            a new relationship, on top of the distance-based baseline.
            Reported only, not used for ranking.

    Returns:
        A JSON-serializable dict. If a failure occurs, 'success' = False
        and 'error' explains what went wrong. If successful, returns, per
        displaced import relationship, the chosen replacement supplier(s),
        landed cost, estimated lead time, and any unmet shortfall, plus an
        overall summary.
    """

    if not shocks:
        return {"success": False, "error": "No shocks provided."}


    scenario_shocks = [{"country": c, "severity": s} for c, s in shocks.items()]

    all_edges = list(network.graph.edges(data=True))
    displaced = [(u, v, d, shocks[v]) for u, v, d in all_edges if v in shocks]

    if not displaced:
        return {
            "success": True,
            "error": None,
            "scenario": {"shocks": scenario_shocks, "capacity_multiplier": capacity_multiplier},
            "reroutes": [],
            "summary": {
                "n_displaced_relationships": 0,
                "total_displaced_value_usd": 0.0,
                "total_unmet_value_usd": 0.0,
                "pct_covered": None,
                "new_trade_relationships_formed": [],
            },
        }

    # Aggregate each remaining supplier's current export value/quantity, to
    # derive its average unit price and its rerouting capacity. A shocked
    # supplier keeps its retained (1 - severity) share of each edge's
    # value/qty rather than being excluded outright.
    export_value_by_supplier: dict[str, float] = defaultdict(float)
    export_qty_by_supplier: dict[str, float] = defaultdict(float)
    for _, target, data in all_edges:
        retained = 1 - shocks.get(target, 0.0)
        if retained <= 0:
            continue
        export_value_by_supplier[target] += (data.get("trade_value_usd", 0) or 0) * retained
        qty = data.get("trade_qty_kg")
        if qty == qty:  # filters NaN
            export_qty_by_supplier[target] += qty * retained

    candidate_stats = {}
    for supplier, total_value in export_value_by_supplier.items():
        total_qty = export_qty_by_supplier.get(supplier, 0)
        if total_qty <= 0:
            continue  # no price basis to rank this candidate
        candidate_stats[supplier] = {
            "avg_unit_price_usd_per_kg": total_value / total_qty,
            "remaining_capacity_usd": total_value * capacity_multiplier,
        }

    existing_tariffs = {
        (u, v): d.get("estimated_tariff_pct")
        for u, v, d in all_edges
        if d.get("estimated_tariff_pct") == d.get("estimated_tariff_pct")
    }

    displaced.sort(key=lambda e: (e[2].get("trade_value_usd", 0) or 0) * e[3], reverse=True)

    reroutes = []
    new_relationships: set[tuple[str, str]] = set()
    total_displaced_value = 0.0
    total_unmet_value = 0.0

    for importer, removed_supplier, data, severity in displaced:
        full_value = data.get("trade_value_usd", 0) or 0
        displaced_value = full_value * severity
        total_displaced_value += displaced_value

        original_qty = data.get("trade_qty_kg")
        original_unit_price = (
            full_value / original_qty if original_qty == original_qty and original_qty else None
        )

        ranked = []
        for candidate, stats in candidate_stats.items():
            if candidate == importer or candidate == removed_supplier or stats["remaining_capacity_usd"] <= 0:
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

            landed_unit_cost = stats["avg_unit_price_usd_per_kg"] * (1 + tariff_pct)
            if is_new_relationship:
                landed_unit_cost *= (1 + onboarding_cost_multiplier)

            distance_km = network._distance_km(importer, candidate)
            freight = estimate_freight_cost_usd_per_kg(distance_km)
            landed_unit_cost += freight["freight_cost_usd_per_kg"]

            lead_time = estimate_lead_time_days(
                distance_km, is_new_relationship, onboarding_lead_time_days
            )

            ranked.append({
                "candidate": candidate,
                "landed_unit_cost_usd_per_kg": landed_unit_cost,
                "freight_cost_usd_per_kg": freight["freight_cost_usd_per_kg"],
                "tariff_pct": tariff_pct,
                "tariff_methodology": tariff_methodology,
                "is_new_trade_relationship": is_new_relationship,
                "distance_km": distance_km,
                "est_supplier_lead_time_days": lead_time["est_supplier_lead_time_days"],
            })
        ranked.sort(key=lambda c: c["landed_unit_cost_usd_per_kg"])

        remaining_to_allocate = displaced_value
        allocations = []
        for candidate_info in ranked:
            if remaining_to_allocate <= 0:
                break
            candidate = candidate_info["candidate"]
            available = candidate_stats[candidate]["remaining_capacity_usd"]
            take = min(remaining_to_allocate, available)
            if take <= 0:
                continue

            candidate_stats[candidate]["remaining_capacity_usd"] -= take
            remaining_to_allocate -= take
            if candidate_info["is_new_trade_relationship"]:
                new_relationships.add((importer, candidate))

            allocations.append({
                "new_supplier": candidate,
                "allocated_value_usd": round(take, 2),
                "landed_unit_cost_usd_per_kg": round(candidate_info["landed_unit_cost_usd_per_kg"], 4),
                "freight_cost_usd_per_kg": candidate_info["freight_cost_usd_per_kg"],
                "tariff_pct": candidate_info["tariff_pct"],
                "tariff_methodology": candidate_info["tariff_methodology"],
                "is_new_trade_relationship": candidate_info["is_new_trade_relationship"],
                "distance_km": round(candidate_info["distance_km"], 1) if candidate_info["distance_km"] is not None else None,
                "est_supplier_lead_time_days": candidate_info["est_supplier_lead_time_days"],
            })

        unmet_value = round(remaining_to_allocate, 2)
        total_unmet_value += unmet_value

        reroutes.append({
            "importer": importer,
            "removed_supplier": removed_supplier,
            "severity": severity,
            "full_trade_value_usd": round(full_value, 2),
            "original_trade_value_usd": round(displaced_value, 2),
            "original_unit_price_usd_per_kg": round(original_unit_price, 4) if original_unit_price is not None else None,
            "allocations": allocations,
            "unmet_value_usd": unmet_value,
            "pct_covered": round((displaced_value - unmet_value) / displaced_value * 100, 2) if displaced_value else None,
        })

    return {
        "success": True,
        "error": None,
        "scenario": {
            "shocks": scenario_shocks,
            "capacity_multiplier": capacity_multiplier,
            "onboarding_cost_multiplier": onboarding_cost_multiplier,
            "onboarding_lead_time_days": onboarding_lead_time_days,
        },
        "reroutes": reroutes,
        "summary": {
            "n_displaced_relationships": len(displaced),
            "total_displaced_value_usd": round(total_displaced_value, 2),
            "total_unmet_value_usd": round(total_unmet_value, 2),
            "pct_covered": round((total_displaced_value - total_unmet_value) / total_displaced_value * 100, 2) if total_displaced_value else None,
            "new_trade_relationships_formed": sorted(f"{u} -> {v}" for u, v in new_relationships),
        },
    }
