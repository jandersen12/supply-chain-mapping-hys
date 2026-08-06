"""
Rule-based shipping cost proxy, for rerouting analysis.

landed_unit_cost previously only accounted for unit price and tariff,
ignoring that a longer haul genuinely costs more to ship - two candidates
priced identically shouldn't rank the same if one is 500km away and the
other is 15,000km away. This adds a flat per-km freight rate as a rough
proxy, anchored to a real-world benchmark: VLCC crude tanker freight runs
roughly $2-3/barrel over an ~18,500 km Persian Gulf -> Asia haul.

Like estimate_tariffs.py and estimate_lead_times.py, this is a rough
placeholder, not a measured freight rate - flagged accordingly. A flat
per-km rate also doesn't capture real freight economics (long-haul rates
flatten out due to economies of scale, short-haul rates carry a fixed-cost
floor); FREIGHT_COST_CAP_USD_PER_KG exists specifically so a single
extreme-distance pair can't dominate the landed cost comparison as a result.
"""

# ~$2.5/barrel over ~18,500 km (Persian Gulf -> Asia VLCC benchmark).
# 1 barrel of crude oil = ~159 kg.
# $2.5 / (159 kg * 18,500 km) = ~$0.0000008 per kg per km.
FREIGHT_RATE_USD_PER_KG_KM = 0.0000008

# Cap on the freight add-on itself, not on distance - keeps a single
# extreme-distance candidate from swamping the tariff/price comparison.
FREIGHT_COST_CAP_USD_PER_KG = 0.05


def estimate_freight_cost_usd_per_kg(distance_km: float | None) -> dict:
    """Estimate a rough shipping cost add-on for a given distance.

    Args:
        distance_km: great-circle distance between importer and candidate,
            or None if unknown (e.g. missing centroid) - in that case no
            freight cost is added rather than guessing a distance.

    Returns:
        {"freight_cost_usd_per_kg": float, "is_placeholder_estimate": True}
    """

    if distance_km is None:
        return {"freight_cost_usd_per_kg": 0.0, "is_placeholder_estimate": True}

    cost = min(distance_km * FREIGHT_RATE_USD_PER_KG_KM, FREIGHT_COST_CAP_USD_PER_KG)
    return {"freight_cost_usd_per_kg": round(cost, 6), "is_placeholder_estimate": True}
