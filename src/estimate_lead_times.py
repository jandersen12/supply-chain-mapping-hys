"""
Rule-based supplier lead-time estimation, for rerouting analysis.

There's no shipment-duration data in the trade dataset - Comtrade gives value
and quantity, not transit time. This estimates a rough lead time from two
placeholder assumptions:

    1. baseline transit + handling time, derived from distance_km using an
       assumed average transit speed
    2. an additional onboarding delay if the candidate is a new trade
       relationship (vendor qualification, contracting, first shipment setup)

Like estimate_tariffs.py, every value here is a rough placeholder, not a
measured lead time - flagged accordingly.
"""

# Fixed days for customs clearance / port handling, regardless of distance.
BASE_HANDLING_DAYS = 7.0

# Rough blended assumption covering sea/air/road at typical mix - not
# mode-specific (mode-of-transport data was investigated and shelved; see
# docs/mode_of_transport_investigation.md).
ASSUMED_TRANSIT_SPEED_KM_PER_DAY = 500.0

# Added on top of baseline transit time when the candidate is a new
# relationship, reflecting vendor qualification/contracting/first-shipment
# friction rather than shipping time itself.
DEFAULT_ONBOARDING_LEAD_TIME_DAYS = 45.0


def estimate_lead_time_days(
    distance_km: float | None,
    is_new_relationship: bool,
    onboarding_lead_time_days: float = DEFAULT_ONBOARDING_LEAD_TIME_DAYS,
) -> dict:
    """Estimate supplier lead time in days for an (importer, candidate) pair.

    Args:
        distance_km: great-circle distance between importer and candidate,
            or None if unknown (e.g. missing centroid).
        is_new_relationship: whether this candidate is not an existing
            supplier for the importer.
        onboarding_lead_time_days: additional days added for a new
            relationship. Set to 0 to disable the onboarding penalty
            entirely, same toggle pattern as capacity_multiplier.

    Returns:
        {
            "est_supplier_lead_time_days": float | None,
            "baseline_lead_time_days": float | None,
            "onboarding_lead_time_days_added": float,
            "is_placeholder_estimate": True,
        }
    """

    if distance_km is None:
        baseline = None
    else:
        baseline = BASE_HANDLING_DAYS + distance_km / ASSUMED_TRANSIT_SPEED_KM_PER_DAY

    added = onboarding_lead_time_days if is_new_relationship else 0.0
    total = None if baseline is None else baseline + added

    return {
        "est_supplier_lead_time_days": round(total, 1) if total is not None else None,
        "baseline_lead_time_days": round(baseline, 1) if baseline is not None else None,
        "onboarding_lead_time_days_added": added,
        "is_placeholder_estimate": True,
    }
