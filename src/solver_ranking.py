"""
Rank-sum scoring for solver comparison results (see solvers/compare.py).

Three metrics, three different units, and only ever a handful of solvers -
z-score normalization would be overkill and unstable at this scale. Instead
each solver is ranked 1st/2nd/... per metric (lower rank = better, so
pct_covered's ranking is inverted since higher is better there), and the
ranks are summed. Lowest total wins. Simple and easy to explain in a tooltip.
"""

import pandas as pd

# Lower value = better for these; pct_covered is higher = better (handled separately).
LOWER_IS_BETTER = ["objective_value", "n_new_relationships"]
HIGHER_IS_BETTER = ["pct_covered"]


def rank_solvers(comparison_df: pd.DataFrame) -> dict:
    """Rank solvers from a run_comparison() result.

    Disqualifies any solver with success=False (no valid metrics to rank).

    Returns:
        {
            "winner": str | None,  # solver name, or None if none succeeded
            "table": pd.DataFrame,  # ranked solvers with per-metric and total rank columns
        }
    """

    eligible = comparison_df[comparison_df["success"]].copy()
    if eligible.empty:
        return {"winner": None, "table": eligible}

    for metric in LOWER_IS_BETTER:
        eligible[f"rank_{metric}"] = eligible[metric].rank(method="min", ascending=True)
    for metric in HIGHER_IS_BETTER:
        eligible[f"rank_{metric}"] = eligible[metric].rank(method="min", ascending=False)

    rank_cols = [f"rank_{m}" for m in LOWER_IS_BETTER + HIGHER_IS_BETTER]
    eligible["total_rank"] = eligible[rank_cols].sum(axis=1)
    eligible = eligible.sort_values("total_rank")

    winner = eligible.iloc[0]["solver"]

    return {"winner": winner, "table": eligible}
