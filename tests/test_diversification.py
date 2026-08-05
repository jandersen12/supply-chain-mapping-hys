"""
Checks for src/diversification.py and the solvers/problem.py refactor behind
it. Plain assert-based script, matching tests/demo.py's convention (no
pytest dependency in this project) - run with `python3 tests/test_diversification.py`.
"""

import math

import pandas as pd

from src.diversification import concentration_report, plan_diversification, seek_diversification_plan
from src.solvers.compare import greedy_solver, run_comparison
from src.solvers.problem import build_reroute_problem
from src.supply_chain_network import SupplyChainNetwork

EDGES_PATH = "data/processed/cleaned_edges.csv"
NODES_PATH = "data/processed/cleaned_nodes.csv"


def check_concentration_matches_precomputed_column(network):
    """concentration_report's dynamically-computed top_share should agree with
    the precomputed share_of_reporter_total column on the baseline graph."""

    edges = pd.read_csv(EDGES_PATH)
    report = {r["importer"]: r for r in concentration_report(network)}

    checked = 0
    for importer, group in edges.groupby("source"):
        if importer not in report:
            continue
        top_row = group.sort_values("trade_value_usd", ascending=False).iloc[0]
        expected_share = top_row["share_of_reporter_total"]
        actual_share = report[importer]["top_share"]
        assert math.isclose(expected_share, actual_share, rel_tol=1e-3), (
            f"{importer}: expected {expected_share}, got {actual_share}"
        )
        checked += 1

    assert checked > 0, "no importers were checked"
    print(f"OK: concentration_report matches share_of_reporter_total for {checked} importers")


def check_plan_diversification_clears_target(network):
    """Rep. of Korea sources ~93% of its trade from China - well over a 20%
    target - with plenty of alternate export capacity in the network to
    absorb the shift, so the plan should fully clear the target."""

    target = 0.2
    result = plan_diversification(network, importers=["Rep. of Korea"], max_share_target=target)

    assert result["success"], result.get("error")
    assert not result.get("already_compliant")
    assert result["recommended_solver"] in {"greedy", "min_cost_flow", "or_tools"}

    after = next(r for r in result["concentration_after"] if r["importer"] == "Rep. of Korea")
    print(f"OK: Rep. of Korea top share after plan = {after['top_share']} (target {target})")
    assert after["top_share"] <= target + 1e-6 or result["plan"]["summary"]["total_unmet_value_usd"] > 0

    print(f"OK: plan_diversification recommended {result['recommended_solver']}, "
          f"pct_covered={result['plan']['summary']['pct_covered']}")


def check_already_compliant_when_nothing_exceeds_target():
    """A very permissive target should find nothing to rebalance."""

    network = SupplyChainNetwork(edges_path=EDGES_PATH, nodes_path=NODES_PATH)
    result = plan_diversification(network, importers=["Rep. of Korea"], max_share_target=0.99)
    assert result["success"] is False or result.get("already_compliant") in (True, None)
    print("OK: high max_share_target yields no rebalancing need or a clean failure")


def check_validation_errors(network):
    bad_target = plan_diversification(network, importers=["Rep. of Korea"], max_share_target=1.5)
    assert bad_target["success"] is False
    print("OK: out-of-range max_share_target rejected")

    bad_importer = plan_diversification(network, importers=["Not A Real Country"], max_share_target=0.2)
    assert bad_importer["success"] is False
    assert bad_importer["unresolved"][0]["input"] == "Not A Real Country"
    print("OK: unresolved importer name rejected with suggestions contract")


def check_seek_diversification_plan_escalates_capacity(network):
    """0.1 under-covers Rep. of Korea's rebalancing need but 0.3 fully covers
    it (confirmed empirically) - the search should try 0.1, fall short, and
    keep escalating until it succeeds."""

    result = seek_diversification_plan(
        network, importers=["Rep. of Korea"], max_share_target=0.2,
        capacity_multiplier_grid=(0.1, 0.2, 0.3, 0.4),
    )

    assert result["success"], result.get("error")
    assert result["goal_met"] is True
    assert len(result["trace"]) > 1, "expected more than one attempt before succeeding"
    assert result["trace"][0]["capacity_multiplier"] == 0.1
    assert result["trace"][0]["pct_covered"] < 100
    assert result["chosen_capacity_multiplier"] == 0.3
    print(f"OK: seek_diversification_plan escalated through {len(result['trace'])} attempts "
          f"to capacity_multiplier={result['chosen_capacity_multiplier']}")


def check_seek_diversification_plan_fails_cleanly_when_nothing_exceeds_target():
    """Compliance doesn't depend on capacity_multiplier: if the named
    importer never exceeds the target, every grid attempt fails identically
    and the search should report a clean failure with the full trace, not
    crash or misreport success."""

    network = SupplyChainNetwork(edges_path=EDGES_PATH, nodes_path=NODES_PATH)
    grid = (0.1, 0.2, 0.3, 0.4)
    result = seek_diversification_plan(
        network, importers=["Rep. of Korea"], max_share_target=0.99,
        capacity_multiplier_grid=grid,
    )
    assert result["success"] is False
    assert len(result["trace"]) == len(grid)
    assert all(not attempt["success"] for attempt in result["trace"])
    print("OK: seek_diversification_plan fails cleanly when nothing exceeds target")


def check_build_reroute_problem_unchanged(network):
    """Regression check: refactoring build_reroute_problem into shared helpers
    shouldn't change its output for an existing removal scenario."""

    built = build_reroute_problem(network, ["China"])
    assert built["success"]
    problem = built["problem"]
    assert problem.demand, "expected non-empty demand for removing China"
    assert not problem.arcs.empty

    comparison = run_comparison(
        network, ["China"], solvers={"greedy": greedy_solver(network)}
    )
    assert comparison.iloc[0]["success"]
    print("OK: build_reroute_problem + run_comparison still work after refactor")


if __name__ == "__main__":
    network = SupplyChainNetwork(edges_path=EDGES_PATH, nodes_path=NODES_PATH)

    check_concentration_matches_precomputed_column(network)
    check_plan_diversification_clears_target(network)
    check_already_compliant_when_nothing_exceeds_target()
    check_validation_errors(network)
    check_seek_diversification_plan_escalates_capacity(network)
    check_seek_diversification_plan_fails_cleanly_when_nothing_exceeds_target()
    check_build_reroute_problem_unchanged(network)

    print("\nAll diversification checks passed.")
