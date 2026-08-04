"""
Benchmarking harness for comparing rerouting solvers.

Every solver ships as a plain function: solve(problem: RerouteProblem) -> SolverResult.
run_comparison() builds one RerouteProblem per scenario (see problem.py) and
runs every registered solver against that exact same problem, so a difference
in results reflects a difference in solving method, not a difference in input
data.

As solvers get built (min_cost_flow, or_tools, gurobi, stochastic), each
should live in its own module under solvers/ and expose a solve(problem)
function matching SolverFn's signature, then get passed into `solvers` here
alongside greedy_solver(network).
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from .problem import RerouteProblem, build_reroute_problem

if TYPE_CHECKING:
    from ..supply_chain_network import SupplyChainNetwork


@dataclass
class SolverResult:
    """Common output shape every solver adapter must return.

    Kept deliberately narrow so results from very different solving methods
    (greedy heuristic, exact MILP, scenario-based stochastic) can sit in the
    same comparison table.

    objective_value is sum(allocated_value_usd * unit_cost_usd_per_kg) across
    allocations - the same quantity a solver formulated against arcs.py would
    minimize. It's a relative comparison metric, not a literal dollar total:
    per problem.py's docstring, flow is tracked in trade-value USD rather
    than physical quantity, so this number isn't dimensionally "dollars
    spent," just a consistent basis for ranking solvers against each other.
    """

    solver_name: str
    success: bool
    allocations: list[dict[str, Any]] = field(default_factory=list)
    objective_value: float | None = None
    total_unmet_value_usd: float | None = None
    pct_covered: float | None = None
    n_new_relationships: int | None = None
    runtime_seconds: float | None = None
    error: str | None = None


SolverFn = Callable[[RerouteProblem], SolverResult]


def _greedy_adapter(problem: RerouteProblem, network: "SupplyChainNetwork") -> SolverResult:
    """Wraps the existing find_rerouting_options as a SolverResult.

    Reruns find_rerouting_options rather than consuming problem.arcs
    directly, since greedy owns its own sequential capacity-depletion loop
    (see supply_chain_network.py) rather than a flat arc list. It's still a
    fair comparison point: same network, same scenario params, so the same
    inputs an exact solver would see via `problem`.
    """

    result = network.find_rerouting_options(
        problem.scenario["removed_countries"],
        capacity_multiplier=problem.scenario["capacity_multiplier"],
        onboarding_cost_multiplier=problem.scenario["onboarding_cost_multiplier"],
        onboarding_lead_time_days=problem.scenario["onboarding_lead_time_days"],
    )
    if not result["success"]:
        return SolverResult(solver_name="greedy", success=False, error=result.get("error"))

    allocations = [
        {"importer": r["importer"], **alloc}
        for r in result["reroutes"]
        for alloc in r["allocations"]
    ]
    objective_value = sum(
        a["allocated_value_usd"] * a["landed_unit_cost_usd_per_kg"] for a in allocations
    )

    return SolverResult(
        solver_name="greedy",
        success=True,
        allocations=allocations,
        objective_value=round(objective_value, 2),
        total_unmet_value_usd=result["summary"]["total_unmet_value_usd"],
        pct_covered=result["summary"]["pct_covered"],
        n_new_relationships=len(result["summary"]["new_trade_relationships_formed"]),
    )


def greedy_solver(network: "SupplyChainNetwork") -> SolverFn:
    """Returns a solve(problem) function bound to this network, for
    inclusion in the `solvers` dict passed to run_comparison, e.g.
    solvers={"greedy": greedy_solver(network), "min_cost_flow": ...}.
    """
    return lambda problem: _greedy_adapter(problem, network)


def run_comparison(
    network: "SupplyChainNetwork",
    removed_countries: list[str],
    solvers: dict[str, SolverFn],
    capacity_multiplier: float = 0.3,
    onboarding_cost_multiplier: float = 0.0,
    onboarding_lead_time_days: float = 45.0,
) -> pd.DataFrame:
    """Run every solver in `solvers` against the same rerouting scenario.

    Args:
        network: loaded SupplyChainNetwork.
        removed_countries: scenario input, same semantics as
            find_rerouting_options.
        solvers: name -> solve(problem) -> SolverResult. Register new solvers
            here as they're built; include greedy_solver(network) as the
            baseline.
        capacity_multiplier, onboarding_cost_multiplier,
            onboarding_lead_time_days: shared problem constraints, applied
            identically to every solver via build_reroute_problem.

    Returns:
        One row per solver: success, objective_value, total_unmet_value_usd,
        pct_covered, n_new_relationships, runtime_seconds, error. A failing
        solver (missing dependency, infeasible problem, exception) shows up
        as a row with success=False rather than raising, so one broken
        solver doesn't block comparing the others.
    """

    built = build_reroute_problem(
        network,
        removed_countries,
        capacity_multiplier=capacity_multiplier,
        onboarding_cost_multiplier=onboarding_cost_multiplier,
        onboarding_lead_time_days=onboarding_lead_time_days,
    )
    if not built["success"]:
        raise ValueError(f"Could not build problem: {built.get('error')}")

    problem = built["problem"]

    rows = []
    for name, solver_fn in solvers.items():
        start = time.perf_counter()
        try:
            result = solver_fn(problem)
            result.runtime_seconds = round(time.perf_counter() - start, 4)
        except Exception as e:
            result = SolverResult(
                solver_name=name,
                success=False,
                error=str(e),
                runtime_seconds=round(time.perf_counter() - start, 4),
            )
        rows.append({
            "solver": name,
            "success": result.success,
            "objective_value": result.objective_value,
            "total_unmet_value_usd": result.total_unmet_value_usd,
            "pct_covered": result.pct_covered,
            "n_new_relationships": result.n_new_relationships,
            "runtime_seconds": result.runtime_seconds,
            "error": result.error,
        })

    return pd.DataFrame(rows)
