# Decision: Shelve the agentic resilience-planning layer

**Date:** 2026-08-05
**Status:** Shelved pending design reconsideration — code reverted from `main` (commit `71d5af4`, "Add agentic layer"), preserved in git history and summarized here.

## Context

Starting from the existing reactive tools (`simulate_removal`, `find_rerouting_options`, `rank_vulnerability` — "if a supplier is cut off, how bad is it?"), we scoped out an agentic layer to answer a proactive question instead: "are we over-concentrated on any single supplier, and how do we rebalance before anything breaks?"

Three initial ideas were considered:
1. **Autonomous risk-briefing agent** — scheduled/on-load `rank_vulnerability` → `simulate_removal` → `find_rerouting_options` pipeline with an LLM-written summary. Judged low on genuine agency (the LLM only narrates a fixed pipeline).
2. **Resilience-planning agent with a goal** — user states a target (e.g. "no supplier over 20%"), the agent iterates toward it. Chosen: this is the one with real autonomous iteration, and it directly extended the solver-comparison work already in progress (`c139b38`).
3. **Natural-language scenario translation** — map real-world events ("Red Sea closure") to affected countries, possibly via web search. Deferred: the failure mode (silently simulating the wrong country) was judged too risky to build without a human-confirmation step, and it wasn't clear it could be built on top of idea 2 without extra grounding work. Idea 2's entry point (`run_resilience_goal(goal_text)`) was deliberately kept generic so idea 3 could plug in later as an input adapter.

Idea 2 was built out in five phases.

## What we built

- **Phase 1–2 (concentration diagnostics + rebalancing):** `src/diversification.py` — `concentration_report()` computes each importer's per-supplier trade-value share dynamically (not from the precomputed CSV column, so it also works on hypothetical post-rebalance graphs); `plan_diversification()` builds a rebalancing problem (shift value away from any supplier exceeding a target share) and solves it via all three registered solvers. Required extending `src/solvers/problem.py` (`build_diversification_problem`, shared `_build_candidate_stats`/`_build_arcs` helpers factored out of `build_reroute_problem`) and adding `src/solvers/greedy.py` (a generic arcs/demand/supply solver, since the existing greedy adapter was hard-wired to the removal scenario via `find_rerouting_options`).
  - **A real bug caught during testing:** the first version could "fix" one over-concentrated supplier by just making another supplier the new dominant one. Fixed by adding a per-arc `max_alloc_usd` cap (enforced in all three solvers) so no candidate's post-allocation share can itself exceed the target.
- **Phase 3 (search loop):** `seek_diversification_plan()` in the same module — tries `capacity_multiplier` ascending through `(0.1, 0.2, 0.3, 0.4)` (deliberately capped well below 1.0 — a supplier realistically can't double its export volume), stopping at the first full-coverage attempt or returning the best effort with a trace of what was tried. Deliberately plain, deterministic Python, not an LLM loop.
- **Phase 4 (LLM boundary):** `src/resilience_agent.py` — `parse_diversification_goal()` (forced tool-call extraction of `{max_share_target, importers}` from free text) and `narrate_diversification_result()` (prose briefing over a trimmed JSON summary, explicitly instructed to ground every number in the data). `run_resilience_goal()` chains parse → search → narrate. Added the `anthropic` SDK dependency (`requirements.txt`) — the project's first real LLM call.
- **Phase 5 (UI):** a fourth Streamlit tab ("Resilience planning") in `app.py` — goal text box, narrative + metrics + trace + before/after concentration table + reused `build_reroute_viz` rendering.

Full per-phase design rationale is in the conversation history that produced this work; the plan documents themselves (`/Users/jordan/.claude/plans/groovy-bubbling-reef.md`, local to the machine this was built on, not committed) have the file-level detail if that machine is still available.

## Where the code still lives

Nothing is lost — `main` was reverted, not rewritten. To bring it back:

```
git show 71d5af4                # inspect the full diff
git cherry-pick 71d5af4          # reapply it onto a branch
# or, if main was reverted via `git revert <revert-commit>`:
git revert <revert-commit-sha>   # un-revert, restoring the code
```

## Open design questions to work through before revisiting

These surfaced while scoping and manually testing the feature, not fully resolved:

- **The goal-parsing schema has no "target supplier" concept.** A phrasing like "reduce dependence on China" is ambiguous under the current `{max_share_target, importers}` tool schema — the model may misplace "China" into `importers` (which means "apply the check to these importers") when the user actually means "specifically watch this supplier across all importers." Worth deciding whether that's a new schema field, a prompt fix, or out of scope.
- **Idea 3 (event → country mapping) is still unresolved.** Whether/how to add a web-search-backed translation layer in front of `run_resilience_goal`, with a human-confirmation step before it feeds a scenario into the search loop, was deferred rather than designed.
- **Modeling assumptions in the search loop** — the `(0.1, 0.2, 0.3, 0.4)` capacity-multiplier grid, "stop at first full coverage," and picking a solver purely by `pct_covered`/`objective_value` were reasonable starting points but not stress-tested against harder scenarios (e.g. many simultaneously over-concentrated importers competing for the same cheap alternate supplier).
- **Whether the search should vary more than `capacity_multiplier`** — e.g. `onboarding_cost_multiplier`, or which importers to prioritize when not everything is achievable — wasn't explored.
- **Whether the two-call LLM boundary (parse, then narrate) is the right shape**, versus a fuller tool-calling loop where the model can ask clarifying questions or inspect intermediate results.

## Alternatives considered for the revert itself

1. **`git revert` (chosen)** — non-destructive, preserves the original commit in history, safe on an already-pushed branch. No force-push required.
2. **`git reset --hard` + force-push** — rejected: rewrites shared history on `main`, which is not warranted here and is harder to undo cleanly if the decision changes again.
3. **Move to a feature branch instead of reverting `main`** — not chosen since the commit was already merged and pushed to `main`; simplest path is to revert now and branch from the preserved commit later if development resumes.

## To revisit this decision

1. Work through the open design questions above — particularly the goal-parsing ambiguity and idea 3's grounding strategy — before resuming implementation.
2. `git cherry-pick 71d5af4` (or revert the revert commit) onto a new branch.
3. Re-run `python3 tests/test_diversification.py` and `python3 tests/demo.py` to confirm the deterministic backend still passes.
4. Re-add `ANTHROPIC_API_KEY` to `.env` if it's been removed, and run `python3 tests/demo_resilience_agent.py` to sanity-check the LLM boundary before wiring the UI back in.
