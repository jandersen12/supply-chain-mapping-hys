# Decision: Shelve mode-of-transportation as a graph/solver dimension

**Date:** 2026-08-03
**Status:** Shelved (not pursuing for now)

## Context

UN Comtrade exposes mode of transportation (sea/air/road/etc.) via `breakdownMode: "plus"`
on the trade data API, instead of the `"classic"` mode originally used in `comtradeapi_data.py`.
The goal was to add mode as a true structural dimension on graph edges (modeled as a
`MultiDiGraph`, one parallel edge per mode between a country pair) and eventually as an
input to the rerouting solver's cost calculation, alongside the existing `distance_km` and
`estimated_tariff_pct` edge attributes.

## What we did

1. Re-pulled gallium (HS 811292) 2025 import data with `breakdownMode: "plus"`.
2. Investigated the response structure: mode is one of three fanning-out dimensions in
   `"plus"` mode (mode, customs procedure, secondary partner). Confirmed the existing
   `isAggregate` flag correctly isolates the one "leaf" (actually-reported) row per
   country pair regardless of which dimension is rolled up, so no new filtering logic
   was needed there.
3. Found that most reporters don't break out mode at all — only report a `TOTAL MOT`
   (mode-unspecified) leaf row.
4. Before building out the multigraph, tariff-style mode-cost lookup, and propagating the
   change through `supply_chain_network.py`, `app.py`, and both notebooks, checked how much
   real signal the mode breakdown would actually add.

## Findings

- **Row/pair coverage**: ~210 of 423 unique reporter-partner pairs (~50%) have *some*
  mode-level leaf row.
- **Value-weighted coverage is much worse: only 18.7%** of total trade value falls under
  pairs with real mode detail. The pairs that do report mode skew toward small, marginal
  trade flows.
- **7 of the top 10 importers by value have 0% mode coverage**, including the largest
  importer by a wide margin (USA), plus EU, Japan, Rep. of Korea, Belgium, Italy, and Hong
  Kong. Only Germany, UK, and Canada report mode, and each does so at or near 100% (it's a
  national reporting practice, not per-shipment randomness).

## Decision

Shelve mode-of-transportation as a graph/solver dimension. The countries with mode
coverage are not the countries that drive centrality/vulnerability results in this
project — a mode-aware solver would have essentially nothing to say about the scenarios
this project is built to answer (e.g. "what happens if the USA loses its gallium supply"),
while adding real structural complexity: `MultiDiGraph` conversion, a new mode-cost lookup
module, and propagation through `supply_chain_network.py`, `app.py`, `eda.ipynb`, and
`node_removal_analysis.ipynb`.

`distance_km` remains the transportation-cost proxy in the edge/solver model.

## Alternatives considered

1. **Drop entirely** (chosen) — revert `comtradeapi_data.py` to `breakdownMode: "classic"`,
   keep the existing tariff/distance model.
2. **Descriptive-only attribute** — keep mode as a display-only field (e.g. Streamlit
   tooltip) without touching graph structure or solver cost logic. Rejected as not worth
   the ingestion/cleaning changes given the coverage gap.
3. **Revisit later** — the coverage gap may be specific to gallium/2025 reporting
   patterns. A different commodity or a multi-year pull might show better mode reporting
   among major importers. Not investigated further; worth a quick coverage check before
   reopening this.

## To revisit this decision

Re-run the value-weighted coverage check (see conversation history / re-derive via
`isAggregate`-based leaf filtering + `motCode != 0`) against a candidate commodity/year
before re-investing engineering effort. If value-weighted coverage among top-10 importers
is materially better, the multigraph design explored here (mode as `MultiDiGraph` edge key,
not composite node key — preserves node-level semantics for centrality/vulnerability
analysis) is still the right structural approach.
