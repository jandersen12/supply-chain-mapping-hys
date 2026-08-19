# Resilient Supply Chains: A graph-based optimization approach

An interactive tool for finding chokepoints in a global trade network and testing how well different optimization strategies recover from losing one.

**[Live demo](https://supply-chain-mapping-hys.streamlit.app/) · Run locally: `streamlit run app.py`**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.51-FF4B4B)
![NetworkX](https://img.shields.io/badge/networkx-3.6-orange)

## Overview

Supply chains optimized for low costs tend to concentrate around a handful of suppliers, which makes them prone to disruption from shocks and slow to recover. This project builds a directed trade-value graph for a single commodity from UN Comtrade data, uses graph theory to identify which nodes are chokepoints, and then simulates a disruption at one of those points to see how different optimization approaches would reroute around it.

The current dataset models **crude oil (HS 2709), 2024**, covering 154 countries and ~1,040 reporter-partner trade relationships. The year 2024 was chosen since it contains the most recent and most complete collection of reported imports by country.

**Try it in the app:** simulate an export shock and watch trade value get displaced, then compare how a greedy heuristic, an exact min-cost-flow solver, and an exact LP solver (OR-Tools) each reroute the lost supply and what it costs.

## What it does

1. **Pulls and cleans trade data** from the UN Comtrade API for a chosen commodity, producing a clean edge list (bilateral trade value/quantity) and node list (per-country trade totals).
2. **Builds a directed graph** (`networkx`) where nodes are countries and edges are import relationships weighted by trade value, with tariff, distance (great circle distance), and freight cost attributes attached.
3. **Finds chokepoints** using structural graph metrics (articulation points, betweenness/closeness centrality, PageRank, and a systematic node removal vulnerability scan.
4. **Simulates disruptions**: apply a partial or full export shock to one or more countries and measure trade value lost, newly isolated countries, and where structural importance shifts to.
5. **Reroutes around the shock** with three interchangeable solvers sharing one problem definition (same arcs, costs, and capacity constraints), so the comparison is apples-to-apples:
   - **Greedy:** cheapest-available-supplier heuristic.
   - **Min-cost flow:** (`networkx`) a flow-network formulation that tries to find the cheapest way to send a resource through a directed graph.
   - **Linear program:** (Google OR-Tools) similar to min-cost-flow except frames the problem as a linear programming problem instead of a flow-based network.
   - Solvers are ranked on a combined score of landed-cost objective, % of demand covered, and number of new trade relationships required.
6. **The app is visualized** in a single-page Streamlit app with a world map of the network, key impact metrics, and solver comparisons.

## Methodology notes

Real shipment level cost, tariff, and lead-time data isn't available in the Comtrade dataset since it reports trade value and quantity only, so several inputs are rule-based estimates rather than measured values, and each one is documented and flagged in the data:

- **Tariffs**— priority-ordered rules (bilateral FTA → regional free-trade zone → importer-specific flat rate → income-group default), applied both to build `estimated_tariffs.csv` and, live, to any candidate reroute pair that doesn't already trade.
- **Freight cost**— a per-km rate anchored to a real VLCC crude-tanker benchmark (~$2.5/barrel over ~18,500km Persian Gulf → Asia).
- **Lead time** — transit time derived from the distance in kilometers between countries plus a fixed onboarding delay penalty for brand-new trade relationships.

These are called out explicitly in the code and in solver output (`is_placeholder_estimate` / `tariff_methodology` columns) rather than presented as measured figures. These are also real points of development for future iterations of this project, since finding data sources to fill in these metrics would create a more realistic model of the network.

## Design decisions

Two features were built, evaluated, and shelved rather than shipped and can be found in the docs/ folder:

- [`docs/mode_of_transport_investigation.md`](docs/mode_of_transport_investigation.md) — shipping mode (sea/air/road) was dropped as a graph dimension due to limited data coverage.
- [`docs/agentic_resilience_planner_shelved.md`](docs/agentic_resilience_planner_shelved.md) — an agentic layer (LLM-driven goal parsing → automated rebalancing search → narrated results) that was built through five phases, then reverted from `main` pending redesign.

## Project structure

```
supply-chain-mapping-hys/
├── app.py                          # Streamlit app
├── data/
│   ├── raw/                        # Raw UN Comtrade pulls (gitignored)
│   ├── processed/                  # Cleaned edges/nodes + derived tables
│   └── archive/                    # Earlier commodity (gallium) pull, kept for reference (gitignored)
├── src/
│   ├── comtradeapi_data.py         # Pulls raw trade data from the UN Comtrade API
│   ├── data_cleaning.py            # Raw data -> cleaned_edges.csv / cleaned_nodes.csv
│   ├── generate_country_centroids.py  # Country -> lat/lon lookup for the map
│   ├── estimate_tariffs.py         # Rule-based tariff estimation
│   ├── estimate_lead_times.py      # Rule-based lead-time estimation
│   ├── estimate_shipping_cost.py   # Distance-based freight cost proxy
│   ├── supply_chain_network.py     # Core graph service: shock simulation, vulnerability ranking, rerouting, greedy algorithm
│   ├── solver_ranking.py           # Rank-sum scoring across solvers
│   ├── map_viz.py                  # pydeck world map visualization
│   └── solvers/
│       ├── problem.py              # Solver-agnostic problem definition (arcs/demand/supply)
│       ├── compare.py              # Runs all solvers on one scenario, returns comparable results
│       ├── min_cost_flow.py        # networkx min-cost-flow solver
│       └── or_tools.py             # OR-Tools LP solver
├── notebooks/
│   ├── eda_gallium.ipynb           # Exploratory graph analysis (structure, centrality, communities)
│   ├── node_removal_analysis.ipynb # Before/after case study on the top structural chokepoint
│   └── oil_solver.ipynb            # Validation pass after switching commodities to crude oil
├── docs/                           # Design-decision write-ups (see above)
├── tests/
│   └── demo.py                     # Scripted walkthrough of SupplyChainNetwork's public API
├── requirements.txt                # libraries used in development
└── requirements.txt                # libraries used in deployment
```

## Data

Source: [UN Comtrade Database](https://comtradeplus.un.org/) via the Comtrade API (requires a free API key).

The original exploratory analysis (`notebooks/eda_gallium.ipynb`, `notebooks/node_removal_analysis.ipynb`) was built on gallium trade data since it was a smaller, more concentrated network that demonstrated the chokepoint-analysis methodology due to the gallium trade's reliance on China. The project later pivoted the live app to crude oil, which is a larger network, once the methodology was validated.

## Setup

```bash
git clone <repo-url>
cd supply-chain-mapping-hys
pip install -r requirements.txt          # or requirements-dev.txt for notebook/EDA extras
```

Create a `.env` file in the project root with a UN Comtrade API key:

```
UN_COMTRADE_API_KEY=your_key_here
```

### Rebuild the data from scratch (optional)

The app ships with processed data already in `data/processed/`, so this step isn't required to run it, but you can use for pulling a fresh Comtrade extract or switching commodities.

```bash
# Outputs raw csv 
mkdir -p data/raw data/processed
python3 src/comtradeapi_data.py

# Outputs cleaned nodes and edges csvs
python3 src/data_cleaning.py data/raw/<raw_file>.csv

# Outputs country centroids and tariff csvs
python3 src/generate_country_centroids.py
python3 src/estimate_tariffs.py

# Adds distance_km to cleaned_edges.csv (uses country_centroids.csv)
python3 src/data_cleaning.py --add-distance

# Adds estimated_tariff_pct to cleaned_edges.csv (uses estimated_tariffs.csv)
python3 src/data_cleaning.py --add-tariffs
```

### Run the app

```bash
streamlit run app.py
```

### Run the demo script

```bash
python3 -m tests.demo
```

## Tech stack

`pandas` · `networkx` · `streamlit` · `pydeck` · `Google OR-Tools` · UN Comtrade API
