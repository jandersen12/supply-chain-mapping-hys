"""Streamlit front end for the supply chain vulnerability network.

Single-page portfolio view: pick a shock scenario in the sidebar, everything
below (map, impact metrics, solver comparison, reroute detail) updates from
that one scenario at once.

Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.map_viz import COLOR_BLUE, COLOR_GREEN, COLOR_MAGENTA, COLOR_RED, COLOR_YELLOW, build_deck, count_unmappable_edges
from src.solver_ranking import rank_solvers
from src.solvers.compare import greedy_solver, run_solvers
from src.solvers import min_cost_flow, or_tools
from src.supply_chain_network import SupplyChainNetwork

st.set_page_config(page_title="Supply Chain Vulnerability Mapping", layout="wide")

CENTROIDS_PATH = "data/processed/country_centroids.csv"

st.markdown(
    f"""
    <style>
    .metric-card {{
        background: rgba(74, 125, 204, 0.08);
        border-left: 4px solid {COLOR_BLUE};
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }}
    .metric-card.danger {{ border-left-color: {COLOR_RED}; background: rgba(230, 31, 31, 0.08); }}
    .metric-card.warn {{ border-left-color: {COLOR_YELLOW}; background: rgba(252, 202, 51, 0.10); }}
    .metric-card.good {{ border-left-color: {COLOR_GREEN}; background: rgba(74, 208, 74, 0.08); }}
    .metric-card .label {{ font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.15rem; }}
    .metric-card .value {{ font-size: 1.9rem; font-weight: 700; line-height: 1.1; }}
    .solver-card {{
        border: 2px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .solver-card.winner {{ border-color: {COLOR_GREEN}; background: rgba(74, 208, 74, 0.10); }}
    .solver-card .name {{ font-size: 1.05rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }}
    .solver-card .badge {{ color: {COLOR_GREEN}; font-weight: 700; font-size: 0.8rem; }}
    .solver-card .stat {{ font-size: 1.3rem; font-weight: 700; margin-top: 0.4rem; }}
    .solver-card .stat-label {{ font-size: 0.75rem; opacity: 0.65; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_network() -> SupplyChainNetwork:
    return SupplyChainNetwork(
        edges_path="data/processed/cleaned_edges.csv",
        nodes_path="data/processed/cleaned_nodes.csv",
        centroids_path=CENTROIDS_PATH,
    )


@st.cache_resource
def load_centroids() -> pd.DataFrame:
    return pd.read_csv(CENTROIDS_PATH)


def metric_card(label: str, value: str, variant: str = "") -> str:
    css_class = f"metric-card {variant}".strip()
    return f'<div class="{css_class}"><div class="label">{label}</div><div class="value">{value}</div></div>'


def solver_card(row: pd.Series, is_winner: bool) -> str:
    css_class = "solver-card winner" if is_winner else "solver-card"
    badge = '<div class="badge">BEST FIT</div>' if is_winner else '<div class="badge">&nbsp;</div>'
    pct_covered = f"{row['pct_covered']:.1f}%" if pd.notna(row["pct_covered"]) else "N/A"
    return f"""
    <div class="{css_class}">
        {badge}
        <div class="name">{row['solver'].replace('_', ' ')}</div>
        <div class="stat">${row['objective_value']:,.0f}</div>
        <div class="stat-label">landed cost objective</div>
        <div class="stat">{pct_covered}</div>
        <div class="stat-label">demand covered</div>
        <div class="stat">{int(row['n_new_relationships'])}</div>
        <div class="stat-label">new relationships formed</div>
    </div>
    """


network = load_network()
centroids_df = load_centroids()

st.title("Supply Chain Vulnerability Mapping")
st.caption("Simulate a disruption to the global crude oil trade network and see how it ripples through, and how three different optimization methods would reroute around it.")

# --- Sidebar: single control panel driving the whole page ---
with st.sidebar:
    st.header("Scenario")
    shock_countries = st.multiselect(
        "Suppliers to shock (export ban, plant shutdown, shipping closure, etc.)",
        options=network.countries,
    )
    severity = st.slider(
        "Severity (fraction of export capacity lost)",
        min_value=0.1, max_value=1.0, value=0.6, step=0.05,
        help="Applied to every selected country. 1.0 = full export stop.",
    )
    capacity_multiplier = st.slider(
        "Rerouting capacity multiplier",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="How much additional trade value a replacement supplier can absorb, as a fraction of its current exports (e.g. 0.5 = up to 50% more).",
    )

shocks = {country: severity for country in shock_countries}

unmappable = count_unmappable_edges(network.graph, centroids_df)

if not shocks:
    st.info("Select one or more suppliers in the sidebar to simulate a disruption.")
    st.pydeck_chart(build_deck(network.graph, centroids_df))
    if unmappable:
        st.caption(f"{unmappable} trade relationships excluded from the map (no location data).")
    st.stop()

shock_result = network.simulate_shock(shocks)

if not shock_result["success"]:
    st.error(shock_result["error"])
    for u in shock_result.get("unresolved", []):
        if u["suggestions"]:
            st.write(f"**{u['input']}** — did you mean: {', '.join(u['suggestions'])}?")
    st.stop()

impact = shock_result["impact"]
g_after = network.shocked_graph(shocks)["graph"]

# --- Hero: map + big-number impact metrics ---
st.markdown("## Disruption impact")

map_col, metrics_col = st.columns([2.2, 1])
with map_col:
    st.pydeck_chart(build_deck(network.graph, centroids_df, highlight_countries=set(shocks)))
    st.caption(
        f"Red = shocked supplier(s) and their affected export flows. "
        + (f"{unmappable} trade relationships excluded (no location data)." if unmappable else "")
    )
with metrics_col:
    st.markdown(
        metric_card("Trade value lost", f"${impact['trade_value_lost_usd']:,.0f}", "danger"),
        unsafe_allow_html=True,
    )
    st.markdown(
        metric_card("% of network value lost", f"{impact['pct_trade_value_lost']:.1f}%", "danger"),
        unsafe_allow_html=True,
    )

if impact["newly_isolated_countries"]:
    st.warning("Isolated: " + ", ".join(impact["newly_isolated_countries"]))

gainers = shock_result["centrality_shifts"]["top_gainers"]
if gainers:
    with st.expander("Countries gaining structural importance (where risk cascades to)"):
        st.dataframe(gainers, width="stretch", hide_index=True)

# --- Solver comparison ---
st.markdown("## Rerouting: which solver finds the best fix?")
st.caption("Same scenario, three solving methods: a greedy heuristic, an exact min-cost-flow solver, and an exact LP solver (OR-Tools).")

solvers = {
    "greedy": greedy_solver(network),
    "min_cost_flow": min_cost_flow.solve,
    "or_tools": or_tools.solve,
}
solver_results = run_solvers(network, shocks, solvers, capacity_multiplier=capacity_multiplier)
comparison_df = pd.DataFrame([
    {
        "solver": name,
        "success": result.success,
        "objective_value": result.objective_value,
        "total_unmet_value_usd": result.total_unmet_value_usd,
        "pct_covered": result.pct_covered,
        "n_new_relationships": result.n_new_relationships,
        "runtime_seconds": result.runtime_seconds,
        "error": result.error,
    }
    for name, result in solver_results.items()
])
ranking = rank_solvers(comparison_df)

failed = comparison_df[~comparison_df["success"]]
if not failed.empty:
    for _, row in failed.iterrows():
        st.warning(f"**{row['solver']}** failed: {row['error']}")

if ranking["winner"] is not None:
    cards_html = '<div style="display:flex; gap:1rem;">'
    for _, row in ranking["table"].iterrows():
        cards_html += f'<div style="flex:1;">{solver_card(row, row["solver"] == ranking["winner"])}</div>'
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
    st.caption(
        f"**{ranking['winner'].replace('_', ' ')}** ranks best overall on landed cost, demand coverage, "
        "and fewest new relationships formed, combined."
    )

    winner_result = solver_results[ranking["winner"]]

    if winner_result.allocations:
        st.markdown(f"### Rerouted network ({ranking['winner'].replace('_', ' ')} — winning solver)")
        st.caption("Gray/blue = unaffected trade. Green = extra volume on an existing relationship. Magenta = a brand new trade relationship.")
        st.pydeck_chart(build_deck(g_after, centroids_df, reroute_allocations=winner_result.allocations))

        if winner_result.total_unmet_value_usd and winner_result.total_unmet_value_usd > 0:
            st.warning(
                f"${winner_result.total_unmet_value_usd:,.0f} of displaced trade value couldn't be fully "
                "rerouted at the current capacity multiplier. Try raising it in the sidebar."
            )

        with st.expander("Rerouting detail by importer"):
            allocations_df = pd.DataFrame(winner_result.allocations)
            for importer, group in allocations_df.groupby("importer"):
                total = group["allocated_value_usd"].sum()
                st.markdown(f"**{importer}** — ${total:,.0f} rerouted")
                st.dataframe(group.drop(columns=["importer"]), width="stretch", hide_index=True)
