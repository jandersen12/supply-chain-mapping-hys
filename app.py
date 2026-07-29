"""Streamlit front end for the supply chain vulnerability network.

Run with: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from src.supply_chain_network import SupplyChainNetwork

st.set_page_config(page_title="Supply Chain Vulnerability Mapping", layout="wide")


@st.cache_resource
def load_network() -> SupplyChainNetwork:
    return SupplyChainNetwork(
        edges_path="data/processed/cleaned_edges.csv",
        nodes_path="data/processed/cleaned_nodes.csv",
    )


def build_network_viz(g, height="600px"):
    net = Network(height=height, width="100%", directed=True, notebook=True, cdn_resources="in_line")

    for node in g.nodes():
        attrs = g.nodes[node]
        raw_value = attrs.get("total_import_value_usd")
        value = raw_value if raw_value == raw_value else 0  # filters NaN
        size = 10 + value / 5e6
        color = "#e74c3c" if attrs.get("is_partner_only") else "#3498db"
        net.add_node(node, label=node, size=size, color=color, title=str(attrs))

    for u, v, attrs in g.edges(data=True):
        weight = attrs.get("trade_value_usd", 0)
        net.add_edge(u, v, value=weight / 1e5, title=f"${weight:,.0f}")

    return net.generate_html()


network = load_network()

st.title("Supply Chain Vulnerability Mapping")
st.caption("Simulate disruptions to a commodity trade network and see the structural and economic impact.")

tab_simulate, tab_rank = st.tabs(["What-if simulation", "Vulnerability ranking"])

with tab_simulate:
    countries = st.multiselect(
        "Countries to remove (export ban, plant shutdown, shipping closure, etc.)",
        options=network.countries,
    )

    if st.button("Run simulation", type="primary", disabled=not countries):
        result = network.simulate_removal(countries)

        if not result["success"]:
            st.error(result["error"])
            for u in result.get("unresolved", []):
                if u["suggestions"]:
                    st.write(f"**{u['input']}** — did you mean: {', '.join(u['suggestions'])}?")
        else:
            impact = result["impact"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Trade value lost", f"${impact['trade_value_lost_usd']:,.0f}", f"-{impact['pct_trade_value_lost']}%")
            col2.metric("Network components", impact["components_after"], impact["components_after"] - impact["components_before"])
            col3.metric("Largest component size", impact["largest_component_after"], impact["largest_component_after"] - impact["largest_component_before"])
            col4.metric("Newly isolated countries", len(impact["newly_isolated_countries"]))

            if impact["network_fragmented"]:
                st.warning("This removal fragments the network into more disconnected components.")

            if impact["newly_isolated_countries"]:
                st.write("**Newly isolated countries:**", ", ".join(impact["newly_isolated_countries"]))

            gainers = result["centrality_shifts"]["top_gainers"]
            if gainers:
                st.subheader("Countries gaining structural importance (risk cascades to)")
                st.dataframe(gainers, use_container_width=True, hide_index=True)

            g_after = network.graph.copy()
            g_after.remove_nodes_from(result["scenario"]["removed_countries"])

            col_before, col_after = st.columns(2)
            with col_before:
                st.subheader("Before removal")
                components.html(build_network_viz(network.graph, height="550px"), height=570, scrolling=True)
            with col_after:
                st.subheader("After removal")
                components.html(build_network_viz(g_after, height="550px"), height=570, scrolling=True)

with tab_rank:
    top_n = st.slider("Number of countries to rank", min_value=5, max_value=len(network.countries), value=10)
    if st.button("Rank vulnerability"):
        rows = network.rank_vulnerability(top_n=top_n)
        st.dataframe(rows, use_container_width=True, hide_index=True)
