"""Streamlit front end for the supply chain vulnerability network.

Run with: streamlit run app.py
"""

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from src.resilience_agent import get_client, run_resilience_goal
from src.supply_chain_network import SupplyChainNetwork

st.set_page_config(page_title="Supply Chain Vulnerability Mapping", layout="wide")


@st.cache_resource
def load_network() -> SupplyChainNetwork:
    return SupplyChainNetwork(
        edges_path="data/processed/cleaned_edges.csv",
        nodes_path="data/processed/cleaned_nodes.csv",
    )


@st.cache_resource
def get_layout(_g) -> dict:
    """Fixed node positions computed once from the full network, reused by every
    panel (before/after/reroute) so nodes never move between views and physics
    never has to re-settle on rerender."""
    return nx.spring_layout(_g, seed=42, k=0.6, iterations=200)


@st.cache_resource
def load_agent_client():
    return get_client()


def build_network_viz(g, layout, height="600px"):
    net = Network(height=height, width="100%", directed=True, notebook=True, cdn_resources="in_line")
    net.toggle_physics(False)

    for node in g.nodes():
        attrs = g.nodes[node]
        raw_value = attrs.get("total_import_value_usd")
        value = raw_value if raw_value == raw_value else 0  # filters NaN
        size = 10 + value / 5e6
        color = "#e74c3c" if attrs.get("is_partner_only") else "#3498db"
        x, y = layout[node]
        net.add_node(node, label=node, size=size, color=color, title=str(attrs), x=x * 800, y=y * 800)

    for u, v, attrs in g.edges(data=True):
        weight = attrs.get("trade_value_usd", 0)
        net.add_edge(u, v, value=weight / 1e5, title=f"${weight:,.0f}")

    return net.generate_html()


def build_reroute_viz(g_after, layout, reroutes, height="600px"):
    """Same base network as build_network_viz, with rerouted supply overlaid:
    green solid = extra volume on an existing trade relationship, orange dashed
    = a brand new trade relationship formed by the reroute."""

    net = Network(height=height, width="100%", directed=True, notebook=True, cdn_resources="in_line")
    net.toggle_physics(False)

    for node in g_after.nodes():
        attrs = g_after.nodes[node]
        raw_value = attrs.get("total_import_value_usd")
        value = raw_value if raw_value == raw_value else 0  # filters NaN
        size = 10 + value / 5e6
        color = "#e74c3c" if attrs.get("is_partner_only") else "#3498db"
        x, y = layout[node]
        net.add_node(node, label=node, size=size, color=color, title=str(attrs), x=x * 800, y=y * 800)

    for u, v, attrs in g_after.edges(data=True):
        weight = attrs.get("trade_value_usd", 0)
        net.add_edge(u, v, value=weight / 1e5, color="#cccccc", title=f"${weight:,.0f}")

    for reroute in reroutes:
        importer = reroute["importer"]
        for alloc in reroute["allocations"]:
            is_new = alloc["is_new_trade_relationship"]
            distance = f"{alloc['distance_km']:.0f} km" if alloc["distance_km"] is not None else "unknown"
            lead_time = (
                f"{alloc['est_supplier_lead_time_days']:.0f} days"
                if alloc["est_supplier_lead_time_days"] is not None
                else "unknown"
            )
            title = (
                f"Rerouted: {importer} <- {alloc['new_supplier']}<br>"
                f"${alloc['allocated_value_usd']:,.0f} allocated<br>"
                f"Landed cost: ${alloc['landed_unit_cost_usd_per_kg']:.2f}/kg<br>"
                f"Tariff: {alloc['tariff_pct'] * 100:.1f}% ({alloc['tariff_methodology']})<br>"
                f"Distance: {distance}<br>"
                f"Est. lead time: {lead_time}<br>"
                f"{'New trade relationship' if is_new else 'Existing trade relationship, increased volume'}"
            )
            net.add_edge(
                importer,
                alloc["new_supplier"],
                value=alloc["allocated_value_usd"] / 1e5,
                color="#e67e22" if is_new else "#27ae60",
                dashes=is_new,
                title=title,
            )

    return net.generate_html()


network = load_network()
layout = get_layout(network.graph)

st.title("Supply Chain Vulnerability Mapping")
st.caption("Simulate disruptions to a commodity trade network and see the structural and economic impact.")

tab_simulate, tab_rank, tab_reroute, tab_resilience = st.tabs(
    ["What-if simulation", "Vulnerability ranking", "Rerouting options", "Resilience planning"]
)

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
                components.html(build_network_viz(network.graph, layout, height="550px"), height=570, scrolling=True)
            with col_after:
                st.subheader("After removal")
                components.html(build_network_viz(g_after, layout, height="550px"), height=570, scrolling=True)

with tab_rank:
    top_n = st.slider("Number of countries to rank", min_value=5, max_value=len(network.countries), value=10)
    if st.button("Rank vulnerability"):
        rows = network.rank_vulnerability(top_n=top_n)
        st.dataframe(rows, use_container_width=True, hide_index=True)

with tab_reroute:
    st.caption("Find the best replacement suppliers for countries that lose their source when a supplier is removed.")

    reroute_countries = st.multiselect(
        "Suppliers to remove (export ban, plant shutdown, shipping closure, etc.)",
        options=network.countries,
        key="reroute_countries",
    )
    capacity_multiplier = st.slider(
        "Capacity multiplier",
        min_value=0.1, max_value=3.0, value=1.0, step=0.1,
        help="How much additional trade value a replacement supplier can absorb, as a multiple of its current total exports. 1.0 = can at most double its current exports.",
    )

    if st.button("Find rerouting options", type="primary", disabled=not reroute_countries):
        result = network.find_rerouting_options(reroute_countries, capacity_multiplier=capacity_multiplier)

        if not result["success"]:
            st.error(result["error"])
            for u in result.get("unresolved", []):
                if u["suggestions"]:
                    st.write(f"**{u['input']}** — did you mean: {', '.join(u['suggestions'])}?")
        else:
            summary = result["summary"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Displaced trade value", f"${summary['total_displaced_value_usd']:,.0f}")
            col2.metric("% covered by rerouting", f"{summary['pct_covered']:.1f}%" if summary["pct_covered"] is not None else "N/A")
            col3.metric("Unmet value", f"${summary['total_unmet_value_usd']:,.0f}")
            col4.metric("New trade relationships", len(summary["new_trade_relationships_formed"]))

            if summary["total_unmet_value_usd"] > 0:
                st.warning(
                    f"${summary['total_unmet_value_usd']:,.0f} of displaced trade value couldn't be fully "
                    "rerouted given the current capacity multiplier. Try raising it."
                )

            if summary["new_trade_relationships_formed"]:
                st.write("**New trade relationships formed:**", ", ".join(summary["new_trade_relationships_formed"]))

            st.subheader("Rerouted network")
            st.caption("Gray = existing trade. Green = extra volume on an existing relationship. Orange dashed = a brand new trade relationship.")
            g_after = network.graph.copy()
            g_after.remove_nodes_from(result["scenario"]["removed_countries"])
            components.html(build_reroute_viz(g_after, layout, result["reroutes"], height="600px"), height=620, scrolling=True)

            st.subheader("Rerouting detail by importer")
            for reroute in result["reroutes"]:
                header = f"{reroute['importer']} — lost {reroute['removed_supplier']} ({reroute['pct_covered']}% covered)"
                with st.expander(header):
                    price_note = (
                        f" (${reroute['original_unit_price_usd_per_kg']:.2f}/kg)"
                        if reroute["original_unit_price_usd_per_kg"] is not None else ""
                    )
                    st.write(f"Original trade value: ${reroute['original_trade_value_usd']:,.0f}{price_note}")
                    if reroute["allocations"]:
                        st.dataframe(reroute["allocations"], use_container_width=True, hide_index=True)
                    if reroute["unmet_value_usd"] > 0:
                        st.warning(f"${reroute['unmet_value_usd']:,.0f} unmet")

with tab_resilience:
    st.caption(
        "Describe a supplier-concentration goal in plain language. The agent searches for a "
        "rebalancing plan that meets it, trying increasingly aggressive capacity assumptions "
        "until the goal is met or it reports its best effort."
    )

    goal_text = st.text_area(
        "Goal",
        placeholder="e.g. No supplier should account for more than 20% of any importer's trade.",
    )

    if st.button("Find a plan", type="primary", disabled=not goal_text):
        try:
            agent_client = load_agent_client()
        except AssertionError as e:
            st.error(f"{e} Add ANTHROPIC_API_KEY to your .env file to use this tab.")
        else:
            with st.spinner("Searching for a diversification plan..."):
                result = run_resilience_goal(agent_client, network, goal_text)

            if not result["success"]:
                st.error(result.get("error"))
                for u in result.get("unresolved", []):
                    if u["suggestions"]:
                        st.write(f"**{u['input']}** — did you mean: {', '.join(u['suggestions'])}?")
            elif result.get("already_compliant"):
                st.success("Already compliant — no importer exceeds the target.")
            else:
                parsed = result.get("parsed_goal") or {}
                if parsed:
                    scope = (
                        f" for {', '.join(parsed['importers'])}" if parsed.get("importers") else " for any importer"
                    )
                    st.caption(
                        f"Interpreted goal: no supplier over {parsed['max_share_target'] * 100:.0f}% "
                        f"of trade value{scope}."
                    )

                st.subheader("Briefing")
                st.write(result["narrative"])

                summary = result["plan"]["summary"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Goal met", "Yes" if result["goal_met"] else "No")
                col2.metric(
                    "Capacity assumption used",
                    f"{result['chosen_capacity_multiplier']:.1f}" if result["chosen_capacity_multiplier"] else "—",
                )
                col3.metric("Recommended solver", result["recommended_solver"])
                col4.metric(
                    "% shift covered",
                    f"{summary['pct_covered']:.1f}%" if summary["pct_covered"] is not None else "N/A",
                )

                st.subheader("Search trace")
                st.dataframe(result["trace"], use_container_width=True, hide_index=True)

                st.subheader("Concentration before / after")
                before_map = {r["importer"]: r for r in result["concentration_before"]}
                after_map = {r["importer"]: r for r in result["concentration_after"]}
                combined = [
                    {
                        "importer": importer,
                        "top_supplier_before": before["top_supplier"],
                        "top_share_before": before["top_share"],
                        "top_share_after": after_map.get(importer, {}).get("top_share"),
                    }
                    for importer, before in before_map.items()
                ]
                st.dataframe(combined, use_container_width=True, hide_index=True)

                st.subheader("Rebalancing plan")
                st.caption("Gray = existing trade. Green = extra volume on an existing relationship. Orange dashed = a brand new trade relationship.")
                components.html(
                    build_reroute_viz(network.graph, layout, result["plan"]["reroutes"], height="600px"),
                    height=620,
                    scrolling=True,
                )

                st.subheader("Rebalancing detail by importer")
                for reroute in result["plan"]["reroutes"]:
                    header = f"{reroute['importer']} — {len(reroute['reduced_suppliers'])} supplier(s) reduced"
                    with st.expander(header):
                        st.write("**Reduced:**")
                        st.dataframe(reroute["reduced_suppliers"], use_container_width=True, hide_index=True)
                        if reroute["allocations"]:
                            st.write("**Reallocated to:**")
                            st.dataframe(reroute["allocations"], use_container_width=True, hide_index=True)
                        if reroute["unmet_value_usd"] > 0:
                            st.warning(f"${reroute['unmet_value_usd']:,.0f} unmet")
