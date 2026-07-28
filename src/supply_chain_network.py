"""Core supply chain graph service.

Loads the cleaned trade network once and exposes a scenario-simulation method ('simulate_removal') designed
to be called repeatedl by an LLM as a tool."""

import difflib

from typing import Any

import networkx as nx
import pandas as pd

class SupplyChainNetwork:
    """
    Wraps the trade network graph and exposes what-if scenario simulations.

    Usage:
        network = SupplyChainNetwork(
            edges_path = "data/processed/cleaned_edges.csv",
            nodes_path = "data/processed/cleaned_nodes.csv"
        )
        result = network.simulate_removal(["USA"])
    """

    def __init__(self, edges_path: str, nodes_path: str):
        edges = pd.read_csv(edges_path)
        nodes =pd.read_csv(nodes_path)

        self.graph = nx.from_pandas_edgelist(
            edges, 
            source="source", 
            target="target", 
            edge_attr=["trade_value_usd", "trade_qty_kg", "share_of_reporter_total"],
            create_using=nx.DiGraph()
        )

        node_attrs = nodes.set_index("country").to_dict(orient="index")
        nx.set_node_attributes(self.graph, node_attrs)

        missing = set(self.graph.nodes() - set(nodes["country"]))
        if missing:
            raise ValueError(f"Graph nodes missing metadata in nodes CSV: {missing}")

        self.countries = sorted(self.graph.nodes())
        self.total_trade_value = self._total_value(self.graph)
        self.baseline = self._snapshot(self.graph)


    # --- Internal Helpers ---

    @staticmethod
    def _total_value(g: nx.DiGraph) -> float:
        return float(sum(d.get("trade_value_usd", 0) for _, _, d in g.edges(data=True)))

    @staticmethod
    def _snapshot(g: nx.DiGraph) -> dict[str, Any]:
        ccs = list(nx.weakly_connected_components(g)) if g.number_of_nodes() else []
        largest_cc = len(max(ccs, key=len)) if ccs else 0

        return {
            "n_edges": g.number_of_edges(),
            "n_nodes": g.number_of_nodes(),
            "n_components": len(ccs),
            "largest_component_size": largest_cc,
            "total_trade_value_usd": round(SupplyChainNetwork._total_value(g), 2)
        }

    def _resolve_country(self, name: str) -> tuple[str | None, list[str]]:
        """Exact match, case-insensitive match, or fuzzy suggestions."""
        if name in self.graph.nodes:
            return name, []

        lower_map = {c.lower(): c for c in self.countries}
        if name.lower() in lower_map:
            return lower_map[name.lower()], []

        suggestions = difflib.get_close_matches(name, self.countries, n=3, cutoff=0.6)
        return None, suggestions


    # --- LLM-callable API ---


    def list_countries(self) -> list[str]:
        """Return every country name known to the graph (for validating/grounding LLM input)."""
        return self.countries

    def simulate_removal(self, countries: list[str]) -> dict[str, Any]:
        """
        Simulate removing one or more countries from the trade network.

        Args:
            countries: list of country names to remove

        Returns:
            A JSON-serializable dict. 
            If a failure occurs, 'success' = False and 'error' explains what went wrong.
            If successfull, returns before/after network stats, the economic & structural impact, and which countries
            gained the most structural importance as a result.
        """

        # Make sure input countries can be found in the graph
        if not countries:
            return {"success": False, "error": "No countries provided.", "suggestions": []}

        resolved: list[str] = []
        unresolved: list[dict[str, Any]] = []

        for name in countries:
            match, suggestions = self._resolve_country(name)
            if match:
                resolved.append(match)
            else:
                unresolved.append({"input": name, "suggestions": suggestions})

        if unresolved:
            return {"success": False, "error": "One or more country names were not found in the network.", "unresolved": unresolved}


        # Create a copy of the graph to show before and after
        g_after = self.graph.copy()
        g_after.remove_nodes_from(resolved)
        after = self._snapshot(g_after)

        # Save key meetrics
        value_lost = round(self.baseline["total_trade_value_usd"] - after["total_trade_value_usd"], 2)
        pct_value_lost = round(value_lost / self.baseline["total_trade_value_usd"] * 100, 2)
        isolated = sorted(n for n in g_after.nodes() if g_after.degree(n) == 0)

        # Compare before and after betweenness centrality
        before_bc = nx.betweenness_centrality(self.graph)
        after_bc = nx.betweenness_centrality(g_after) if g_after.number_of_nodes() > 2 else {}
        shifts = []

        for country in g_after.nodes():
            delta = after_bc.get(country, 0.0) - before_bc.get(country, 0.0)
            if delta > 0:
                shifts.append({"country": country, "centrality_delta": round(delta, 4)})
        shifts.sort(key=lambda x: x["centrality_delta"], reverse=True)


        return {
            "success": True,
            "error": None,
            "scenario": {"removed_countries": resolved},
            "baseline": self.baseline,
            "after_removal": after,
            "impact": {
                "trade_value_lost_usd": value_lost,
                "pct_trade_value_lost": pct_value_lost,
                "components_before": self.baseline["n_components"],
                "components_after": after["n_components"],
                "largest_component_before": self.baseline["largest_component_size"],
                "largest_component_after": after["largest_component_size"],
                "network_fragmented": after["n_components"] > self.baseline["n_components"],
                "newly_isolated_countries": isolated,
            },
            "centrality_shifts": {
                "description": "Countries gaining structural importance (betweenness centrality) after removal, i.e. where risk cascades to.",
                "top_gainers": shifts[:5],
            },
        }

    def rank_vulnerability(self, top_n: int = 10) -> list[dict[str, Any]]:
        """
        Rank every country by combined structural & economic impact if it
        alone were removed. Useful for an LLM answering open-ended questions
        like "which countries are the biggest risk?" without needing to
        pick a specific country first.
        """
        rows = []

        for country in self.countries:
            result = self.simulate_removal([country])
            impact = result["impact"]
            rows.append(
                {
                    "country": country,
                    "components_after": impact["components_after"],
                    "pct_trade_value_lost": impact["pct_trade_value_lost"],
                    "n_isolated": len(impact["newly_isolated_countries"]),
                }
            )
        rows.sort(key=lambda r: (r["components_after"], r["pct_trade_value_lost"]), reverse=True)
        return rows[:top_n]
