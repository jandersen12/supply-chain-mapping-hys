"""Core supply chain graph service.

Loads the cleaned trade network once and exposes a scenario-simulation method ('simulate_shock')."""

import math

from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .estimate_tariffs import derive_importer_default_rates

class SupplyChainNetwork:
    """
    Wraps the trade network graph and exposes what-if scenario simulations.

    Usage:
        network = SupplyChainNetwork(
            edges_path = "data/processed/cleaned_edges.csv",
            nodes_path = "data/processed/cleaned_nodes.csv"
        )
        result = network.simulate_shock({"USA": 1.0})
    """

    def __init__(self, edges_path: str, nodes_path: str, centroids_path: str = "data/processed/country_centroids.csv"):
        edges = pd.read_csv(edges_path)
        nodes =pd.read_csv(nodes_path)

        self.graph = nx.from_pandas_edgelist(
            edges,
            source="source",
            target="target",
            edge_attr=["trade_value_usd", "trade_qty_kg", "share_of_reporter_total", "distance_km", "estimated_tariff_pct"],
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

        self._importer_default_rates = derive_importer_default_rates(
            edges[["source", "target", "estimated_tariff_pct", "tariff_methodology"]]
        )

        # Lets rerouting logic compute distance between any two countries for candidate reroute pairs
        centroids_path = Path(centroids_path)
        self._centroids: dict[str, tuple[float, float]] = {}
        if centroids_path.exists():
            centroids_df = pd.read_csv(centroids_path)
            self._centroids = {row.country: (row.lat, row.lon) for row in centroids_df.itertuples()}


    # --- Internal Helpers ---

    @staticmethod
    def _total_value(g: nx.DiGraph) -> float:
        """Pulls the trade value from the attribute dictionary (d) from every edge in the graph (g).
        If it doesn't exist, defaults to 0 instead."""
        return float(sum(d.get("trade_value_usd", 0) for _, _, d in g.edges(data=True)))

    @staticmethod
    def _snapshot(g: nx.DiGraph) -> dict[str, Any]:
        """Returns a snapshot of the graph's (g) key attributes."""
        ccs = list(nx.weakly_connected_components(g)) if g.number_of_nodes() else []
        largest_cc = len(max(ccs, key=len)) if ccs else 0

        return {
            "n_edges": g.number_of_edges(),
            "n_nodes": g.number_of_nodes(),
            "n_components": len(ccs),
            "largest_component_size": largest_cc,
            "total_trade_value_usd": round(SupplyChainNetwork._total_value(g), 2)
        }


    def _distance_km(self, a: str, b: str) -> float | None:
        """Great-circle distance between two countries' centroids, or None if either is missing."""

        if a not in self._centroids or b not in self._centroids:
            return None

        lat1, lon1 = self._centroids[a]
        lat2, lon2 = self._centroids[b]
        lat1, lon1, lat2, lon2 = (math.radians(x) for x in (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * 6371.0 * math.asin(math.sqrt(h))


    def list_countries(self) -> list[str]:
        """Return every country name known to the graph (for validating/grounding LLM input)."""
        return self.countries

    
    # --- Shock Simulation Functions ---

    @staticmethod
    def _apply_shocks(g: nx.DiGraph, shocks: dict[str, float]) -> nx.DiGraph:
        """Return a copy of g with each country's export edges scaled down by its severity (edges scaled to zero are dropped)."""

        g_after = g.copy()
        edges_to_drop = []
        for country, severity in shocks.items():
            retained = 1 - severity
            for u, v, data in g_after.in_edges(country, data=True):
                if retained <= 0:
                    edges_to_drop.append((u, v))
                    continue
                data["trade_value_usd"] = data.get("trade_value_usd", 0) * retained
                qty = data.get("trade_qty_kg")
                if qty == qty:  # filters NaN
                    data["trade_qty_kg"] = qty * retained
        g_after.remove_edges_from(edges_to_drop)
        return g_after


    def shocked_graph(self, shocks: dict[str, float]) -> dict[str, Any]:
        """Return the post-shock graph for a validated country -> severity shock map.

        Exposed separately from simulate_shock so callers that just need the
        graph for visualization (e.g. app.py) don't have to duplicate the shock-application logic.

        Returns {"success": True, "graph": nx.DiGraph, "shocks": {...}} or {"success": False, "error": str}.
        """

        if not shocks:
            return {"success": False, "error": "No shocks provided."}

        return {"success": True, "graph": self._apply_shocks(self.graph, shocks), "shocks": shocks}


    def simulate_shock(self, shocks: dict[str, float]) -> dict[str, Any]:
        """
        Simulate one or more countries losing export capacity.

        The function scales down the trade_value_usd and trade_qty_kg of its export edges by its severity. 
        A country that loses 100% of its export capacity still shows up as an importer, it just stops exporting. 
        Edges scaled to zero are dropped from the after-shock graph so structural stats (components, isolation) still work.

        Args:
            shocks: country name -> severity, where severity is the fraction of that country's export capacity lost.

        Returns:
            A JSON-serializable dict.
            If a failure occurs, 'success' = False and 'error' explains what went wrong.
            If successfull, returns before/after network stats, the economic & structural impact, and which countries
            gained the most structural importance as a result.
        """

        built = self.shocked_graph(shocks)
        if not built["success"]:
            return {"success": False, "error": built["error"]}

        shocks = built["shocks"]
        g_after = built["graph"]
        after = self._snapshot(g_after)

        # Save key metrics
        value_lost = round(self.baseline["total_trade_value_usd"] - after["total_trade_value_usd"], 2)
        pct_value_lost = round(value_lost / self.baseline["total_trade_value_usd"] * 100, 2)
        isolated = sorted(n for n in g_after.nodes() if g_after.degree(n) == 0) # checks for any newly isolated countries

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
            "scenario": {"shocks": [{"country": c, "severity": s} for c, s in shocks.items()]},
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
                "description": "Countries where risk cascades to after shock (higher betweeeness centrality than before).",
                "top_gainers": shifts[:5],
            },
        }

    # --- Function to Rank Vulnerability of Country Loss ---

    def rank_vulnerability(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Rank every country by combined structural & economic impact if it were removed. """
        rows = []

        for country in self.countries:
            result = self.simulate_shock({country: 1.0})
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
    
