"""Core supply chain graph service.

Loads the cleaned trade network once and exposes a scenario-simulation method ('simulate_removal') designed
to be called repeatedl by an LLM as a tool."""

import difflib
import math

from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .estimate_lead_times import estimate_lead_time_days
from .estimate_tariffs import derive_importer_default_rates, estimate_tariff_pct

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

        # Used by find_rerouting_options: each importer's general tariff default
        # (for estimating tariffs on candidate pairs that don't already trade)
        # and country centroids (for distance between any pair, not just existing edges).
        self._importer_default_rates = derive_importer_default_rates(
            edges[["source", "target", "estimated_tariff_pct", "tariff_methodology"]]
        )

        centroids_path = Path(centroids_path)
        self._centroids: dict[str, tuple[float, float]] = {}
        if centroids_path.exists():
            centroids_df = pd.read_csv(centroids_path)
            self._centroids = {row.country: (row.lat, row.lon) for row in centroids_df.itertuples()}


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

    def find_rerouting_options(
        self,
        removed_countries: list[str],
        capacity_multiplier: float = 0.3,
        onboarding_cost_multiplier: float = 0.0,
        onboarding_lead_time_days: float = 45.0,
    ) -> dict[str, Any]:
        """
        For each importer that sourced this commodity from a country being removed,
        find the best replacement supplier(s) among the remaining network.

        Rerouting cost is modeled as landed unit cost: a candidate's average
        export unit price (trade_value_usd / trade_qty_kg across its existing
        trade) times (1 + the tariff the importer would pay that candidate).
        If the importer already trades with the candidate, the existing
        estimated_tariff_pct is used; otherwise a tariff is estimated with the
        same rule engine that produced estimated_tariffs.csv (see
        estimate_tariffs.py) and flagged as such.

        Forming a brand-new supplier relationship carries real-world friction
        (vendor qualification, contracting, first-shipment setup) that an
        existing relationship doesn't. Two separate levers model this rather
        than one, since cost and time are different currencies and shouldn't
        be silently combined:
          - onboarding_cost_multiplier inflates a new candidate's landed unit
            cost by this fraction, same toggle pattern as capacity_multiplier.
            0.0 (default) disables it.
          - onboarding_lead_time_days is added to a new candidate's estimated
            lead time (see estimate_lead_times.py). It's reported per
            allocation but does not affect ranking/cost - lead time isn't
            folded into the dollar objective here to avoid stacking an
            invented $/day conversion on top of the tariff and onboarding-cost
            placeholders already in play.

        Candidates are capacity-constrained: each can absorb up to
        capacity_multiplier times its current total export value in
        *additional* rerouted trade, so a single cheap supplier can't be
        assigned everyone's lost volume. Displaced trade is processed
        largest-value-lost first and split across the cheapest available
        candidates (by landed unit cost) until covered or capacity runs out.
        This is a greedy heuristic, not a globally optimal assignment - it can
        be locally suboptimal when multiple importers compete for the same
        cheap candidate - but is fast and transparent for this dataset's size.

        Args:
            removed_countries: countries whose export capacity is gone (e.g.
                an export ban) - same semantics as simulate_removal.
            capacity_multiplier: how much additional trade value, as a
                multiple of a candidate's current total exports, it can
                absorb. Default 0.3 means a candidate can at most absorb 30%
                of its current export value in additional rerouted trade (short stock assumption).
            onboarding_cost_multiplier: fractional landed-cost penalty applied
                to candidates that aren't an existing supplier for the
                importer. Default 0.0 (no penalty).
            onboarding_lead_time_days: additional estimated lead-time days for
                a new relationship, on top of the distance-based baseline.
                Reported only, not used for ranking.

        Returns:
            A JSON-serializable dict. If a failure occurs, 'success' = False
            and 'error' explains what went wrong. If successful, returns, per
            displaced import relationship, the chosen replacement supplier(s),
            landed cost, estimated lead time, and any unmet shortfall, plus an
            overall summary.
        """

        if not removed_countries:
            return {"success": False, "error": "No countries provided.", "suggestions": []}

        if capacity_multiplier <= 0:
            return {"success": False, "error": "capacity_multiplier must be > 0.", "suggestions": []}

        resolved: list[str] = []
        unresolved: list[dict[str, Any]] = []

        for name in removed_countries:
            match, suggestions = self._resolve_country(name)
            if match:
                resolved.append(match)
            else:
                unresolved.append({"input": name, "suggestions": suggestions})

        if unresolved:
            return {"success": False, "error": "One or more country names were not found in the network.", "unresolved": unresolved}

        all_edges = list(self.graph.edges(data=True))
        displaced = [(u, v, d) for u, v, d in all_edges if v in resolved]

        if not displaced:
            return {
                "success": True,
                "error": None,
                "scenario": {"removed_countries": resolved, "capacity_multiplier": capacity_multiplier},
                "reroutes": [],
                "summary": {
                    "n_displaced_relationships": 0,
                    "total_displaced_value_usd": 0.0,
                    "total_unmet_value_usd": 0.0,
                    "pct_covered": None,
                    "new_trade_relationships_formed": [],
                },
            }

        # Aggregate each remaining supplier's current export value/quantity, to
        # derive its average unit price and its rerouting capacity.
        export_value_by_supplier: dict[str, float] = defaultdict(float)
        export_qty_by_supplier: dict[str, float] = defaultdict(float)
        for _, target, data in all_edges:
            if target in resolved:
                continue
            export_value_by_supplier[target] += data.get("trade_value_usd", 0) or 0
            qty = data.get("trade_qty_kg")
            if qty == qty:  # filters NaN
                export_qty_by_supplier[target] += qty

        candidate_stats = {}
        for supplier, total_value in export_value_by_supplier.items():
            total_qty = export_qty_by_supplier.get(supplier, 0)
            if total_qty <= 0:
                continue  # no price basis to rank this candidate
            candidate_stats[supplier] = {
                "avg_unit_price_usd_per_kg": total_value / total_qty,
                "remaining_capacity_usd": total_value * capacity_multiplier,
            }

        existing_tariffs = {
            (u, v): d.get("estimated_tariff_pct")
            for u, v, d in all_edges
            if d.get("estimated_tariff_pct") == d.get("estimated_tariff_pct")
        }

        displaced.sort(key=lambda e: e[2].get("trade_value_usd", 0) or 0, reverse=True)

        reroutes = []
        new_relationships: set[tuple[str, str]] = set()
        total_displaced_value = 0.0
        total_unmet_value = 0.0

        for importer, removed_supplier, data in displaced:
            displaced_value = data.get("trade_value_usd", 0) or 0
            total_displaced_value += displaced_value

            original_qty = data.get("trade_qty_kg")
            original_unit_price = (
                displaced_value / original_qty if original_qty == original_qty and original_qty else None
            )

            ranked = []
            for candidate, stats in candidate_stats.items():
                if candidate == importer or stats["remaining_capacity_usd"] <= 0:
                    continue

                if (importer, candidate) in existing_tariffs:
                    tariff_pct = existing_tariffs[(importer, candidate)]
                    tariff_methodology = "existing_trade_relationship"
                    is_new_relationship = False
                else:
                    est = estimate_tariff_pct(importer, candidate, self._importer_default_rates)
                    tariff_pct = est["estimated_tariff_pct"]
                    tariff_methodology = est["tariff_methodology"]
                    is_new_relationship = True

                landed_unit_cost = stats["avg_unit_price_usd_per_kg"] * (1 + tariff_pct)
                if is_new_relationship:
                    landed_unit_cost *= (1 + onboarding_cost_multiplier)

                distance_km = self._distance_km(importer, candidate)
                lead_time = estimate_lead_time_days(
                    distance_km, is_new_relationship, onboarding_lead_time_days
                )

                ranked.append({
                    "candidate": candidate,
                    "landed_unit_cost_usd_per_kg": landed_unit_cost,
                    "tariff_pct": tariff_pct,
                    "tariff_methodology": tariff_methodology,
                    "is_new_trade_relationship": is_new_relationship,
                    "distance_km": distance_km,
                    "est_supplier_lead_time_days": lead_time["est_supplier_lead_time_days"],
                })
            ranked.sort(key=lambda c: c["landed_unit_cost_usd_per_kg"])

            remaining_to_allocate = displaced_value
            allocations = []
            for candidate_info in ranked:
                if remaining_to_allocate <= 0:
                    break
                candidate = candidate_info["candidate"]
                available = candidate_stats[candidate]["remaining_capacity_usd"]
                take = min(remaining_to_allocate, available)
                if take <= 0:
                    continue

                candidate_stats[candidate]["remaining_capacity_usd"] -= take
                remaining_to_allocate -= take
                if candidate_info["is_new_trade_relationship"]:
                    new_relationships.add((importer, candidate))

                allocations.append({
                    "new_supplier": candidate,
                    "allocated_value_usd": round(take, 2),
                    "landed_unit_cost_usd_per_kg": round(candidate_info["landed_unit_cost_usd_per_kg"], 4),
                    "tariff_pct": candidate_info["tariff_pct"],
                    "tariff_methodology": candidate_info["tariff_methodology"],
                    "is_new_trade_relationship": candidate_info["is_new_trade_relationship"],
                    "distance_km": round(candidate_info["distance_km"], 1) if candidate_info["distance_km"] is not None else None,
                    "est_supplier_lead_time_days": candidate_info["est_supplier_lead_time_days"],
                })

            unmet_value = round(remaining_to_allocate, 2)
            total_unmet_value += unmet_value

            reroutes.append({
                "importer": importer,
                "removed_supplier": removed_supplier,
                "original_trade_value_usd": round(displaced_value, 2),
                "original_unit_price_usd_per_kg": round(original_unit_price, 4) if original_unit_price is not None else None,
                "allocations": allocations,
                "unmet_value_usd": unmet_value,
                "pct_covered": round((displaced_value - unmet_value) / displaced_value * 100, 2) if displaced_value else None,
            })

        return {
            "success": True,
            "error": None,
            "scenario": {
                "removed_countries": resolved,
                "capacity_multiplier": capacity_multiplier,
                "onboarding_cost_multiplier": onboarding_cost_multiplier,
                "onboarding_lead_time_days": onboarding_lead_time_days,
            },
            "reroutes": reroutes,
            "summary": {
                "n_displaced_relationships": len(displaced),
                "total_displaced_value_usd": round(total_displaced_value, 2),
                "total_unmet_value_usd": round(total_unmet_value, 2),
                "pct_covered": round((total_displaced_value - total_unmet_value) / total_displaced_value * 100, 2) if total_displaced_value else None,
                "new_trade_relationships_formed": sorted(f"{u} -> {v}" for u, v in new_relationships),
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
