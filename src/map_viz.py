"""
Geographic (world map) visualization of the trade network via pydeck.

Replaces the force-directed pyvis layout with real country coordinates:
nodes are placed at their centroid lat/lon, edges are drawn as arcs between
them. Used by app.py for the single-page portfolio view.

Edges with no centroid for source and/or target (the "Areas, nes"/"Other
Europe, nes" residual categories) can't be placed on a map and are silently
skipped here - callers should surface the excluded count separately (see
count_unmappable_edges below) rather than let them disappear unremarked.
"""

import math
from typing import Any

import networkx as nx
import pandas as pd
import pydeck as pdk

# Portfolio color scheme, applied by semantic role rather than arbitrarily.
COLOR_RED = "#e61f1f"       # loss / disruption / shocked country
COLOR_YELLOW = "#fcca33"    # caution / partial coverage
COLOR_GREEN = "#4ad04a"     # winner / good outcome / existing-relationship reroute
COLOR_BLUE = "#4a7dcc"      # neutral / baseline
COLOR_MAGENTA = "#b11a7c"   # accent / new trade relationship


def _hex_to_rgba(hex_color: str, alpha: int = 180) -> list[int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]


def _arc_width(value: float) -> float:
    """Map a USD trade value (ranging from <$1 to $100B+ in this dataset) to a
    thin, readable arc width in pixels via a log scale. A linear/sqrt mapping
    blows up to hundreds of pixels for the largest flows and renders as solid
    filled ribbons instead of lines."""
    return min(3.5, max(0.4, 0.3 + math.log10(value + 1) * 0.3))


def _centroid_lookup(centroids_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {row.country: (row.lat, row.lon) for row in centroids_df.itertuples()}


def count_unmappable_edges(graph: nx.DiGraph, centroids_df: pd.DataFrame) -> int:
    """Count edges whose source or target has no centroid (can't be placed on the map)."""
    centroids = _centroid_lookup(centroids_df)
    return sum(1 for u, v in graph.edges() if u not in centroids or v not in centroids)


def _node_rows(graph: nx.DiGraph, centroids: dict, highlight: set[str]) -> list[dict[str, Any]]:
    rows = []
    for n in graph.nodes():
        if n not in centroids:
            continue
        lat, lon = centroids[n]
        value = graph.nodes[n].get("total_import_value_usd")
        value = value if value == value else 0  # filters NaN
        rows.append({
            "country": n,
            "lon": lon,
            "lat": lat,
            "radius": 20_000 + (value ** 0.5) * 3,
            "color": _hex_to_rgba(COLOR_RED, 220) if n in highlight else _hex_to_rgba(COLOR_BLUE, 180),
            "tooltip": f"{n}: ${value:,.0f} total imports",
        })
    return rows


def _base_edge_rows(
    graph: nx.DiGraph, centroids: dict, highlight: set[str], only_affected: bool = False
) -> list[dict[str, Any]]:
    """Baseline trade arcs. Target = exporter/partner (see supply_chain_network.py's
    edge-direction convention), so an arc is 'affected' when its target is shocked.

    only_affected=True (used once a shock scenario is selected) drops every
    unaffected (blue) arc entirely rather than just fading it - the map shows
    only the red, actually-disrupted flows instead of the full baseline
    network, which was competing for attention against them."""
    rows = []
    for u, v, data in graph.edges(data=True):
        if u not in centroids or v not in centroids:
            continue
        affected = v in highlight
        if only_affected and not affected:
            continue
        value = data.get("trade_value_usd", 0) or 0
        color = _hex_to_rgba(COLOR_RED, 140) if affected else _hex_to_rgba(COLOR_BLUE, 60)
        rows.append({
            "source_lon": centroids[u][1], "source_lat": centroids[u][0],
            "target_lon": centroids[v][1], "target_lat": centroids[v][0],
            "source": u, "target": v,
            "value": value,
            "width": _arc_width(value),
            "color": color,
            "tooltip": f"{u} imports from {v}: ${value:,.0f}",
        })
    return rows


def _reroute_edge_rows(centroids: dict, allocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build reroute arcs from a flat allocations list - the shape every
    solver's SolverResult.allocations uses (see solvers/compare.py), so this
    works identically regardless of which solver's result is passed in."""
    rows = []
    for alloc in allocations:
        importer = alloc["importer"]
        supplier = alloc["new_supplier"]
        if importer not in centroids or supplier not in centroids:
            continue
        is_new = alloc["is_new_trade_relationship"]
        rows.append({
            "source_lon": centroids[supplier][1], "source_lat": centroids[supplier][0],
            "target_lon": centroids[importer][1], "target_lat": centroids[importer][0],
            "source": supplier, "target": importer,
            "value": alloc["allocated_value_usd"],
            "width": _arc_width(alloc["allocated_value_usd"]),
            "color": _hex_to_rgba(COLOR_MAGENTA, 200) if is_new else _hex_to_rgba(COLOR_GREEN, 200),
            "tooltip": (
                f"Rerouted: {importer} <- {supplier}: ${alloc['allocated_value_usd']:,.0f}"
                f"{' (new relationship)' if is_new else ' (existing relationship, increased volume)'}"
            ),
        })
    return rows


def build_deck(
    graph: nx.DiGraph,
    centroids_df: pd.DataFrame,
    highlight_countries: set[str] | None = None,
    reroute_allocations: list[dict[str, Any]] | None = None,
) -> pdk.Deck:
    """Build a world-map pydeck Deck of the trade network.

    Args:
        graph: the network graph (baseline or post-shock) to draw.
        centroids_df: country_centroids.csv-shaped DataFrame.
        highlight_countries: countries to draw in red (e.g. shocked suppliers)
            and whose outgoing (export) arcs are drawn in red too. Once set,
            unaffected (blue) baseline arcs are dropped entirely rather than
            shown alongside them - only the actually-disrupted flows are
            drawn, plus the reroute overlay if present.
        reroute_allocations: optional flat list of allocation dicts (the
            shape of any SolverResult.allocations - importer, new_supplier,
            allocated_value_usd, is_new_trade_relationship, ...) to overlay
            as green (existing relationship) / magenta (new relationship)
            arcs. Works with any solver's result, not just greedy's. When
            set, baseline arcs are dropped entirely - only the reroute arcs
            are drawn.

    Returns:
        A pydeck.Deck ready for st.pydeck_chart.
    """

    centroids = _centroid_lookup(centroids_df)
    highlight = highlight_countries or set()

    node_rows = _node_rows(graph, centroids, highlight)
    if reroute_allocations:
        edge_rows = []  # reroute view: only the green/magenta reroute arcs, no baseline
    elif highlight:
        edge_rows = _base_edge_rows(graph, centroids, highlight, only_affected=True)  # shock view: only red
    else:
        edge_rows = _base_edge_rows(graph, centroids, highlight)  # landing page: full baseline
    reroute_rows = _reroute_edge_rows(centroids, reroute_allocations) if reroute_allocations else []

    layers = [
        pdk.Layer(
            "ArcLayer",
            data=edge_rows,
            get_source_position=["source_lon", "source_lat"],
            get_target_position=["target_lon", "target_lat"],
            get_source_color="color",
            get_target_color="color",
            get_width="width",
            width_min_pixels=1,
            width_max_pixels=4,
            pickable=True,
        ),
    ]

    if reroute_rows:
        layers.append(
            pdk.Layer(
                "ArcLayer",
                data=reroute_rows,
                get_source_position=["source_lon", "source_lat"],
                get_target_position=["target_lon", "target_lat"],
                get_source_color="color",
                get_target_color="color",
                get_width="width",
                width_min_pixels=1,
                width_max_pixels=5,
                pickable=True,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=node_rows,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
        )
    )

    view_state = pdk.ViewState(latitude=15, longitude=10, zoom=1.1, pitch=0)

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=None,
        tooltip={"text": "{tooltip}"},
    )
