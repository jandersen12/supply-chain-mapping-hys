"""
Process the raw data from UN Comtrade and clean it for analysis and graph building.

The cleaning process involves checks for the structural schema, missingness, units, and consistency.

Input:
    A raw comtrade export in csv with the standard Comtrade column schema.

OUtput:
    cleaned_edges.csv - reporter -> partner trade edges
    cleaned_nodes.csv - one row per country with total reported trade and reporting activity
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

OUTPUT_DIR = Path("data/processed")

# --- Load data ---

def load_raw(paths: list[str]):
    """Load one or more raw comtrade CSVs and concatenate them."""

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df['_source_file'] = Path(p).name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    return combined


# --- Drop Aggregate Rows ---

COUNTRY_NAME_FIXES = {"Other Asia, nes": "Taiwan"}      # rename for clarity
EXCLUDED_ENTITIES = {"European Union"}                  # total for EU duplicate totals for its member states


def filter_edges(df: pd.DataFrame):
    """Keep only bilateral records: one row per (reporter, partner) pair.

    Excludes 'World' partner rows, self-trade rows, and rows involving
    EXCLUDED_ENTITIES, and collapses each remaining pair down to its single
    totals row.

    UN Comtrade's "plus" breakdownMode fans every bilateral pair out across
    three extra dimensions:
        - mode of transport (motCode)
        - customs procedure (customsCode)
        - secondary partner (partner2Code)
    each with its own "TOTAL" value (0, "C00", 0 respectively). Keeping only the row
    where all three are at that value reduces each pair back down to the
    one row "classic" mode would have returned, but maintains the disaggregated rows for
    future use/analysis.

    Classic-mode files trivially satisfy this same condition on every row (those columns are always at
    their only/TOTAL value there), so this filter is safe regardless of
    which breakdownMode a file was pulled with.
    """

    df = df.replace({"reporterDesc": COUNTRY_NAME_FIXES, "partnerDesc": COUNTRY_NAME_FIXES})

    world_mask = df['partnerDesc'].str.strip().str.lower() == 'world'
    self_mask = df['reporterDesc'] == df['partnerDesc']
    excluded_mask = df['reporterDesc'].isin(EXCLUDED_ENTITIES) | df['partnerDesc'].isin(EXCLUDED_ENTITIES)
    totals_mask = (
        (df['motCode'] == 0) & (df['customsCode'] == 'C00') & (df['partner2Code'] == 0)
    )

    clean = df[~world_mask & ~self_mask & ~excluded_mask & totals_mask].copy()

    return clean


# --- Build nodes and edges ---

def build_edge_table(clean: pd.DataFrame):
    """Construct the final edge table: reporter -> partner, weighted."""

    edges = clean[[
        'reporterDesc',
        'partnerDesc',
        'refYear',
        'cmdCode',
        'primaryValue',
        'qty'
    ]].rename(columns={"reporterDesc": "source",
                       "partnerDesc": "target",
                       "refYear": "year",
                       "cmdCode": "cmd_code",
                       "primaryValue": "trade_value_usd",
                       "qty": "trade_qty_kg"
                       })

    # Aggregate will show up in every row of the dataframe so we can calculate share of total below
    reporter_year_totals = (
        edges.groupby(['source', 'year'])["trade_value_usd"].transform("sum")
    )

    # Calculate share of reporter's total for each row
    edges['share_of_reporter_total'] = np.where(
        reporter_year_totals > 0, edges['trade_value_usd'] / reporter_year_totals, np.nan
    )

    edges = edges.sort_values(['source', 'year', 'trade_value_usd'], ascending=[True, True, False])

    return edges.reset_index(drop=True)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points (vectorized)."""

    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    earth_radius_km = 6371.0
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def add_distance_km(edges: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Add a distance_km column to the edge table using country centroid lat/lon.

    Distance is the great-circle (haversine) distance between the source
    (reporter) and target (partner) country centroids. Rows whose source or
    target isn't present in the centroids table get NaN.
    """

    edges = edges.merge(
        centroids.rename(columns={"country": "source", "lat": "source_lat", "lon": "source_lon"}),
        on="source",
        how="left",
    )
    edges = edges.merge(
        centroids.rename(columns={"country": "target", "lat": "target_lat", "lon": "target_lon"}),
        on="target",
        how="left",
    )

    edges["distance_km"] = _haversine_km(
        edges["source_lat"], edges["source_lon"], edges["target_lat"], edges["target_lon"]
    )

    n_missing = edges["distance_km"].isna().sum()
    if n_missing:
        missing_countries = sorted(
            set(edges.loc[edges["source_lat"].isna(), "source"])
            | set(edges.loc[edges["target_lat"].isna(), "target"])
        )
        print(
            f"Warning: distance_km missing for {n_missing:,} edges, no centroid for: "
            f"{missing_countries}",
            file=sys.stderr,
        )

    return edges.drop(columns=["source_lat", "source_lon", "target_lat", "target_lon"])


def add_tariffs(edges: pd.DataFrame, tariffs: pd.DataFrame) -> pd.DataFrame:
    """Add estimated_tariff_pct (and its methodology/placeholder flag) to the edge table.

    Joined on (source, target). Tariff estimates are currently country-pair level
    (not year/commodity specific), so this assumes at most one tariff row per
    source-target pair.
    """

    dupes = tariffs.duplicated(subset=["source", "target"]).sum()
    if dupes:
        print(
            f"Warning: estimated_tariffs.csv has {dupes} duplicate (source, target) "
            "pairs; merge will fan out edges.",
            file=sys.stderr,
        )

    edges = edges.merge(tariffs, on=["source", "target"], how="left")

    n_missing = edges["estimated_tariff_pct"].isna().sum()
    if n_missing:
        missing_pairs = edges.loc[
            edges["estimated_tariff_pct"].isna(), ["source", "target"]
        ].drop_duplicates()
        print(
            f"Warning: no tariff estimate for {n_missing:,} edges: "
            f"{list(missing_pairs.itertuples(index=False, name=None))}",
            file=sys.stderr,
        )

    return edges


def build_node_table(edges: pd.DataFrame):
    """Construct a node table covering every country that appears as either a reporter or partner."""

    reporters = set(edges['source'].unique())
    partners = set(edges['target'].unique())
    all_countries = sorted(reporters | partners)

    reporter_stats = (edges.groupby('source').agg(
        total_import_value_usd=('trade_value_usd', 'sum'),
        total_import_qty_kg=('trade_qty_kg', 'sum'),
        n_partners=('target', 'nunique')
    ).reset_index().rename(columns={"source":"country"}))

    partner_stats = (edges.groupby("target").agg(
        n_reporters_sourcing_from=('source', 'nunique'),
        total_named_as_source_value_usd=("trade_value_usd", "sum")
    ).reset_index().rename(columns={"target": "country"}))

    nodes = pd.DataFrame({"country": all_countries})
    nodes = nodes.merge(reporter_stats, on="country", how="left")
    nodes = nodes.merge(partner_stats, on="country", how="left")
    nodes["is_reporter"] = nodes["country"].isin(reporters)
    nodes["is_partner_only"] = ~nodes["is_reporter"]

    return nodes.sort_values("total_import_value_usd", ascending=False, na_position="last").reset_index(drop=True)


# --- Main pipeline ---

def main(paths: list[str]):
    """Runs all functions above on the dataset and saves node and edge CSVs to output directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw(paths)

    filtered = filter_edges(raw)

    edges = build_edge_table(filtered)
    nodes = build_node_table(edges)

    edges.to_csv(OUTPUT_DIR/"cleaned_edges.csv", index=False)
    nodes.to_csv(OUTPUT_DIR/"cleaned_nodes.csv", index=False)

    print(f"Wrote {len(edges):,} edges and {len(nodes):,} nodes to {OUTPUT_DIR}/")


def main_add_distance():
    """Add distance_km to the existing cleaned_edges.csv using country_centroids.csv.

    Run after generate_country_centroids.py, since that script depends on
    cleaned_edges.csv/cleaned_nodes.csv already existing.
    """

    edges_path = OUTPUT_DIR / "cleaned_edges.csv"
    centroids_path = OUTPUT_DIR / "country_centroids.csv"

    if not centroids_path.exists():
        print(
            f"{centroids_path} not found. Run "
            "`python src/generate_country_centroids.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    edges = pd.read_csv(edges_path)
    centroids = pd.read_csv(centroids_path)

    edges = add_distance_km(edges, centroids)
    edges.to_csv(edges_path, index=False)

    print(f"Added distance_km to {len(edges):,} edges in {edges_path}")


def main_add_tariffs():
    """Add estimated_tariff_pct to the existing cleaned_edges.csv using estimated_tariffs.csv."""

    edges_path = OUTPUT_DIR / "cleaned_edges.csv"
    tariffs_path = OUTPUT_DIR / "estimated_tariffs.csv"

    if not tariffs_path.exists():
        print(f"{tariffs_path} not found.", file=sys.stderr)
        sys.exit(1)

    edges = pd.read_csv(edges_path)
    tariffs = pd.read_csv(tariffs_path)

    edges = add_tariffs(edges, tariffs)
    edges.to_csv(edges_path, index=False)

    print(f"Added estimated_tariff_pct to {len(edges):,} edges in {edges_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            f"Usage: python {Path(__file__).name} raw_file.csv [more_years.csv ...]\n"
            f"       python {Path(__file__).name} --add-distance\n"
            f"       python {Path(__file__).name} --add-tariffs"
        )
        sys.exit(1)

    if sys.argv[1] == "--add-distance":
        main_add_distance()
    elif sys.argv[1] == "--add-tariffs":
        main_add_tariffs()
    else:
        main(sys.argv[1:])