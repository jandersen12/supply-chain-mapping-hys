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

def filter_edges(df: pd.DataFrame):
    """Keep only bilateral records.
    Exclude 'World' rows and isAggregate == True rows.
    """

    world_mask = df['partnerDesc'].str.strip().str.lower() == 'world'
    n_world = world_mask.sum()

    agg_mask = df['isAggregate'] == True
    n_agg_only = (agg_mask & ~world_mask).sum()

    self_mask = df['reporterDesc'] == df['partnerDesc']
    n_self = self_mask.sum()

    clean = df[~world_mask & ~agg_mask & ~self_mask].copy()

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} raw_file.csv [more_years.csv ...]")
        sys.exit(1)
    main(sys.argv[1:])