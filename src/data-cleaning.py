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

OUTPUT_DIR = Path("../data/processed/")

# --- Load data ---

def load_raw(paths: list[str]):
    """Load one or more raw comtrade CSVs and concatenate them."""

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df['_source_file'] = Path(p).name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    return combined, "Data loading complete."


