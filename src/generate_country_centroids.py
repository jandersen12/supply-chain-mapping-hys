"""
Generate a country_centroids.csv lookup table (country, lat, lon) covering every
country name that appears in the cleaned nodes/edges tables.

Comtrade uses its own country naming convention (e.g. "USA", "Rep. of Korea",
"China, Hong Kong SAR", "Other Asia, nes" for Taiwan), so this script ships a
static centroid table keyed on those names plus a set of common aliases, rather
than depending on an external geocoding call.

Input:
    data/processed/cleaned_nodes.csv (column: country)
    data/processed/cleaned_edges.csv (columns: source, target)

Output:
    data/processed/country_centroids.csv - one row per country with lat/lon.
"""

import sys
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("data/processed")

# Approximate geographic centroid (decimal degrees) for each country, keyed on
# the name as it should appear in the output file.
CENTROIDS = {
    "Australia": (-25.2744, 133.7751),
    "Austria": (47.5162, 14.5501),
    "Belgium": (50.5039, 4.4699),
    "Brazil": (-14.2350, -51.9253),
    "Bulgaria": (42.7339, 25.4858),
    "Canada": (56.1304, -106.3468),
    "Chile": (-35.6751, -71.5430),
    "China": (35.8617, 104.1954),
    "China, Hong Kong SAR": (22.3193, 114.1694),
    "Colombia": (4.5709, -74.2973),
    "Croatia": (45.1000, 15.2000),
    "Czechia": (49.8175, 15.4730),
    "Denmark": (56.2639, 9.5018),
    "Egypt": (26.8206, 30.8025),
    "El Salvador": (13.7942, -88.8965),
    "Estonia": (58.5953, 25.0136),
    "Finland": (61.9241, 25.7482),
    "France": (46.2276, 2.2137),
    "Germany": (51.1657, 10.4515),
    "Hungary": (47.1625, 19.5033),
    "Iceland": (64.9631, -19.0208),
    "India": (20.5937, 78.9629),
    "Ireland": (53.1424, -7.6921),
    "Israel": (31.0461, 34.8516),
    "Italy": (41.8719, 12.5674),
    "Japan": (36.2048, 138.2529),
    "Kazakhstan": (48.0196, 66.9237),
    "Lithuania": (55.1694, 23.8813),
    "Luxembourg": (49.8153, 6.1296),
    "Malaysia": (4.2105, 101.9758),
    "Morocco": (31.7917, -7.0926),
    "Netherlands": (52.1326, 5.2913),
    "New Zealand": (-40.9006, 174.8860),
    # Comtrade reports Taiwan under the "Other Asia, nes" label.
    "Other Asia, nes": (23.6978, 120.9605),
    "Philippines": (12.8797, 121.7740),
    "Poland": (51.9194, 19.1451),
    "Portugal": (39.3999, -8.2245),
    "Rep. of Korea": (35.9078, 127.7669),
    "Romania": (45.9432, 24.9668),
    "Russian Federation": (61.5240, 105.3188),
    "Singapore": (1.3521, 103.8198),
    "Slovakia": (48.6690, 19.6990),
    "Spain": (40.4637, -3.7492),
    "Sri Lanka": (7.8731, 80.7718),
    "Sweden": (60.1282, 18.6435),
    "Switzerland": (46.8182, 8.2275),
    "Thailand": (15.8700, 100.9925),
    "Türkiye": (38.9637, 35.2433),
    "USA": (37.0902, -95.7129),
    "Ukraine": (48.3794, 31.1656),
    "United Kingdom": (55.3781, -3.4360),
}

# Common alternate spellings/names mapped to the canonical key in CENTROIDS above.
ALIASES = {
    "United States": "USA",
    "United States of America": "USA",
    "South Korea": "Rep. of Korea",
    "Korea, Rep.": "Rep. of Korea",
    "Hong Kong": "China, Hong Kong SAR",
    "Hong Kong SAR": "China, Hong Kong SAR",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "Taiwan": "Other Asia, nes",
    "Chinese Taipei": "Other Asia, nes",
    "Russia": "Russian Federation",
}


def collect_country_names(nodes_path: Path, edges_path: Path) -> set[str]:
    """Gather the union of country names referenced in the cleaned nodes/edges tables."""

    names = set()

    if nodes_path.exists():
        nodes = pd.read_csv(nodes_path)
        names.update(nodes["country"].dropna().unique())

    if edges_path.exists():
        edges = pd.read_csv(edges_path)
        names.update(edges["source"].dropna().unique())
        names.update(edges["target"].dropna().unique())

    return names


def build_centroids_table(country_names: set[str]) -> pd.DataFrame:
    """Resolve each country name to a (lat, lon) pair, reporting any that can't be found."""

    rows = []
    unmatched = []

    for name in sorted(country_names):
        key = name if name in CENTROIDS else ALIASES.get(name)
        if key is None:
            unmatched.append(name)
            continue
        lat, lon = CENTROIDS[key]
        rows.append({"country": name, "lat": lat, "lon": lon})

    if unmatched:
        print(
            f"Warning: no centroid found for {len(unmatched)} countr"
            f"{'y' if len(unmatched) == 1 else 'ies'}: {unmatched}",
            file=sys.stderr,
        )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    country_names = collect_country_names(
        OUTPUT_DIR / "cleaned_nodes.csv", OUTPUT_DIR / "cleaned_edges.csv"
    )

    if not country_names:
        print(
            "No countries found in data/processed/cleaned_nodes.csv or "
            "cleaned_edges.csv. Run data_cleaning.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    centroids = build_centroids_table(country_names)
    centroids.to_csv(OUTPUT_DIR / "country_centroids.csv", index=False)

    print(f"Wrote {len(centroids):,} country centroids to {OUTPUT_DIR}/country_centroids.csv")


if __name__ == "__main__":
    main()
