"""
Generate a country_centroids.csv lookup table (country, lat, lon) covering every
country name that appears in the cleaned nodes/edges tables.

Comtrade uses its own country naming convention (e.g. "USA", "Rep. of Korea",
"China, Hong Kong SAR", "Other Asia, nes" for Taiwan), so this script ships a
static centroid table keyed on those names plus a set of common aliases, instead of
depending on an external geocoding call.

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
#
# Two partner categories left out: "Areas, nes" and "Other Europe, nes" are 
# undisclosed-partner residual categories, not real countries. They're left to
# fall through to add_distance_km's NaN handling.

CENTROIDS = {
    "Albania": (41.1533, 20.1683),
    "Algeria": (28.0339, 1.6596),
    "Angola": (-11.2027, 17.8739),
    "Antigua and Barbuda": (17.0608, -61.7964),
    "Argentina": (-38.4161, -63.6167),
    "Armenia": (40.0691, 45.0382),
    "Aruba": (12.5211, -69.9683),
    "Australia": (-25.2744, 133.7751),
    "Austria": (47.5162, 14.5501),
    "Azerbaijan": (40.1431, 47.5769),
    "Bahamas": (25.0343, -77.3963),
    "Bahrain": (25.9304, 50.6378),
    "Barbados": (13.1939, -59.5432),
    "Belgium": (50.5039, 4.4699),
    "Belize": (17.1899, -88.4976),
    "Bolivia (Plurinational State of)": (-16.2902, -63.5887),
    "Br. Virgin Isds": (18.4207, -64.6400),
    "Brazil": (-14.2350, -51.9253),
    "Brunei Darussalam": (4.5353, 114.7277),
    "Bulgaria": (42.7339, 25.4858),
    "Burkina Faso": (12.2383, -1.5616),
    "Cabo Verde": (16.5388, -23.0418),
    "Cambodia": (12.5657, 104.9910),
    "Cameroon": (7.3697, 12.3547),
    "Canada": (56.1304, -106.3468),
    "Central African Rep.": (6.6111, 20.9394),
    "Chad": (15.4542, 18.7322),
    "Chile": (-35.6751, -71.5430),
    "China": (35.8617, 104.1954),
    "China, Hong Kong SAR": (22.3193, 114.1694),
    "Colombia": (4.5709, -74.2973),
    "Congo": (-0.2280, 15.8277),
    "Costa Rica": (9.7489, -83.7534),
    "Croatia": (45.1000, 15.2000),
    "Cuba": (21.5218, -77.7812),
    "Curaçao": (12.1696, -68.9900),
    "Cyprus": (35.1264, 33.4299),
    "Czechia": (49.8175, 15.4730),
    "Côte d'Ivoire": (7.5400, -5.5471),
    "Dem. Rep. of the Congo": (-4.0383, 21.7587),
    "Denmark": (56.2639, 9.5018),
    "Dominican Rep.": (18.7357, -70.1627),
    "Ecuador": (-1.8312, -78.1834),
    "Egypt": (26.8206, 30.8025),
    "El Salvador": (13.7942, -88.8965),
    "Equatorial Guinea": (1.6508, 10.2679),
    "Estonia": (58.5953, 25.0136),
    "Eswatini": (-26.5225, 31.4659),
    "Fiji": (-17.7134, 178.0650),
    "Finland": (61.9241, 25.7482),
    "France": (46.2276, 2.2137),
    "Gabon": (-0.8037, 11.6094),
    "Georgia": (42.3154, 43.3569),
    "Germany": (51.1657, 10.4515),
    "Ghana": (7.9465, -1.0232),
    "Gibraltar": (36.1408, -5.3536),
    "Greece": (39.0742, 21.8243),
    "Greenland": (71.7069, -42.6043),
    "Grenada": (12.1165, -61.6790),
    "Guatemala": (15.7835, -90.2308),
    "Guyana": (4.8604, -58.9302),
    "Hungary": (47.1625, 19.5033),
    "Iceland": (64.9631, -19.0208),
    "India": (20.5937, 78.9629),
    "Indonesia": (-0.7893, 113.9213),
    "Iran": (32.4279, 53.6880),
    "Iraq": (33.2232, 43.6793),
    "Ireland": (53.1424, -7.6921),
    "Israel": (31.0461, 34.8516),
    "Italy": (41.8719, 12.5674),
    "Jamaica": (18.1096, -77.2975),
    "Japan": (36.2048, 138.2529),
    "Jordan": (30.5852, 36.2384),
    "Kazakhstan": (48.0196, 66.9237),
    "Kenya": (-0.0236, 37.9062),
    "Kuwait": (29.3117, 47.4818),
    "Kyrgyzstan": (41.2044, 74.7661),
    "Latvia": (56.8796, 24.6032),
    "Lesotho": (-29.6100, 28.2336),
    "Liberia": (6.4281, -9.4295),
    "Libya": (26.3351, 17.2283),
    "Lithuania": (55.1694, 23.8813),
    "Luxembourg": (49.8153, 6.1296),
    "Madagascar": (-18.7669, 46.8691),
    "Malawi": (-13.2543, 34.3015),
    "Malaysia": (4.2105, 101.9758),
    "Maldives": (3.2028, 73.2207),
    "Malta": (35.9375, 14.3754),
    "Marshall Isds": (7.1315, 171.1845),
    "Mauritius": (-20.3484, 57.5522),
    "Mexico": (23.6345, -102.5528),
    "Mongolia": (46.8625, 103.8467),
    "Montserrat": (16.7425, -62.1874),
    "Morocco": (31.7917, -7.0926),
    "Mozambique": (-18.6657, 35.5296),
    "Myanmar": (21.9162, 95.9560),
    "Namibia": (-22.9576, 18.4904),
    "Netherlands": (52.1326, 5.2913),
    "New Caledonia": (-20.9043, 165.6180),
    "New Zealand": (-40.9006, 174.8860),
    "Nicaragua": (12.8654, -85.2072),
    "Niger": (17.6078, 8.0817),
    "Nigeria": (9.0820, 8.6753),
    "Norway": (60.4720, 8.4689),
    "Oman": (21.4735, 55.9754),
    "Pakistan": (30.3753, 69.3451),
    "Panama": (8.5380, -80.7821),
    "Papua New Guinea": (-6.3149, 143.9555),
    "Paraguay": (-23.4425, -58.4438),
    "Peru": (-9.1900, -75.0152),
    "Philippines": (12.8797, 121.7740),
    "Poland": (51.9194, 19.1451),
    "Portugal": (39.3999, -8.2245),
    "Qatar": (25.3548, 51.1839),
    "Rep. of Korea": (35.9078, 127.7669),
    "Rep. of Moldova": (47.4116, 28.3699),
    "Romania": (45.9432, 24.9668),
    "Russian Federation": (61.5240, 105.3188),
    "Samoa": (-13.7590, -172.1046),
    "Sao Tome and Principe": (0.1864, 6.6131),
    "Saudi Arabia": (23.8859, 45.0792),
    "Senegal": (14.4974, -14.4524),
    "Serbia": (44.0165, 21.0059),
    "Singapore": (1.3521, 103.8198),
    "Slovakia": (48.6690, 19.6990),
    "Slovenia": (46.1512, 14.9955),
    "South Africa": (-30.5595, 22.9375),
    "South Sudan": (6.8770, 31.3070),
    "Spain": (40.4637, -3.7492),
    "Sri Lanka": (7.8731, 80.7718),
    "Sudan": (12.8628, 30.2176),
    "Sweden": (60.1282, 18.6435),
    "Switzerland": (46.8182, 8.2275),
    "Taiwan": (23.6978, 120.9605),
    "Thailand": (15.8700, 100.9925),
    "Timor-Leste": (-8.8742, 125.7275),
    "Togo": (8.6195, 0.8248),
    "Trinidad and Tobago": (10.6918, -61.2225),
    "Tunisia": (33.8869, 9.5375),
    "Turkmenistan": (38.9697, 59.5563),
    "Türkiye": (38.9637, 35.2433),
    "USA": (37.0902, -95.7129),
    "Uganda": (1.3733, 32.2903),
    "Ukraine": (48.3794, 31.1656),
    "United Arab Emirates": (23.4241, 53.8478),
    "United Kingdom": (55.3781, -3.4360),
    "United Rep. of Tanzania": (-6.3690, 34.8888),
    "Uruguay": (-32.5228, -55.7658),
    "Uzbekistan": (41.3775, 64.5853),
    "Venezuela": (6.4238, -66.5897),
    "Viet Nam": (14.0583, 108.2772),
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
    # Comtrade reports Taiwan under the "Other Asia, nes" label; data_cleaning.py
    # renames it to "Taiwan" before this script ever sees it, but both aliases
    # are kept here in case this script is ever run against older raw data.
    "Other Asia, nes": "Taiwan",
    "Chinese Taipei": "Taiwan",
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
