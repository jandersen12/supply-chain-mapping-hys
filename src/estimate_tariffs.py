"""
Rule-based tariff estimation, and estimated_tariffs.csv generation.

Two functions:

1. estimate_tariff_pct() applies a small priority-ordered rule set to any
   (importer, partner) pair:
       1. a specific bilateral FTA pair (e.g. USA <-> Canada, 0%)
       2. both countries in the EU/EEA/UK/CH free-trade zone (0%)
       3. a special flat rate for a handful of importers regardless of partner
          (Hong Kong as a free port, Singapore's low-tariff FTA network default)
       4. otherwise, that importer's general default rate, derived from a
          rough World Bank-style income classification (see
          COUNTRY_INCOME_GROUP below)
   Used both to generate the tariffs table below and then again at graph-load time, by
   SupplyChainNetwork.find_rerouting_options() for candidate pairs that don't
   already have a real trade relationship (see supply_chain_network.py).

2. build_tariff_table()/main() generates estimated_tariffs.csv from cleaned_edges.csv by applying estimate_tariff_pct() to every
   (source, target) pair in it. Run directly: `python src/estimate_tariffs.py`.
"""

from pathlib import Path

import pandas as pd

# Countries in dataset that are members of the EU/EEA/UK/CH free-trade zone.
EU_EEA_UK_CH_ZONE = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden", "Switzerland", "United Kingdom",
}

# Flat rate charged by these importers regardless of partner (i.e. Hong Kong as a free port, Singapore's FTA network).
SPECIAL_IMPORTER_RATES = {
    "China, Hong Kong SAR": (0.0, "hk_free_port"),
    "Singapore": (0.005, "sg_low_tariff_fta_network"),
}

# Known bilateral FTA pairs that override the general default (0% either way).
BILATERAL_FTA_PAIRS = {
    frozenset({"USA", "Canada"}): "bilateral_fta_bloc",
}

# Rough World Bank-style income classification per importer
# use to derive a baseline placeholder tariff rate when no more specific rule above applies

# Countries handled by SPECIAL_IMPORTER_RATES above are omitted

COUNTRY_INCOME_GROUP: dict[str, str] = {
    "Algeria": "lower_middle_income",
    "Angola": "lower_middle_income",
    "Antigua and Barbuda": "high_income",
    "Armenia": "upper_middle_income",
    "Australia": "high_income",
    "Austria": "high_income",
    "Azerbaijan": "upper_middle_income",
    "Bahamas": "high_income",
    "Bahrain": "high_income",
    "Belgium": "high_income",
    "Belize": "upper_middle_income",
    "Bolivia (Plurinational State of)": "lower_middle_income",
    "Brazil": "upper_middle_income",
    "Brunei Darussalam": "high_income",
    "Bulgaria": "high_income",
    "Burkina Faso": "low_income",
    "Cabo Verde": "lower_middle_income",
    "Cambodia": "lower_middle_income",
    "Canada": "high_income",
    "Central African Rep.": "low_income",
    "Chile": "high_income",
    "China": "upper_middle_income",
    "Colombia": "upper_middle_income",
    "Costa Rica": "upper_middle_income",
    "Croatia": "high_income",
    "Cyprus": "high_income",
    "Czechia": "high_income",
    "Côte d'Ivoire": "lower_middle_income",
    "Denmark": "high_income",
    "Dominican Rep.": "upper_middle_income",
    "Ecuador": "upper_middle_income",
    "Egypt": "lower_middle_income",
    "El Salvador": "lower_middle_income",
    "Estonia": "high_income",
    "Fiji": "upper_middle_income",
    "Finland": "high_income",
    "France": "high_income",
    "Georgia": "upper_middle_income",
    "Germany": "high_income",
    "Ghana": "lower_middle_income",
    "Greece": "high_income",
    "Grenada": "upper_middle_income",
    "Guatemala": "upper_middle_income",
    "Guyana": "high_income",
    "Hungary": "high_income",
    "Iceland": "high_income",
    "India": "lower_middle_income",
    "Indonesia": "upper_middle_income",
    "Ireland": "high_income",
    "Israel": "high_income",
    "Italy": "high_income",
    "Jamaica": "upper_middle_income",
    "Japan": "high_income",
    "Jordan": "upper_middle_income",
    "Kazakhstan": "upper_middle_income",
    "Kenya": "lower_middle_income",
    "Kyrgyzstan": "lower_middle_income",
    "Latvia": "high_income",
    "Lesotho": "lower_middle_income",
    "Lithuania": "high_income",
    "Luxembourg": "high_income",
    "Malawi": "low_income",
    "Malaysia": "upper_middle_income",
    "Maldives": "upper_middle_income",
    "Malta": "high_income",
    "Montserrat": "high_income",
    "Morocco": "lower_middle_income",
    "Mozambique": "low_income",
    "Namibia": "upper_middle_income",
    "Netherlands": "high_income",
    "New Zealand": "high_income",
    "Nicaragua": "lower_middle_income",
    "Norway": "high_income",
    "Oman": "high_income",
    "Pakistan": "lower_middle_income",
    "Paraguay": "upper_middle_income",
    "Peru": "upper_middle_income",
    "Philippines": "lower_middle_income",
    "Poland": "high_income",
    "Portugal": "high_income",
    "Qatar": "high_income",
    "Rep. of Korea": "high_income",
    "Rep. of Moldova": "upper_middle_income",
    "Romania": "high_income",
    "Samoa": "upper_middle_income",
    "Senegal": "lower_middle_income",
    "Serbia": "upper_middle_income",
    "Slovakia": "high_income",
    "Slovenia": "high_income",
    "South Africa": "upper_middle_income",
    "Spain": "high_income",
    "Sri Lanka": "lower_middle_income",
    "Sweden": "high_income",
    "Switzerland": "high_income",
    "Taiwan": "high_income",
    "Thailand": "upper_middle_income",
    "Trinidad and Tobago": "high_income",
    "Tunisia": "lower_middle_income",
    "USA": "high_income",
    "Uganda": "low_income",
    "Ukraine": "lower_middle_income",
    "United Kingdom": "high_income",
    "United Rep. of Tanzania": "lower_middle_income",
    "Uruguay": "high_income",
    "Uzbekistan": "lower_middle_income",
}

# Default tariff rate per income group based on unweighted simple mean (World Bank)

INCOME_GROUP_DEFAULT_RATES: dict[str, float] = {
    "high_income": 0.02,
    "upper_middle_income": 0.04,
    "lower_middle_income": 0.08,
    "low_income": 0.11,
}

# Fallback for an importer with no income-group classification at all.
FALLBACK_DEFAULT_RATE = 0.05
FALLBACK_DEFAULT_METHODOLOGY = "importer_default_unclassified"


def income_group_default_rates() -> dict[str, tuple[float, str]]:
    """Build each importer's default rate directly from COUNTRY_INCOME_GROUP/INCOME_GROUP_DEFAULT_RATES."""

    return {
        country: (INCOME_GROUP_DEFAULT_RATES[group], f"importer_default_{group}")
        for country, group in COUNTRY_INCOME_GROUP.items()
    }


def derive_importer_default_rates(tariffs: pd.DataFrame) -> dict[str, tuple[float, str]]:
    """Derive each importer's general default rate from a tariff table.

    Expects columns [source, target, estimated_tariff_pct, tariff_methodology] as found in estimated_tariffs.csv.

    Excludes rows explained by a more specific rule (zone or bilateral FTA), then takes the most common remaining rate/methodology per importer.
    e.g. Egypt -> (0.08, "importer_default_lower_middle_income").

    Used at graph-load time (SupplyChainNetwork.__init__), where the edges already have estimated_tariff_pct/tariff_methodology baked in from
    build_tariff_table()/add_tariffs().
    """

    default_rows = tariffs[~tariffs["tariff_methodology"].isin(
        ["eu_eea_uk_ch_zone", "bilateral_fta_bloc"]
    )]

    rates = {}
    for importer, group in default_rows.groupby("source"):
        rate = group["estimated_tariff_pct"].mode()[0]
        methodology = group["tariff_methodology"].mode()[0]
        rates[importer] = (float(rate), methodology)

    return rates


def load_importer_default_rates(tariffs_path: Path | str) -> dict[str, tuple[float, str]]:
    """Same as derive_importer_default_rates, reading the tariff table from a CSV path."""

    return derive_importer_default_rates(pd.read_csv(tariffs_path))


def estimate_tariff_pct(
    importer: str, partner: str, importer_default_rates: dict[str, tuple[float, str]]
) -> dict:
    """Estimate a tariff rate for an (importer, partner) pair using the rule priority: 
        bilateral FTA pair > shared EU/EEA/UK/CH zone > special flat-rate importer > importer's general default.

    Args:
        importer: the country recorded as importing the good.
        partner: the country recorded as supplying the good.
        importer_default_rates: output of income_group_default_rates() (if from-scratch generation) 
            or load_importer_default_rates()/derive_importer_default_rates() (if re-deriving from an existing table).

    Returns:
        {"estimated_tariff_pct": float, "tariff_methodology": str, "is_placeholder_estimate": True}
    """

    if frozenset({importer, partner}) in BILATERAL_FTA_PAIRS:
        return {
            "estimated_tariff_pct": 0.0,
            "tariff_methodology": BILATERAL_FTA_PAIRS[frozenset({importer, partner})],
            "is_placeholder_estimate": True,
        }

    if importer in EU_EEA_UK_CH_ZONE and partner in EU_EEA_UK_CH_ZONE:
        return {
            "estimated_tariff_pct": 0.0,
            "tariff_methodology": "eu_eea_uk_ch_zone",
            "is_placeholder_estimate": True,
        }

    if importer in SPECIAL_IMPORTER_RATES:
        rate, methodology = SPECIAL_IMPORTER_RATES[importer]
        return {
            "estimated_tariff_pct": rate,
            "tariff_methodology": methodology,
            "is_placeholder_estimate": True,
        }

    rate, methodology = importer_default_rates.get(
        importer, (FALLBACK_DEFAULT_RATE, FALLBACK_DEFAULT_METHODOLOGY)
    )
    return {
        "estimated_tariff_pct": rate,
        "tariff_methodology": methodology,
        "is_placeholder_estimate": True,
    }


def build_tariff_table(edges: pd.DataFrame) -> pd.DataFrame:
    """Generate a full estimated_tariffs.csv table by applying estimate_tariff_pct() to every unique (source, target) pair in `edges`.

    Args:
        edges: a cleaned_edges.csv-shaped DataFrame with at least "source" and "target" columns.

    Returns:
        One row per unique (source, target) pair: source, target, estimated_tariff_pct, tariff_methodology, is_placeholder_estimate.
    """

    default_rates = income_group_default_rates()
    pairs = edges[["source", "target"]].drop_duplicates()

    rows = []
    for source, target in pairs.itertuples(index=False):
        est = estimate_tariff_pct(source, target, default_rates)
        rows.append({"source": source, "target": target, **est})

    return pd.DataFrame(
        rows,
        columns=["source", "target", "estimated_tariff_pct", "tariff_methodology", "is_placeholder_estimate"],
    )


def main(
    edges_path: str = "data/processed/cleaned_edges.csv",
    output_path: str = "data/processed/estimated_tariffs.csv",
):
    edges = pd.read_csv(edges_path)
    table = build_tariff_table(edges)
    table.to_csv(output_path, index=False)

    n_pairs = edges[["source", "target"]].drop_duplicates().shape[0]
    print(f"Wrote {len(table):,} tariff estimates ({n_pairs:,} unique (source, target) pairs) to {output_path}")


if __name__ == "__main__":
    main()
