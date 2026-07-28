# Supply Chain Vulnerability Mapping and Optimized Solutions Project

A graph database solution to supply chain analytics and strategic solutions for resilience.

## Description

Supply chains have prioritized efficiency and lower upfront costs at the expense of network resilience. This has resulted in supply chain systems that lack the flexibility to respond to global shocks, often leading to inflation and logistical challenges that can take months or years to recover. By creating digital graph database models of commodity supply chains, companies can surface chokepoints in the system and create plans that enable them to respond to shocks with strategic solutions that limit negative impacts. 

This project aims to model the supply chain network for a single commodity and surface chokepoints its chokepoints. By leveraging graph algorithms and optimization libraries in python, the project analysis will produce insights into the cost of reallocating resources to respond to supply chain failures. Finally, there will be a natural langauge interface that a user can interact with to model potential scenarios. The natural language inerface will translate the scenario into model inputs and then translate the result back into natural langauge for the user, enabling businesses to translate "what-if" scenarios into action plans for resilience. 

### Data

[UN Comtrade Database](https://comtradeplus.un.org/)

## Structure

```
favorita-demand-forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── results/
├── src/
│   ├── __init__.py
│   ├── comtradeapi-data.py     # Pulls data from UN Comtrade
│   └── data-cleaning.py        # Process raw data + build nodes and edges
├── .gitignore
├── README.md
└── requirements.txt
```

## Reproducibility

**Initial setup**

1. Clone the repository
2. Create the following folders inside your cloned repository:
    - data/processed
    - data/raw
3. Ensure you have a UN Comtrade account setup and an API key
4. Save your API key to a local .env file

**Code**
Once all of those steps are complete, run the code as follows:

```
pip install -r requirements.txt
python3 src/comtradeapi_data.py
python3 src/data_cleaning.py data/raw/comtrade_gallium_imports_2025_raw.csv
```