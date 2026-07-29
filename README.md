# Supply Chain Vulnerability Mapping and Optimized Solutions Project

A graph database solution to supply chain analytics and strategic solutions for resilience.

## Description

Supply chains have prioritized efficiency and lower upfront costs at the expense of network resilience. This has resulted in supply chain systems that lack the flexibility to respond to global shocks, often leading to inflation and logistical challenges that can take months or years to recover. By creating digital graph database models of commodity supply chains, companies can surface chokepoints in the system and create plans that enable them to respond to shocks with strategic solutions that limit negative impacts. 

This project models the supply chain network for a single commodity to identify critical nodes where failures in the chain would be the most disruptive. Using graph algorithms and optimization techniques, it estimates the cost of rerouting resources when a failure occurs at one of these points. A natural language interface also lets users simulate "what-if" scenarios, helping businesses turn resilience planning into concrete action.


### Data

[UN Comtrade Database](https://comtradeplus.un.org/)

## Structure

```
favorita-demand-forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── eda.ipynb
│   └── node_removal_analysis.ipynb
├── outputs/
│   ├── figures/
│   └── results/
├── src/
│   ├── __init__.py
│   ├── comtradeapi_data.py     # Pulls data from UN Comtrade
│   ├── data_cleaning.py        # Process raw data + build nodes and edges
│   ├── supply_chain_network.py # Builds network, calculates key metrics
│   └── tool_schemas.py         # Tool format for LLM calls
├── tests/
│   ├── demo.py                 # script to simulate supply_chain_network.py
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