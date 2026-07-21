# Supply Chain Vulnerability Mapping and Optimized Solutions Project

Hack Your Summer

## Project Structure

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
python3 src/comtradeapi-data.py
```