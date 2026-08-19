"""
Demo showing how to call the supply chain network file.
"""

import json

from src.supply_chain_network import SupplyChainNetwork

if __name__ == "__main__":
    network = SupplyChainNetwork(
        edges_path="data/processed/cleaned_edges.csv",
        nodes_path="data/processed/cleaned_nodes.csv",
    )

    print("=== Single-country full shock: 'USA' loses 100% of exports ===")
    result = network.simulate_shock({"USA": 1.0})
    print(json.dumps(result, indent=2))

    print("\n=== Partial shock: 'Saudi Arabia' loses 40% of exports ===")
    result = network.simulate_shock({"Saudi Arabia": 0.4})
    print(json.dumps(result["impact"], indent=2))

    print("\n=== Open-ended: rank_vulnerability ===")
    print(json.dumps(network.rank_vulnerability(top_n=5), indent=2))