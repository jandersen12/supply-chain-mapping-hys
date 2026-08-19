"""
Demo showing how to call the supply chain network file. either directly, or from an LLM tool_use block.
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

    print("\n=== Partial shock: 'usa' loses 40%, 'Japan' loses 100% (case-insensitive, multi-node) ===")
    result = network.simulate_shock({"usa": 0.4, "Japan": 1.0})
    print(json.dumps(result["impact"], indent=2))

    print("\n=== Bad input: unrecognized country name ===")
    result = network.simulate_shock({"Sourth Korea": 1.0})
    print(json.dumps(result, indent=2))

    print("\n=== Open-ended: rank_vulnerability ===")
    print(json.dumps(network.rank_vulnerability(top_n=5), indent=2))

    print("\n=== Simulated LLM tool_use dispatch ===")
    fake_llm_tool_call = {"tool_name": "simulate_disruption", "tool_input": {"shocks": {"China": 1.0}}}
    output = dispatch_tool_call(
        network, fake_llm_tool_call["tool_name"], fake_llm_tool_call["tool_input"]
    )
    print(json.dumps(output["impact"], indent=2))
