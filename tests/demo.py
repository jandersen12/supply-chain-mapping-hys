"""
Demo showing how to call the supply chain network file. either directly, or from an LLM tool_use block.
"""

import json

from src.supply_chain_network import SupplyChainNetwork
from src.tool_schemas import dispatch_tool_call

if __name__ == "__main__":
    network = SupplyChainNetwork(
        edges_path="data/processed/cleaned_edges.csv",
        nodes_path="data/processed/cleaned_nodes.csv",
    )

    print("=== Single-country what-if: 'USA' ===")
    result = network.simulate_removal(["USA"])
    print(json.dumps(result, indent=2))

    print("\n=== Compound what-if: 'usa' + 'Japan' (case-insensitive, multi-node) ===")
    result = network.simulate_removal(["usa", "Japan"])
    print(json.dumps(result["impact"], indent=2))

    print("\n=== Bad input: unrecognized country name ===")
    result = network.simulate_removal(["Sourth Korea"])
    print(json.dumps(result, indent=2))

    print("\n=== Open-ended: rank_vulnerability ===")
    print(json.dumps(network.rank_vulnerability(top_n=5), indent=2))

    print("\n=== Simulated LLM tool_use dispatch ===")
    fake_llm_tool_call = {"tool_name": "simulate_node_removal", "tool_input": {"countries": ["China"]}}
    output = dispatch_tool_call(
        network, fake_llm_tool_call["tool_name"], fake_llm_tool_call["tool_input"]
    )
    print(json.dumps(output["impact"], indent=2))
