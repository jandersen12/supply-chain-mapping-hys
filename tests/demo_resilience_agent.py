"""
Manual demo of the LLM boundary in src/resilience_agent.py - requires
ANTHROPIC_API_KEY in .env. Not part of the deterministic test suite (that's
tests/test_diversification.py); run this by hand to eyeball the parsed goal,
search trace, and narrative for a few example phrasings.

Run with: python3 tests/demo_resilience_agent.py
"""

import json

from src.resilience_agent import get_client, run_resilience_goal
from src.supply_chain_network import SupplyChainNetwork

EDGES_PATH = "data/processed/cleaned_edges.csv"
NODES_PATH = "data/processed/cleaned_nodes.csv"

EXAMPLE_GOALS = [
    "Make sure no single supplier accounts for more than 20% of any importer's trade.",
    "I'm worried about Rep. of Korea's exposure to China - get them under 15% dependence on any one supplier.",
    "Reduce our supply concentration risk.",  # no threshold named - should default to 0.2
]

if __name__ == "__main__":
    client = get_client()
    network = SupplyChainNetwork(edges_path=EDGES_PATH, nodes_path=NODES_PATH)

    for goal_text in EXAMPLE_GOALS:
        print(f"\n{'=' * 70}\nGoal: {goal_text}\n{'=' * 70}")

        result = run_resilience_goal(client, network, goal_text)

        print("\n--- Parsed goal ---")
        print(json.dumps(result.get("parsed_goal"), indent=2))

        if not result["success"]:
            print("\n--- Failed ---")
            print(result.get("error"))
            continue

        print("\n--- Trace ---")
        print(json.dumps(result.get("trace", []), indent=2))

        print(f"\n--- goal_met: {result.get('goal_met')}, "
              f"chosen_capacity_multiplier: {result.get('chosen_capacity_multiplier')} ---")

        print("\n--- Narrative ---")
        print(result["narrative"])
