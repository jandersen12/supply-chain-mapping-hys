"""
Tool definitions for exposing SupplyChainNetwork methods to an LLM via
function calling (Anthropic Messages API `tools` format). Pass TOOLS into
the `tools` param of a messages.create() call; route tool_use blocks to
`dispatch_tool_call`.
"""

from typing import Any

from src.supply_chain_network import SupplyChainNetwork

TOOLS = [
    {
        "name": "simulate_node_removal",
        "description": (
            "Simulate a disruption (export ban, plant shutdown, shipping "
            "corridor closure, etc.) by removing one or more countries from "
            "the supply chain trade network and reporting the structural "
            "and economic impact. Use this whenever the user asks a "
            "'what-if' question about a specific country or set of "
            "countries being cut off."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One or more country names to remove simultaneously, "
                        "e.g. ['USA'] or ['USA', 'Japan']. Names should match "
                        "how they appear in the trade data (call "
                        "list_countries first if unsure)."
                    ),
                }
            },
            "required": ["countries"],
        },
    },
    {
        "name": "list_countries",
        "description": (
            "List every country name known to the supply chain network. "
            "Use this to validate or disambiguate a country name mentioned "
            "by the user before calling simulate_node_removal."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "rank_vulnerability",
        "description": (
            "Rank countries by how disruptive their individual removal "
            "would be, combining structural fragmentation and trade value "
            "lost. Use this for open-ended questions like 'which countries "
            "are the biggest risk?' rather than a specific what-if."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "How many top-ranked countries to return (default 10).",
                }
            },
        },
    },
]


def dispatch_tool_call(network: SupplyChainNetwork, tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Route a tool_use block's name/input to the matching SupplyChainNetwork method."""
    if tool_name == "simulate_node_removal":
        return network.simulate_removal(tool_input["countries"])
    if tool_name == "list_countries":
        return network.list_countries()
    if tool_name == "rank_vulnerability":
        return network.rank_vulnerability(top_n=tool_input.get("top_n", 10))
    raise ValueError(f"Unknown tool: {tool_name}")
