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
        "name": "simulate_disruption",
        "description": (
            "Simulate a disruption (export ban, plant shutdown, shipping "
            "corridor closure, etc.) by reducing one or more countries' "
            "export capacity in the supply chain trade network and "
            "reporting the structural and economic impact. A country's own "
            "imports are unaffected - only what it supplies to others "
            "shrinks. Use this whenever the user asks a 'what-if' question "
            "about a specific country or set of countries losing export "
            "capacity, whether partially (e.g. a 40% output cut) or "
            "entirely (e.g. a full export ban, severity 1.0)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "shocks": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": (
                        "Map of country name to severity: the fraction of "
                        "that country's export capacity lost, in (0, 1]. "
                        "E.g. {'USA': 0.4} models a 40% drop in USA's "
                        "exports; {'USA': 1.0, 'Japan': 1.0} models both "
                        "fully cut off. Names should match how they appear "
                        "in the trade data (call list_countries first if "
                        "unsure)."
                    ),
                }
            },
            "required": ["shocks"],
        },
    },
    {
        "name": "list_countries",
        "description": (
            "List every country name known to the supply chain network. "
            "Use this to validate or disambiguate a country name mentioned "
            "by the user before calling simulate_disruption."
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
    if tool_name == "simulate_disruption":
        return network.simulate_shock(tool_input["shocks"])
    if tool_name == "list_countries":
        return network.list_countries()
    if tool_name == "rank_vulnerability":
        return network.rank_vulnerability(top_n=tool_input.get("top_n", 10))
    raise ValueError(f"Unknown tool: {tool_name}")
