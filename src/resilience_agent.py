"""
LLM boundary for the resilience-planning agent: parsing a free-text
diversification goal into the structured input seek_diversification_plan
needs, and narrating its result back into a briefing.

Deliberately narrow - two isolated LLM calls, not an agentic tool-use loop.
The search itself (diversification.seek_diversification_plan) stays plain,
deterministic Python; see that module's docstring for why. Functions here
take an anthropic.Anthropic client as an argument rather than constructing
one at import time, so importing this module never requires
ANTHROPIC_API_KEY to be set.
"""

import json
import os
from typing import TYPE_CHECKING, Any

import anthropic
from dotenv import load_dotenv

if TYPE_CHECKING:
    from .supply_chain_network import SupplyChainNetwork

MODEL = "claude-sonnet-5"

GOAL_TOOL = {
    "name": "set_diversification_goal",
    "description": (
        "Extract a supplier-diversification goal: the maximum share of an "
        "importer's trade value any single supplier should hold, and "
        "(optionally) which importers to apply it to."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "max_share_target": {
                "type": "number",
                "description": (
                    "Maximum fraction (0-1) any single supplier should hold "
                    "of an importer's trade value. Convert percentages to "
                    "fractions (e.g. '20%' or 'under 20%' -> 0.2). Default "
                    "to 0.2 if the user gave no numeric threshold."
                ),
            },
            "importers": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Importer country names the user explicitly named. Omit "
                    "this field entirely if the user didn't name specific "
                    "importers, so every over-concentrated importer is "
                    "considered."
                ),
            },
        },
        "required": ["max_share_target"],
    },
}

GOAL_SYSTEM_PROMPT = (
    "Extract a supplier-diversification goal from the user's request using "
    "the set_diversification_goal tool. max_share_target is a fraction "
    "between 0 and 1, not a percentage - convert percentages by dividing by "
    "100. If the user names specific importer countries, list them in "
    "importers; otherwise omit that field so every over-concentrated "
    "importer is considered."
)

NARRATION_SYSTEM_PROMPT = (
    "You are summarizing the result of an automated supplier-diversification "
    "planning run for a supply-chain risk analyst. You'll be given the "
    "user's original goal and a JSON result covering: the search trace "
    "across capacity_multiplier assumptions, before/after supplier "
    "concentration, which solver was recommended, and the resulting "
    "allocation plan summary. Write a concise briefing: state plainly "
    "whether the goal was met, give the key before/after numbers, and call "
    "out anything worth scrutinizing (an unusually large capacity "
    "assumption, new trade relationships formed, unmet value). Ground every "
    "number in the provided JSON - never invent one. If the goal wasn't "
    "fully met, say so plainly and explain why (capacity-constrained, no "
    "alternate suppliers, etc)."
)


def get_client() -> anthropic.Anthropic:
    """Load ANTHROPIC_API_KEY from .env, same pattern as comtradeapi_data.py.
    Only called by callers that actually need a client (a demo script, the
    UI) - never by the library functions below, so importing this module
    doesn't require credentials to be set."""

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    assert api_key, "ANTHROPIC_API_KEY not found in .env file."
    return anthropic.Anthropic(api_key=api_key)


def parse_diversification_goal(client: anthropic.Anthropic, text: str) -> dict[str, Any]:
    """Parse a free-text diversification goal into structured arguments for
    seek_diversification_plan.

    Returns {"success": True, "goal": {"max_share_target": float, "importers": list[str] | None}}
    or {"success": False, "error": str}.
    """

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=GOAL_SYSTEM_PROMPT,
            thinking={"type": "disabled"},
            tools=[GOAL_TOOL],
            tool_choice={"type": "tool", "name": "set_diversification_goal"},
            messages=[{"role": "user", "content": text}],
        )
    except anthropic.APIError as e:
        return {"success": False, "error": f"Could not reach the model: {e}"}

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        return {"success": False, "error": "Model did not return a parsed goal."}

    max_share_target = tool_use.input.get("max_share_target")
    if not isinstance(max_share_target, (int, float)) or not (0 < max_share_target < 1):
        return {
            "success": False,
            "error": f"Parsed max_share_target ({max_share_target!r}) is not a fraction between 0 and 1.",
        }

    return {
        "success": True,
        "goal": {
            "max_share_target": float(max_share_target),
            "importers": tool_use.input.get("importers"),
        },
    }


def _summarize_for_narration(result: dict[str, Any]) -> dict[str, Any]:
    """Trim a seek_diversification_plan result to the fields worth narrating,
    dropping the per-allocation arc details so the narration prompt stays
    compact."""

    summary = {
        "success": result.get("success"),
        "error": result.get("error"),
        "already_compliant": result.get("already_compliant"),
        "goal_met": result.get("goal_met"),
        "chosen_capacity_multiplier": result.get("chosen_capacity_multiplier"),
        "trace": result.get("trace"),
        "concentration_before": result.get("concentration_before"),
        "concentration_after": result.get("concentration_after"),
        "recommended_solver": result.get("recommended_solver"),
    }
    plan = result.get("plan")
    if plan:
        summary["plan_summary"] = plan.get("summary")
    return {k: v for k, v in summary.items() if v is not None}


def narrate_diversification_result(
    client: anthropic.Anthropic, goal_text: str, result: dict[str, Any]
) -> str:
    """Turn a seek_diversification_plan result into a prose briefing."""

    payload = json.dumps(_summarize_for_narration(result), indent=2, default=str)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=NARRATION_SYSTEM_PROMPT,
            thinking={"type": "disabled"},
            messages=[{
                "role": "user",
                "content": f"User's goal: {goal_text}\n\nResult:\n{payload}",
            }],
        )
    except anthropic.APIError as e:
        return f"(Could not generate a narrative: {e})"

    return "".join(b.text for b in response.content if b.type == "text")


def run_resilience_goal(
    client: anthropic.Anthropic,
    network: "SupplyChainNetwork",
    goal_text: str,
    **search_kwargs: Any,
) -> dict[str, Any]:
    """Orchestrate: parse the free-text goal, search for a plan, narrate the
    result. Single entry point - takes a goal_text in, returns the search
    result plus goal_text/parsed_goal/narrative."""

    from .diversification import seek_diversification_plan

    parsed = parse_diversification_goal(client, goal_text)
    if not parsed["success"]:
        return parsed

    goal = parsed["goal"]
    search_result = seek_diversification_plan(
        network,
        importers=goal.get("importers"),
        max_share_target=goal["max_share_target"],
        **search_kwargs,
    )

    narrative = narrate_diversification_result(client, goal_text, search_result)

    return {
        **search_result,
        "goal_text": goal_text,
        "parsed_goal": goal,
        "narrative": narrative,
    }
