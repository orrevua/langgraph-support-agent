"""Graph state: the single, typed object every node reads from and writes partial updates to.

Reducers (the second argument of `Annotated[...]`) decide how a node's update is merged into
the existing state. Without a reducer, a node's value simply *overwrites* the old one.

* `messages` uses `add_messages`: new messages are APPENDED, and a message with an existing
  `id` REPLACES the old one (we use that to edit a tool call after a human changes its args).
* `agent_steps` uses `count_steps`: nodes return `1` to add one step; a new user turn sends
  `RESET_STEPS` to start counting from zero again.
* `approved_tool_calls` uses `merge_dicts`: approvals accumulate over the thread.

Docs: https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

RESET_STEPS = -1


def count_steps(current: int | None, update: int) -> int:
    if update == RESET_STEPS:
        return 0
    return (current or 0) + update


def merge_dicts(current: dict[str, str] | None, update: dict[str, str]) -> dict[str, str]:
    return {**(current or {}), **update}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # Who is authenticated. Set once per thread by the API; tools read it from here.
    customer_id: str
    # Model calls in the current user turn (guards against infinite tool loops).
    agent_steps: Annotated[int, count_steps]
    # tool_call_id -> reviewer, for sensitive calls a human approved.
    approved_tool_calls: Annotated[dict[str, str], merge_dicts]
    # Guardrail findings for the current turn (e.g. ["prompt_injection"]).
    flags: list[str]


def user_turn(text: str, customer_id: str) -> dict[str, Any]:
    """The input for one user message: append it, (re)state the customer, reset the step count."""
    return {"messages": [HumanMessage(text)], "customer_id": customer_id, "agent_steps": RESET_STEPS}
