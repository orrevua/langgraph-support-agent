"""The support agent as an explicit LangGraph StateGraph (a hand-written ReAct loop).

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

We deliberately do NOT use `create_react_agent`: writing the loop yourself shows exactly where
you can add guardrails, approvals and limits, which is the whole point of this project.

Docs: https://langchain-ai.github.io/langgraph/concepts/low_level/
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer

from supportagent.agent.prompts import SYSTEM_PROMPT
from supportagent.agent.state import AgentState
from supportagent.config import Settings, get_settings
from supportagent.tools import SENSITIVE_TOOLS, ToolContext, ToolError, build_tools

log = logging.getLogger(__name__)

STEP_LIMIT_REPLY = (
    "I'm sorry, I couldn't resolve this automatically. I've escalated your conversation to a "
    "human specialist, who will get back to you shortly."
)


def last_ai_message(state: AgentState) -> AIMessage | None:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message
    return None


def make_agent_node(model: BaseChatModel, tools: list[BaseTool], settings: Settings):
    # bind_tools sends the tool schemas with every request. Note the injected `customer_id`
    # and `approved_by` arguments do NOT appear in these schemas.
    model_with_tools = model.bind_tools(tools)

    def agent(state: AgentState) -> dict:
        """Call the model once. It either answers (no tool calls) or requests tools."""
        if state.get("agent_steps", 0) >= settings.max_agent_steps:
            log.warning("step limit reached for customer %s", state.get("customer_id"))
            return {"messages": [AIMessage(STEP_LIMIT_REPLY)]}
        system = SystemMessage(
            SYSTEM_PROMPT.format(customer_id=state["customer_id"], today=date.today().isoformat())
        )
        response = model_with_tools.invoke([system, *state["messages"]])
        # Partial update: add_messages appends `response`, count_steps adds 1.
        return {"messages": [response], "agent_steps": 1}

    return agent


def make_tools_node(tools: list[BaseTool]):
    tools_by_name = {t.name: t for t in tools}

    def run_tools(state: AgentState) -> dict:
        """Execute every tool call of the last AIMessage and return one ToolMessage per call."""
        ai = last_ai_message(state)
        assert ai is not None, "tools node reached without an AIMessage"
        # Calls that already have an answer (e.g. a rejection added by the reviewer) are skipped.
        answered = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
        approved = state.get("approved_tool_calls", {})
        results: list[ToolMessage] = []
        for call in ai.tool_calls:
            if call["id"] in answered:
                continue
            name, call_id = call["name"], call["id"]
            log.info("tool call %s args=%s", name, json.dumps(call["args"], ensure_ascii=False))
            tool = tools_by_name.get(name)
            if tool is None:
                results.append(_error(call_id, name, f"Unknown tool {name!r}."))
                continue
            # Inject the hidden arguments from trusted state, never from the model.
            args = {**call["args"], "customer_id": state["customer_id"]}
            if name in SENSITIVE_TOOLS:
                if call_id not in approved:
                    results.append(_error(call_id, name, "This action requires human approval."))
                    continue
                args["approved_by"] = approved[call_id]
            try:
                output = tool.invoke(args)
                results.append(ToolMessage(content=output, tool_call_id=call_id, name=name))
            except ToolError as exc:  # expected business errors: let the model explain them
                results.append(_error(call_id, name, str(exc)))
            except Exception as exc:  # bugs, validation errors: never crash the conversation
                log.exception("tool %s failed", name)
                results.append(
                    _error(call_id, name, f"Internal error while running {name}: {type(exc).__name__}")
                )
        return {"messages": results}

    return run_tools


def _error(call_id: str, name: str, text: str) -> ToolMessage:
    return ToolMessage(content=f"ERROR: {text}", tool_call_id=call_id, name=name, status="error")


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """Conditional edge: keep looping while the model asks for tools."""
    ai = last_ai_message(state)
    if ai is not None and ai.tool_calls and state["messages"][-1] is ai:
        return "tools"
    return END


def build_graph(
    ctx: ToolContext,
    model: BaseChatModel,
    *,
    checkpointer: Checkpointer = None,
    settings: Settings | None = None,
):
    settings = settings or get_settings()
    tools = build_tools(ctx)

    builder = StateGraph(AgentState)
    builder.add_node("agent", make_agent_node(model, tools, settings))
    builder.add_node("tools", make_tools_node(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
