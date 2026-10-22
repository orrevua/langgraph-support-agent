"""An offline, deterministic chat model used when no API key is available.

Why not just use `FakeListChatModel` from langchain-core? That one replays a fixed list of
responses, which breaks as soon as the conversation takes a different path. An agent needs a
model that *reacts* to what it sees (the user's request, tool results, an approval rejection),
so this module implements a tiny "scripted policy" model:

    messages + bound tools  --policy-->  AIMessage (text and/or tool calls)

It plugs into LangChain exactly like `ChatAnthropic` does, because it subclasses
`BaseChatModel` and implements `_generate` and `bind_tools`. That means `with_structured_output`,
`ChatPromptTemplate | model` pipes, and LangGraph nodes all work unchanged.

Policies are plain functions; the first one that returns a message wins. See
`supportagent.llm.policy` for the support-agent policy that drives the evals.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

# A policy sees the conversation and the tool schemas (OpenAI format, which is what
# `convert_to_openai_tool` produces) and returns a message, or None to defer to the next policy.
Policy = Callable[[Sequence[BaseMessage], list[dict[str, Any]]], AIMessage | None]
# A structured-output handler turns the user's text into a dict matching a schema.
StructuredHandler = Callable[[str], dict[str, Any]]


def tool_call(name: str, args: dict[str, Any], messages: Sequence[BaseMessage]) -> AIMessage:
    """Build an AIMessage that requests one tool call, with a deterministic call id."""
    call_id = f"call_{len(messages)}_{name}"
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def last_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text if hasattr(message, "text") else str(message.content)
    return ""


def generic_policy(messages: Sequence[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
    """A fallback policy that makes simple tool-calling demos work offline.

    1. If the previous message is a tool result, answer using it.
    2. If tools are bound, pick the one whose name/description best overlaps the request and
       fill its arguments heuristically (numbers from the text, enum values that appear, ...).
    3. Otherwise produce a short extractive answer.
    """
    if messages and isinstance(messages[-1], ToolMessage):
        results = [m.content for m in messages if isinstance(m, ToolMessage)]
        return AIMessage(content=f"Based on the tool result(s): {'; '.join(map(str, results))}")

    text = last_human_text(messages)
    words = set(re.findall(r"[a-z]+", text.lower()))
    best, best_score = None, 0
    for spec in tools:
        fn = spec["function"]
        vocab = set(
            re.findall(r"[a-z]+", f"{fn['name'].replace('_', ' ')} {fn.get('description', '')}".lower())
        )
        score = len(words & vocab)
        if score > best_score:
            best, best_score = fn, score
    if best is not None:
        return tool_call(best["name"], _fill_args(best.get("parameters", {}), text), messages)

    first_sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return AIMessage(content=f"[offline fake model] You asked: {first_sentence}")


def _fill_args(schema: dict[str, Any], text: str) -> dict[str, Any]:
    numbers = re.findall(r"\d+(?:[.,]\d+)?", text)
    args: dict[str, Any] = {}
    for name, prop in schema.get("properties", {}).items():
        kind = prop.get("type")
        if "enum" in prop:
            args[name] = next((v for v in prop["enum"] if str(v).lower() in text.lower()), prop["enum"][0])
        elif kind in ("integer", "number") and numbers:
            raw = numbers.pop(0).replace(",", ".")
            args[name] = int(float(raw)) if kind == "integer" else float(raw)
        elif kind == "string":
            args[name] = text
        elif "default" in prop:
            args[name] = prop["default"]
    return args


class ScriptedChatModel(BaseChatModel):
    """A deterministic chat model driven by policy functions (see module docstring)."""

    policies: list[Policy] = Field(default_factory=lambda: [generic_policy])
    structured_handlers: dict[str, StructuredHandler] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any) -> Runnable:
        # Real providers translate tools into their API format. We keep the OpenAI-style JSON
        # schema so policies can inspect names, descriptions and parameters.
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools: list[dict[str, Any]] = kwargs.get("tools") or []
        tool_choice = kwargs.get("tool_choice")

        # `with_structured_output` binds exactly one tool (the schema) with tool_choice="any".
        if tool_choice and len(tools) == 1:
            name = tools[0]["function"]["name"]
            handler = self.structured_handlers.get(name)
            if handler is None:
                raise NotImplementedError(f"No offline structured-output handler for schema {name!r}")
            args = handler(last_human_text(messages))
            message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])
            return ChatResult(generations=[ChatGeneration(message=message)])

        for policy in self.policies:
            message = policy(messages, tools)
            if message is not None:
                return ChatResult(generations=[ChatGeneration(message=message)])
        raise RuntimeError("No policy produced a response")


def parse_json(content: Any) -> Any:
    """Tool results are JSON strings; parse them leniently for policies."""
    if isinstance(content, (dict, list)):
        return content
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content
