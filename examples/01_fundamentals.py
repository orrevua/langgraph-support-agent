"""Milestone 1 - LangChain fundamentals, one concept per section.

Run it offline (default, no API key needed):

    uv run python examples/01_fundamentals.py

or against Claude:

    ANTHROPIC_API_KEY=sk-... uv run python examples/01_fundamentals.py

Concepts covered:
  1. Chat models and messages (SystemMessage / HumanMessage / AIMessage)
  2. ChatPromptTemplate and LCEL pipes (`prompt | model | parser`)
  3. Structured output with Pydantic (`model.with_structured_output(TicketTriage)`)
  4. Tools: the @tool decorator, tool schemas, and the tool-calling round trip

Docs: https://python.langchain.com/docs/concepts/
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from supportagent.llm import describe_provider, get_chat_model
from supportagent.triage import TicketTriage


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------------------------
# 1. A chat model takes a list of messages and returns an AIMessage.
# ---------------------------------------------------------------------------------------------
# `get_chat_model()` returns ChatAnthropic when ANTHROPIC_API_KEY is set, otherwise our
# offline ScriptedChatModel. Both implement the same `BaseChatModel` interface, so nothing
# below cares which one it got. This is the main reason to code against LangChain's
# abstractions: swapping providers (or faking one in tests) is free.
model = get_chat_model()


def demo_messages() -> None:
    section(f"1. Chat model + messages  [provider: {describe_provider()}]")
    messages = [
        # The system message sets behaviour; providers treat it specially (Anthropic sends
        # it as the top-level `system` parameter).
        SystemMessage("You are a concise support assistant for a Brazilian fintech."),
        HumanMessage("In one sentence: what is Pix?"),
    ]
    reply = model.invoke(messages)  # -> AIMessage
    print(type(reply).__name__, "->", reply.content)
    # `usage_metadata` holds token counts for real providers (None for the fake).
    print("usage:", reply.usage_metadata)


# ---------------------------------------------------------------------------------------------
# 2. Prompt templates + LCEL (LangChain Expression Language).
# ---------------------------------------------------------------------------------------------
# A ChatPromptTemplate is a Runnable that turns a dict of variables into a list of messages.
# The `|` operator chains Runnables: the output of one becomes the input of the next.
# StrOutputParser extracts `.content` from the AIMessage so the chain returns a plain string.
reply_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You write short, polite replies for {company}. Never promise refunds."),
        ("human", "Customer wrote: {customer_message}\nWrite a reply in English."),
    ]
)


def demo_prompt_template() -> None:
    section("2. ChatPromptTemplate | model | StrOutputParser")
    chain = reply_prompt | model | StrOutputParser()
    # Inspect what the template renders before sending it anywhere - useful for debugging.
    rendered = reply_prompt.invoke({"company": "Aurora Bank", "customer_message": "My Pix is late."})
    for message in rendered.to_messages():
        print(f"  [{message.type}] {message.content}")
    print("chain output:", chain.invoke({"company": "Aurora Bank", "customer_message": "My Pix is late."}))


# ---------------------------------------------------------------------------------------------
# 3. Structured output: get a validated Pydantic object instead of text.
# ---------------------------------------------------------------------------------------------
# Under the hood, `with_structured_output` turns the Pydantic model into a tool schema, forces
# the model to "call" it, and parses the arguments back into `TicketTriage`. If the model
# returns something that does not validate, you get an exception instead of silent garbage.
def demo_structured_output() -> None:
    section("3. Structured output (ticket triage)")
    triage_chain = ChatPromptTemplate.from_messages(
        [
            ("system", "Classify the customer's support message."),
            ("human", "{message}"),
        ]
    ) | model.with_structured_output(TicketTriage)
    for text in [
        "I sent a Pix to my landlord yesterday and it never arrived!",
        "My credit card was stolen, please help",
        "How do I change my app password?",
    ]:
        result: TicketTriage = triage_chain.invoke({"message": text})
        print(f"  {text!r}\n    -> {result.model_dump()}")


# ---------------------------------------------------------------------------------------------
# 4. Tools: @tool turns a typed, documented function into something a model can call.
# ---------------------------------------------------------------------------------------------
# The function name, docstring and type hints become the JSON schema the model sees, so write
# them for the model: say *when* to use the tool and what each argument means.
@tool
def pix_limit(period: Literal["daytime", "nighttime"]) -> str:
    """Return the Pix transfer limit for a period of the day (daytime 6h-20h, nighttime 20h-6h)."""
    # Brazilian banks must apply a lower default Pix limit at night (Central Bank rule).
    return "R$ 5.000,00" if period == "daytime" else "R$ 1.000,00"


@tool
def ted_fee(amount_brl: float) -> str:
    """Compute the fee to send a TED bank transfer of the given amount in BRL."""
    fee = 0.0 if amount_brl >= 1000 else 8.50
    return f"R$ {fee:.2f}".replace(".", ",")


def demo_tools() -> None:
    section("4. Tools and the tool-calling round trip")
    # The schema the model will see:
    print("schema:", pix_limit.args_schema.model_json_schema())

    # bind_tools attaches tool schemas to every request; the model may answer with tool_calls.
    model_with_tools = model.bind_tools([pix_limit, ted_fee])
    tools_by_name = {"pix_limit": pix_limit, "ted_fee": ted_fee}

    messages = [HumanMessage("What is my Pix limit during the nighttime period?")]
    ai = model_with_tools.invoke(messages)
    print("model asked for:", ai.tool_calls)
    messages.append(ai)

    # The model never executes anything: *we* run the tool and send back a ToolMessage that
    # references the tool_call_id. This loop is exactly what the LangGraph agent automates
    # in milestone 2 (see src/supportagent/agent/graph.py).
    for call in ai.tool_calls:
        result = tools_by_name[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    final = model_with_tools.invoke(messages)
    print("final answer:", final.content)


if __name__ == "__main__":
    demo_messages()
    demo_prompt_template()
    demo_structured_output()
    demo_tools()
