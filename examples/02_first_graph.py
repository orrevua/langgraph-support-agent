"""Milestone 2 - the explicit ReAct loop, step by step.

    uv run python examples/02_first_graph.py

Uses the in-memory repository and the offline fake model by default. Read
`src/supportagent/agent/graph.py` alongside this output: every printed block is one node
execution (one LangGraph "super-step").
"""

from __future__ import annotations

from supportagent.agent.state import user_turn
from supportagent.config import Settings
from supportagent.llm import describe_provider
from supportagent.runtime import build_agent

settings = Settings(storage="memory")
graph, _ = build_agent(settings)

print(f"provider: {describe_provider(settings)}\n")
print("Graph structure (mermaid, paste into https://mermaid.live):")
print(graph.get_graph().draw_mermaid())

question = "I paid my rent via Pix on Wednesday but my landlord says it never arrived. What happened?"
print(f"\nUSER: {question}\n")

# stream_mode="updates" yields {node_name: partial_state_update} after each node runs, so you
# can watch the loop agent -> tools -> agent -> ... happen.
for update in graph.stream(user_turn(question, customer_id="C001"), stream_mode="updates"):
    for node, delta in update.items():
        print(f"--- node: {node} (steps this turn: +{delta.get('agent_steps', 0)})")
        for message in delta.get("messages", []):
            if getattr(message, "tool_calls", None):
                for call in message.tool_calls:
                    print(f"    model -> tool call {call['name']}({call['args']})")
            else:
                print(f"    [{message.type}] {str(message.content)[:300]}")
