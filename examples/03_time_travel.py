"""Milestone 3 - persistence, history and time travel.

    uv run python examples/03_time_travel.py            # Postgres checkpointer (p2_agent)
    uv run python examples/03_time_travel.py --memory   # MemorySaver, no database needed

What it shows:
  1. thread_id = one conversation. Two turns on the same thread share memory.
  2. get_state_history(): every super-step left a checkpoint you can inspect.
  3. Replay: invoke(None, config_with_checkpoint_id) re-runs the graph from an old checkpoint.
  4. Fork: invoke(new_input, config_with_checkpoint_id) branches the conversation; the original
     branch is untouched (checkpoints are immutable, history is append-only).

Docs: https://langchain-ai.github.io/langgraph/concepts/time-travel/
"""

from __future__ import annotations

import argparse
import uuid

from supportagent.agent.state import user_turn
from supportagent.checkpointing import open_checkpointer
from supportagent.config import Settings, get_settings
from supportagent.repository import InMemoryBankRepository
from supportagent.runtime import build_agent


def describe(snapshot) -> str:
    last = snapshot.values.get("messages", [])[-1:] or [None]
    msg = last[0]
    if msg is None:
        what = "(empty)"
    elif getattr(msg, "tool_calls", None):
        what = "ai -> " + ", ".join(c["name"] for c in msg.tool_calls)
    else:
        what = f"{msg.type}: {str(msg.content)[:60]!r}"
    step = snapshot.metadata.get("step")
    return f"step={step:>2} next={snapshot.next!s:<12} msgs={len(snapshot.values.get('messages', [])):>2} | {what}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", action="store_true", help="use MemorySaver instead of Postgres")
    args = parser.parse_args()

    base = get_settings()
    # The domain data stays in memory here (this example is about checkpoints, not tools).
    settings = Settings(**{**base.model_dump(), "storage": "memory"})
    kind = "memory" if args.memory else "postgres"

    with open_checkpointer(kind, base.database_url) as checkpointer:
        graph, _ = build_agent(settings, repo=InMemoryBankRepository(), checkpointer=checkpointer)
        thread_id = f"time-travel-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        print(f"checkpointer={type(checkpointer).__name__} thread_id={thread_id}\n")

        # 1. Two turns on the same thread ------------------------------------------------
        graph.invoke(user_turn("What's my balance?", "C001"), config)
        graph.invoke(user_turn("Can you list my recent transactions?", "C001"), config)
        state = graph.get_state(config)
        print(f"after 2 turns: {len(state.values['messages'])} messages in the thread\n")

        # 2. History, newest first ------------------------------------------------------
        history = list(graph.get_state_history(config))
        print(f"{len(history)} checkpoints (newest first):")
        for snap in history:
            print("  ", snap.config["configurable"]["checkpoint_id"][:13], describe(snap))

        # The checkpoint where turn 1 had finished: nothing left to run (`next` is empty) and
        # the last message is the final AI answer. We search by content, not by index.
        end_of_turn_1 = next(
            s
            for s in reversed(history)  # oldest first
            if not s.next and s.values.get("messages") and s.values["messages"][-1].type == "ai"
        )
        print(f"\nend of turn 1: {describe(end_of_turn_1)}")

        # 3. Replay: resume from the checkpoint that was about to run the first tool call.
        before_tools = next(s for s in reversed(history) if s.next == ("tools",))
        print(f"\nreplaying from: {describe(before_tools)}")
        replayed = graph.invoke(None, before_tools.config)
        print("replay final answer:", replayed["messages"][-1].content.splitlines()[2][:80])

        # 4. Fork: ask something different right after turn 1 -----------------------------
        fork = graph.invoke(user_turn("What's the status of TX-1005?", "C001"), end_of_turn_1.config)
        human_turns = [m.content for m in fork["messages"] if m.type == "human"]
        print("\nfork branch human turns:", human_turns)
        print("fork answer:", fork["messages"][-1].content.splitlines()[2][:100])

        # The latest checkpoint of the thread is now the fork; the old branch is still in history.
        print(f"\nhistory now has {len(list(graph.get_state_history(config)))} checkpoints (both branches).")


if __name__ == "__main__":
    main()
