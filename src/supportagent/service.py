"""A thin service layer over the compiled graph, shared by the API, the CLI demo and the evals.

It hides LangGraph plumbing (configs, snapshots, `Command(resume=...)`) behind three verbs:
`chat`, `pending` and `decide`.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel

from supportagent.agent.review import EDITABLE_ARGS, PendingAction, ReviewDecision
from supportagent.agent.state import user_turn
from supportagent.config import Settings


class ThreadNotFound(Exception):
    pass


class ThreadBusy(Exception):
    """A new message arrived while the thread waits for an approval decision."""


class NothingPending(Exception):
    pass


class InvalidEdit(Exception):
    pass


class TurnResult(BaseModel):
    thread_id: str
    status: Literal["completed", "pending_approval"]
    reply: str | None = None
    pending: list[PendingAction] = []
    tools_called: list[str] = []


class AgentService:
    def __init__(self, graph: Any, settings: Settings) -> None:
        self.graph = graph
        self.settings = settings

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.settings.recursion_limit}

    def owner(self, thread_id: str) -> str | None:
        return self.graph.get_state(self._config(thread_id)).values.get("customer_id")

    def pending(self, thread_id: str) -> list[PendingAction]:
        snapshot = self.graph.get_state(self._config(thread_id))
        return [PendingAction.model_validate(i.value) for i in snapshot.interrupts]

    def chat(self, thread_id: str, customer_id: str, text: str) -> TurnResult:
        owner = self.owner(thread_id)
        if owner is not None and owner != customer_id:
            raise ThreadNotFound(thread_id)  # do not reveal that the thread exists
        if self.pending(thread_id):
            raise ThreadBusy(thread_id)
        return self._run(thread_id, user_turn(text, customer_id))

    def decide(self, thread_id: str, decision: ReviewDecision) -> TurnResult:
        pending = self.pending(thread_id)
        if not pending:
            raise NothingPending(thread_id)
        if decision.edited_args:
            allowed = EDITABLE_ARGS.get(pending[0].tool, set())
            bad = set(decision.edited_args) - allowed
            if bad:
                raise InvalidEdit(f"cannot edit {sorted(bad)}; editable: {sorted(allowed)}")
        return self._run(thread_id, Command(resume=decision.model_dump()))

    def _run(self, thread_id: str, graph_input: Any) -> TurnResult:
        config = self._config(thread_id)
        before = len(self.graph.get_state(config).values.get("messages", []))
        try:
            state = self.graph.invoke(graph_input, config)
        except GraphRecursionError:
            # Backstop for runaway loops; the agent node's step limit normally triggers first.
            from supportagent.agent.graph import STEP_LIMIT_REPLY

            return TurnResult(thread_id=thread_id, status="completed", reply=STEP_LIMIT_REPLY)
        messages = state.get("messages", [])
        tools = [c["name"] for m in messages[before:] if isinstance(m, AIMessage) for c in m.tool_calls]
        pending = self.pending(thread_id)
        if pending:
            return TurnResult(thread_id=thread_id, status="pending_approval", pending=pending, tools_called=tools)
        reply = messages[-1].content if messages and isinstance(messages[-1], AIMessage) else None
        return TurnResult(thread_id=thread_id, status="completed", reply=reply, tools_called=tools)
