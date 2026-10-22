"""Human-in-the-loop review of sensitive tool calls, built on `interrupt()`.

How `interrupt()` works (the key idea of this milestone):

1. Inside a node, `interrupt(payload)` raises a special exception. LangGraph catches it, saves
   a checkpoint, and returns control to the caller with the payload exposed as a pending
   interrupt (`graph.get_state(config).interrupts`). Nothing after the call runs yet.
2. Later - seconds or days, even after a server restart, because the checkpoint is in
   Postgres - the caller resumes with `graph.invoke(Command(resume=decision), config)`.
3. LangGraph RE-RUNS THE NODE FROM THE TOP. This time `interrupt()` returns `decision`
   instead of raising. Consequence: code before `interrupt()` runs twice, so keep side effects
   (like writing the audit log) AFTER all interrupts in the node.

With several sensitive calls in one AI message, the node calls `interrupt()` once per call;
LangGraph matches resume values to interrupts by order, so they are reviewed one at a time.

Docs: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from supportagent.agent.state import AgentState
from supportagent.domain.models import ApprovalRecord, format_brl
from supportagent.tools import SENSITIVE_TOOLS, ToolContext, refund_eligibility, to_cents

# Reviewers may adjust these arguments; changing *what* is acted on (transaction/card) is not
# an edit, it is a different action, so it must be rejected instead.
EDITABLE_ARGS: dict[str, set[str]] = {"issue_refund": {"amount_brl", "reason"}, "block_card": {"reason"}}


class ReviewDecision(BaseModel):
    """The value a reviewer sends back through `Command(resume=...)`."""

    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=100)
    edited_args: dict[str, Any] | None = None
    reason: str | None = Field(default=None, max_length=500)


class PendingAction(BaseModel):
    """What the graph exposes while paused (the `interrupt()` payload)."""

    action_id: str  # the tool_call_id
    tool: str
    args: dict[str, Any]
    summary: str
    editable_args: list[str]
    context: dict[str, Any] = {}


def describe_action(name: str, args: dict[str, Any], ctx: ToolContext, customer_id: str) -> PendingAction:
    context: dict[str, Any] = {"customer_id": customer_id}
    if name == "issue_refund":
        try:
            amount = format_brl(to_cents(args.get("amount_brl", 0)))
        except Exception:
            amount = str(args.get("amount_brl"))
        summary = f"Refund {amount} for {args.get('transaction_id')} ({args.get('reason', 'no reason')})"
        txn = ctx.repo.get_transaction(str(args.get("transaction_id", "")).upper())
        if txn is not None and txn.customer_id == customer_id:
            context["transaction"] = {
                "amount": format_brl(txn.amount_cents),
                "status": txn.status,
                "kind": txn.kind,
                "counterparty": txn.counterparty,
            }
            context["eligibility"] = refund_eligibility(
                txn, ctx.repo.list_refunds(txn.id), ctx.settings.refund_hard_limit_cents
            )
    else:
        summary = f"Block card {args.get('card_id')} (reason: {args.get('reason')})"
        card = ctx.repo.get_card(str(args.get("card_id", "")).upper())
        if card is not None and card.customer_id == customer_id:
            context["card"] = {"kind": card.kind, "last4": card.last4, "status": card.status}
    return PendingAction(
        action_id="",
        tool=name,
        args=args,
        summary=summary,
        editable_args=sorted(EDITABLE_ARGS.get(name, set())),
        context=context,
    )


def make_human_review_node(ctx: ToolContext):
    def human_review(state: AgentState, config: RunnableConfig) -> dict:
        ai = state["messages"][-1]
        assert isinstance(ai, AIMessage)
        thread_id = config["configurable"].get("thread_id", "")

        new_calls, rejections, approvals, audit = [], [], {}, []
        for call in ai.tool_calls:
            if call["name"] not in SENSITIVE_TOOLS:
                new_calls.append(call)
                continue
            pending = describe_action(call["name"], call["args"], ctx, state["customer_id"])
            pending.action_id = call["id"]
            # --- the pause -------------------------------------------------------------
            decision = ReviewDecision.model_validate(interrupt(pending.model_dump()))
            # --- resumed: `decision` is what the reviewer sent ----------------------------
            if decision.decision == "approve":
                edits = {
                    k: v for k, v in (decision.edited_args or {}).items() if k in EDITABLE_ARGS.get(call["name"], set())
                }
                final_args = {**call["args"], **edits}
                new_calls.append({**call, "args": final_args})
                approvals[call["id"]] = decision.reviewer
            else:
                final_args = None
                new_calls.append(call)
                rejections.append(
                    ToolMessage(
                        content=f"ERROR: Action rejected by reviewer {decision.reviewer}: "
                        f"{decision.reason or 'no reason given'}. Do not retry it.",
                        tool_call_id=call["id"],
                        name=call["name"],
                        status="error",
                    )
                )
            audit.append(
                ApprovalRecord(
                    thread_id=thread_id,
                    tool_call_id=call["id"],
                    tool_name=call["name"],
                    proposed_args=call["args"],
                    final_args=final_args,
                    decision="approved" if decision.decision == "approve" else "rejected",
                    reviewer=decision.reviewer,
                    reason=decision.reason,
                )
            )

        # Side effects only after every interrupt in this node has been answered (see docstring).
        for record in audit:
            ctx.repo.record_approval(record)

        # Same message id -> add_messages REPLACES the AI message, so the (possibly edited) args
        # are what the tools node executes and what the model sees on the next turn.
        updated = AIMessage(content=ai.content, tool_calls=new_calls, id=ai.id)
        return {"messages": [updated, *rejections], "approved_tool_calls": approvals}

    return human_review
