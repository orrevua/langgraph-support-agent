"""The scripted "brain" of the offline fake model for the support agent.

A real LLM decides what to do next from the system prompt, the conversation and tool results.
This module does the same with plain Python rules, so the full agent loop (tool calls, tool
errors, human approvals, rejections) runs deterministically without an API key.

It is deliberately written like a tiny planner:

    intent = classify(user message)
    next_action = PLAN[intent](what happened so far in this turn)

where `next_action` is either a tool call or the final answer. Read it next to the system
prompt in `agent/prompts.py`: each rule mirrors an instruction given to the real model.
It is NOT smart: the evals in `evals/scenarios.yaml` exist to compare it to a real model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from supportagent.llm.fake import parse_json, tool_call

TXN_RE = re.compile(r"\bTX-\d{4}\b", re.IGNORECASE)
CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

THIRD_PARTY = (
    "brother's",
    "sister's",
    "wife's",
    "husband's",
    "mother's",
    "father's",
    "friend's",
    "someone else",
    "another customer",
    "other customer",
    "other person",
)

OUT_OF_SCOPE_REPLY = (
    "I'm Aurora Bank's support assistant, so I can only help with your accounts, cards, Pix "
    "transfers and refunds. I can't help with that request, but feel free to ask me anything "
    "about your banking."
)
UNAUTHORIZED_REPLY = (
    "For your security and other customers' privacy, I can only access information about your "
    "own account. The account holder needs to contact us directly from their own login."
)
GREETING_REPLY = (
    "Hi! I'm Aurora Bank's assistant. I can check your balance and transactions, investigate a "
    "Pix that didn't arrive, request refunds or block a card. How can I help?"
)


@dataclass
class Turn:
    """Everything that happened since the last user message."""

    text: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, tuple[str, Any, bool]] = field(default_factory=dict)  # id -> (name, data, error)

    @classmethod
    def from_messages(cls, messages: Sequence[BaseMessage]) -> Turn:
        start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        turn = cls(text=str(messages[start].content) if start >= 0 else "")
        for message in messages[start + 1 :]:
            if isinstance(message, AIMessage):
                turn.calls.extend(message.tool_calls)
            elif isinstance(message, ToolMessage):
                content = str(message.content)
                is_error = message.status == "error" or content.startswith("ERROR:")
                data = content.removeprefix("ERROR: ") if is_error else parse_json(content)
                turn.results[message.tool_call_id] = (message.name or "", data, is_error)
        return turn

    def called(self, name: str) -> bool:
        return any(c["name"] == name for c in self.calls)

    def result(self, name: str) -> tuple[Any, bool] | None:
        """Latest (data, is_error) for a tool, or None if it has not returned yet."""
        for call in reversed(self.calls):
            if call["name"] == name and call["id"] in self.results:
                _, data, err = self.results[call["id"]]
                return data, err
        return None

    @property
    def lower(self) -> str:
        return self.text.lower()


def support_policy(messages: Sequence[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage | None:
    names = {t["function"]["name"] for t in tools}
    if "draft_reply" not in names:
        return None  # not the support agent: let the generic policy answer

    turn = Turn.from_messages(messages)
    drafted = turn.result("draft_reply")
    if drafted is not None:
        data, err = drafted
        return AIMessage(content=data["reply"] if not err and isinstance(data, dict) else str(data))

    def call(name: str, **args: Any) -> AIMessage:
        return tool_call(name, args, messages)

    def reply(message: str, tone: str = "neutral") -> AIMessage:
        return call("draft_reply", message=message, tone=tone)

    # --- 1. Identity / authorization ------------------------------------------------------
    identifier = (CPF_RE.search(turn.text) or EMAIL_RE.search(turn.text) or [None])[0]
    if any(p in turn.lower for p in THIRD_PARTY):
        return AIMessage(content=UNAUTHORIZED_REPLY)
    if identifier:
        lookup = turn.result("lookup_customer")
        if lookup is None:
            return call("lookup_customer", identifier=identifier)
        data, err = lookup
        if err or not data.get("identifier_belongs_to_customer"):
            return AIMessage(content=UNAUTHORIZED_REPLY)
        # It is the customer's own identifier: carry on with the actual request.

    intent = classify(turn.lower)
    planner = PLANS.get(intent)
    if planner is None:
        return AIMessage(content=GREETING_REPLY if intent == "greeting" else OUT_OF_SCOPE_REPLY)
    return planner(turn, call, reply)


def classify(text: str) -> str:
    has_txn = bool(TXN_RE.search(text))
    if any(k in text for k in ("refund", "estorno", "reembolso", "money back", "charged twice")):
        return "refund"
    if ("card" in text or "cartão" in text or "cartao" in text) and any(
        k in text for k in ("block", "lost", "stolen", "bloque", "perdi", "roubad")
    ):
        return "block_card"
    pix_problem = any(
        k in text
        for k in (
            "arriv",
            "fail",
            "never",
            "didn't",
            "did not",
            "not received",
            "pending",
            "stuck",
            "missing",
        )
    )
    if (has_txn and any(k in text for k in ("status", "happened", "what", "where", "check"))) or (
        "pix" in text and pix_problem
    ):
        return "payment_status"
    if any(
        k in text for k in ("transaction", "statement", "extrato", "recent", "history", "spent", "purchases")
    ):
        return "transactions"
    if any(k in text for k in ("balance", "saldo", "how much money", "account summary", "my accounts")):
        return "balance"
    if re.fullmatch(r"\W*(hi|hello|hey|oi|olá|ola|good (morning|afternoon|evening))\W*", text):
        return "greeting"
    return "out_of_scope"


# --- Planners: each returns the next action for its intent ---------------------------------


def plan_balance(turn: Turn, call, reply) -> AIMessage:
    summary = turn.result("get_account_summary")
    if summary is None:
        return call("get_account_summary")
    data, err = summary
    if err:
        return reply(f"I couldn't load your accounts right now ({data}).", "empathetic")
    lines = [
        f"- {a['kind'].capitalize()} account {a['account_id']}: {a['balance']}" for a in data["accounts"]
    ]
    return reply("Here are your balances:\n" + "\n".join(lines) + f"\nTotal: {data['total_balance']}.")


def plan_transactions(turn: Turn, call, reply) -> AIMessage:
    listing = turn.result("list_recent_transactions")
    if listing is None:
        return call("list_recent_transactions", limit=5)
    data, err = listing
    if err:
        return reply(f"I couldn't load your transactions right now ({data}).", "empathetic")
    lines = [
        f"- {t['date']} {t['transaction_id']} {t['kind'].replace('_', ' ')} {t['amount']} "
        f"({t['counterparty']}) - {t['status']}"
        for t in data
    ]
    return reply("Your most recent transactions:\n" + "\n".join(lines))


def plan_payment_status(turn: Turn, call, reply) -> AIMessage:
    match = TXN_RE.search(turn.text)
    if match is None:
        listing = turn.result("list_recent_transactions")
        if listing is None:
            return call("list_recent_transactions", limit=10)
        data, err = listing
        if err:
            return reply(f"I couldn't load your transactions right now ({data}).", "empathetic")
        candidate = pick_problem_pix(data, turn.lower)
        if candidate is None:
            return reply("I couldn't find any failed or pending Pix among your recent transactions.")
        txn_id = candidate["transaction_id"]
    else:
        txn_id = match.group(0).upper()

    status = turn.result("check_transaction_status")
    if status is None:
        return call("check_transaction_status", transaction_id=txn_id)
    data, err = status
    if err:
        return reply(f"I couldn't check {txn_id}: {data}", "empathetic")
    return reply(explain_status(data), "empathetic" if data["status"] in ("failed", "pending") else "neutral")


def plan_refund(turn: Turn, call, reply) -> AIMessage:
    match = TXN_RE.search(turn.text)
    if match is None:
        return AIMessage(
            content="I can help with a refund. Which transaction is it? Please share its ID (e.g. TX-1234)."
        )
    txn_id = match.group(0).upper()
    status = turn.result("check_transaction_status")
    if status is None:
        return call("check_transaction_status", transaction_id=txn_id)
    data, err = status
    if err:
        return reply(f"I couldn't check {txn_id}: {data}", "empathetic")
    refund = data["refund"]
    if not refund["refundable"]:
        return reply(f"Transaction {txn_id} is not eligible for a refund: {refund['reason']}")

    result = turn.result("issue_refund")
    if result is None and not turn.called("issue_refund"):
        reason = (
            f"Failed Pix ({data['network'].get('failure_code')}) not returned by the network"
            if data["kind"] == "pix_out"
            else "Disputed card charge"
        )
        return call(
            "issue_refund",
            transaction_id=txn_id,
            amount_brl=refund["max_refund_cents"] / 100,
            reason=reason,
        )
    if result is None:  # pragma: no cover - the graph always answers a call before re-asking
        return AIMessage(content="Your refund request is waiting for review.")
    data, err = result
    if err:
        if "rejected by" in str(data).lower():
            return reply(
                f"Your refund request for {txn_id} was reviewed by our team and was not approved, "
                "so no refund was issued. A specialist can give you more details if needed.",
                "empathetic",
            )
        return reply(f"I couldn't issue the refund for {txn_id}: {data}", "empathetic")
    return reply(
        f"Your refund {data['refund_id']} of {data['amount']} for {txn_id} was approved and credited to "
        f"your account {data['credited_to_account']}."
    )


def plan_block_card(turn: Turn, call, reply) -> AIMessage:
    summary = turn.result("get_account_summary")
    if summary is None:
        return call("get_account_summary")
    data, err = summary
    if err:
        return reply(f"I couldn't load your cards right now ({data}).", "empathetic")
    active = [c for c in data["cards"] if c["status"] == "active"]
    wanted = [c for c in active if c["kind"] in turn.lower]
    candidates = wanted or active
    if not candidates:
        return reply("All your cards are already blocked.")
    if len(candidates) > 1:
        options = ", ".join(f"{c['kind']} card ending in {c['last4']}" for c in candidates)
        return AIMessage(content=f"Which card should I block? You have: {options}.")
    card = candidates[0]

    result = turn.result("block_card")
    if result is None:
        reason = "stolen" if any(k in turn.lower for k in ("stolen", "roubad")) else "lost"
        return call("block_card", card_id=card["card_id"], reason=reason)
    data, err = result
    if err:
        if "rejected by" in str(data).lower():
            return reply(
                f"The block of your {card['kind']} card ending in {card['last4']} was not approved by our "
                "team. Please call us so we can verify your identity.",
                "empathetic",
            )
        return reply(f"I couldn't block the card: {data}", "empathetic")
    return reply(
        f"Your {data['kind']} card ending in {data['last4']} is now blocked. {data['next_steps']}",
        "empathetic",
    )


def pick_problem_pix(transactions: list[dict], text: str) -> dict | None:
    candidates = [t for t in transactions if t["kind"] == "pix_out" and t["status"] in ("failed", "pending")]
    words = set(re.findall(r"[a-z]{3,}", text))
    for txn in candidates:  # prefer the one the customer described ("rent", "gift", a name...)
        vocab = set(re.findall(r"[a-z]{3,}", f"{txn['description']} {txn['counterparty']}".lower()))
        if words & vocab:
            return txn
    return candidates[0] if candidates else None


def explain_status(data: dict) -> str:
    net = data["network"]
    kind = data["kind"].replace("_", " ")
    head = f"Your {kind} {data['transaction_id']} of {data['amount']} to {data['counterparty']}"
    status = net["network_status"]
    if status == "REJECTED":
        refund = data["refund"]
        text = (
            f"{head} failed: it was rejected by the receiving institution "
            f"({net['failure_code']}: {net['failure_reason']}). The money has not been returned to your "
            "account yet."
        )
        if refund["refundable"]:
            text += f" It is eligible for a refund of up to {refund['max_refund']}; just ask me to refund it."
        return text
    if status == "RETURNED":
        return (
            f"{head} failed ({net['failure_reason']}), but the funds were already returned to your account."
        )
    if status == "IN_PROCESS":
        return f"{head} is still pending, being processed by the receiving bank. {net['expected_resolution']}"
    if status == "SETTLED":
        return (
            f"{head} was completed and settled. If the recipient can't see it, they can ask their bank "
            f"using the end-to-end ID {net['end_to_end_id']}."
        )
    if status == "IN_DISPUTE":
        return (
            f"{head} is under dispute. It is eligible for a refund of up to {data['refund']['max_refund']}."
        )
    return f"{head} was captured successfully by the card network."


PLANS = {
    "balance": plan_balance,
    "transactions": plan_transactions,
    "payment_status": plan_payment_status,
    "refund": plan_refund,
    "block_card": plan_block_card,
}
