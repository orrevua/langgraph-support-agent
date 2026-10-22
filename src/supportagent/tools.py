"""The support agent's tools.

Design notes (worth reading before the code):

* **Dependency injection via a factory.** `build_tools(ctx)` closes over a `ToolContext`
  (repository, payments client, settings), so the same tool code runs against the in-memory
  repo in tests and Postgres in the API. No globals.

* **Injected arguments.** `customer_id` is annotated with `InjectedToolArg`: it is *hidden* from
  the schema the model sees and filled in by our tools node from the graph state (the
  authenticated session). The model therefore cannot ask for another customer's data just by
  passing a different id - authorization is enforced in code, not in the prompt.

* **Errors are values.** Business-rule violations raise `ToolError`; the tools node turns them
  into a ToolMessage with `status="error"` so the model can explain or recover, instead of the
  whole graph crashing.

* **Sensitive tools** (`SENSITIVE_TOOLS`) move money or change account state. The graph pauses
  for human approval before running them (see `agent/graph.py`), and the tools node refuses to
  run them without a recorded approval.

* Tool outputs are JSON strings: easy for models to read, easy for the fake policy to parse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool, InjectedToolArg, tool

from supportagent.config import Settings
from supportagent.domain.models import Refund, Transaction, TxnStatus, format_brl
from supportagent.guardrails import mask_cpf, mask_email, mask_phone
from supportagent.payments import PaymentNotFound, PaymentsClient, PaymentsUnavailable
from supportagent.repository.base import BankRepository
from supportagent.resilience import retry_with_backoff

SENSITIVE_TOOLS = frozenset({"issue_refund", "block_card"})


class ToolError(Exception):
    """An expected, explainable failure (not found, not allowed, over limit, ...)."""


@dataclass
class ToolContext:
    repo: BankRepository
    payments: PaymentsClient
    settings: Settings


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _owned_transaction(ctx: ToolContext, transaction_id: str, customer_id: str) -> Transaction:
    txn = ctx.repo.get_transaction(transaction_id.strip().upper())
    # Same message whether it does not exist or belongs to someone else: do not leak existence.
    if txn is None or txn.customer_id != customer_id:
        raise ToolError(f"Transaction {transaction_id} was not found on this customer's accounts.")
    return txn


def refund_eligibility(txn: Transaction, refunds: list[Refund], hard_limit_cents: int) -> dict[str, Any]:
    """Pure business rule: can this transaction be refunded, and up to how much?

    - failed Pix whose money was debited but not returned by the network -> refundable
    - disputed card purchase (e.g. duplicate charge)                    -> refundable
    - anything else                                                       -> not refundable
    """
    already = sum(r.amount_cents for r in refunds)
    remaining = max(0, txn.amount_cents - already)
    if txn.kind == "pix_in":
        reason = "Incoming transfers cannot be refunded by us."
    elif txn.status == "failed" and txn.funds_returned:
        reason = "The payments network already returned the funds automatically."
    elif txn.status == "failed" or txn.status == "disputed":
        reason = None
    else:
        reason = f"Transactions with status '{txn.status}' are not eligible for a refund."
    if reason is None and remaining == 0:
        reason = "This transaction was already fully refunded."
    refundable = reason is None
    return {
        "refundable": refundable,
        "reason": reason,
        "already_refunded": format_brl(already),
        "max_refund": format_brl(min(remaining, hard_limit_cents)) if refundable else format_brl(0),
        "max_refund_cents": min(remaining, hard_limit_cents) if refundable else 0,
        "needs_escalation": refundable and remaining > hard_limit_cents,
    }


def to_cents(amount_brl: float | str) -> int:
    try:
        value = Decimal(str(amount_brl))
    except InvalidOperation as exc:
        raise ToolError(f"Invalid amount: {amount_brl!r}") from exc
    return int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    repo, settings = ctx.repo, ctx.settings

    @tool
    def lookup_customer(
        customer_id: Annotated[str, InjectedToolArg],
        identifier: str | None = None,
    ) -> str:
        """Get the authenticated customer's profile (name, tier, masked contact data).

        If the user mentions a CPF or e-mail, pass it as `identifier` to check whether it belongs
        to the authenticated customer. Data about other people is never returned.
        """
        customer = repo.get_customer(customer_id)
        if customer is None:
            raise ToolError("Authenticated customer not found.")
        result: dict[str, Any] = {
            "customer_id": customer.id,
            "name": customer.full_name,
            "first_name": customer.first_name,
            "tier": customer.tier,
            "cpf": mask_cpf(customer.cpf),
            "email": mask_email(customer.email),
            "phone": mask_phone(customer.phone),
        }
        if identifier:
            ident = identifier.strip().lower()
            digits = "".join(ch for ch in ident if ch.isdigit())
            own_digits = "".join(ch for ch in customer.cpf if ch.isdigit())
            result["identifier_belongs_to_customer"] = ident == customer.email.lower() or (
                bool(digits) and digits == own_digits
            )
        return _dump(result)

    @tool
    def get_account_summary(customer_id: Annotated[str, InjectedToolArg]) -> str:
        """Get the customer's accounts with balances and their cards (masked) with status."""
        accounts = repo.list_accounts(customer_id)
        cards = repo.list_cards(customer_id)
        return _dump(
            {
                "accounts": [
                    {"account_id": a.id, "kind": a.kind, "balance": format_brl(a.balance_cents)} for a in accounts
                ],
                "total_balance": format_brl(sum(a.balance_cents for a in accounts)),
                "cards": [{"card_id": c.id, "kind": c.kind, "last4": c.last4, "status": c.status} for c in cards],
            }
        )

    @tool
    def list_recent_transactions(
        customer_id: Annotated[str, InjectedToolArg],
        limit: int = 5,
        status: TxnStatus | None = None,
    ) -> str:
        """List the customer's most recent transactions (newest first), optionally by status.

        Args:
            limit: how many transactions to return (1-20).
            status: only return transactions with this status (completed, pending, failed, disputed).
        """
        if not 1 <= limit <= 20:
            raise ToolError("limit must be between 1 and 20")
        txns = repo.list_transactions(customer_id, limit=limit, status=status)
        return _dump(
            [
                {
                    "transaction_id": t.id,
                    "date": t.created_at.date().isoformat(),
                    "kind": t.kind,
                    "amount": format_brl(t.amount_cents),
                    "counterparty": t.counterparty,
                    "description": t.description,
                    "status": t.status,
                }
                for t in txns
            ]
        )

    @tool
    def check_transaction_status(customer_id: Annotated[str, InjectedToolArg], transaction_id: str) -> str:
        """Ask the payments network for the live status of one transaction (e.g. a Pix that did
        not arrive). Also says whether it can be refunded and up to which amount."""
        txn = _owned_transaction(ctx, transaction_id, customer_id)
        try:
            status = retry_with_backoff(
                lambda: ctx.payments.get_status(txn.id),
                attempts=settings.payments_retry_attempts,
                base_delay=settings.payments_retry_base_delay_s,
                retry_on=(PaymentsUnavailable,),
            )
        except PaymentsUnavailable as exc:
            raise ToolError(
                "The payments network is temporarily unavailable. Please try again in a few minutes."
            ) from exc
        except PaymentNotFound as exc:
            raise ToolError(f"The payments network has no record of {txn.id}.") from exc
        return _dump(
            {
                "transaction_id": txn.id,
                "kind": txn.kind,
                "amount": format_brl(txn.amount_cents),
                "counterparty": txn.counterparty,
                "description": txn.description,
                "status": txn.status,
                "network": status.model_dump(mode="json", exclude={"checked_at", "transaction_id"}),
                "refund": refund_eligibility(txn, repo.list_refunds(txn.id), settings.refund_hard_limit_cents),
            }
        )

    @tool
    def issue_refund(
        customer_id: Annotated[str, InjectedToolArg],
        approved_by: Annotated[str, InjectedToolArg],
        transaction_id: str,
        amount_brl: float,
        reason: str,
    ) -> str:
        """SENSITIVE - requires human approval. Refund money to the customer for an eligible
        transaction (failed Pix not returned by the network, or a disputed card charge).
        Always call check_transaction_status first to get the maximum refundable amount.

        Args:
            transaction_id: e.g. "TX-1003".
            amount_brl: amount in reais, e.g. 1200.00. Must not exceed the refundable amount.
            reason: short justification shown to the reviewer and stored with the refund.
        """
        txn = _owned_transaction(ctx, transaction_id, customer_id)
        cents = to_cents(amount_brl)
        if cents <= 0:
            raise ToolError("Refund amount must be positive.")
        if cents > settings.refund_hard_limit_cents:
            raise ToolError(
                f"Refunds above {format_brl(settings.refund_hard_limit_cents)} must be escalated to the "
                "back-office team and cannot be issued in chat."
            )
        eligibility = refund_eligibility(txn, repo.list_refunds(txn.id), settings.refund_hard_limit_cents)
        if not eligibility["refundable"]:
            raise ToolError(f"Transaction {txn.id} is not refundable: {eligibility['reason']}")
        if cents > eligibility["max_refund_cents"]:
            raise ToolError(f"Amount exceeds the refundable maximum of {eligibility['max_refund']}.")
        refund = repo.create_refund(
            transaction_id=txn.id,
            customer_id=customer_id,
            amount_cents=cents,
            reason=reason[:200],
            approved_by=approved_by,
        )
        return _dump(
            {
                "status": "refunded",
                "refund_id": refund.id,
                "transaction_id": txn.id,
                "amount": format_brl(cents),
                "credited_to_account": txn.account_id,
                "approved_by": approved_by,
            }
        )

    @tool
    def block_card(
        customer_id: Annotated[str, InjectedToolArg],
        approved_by: Annotated[str, InjectedToolArg],
        card_id: str,
        reason: Literal["lost", "stolen", "suspected_fraud", "customer_request"],
    ) -> str:
        """SENSITIVE - requires human approval. Permanently block one of the customer's cards.
        Get the card_id from get_account_summary first."""
        card = repo.get_card(card_id.strip().upper())
        if card is None or card.customer_id != customer_id:
            raise ToolError(f"Card {card_id} was not found for this customer.")
        if card.status == "blocked":
            return _dump({"status": "already_blocked", "card_id": card.id, "last4": card.last4})
        card = repo.set_card_status(card.id, "blocked")
        return _dump(
            {
                "status": "blocked",
                "card_id": card.id,
                "kind": card.kind,
                "last4": card.last4,
                "reason": reason,
                "approved_by": approved_by,
                "next_steps": "You can request a replacement in the app under Cards > Request new card.",
            }
        )

    @tool
    def draft_reply(
        customer_id: Annotated[str, InjectedToolArg],
        message: str,
        tone: Literal["neutral", "empathetic"] = "neutral",
    ) -> str:
        """Format the final customer-facing reply (greeting + your message + sign-off).
        Call this once at the end, then answer the customer with the returned `reply` text."""
        customer = repo.get_customer(customer_id)
        name = customer.first_name if customer else "there"
        opening = "I'm sorry for the trouble. " if tone == "empathetic" else ""
        reply = (
            f"Hi {name},\n\n{opening}{message.strip()}\n\n"
            "If there is anything else, just reply here.\nAurora Bank Support"
        )
        return _dump({"reply": reply})

    return [
        lookup_customer,
        get_account_summary,
        list_recent_transactions,
        check_transaction_status,
        draft_reply,
        issue_refund,
        block_card,
    ]
