"""In-memory repository seeded from `domain.seed`. Each instance gets its own deep copy."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from supportagent.domain import seed
from supportagent.domain.models import (
    Account,
    ApprovalRecord,
    Card,
    CardStatus,
    Customer,
    Refund,
    Transaction,
    TxnStatus,
)


class InMemoryBankRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.customers = {c.id: c.model_copy(deep=True) for c in seed.CUSTOMERS}
        self.accounts = {a.id: a.model_copy(deep=True) for a in seed.ACCOUNTS}
        self.cards = {c.id: c.model_copy(deep=True) for c in seed.CARDS}
        self.transactions = {t.id: t.model_copy(deep=True) for t in seed.TRANSACTIONS}
        self.refunds = {r.id: r.model_copy(deep=True) for r in seed.REFUNDS}
        self.approvals: list[ApprovalRecord] = []

    def get_customer(self, customer_id: str) -> Customer | None:
        return self.customers.get(customer_id)

    def list_accounts(self, customer_id: str) -> list[Account]:
        return [a for a in self.accounts.values() if a.customer_id == customer_id]

    def list_cards(self, customer_id: str) -> list[Card]:
        return [c for c in self.cards.values() if c.customer_id == customer_id]

    def get_card(self, card_id: str) -> Card | None:
        return self.cards.get(card_id)

    def set_card_status(self, card_id: str, status: CardStatus) -> Card:
        card = self.cards[card_id]
        card.status = status
        return card

    def list_transactions(
        self, customer_id: str, *, limit: int = 10, status: TxnStatus | None = None
    ) -> list[Transaction]:
        rows = [
            t
            for t in self.transactions.values()
            if t.customer_id == customer_id and (status is None or t.status == status)
        ]
        return sorted(rows, key=lambda t: t.created_at, reverse=True)[:limit]

    def get_transaction(self, transaction_id: str) -> Transaction | None:
        return self.transactions.get(transaction_id)

    def list_refunds(self, transaction_id: str) -> list[Refund]:
        return [r for r in self.refunds.values() if r.transaction_id == transaction_id]

    def create_refund(
        self,
        *,
        transaction_id: str,
        customer_id: str,
        amount_cents: int,
        reason: str,
        approved_by: str | None,
    ) -> Refund:
        with self._lock:
            refund = Refund(
                id=f"RF-{len(self.refunds) + 1:03d}",
                transaction_id=transaction_id,
                customer_id=customer_id,
                amount_cents=amount_cents,
                reason=reason,
                created_at=datetime.now(UTC),
                approved_by=approved_by,
            )
            self.refunds[refund.id] = refund
            txn = self.transactions[transaction_id]
            self.accounts[txn.account_id].balance_cents += amount_cents
            return refund

    def record_approval(self, record: ApprovalRecord) -> ApprovalRecord:
        with self._lock:
            stored = record.model_copy(update={"id": len(self.approvals) + 1, "created_at": datetime.now(UTC)})
            self.approvals.append(stored)
            return stored

    def list_approvals(self, thread_id: str | None = None) -> list[ApprovalRecord]:
        return [a for a in self.approvals if thread_id is None or a.thread_id == thread_id]
