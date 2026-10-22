"""The repository interface (a `typing.Protocol`, i.e. structural typing: any class with
these methods qualifies, no inheritance required)."""

from __future__ import annotations

from typing import Protocol

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


class BankRepository(Protocol):
    def get_customer(self, customer_id: str) -> Customer | None: ...
    def list_accounts(self, customer_id: str) -> list[Account]: ...
    def list_cards(self, customer_id: str) -> list[Card]: ...
    def get_card(self, card_id: str) -> Card | None: ...
    def set_card_status(self, card_id: str, status: CardStatus) -> Card: ...
    def list_transactions(
        self, customer_id: str, *, limit: int = 10, status: TxnStatus | None = None
    ) -> list[Transaction]: ...
    def get_transaction(self, transaction_id: str) -> Transaction | None: ...
    def list_refunds(self, transaction_id: str) -> list[Refund]: ...
    def create_refund(
        self,
        *,
        transaction_id: str,
        customer_id: str,
        amount_cents: int,
        reason: str,
        approved_by: str | None,
    ) -> Refund: ...
    def record_approval(self, record: ApprovalRecord) -> ApprovalRecord: ...
    def list_approvals(self, thread_id: str | None = None) -> list[ApprovalRecord]: ...
