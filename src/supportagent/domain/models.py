"""Pydantic models for the mock fintech.

Money is always stored as integer cents (`amount_cents`) to avoid float rounding; use
`format_brl` to display it the Brazilian way (R$ 1.234,56).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

TxnKind = Literal["pix_out", "pix_in", "card_purchase"]
TxnStatus = Literal["completed", "pending", "failed", "disputed"]
CardStatus = Literal["active", "blocked"]


def format_brl(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    reais, centavos = divmod(abs(cents), 100)
    return f"{sign}R$ {reais:,}".replace(",", ".") + f",{centavos:02d}"


class Customer(BaseModel):
    id: str
    full_name: str
    cpf: str  # Brazilian taxpayer id - PII, never log it unmasked
    email: str
    phone: str
    tier: Literal["standard", "premium"] = "standard"

    @property
    def first_name(self) -> str:
        return self.full_name.split()[0]


class Account(BaseModel):
    id: str
    customer_id: str
    kind: Literal["checking", "savings"]
    balance_cents: int


class Card(BaseModel):
    id: str
    customer_id: str
    kind: Literal["debit", "credit"]
    last4: str
    status: CardStatus = "active"


class Transaction(BaseModel):
    id: str
    account_id: str
    customer_id: str
    kind: TxnKind
    amount_cents: int
    counterparty: str
    description: str
    status: TxnStatus
    created_at: datetime
    card_id: str | None = None
    # Pix-specific fields (filled by the payments network)
    end_to_end_id: str | None = None
    failure_code: str | None = None
    failure_reason: str | None = None
    funds_returned: bool = False


class Refund(BaseModel):
    id: str
    transaction_id: str
    customer_id: str
    amount_cents: int
    reason: str
    status: Literal["completed", "pending"] = "completed"
    created_at: datetime
    approved_by: str | None = None


class ApprovalRecord(BaseModel):
    """One human decision about a sensitive action (the audit trail)."""

    id: int | None = None
    thread_id: str
    tool_call_id: str
    tool_name: str
    proposed_args: dict
    final_args: dict | None
    decision: Literal["approved", "rejected"]
    reviewer: str
    reason: str | None = None
    created_at: datetime | None = None
