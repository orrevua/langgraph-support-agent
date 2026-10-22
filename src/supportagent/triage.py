"""Ticket triage: a structured-output example used in `examples/01_fundamentals.py`.

`with_structured_output(TicketTriage)` makes the model return an instance of this Pydantic
model instead of free text. Field descriptions are sent to the model as part of the schema, so
they double as instructions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TicketTriage(BaseModel):
    """Classification of an incoming customer-support message."""

    category: Literal["pix_transfer", "card", "refund", "account", "other"] = Field(
        description="Main topic of the message."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="high = money missing or card compromised; low = general question."
    )
    sentiment: Literal["negative", "neutral", "positive"]
    summary: str = Field(description="One sentence summary, in English, without personal data.")


_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("refund", ("refund", "estorno", "reembolso", "money back", "charged twice", "duplicate")),
    ("card", ("card", "cartão", "cartao", "lost", "stolen", "block")),
    ("pix_transfer", ("pix", "transfer", "transferência", "sent")),
    ("account", ("balance", "saldo", "account", "statement", "extrato")),
]


def heuristic_triage(text: str) -> dict:
    """Keyword triage used by the offline fake model (a real model does this from the schema)."""
    lower = text.lower()
    category = next((cat for cat, keys in _CATEGORY_KEYWORDS if any(k in lower for k in keys)), "other")
    urgent = any(k in lower for k in ("stolen", "fraud", "never arrived", "missing", "urgent", "not received"))
    negative = any(k in lower for k in ("angry", "terrible", "!", "still", "never", "unacceptable"))
    return {
        "category": category,
        "urgency": "high" if urgent else ("medium" if category != "other" else "low"),
        "sentiment": "negative" if negative else "neutral",
        "summary": f"Customer message about {category.replace('_', ' ')}.",
    }
