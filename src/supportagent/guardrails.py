"""Guardrails: PII masking (more is added in the hardening milestone)."""

from __future__ import annotations

import re

CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,15}(\d{4})\b")
PHONE_RE = re.compile(r"\+?55\s?\(?\d{2}\)?\s?9?\d{4}-?\d{4}")


def mask_cpf(cpf: str) -> str:
    digits = "".join(ch for ch in cpf if ch.isdigit())
    return f"***.{digits[3:6]}.***-**" if len(digits) == 11 else "***"


def mask_email(email: str) -> str:
    return EMAIL_RE.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", email)


def mask_phone(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    return f"(**) *****-{digits[-4:]}" if len(digits) >= 8 else "***"
