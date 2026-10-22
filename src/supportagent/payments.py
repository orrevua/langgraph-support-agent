"""A mock "payments API" (think: the bank's Pix/SPI gateway) with configurable flakiness.

Two ways to use it, both behind the same `PaymentsClient` protocol:

- `MockPaymentsClient`: in-process, used by tests, evals and the default app config.
- `HttpPaymentsClient`: calls the FastAPI router below over HTTP, to see real network
  behaviour (timeouts, 503s). The router is mounted at `/mock-payments` by `supportagent.api`.

Failures are *transient* (`PaymentsUnavailable`) so the tool layer can retry them with backoff.
"""

from __future__ import annotations

import random
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from supportagent.repository.base import BankRepository


class PaymentsUnavailable(Exception):
    """Transient failure (timeout / 503). Safe to retry."""


class PaymentNotFound(Exception):
    """The payments network has no record of this transaction. Not retryable."""


NetworkStatus = Literal["SETTLED", "IN_PROCESS", "REJECTED", "RETURNED", "CAPTURED", "IN_DISPUTE"]


class PaymentStatus(BaseModel):
    transaction_id: str
    end_to_end_id: str | None
    network_status: NetworkStatus
    failure_code: str | None = None
    failure_reason: str | None = None
    funds_returned: bool = False
    expected_resolution: str | None = None
    checked_at: datetime


class PaymentsClient(Protocol):
    def get_status(self, transaction_id: str) -> PaymentStatus: ...


class MockPaymentsClient:
    """Derives the network status from the seeded transactions, with optional chaos.

    fail_first: the first N calls raise PaymentsUnavailable (deterministic, great for tests).
    failure_rate: probability of a random transient failure (seeded RNG -> reproducible).
    latency_ms: artificial delay per call.
    """

    def __init__(
        self,
        repo: BankRepository,
        *,
        latency_ms: int = 0,
        fail_first: int = 0,
        failure_rate: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.repo = repo
        self.latency_ms = latency_ms
        self.fail_first = fail_first
        self.failure_rate = failure_rate
        self.calls = 0
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def get_status(self, transaction_id: str) -> PaymentStatus:
        with self._lock:
            self.calls += 1
            call_number = self.calls
            roll = self._rng.random()
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        if call_number <= self.fail_first or roll < self.failure_rate:
            raise PaymentsUnavailable("payments gateway returned 503 Service Unavailable")

        txn = self.repo.get_transaction(transaction_id)
        if txn is None:
            raise PaymentNotFound(transaction_id)
        now = datetime.now(UTC)
        status: NetworkStatus
        expected = None
        if txn.kind == "card_purchase":
            status = "IN_DISPUTE" if txn.status == "disputed" else "CAPTURED"
        elif txn.status == "completed":
            status = "SETTLED"
        elif txn.status == "pending":
            status = "IN_PROCESS"
            expected = "Pix usually settles within minutes; the receiving bank has up to 1 business day."
        elif txn.funds_returned:
            status = "RETURNED"
        else:
            status = "REJECTED"
            expected = (now + timedelta(days=1)).date().isoformat()
        return PaymentStatus(
            transaction_id=txn.id,
            end_to_end_id=txn.end_to_end_id,
            network_status=status,
            failure_code=txn.failure_code,
            failure_reason=txn.failure_reason,
            funds_returned=txn.funds_returned,
            expected_resolution=expected,
            checked_at=now,
        )


class HttpPaymentsClient:
    """Talks to the mock payments router over HTTP. Maps HTTP errors to our exceptions."""

    def __init__(self, base_url: str, *, timeout_s: float = 2.0, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def get_status(self, transaction_id: str) -> PaymentStatus:
        try:
            response = self._client.get(f"{self.base_url}/v1/transactions/{transaction_id}/status")
        except httpx.TransportError as exc:  # timeouts, connection refused, ...
            raise PaymentsUnavailable(str(exc)) from exc
        if response.status_code == 404:
            raise PaymentNotFound(transaction_id)
        if response.status_code >= 500:
            raise PaymentsUnavailable(f"HTTP {response.status_code}")
        response.raise_for_status()
        return PaymentStatus.model_validate(response.json())


def build_payments_router(client: PaymentsClient) -> APIRouter:
    router = APIRouter(tags=["mock-payments"])

    @router.get("/v1/transactions/{transaction_id}/status", response_model=PaymentStatus)
    def get_status(transaction_id: str) -> PaymentStatus:
        try:
            return client.get_status(transaction_id)
        except PaymentsUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except PaymentNotFound as exc:
            raise HTTPException(status_code=404, detail="transaction not found") from exc

    return router
