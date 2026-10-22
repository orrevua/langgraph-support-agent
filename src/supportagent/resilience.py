"""Retries with exponential backoff and jitter, for flaky dependencies (the payments API).

Only *transient* errors are retried (timeouts, 503s). Retrying a "not found" or a validation
error just wastes time, so callers pass the exception types that are safe to retry.

LangGraph also offers `RetryPolicy` at the node level (`add_node(..., retry_policy=...)`). We
retry inside the tool instead so that one flaky call does not re-run the whole tools node
(which might contain other, non-idempotent tool calls).
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

log = logging.getLogger(__name__)


def retry_with_backoff[T](
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call `fn` up to `attempts` times. Delay doubles each time (+ up to 10% jitter)."""
    rng = rng or random.Random()
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay += delay * 0.1 * rng.random()  # jitter avoids synchronized retry storms
            log.warning("transient error (attempt %d/%d): %s; retrying in %.2fs", attempt, attempts, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")
