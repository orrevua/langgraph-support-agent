import pytest

from supportagent.resilience import retry_with_backoff


class Flaky:
    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("boom")
        return "ok"


def test_retries_then_succeeds_with_exponential_delays():
    delays = []
    fn = Flaky(2)
    assert retry_with_backoff(fn, attempts=3, base_delay=0.1, retry_on=(TimeoutError,), sleep=delays.append) == "ok"
    assert fn.calls == 3
    assert 0.1 <= delays[0] <= 0.11 and 0.2 <= delays[1] <= 0.22


def test_gives_up_after_attempts():
    with pytest.raises(TimeoutError):
        retry_with_backoff(Flaky(5), attempts=3, retry_on=(TimeoutError,), sleep=lambda _: None)


def test_does_not_retry_other_errors():
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        retry_with_backoff(bad, retry_on=(TimeoutError,), sleep=lambda _: None)
    assert len(calls) == 1
