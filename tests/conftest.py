import os

import pytest

# Tests are always offline, even if the developer has an API key exported.
os.environ["LLM_PROVIDER"] = "fake"
os.environ.pop("ANTHROPIC_API_KEY", None)

from supportagent.config import Settings  # noqa: E402
from supportagent.payments import MockPaymentsClient  # noqa: E402
from supportagent.repository import InMemoryBankRepository  # noqa: E402
from supportagent.tools import ToolContext  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings(storage="memory", llm_provider="fake", payments_retry_base_delay_s=0.0, _env_file=None)


@pytest.fixture
def repo() -> InMemoryBankRepository:
    return InMemoryBankRepository()


@pytest.fixture
def payments(repo) -> MockPaymentsClient:
    return MockPaymentsClient(repo)


@pytest.fixture
def ctx(repo, payments, settings) -> ToolContext:
    return ToolContext(repo=repo, payments=payments, settings=settings)
