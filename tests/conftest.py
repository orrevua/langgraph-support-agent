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


def _postgres_available(url: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def database_url() -> str:
    url = Settings(_env_file=None).database_url
    if not _postgres_available(url):
        pytest.skip("Postgres not reachable")
    return url


@pytest.fixture
def pg_repo(database_url):
    from supportagent.repository.postgres import PostgresBankRepository

    repo = PostgresBankRepository(database_url)
    repo.reset_and_seed()
    yield repo
    repo.close()
