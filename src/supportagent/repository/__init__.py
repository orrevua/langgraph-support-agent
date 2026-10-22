"""Data access. Tools depend on the `BankRepository` protocol, never on a concrete database.

- `InMemoryBankRepository`: fast, isolated, used by unit tests and offline evals.
- `PostgresBankRepository`: the real thing, used by the API and demo (`STORAGE=postgres`).
"""

from supportagent.repository.base import BankRepository
from supportagent.repository.memory import InMemoryBankRepository

__all__ = ["BankRepository", "InMemoryBankRepository", "build_repository"]


def build_repository(settings=None) -> BankRepository:
    from supportagent.config import get_settings

    settings = settings or get_settings()
    if settings.storage == "memory":
        return InMemoryBankRepository()
    from supportagent.repository.postgres import PostgresBankRepository

    return PostgresBankRepository(settings.database_url)
