"""Database CLI: `uv run python -m supportagent.db seed` creates the schema and (re)seeds it.

Also creates LangGraph's checkpoint tables, so a fresh database is fully ready.
"""

from __future__ import annotations

import argparse

from supportagent.checkpointing import setup_postgres_checkpointer
from supportagent.config import get_settings
from supportagent.repository.postgres import PostgresBankRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seed"], help="seed: create tables and load seed data")
    parser.parse_args()
    settings = get_settings()
    repo = PostgresBankRepository(settings.database_url)
    try:
        repo.reset_and_seed()
    finally:
        repo.close()
    setup_postgres_checkpointer(settings.database_url)
    print(f"Seeded {settings.database_url.rsplit('/', 1)[-1]} and created checkpoint tables.")


if __name__ == "__main__":
    main()
