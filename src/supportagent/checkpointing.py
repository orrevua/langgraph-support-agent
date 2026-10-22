"""Checkpointers: how LangGraph remembers conversations.

A checkpointer saves a snapshot of the graph state after every super-step, keyed by
`thread_id` (one per conversation). That gives you, for free:

* multi-turn memory: invoke again with the same thread_id and the old messages are there;
* human-in-the-loop: a paused graph (interrupt) is just a saved checkpoint waiting for input;
* time travel: list old checkpoints and resume or fork from any of them.

`MemorySaver` keeps everything in a dict (tests, notebooks). `PostgresSaver` stores checkpoints
in Postgres so conversations survive restarts and can be shared by several API workers.

Docs: https://langchain-ai.github.io/langgraph/concepts/persistence/
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def setup_postgres_checkpointer(database_url: str) -> None:
    """Create/migrate LangGraph's checkpoint tables (safe to call repeatedly)."""
    with PostgresSaver.from_conn_string(database_url) as saver:
        saver.setup()


@contextmanager
def open_checkpointer(kind: Literal["postgres", "memory"], database_url: str) -> Iterator[BaseCheckpointSaver]:
    if kind == "memory":
        yield MemorySaver()
        return
    # PostgresSaver needs autocommit and dict rows; prepare_threshold=0 disables server-side
    # prepared statements (required when running behind PgBouncer in transaction mode).
    pool = ConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    try:
        saver = PostgresSaver(pool)
        saver.setup()
        yield saver
    finally:
        pool.close()
