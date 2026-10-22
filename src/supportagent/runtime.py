"""Wiring helpers: build the repository, payments client, model and graph from settings."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.types import Checkpointer

from supportagent.config import Settings, get_settings
from supportagent.llm import get_chat_model
from supportagent.payments import HttpPaymentsClient, MockPaymentsClient, PaymentsClient
from supportagent.repository import BankRepository, build_repository
from supportagent.tools import ToolContext


def build_payments(repo: BankRepository, settings: Settings) -> PaymentsClient:
    if settings.payments_mode == "http":
        return HttpPaymentsClient(settings.payments_base_url)
    return MockPaymentsClient(
        repo,
        latency_ms=settings.payments_latency_ms,
        fail_first=settings.payments_fail_first,
        failure_rate=settings.payments_failure_rate,
    )


def build_agent(
    settings: Settings | None = None,
    *,
    repo: BankRepository | None = None,
    payments: PaymentsClient | None = None,
    model: BaseChatModel | None = None,
    checkpointer: Checkpointer = None,
):
    """Return (compiled graph, tool context). Any dependency can be overridden (tests do)."""
    from supportagent.agent.graph import build_graph

    settings = settings or get_settings()
    repo = repo or build_repository(settings)
    ctx = ToolContext(repo=repo, payments=payments or build_payments(repo, settings), settings=settings)
    graph = build_graph(ctx, model or get_chat_model(settings), checkpointer=checkpointer, settings=settings)
    return graph, ctx
