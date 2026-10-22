"""Application settings, loaded from environment variables (and an optional `.env`).

pydantic-settings maps each field to an env var of the same name (case-insensitive),
so `LLM_PROVIDER=fake` sets `Settings.llm_provider`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM -------------------------------------------------------------------------
    # "auto" picks Anthropic when ANTHROPIC_API_KEY is set, otherwise the offline fake.
    llm_provider: Literal["auto", "anthropic", "fake"] = "auto"
    llm_model: str = "claude-sonnet-5-5"
    llm_fast_model: str = "claude-haiku-5-5"
    anthropic_api_key: str | None = None

    # --- Storage ---------------------------------------------------------------------
    database_url: str = "postgresql://postgres:postgres@localhost:5432/p2_agent"
    # "postgres" for the API/demo, "memory" for tests and quick experiments.
    storage: Literal["postgres", "memory"] = "postgres"

    # --- Agent limits / guardrails ----------------------------------------------------
    max_agent_steps: int = Field(8, description="Model calls allowed per user turn.")
    recursion_limit: int = Field(30, description="LangGraph super-step limit per run.")
    max_message_chars: int = 2000
    # Refunds above this (in BRL cents) are refused outright by the tool, approval or not.
    refund_hard_limit_cents: int = 200_000  # R$ 2.000,00

    # --- Mock payments API ------------------------------------------------------------
    payments_mode: Literal["inprocess", "http"] = "inprocess"
    payments_base_url: str = "http://localhost:8000/mock-payments"
    payments_latency_ms: int = 0
    payments_fail_first: int = 0  # simulate N transient failures, then succeed
    payments_failure_rate: float = 0.0  # random transient failures (seeded)
    payments_retry_attempts: int = 3
    payments_retry_base_delay_s: float = 0.05

    @property
    def resolved_provider(self) -> Literal["anthropic", "fake"]:
        if self.llm_provider == "auto":
            return "anthropic" if self.anthropic_api_key else "fake"
        return self.llm_provider


@lru_cache
def get_settings() -> Settings:
    return Settings()
