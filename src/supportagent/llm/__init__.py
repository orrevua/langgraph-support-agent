"""Chat-model factory: one function decides between Anthropic and the offline fake.

Everything else in the code base depends on `BaseChatModel`, never on a concrete provider, so
switching providers is a configuration change (`LLM_PROVIDER`, `LLM_MODEL`).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from supportagent.config import Settings, get_settings
from supportagent.llm.fake import ScriptedChatModel, generic_policy
from supportagent.llm.policy import support_policy
from supportagent.triage import heuristic_triage


def get_chat_model(settings: Settings | None = None, *, fast: bool = False) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.resolved_provider == "anthropic":
        # Imported lazily so the offline path never needs the Anthropic SDK configured.
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_fast_model if fast else settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=1024,
            max_retries=3,
        )
    return build_fake_model()


def build_fake_model() -> ScriptedChatModel:
    # The support policy handles the agent; the generic one handles everything else
    # (e.g. the tool-calling demo in examples/01_fundamentals.py).
    return ScriptedChatModel(
        policies=[support_policy, generic_policy],
        structured_handlers={"TicketTriage": heuristic_triage},
    )


def describe_provider(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if settings.resolved_provider == "anthropic":
        return f"anthropic ({settings.llm_model})"
    return "offline scripted fake model"
