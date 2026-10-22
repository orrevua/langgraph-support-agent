"""Chat-model factory: one function decides between Anthropic and the offline fake.

Everything else in the code base depends on `BaseChatModel`, never on a concrete provider, so
switching providers is a configuration change (`LLM_PROVIDER`, `LLM_MODEL`).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from supportagent.config import Settings, get_settings
from supportagent.llm.fake import ScriptedChatModel, generic_policy
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
    policies = []
    try:  # the support-agent policy is added in milestone 2
        from supportagent.llm.policy import support_policy

        policies.append(support_policy)
    except ImportError:  # pragma: no cover
        pass
    policies.append(generic_policy)
    return ScriptedChatModel(policies=policies, structured_handlers={"TicketTriage": heuristic_triage})


def describe_provider(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if settings.resolved_provider == "anthropic":
        return f"anthropic ({settings.llm_model})"
    return "offline scripted fake model"
