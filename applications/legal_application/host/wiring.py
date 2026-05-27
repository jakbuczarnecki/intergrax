# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Construct Tier-2 Legal Agent and dependencies from :class:`~legal_application.host.settings.LegalBackendSettings`."""

from __future__ import annotations

from legal.config.legal_agent_product_profiles import LegalAgentProductProfile
from legal.legal_agent import LegalAgent
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.session.session_storage import SessionStorage
from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage

from legal_application.host.settings import LegalBackendSettings


def build_legal_agent(settings: LegalBackendSettings) -> LegalAgent:
    """
    Session storage: SQLite when ``LEGAL_SESSION_SQLITE_PATH`` is set; otherwise in-memory (dev only).

    LLM provider from ``LEGAL_LLM_PROVIDER`` (must be a :class:`~intergrax.llm_adapters.contracts.llm_provider.LLMProvider` value).

    Tier-2 configuration from :class:`~legal.config.legal_agent_product_profiles.LegalAgentProductProfile`.
    """
    try:
        profile = LegalAgentProductProfile(settings.legal_product_profile)
    except ValueError as exc:
        raise ValueError(
            f"Invalid LEGAL_PRODUCT_PROFILE={settings.legal_product_profile!r}. "
            f"Choose one of: {[p.value for p in LegalAgentProductProfile]}."
        ) from exc

    try:
        provider = LLMProvider(settings.legal_llm_provider)
    except ValueError as exc:
        raise ValueError(
            f"Invalid LEGAL_LLM_PROVIDER={settings.legal_llm_provider!r}. "
            f"Choose one of: {[p.value for p in LLMProvider]}."
        ) from exc

    llm = LLMAdapterRegistry.create(provider)

    storage: SessionStorage
    if settings.session_sqlite_path:
        storage = SQLiteSessionStorage(settings.session_sqlite_path)
    else:
        storage = InMemorySessionStorage()

    session_manager = SessionManager(storage=storage)

    cfg = profile.make_config(
        session_manager=session_manager,
        llm_adapter=llm,
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
    )
    return LegalAgent(config=cfg)
