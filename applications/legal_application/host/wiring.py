# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Construct Tier-2 Legal Agent and dependencies from :class:`~legal_application.host.settings.LegalBackendSettings`."""

from __future__ import annotations

from legal.config.legal_agent_product_profiles import LegalAgentProductProfile
from legal.legal_agent import LegalAgent
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.session.session_storage import SessionStorage
from intergrax.integrations.providers.relational_store.sqlite import create_sqlite_session_storage

from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications._shared.skill_wiring import build_application_skill_wiring
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.skills.registry.profile import SkillProfile
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from legal_application.manifest import LEGAL_APPLICATION_MANIFEST

from legal_application.host.settings import LegalBackendSettings
from legal_application.host.tool_wiring import wire_legal_tools


def build_legal_manifest(settings: LegalBackendSettings) -> ApplicationManifest:
    """Legal manifest with runtime contract id from settings."""
    base = LEGAL_APPLICATION_MANIFEST.agents[0]
    binding = base.model_copy(update={"contract_id": settings.legal_default_agent_id})
    return LEGAL_APPLICATION_MANIFEST.model_copy(update={"agents": [binding]})


def build_legal_registry(settings: LegalBackendSettings) -> AgentRegistry:
    """Materialize Legal agent via unified Tier-3 wiring (factory + tool catalog)."""
    manifest = build_legal_manifest(settings)
    tool_wiring = wire_legal_tools(settings=settings)
    skill_wiring = build_application_skill_wiring(SkillProfile(enabled_bundles=["legal"]))
    tool_registry = tool_wiring.registry
    if not tool_wiring.profile.enabled and not tool_wiring.profile.enabled_bundles:
        tool_registry = None
    ctx = ApplicationBuildContext.for_manifest(
        manifest,
        settings=settings,
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
        skill_profile=skill_wiring.profile,
        skill_registry=skill_wiring.registry,
        tool_registry=tool_registry,
        policy_bundle=build_runtime_policy_bundle(
            domain_fragments={"legal.contract_review.policy": "legal.contract_review.policy"},
        ),
    )
    return build_application_registry(manifest, ctx)


def build_legal_agent(
    settings: LegalBackendSettings,
    *,
    ctx: ApplicationBuildContext | None = None,
) -> LegalAgent:
    """
    Session storage: SQLite when ``LEGAL_SESSION_SQLITE_PATH`` is set; otherwise in-memory (dev only).

    LLM from :class:`~intergrax.llm_adapters.registry.profile.LLMProfile` using
    ``LEGAL_LLM_PROVIDER`` and optional ``LEGAL_LLM_MODEL``.

    Tier-2 configuration from :class:`~legal.config.legal_agent_product_profiles.LegalAgentProductProfile`.
    Tool catalog from ``ctx.tool_profile`` / ``ctx.tool_wiring_context`` when provided.
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

    llm_profile = LLMProfile(
        provider=provider,
        model=settings.legal_llm_model,
        options={"max_retries": 2},
    )
    llm = llm_profile.create_adapter()

    storage: SessionStorage
    if settings.session_sqlite_path:
        storage = create_sqlite_session_storage(db_path=settings.session_sqlite_path)  # type: ignore[assignment]
    else:
        storage = InMemorySessionStorage()

    session_manager = SessionManager(storage=storage)

    tool_profile = ctx.tool_profile if ctx is not None else None
    tool_wiring_context = ctx.tool_wiring_context if ctx is not None else None
    if tool_profile is None and settings.enabled_tool_ids:
        tool_wiring = wire_legal_tools(settings=settings)
        tool_profile = tool_wiring.profile
        tool_wiring_context = tool_wiring.wiring_context

    has_rag_tools = bool(tool_profile and tool_profile.is_tool_enabled("rag.retrieve"))
    has_web_tools = bool(tool_profile and tool_profile.is_tool_enabled("websearch.query"))

    cfg = profile.make_config(
        session_manager=session_manager,
        llm_adapter=llm,
        production_mode=False,
        enable_rag=settings.enable_rag and has_rag_tools,
        enable_websearch=settings.enable_websearch and has_web_tools,
        use_legal_tool_decision=settings.use_legal_tool_decision,
        tools_mode=settings.tools_mode,  # type: ignore[arg-type]
        tool_profile=tool_profile,
        tool_wiring_context=tool_wiring_context,
    )
    return LegalAgent(config=cfg)
