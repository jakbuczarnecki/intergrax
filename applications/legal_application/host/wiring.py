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

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from legal_application.host.settings import LegalBackendSettings


def build_legal_manifest(settings: LegalBackendSettings) -> ApplicationManifest:
    """Legal manifest with runtime contract id from settings."""
    from legal_application.manifest import LEGAL_APPLICATION_MANIFEST

    base = LEGAL_APPLICATION_MANIFEST.agents[0]
    binding = base.model_copy(update={"contract_id": settings.legal_default_agent_id})
    return LEGAL_APPLICATION_MANIFEST.model_copy(update={"agents": [binding]})


def build_legal_environment_profile(settings: LegalBackendSettings) -> ApplicationEnvironmentProfile:
    """Product environment for legal host (H-APP.5.2)."""
    from intergrax.runtime.modality.modality_profile import ModalityProfile, ModalityPlane
    from legal_application.manifest import LEGAL_APPLICATION_MANIFEST

    tool_ids = list(settings.enabled_tool_ids)
    modality_profile = None
    if settings.enable_modality_tools:
        modality_profile = ModalityProfile(
            profile_id="legal.modality",
            allowed_planes={ModalityPlane.DEDICATED_INFERENCE},
        )
        for tool_id in (
            "vision.detect",
            "vision.segment",
            "vision.ocr_regions",
            "speech.synthesize",
            "speech.transcribe",
            "ml.predict",
            "ml.explain",
            "ml.batch_predict",
        ):
            if tool_id not in tool_ids:
                tool_ids.append(tool_id)
    return ApplicationEnvironmentProfile.product_defaults(
        profile_id="legal.product",
        skill_bundles=["legal"],
        tool_ids=tool_ids,
        domain_fragments={"legal.contract_review.policy": "legal.contract_review.policy"},
    ).model_copy(
        update={
            "integration_profile": LEGAL_APPLICATION_MANIFEST.integration_profile,
            "modality_profile": modality_profile,
        },
    )


def build_legal_registry(settings: LegalBackendSettings) -> AgentRegistry:
    """Materialize Legal agent via unified Tier-3 environment wiring."""
    manifest = build_legal_manifest(settings)
    env = manifest.environment or build_legal_environment_profile(settings)
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(manifest, env, settings=settings)
    return build_application_registry(manifest, env_wiring.build_context)


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
        policy_bundle=ctx.policy_bundle if ctx is not None else None,
    )
    return LegalAgent(config=cfg)
