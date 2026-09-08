# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment profile for research_application."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from research_application.host.settings import ResearchBackendSettings
from research_application.host.skill_wiring import (
    RESEARCH_BUNDLE_ID,
    build_research_skill_profile,
)


class ResearchHostConfigurationError(RuntimeError):
    """Research host composition rejected incomplete orchestration configuration."""


def resolve_research_llm_profile(
    settings: ResearchBackendSettings,
) -> LLMProfile | None:
    """Declarative Research-owned LLM selection (no adapter materialization)."""
    provider = settings.llm_provider
    if provider is not None and provider.strip():
        return LLMProfile(provider=provider.strip(), model=settings.llm_model)
    research_env_profile = llm_profile_from_env(prefix="RESEARCH_LLM")
    if research_env_profile is not None:
        return research_env_profile
    return llm_profile_from_env(prefix="INTERGRAX_LLM")


def require_research_orchestration_llm_profile(
    settings: ResearchBackendSettings,
    profile: ApplicationEnvironmentProfile,
) -> None:
    """Fail closed before runtime assembly when engine orchestration lacks LLM selection."""
    if not settings.use_nexus_loop:
        return
    if profile.llm_profile is None:
        raise ResearchHostConfigurationError(
            "Research host requires explicit LLM selection for Nexus orchestration "
            "(planner_kind=engine): set RESEARCH_LLM_PROVIDER/RESEARCH_LLM_MODEL "
            "or INTERGRAX_LLM_PROVIDER/INTERGRAX_LLM_MODEL."
        )


def build_research_environment_profile(
    settings: ResearchBackendSettings | None = None,
) -> ApplicationEnvironmentProfile:
    """Product environment for research host (DX-1.4)."""
    settings = settings or ResearchBackendSettings.from_env()
    enabled_tools = list(settings.enabled_tool_ids)
    for tool_id in RESEARCH_LITERATURE_SCAN.tool_ids:
        if tool_id not in enabled_tools:
            enabled_tools.append(tool_id)
    llm_profile = resolve_research_llm_profile(settings)
    return (
        ApplicationEnvironmentProfile.product_defaults(
            profile_id="research.product",
            skill_bundles=[RESEARCH_BUNDLE_ID],
            tool_ids=enabled_tools,
        )
        .model_copy(
            update={
                "integration_profile": IntegrationProfile.research_product(),
                "skill_profile": build_research_skill_profile(),
                "llm_profile": llm_profile,
            },
        )
        .with_harness_memory()
        .with_reference_host_platform_defaults()
    )
