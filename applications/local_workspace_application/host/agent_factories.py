# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.integrations.contracts.integration_profile import IntegrationProfile

from local_indexer.local_indexer_agent import LocalIndexerAgent
from local_search.local_search_agent import LocalSearchAgent
from local_synthesizer.local_synthesizer_agent import LocalSynthesizerAgent
from local_workspace_application.host.agent_builders import LOCAL_WORKSPACE_AGENT_BUILDERS
from model_routing_qualifier.model_routing import (
    build_invoke_fail_profile,
    build_profile_a,
    build_profile_b,
)
from model_routing_qualifier.model_routing_qualifier_agent import ModelRoutingQualifierAgent
from model_routing_qualifier.routing_profile import build_q4_qualification_routing_profile
from tool_selection_qualifier.tool_selection_qualifier_agent import ToolSelectionQualifierAgent
from web_search_qualifier.steps.web_search_job import ResolvedSearchProvider
from web_search_qualifier.web_search_qualifier_agent import WebSearchQualifierAgent


def build_local_workspace_local_indexer_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> LocalIndexerAgent:
    _ = ctx, binding
    factory = LOCAL_WORKSPACE_AGENT_BUILDERS.get(LocalIndexerAgent)
    if factory is None:
        raise ValueError(f"No builder registered for {binding.import_path!r}")
    return factory(ctx, binding)


def build_local_workspace_local_search_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> LocalSearchAgent:
    _ = ctx, binding
    factory = LOCAL_WORKSPACE_AGENT_BUILDERS.get(LocalSearchAgent)
    if factory is None:
        raise ValueError(f"No builder registered for {binding.import_path!r}")
    return factory(ctx, binding)


def build_local_workspace_local_synthesizer_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> LocalSynthesizerAgent:
    _ = ctx, binding
    factory = LOCAL_WORKSPACE_AGENT_BUILDERS.get(LocalSynthesizerAgent)
    if factory is None:
        raise ValueError(f"No builder registered for {binding.import_path!r}")
    return factory(ctx, binding)


def build_local_workspace_tool_selection_qualifier_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> ToolSelectionQualifierAgent:
    from intergrax.applications._shared.llm_resolver import resolve_llm_adapter

    adapter = resolve_llm_adapter(ctx.environment)
    _ = binding
    return ToolSelectionQualifierAgent(llm_adapter=adapter)


def local_workspace_profile_has_search_provider(profile: IntegrationProfile) -> bool:
    """Single source of truth: the profile carries a ``search_provider`` slot (roster enablement and builder)."""
    return profile.search_provider is not None


def _resolve_search_provider_from_profile(ctx: ApplicationBuildContext) -> ResolvedSearchProvider:
    """Resolve the web-search provider from the ``IntegrationProfile`` slot (no ambient credentials)."""
    from intergrax.integrations.contracts.base import (
        IntegrationCategory,
        IntegrationConfigurationError,
    )
    from intergrax.integrations.registry.factory import resolve_from_profile

    if ctx.environment is None:
        raise IntegrationConfigurationError(
            "web_search_qualifier requires an application environment with an IntegrationProfile",
        )
    profile = ctx.environment.integration_profile
    if not local_workspace_profile_has_search_provider(profile):
        raise IntegrationConfigurationError(
            "web_search_qualifier requires an IntegrationProfile search_provider slot; "
            "ambient credentials never activate a provider",
        )
    provider = resolve_from_profile(profile, IntegrationCategory.SEARCH_PROVIDER)
    provider_id = (
        profile.slug_for_category(IntegrationCategory.SEARCH_PROVIDER) or IntegrationCategory.SEARCH_PROVIDER.value
    )
    return ResolvedSearchProvider(provider_id=provider_id, provider=provider)


def build_local_workspace_web_search_qualifier_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> WebSearchQualifierAgent:
    from intergrax.applications._shared.llm_resolver import resolve_llm_adapter

    adapter = resolve_llm_adapter(ctx.environment)
    _ = binding
    return WebSearchQualifierAgent(
        llm_adapter=adapter,
        search_provider=_resolve_search_provider_from_profile(ctx),
    )


def build_local_workspace_model_routing_qualifier_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> ModelRoutingQualifierAgent:
    profile_a = build_profile_a()
    profile_b = build_profile_b()
    invoke_fail_profile = build_invoke_fail_profile()
    env = ctx.environment.model_copy(
        update={
            "llm_profile": profile_a,
            "llm_routing_profile": build_q4_qualification_routing_profile(
                profile_a=profile_a,
                profile_b=profile_b,
                invoke_fail_profile=invoke_fail_profile,
            ),
        },
    )
    from intergrax.applications._shared.llm_resolver import resolve_llm_adapter

    adapter = resolve_llm_adapter(env)
    _ = binding
    return ModelRoutingQualifierAgent(
        llm_adapter=adapter,
        routing_profile=env.llm_routing_profile,
    )
