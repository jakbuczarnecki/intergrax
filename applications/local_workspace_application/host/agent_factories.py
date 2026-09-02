# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding
from local_indexer.local_indexer_agent import LocalIndexerAgent
from local_search.local_search_agent import LocalSearchAgent
from local_synthesizer.local_synthesizer_agent import LocalSynthesizerAgent
from tool_selection_qualifier.tool_selection_qualifier_agent import ToolSelectionQualifierAgent
from web_search_qualifier.web_search_qualifier_agent import WebSearchQualifierAgent
from model_routing_qualifier.model_routing_qualifier_agent import ModelRoutingQualifierAgent
from model_routing_qualifier.model_routing import (
    build_invoke_fail_profile,
    build_profile_a,
    build_profile_b,
)
from model_routing_qualifier.routing_profile import build_q4_qualification_routing_profile
from local_workspace_application.host.agent_builders import LOCAL_WORKSPACE_AGENT_BUILDERS


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


def build_local_workspace_web_search_qualifier_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> WebSearchQualifierAgent:
    from intergrax.applications._shared.llm_resolver import resolve_llm_adapter

    adapter = resolve_llm_adapter(ctx.environment)
    _ = binding
    return WebSearchQualifierAgent(llm_adapter=adapter)


def build_local_workspace_model_routing_qualifier_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> ModelRoutingQualifierAgent:
    from intergrax.applications._shared.llm_resolver import resolve_llm_adapter

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
    adapter = resolve_llm_adapter(env)
    _ = binding
    return ModelRoutingQualifierAgent(llm_adapter=adapter, routing_profile=env.llm_routing_profile)
