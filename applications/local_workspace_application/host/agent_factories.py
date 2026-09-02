# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding
from local_indexer.local_indexer_agent import LocalIndexerAgent
from local_search.local_search_agent import LocalSearchAgent
from local_synthesizer.local_synthesizer_agent import LocalSynthesizerAgent
from tool_selection_qualifier.tool_selection_qualifier_agent import ToolSelectionQualifierAgent
from web_search_qualifier.web_search_qualifier_agent import WebSearchQualifierAgent
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
