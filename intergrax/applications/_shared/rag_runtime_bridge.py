# © Artur Czarnecki. All rights reserved.

"""Map RAG stack / tool wiring artifacts to RuntimeConfig (Phase RAG-1)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack, create_default_rag_stack
from intergrax.rag.profiles.rag_profile import RagProfile, production_rag_profile
from intergrax.rag.profiles.runtime_rag_sync import sync_rag_profile_from_runtime_config
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.tools.registry.wiring import ToolWiringContext


def apply_rag_stack_to_runtime_config(
    config: RuntimeConfig,
    stack: RagStack,
) -> RuntimeConfig:
    """Attach composed RAG managers and profile to runtime config."""
    config.vectorstore_manager = stack.vectorstore_manager
    config.embedding_manager = stack.embedding_manager
    config.retriever_manager = stack.retriever_manager
    config.reranker_manager = stack.reranker_manager
    config.retrieval_service = stack.retrieval_service
    config.rag_profile = stack.profile
    sync_rag_profile_from_runtime_config(config, base=stack.profile)
    return config


def apply_rag_from_tool_wiring_context(
    config: RuntimeConfig,
    wiring_context: ToolWiringContext,
) -> RuntimeConfig:
    """Copy RAG managers from ``ToolWiringContext`` when present."""
    if wiring_context.vectorstore_manager is not None:
        config.vectorstore_manager = wiring_context.vectorstore_manager
    if wiring_context.embedding_manager is not None:
        config.embedding_manager = wiring_context.embedding_manager
    if wiring_context.retriever_manager is not None:
        config.retriever_manager = wiring_context.retriever_manager
    if wiring_context.reranker_manager is not None:
        config.reranker_manager = wiring_context.reranker_manager
    if wiring_context.retrieval_service is not None:
        config.retrieval_service = wiring_context.retrieval_service
    if wiring_context.rag_profile is not None:
        config.rag_profile = wiring_context.rag_profile
    if config.rag_profile is not None:
        sync_rag_profile_from_runtime_config(config, base=config.rag_profile)
    return config


def resolve_rag_profile_for_environment(
    env: ApplicationEnvironmentProfile,
    *,
    base: RagProfile | None = None,
) -> RagProfile | None:
    """Apply production Graph RAG defaults when RAG is enabled on product hosts."""
    if not env.context_profile.enable_rag:
        return None
    profile = base or production_rag_profile()
    if env.application_profile is ApplicationProfile.PRODUCT:
        prod = production_rag_profile()
        return profile.model_copy(
            update={
                "graph_rag_enabled": prod.graph_rag_enabled,
                "graph_rag_hops": prod.graph_rag_hops,
                "graph_indexer_mode": prod.graph_indexer_mode,
                "graph_store_backend": prod.graph_store_backend,
            }
        )
    return profile


def resolve_rag_stack_for_environment(
    env: ApplicationEnvironmentProfile,
    *,
    integration_profile: IntegrationProfile | None = None,
    llm_adapter: LLMAdapter | None = None,
) -> RagStack | None:
    """Build default RAG stack when context profile enables RAG."""
    if not env.context_profile.enable_rag:
        return None
    profile = integration_profile or env.integration_profile
    rag_profile = resolve_rag_profile_for_environment(env)
    return create_default_rag_stack(
        integration_profile=profile,
        llm_for_contextual=llm_adapter,
        profile=rag_profile,
    )


def apply_rag_for_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
    *,
    tool_wiring_context: ToolWiringContext | None = None,
) -> RuntimeConfig:
    """
    Apply RAG managers to ``RuntimeConfig`` when RAG is enabled.

    Prefers wired ``ToolWiringContext`` managers; otherwise composes a default stack.
    """
    if not config.enable_rag:
        return config
    if tool_wiring_context is not None and tool_wiring_context.vectorstore_manager is not None:
        return apply_rag_from_tool_wiring_context(config, tool_wiring_context)
    stack = resolve_rag_stack_for_environment(
        env,
        integration_profile=config.integration_profile,
        llm_adapter=config.llm_adapter,
    )
    if stack is None:
        return config
    return apply_rag_stack_to_runtime_config(config, stack)
