# © Artur Czarnecki. All rights reserved.

"""Copy RAG managers from tool wiring into runtime config (Phase RAG-1)."""

from __future__ import annotations

from intergrax.rag.profiles.runtime_rag_sync import sync_rag_profile_from_runtime_config
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.tools.registry.wiring import ToolWiringContext


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
