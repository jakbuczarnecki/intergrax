# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Compose default RAG managers + profile for Tier-3 wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.contextual.chunk_enricher import ContextualChunkEnricher
from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.profiles.rag_profile import RagProfile, rag_profile_from_env
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import (
    create_default_reranker_engine,
    create_default_reranker_registry,
)
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.rerankers.re_ranker_manager import ReRankerManager
from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import create_vectorstore_manager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager


@dataclass(frozen=True)
class RagStack:
    profile: RagProfile
    vectorstore_manager: BaseVectorstoreManager
    embedding_manager: BaseEmbeddingManager
    retriever_manager: BaseRetrieverManager
    reranker_manager: BaseRerankerManager
    retrieval_service: RetrievalService
    contextual_enricher: Optional[ContextualChunkEnricher] = None
    graph_store: Optional[GraphStore] = None


def create_default_rag_stack(
    *,
    integration_profile: Optional[IntegrationProfile] = None,
    vectorstore_manager: Optional[BaseVectorstoreManager] = None,
    embedding_manager: Optional[BaseEmbeddingManager] = None,
    profile: Optional[RagProfile] = None,
    llm_for_contextual: Optional[LLMAdapter] = None,
    graph_store: Optional[GraphStore] = None,
) -> RagStack:
    profile = profile or rag_profile_from_env()
    if vectorstore_manager is None:
        vectorstore_manager = create_vectorstore_manager(profile=integration_profile)
    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    if graph_store is None and profile.graph_rag_enabled:
        graph_store = create_rag_graph_store(profile=profile)

    retriever_manager = create_default_retriever_manager(
        vector_store=vectorstore_manager,
        embedding_manager=embedding_manager,
        graph_store=graph_store,
        profile=profile,
        llm_for_query_expansion=llm_for_contextual,
    )
    registry = create_default_reranker_registry(
        embedding_manager=embedding_manager,
        integration_profile=integration_profile,
    )
    reranker_manager = ReRankerManager(
        engine=create_default_reranker_engine(
            embedding_manager=embedding_manager,
            registry=registry,
        )
    )

    contextual: Optional[ContextualChunkEnricher] = None
    if profile.contextual_enrich == "on" and llm_for_contextual is not None:
        contextual = ContextualChunkEnricher(llm_for_contextual)

    retrieval_service = RetrievalService(
        retriever_manager=retriever_manager,
        reranker_manager=reranker_manager if profile.enable_rerank else None,
        profile=profile,
        llm_for_agentic=llm_for_contextual,
    )

    return RagStack(
        profile=profile,
        vectorstore_manager=vectorstore_manager,
        embedding_manager=embedding_manager,
        retriever_manager=retriever_manager,
        reranker_manager=reranker_manager,
        retrieval_service=retrieval_service,
        contextual_enricher=contextual,
        graph_store=graph_store,
    )
