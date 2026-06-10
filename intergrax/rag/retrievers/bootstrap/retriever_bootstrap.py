# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.query.query_expander import query_expander_from_profile
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.retrievers.engine.retriever_engine import RetrieverEngine
from intergrax.rag.retrievers.providers.fusion_retriever import FusionRetriever
from intergrax.rag.retrievers.providers.hierarchical_retriever import HierarchicalRetriever
from intergrax.rag.retrievers.providers.hybrid_retriever import HybridRetriever
from intergrax.rag.retrievers.providers.mmr_retriever import MMRRetriever
from intergrax.rag.retrievers.providers.multiquery_retriever import MultiQueryRetriever
from intergrax.rag.retrievers.providers.parent_child_retriever import ParentChildRetriever
from intergrax.rag.retrievers.providers.vector_similarity_retriever import VectorSimilarityRetriever
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.retrievers.retriever_manager import RetrieverManager
from intergrax.rag.retrievers.pipeline.retriever_pipeline import RetrieverPipeline
from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.retrievers.providers.graph_rag_retriever import GraphRagRetriever
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import create_default_vectorstore_manager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager


def create_default_retriever_registry(
    *,
    vector_store: BaseVectorstoreManager | None = None,
    embedding_manager: BaseEmbeddingManager | None = None,
    registry: RetrieverRegistry | None = None,    
    toc_vector_store: BaseVectorstoreManager | None = None,
    graph_store: GraphStore | None = None,
    profile: RagProfile | None = None,
    llm_for_query_expansion: LLMAdapter | None = None,
) -> RetrieverRegistry:
    """
    Create RetrieverRegistry with built-in retriever providers registered.

    Allows dependency override by providing a custom registry.
    """

    if vector_store is None:
        vector_store = create_default_vectorstore_manager()

    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    if registry is None:
        registry = RetrieverRegistry()

        registry.register(
            VectorSimilarityRetriever(
                vector_store=vector_store,
                embedding_manager=embedding_manager,
            )
        )
        registry.register(
            MMRRetriever(
                vector_store=vector_store,
                embedding_manager=embedding_manager,
            )
        )
        registry.register(
            ParentChildRetriever(
                vector_store=vector_store,
                embedding_manager=embedding_manager,
            )
        )
        registry.register(
            HybridRetriever(
                vector_store=vector_store,
                embedding_manager=embedding_manager,
            )
        )
        query_expander = None
        if profile is not None:
            query_expander = query_expander_from_profile(
                mode=profile.query_expansion,
                llm=llm_for_query_expansion,
            )
        registry.register(
            MultiQueryRetriever(
                vector_store=vector_store,
                embedding_manager=embedding_manager,
                query_expander=query_expander,
            )
        )
        registry.register(
            HierarchicalRetriever(
                chunks_store=vector_store,
                embedding_manager=embedding_manager,
                toc_store=toc_vector_store,
            )
        )
        registry.register(
            FusionRetriever(
                registry=registry,
                retrievers=[
                    VectorSimilarityRetriever.name(),
                    HybridRetriever.name(),
                    ParentChildRetriever.name(),
                ],
            )
        )
        if graph_store is not None:
            registry.register(
                GraphRagRetriever(
                    vector_store=vector_store,
                    embedding_manager=embedding_manager,
                    graph_store=graph_store,
                )
            )

    return registry


def create_default_retriever_engine(
    *,
    vector_store: BaseVectorstoreManager | None = None,
    embedding_manager: BaseEmbeddingManager | None = None,
    registry: RetrieverRegistry | None = None,
    graph_store: GraphStore | None = None,
    profile: RagProfile | None = None,
    llm_for_query_expansion: LLMAdapter | None = None,
    toc_vector_store: BaseVectorstoreManager | None = None,
) -> RetrieverEngine:
    """
    Create RetrieverEngine with default retriever providers registered.
    """

    if vector_store is None:
        vector_store = create_default_vectorstore_manager()

    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    if registry is None:
        registry = create_default_retriever_registry(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            graph_store=graph_store,
            profile=profile,
            llm_for_query_expansion=llm_for_query_expansion,
            toc_vector_store=toc_vector_store,
        )

    return RetrieverEngine(
        registry=registry,
    )


def create_default_retriever_pipeline(
    *,
    vector_store: BaseVectorstoreManager | None = None,
    embedding_manager: BaseEmbeddingManager | None = None,
    registry: RetrieverRegistry | None = None,
    graph_store: GraphStore | None = None,
    profile: RagProfile | None = None,
    llm_for_query_expansion: LLMAdapter | None = None,
    toc_vector_store: BaseVectorstoreManager | None = None,
) -> RetrieverPipeline:
    """
    Create RetrieverPipeline using the default retriever engine.
    """

    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    if registry is None:
        registry = create_default_retriever_registry(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            graph_store=graph_store,
            profile=profile,
            llm_for_query_expansion=llm_for_query_expansion,
            toc_vector_store=toc_vector_store,
        )

    engine = create_default_retriever_engine(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        registry=registry,
        graph_store=graph_store,
        profile=profile,
        llm_for_query_expansion=llm_for_query_expansion,
        toc_vector_store=toc_vector_store,
    )

    return RetrieverPipeline(
        engine=engine,        
        embedding_manager=embedding_manager
    )


def create_default_retriever_manager(
    *,
    vector_store: BaseVectorstoreManager | None = None,
    embedding_manager: BaseEmbeddingManager | None = None,
    registry: RetrieverRegistry | None = None,
    graph_store: GraphStore | None = None,
    profile: RagProfile | None = None,
    llm_for_query_expansion: LLMAdapter | None = None,
    toc_vector_store: BaseVectorstoreManager | None = None,
) -> BaseRetrieverManager:
    """
    Create RetrieverManager using the default retriever pipeline.
    """

    if vector_store is None:
        vector_store = create_default_vectorstore_manager()

    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    pipeline = create_default_retriever_pipeline(        
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        registry=registry,
        graph_store=graph_store,
        profile=profile,
        llm_for_query_expansion=llm_for_query_expansion,
        toc_vector_store=toc_vector_store,
    )

    return RetrieverManager(
        pipeline=pipeline,        
    )