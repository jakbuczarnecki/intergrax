# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
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
from intergrax.rag.retrievers.retriever_pipeline import RetrieverPipeline
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager


def create_default_retriever_registry(
    *,
    vector_store: BaseVectorstoreManager,
    embedding_manager: BaseEmbeddingManager,
    registry: RetrieverRegistry | None = None,    
    toc_vector_store: BaseVectorstoreManager | None = None,
) -> RetrieverRegistry:
    """
    Create RetrieverRegistry with built-in retriever providers registered.

    Allows dependency override by providing a custom registry.
    """

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
        registry.register(
            MultiQueryRetriever(
                vector_store=vector_store,
                embedding_manager=embedding_manager,
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

    return registry


def create_default_retriever_engine(
    *,
    vector_store: BaseVectorstoreManager,
    embedding_manager: BaseEmbeddingManager,
    registry: RetrieverRegistry | None = None,
) -> RetrieverEngine:
    """
    Create RetrieverEngine with default retriever providers registered.
    """

    if registry is None:
        registry = create_default_retriever_registry(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
        )

    return RetrieverEngine(
        registry=registry,
    )


def create_default_retriever_pipeline(
    *,
    vector_store: BaseVectorstoreManager,
    embedding_manager: BaseEmbeddingManager,
    registry: RetrieverRegistry | None = None,
) -> RetrieverPipeline:
    """
    Create RetrieverPipeline using the default retriever engine.
    """

    if registry is None:
        registry = create_default_retriever_registry(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
        )

    engine = create_default_retriever_engine(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        registry=registry,
    )

    return RetrieverPipeline(
        engine=engine,        
        embedding_manager=embedding_manager
    )


def create_default_retriever_manager(
    *,
    vector_store: BaseVectorstoreManager,
    embedding_manager: BaseEmbeddingManager,
    registry: RetrieverRegistry | None = None,
) -> BaseRetrieverManager:
    """
    Create RetrieverManager using the default retriever pipeline.
    """

    pipeline = create_default_retriever_pipeline(        
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        registry=registry,
    )

    return RetrieverManager(
        pipeline=pipeline,        
    )