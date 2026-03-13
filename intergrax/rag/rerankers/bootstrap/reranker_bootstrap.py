# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager

from intergrax.rag.rerankers.cache.base_rerank_cache import BaseRerankCache
from intergrax.rag.rerankers.cache.rerank_cache import RerankCache
from intergrax.rag.rerankers.providers.cohere_reranker import CohereReranker
from intergrax.rag.rerankers.providers.cross_encoder_reranker import CrossEncoderReranker
from intergrax.rag.rerankers.providers.jina_reranker import JinaReranker
from intergrax.rag.rerankers.providers.semantic_reranker import SemanticReranker
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry
from intergrax.rag.rerankers.engine.reranker_engine import RerankerEngine
from intergrax.rag.rerankers.pipeline.reranker_pipeline import RerankerPipeline

from intergrax.rag.rerankers.providers.embedding_cosine_reranker import (
    EmbeddingCosineReranker,
)


def create_default_reranker_registry(
    *,
    embedding_manager: BaseEmbeddingManager | None = None,
    registry: RerankerRegistry | None = None,
) -> RerankerRegistry:
    """
    Create RerankerRegistry with built-in reranker providers registered.

    Allows dependency override by providing a custom registry.
    """

    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    if registry is None:

        registry = RerankerRegistry()

        registry.register(
            EmbeddingCosineReranker(
                embedding_manager=embedding_manager,
            )
        )

        registry.register(
            CrossEncoderReranker()
        )

        registry.register(
            SemanticReranker()
        )

        registry.register(
            CohereReranker()
        )

        registry.register(
            JinaReranker()
        )

    return registry


def create_default_reranker_engine(
    *,
    embedding_manager: BaseEmbeddingManager | None = None,
    registry: RerankerRegistry | None = None,
    cache: BaseRerankCache | None = None
) -> RerankerEngine:
    """
    Create RerankerEngine with default reranker providers registered.
    """

    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    if registry is None:
        registry = create_default_reranker_registry(
            embedding_manager=embedding_manager,
        )

    if cache is None:
        cache = RerankCache()

    return RerankerEngine(
        registry=registry,
        cache=cache,
    )


def create_default_reranker_pipeline(
    *,
    embedding_manager: BaseEmbeddingManager | None = None,
    registry: RerankerRegistry | None = None,
) -> RerankerPipeline:
    """
    Create RerankerPipeline using the default reranker engine.
    """

    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    if registry is None:

        registry = create_default_reranker_registry(
            embedding_manager=embedding_manager,
        )

    engine = create_default_reranker_engine(
        embedding_manager=embedding_manager,
        registry=registry,
    )

    return RerankerPipeline(
        engine=engine,
    )