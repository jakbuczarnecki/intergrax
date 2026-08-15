# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.core.plugins.discovery import EP_RAG_RERANKERS, register_plugins
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager

from intergrax.rag.rerankers.cache.base_rerank_cache import BaseRerankCache
from intergrax.rag.rerankers.cache.rerank_cache import RerankCache
from intergrax.rag.rerankers.contracts.base_reranker import (
    BaseReranker,
    BaseRerankerPlugin,
)
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.rerankers.providers.cohere_reranker import CohereReranker
from intergrax.rag.rerankers.providers.cross_encoder_reranker import CrossEncoderReranker
from intergrax.rag.rerankers.providers.jina_reranker import JinaReranker
from intergrax.rag.rerankers.providers.semantic_reranker import SemanticReranker
from intergrax.rag.rerankers.re_ranker_manager import ReRankerManager
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry
from intergrax.rag.rerankers.engine.reranker_engine import RerankerEngine
from intergrax.rag.rerankers.pipeline.reranker_pipeline import RerankerPipeline

from intergrax.rag.rerankers.providers.embedding_cosine_reranker import (
    EmbeddingCosineReranker,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.profile import IntegrationProfile


def _register_entry_point_rerankers(
    registry: RerankerRegistry,
    *,
    embedding_manager: BaseEmbeddingManager,
    discover_entry_points: bool | None,
) -> None:
    if discover_entry_points is None:
        discover_entry_points = discover_plugins_enabled()

    def _register_entry_point(plugin_type: type) -> None:
        if issubclass(plugin_type, BaseRerankerPlugin):
            reranker = plugin_type.create(embedding_manager=embedding_manager)
        elif issubclass(plugin_type, BaseReranker):
            reranker = plugin_type()
        else:
            raise TypeError(
                "RAG reranker plugin must subclass BaseReranker or "
                f"BaseRerankerPlugin: {plugin_type!r}"
            )
        if not isinstance(reranker, BaseReranker):
            raise TypeError(
                f"RAG reranker plugin factory must return BaseReranker: {plugin_type!r}"
            )
        registry.register(reranker)

    register_plugins(
        EP_RAG_RERANKERS,
        _register_entry_point,
        discover_entry_points=discover_entry_points,
    )


def create_default_reranker_registry(
    *,
    embedding_manager: BaseEmbeddingManager | None = None,
    registry: RerankerRegistry | None = None,
    integration_profile: IntegrationProfile | None = None,
    discover_entry_points: bool | None = None,
) -> RerankerRegistry:
    """
    Create RerankerRegistry with built-in reranker providers registered.

    Allows dependency override by providing a custom registry.
    """

    if discover_entry_points is None:
        discover_entry_points = discover_plugins_enabled()

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

        preferred = None
        if integration_profile is not None:
            preferred = integration_profile.slug_for_category(IntegrationCategory.RERANK_PROVIDER)

        if preferred in (None, "cohere_rerank"):
            registry.register(CohereReranker())
        if preferred in (None, "jina_rerank"):
            registry.register(JinaReranker())

    _register_entry_point_rerankers(
        registry,
        embedding_manager=embedding_manager,
        discover_entry_points=discover_entry_points,
    )

    return registry


def create_default_reranker_engine(
    *,
    embedding_manager: BaseEmbeddingManager | None = None,
    registry: RerankerRegistry | None = None,
    cache: BaseRerankCache | None = None,
    discover_entry_points: bool | None = None,
) -> RerankerEngine:
    """
    Create RerankerEngine with default reranker providers registered.
    """

    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    if registry is None:
        registry = create_default_reranker_registry(
            embedding_manager=embedding_manager,
            discover_entry_points=False,
        )

    _register_entry_point_rerankers(
        registry,
        embedding_manager=embedding_manager,
        discover_entry_points=discover_entry_points,
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
    discover_entry_points: bool | None = None,
) -> RerankerPipeline:
    """
    Create RerankerPipeline using the default reranker engine.
    """

    if embedding_manager is None:
        embedding_manager = create_default_embedding_manager()

    if registry is None:

        registry = create_default_reranker_registry(
            embedding_manager=embedding_manager,
            discover_entry_points=False,
        )

    engine = create_default_reranker_engine(
        embedding_manager=embedding_manager,
        registry=registry,
        discover_entry_points=discover_entry_points,
    )

    return RerankerPipeline(
        engine=engine,
    )


def create_default_reranker_manager(
    *,
    embedding_manager: BaseEmbeddingManager | None = None,
    engine: RerankerEngine | None = None,
    discover_entry_points: bool | None = None,
) -> BaseRerankerManager:
    if engine is None:
        engine = create_default_reranker_engine(
            embedding_manager=embedding_manager,
            discover_entry_points=discover_entry_points,
        )

    return ReRankerManager(
        engine=engine,
    )