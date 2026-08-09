# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.core.plugins.discovery import EP_RAG_CHUNKERS, register_plugins
from intergrax.rag.document_splitters.contracts.base_documents_splitter import BaseDocumentsSplitter
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.documents_splitter import DocumentsSplitter
from intergrax.rag.document_splitters.engine.chunking_engine import ChunkingEngine
from intergrax.rag.document_splitters.registry.plugin_registry import apply_chunking_strategy_plugins
from intergrax.rag.document_splitters.registry.strategy_registry import (
    ChunkingStrategyRegistry,
)

from intergrax.rag.document_splitters.strategies.recursive_chunking_strategy import (
    RecursiveChunkingStrategy,
)
from intergrax.rag.document_splitters.strategies.semantic_chunking_strategy import (
    SemanticChunkingStrategy,
)
from intergrax.rag.document_splitters.strategies.parent_child_chunking_strategy import (
    ParentChildChunkingStrategy,
)
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager


def create_default_chunking_engine(
    registry: ChunkingStrategyRegistry | None = None,
    *,
    discover_entry_points: bool | None = None,
) -> ChunkingEngine:
    """
    Create a ChunkingEngine with default chunking strategies registered.

    Allows dependency override by providing a custom registry.
    """

    if discover_entry_points is None:
        discover_entry_points = discover_plugins_enabled()

    if registry is None:
        registry = ChunkingStrategyRegistry()

        registry.register(RecursiveChunkingStrategy())
        registry.register(SemanticChunkingStrategy(embedding_manager=create_default_embedding_manager()))
        registry.register(ParentChildChunkingStrategy())
        from intergrax.rag.document_splitters.strategies.docling_chunking_strategy import (
            DoclingChunkingStrategy,
        )

        registry.register(DoclingChunkingStrategy())

    apply_chunking_strategy_plugins(registry)

    def _register_entry_point(plugin_type: type) -> None:
        if not issubclass(plugin_type, BaseChunkingStrategy):
            raise TypeError(
                f"RAG chunker plugin must subclass BaseChunkingStrategy: {plugin_type!r}"
            )
        registry.register(plugin_type())

    register_plugins(
        EP_RAG_CHUNKERS,
        _register_entry_point,
        discover_entry_points=discover_entry_points,
    )

    return ChunkingEngine(
        registry=registry,
    )


def create_default_document_splitter(
    registry: ChunkingStrategyRegistry | None = None,
    *,
    discover_entry_points: bool | None = None,
) -> BaseDocumentsSplitter:
    
    return DocumentsSplitter(
        engine=create_default_chunking_engine(
            registry=registry,
            discover_entry_points=discover_entry_points,
        )
    )