# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.document_splitters.contracts.base_documents_splitter import BaseDocumentsSplitter
from intergrax.rag.document_splitters.documents_splitter import DocumentsSplitter
from intergrax.rag.document_splitters.engine.chunking_engine import ChunkingEngine
from intergrax.rag.document_splitters.registry.strategy_registry import (
    ChunkingStrategyRegistry,
)

from intergrax.rag.document_splitters.strategies.recursive_chunking_strategy import (
    RecursiveChunkingStrategy,
)
from intergrax.rag.document_splitters.strategies.langchain_recursive_chunking_strategy import (
    LangChainRecursiveChunkingStrategy,
)
from intergrax.rag.document_splitters.strategies.semantic_chunking_strategy import (
    SemanticChunkingStrategy,
)
from intergrax.rag.document_splitters.strategies.parent_child_chunking_strategy import (
    ParentChildChunkingStrategy,
)
from intergrax.rag.document_splitters.strategies.docling_chunking_strategy import (
    DoclingChunkingStrategy,
)
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager


def create_default_chunking_engine(
    registry: ChunkingStrategyRegistry | None = None,
) -> ChunkingEngine:
    """
    Create a ChunkingEngine with default chunking strategies registered.

    Allows dependency override by providing a custom registry.
    """

    if registry is None:
        registry = ChunkingStrategyRegistry()

        registry.register(RecursiveChunkingStrategy())
        registry.register(LangChainRecursiveChunkingStrategy())
        registry.register(SemanticChunkingStrategy(embedding_manager=create_default_embedding_manager()))
        registry.register(ParentChildChunkingStrategy())
        registry.register(DoclingChunkingStrategy())

    return ChunkingEngine(
        registry=registry,
    )


def create_default_document_splitter(
    registry: ChunkingStrategyRegistry | None = None,
) -> BaseDocumentsSplitter:
    
    return DocumentsSplitter(
        engine=create_default_chunking_engine(registry=registry)
    )