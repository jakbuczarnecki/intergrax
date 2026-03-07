# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.registry.strategy_registry import ChunkingStrategyRegistry



class ChunkingEngine:
    """
    Execution engine responsible for applying a chunking strategy
    to a collection of documents.

    The engine resolves the strategy from the registry and delegates
    the chunking execution to the selected strategy.
    """

    def __init__(
        self,
        registry: ChunkingStrategyRegistry,
    ) -> None:
        self._registry = registry

    def chunk(
        self,
        documents: Sequence[Document],
        strategy_id: str,
    ) -> Sequence[Document]:
        """
        Execute chunking using the specified strategy.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents produced by the ingestion pipeline.

        strategy_id : str
            Identifier of the chunking strategy.

        Returns
        -------
        Sequence[Document]
            Chunked documents.
        """

        strategy: BaseChunkingStrategy = self._registry.resolve(strategy_id)

        chunks = strategy.chunk(documents)

        return chunks