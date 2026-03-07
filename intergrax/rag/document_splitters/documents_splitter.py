
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional, Sequence

from langchain_core.documents import Document

from intergrax.rag.document_splitters.engine.chunking_engine import ChunkingEngine
from intergrax.rag.document_splitters.strategies.langchain_recursive_chunking_strategy import LangChainRecursiveChunkingStrategy


class DocumentsSplitter:
    """
    Entry point for document chunking.

    Responsibilities
    ----------------
    - accept documents produced by the ingestion pipeline
    - delegate chunking execution to ChunkingEngine

    This class intentionally contains no chunking logic.
    All chunking behaviour is implemented by chunking strategies
    registered in ChunkingStrategyRegistry and executed by ChunkingEngine.
    """

    DEFAULT_STRATEGY = LangChainRecursiveChunkingStrategy

    def __init__(
        self,
        *,
        engine: ChunkingEngine,
    ) -> None:
        self._engine = engine


    def split_documents(
        self,
        documents: Sequence[Document],
        strategy_id: Optional[str] = None,
    ) -> Sequence[Document]:
        """
        Split documents using a configured chunking strategy.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents produced by the ingestion pipeline.

        strategy_id : str
            Identifier of the chunking strategy.

        Returns
        -------
        Sequence[Document]
            Chunked documents produced by the selected strategy.
        """

        if strategy_id is None:
            strategy_id = self.DEFAULT_STRATEGY.strategy_id()

        return self._engine.chunk(
            documents=documents,
            strategy_id=strategy_id,
        )