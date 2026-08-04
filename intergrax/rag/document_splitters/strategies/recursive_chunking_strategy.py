# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy


class RecursiveChunkingStrategy(BaseChunkingStrategy):
    """
    Deterministic recursive chunking strategy.

    Splits documents into smaller chunks using a maximum character size
    and overlap to preserve context continuity.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @classmethod
    def strategy_id(cls) -> str:
        return "recursive"

    def chunk(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> Sequence[KnowledgeDocument]:

        chunks: list[KnowledgeDocument] = []

        for doc in documents:
            text = doc.content
            start = 0
            chunk_index = 0

            while start < len(text):
                end = start + self._chunk_size
                chunk_text = text[start:end]

                chunks.append(
                    build_derived_chunk(
                        doc,
                        content=chunk_text,
                        strategy_id=self.strategy_id(),
                        chunk_index=chunk_index,
                    )
                )

                start = end - self._chunk_overlap
                chunk_index += 1

        return chunks
