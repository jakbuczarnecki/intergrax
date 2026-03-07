# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence, List

from langchain_core.documents import Document

from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.contracts.chunk_metadata_contract import ChunkMetadataKey


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
        documents: Sequence[Document],
    ) -> Sequence[Document]:

        chunks: List[Document] = []

        for doc in documents:
            text = doc.page_content
            metadata = dict(doc.metadata)

            start = 0
            chunk_index = 0

            while start < len(text):

                end = start + self._chunk_size
                chunk_text = text[start:end]

                chunk_metadata = dict(metadata)
                chunk_metadata[ChunkMetadataKey.CHUNK_INDEX] = chunk_index
                chunk_metadata[ChunkMetadataKey.CHUNK_STRATEGY] = self.strategy_id()

                chunk = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )

                chunks.append(chunk)

                start = end - self._chunk_overlap
                chunk_index += 1

        return chunks