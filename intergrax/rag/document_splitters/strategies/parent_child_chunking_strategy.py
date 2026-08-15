# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey


class ParentChildChunkingStrategy(BaseChunkingStrategy):
    """
    Parent-child chunking strategy.

    Generates hierarchical chunks where smaller child chunks reference
    a larger parent chunk containing broader context.
    """

    def __init__(
        self,
        parent_size: int = 2000,
        child_size: int = 400,
        child_overlap: int = 100,
    ) -> None:

        if child_overlap >= child_size:
            raise ValueError("child_overlap must be smaller than child_size")

        if child_size >= parent_size:
            raise ValueError("child_size must be smaller than parent_size")

        self._parent_size = parent_size
        self._child_size = child_size
        self._child_overlap = child_overlap

    @classmethod
    def strategy_id(cls) -> str:
        return "parent_child"

    def chunk(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> Sequence[KnowledgeDocument]:

        chunks: list[KnowledgeDocument] = []

        for doc in documents:
            text = doc.content
            source_id = doc.identity.document_id

            parent_start = 0
            parent_index = 0
            chunk_index = 0

            while parent_start < len(text):
                parent_end = parent_start + self._parent_size
                parent_text = text[parent_start:parent_end]

                parent_id = f"{source_id}:parent_{parent_index}"

                child_start = 0

                while child_start < len(parent_text):
                    child_end = child_start + self._child_size
                    child_text = parent_text[child_start:child_end]

                    chunks.append(
                        build_derived_chunk(
                            doc,
                            content=child_text,
                            strategy_id=self.strategy_id(),
                            chunk_index=chunk_index,
                            metadata_updates={
                                ChunkMetadataKey.PARENT_CHUNK_ID.value: parent_id,
                                ChunkMetadataKey.SECTION.value: parent_id,
                                ChunkMetadataKey.PARENT_CHUNK_INDEX.value: parent_index,
                            },
                        )
                    )

                    child_start = child_end - self._child_overlap
                    chunk_index += 1

                parent_start = parent_end
                parent_index += 1

        return chunks
