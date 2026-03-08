# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence, List

from langchain_core.documents import Document

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
        documents: Sequence[Document],
    ) -> Sequence[Document]:

        chunks: List[Document] = []

        for doc in documents:

            text = doc.page_content
            metadata = dict(doc.metadata)

            parent_start = 0
            parent_index = 0

            while parent_start < len(text):

                parent_end = parent_start + self._parent_size
                parent_text = text[parent_start:parent_end]

                parent_id = f"parent_{parent_index}"

                child_start = 0
                child_index = 0

                while child_start < len(parent_text):

                    child_end = child_start + self._child_size
                    child_text = parent_text[child_start:child_end]

                    child_metadata = dict(metadata)
                    child_metadata[ChunkMetadataKey.PARENT_CHUNK_ID] = parent_id
                    child_metadata[ChunkMetadataKey.PARENT_CHUNK_INDEX] = parent_index
                    child_metadata[ChunkMetadataKey.CHUNK_SIZE] = len(text)
                    child_metadata[ChunkMetadataKey.CHUNK_INDEX] = child_index                
                    child_metadata[ChunkMetadataKey.CHUNK_STRATEGY] = self.strategy_id()

                    chunk = Document(
                        page_content=child_text,
                        metadata=child_metadata,
                    )

                    chunks.append(chunk)

                    child_start = child_end - self._child_overlap
                    child_index += 1

                parent_start = parent_end
                parent_index += 1

        return chunks