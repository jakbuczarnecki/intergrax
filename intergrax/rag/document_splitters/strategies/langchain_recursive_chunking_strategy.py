# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey


class LangChainRecursiveChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy based on LangChain RecursiveCharacterTextSplitter.

    This strategy is widely used in production RAG systems and serves as a
    reliable baseline for chunking textual documents.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @classmethod
    def strategy_id(cls) -> str:
        return "langchain_recursive"

    def chunk(
        self,
        documents: Sequence[Document],
    ) -> Sequence[Document]:

        result_chunks: List[Document] = []

        for doc in documents:

            splits = self._splitter.split_text(doc.page_content)

            for index, text in enumerate(splits):

                metadata = dict(doc.metadata)
                metadata[ChunkMetadataKey.CHUNK_INDEX] = index                
                metadata[ChunkMetadataKey.CHUNK_STRATEGY] = self.strategy_id()
                metadata[ChunkMetadataKey.CHUNK_SIZE] = len(text)

                chunk = Document(
                    page_content=text,
                    metadata=metadata,
                )

                result_chunks.append(chunk)

        return result_chunks