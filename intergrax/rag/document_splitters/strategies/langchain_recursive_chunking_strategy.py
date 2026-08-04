# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy


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

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ModuleNotFoundError as exc:
            if exc.name == "langchain_text_splitters":
                raise RuntimeError(
                    "LangChain recursive chunking requires the "
                    "'rag-langchain-splitters' optional dependency. "
                    "Install Intergrax-ai[rag-langchain-splitters]."
                ) from exc
            raise

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @classmethod
    def strategy_id(cls) -> str:
        return "langchain_recursive"

    def chunk(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> Sequence[KnowledgeDocument]:

        result_chunks: list[KnowledgeDocument] = []

        for doc in documents:
            splits = self._splitter.split_text(doc.content)

            for index, text in enumerate(splits):
                result_chunks.append(
                    build_derived_chunk(
                        doc,
                        content=text,
                        strategy_id=self.strategy_id(),
                        chunk_index=index,
                    )
                )

        return result_chunks
