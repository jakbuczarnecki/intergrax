# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.rag.indexing.contracts.index_strategy import IndexStrategy


class SingleIndexStrategy(IndexStrategy):
    """
    Default indexing strategy.

    Builds a single vector index where all documents/chunks are embedded
    and inserted into a single vectorstore collection.
    """

    def build_index(
        self,
        *,
        documents: List[Document],
        embed_manager: EmbeddingManager,
        vectorstore: VectorstoreManager,
    ) -> None:

        if not documents:
            return

        embeddings, aligned_docs = embed_manager.embed_documents(documents)

        vectorstore.add_documents(
            documents=aligned_docs,
            embeddings=embeddings,
        )