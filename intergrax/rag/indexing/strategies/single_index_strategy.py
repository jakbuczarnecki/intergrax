# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import validate_embedding_matrix
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
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
        embed_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
    ) -> None:

        if not documents:
            return

        raw_embeddings = embed_manager.embed_texts(
            [document.page_content for document in documents]
        )
        embeddings = validate_embedding_matrix(
            raw_embeddings,
            expected_rows=len(documents),
        )

        vectorstore.add_documents(
            documents=documents,
            embeddings=embeddings,
        )