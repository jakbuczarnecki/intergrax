# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    to_legacy_rag_document,
)
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
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
        documents: Sequence[KnowledgeDocument],
        embed_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
    ) -> None:

        if not documents:
            return

        result = embed_manager.embed_documents(documents)
        legacy_documents = [
            to_legacy_rag_document(document) for document in result.documents
        ]

        vectorstore.add_documents(
            documents=legacy_documents,
            embeddings=result.embeddings,
        )