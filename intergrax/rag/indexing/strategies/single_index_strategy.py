# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
)
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
    ) -> Sequence[str]:

        if not documents:
            return []

        result = embed_manager.embed_documents(documents)
        records = [
            VectorStoreRecord(
                document=document,
                embedding=result.embeddings[index],
                vector_id=document.identity.document_id,
            )
            for index, document in enumerate(result.documents)
        ]
        stored_ids = vectorstore.add_records(records)
        if stored_ids is None:
            return [record.vector_id for record in records]
        persisted_ids = list(stored_ids)
        if len(persisted_ids) != len(records):
            raise ValueError("vectorstore returned an unexpected number of vector IDs")
        return persisted_ids