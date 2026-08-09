# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager

from intergrax.rag.indexing.contracts.index_strategy import IndexStrategy
from intergrax.rag.indexing.pipeline.indexing_pipeline import IndexingPipeline
from intergrax.rag.indexing.strategies.single_index_strategy import SingleIndexStrategy


class IndexingManager:
    """
    Entry point for document indexing.

    Responsible for constructing the indexing pipeline
    and executing the indexing process.
    """

    def __init__(
        self,
        *,
        embed_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
        strategy: Optional[IndexStrategy] = None,
    ) -> None:

        if strategy is None:
            strategy = SingleIndexStrategy()

        self.embed_manager = embed_manager
        self.vectorstore = vectorstore

        self.pipeline = IndexingPipeline(
            strategy=strategy
        )

    def index_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> Sequence[str]:
        """
        Execute indexing for provided documents and return persisted vector IDs.
        """

        materialized_documents = tuple(documents)
        if not materialized_documents:
            return []

        validated_documents: list[KnowledgeDocument] = []
        for document in materialized_documents:
            if not isinstance(document, KnowledgeDocument):
                raise TypeError("documents must contain only KnowledgeDocument values")
            validated_documents.append(
                KnowledgeDocument.model_validate(document.model_dump(mode="python"))
            )

        return self.pipeline.run(
            documents=tuple(validated_documents),
            embed_manager=self.embed_manager,
            vectorstore=self.vectorstore,
        )