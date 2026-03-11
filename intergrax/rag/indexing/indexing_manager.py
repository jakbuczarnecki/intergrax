# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

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
        embed_manager: EmbeddingManager,
        vectorstore: VectorstoreManager,
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
        documents: List[Document],
    ) -> None:
        """
        Execute indexing for provided documents.
        """

        if not documents:
            return

        self.pipeline.run(
            documents=documents,
            embed_manager=self.embed_manager,
            vectorstore=self.vectorstore,
        )