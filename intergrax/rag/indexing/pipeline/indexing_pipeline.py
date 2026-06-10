# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.indexing.contracts.index_strategy import IndexStrategy


class IndexingPipeline:
    """
    Pipeline responsible for executing document indexing.

    Delegates the actual indexing logic to the configured IndexStrategy.
    """

    def __init__(
        self,
        *,
        strategy: IndexStrategy,
    ):
        self.strategy = strategy

    def run(
        self,
        *,
        documents: List[Document],
        embed_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
    ) -> None:

        if not documents:
            return

        self.strategy.build_index(
            documents=documents,
            embed_manager=embed_manager,
            vectorstore=vectorstore,
        )