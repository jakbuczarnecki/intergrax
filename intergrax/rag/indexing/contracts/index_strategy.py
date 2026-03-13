# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


class IndexStrategy(ABC):
    """
    Contract for index construction strategies.

    Implementations define how documents are transformed into
    vectorstore indexes.

    The strategy receives documents and is responsible for:

        1. generating embeddings
        2. inserting vectors into the target vectorstore

    Strategies must remain deterministic and stateless.
    """

    @abstractmethod
    def build_index(
        self,
        *,
        documents: List[Document],
        embed_manager: EmbeddingManager,
        vectorstore: VectorstoreManager,
    ) -> None:
        """
        Build a vector index from the provided documents.

        Parameters
        ----------
        documents
            Input documents or chunks to index.

        embed_manager
            Embedding manager responsible for generating vector embeddings.

        vectorstore
            Target vectorstore where indexed documents will be inserted.
        """
        raise NotImplementedError