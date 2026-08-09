# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager


class IndexStrategy(ABC):
    """
    Contract for index construction strategies.

    Implementations define how native documents are transformed into
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
        documents: Sequence[KnowledgeDocument],
        embed_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
    ) -> Sequence[str]:
        """
        Build a vector index from the provided documents and return the
        persisted vector IDs for those documents.

        Parameters
        ----------
        documents
            Native input documents or chunks to index. The strategy preserves
            their order and does not mutate them.

        embed_manager
            Embedding manager responsible for generating vector embeddings.

        vectorstore
            Temporary legacy vectorstore consumer where indexed documents will
            be inserted.
        """
        raise NotImplementedError