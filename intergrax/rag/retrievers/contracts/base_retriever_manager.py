# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import abstractmethod
from typing import List, Sequence

from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope


class BaseRetrieverManager:
    
    @abstractmethod
    def retrieve(
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter=None,
        scope: VectorStoreScope | None = None,
        include_embeddings: bool = False,
    ) -> List[RetrievalHit]:
        """
        Retrieve candidates for query text.
        """
        raise NotImplementedError
    
    
    @abstractmethod
    def retrieve_query(
        self,
        query: RetrieverQuery,
        retriever_id: str,
    ) -> List[RetrievalHit]:
        """
        Retrieve using preconstructed RetrieverQuery.
        """
        raise NotImplementedError