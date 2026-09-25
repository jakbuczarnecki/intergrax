# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import abstractmethod
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.retrievers.engine.retriever_execution import RetrieverExecutionMetadata
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter


class BaseRetrieverManager:

    @property
    def last_execution(self) -> RetrieverExecutionMetadata | None:
        """Optional execution metadata from the most recent retrieve call."""
        return None

    @property
    def supports_scoped_retrieval(self) -> bool:
        """Whether this manager explicitly supports scoped retrieval."""
        return False
    
    @abstractmethod
    def retrieve(
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: NDArray[np.float32] | Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
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