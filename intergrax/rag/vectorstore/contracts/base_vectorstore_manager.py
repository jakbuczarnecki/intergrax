# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from langchain_core.documents import Document

from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.rag.vectorstore.contracts.vector_store import VectorStoreHit


class BaseVectorstoreManager(ABC):


    @abstractmethod
    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError