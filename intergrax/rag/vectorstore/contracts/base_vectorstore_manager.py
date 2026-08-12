# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)


class BaseVectorstoreManager(ABC):
    @property
    def bound_scope(self) -> VectorStoreScope | None:
        """Public immutable scope bound by an integration, when available."""
        return None


    @abstractmethod
    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope | None = None,
    ) -> Sequence[str] | None:
        """Native write contract implemented by concrete managers."""
        raise NotImplementedError

    def add_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        embeddings: Sequence[Sequence[float]] | NDArray[np.float32],
        *,
        ids: Sequence[str] | None = None,
        scope: VectorStoreScope | None = None,
        base_metadata: object | None = None,
    ) -> Sequence[str] | None:
        """Native convenience wrapper; LangChain documents are not accepted."""
        if base_metadata:
            raise ValueError("base_metadata is not supported by the native contract")
        docs = list(documents)
        vectors = list(embeddings)
        if len(docs) != len(vectors):
            raise ValueError("documents and embeddings length mismatch")
        if ids is not None and len(ids) != len(docs):
            raise ValueError("ids and documents length mismatch")
        if any(not isinstance(document, KnowledgeDocument) for document in docs):
            raise TypeError("documents must contain only KnowledgeDocument values")
        records = [
            VectorStoreRecord(
                document=document,
                embedding=vector,
                vector_id=ids[index]
                if ids is not None
                else document.identity.document_id,
            )
            for index, (document, vector) in enumerate(zip(docs, vectors))
        ]
        return self.add_records(records, scope=scope)

    @abstractmethod
    def query(
        self,
        query_embedding: NDArray[np.float32] | Sequence[float],
        *,
        scope: VectorStoreScope | None = None,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        ids: Sequence[str],
        *,
        scope: VectorStoreScope | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_source_record_ids(
        self,
        *,
        source_id: str,
        scope: VectorStoreScope | None = None,
    ) -> Sequence[str]:
        """Return all persisted vector IDs owned by one canonical source."""
        raise NotImplementedError

    @abstractmethod
    def count(self, *, scope: VectorStoreScope | None = None) -> int:
        raise NotImplementedError

    def supports_native_hybrid_search(self) -> bool:
        """Return whether this manager can execute provider-native hybrid search."""
        return False