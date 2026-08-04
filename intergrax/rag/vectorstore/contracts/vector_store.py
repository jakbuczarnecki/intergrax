# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

from intergrax.knowledge.contracts.validation import JsonValue
from intergrax.rag.vectorstore.contracts.native_vectorstore import MetadataFilter


@dataclass(frozen=True)
class LegacyVectorStoreHit:
    """
    Legacy provider result owned by LCI-3D.

    ``VectorstoreManager`` converts this private transport shape into the
    native hit before returning it to core callers.
    """
    id: str
    content: str
    metadata: dict[str, JsonValue]
    similarity_score: float
    rank: int
    embedding: Optional[List[float]] = None


VectorStoreHit = LegacyVectorStoreHit


class VectorStore(ABC):
    """
    Legacy provider compatibility port owned by LCI-3D.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: Sequence[object],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Add or upsert documents with corresponding embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[LegacyVectorStoreHit]:
        """
        Query top_k most similar vectors.
        Must return similarity_score normalized to [0,1].
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """
        Delete vectors by ids.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """
        Return number of stored vectors.
        """
        raise NotImplementedError

    def list_collections(self) -> List[str]:
        """Return logical collection names exposed by this store (default: single active collection)."""
        name = attribute_access.optional(self, "collection_name", None)
        if name is None and hasattr(self, "cfg"):
            cfg = attribute_access.optional(self, "cfg")
            name = attribute_access.optional(cfg, "collection_name", None)
        if name:
            return [str(name)]
        tenant_id = attribute_access.optional(self, "_tenant_id", "default")
        return [f"inmemory:{tenant_id}"]