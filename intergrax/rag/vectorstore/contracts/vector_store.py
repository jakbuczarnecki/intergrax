# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)

__all__ = [
    "MetadataFilter",
    "VectorStore",
    "VectorStoreHit",
    "VectorStoreRecord",
    "VectorStoreScope",
]


class VectorStore(ABC):
    """Native provider-facing vector-store port."""

    @abstractmethod
    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str] | None:
        """Add or upsert native records within the authoritative scope."""
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embedding: NDArray[np.float32] | Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        """Return native hits with scores normalized to ``[0, 1]``."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        """Delete only IDs belonging to the authoritative scope."""
        raise NotImplementedError

    @abstractmethod
    def count(self, *, scope: VectorStoreScope) -> int:
        """Count only vectors belonging to the authoritative scope."""
        raise NotImplementedError

    def list_source_record_ids(
        self,
        *,
        source_id: str,
        scope: VectorStoreScope,
        root_document_id: str | None = None,
    ) -> Sequence[str]:
        """Return persisted vector IDs owned by one canonical source."""
        raise RuntimeError("vectorstore_source_record_lookup_not_supported")

    def list_collections(self) -> list[str]:
        """Return logical collection names exposed by this store (default: single active collection)."""
        name = attribute_access.optional(self, "collection_name", None)
        if name is None and hasattr(self, "cfg"):
            cfg = attribute_access.optional(self, "cfg")
            name = attribute_access.optional(cfg, "collection_name", None)
        if name:
            return [str(name)]
        tenant_id = attribute_access.optional(self, "_tenant_id", "default")
        return [f"inmemory:{tenant_id}"]