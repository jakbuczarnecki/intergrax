# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document


@dataclass(frozen=True)
class MetadataFilter:
    """
    Provider-agnostic metadata filter model.
    Providers are responsible for translating this
    into native filtering mechanisms.
    """
    conditions: Dict[str, Any]


@dataclass(frozen=True)
class VectorStoreHit:
    """
    Unified vector store hit model returned by all providers.
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int
    embedding: Optional[List[float]] = None


class VectorStore(ABC):
    """
    Contract for all vector store providers.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Add or upsert documents with corresponding embeddings.
        """

    @abstractmethod
    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
        """
        Query top_k most similar vectors.
        Must return similarity_score normalized to [0,1].
        """

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """
        Delete vectors by ids.
        """

    @abstractmethod
    def count(self) -> int:
        """
        Return number of stored vectors.
        """