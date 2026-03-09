# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter



@dataclass(frozen=True)
class RetrieverQuery:
    """
    Standardized retrieval query context passed to retriever strategies.
    """

    query_text: str

    # optional precomputed embedding
    query_embedding: Sequence[float] | None

    # number of results requested
    top_k: int

    # metadata filtering compatible with VectorStore contract
    metadata_filter: MetadataFilter | None = None

    # request embeddings in returned candidates
    include_embeddings: bool = False


@dataclass
class RetrieverCandidate:
    """
    Standardized retrieval result candidate.
    """

    id: str
    content: str
    metadata: Dict[str, object]

    # normalized similarity score in range [0,1]
    score: float

    # optional embedding returned by vector store
    embedding: Sequence[float] | None = None

    # optional rank assigned by retriever strategy
    rank: int | None = None


class BaseRetriever(ABC):
    """
    Base contract for all Intergrax retrieval strategies.

    Implementations define the strategy used to retrieve
    candidate documents from vector stores or other sources.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Unique retriever identifier used by the registry.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: RetrieverQuery,
    ) -> List[RetrieverCandidate]:
        """
        Execute retrieval for a given query context.
        """
        raise NotImplementedError