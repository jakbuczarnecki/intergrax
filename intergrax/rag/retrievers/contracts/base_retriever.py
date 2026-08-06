# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreHit,
    VectorStoreScope,
)

if TYPE_CHECKING:
    from intergrax.rag.retrieval.retrieval_result import RetrievalChunk


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

    # authoritative vector-store routing scope
    scope: VectorStoreScope | None = None

    # request embeddings in returned candidates
    include_embeddings: bool = False


def _copy_document(document: KnowledgeDocument) -> KnowledgeDocument:
    if not isinstance(document, KnowledgeDocument):
        raise TypeError("document must be a KnowledgeDocument")
    try:
        return KnowledgeDocument.model_validate(document.model_dump(mode="python"))
    except Exception as exc:
        raise ValueError("document failed full revalidation") from exc


def _copy_embedding(
    embedding: Sequence[float] | NDArray[np.float32] | None,
) -> NDArray[np.float32] | None:
    if embedding is None:
        return None
    try:
        copied = np.array(embedding, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding must be a numeric vector") from exc
    if copied.ndim != 1 or copied.size == 0:
        raise ValueError("embedding must be a non-empty 1D vector")
    if not np.isfinite(copied).all():
        raise ValueError("embedding must contain only finite values")
    copied.setflags(write=False)
    return copied


def _validate_score(score: object) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be a number, not bool")
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("score must be finite and in [0.0, 1.0]")
    return value


def _validate_rank(rank: object) -> int:
    if type(rank) is not int or rank < 0:
        raise TypeError("rank must be an exact non-negative int")
    return rank


@dataclass(frozen=True)
class RetrievalHit:
    """Immutable, provider-neutral result at the retriever boundary.

    Scores are normalized similarities in the inclusive range ``[0.0, 1.0]``.
    Provider-specific ``VectorStoreHit`` values are converted with
    :meth:`from_vector_store_hit`; they never cross this boundary.
    """

    document: KnowledgeDocument
    score: float
    rank: int
    channel: str
    vector_id: str | None = None
    embedding: NDArray[np.float32] | None = None
    query_id: str | None = None
    query_text: str | None = None
    parent_vector_id: str | None = None
    child_vector_id: str | None = None
    source_rank: int | None = None
    retriever_name: str | None = None

    def __post_init__(self) -> None:
        document = _copy_document(self.document)
        score = _validate_score(self.score)
        rank = _validate_rank(self.rank)
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("channel must be a non-empty string")
        vector_id = self.vector_id
        if vector_id is not None and (not isinstance(vector_id, str) or not vector_id.strip()):
            raise ValueError("vector_id must be a non-empty string when provided")
        for field_name in (
            "query_id",
            "query_text",
            "parent_vector_id",
            "child_vector_id",
            "retriever_name",
        ):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string when provided")
        source_rank = self.source_rank
        if source_rank is not None:
            source_rank = _validate_rank(source_rank)
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "vector_id", vector_id)
        object.__setattr__(self, "embedding", _copy_embedding(self.embedding))
        object.__setattr__(self, "source_rank", source_rank)

    @classmethod
    def from_vector_store_hit(
        cls,
        hit: VectorStoreHit,
        *,
        channel: str,
        rank: int | None = None,
        query_id: str | None = None,
        query_text: str | None = None,
        parent_vector_id: str | None = None,
        child_vector_id: str | None = None,
        retriever_name: str | None = None,
    ) -> RetrievalHit:
        if not isinstance(hit, VectorStoreHit):
            raise TypeError("retriever input must be a VectorStoreHit")
        return cls(
            document=hit.document,
            score=hit.similarity_score,
            rank=hit.rank if rank is None else rank,
            channel=channel,
            vector_id=hit.vector_id,
            embedding=hit.embedding,
            query_id=query_id,
            query_text=query_text,
            parent_vector_id=parent_vector_id,
            child_vector_id=child_vector_id,
            source_rank=hit.rank,
            retriever_name=retriever_name,
        )

    @property
    def id(self) -> str:
        """Compatibility accessor for the retrieval service during LCI-4A."""
        return self.document.identity.document_id

    @property
    def content(self) -> str:
        return self.document.content

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "document": self.document.model_dump(mode="json"),
            "score": self.score,
            "rank": self.rank,
            "channel": self.channel,
            "vector_id": self.vector_id,
            "embedding": None if self.embedding is None else self.embedding.tolist(),
        }
        for field_name in (
            "query_id",
            "query_text",
            "parent_vector_id",
            "child_vector_id",
            "source_rank",
            "retriever_name",
        ):
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        return payload

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        del mode
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# Temporary typed adapter retained for LCI-4B reranker boundary.
def retrieval_hit_to_chunk(hit: RetrievalHit) -> RetrievalChunk:
    """Adapt a native hit without tunneling retrieval fields through metadata."""
    if not isinstance(hit, RetrievalHit):
        raise TypeError("hit must be a RetrievalHit")

    from intergrax.rag.retrieval.retrieval_result import RetrievalChunk

    user_metadata = dict(hit.document.metadata)
    return RetrievalChunk(
        id=hit.document.identity.document_id,
        text=hit.document.content,
        score=hit.score,
        rank=hit.rank,
        channel=hit.channel,
        vector_id=hit.vector_id,
        scope=hit.document.scope.model_dump(mode="json"),
        provenance=hit.document.provenance.model_dump(mode="json"),
        user_metadata=dict(user_metadata),
        metadata=dict(user_metadata),
    )


@dataclass(frozen=True)
class RetrievalResult:
    """Immutable envelope for public retrieval APIs that need query context."""

    hits: tuple[RetrievalHit, ...]
    query: str
    requested_top_k: int | None = None
    retrieval_mode: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        hits = tuple(self.hits)
        if any(not isinstance(hit, RetrievalHit) for hit in hits):
            raise TypeError("hits must contain only RetrievalHit values")
        requested_top_k = self.requested_top_k
        if requested_top_k is not None:
            requested_top_k = _validate_rank(requested_top_k)
        if self.retrieval_mode is not None and (
            not isinstance(self.retrieval_mode, str) or not self.retrieval_mode.strip()
        ):
            raise ValueError("retrieval_mode must be a non-empty string when provided")
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "requested_top_k", requested_top_k)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hits": [hit.to_dict() for hit in self.hits],
            "query": self.query,
        }
        if self.requested_top_k is not None:
            payload["requested_top_k"] = self.requested_top_k
        if self.retrieval_mode is not None:
            payload["retrieval_mode"] = self.retrieval_mode
        return payload

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        del mode
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class RetrieverCandidate:
    """LCI-4C compatibility shape retained only for the graph boundary."""

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

    requires_query_embedding: bool = True

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
    ) -> Sequence[RetrievalHit]:
        """
        Execute retrieval for a given query context.
        """
        raise NotImplementedError