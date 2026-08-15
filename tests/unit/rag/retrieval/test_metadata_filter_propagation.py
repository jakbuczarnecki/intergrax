# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import List, Sequence

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    MetadataMembershipCondition,
    VectorStoreScope,
)

pytestmark = pytest.mark.unit


class _StubRetriever(BaseRetriever):
    @classmethod
    def name(cls) -> str:
        return "stub"

    def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:
        return []


class _CapturingRetrieverManager(BaseRetrieverManager):
    def __init__(self) -> None:
        self.last_query: RetrieverQuery | None = None

    @property
    def supports_scoped_retrieval(self) -> bool:
        return True

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
    ) -> List[RetrieverCandidate]:
        self.last_query = RetrieverQuery(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            scope=scope,
        )
        return []

    def retrieve_query(self, query: RetrieverQuery, retriever_id: str) -> List[RetrieverCandidate]:
        self.last_query = query
        return []


def test_retrieval_request_carries_membership_filter_unchanged() -> None:
    scope = VectorStoreScope(tenant_id="tenant-a", namespace="rag")
    metadata_filter = MetadataFilter(
        membership=(
            MetadataMembershipCondition(
                field="source_id",
                allowed_values=("src-a", "src-b"),
            ),
        )
    )
    manager = _CapturingRetrieverManager()
    service = RetrievalService(
        retriever_manager=manager,
        profile=RagProfile(
            retriever_id="vector_similarity",
            enable_rerank=False,
            route_mode="off",
        ),
    )

    service.retrieve(
        RetrievalRequest(
            query="hello",
            scope=scope,
            metadata_filter=metadata_filter,
        )
    )

    assert manager.last_query is not None
    assert manager.last_query.metadata_filter is metadata_filter
    assert manager.last_query.metadata_filter.membership[0].allowed_values == (
        "src-a",
        "src-b",
    )


def test_retrieval_request_without_membership_keeps_legacy_behavior() -> None:
    scope = VectorStoreScope(tenant_id="tenant-a")
    manager = _CapturingRetrieverManager()
    service = RetrievalService(
        retriever_manager=manager,
        profile=RagProfile(
            retriever_id="vector_similarity",
            enable_rerank=False,
            route_mode="off",
        ),
    )

    service.retrieve(RetrievalRequest(query="hello", scope=scope))

    assert manager.last_query is not None
    assert manager.last_query.metadata_filter is None
