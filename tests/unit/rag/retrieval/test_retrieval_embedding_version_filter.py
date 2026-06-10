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

pytestmark = pytest.mark.gate


class _VersionedRetriever(BaseRetriever):
    @classmethod
    def name(cls) -> str:
        return "versioned"

    def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:
        return [
            RetrieverCandidate(
                id="new",
                content="fresh",
                metadata={"embedding_model_version": "v2"},
                score=0.9,
            ),
            RetrieverCandidate(
                id="old",
                content="stale",
                metadata={"embedding_model_version": "v1"},
                score=0.85,
            ),
        ]


class _VersionedRetrieverManager(BaseRetrieverManager):
    def retrieve(
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter=None,
        include_embeddings: bool = False,
    ) -> List[RetrieverCandidate]:
        return _VersionedRetriever().retrieve(
            RetrieverQuery(
                query_text=query_text,
                query_embedding=None,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        )

    def retrieve_query(self, query: RetrieverQuery, retriever_id: str) -> List[RetrieverCandidate]:
        return self.retrieve(query.query_text, retriever_id=retriever_id, top_k=query.top_k)


def test_retrieval_service_filters_mismatched_embedding_versions() -> None:
    service = RetrievalService(
        retriever_manager=_VersionedRetrieverManager(),
        profile=RagProfile(
            enable_rerank=False,
            embedding_model_version="v2",
            embedding_version_filter_on_retrieve=True,
        ),
    )
    result = service.retrieve(RetrievalRequest(query="policy"))

    assert result.used is True
    assert [chunk.id for chunk in result.chunks] == ["new"]
    assert result.trace.embedding_version_filtered_count == 1


def test_retrieval_service_returns_mismatch_reason_when_all_filtered() -> None:
    service = RetrievalService(
        retriever_manager=_VersionedRetrieverManager(),
        profile=RagProfile(
            enable_rerank=False,
            embedding_model_version="v3",
            embedding_version_filter_on_retrieve=True,
        ),
    )
    result = service.retrieve(RetrievalRequest(query="policy"))

    assert result.used is False
    assert result.reason == "embedding_version_mismatch"
    assert result.trace.embedding_version_filtered_count == 2
