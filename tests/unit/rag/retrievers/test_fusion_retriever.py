# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.retrievers.providers.fusion_retriever import FusionRetriever
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry


pytestmark = pytest.mark.unit


def _hit(identifier: str, score: float, rank: int) -> RetrievalHit:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": identifier, "root_document_id": identifier},
            "scope": {"tenant_id": "tenant-a", "namespace": "namespace-a"},
            "content": f"doc-{identifier}",
            "metadata": {},
            "provenance": {"source_kind": "test", "source_id": identifier},
        }
    )
    return RetrievalHit(
        document=document,
        score=score,
        rank=rank,
        channel="dense",
        vector_id=identifier,
    )


class FakeRetrieverA(BaseRetriever):

    @classmethod
    def name(cls) -> str:
        return "fake_a"

    def retrieve(self, query: RetrieverQuery) -> Sequence[RetrievalHit]:

        return [
            _hit("a", 0.9, 0),
            _hit("b", 0.8, 1),
        ]


class FakeRetrieverB(BaseRetriever):

    @classmethod
    def name(cls) -> str:
        return "fake_b"

    def retrieve(self, query: RetrieverQuery) -> Sequence[RetrievalHit]:

        return [
            _hit("a", 0.7, 0),
            _hit("c", 0.85, 1),
        ]


def test_fusion_retriever_merges_and_deduplicates():

    registry = RetrieverRegistry()

    registry.register(FakeRetrieverA())
    registry.register(FakeRetrieverB())

    retriever = FusionRetriever(
        registry=registry,
        retrievers=["fake_a", "fake_b"]
    )

    query = RetrieverQuery(
        query_text="test query",
        query_embedding=None,
        top_k=2,
        metadata_filter=None,
        include_embeddings=False,
    )

    results = retriever.retrieve(query)

    assert len(results) == 2

    assert isinstance(results, tuple)
    assert all(isinstance(result, RetrievalHit) for result in results)
    ids = {r.vector_id for r in results}

    # ensure merged and deduplicated
    assert ids.issubset({"a", "b", "c"})

    # ensure metadata propagated
    assert [result.rank for result in results] == [0, 1]
    assert all(result.channel == "hybrid" for result in results)