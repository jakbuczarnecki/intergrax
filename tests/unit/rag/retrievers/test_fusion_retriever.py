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


def _hit(
    identifier: str,
    score: float,
    rank: int,
    *,
    workspace_id: str | None = None,
) -> RetrievalHit:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": identifier, "root_document_id": identifier},
            "scope": {
                "tenant_id": "tenant-a",
                "namespace": "namespace-a",
                "workspace_id": workspace_id,
            },
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


def test_fusion_does_not_merge_same_vector_id_across_workspaces() -> None:
    class WorkspaceRetrieverA(BaseRetriever):
        def __init__(self, name: str, workspace_id: str) -> None:
            self._name = name
            self._workspace_id = workspace_id

        @classmethod
        def name(cls) -> str:
            return "workspace_a"

        def retrieve(self, query: RetrieverQuery) -> Sequence[RetrievalHit]:
            return [_hit(
                "same-vector",
                0.9,
                0,
                workspace_id=self._workspace_id,
            )]

    class WorkspaceRetrieverB(WorkspaceRetrieverA):
        @classmethod
        def name(cls) -> str:
            return "workspace_b"

    registry = RetrieverRegistry()
    registry.register(WorkspaceRetrieverA("workspace_a", "workspace-a"))
    registry.register(WorkspaceRetrieverB("workspace_b", "workspace-b"))

    results = FusionRetriever(
        registry=registry,
        retrievers=["workspace_a", "workspace_b"],
    ).retrieve(
        RetrieverQuery(query_text="test", query_embedding=None, top_k=2)
    )

    assert len(results) == 2
    assert {result.document.scope.workspace_id for result in results} == {
        "workspace-a",
        "workspace-b",
    }