# © Artur Czarnecki. All rights reserved.
# pyright: reportArgumentType=false, reportCallIssue=false

"""B-P6-R item 3 - the deep-tier agentic loop carries the typed freshness verdict (PDC-D2-R005)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.governance.embedding_version_policy import clear_reindex_queue_hooks
from intergrax.rag.governance.source_freshness import SourceFreshnessObservation, SourceFreshnessStatus
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever import RetrievalHit
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_reindex_hooks() -> Any:
    clear_reindex_queue_hooks()
    yield
    clear_reindex_queue_hooks()


class _Manager(BaseRetrieverManager):
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self._hits = hits
        self.calls = 0

    def retrieve(  # type: ignore[override]
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter: Any = None,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> list[RetrievalHit]:
        self.calls += 1
        return list(self._hits)

    def retrieve_query(self, query: Any, retriever_id: str) -> list[Any]:
        self.calls += 1
        return list(self._hits)


class _PerPassManager(_Manager):
    """Returns a different hit list on each pass (the refined query surfaces other sources)."""

    def __init__(self, passes: list[list[RetrievalHit]]) -> None:
        super().__init__([])
        self._passes = passes

    def retrieve(self, query_text: str, **kwargs: Any) -> list[RetrievalHit]:  # type: ignore[override]
        self.calls += 1
        return list(self._passes[min(self.calls, len(self._passes)) - 1])


def _hit(document_id: str, source: str, *, revision: str) -> RetrievalHit:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": document_id, "root_document_id": document_id},
            "scope": {"tenant_id": "tenant-a", "namespace": "namespace-a"},
            "content": f"content of {document_id}",
            "metadata": {"source_path": source},
            "provenance": {"source_kind": "test", "source_id": source, "source_revision": revision},
        }
    )
    return RetrievalHit(document=document, score=0.9, rank=0, channel="dense", vector_id=f"vec-{document_id}")


def _deep_service(manager: _Manager, **kwargs: Any) -> RetrievalService:
    # min_chunks above the hit count keeps the loop refining, so it really is a multi-pass retrieval.
    profile = RagProfile(
        enable_rerank=False,
        agentic_enabled=True,
        agentic_max_iterations=3,
        agentic_min_chunks=5,
    )
    return RetrievalService(retriever_manager=manager, profile=profile, **kwargs)


_REQUEST = RetrievalRequest(query="what is the refund policy for enterprise plans", route_tier_override="deep")


def test_multi_pass_retrieval_with_a_stale_source_returns_the_verdict() -> None:
    manager = _Manager([_hit("a", "docs/a.md", revision="rev-1"), _hit("b", "docs/b.md", revision="rev-4")])
    current = {
        "docs/a.md": SourceFreshnessObservation(source_id="docs/a.md", revision="rev-2"),
        "docs/b.md": SourceFreshnessObservation(source_id="docs/b.md", revision="rev-4"),
    }

    result = _deep_service(manager, source_observer=current.get).retrieve(_REQUEST)

    assert manager.calls > 1  # more than one pass ran
    assert result.trace.agentic_iteration is not None and result.trace.agentic_iteration > 1
    assert result.used is True
    assert result.freshness is not None
    by_source = {v.source_id: v for v in result.freshness.verdicts}
    assert by_source["docs/a.md"].status is SourceFreshnessStatus.STALE
    assert by_source["docs/b.md"].status is SourceFreshnessStatus.FRESH
    assert result.freshness.stale_source_ids == ("docs/a.md",)


def test_multi_pass_retrieval_without_an_observer_keeps_none() -> None:
    manager = _Manager([_hit("a", "docs/a.md", revision="rev-1")])

    result = _deep_service(manager).retrieve(_REQUEST)

    assert manager.calls > 1
    assert result.used is True
    assert result.freshness is None


def test_the_verdict_is_the_one_of_the_final_pass_result() -> None:
    """The loop returns the underlying single-pass result, so the verdict describes exactly its chunks."""
    manager = _PerPassManager(
        [
            [_hit("a", "docs/a.md", revision="rev-1")],
            [_hit("b", "docs/b.md", revision="rev-4")],
        ]
    )
    current = {
        "docs/a.md": SourceFreshnessObservation(source_id="docs/a.md", revision="rev-2"),
        "docs/b.md": SourceFreshnessObservation(source_id="docs/b.md", revision="rev-4"),
    }

    result = _deep_service(manager, source_observer=current.get).retrieve(_REQUEST)

    assert [chunk.id for chunk in result.chunks] == ["b"]
    assert result.freshness is not None
    assert [v.source_id for v in result.freshness.verdicts] == ["docs/b.md"]
    assert result.freshness.stale_source_ids == ()
