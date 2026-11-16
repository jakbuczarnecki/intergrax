# © Artur Czarnecki. All rights reserved.
# pyright: reportArgumentType=false, reportCallIssue=false

"""PDC-D2 R005 end to end: ingest stamp -> RetrievalService verdict -> existing reindex hook."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.governance.embedding_version_policy import (
    ReindexQueueRequest,
    clear_reindex_queue_hooks,
    register_reindex_queue_hook,
)
from intergrax.rag.governance.source_freshness import (
    SOURCE_MODIFIED_AT_METADATA_KEY,
    SOURCE_REVISION_METADATA_KEY,
    SourceFreshnessObservation,
    SourceFreshnessStatus,
    observe_file_source,
)
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever import RetrievalHit
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _isolated_reindex_hooks() -> Any:
    clear_reindex_queue_hooks()
    yield
    clear_reindex_queue_hooks()


class _FixedHitsManager(BaseRetrieverManager):
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self._hits = hits

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
        return list(self._hits)

    def retrieve_query(self, query: Any, retriever_id: str) -> list[Any]:
        return list(self._hits)


def _service(hits: list[RetrievalHit], **kwargs: Any) -> RetrievalService:
    return RetrievalService(
        retriever_manager=_FixedHitsManager(hits),
        profile=RagProfile(enable_rerank=False),
        **kwargs,
    )


def _hit(
    document_id: str,
    source: str,
    *,
    revision: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RetrievalHit:
    """Hit whose document carries the canonical ``provenance.source_revision`` stamp."""
    provenance: dict[str, Any] = {"source_kind": "test", "source_id": source}
    if revision is not None:
        provenance["source_revision"] = revision
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": document_id, "root_document_id": document_id},
            "scope": {"tenant_id": "tenant-a", "namespace": "namespace-a"},
            "content": f"content of {document_id}",
            "metadata": {"source_path": source, **(metadata or {})},
            "provenance": provenance,
        }
    )
    return RetrievalHit(document=document, score=0.9, rank=0, channel="dense", vector_id=f"vec-{document_id}")


# ------------------------------------------------------------------ retrieval side


def test_retrieval_without_observer_keeps_todays_behavior() -> None:
    queued: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(queued.append)
    hits = [_hit("a", "docs/a.md", revision="rev-1")]

    result = _service(hits).retrieve(RetrievalRequest(query="policy"))

    assert result.used is True
    assert result.reason == "ok"
    assert [chunk.id for chunk in result.chunks] == ["a"]
    assert result.freshness is None
    assert queued == []


def test_missing_observation_is_unknown_never_fresh_and_chunks_are_kept() -> None:
    queued: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(queued.append)
    hits = [_hit("a", "docs/a.md", revision="rev-1")]

    result = _service(hits, source_observer=lambda source_id: None).retrieve(RetrievalRequest(query="policy"))

    assert result.used is True
    assert [chunk.id for chunk in result.chunks] == ["a"]
    assert result.freshness is not None
    verdict = result.freshness.verdicts[0]
    assert verdict.status is SourceFreshnessStatus.UNKNOWN
    assert verdict.reason == "source_not_observed"
    assert result.freshness.stale_source_ids == ()
    assert queued == []


def test_failing_observer_degrades_to_unknown_not_fresh() -> None:
    def broken(source_id: str) -> SourceFreshnessObservation | None:
        raise RuntimeError("source system unavailable")

    hits = [_hit("a", "docs/a.md", revision="rev-1")]
    result = _service(hits, source_observer=broken).retrieve(RetrievalRequest(query="policy"))

    assert result.used is True
    assert result.freshness is not None
    assert result.freshness.verdicts[0].status is SourceFreshnessStatus.UNKNOWN


def test_stale_source_gets_a_typed_verdict_and_reindex_through_the_existing_hook() -> None:
    queued: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(queued.append)
    hits = [
        _hit("a", "docs/a.md", revision="rev-1"),
        _hit("b", "docs/b.md", revision="rev-4"),
    ]
    current = {
        "docs/a.md": SourceFreshnessObservation(source_id="docs/a.md", revision="rev-2"),
        "docs/b.md": SourceFreshnessObservation(source_id="docs/b.md", revision="rev-4"),
    }

    result = _service(hits, source_observer=current.get).retrieve(RetrievalRequest(query="policy"))

    assert result.used is True
    assert [chunk.id for chunk in result.chunks] == ["a", "b"]  # annotation only, nothing dropped
    assert result.freshness is not None
    by_source = {v.source_id: v for v in result.freshness.verdicts}
    assert by_source["docs/a.md"].status is SourceFreshnessStatus.STALE
    assert by_source["docs/b.md"].status is SourceFreshnessStatus.FRESH
    assert result.freshness.stale_source_ids == ("docs/a.md",)
    assert result.freshness.reindex_requested is True
    assert [(r.source_path, r.reason) for r in queued] == [("docs/a.md", "source_revision_stale")]


# ------------------------------------------------------------------ ingest -> retrieval


class _RecordingLoader:
    """Loader stub that honours ``call_custom_metadata`` like the real loaders."""

    def load_document(self, source: str, **kwargs: Any) -> list[KnowledgeDocument]:
        callback = kwargs.get("call_custom_metadata")
        metadata: dict[str, Any] = {"source": source}
        if callable(callback):
            metadata.update(callback(None, source))
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {"document_id": "doc-1", "root_document_id": "doc-1"},
                    "scope": {"tenant_id": "tenant.test"},
                    "content": Path(source).read_text(encoding="utf-8"),
                    "metadata": metadata,
                    "provenance": {"source_kind": "test", "source_id": source},
                }
            )
        ]


class _StubSplitter:
    def split_documents(self, docs: Any, strategy_id: Any = None) -> list[Any]:
        return list(docs)


class _StubEmbedding:
    def embed_texts(self, texts: Any) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class _CapturingVectorstore:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def add_records(self, records: Any, *, scope: Any = None) -> list[str]:
        self.records.extend(records)
        return [f"id-{i}" for i, _ in enumerate(records)]

    def list_source_record_ids(self, *, source_id: str, scope: object, root_document_id: Any = None) -> tuple[str, ...]:
        return ()

    def count(self, *, scope: object) -> int:
        return 0

    def delete(self, ids: Any, *, scope: Any = None) -> None:
        return None


def _ingest(source: Path, **extra_metadata: Any) -> _CapturingVectorstore:
    store = _CapturingVectorstore()
    pipeline = IngestPipeline(
        loader=_RecordingLoader(),
        splitter=_StubSplitter(),
        embedding_manager=_StubEmbedding(),
        vectorstore=store,
        profile=RagProfile(),
    )
    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={"tenant_id": "tenant.test", **extra_metadata},
        )
    )
    assert result.used is True, result.reason
    return store


def test_ingest_stamps_the_observed_source_revision_and_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "policy.md"
    source.write_text("version one", encoding="utf-8")

    store = _ingest(source)

    document = store.records[0].document
    metadata = document.metadata
    observed = observe_file_source(source)
    assert document.provenance.source_revision == observed.revision
    assert metadata[SOURCE_MODIFIED_AT_METADATA_KEY] == observed.modified_at.isoformat()  # type: ignore[union-attr]
    assert metadata["source_path"] == str(source)


def test_ingest_keeps_a_caller_supplied_revision(tmp_path: Path) -> None:
    source = tmp_path / "policy.md"
    source.write_text("version one", encoding="utf-8")

    store = _ingest(source, **{SOURCE_REVISION_METADATA_KEY: "git-abc123"})

    assert store.records[0].document.provenance.source_revision == "git-abc123"


def test_edit_after_ingest_is_reported_stale_end_to_end(tmp_path: Path) -> None:
    queued: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(queued.append)
    source = tmp_path / "policy.md"
    source.write_text("version one", encoding="utf-8")
    store = _ingest(source)
    indexed_metadata = dict(store.records[0].document.metadata)
    indexed_revision = store.records[0].document.provenance.source_revision

    def retrieve_indexed() -> Any:
        hit = _hit(
            "doc-1",
            str(source),
            revision=indexed_revision,
            metadata={k: v for k, v in indexed_metadata.items() if k != "source_path"},
        )
        service = _service([hit], source_observer=lambda sid: observe_file_source(sid))
        return service.retrieve(RetrievalRequest(query="policy"))

    unchanged = retrieve_indexed()
    assert unchanged.freshness is not None
    assert unchanged.freshness.verdicts[0].status is SourceFreshnessStatus.FRESH
    assert queued == []

    source.write_text("version two, edited after indexing", encoding="utf-8")
    later = os.stat(source).st_mtime_ns + 5_000_000_000
    os.utime(source, ns=(later, later))

    edited = retrieve_indexed()
    assert edited.used is True
    assert edited.freshness is not None
    assert edited.freshness.verdicts[0].status is SourceFreshnessStatus.STALE
    assert edited.freshness.reindex_requested is True
    assert [r.source_path for r in queued] == [str(source)]
