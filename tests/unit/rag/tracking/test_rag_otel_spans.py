# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Sequence

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument

from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.instrumentation_span_attributes import (
    INTERGRAX_ATTEMPT_ID_ATTR,
    INTERGRAX_EXECUTION_ID_ATTR,
    INTERGRAX_RUN_ID_ATTR,
)
from intergrax.rag.tracking.rag_spans import (
    RAG_OTEL_SPAN_NAMES,
    RAG_OTEL_TRACER_NAME,
    is_rag_otel_spans_enabled,
    rag_span,
    set_rag_otel_spans_enabled,
)

pytestmark = pytest.mark.gate


@pytest.fixture(autouse=True)
def _enable_rag_spans() -> Generator[None, None, None]:
    set_rag_otel_spans_enabled(True)
    yield
    set_rag_otel_spans_enabled(False)


class _StubRetriever(BaseRetriever):
    @classmethod
    def name(cls) -> str:
        return "stub"

    def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:
        return [
            RetrieverCandidate(
                id="c1",
                content=f"answer for {query.query_text}",
                metadata={},
                score=0.9,
            )
        ]


class _StubRetrieverManager(BaseRetrieverManager):
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
        return _StubRetriever().retrieve(
            RetrieverQuery(
                query_text=query_text,
                query_embedding=None,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        )

    def retrieve_query(self, query: RetrieverQuery, retriever_id: str) -> List[RetrieverCandidate]:
        return _StubRetriever().retrieve(query)


class _StubLoader:
    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {"document_id": "otel-doc", "root_document_id": "otel-doc"},
                    "scope": {"tenant_id": "otel-test"},
                    "content": "chunk body",
                    "metadata": {"source": source},
                    "provenance": {"source_kind": "test", "source_id": source},
                }
            )
        ]


class _StubSplitter:
    def split_documents(self, docs, strategy_id=None):
        return list(docs)


class _StubEmbedding:
    def embed_documents(self, docs):
        from types import SimpleNamespace

        return SimpleNamespace(documents=docs, embeddings=[[0.1, 0.2]] * len(docs))

    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


class _StubVectorstore:
    def add_documents(self, **kwargs: object) -> list[str]:
        return ["id-0"]

    def add_records(self, records, *, scope):
        return [record.vector_id for record in records]

    def list_source_record_ids(
        self,
        *,
        source_id: str,
        scope: object,
        root_document_id: str | None = None,
    ) -> tuple[str, ...]:
        del source_id, scope, root_document_id
        return ()

    def count(self, *, scope: object) -> int:
        del scope
        return 0


def test_rag_otel_span_registry_is_stable() -> None:
    assert RAG_OTEL_TRACER_NAME == "intergrax.rag"
    assert "rag.retrieve" in RAG_OTEL_SPAN_NAMES
    assert "rag.ingest.index" in RAG_OTEL_SPAN_NAMES


def test_rag_otel_spans_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import intergrax.rag.tracking.rag_spans as rag_spans_module

    rag_spans_module._rag_otel_spans_enabled_override = None
    monkeypatch.delenv("INTERGRAX_RAG_OTEL_SPANS_ENABLED", raising=False)
    assert is_rag_otel_spans_enabled() is True

    monkeypatch.setenv("INTERGRAX_RAG_OTEL_SPANS_ENABLED", "false")
    assert is_rag_otel_spans_enabled() is False


def test_retrieval_service_emits_otel_spans(span_exporter: object) -> None:
    service = RetrievalService(
        retriever_manager=_StubRetrieverManager(),
        profile=RagProfile(enable_rerank=False),
    )
    result = service.retrieve(RetrievalRequest(query="hello world"))

    assert result.used is True
    names = [span.name for span in span_exporter.get_finished_spans()]
    assert "rag.retrieve" in names
    assert "rag.retrieve.single_pass" in names


def test_ingest_pipeline_emits_stage_otel_spans(
    span_exporter: object,
    tmp_path: Path,
) -> None:
    source = tmp_path / "doc.txt"
    source.write_text("hello ingest", encoding="utf-8")

    pipeline = IngestPipeline(
        loader=_StubLoader(),
        splitter=_StubSplitter(),
        embedding_manager=_StubEmbedding(),
        vectorstore=_StubVectorstore(),
    )
    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={"tenant_id": "otel-test"},
        )
    )

    assert result.used is True
    names = [span.name for span in span_exporter.get_finished_spans()]
    assert names == [
        "rag.ingest.load",
        "rag.ingest.chunk",
        "rag.ingest.index",
        "rag.ingest",
    ]


def test_rag_span_correlates_active_execution_identity(span_exporter: object) -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        with rag_span(
            "rag.retrieve",
            attributes={
                "rag.query": "must not export",
                "rag.query.length": 4,
            },
        ):
            pass
    finally:
        reset_active_execution_identity(token)

    span = next(span for span in span_exporter.get_finished_spans() if span.name == "rag.retrieve")
    attributes = dict(span.attributes)
    assert attributes[INTERGRAX_RUN_ID_ATTR] == str(run_id)
    assert attributes[INTERGRAX_ATTEMPT_ID_ATTR] == str(attempt_id)
    assert attributes[INTERGRAX_EXECUTION_ID_ATTR] == str(execution_id)
    assert attributes["rag.query.length"] == 4
    assert "rag.query" not in attributes


def test_rag_span_instrumentation_failure_does_not_break_business(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken_get_tracer(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("otel provider unavailable")

    monkeypatch.setattr("opentelemetry.trace.get_tracer", _broken_get_tracer)

    outcome = "ok"
    with rag_span("rag.retrieve"):
        outcome = "still-ok"
    assert outcome == "still-ok"


def test_rag_span_business_exception_propagates(span_exporter: object) -> None:
    with pytest.raises(ValueError, match="business failure"):
        with rag_span("rag.retrieve"):
            raise ValueError("business failure")

    span = next(span for span in span_exporter.get_finished_spans() if span.name == "rag.retrieve")
    assert span.status.status_code.name == "ERROR"
