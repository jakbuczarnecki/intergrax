# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Sequence

import pytest
from opentelemetry import trace
from intergrax.knowledge.contracts import KnowledgeDocument
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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
from intergrax.rag.tracking.rag_spans import (
    RAG_OTEL_SPAN_NAMES,
    RAG_OTEL_TRACER_NAME,
    is_rag_otel_spans_enabled,
    set_rag_otel_spans_enabled,
)

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def span_exporter() -> Generator[InMemorySpanExporter, None, None]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter


@pytest.fixture(autouse=True)
def _enable_rag_spans(
    span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    span_exporter.clear()
    set_rag_otel_spans_enabled(True)
    yield
    set_rag_otel_spans_enabled(False)
    span_exporter.clear()


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


def test_retrieval_service_emits_otel_spans(span_exporter: InMemorySpanExporter) -> None:
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
    span_exporter: InMemorySpanExporter,
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
