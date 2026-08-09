# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.ingest.ingest_policy import (
    SEMANTIC_CHUNKING_SIZE_EXCEEDED_REASON,
    semantic_chunking_allowed,
)
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _LoaderReturnsLargeDoc:
    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        return [_document("word. " * 200, source)]


class _NoopSplitter:
    def split_documents(self, docs, strategy_id=None):
        raise AssertionError("splitter must not run when semantic size guard blocks ingest")


class _NoopEmbedding:
    def embed_texts(self, texts):
        raise AssertionError("embed must not run when semantic size guard blocks ingest")


class _NoopVectorstore:
    def add_records(self, records, *, scope=None) -> None:
        raise AssertionError("vectorstore must not run when semantic size guard blocks ingest")

    def list_source_record_ids(self, *, source_id: str, scope: object) -> tuple[str, ...]:
        return ()

    def count(self, *, scope: object) -> int:
        return 0

    def delete(self, ids, *, scope=None) -> None:
        del ids, scope


def test_semantic_chunking_allowed_rejects_oversized_document() -> None:
    docs = [_document("a" * 500, "source")]
    allowed, reason, chars = semantic_chunking_allowed(
        docs=docs,
        strategy_id="semantic",
        profile=RagProfile(semantic_chunking_max_chars=100),
    )
    assert allowed is False
    assert SEMANTIC_CHUNKING_SIZE_EXCEEDED_REASON in reason
    assert chars == 500


def test_semantic_chunking_allowed_skips_non_semantic_strategy() -> None:
    docs = [_document("a" * 500, "source")]
    allowed, reason, _ = semantic_chunking_allowed(
        docs=docs,
        strategy_id="langchain_recursive",
        profile=RagProfile(semantic_chunking_max_chars=100),
    )
    assert allowed is True
    assert reason == "ok"


def test_ingest_pipeline_rejects_oversized_semantic_doc_before_chunk(tmp_path: Path) -> None:
    source = tmp_path / "doc.txt"
    source.write_text("placeholder", encoding="utf-8")

    pipeline = IngestPipeline(
        loader=_LoaderReturnsLargeDoc(),
        splitter=_NoopSplitter(),
        embedding_manager=_NoopEmbedding(),
        vectorstore=_NoopVectorstore(),
        profile=RagProfile(
            chunking_strategy_id="semantic",
            semantic_chunking_max_chars=100,
            sync_ingest_max_bytes=1_000_000,
        ),
    )
    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={"tenant_id": "tenant.test"},
        )
    )

    assert result.used is False
    assert SEMANTIC_CHUNKING_SIZE_EXCEEDED_REASON in result.reason
    assert result.async_job_recommended is True


def _document(content: str, source_id: str) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "doc-1", "root_document_id": "doc-1"},
            "scope": {"tenant_id": "tenant.test"},
            "content": content,
            "metadata": {},
            "provenance": {"source_kind": "test", "source_id": source_id},
        }
    )
