# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.ingest.ingest_policy import (
    SEMANTIC_CHUNKING_SIZE_EXCEEDED_REASON,
    semantic_chunking_allowed,
)
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _LoaderReturnsLargeDoc:
    def load_document(self, source: str, **kwargs: object) -> list[Document]:
        return [Document(page_content="word. " * 200)]


class _NoopSplitter:
    def split_documents(self, docs, strategy_id=None):
        raise AssertionError("splitter must not run when semantic size guard blocks ingest")


class _NoopEmbedding:
    def embed_documents(self, docs):
        raise AssertionError("embed must not run when semantic size guard blocks ingest")


class _NoopVectorstore:
    def add_documents(self, **kwargs: object) -> None:
        raise AssertionError("vectorstore must not run when semantic size guard blocks ingest")


def test_semantic_chunking_allowed_rejects_oversized_document() -> None:
    docs = [Document(page_content="a" * 500)]
    allowed, reason, chars = semantic_chunking_allowed(
        docs=docs,
        strategy_id="semantic",
        profile=RagProfile(semantic_chunking_max_chars=100),
    )
    assert allowed is False
    assert SEMANTIC_CHUNKING_SIZE_EXCEEDED_REASON in reason
    assert chars == 500


def test_semantic_chunking_allowed_skips_non_semantic_strategy() -> None:
    docs = [Document(page_content="a" * 500)]
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
    result = pipeline.run(IngestRequest(source_path=str(source)))

    assert result.used is False
    assert SEMANTIC_CHUNKING_SIZE_EXCEEDED_REASON in result.reason
    assert result.async_job_recommended is True
