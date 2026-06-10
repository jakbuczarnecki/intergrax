# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.ingest.ingest_policy import SYNC_INGEST_SIZE_EXCEEDED_REASON, sync_ingest_allowed
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _NoopLoader:
    def load_document(self, source: str, **kwargs: object) -> list:
        raise AssertionError("loader must not run when sync ingest is blocked")


class _NoopSplitter:
    def split_documents(self, docs, strategy_id=None):
        return docs


class _NoopEmbedding:
    def embed_documents(self, docs):
        raise AssertionError("embed must not run when sync ingest is blocked")


class _NoopVectorstore:
    def add_documents(self, **kwargs: object) -> None:
        raise AssertionError("vectorstore must not run when sync ingest is blocked")


def test_sync_ingest_allowed_rejects_oversized_file(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    source.write_bytes(b"x" * 128)

    allowed, reason, size = sync_ingest_allowed(
        path=source,
        profile=RagProfile(sync_ingest_max_bytes=64),
    )
    assert allowed is False
    assert SYNC_INGEST_SIZE_EXCEEDED_REASON in reason
    assert size == 128


def test_ingest_pipeline_rejects_oversized_file_before_load(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    source.write_bytes(b"x" * 256)

    pipeline = IngestPipeline(
        loader=_NoopLoader(),
        splitter=_NoopSplitter(),
        embedding_manager=_NoopEmbedding(),
        vectorstore=_NoopVectorstore(),
        profile=RagProfile(sync_ingest_max_bytes=128),
    )
    result = pipeline.run(IngestRequest(source_path=str(source)))

    assert result.used is False
    assert SYNC_INGEST_SIZE_EXCEEDED_REASON in result.reason
    assert result.file_size_bytes == 256
    assert result.async_job_recommended is True
