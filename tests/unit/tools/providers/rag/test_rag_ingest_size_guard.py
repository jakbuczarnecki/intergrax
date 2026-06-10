# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.rag.ingest.ingest_policy import SYNC_INGEST_SIZE_EXCEEDED_REASON
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.tools.providers.rag.ingest_contracts import RagIngestInput
from intergrax.tools.providers.rag.ingest_service import perform_rag_ingest
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _LoaderShouldNotRun:
    def load_document(self, source: str, **kwargs: object) -> list:
        raise AssertionError("sync ingest should be blocked before loader")


class _Splitter:
    def split_documents(self, docs):
        return docs


class _Embedding:
    class _Result:
        def __init__(self, docs, embeddings):
            self.documents = docs
            self.embeddings = embeddings

    def embed_documents(self, docs):
        return self._Result(docs, [[0.1, 0.2] for _ in docs])


class _Vectorstore:
    def add_documents(self, **kwargs: object) -> list[str]:
        raise AssertionError("sync ingest should be blocked before vectorstore")


def test_rag_ingest_rejects_oversized_source(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    source.write_bytes(b"x" * 512)

    out = perform_rag_ingest(
        ToolWiringContext(
            vectorstore_manager=_Vectorstore(),
            embedding_manager=_Embedding(),
            rag_profile=RagProfile(sync_ingest_max_bytes=128),
            extras={
                "documents_loader": _LoaderShouldNotRun(),
                "documents_splitter": _Splitter(),
            },
        ),
        RagIngestInput(source_path=str(source), tenant_id="tenant-a"),
    )

    assert out.used is False
    assert SYNC_INGEST_SIZE_EXCEEDED_REASON in out.reason
    assert out.file_size_bytes == 512
    assert out.async_job_recommended is True
