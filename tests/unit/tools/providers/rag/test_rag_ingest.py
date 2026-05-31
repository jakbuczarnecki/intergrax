# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from intergrax.tools.providers.rag.ingest_contracts import RagIngestInput
from intergrax.tools.providers.rag.ingest_service import perform_rag_ingest
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class FakeEmbeddingManager:
    class _Result:
        def __init__(self, docs, embeddings):
            self.documents = docs
            self.embeddings = embeddings

    def embed_documents(self, docs):
        return self._Result(docs, [[0.1, 0.2] for _ in docs])


class FakeVectorstoreManager:
    def __init__(self) -> None:
        self.added: list[str] = []

    def add_documents(self, *, documents, embeddings, ids, base_metadata):
        self.added = list(ids)
        return ids


class FakeDocumentsLoader:
    def load_document(self, source: str, *, use_default_metadata=True, call_custom_metadata=None):
        from langchain_core.documents import Document

        return [
            Document(
                page_content="hello world",
                metadata={
                    "integration_parser_trace": {
                        "parser_id": "docling.local",
                        "attempts": [{"parser_id": "docling.local", "status": "success"}],
                    }
                },
            )
        ]


class FakeSplitter:
    def split_documents(self, docs):
        return docs


def test_rag_ingest_indexes_document(tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_text("x", encoding="utf-8")

    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager(),
        embedding_manager=FakeEmbeddingManager(),
        extras={
            "documents_loader": FakeDocumentsLoader(),
            "documents_splitter": FakeSplitter(),
        },
    )

    out = perform_rag_ingest(
        ctx,
        RagIngestInput(source_path=str(source), session_id="s1", tenant_id="t1"),
    )

    assert out.used is True
    assert out.num_chunks == 1
    assert out.parser_id == "docling.local"
    assert out.parser_trace.get("parser_id") == "docling.local"
    assert out.reason == "ok"


def test_rag_ingest_missing_vectorstore() -> None:
    ctx = ToolWiringContext(embedding_manager=FakeEmbeddingManager())
    out = perform_rag_ingest(ctx, RagIngestInput(source_path="/tmp/x.pdf"))
    assert out.used is False
    assert "not_configured" in out.reason
