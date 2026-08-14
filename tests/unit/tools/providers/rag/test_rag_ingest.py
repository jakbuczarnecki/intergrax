# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
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

    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeVectorstoreManager:
    def __init__(self) -> None:
        self.added: list[str] = []

    def add_documents(self, *, documents, embeddings, ids, base_metadata):
        self.added = list(ids)
        return ids

    def add_records(self, records, *, scope):
        self.added = [record.vector_id for record in records]
        return self.added

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


class FakeDocumentsLoader:
    def load_document(
        self,
        source: str,
        *,
        tenant_id: str,
        namespace=None,
        use_default_metadata=True,
        call_custom_metadata=None,
    ):
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {"document_id": "ingest-doc", "root_document_id": "ingest-doc"},
                    "scope": {"tenant_id": tenant_id, "namespace": namespace},
                    "content": "hello world",
                    "metadata": {
                        "integration_parser_trace": {
                            "parser_id": "docling.local",
                            "attempts": [{"parser_id": "docling.local", "status": "success"}],
                        }
                    },
                    "provenance": {"source_kind": "test", "source_id": source},
                }
            )
        ]


class FakeSplitter:
    def split_documents(self, docs, strategy_id=None):
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
