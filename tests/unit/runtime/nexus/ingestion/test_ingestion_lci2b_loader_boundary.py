# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.llm.messages import AttachmentRef
from intergrax.runtime.nexus.ingestion.attachments import AttachmentResolver
from intergrax.runtime.nexus.ingestion.ingestion_service import AttachmentIngestionService


class _Resolver(AttachmentResolver):
    def __init__(self, path: Path) -> None:
        self._path = path

    async def resolve_to_path(self, attachment: AttachmentRef) -> Path:
        return self._path


class _TrackingLoader:
    def __init__(self, docs: list[KnowledgeDocument]) -> None:
        self.docs = docs
        self.kwargs: dict[str, object] = {}

    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        self.kwargs = dict(kwargs)
        return self.docs


class _LangChainSplitter:
    def __init__(self) -> None:
        self.received: list[Document] | None = None

    def split_documents(self, docs):
        self.received = list(docs)
        return docs


class _NoopEmbedding:
    def embed_documents(self, docs):
        return MagicMock(documents=docs, embeddings=[[0.0]])


class _NoopVectorstore:
    def add_documents(self, **kwargs: object) -> list[str]:
        return ["vec-0"]


@pytest.mark.asyncio
async def test_attachment_ingestion_passes_tenant_and_workspace_namespace(tmp_path: Path) -> None:
    attachment_path = tmp_path / "note.txt"
    attachment_path.write_text("hello", encoding="utf-8")

    native_doc = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "docid1234567890ab",
                "root_document_id": "docid1234567890ab",
            },
            "scope": {"tenant_id": "tenant.test", "namespace": "workspace-1"},
            "content": "hello",
            "metadata": {
                "source": str(attachment_path),
                "parser": "tests.dummy",
                "position": 0,
            },
            "provenance": {
                "source_kind": "file",
                "source_id": str(attachment_path),
                "provider_id": "tests.dummy",
            },
        }
    )

    loader = _TrackingLoader([native_doc])
    splitter = _LangChainSplitter()

    service = AttachmentIngestionService(
        resolver=_Resolver(attachment_path),
        embedding_manager=_NoopEmbedding(),
        vectorstore_manager=_NoopVectorstore(),
        loader=loader,
        splitter=splitter,
    )

    attachment = AttachmentRef(id="att-1", type="file", uri=f"file://{attachment_path}")

    result = await service.ingest_attachments_for_session(
        [attachment],
        session_id="sess-1",
        user_id="user-1",
        tenant_id="tenant.test",
        workspace_id="workspace-1",
    )

    assert result[0].num_chunks == 1
    assert loader.kwargs["tenant_id"] == "tenant.test"
    assert loader.kwargs["namespace"] == "workspace-1"
    assert splitter.received is not None
    assert isinstance(splitter.received[0], Document)
    assert splitter.received[0].page_content == "hello"
