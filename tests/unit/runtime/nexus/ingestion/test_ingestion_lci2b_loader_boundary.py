# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.llm.messages import AttachmentRef
from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata
from intergrax.rag.document_loaders.documents_loader import DocumentsLoader
from intergrax.rag.document_loaders.pipeline.metadata_pipeline import MetadataPipeline
from intergrax.rag.document_loaders.pipeline.normalizer_pipeline import NormalizerPipeline
from intergrax.rag.document_loaders.registry.document_handler_registry import DocumentHandlerRegistry
from intergrax.runtime.nexus.ingestion.attachments import AttachmentResolver
from intergrax.runtime.nexus.ingestion.ingestion_service import AttachmentIngestionService


class _Resolver(AttachmentResolver):
    def __init__(self, path: Path) -> None:
        self._path = path

    async def resolve_to_path(self, attachment: AttachmentRef) -> Path:
        return self._path


class _PassthroughParser(BaseDocumentParser):
    @classmethod
    def parser_id(cls) -> str:
        return "tests.passthrough"

    def is_available(self) -> bool:
        return True

    def load(self, source: str):
        return [
            ParsedDocumentFragment(
                text="hello",
                metadata=build_loader_metadata(
                    source=source,
                    parser=self.parser_id(),
                    position=0,
                ),
            )
        ]


class _PassthroughHandler(BaseDocumentHandler):
    def supports(self, source: str) -> bool:
        return True

    def confidence(self, source: str) -> float:
        return 1.0

    def build_parsers(self):
        return [_PassthroughParser()]


class _NativeSplitter:
    def __init__(self) -> None:
        self.received: list[KnowledgeDocument] | None = None

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
async def test_attachment_ingestion_uses_real_loader_callback_and_scope(tmp_path: Path) -> None:
    attachment_path = tmp_path / "note.txt"
    attachment_path.write_text("hello", encoding="utf-8")

    registry = DocumentHandlerRegistry()
    registry.register(_PassthroughHandler())

    loader = DocumentsLoader(
        registry=registry,
        normalizer_pipeline=NormalizerPipeline(normalizers=[]),
        metadata_pipeline=MetadataPipeline(providers=[]),
    )
    splitter = _NativeSplitter()

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
    assert result[0].metadata.get("reason") != "ingestion_failed"
    assert splitter.received is not None
    assert isinstance(splitter.received[0], KnowledgeDocument)
    assert splitter.received[0].content == "hello"

    loaded = loader.load_document(
        str(attachment_path),
        tenant_id="tenant.test",
        namespace="workspace-1",
        use_default_metadata=False,
        call_custom_metadata=lambda doc, source: {
            "attachment_id": attachment.id,
            "session_id": "sess-1",
            "user_id": "user-1",
            "workspace_id": "workspace-1",
        },
    )
    assert loaded[0].scope.tenant_id == "tenant.test"
    assert loaded[0].scope.namespace == "workspace-1"
    assert "tenant_id" not in loaded[0].metadata
