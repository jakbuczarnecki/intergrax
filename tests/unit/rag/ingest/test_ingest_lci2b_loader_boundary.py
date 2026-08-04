# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    attach_parser_native_handle,
)
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest


pytestmark = pytest.mark.unit


class _TrackingLoader:
    def __init__(self) -> None:
        self.called = False

    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        self.called = True
        return []


class _NativeSplitter:
    def __init__(self) -> None:
        self.received: list[KnowledgeDocument] | None = None

    def split_documents(self, docs, strategy_id=None):
        self.received = list(docs)
        return docs


class _NoopEmbedding:
    def embed_documents(self, docs):
        return MagicMock(documents=docs, embeddings=[[0.0]])


class _NoopVectorstore:
    def add_documents(self, **kwargs: object) -> list[str]:
        return ["id-0"]


def test_ingest_pipeline_missing_tenant_skips_loader(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("hello", encoding="utf-8")

    loader = _TrackingLoader()
    splitter = _NativeSplitter()

    pipeline = IngestPipeline(
        loader=loader,
        splitter=splitter,
        embedding_manager=_NoopEmbedding(),
        vectorstore=_NoopVectorstore(),
    )

    result = pipeline.run(IngestRequest(source_path=str(source)))

    assert result.used is False
    assert result.reason == "missing_tenant_id"
    assert loader.called is False
    assert splitter.received is None


def test_ingest_pipeline_passes_native_docs_to_splitter(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("hello", encoding="utf-8")

    native_doc = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "docid1234567890ab",
                "root_document_id": "docid1234567890ab",
            },
            "scope": {"tenant_id": "tenant.test"},
            "content": "hello",
            "metadata": {
                "source": str(source),
                "parser": "tests.dummy",
                "position": 0,
            },
            "provenance": {
                "source_kind": "file",
                "source_id": str(source),
                "provider_id": "tests.dummy",
            },
        }
    )

    class _NativeLoader:
        def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
            return [native_doc]

    splitter = _NativeSplitter()

    pipeline = IngestPipeline(
        loader=_NativeLoader(),
        splitter=splitter,
        embedding_manager=_NoopEmbedding(),
        vectorstore=_NoopVectorstore(),
    )

    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={"tenant_id": "tenant.test"},
        )
    )

    assert result.used is True
    assert splitter.received is not None
    assert len(splitter.received) == 1
    assert isinstance(splitter.received[0], KnowledgeDocument)
    assert splitter.received[0].content == "hello"


def test_ingest_pipeline_converts_chunks_to_legacy_after_splitter(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("hello", encoding="utf-8")

    handle = object()
    native_doc = attach_parser_native_handle(
        KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {
                    "document_id": "docid1234567890ab",
                    "root_document_id": "docid1234567890ab",
                },
                "scope": {"tenant_id": "tenant.test"},
                "content": "hello",
                "metadata": {
                    "source": str(source),
                    "parser": "tests.dummy",
                    "position": 0,
                },
                "provenance": {
                    "source_kind": "file",
                    "source_id": str(source),
                    "provider_id": "tests.dummy",
                },
            }
        ),
        handle,
    )

    class _NativeLoader:
        def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
            return [native_doc]

    class _ReturningSplitter:
        def __init__(self) -> None:
            self.received: list[KnowledgeDocument] | None = None

        def split_documents(self, docs, strategy_id=None):
            self.received = list(docs)
            return docs

    class _CapturingEmbedding:
        def __init__(self) -> None:
            self.received: list[Document] | None = None

        def embed_documents(self, docs):
            self.received = list(docs)
            return MagicMock(documents=docs, embeddings=[[0.0]])

    embedding = _CapturingEmbedding()
    splitter = _ReturningSplitter()

    pipeline = IngestPipeline(
        loader=_NativeLoader(),
        splitter=splitter,
        embedding_manager=embedding,
        vectorstore=_NoopVectorstore(),
    )

    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={"tenant_id": "tenant.test"},
        )
    )

    assert result.used is True
    assert splitter.received is not None
    assert isinstance(splitter.received[0], KnowledgeDocument)
    assert DocumentMetadataKey.DOCLING_DOCUMENT_META.value not in splitter.received[0].metadata
    assert embedding.received is not None
    assert isinstance(embedding.received[0], Document)
    legacy_meta = embedding.received[0].metadata
    assert legacy_meta[DocumentMetadataKey.DOCUMENT_ID.value] == "docid1234567890ab"
    assert DocumentMetadataKey.DOCLING_DOCUMENT_META.value not in legacy_meta
