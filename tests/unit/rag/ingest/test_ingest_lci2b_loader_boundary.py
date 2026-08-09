# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
import pytest
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.knowledge.contracts.document import RESERVED_METADATA_KEYS
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    attach_parser_native_handle,
)
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile


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
    def embed_texts(self, texts):
        return [[0.0] for _ in texts]

    def embed_documents(self, docs):
        raise AssertionError("native ingest must not call embed_documents")


class _NoopVectorstore:
    def __init__(self) -> None:
        self.received: dict[str, object] | None = None

    def add_records(self, records, *, scope=None) -> list[str]:
        self.received = {"records": records, "scope": scope}
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

    class _CapturingContextual:
        def __init__(self) -> None:
            self.documents: list[KnowledgeDocument] | None = None
            self.chunks: list[KnowledgeDocument] | None = None

        def enrich(self, documents, chunks):
            self.documents = list(documents)
            self.chunks = list(chunks)
            return list(chunks)

    contextual = _CapturingContextual()
    pipeline = IngestPipeline(
        loader=_NativeLoader(),
        splitter=splitter,
        embedding_manager=_NoopEmbedding(),
        vectorstore=_NoopVectorstore(),
        profile=RagProfile(contextual_enrich="on"),
        contextual_enricher=contextual,
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
    assert contextual.documents is not None
    assert contextual.chunks is not None
    assert all(isinstance(doc, KnowledgeDocument) for doc in contextual.documents)
    assert all(isinstance(chunk, KnowledgeDocument) for chunk in contextual.chunks)


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

    class _NativeContextual:
        def enrich(self, documents, chunks):
            assert all(isinstance(doc, KnowledgeDocument) for doc in documents)
            assert all(isinstance(chunk, KnowledgeDocument) for chunk in chunks)
            payload = chunks[0].model_dump(mode="json")
            payload["content"] = "Native context\n\n" + chunks[0].content
            payload["metadata"]["contextual_enrich"] = True
            return [KnowledgeDocument.model_validate(payload)]

    class _CapturingEmbedding:
        def __init__(self) -> None:
            self.received: list[str] | None = None

        def embed_texts(self, texts):
            self.received = list(texts)
            return [[0.0] for _ in texts]

        def embed_documents(self, docs):
            raise AssertionError("native ingest must not call embed_documents")

    embedding = _CapturingEmbedding()
    splitter = _ReturningSplitter()
    vectorstore = _NoopVectorstore()
    contextual = _NativeContextual()

    pipeline = IngestPipeline(
        loader=_NativeLoader(),
        splitter=splitter,
        embedding_manager=embedding,
        vectorstore=vectorstore,
        profile=RagProfile(contextual_enrich="on"),
        contextual_enricher=contextual,
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
    assert embedding.received == ["Native context\n\nhello"]
    assert vectorstore.received is not None
    record = vectorstore.received["records"][0]
    assert isinstance(record.document, KnowledgeDocument)
    assert record.document.content == "Native context\n\nhello"
    assert DocumentMetadataKey.DOCLING_DOCUMENT_META.value not in record.document.metadata


def test_ingest_pipeline_isolates_reserved_metadata_at_native_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "managed.txt"
    source.write_text("hello", encoding="utf-8")
    base_metadata = {
        "tenant_id": "tenant.test",
        "namespace": "managed",
        "source_id": "managed-source-id",
        "document_id": "managed-document-id",
        "content_hash": "sha256:managed",
        "workspace_id": "workspace-1",
        "operation_id": "operation-1",
        "source_path": str(source),
        "file_name": "managed.txt",
    }
    original_metadata = dict(base_metadata)

    native_doc = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "native-doc-id",
                "root_document_id": "native-doc-id",
            },
            "scope": {"tenant_id": "tenant.test", "namespace": "managed"},
            "content": "hello",
            "metadata": {"parser": "tests.boundary"},
            "provenance": {
                "source_kind": "file",
                "source_id": str(source),
            },
        }
    )

    class _BoundaryLoader:
        def __init__(self) -> None:
            self.custom_metadata: dict[str, object] | None = None

        def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
            callback = kwargs["call_custom_metadata"]
            assert callable(callback)
            custom_metadata = callback(native_doc, source)
            self.custom_metadata = dict(custom_metadata)
            payload = native_doc.model_dump(mode="python")
            payload["metadata"] = {
                **dict(native_doc.metadata),
                **self.custom_metadata,
            }
            return [KnowledgeDocument.model_validate(payload)]

    class _CapturingSplitter:
        def __init__(self) -> None:
            self.received: list[KnowledgeDocument] | None = None

        def split_documents(self, docs, strategy_id=None):
            self.received = list(docs)
            return docs

    class _CapturingEmbedding:
        def __init__(self) -> None:
            self.received: list[str] | None = None

        def embed_texts(self, texts):
            self.received = list(texts)
            return [[0.0] for _ in texts]

        def embed_documents(self, docs):
            raise AssertionError("native ingest must not call embed_documents")

    class _CapturingVectorstore:
        def __init__(self) -> None:
            self.received: dict[str, object] | None = None

        def add_records(self, records, *, scope=None) -> list[str]:
            self.received = {"records": records, "scope": scope}
            return ["id-0"]

    loader = _BoundaryLoader()
    splitter = _CapturingSplitter()
    embedding = _CapturingEmbedding()
    vectorstore = _CapturingVectorstore()
    pipeline = IngestPipeline(
        loader=loader,
        splitter=splitter,
        embedding_manager=embedding,
        vectorstore=vectorstore,
    )

    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata=base_metadata,
            workspace_id="workspace-1",
        )
    )

    assert result.used is True
    assert result.reason == "ok"
    assert loader.custom_metadata is not None
    assert RESERVED_METADATA_KEYS.isdisjoint(loader.custom_metadata)
    assert loader.custom_metadata == {
        "operation_id": "operation-1",
        "source_path": str(source),
        "file_name": "managed.txt",
    }
    assert splitter.received is not None
    assert embedding.received is not None
    assert vectorstore.received is not None

    assert embedding.received == ["hello"]
    native_metadata = splitter.received[0].metadata
    assert RESERVED_METADATA_KEYS.isdisjoint(native_metadata)
    assert splitter.received[0].scope.tenant_id == "tenant.test"
    assert splitter.received[0].scope.namespace == "managed"
    assert splitter.received[0].scope.workspace_id == "workspace-1"
    assert vectorstore.received["scope"].tenant_id == "tenant.test"
    assert vectorstore.received["scope"].namespace == "managed"
    assert vectorstore.received["scope"].workspace_id == "workspace-1"
    assert "workspace_id" not in vectorstore.received["records"][0].document.metadata
    assert base_metadata == original_metadata


def test_ingest_pipeline_propagates_splitter_typeerror_without_retry(tmp_path: Path) -> None:
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

    class _FailingSplitter:
        def __init__(self) -> None:
            self.calls = 0

        def split_documents(self, docs, strategy_id=None):
            self.calls += 1
            raise TypeError("internal splitter failure")

    class _TrackingEmbedding:
        def __init__(self) -> None:
            self.called = False

        def embed_texts(self, texts):
            self.called = True
            return [[0.0] for _ in texts]

    class _TrackingVectorstore:
        def __init__(self) -> None:
            self.called = False

        def add_records(self, records, *, scope=None) -> list[str]:
            self.called = True
            return ["id-0"]

    splitter = _FailingSplitter()
    embedding = _TrackingEmbedding()
    vectorstore = _TrackingVectorstore()

    pipeline = IngestPipeline(
        loader=_NativeLoader(),
        splitter=splitter,
        embedding_manager=embedding,
        vectorstore=vectorstore,
    )

    with pytest.raises(TypeError, match="internal splitter failure"):
        pipeline.run(
            IngestRequest(
                source_path=str(source),
                base_metadata={"tenant_id": "tenant.test"},
                chunking_strategy_id="recursive",
            )
        )

    assert splitter.calls == 1
    assert embedding.called is False
    assert vectorstore.called is False
