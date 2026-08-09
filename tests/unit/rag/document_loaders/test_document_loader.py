# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

import pytest

from pydantic import ValidationError

from intergrax.knowledge.contracts import KnowledgeDocument, KnowledgeDocumentScope
from intergrax.knowledge.contracts.document import dump_knowledge_document
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    attach_parser_native_handle,
    copy_parser_runtime_state,
    get_parser_native_handle,
    to_legacy_rag_document,
)
from intergrax.rag.document_loaders.bootstrap.default_loader import create_default_normalizer_pipeline
from intergrax.rag.document_loaders.documents_loader import DocumentsLoader
from intergrax.rag.document_loaders.contracts.base_document_handler import (
    BaseDocumentHandler,
)
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.document_loaders.registry.document_handler_registry import (
    DocumentHandlerRegistry,
)


pytestmark = pytest.mark.unit

_TENANT = "tenant.test"
_SCOPE = KnowledgeDocumentScope(tenant_id=_TENANT)


def _sample_knowledge_doc(content: str = "A") -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "docid1234567890ab",
                "root_document_id": "docid1234567890ab",
            },
            "scope": {"tenant_id": _TENANT},
            "content": content,
            "metadata": {
                "source": "file.pdf",
                "parser": "tests.dummy",
                "position": 0,
            },
            "provenance": {
                "source_kind": "file",
                "source_id": "file.pdf",
                "provider_id": "tests.dummy",
            },
        }
    )


class _DummyHandler(BaseDocumentHandler):

    def __init__(self, docs: Sequence[KnowledgeDocument]):
        self.docs = list(docs)
        self.load_called = False
        self.last_scope: KnowledgeDocumentScope | None = None

    def supports(self, source: str) -> bool:
        return True

    def confidence(self, source: str) -> float:
        return 1.0

    def build_parsers(self):
        return []

    def load(self, source: str, *, scope: KnowledgeDocumentScope):
        self.load_called = True
        self.last_scope = scope
        result: list[KnowledgeDocument] = []
        for doc in self.docs:
            payload = doc.model_dump(mode="python")
            payload["scope"] = {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
            }
            result.append(copy_parser_runtime_state(doc, KnowledgeDocument.model_validate(payload)))
        return result


class _DummyMetadataPipeline:

    def __init__(self):
        self.called = False

    def enrich(self, docs, source):
        self.called = True
        return docs


def test_loader_calls_handler_load():

    docs = [_sample_knowledge_doc()]

    handler = _DummyHandler(docs)

    registry = DocumentHandlerRegistry()
    registry.register(handler)

    metadata_pipeline = _DummyMetadataPipeline()
    normalizer_pipeline = create_default_normalizer_pipeline()

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=metadata_pipeline,
        normalizer_pipeline=normalizer_pipeline,
    )

    result = loader.load_document("file.pdf", tenant_id=_TENANT)

    assert handler.load_called
    assert handler.last_scope == _SCOPE
    assert len(result) == 1
    assert isinstance(result[0], KnowledgeDocument)
    assert result[0].content == "A"


def test_loader_runs_metadata_pipeline():

    docs = [_sample_knowledge_doc()]

    handler = _DummyHandler(docs)

    registry = DocumentHandlerRegistry()
    registry.register(handler)

    metadata_pipeline = _DummyMetadataPipeline()
    normalizer_pipeline = create_default_normalizer_pipeline()

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=metadata_pipeline,
        normalizer_pipeline=normalizer_pipeline,
    )

    loader.load_document("file.pdf", tenant_id=_TENANT)

    assert metadata_pipeline.called


def test_loader_returns_empty_when_handler_returns_none():

    handler = _DummyHandler([])

    registry = DocumentHandlerRegistry()
    registry.register(handler)

    metadata_pipeline = _DummyMetadataPipeline()
    normalizer_pipeline = create_default_normalizer_pipeline()

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=metadata_pipeline,
        normalizer_pipeline=normalizer_pipeline,
    )

    result = loader.load_document("file.pdf", tenant_id=_TENANT)

    assert result == []


def test_loader_returns_empty_on_exception():

    class _FailingHandler(_DummyHandler):

        def load(self, source: str, *, scope: KnowledgeDocumentScope):
            raise RuntimeError("boom")

    handler = _FailingHandler([])

    registry = DocumentHandlerRegistry()
    registry.register(handler)

    metadata_pipeline = _DummyMetadataPipeline()
    normalizer_pipeline = create_default_normalizer_pipeline()

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=metadata_pipeline,
        normalizer_pipeline=normalizer_pipeline,
    )

    result = loader.load_document("file.pdf", tenant_id=_TENANT)

    assert result == []


def test_loader_invalid_tenant_raises():

    loader = DocumentsLoader(
        registry=DocumentHandlerRegistry(),
        metadata_pipeline=_DummyMetadataPipeline(),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    with pytest.raises(ValueError):
        loader.load_document("file.pdf", tenant_id="")


def test_loader_preserves_scope_through_native_pipelines():

    docs = [_sample_knowledge_doc()]

    handler = _DummyHandler(docs)
    registry = DocumentHandlerRegistry()
    registry.register(handler)

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=_DummyMetadataPipeline(),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    namespace = "workspace.ns"
    result = loader.load_document("file.pdf", tenant_id=_TENANT, namespace=namespace)

    assert result[0].scope.tenant_id == _TENANT
    assert result[0].scope.namespace == namespace
    assert result[0].identity.document_id == docs[0].identity.document_id
    assert result[0].provenance.source_kind == docs[0].provenance.source_kind


def test_loader_handler_output_passes_directly_to_normalizer_pipeline():

    docs = [_sample_knowledge_doc(content="A   B")]

    handler = _DummyHandler(docs)
    registry = DocumentHandlerRegistry()
    registry.register(handler)

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=_DummyMetadataPipeline(),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    result = loader.load_document("file.pdf", tenant_id=_TENANT)

    assert result[0].content == "A B"


def test_loader_normalized_output_passes_directly_to_metadata_pipeline(tmp_path):

    from intergrax.rag.document_loaders.metadata.default_metadata_provider import (
        DefaultMetadataProvider,
    )
    from intergrax.rag.document_loaders.pipeline.metadata_pipeline import MetadataPipeline

    source = tmp_path / "sample.pdf"
    source.write_text("pdf", encoding="utf-8")

    docs = [_sample_knowledge_doc()]

    handler = _DummyHandler(docs)
    registry = DocumentHandlerRegistry()
    registry.register(handler)

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=MetadataPipeline([DefaultMetadataProvider()]),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    result = loader.load_document(str(source), tenant_id=_TENANT)

    assert result[0].metadata[DocumentMetadataKey.SOURCE_NAME] == "sample.pdf"


def test_loader_does_not_use_compatibility_bridge(monkeypatch):

    docs = [_sample_knowledge_doc()]

    handler = _DummyHandler(docs)
    registry = DocumentHandlerRegistry()
    registry.register(handler)

    def _forbidden(*args, **kwargs):
        raise AssertionError("LangChain compatibility bridge must not be called")

    monkeypatch.setattr(
        "intergrax.compat.langchain.to_langchain_document",
        _forbidden,
    )
    monkeypatch.setattr(
        "intergrax.compat.langchain.from_langchain_document",
        _forbidden,
    )

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=_DummyMetadataPipeline(),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    result = loader.load_document("file.pdf", tenant_id=_TENANT)

    assert len(result) == 1
    assert isinstance(result[0], KnowledgeDocument)


def test_loader_custom_metadata_merged_and_validated():

    docs = [_sample_knowledge_doc()]

    handler = _DummyHandler(docs)
    registry = DocumentHandlerRegistry()
    registry.register(handler)

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=_DummyMetadataPipeline(),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    def _custom(doc: KnowledgeDocument, source: str):
        return {"custom_key": "custom_value"}

    result = loader.load_document(
        "file.pdf",
        tenant_id=_TENANT,
        call_custom_metadata=_custom,
    )

    assert result[0].metadata["custom_key"] == "custom_value"


def test_loader_rejects_reserved_custom_metadata():

    docs = [_sample_knowledge_doc()]

    handler = _DummyHandler(docs)
    registry = DocumentHandlerRegistry()
    registry.register(handler)

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=_DummyMetadataPipeline(),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    def _custom(doc: KnowledgeDocument, source: str):
        return {"tenant_id": "override"}

    with pytest.raises(ValidationError):
        loader.load_document(
            "file.pdf",
            tenant_id=_TENANT,
            call_custom_metadata=_custom,
        )


def test_loader_does_not_mutate_source_document():

    source_doc = _sample_knowledge_doc()
    original_metadata = dict(source_doc.metadata)

    handler = _DummyHandler([source_doc])
    registry = DocumentHandlerRegistry()
    registry.register(handler)

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=_DummyMetadataPipeline(),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    loader.load_document("file.pdf", tenant_id=_TENANT)

    assert dict(source_doc.metadata) == original_metadata


def test_loader_preserves_parser_runtime_state_through_pipelines():

    handle = object()
    source_doc = attach_parser_native_handle(_sample_knowledge_doc(), handle)

    handler = _DummyHandler([source_doc])
    registry = DocumentHandlerRegistry()
    registry.register(handler)

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=_DummyMetadataPipeline(),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    result = loader.load_document("file.pdf", tenant_id=_TENANT)

    assert get_parser_native_handle(result[0]) is handle
    assert DocumentMetadataKey.DOCLING_DOCUMENT_META.value not in result[0].metadata
    assert '"_docling_document"' not in dump_knowledge_document(result[0]).decode("utf-8")


def test_loader_preserves_parser_runtime_state_through_custom_metadata():

    handle = object()
    source_doc = attach_parser_native_handle(_sample_knowledge_doc(), handle)

    handler = _DummyHandler([source_doc])
    registry = DocumentHandlerRegistry()
    registry.register(handler)

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=_DummyMetadataPipeline(),
        normalizer_pipeline=create_default_normalizer_pipeline(),
    )

    def _custom(doc: KnowledgeDocument, source: str):
        return {"custom_key": "custom_value"}

    result = loader.load_document(
        "file.pdf",
        tenant_id=_TENANT,
        call_custom_metadata=_custom,
    )

    assert get_parser_native_handle(result[0]) is handle
    assert result[0].metadata["custom_key"] == "custom_value"


def test_to_legacy_rag_document_does_not_mutate_source():

    handle = object()
    source_doc = attach_parser_native_handle(_sample_knowledge_doc(), handle)
    original_metadata = dict(source_doc.metadata)

    to_legacy_rag_document(source_doc)

    assert dict(source_doc.metadata) == original_metadata
    assert DocumentMetadataKey.DOCLING_DOCUMENT_META.value not in source_doc.metadata
