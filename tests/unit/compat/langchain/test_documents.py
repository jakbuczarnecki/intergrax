# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for LangChain Document compatibility bridge."""

from __future__ import annotations

import importlib
import math
import sys
from types import ModuleType

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from intergrax.compat.langchain import (
    LangChainCompatibilityUnavailableError,
    LangChainDocumentBridgeError,
    from_langchain_document,
    to_langchain_document,
)
from intergrax.knowledge.contracts import (
    KnowledgeDocument,
    KnowledgeDocumentIdentity,
    KnowledgeDocumentProvenance,
    KnowledgeDocumentScope,
)


def _transport_metadata(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "tenant_id": "tenant-1",
        "source_kind": "file",
        "source_id": "abc123",
    }
    payload.update(overrides)
    return payload


def _lc_document(
    *,
    page_content: str = "Hello knowledge",
    document_id: str | None = "file:abc123",
    metadata: dict[str, object] | None = None,
) -> Document:
    return Document(
        id=document_id,
        page_content=page_content,
        metadata=_transport_metadata(**(metadata or {})),
    )


def _native_document(**overrides: object) -> KnowledgeDocument:
    payload: dict[str, object] = {
        "schema_version": 1,
        "identity": {
            "document_id": "file:abc123",
            "root_document_id": "file:abc123",
            "parent_document_id": None,
        },
        "scope": {"tenant_id": "tenant-1", "namespace": None, "workspace_id": None},
        "content": "Hello knowledge",
        "metadata": {},
        "provenance": {
            "source_kind": "file",
            "source_id": "abc123",
            "source_parent_id": None,
            "provider_id": None,
            "source_revision": None,
            "source_uri": None,
            "content_hash": None,
        },
    }
    payload.update(overrides)
    return KnowledgeDocument.model_validate(payload)


@pytest.mark.unit
def test_compat_package_import_without_eager_langchain_documents_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = importlib.import_module

    def blocked_import(name: str, package: str | None = None) -> ModuleType:
        if name == "langchain_core.documents" or name.startswith("langchain_core.documents."):
            raise ImportError("blocked for test")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked_import)
    module_name = "intergrax.compat.langchain"
    sys.modules.pop(module_name, None)
    sys.modules.pop("intergrax.compat.langchain.documents", None)

    imported = importlib.import_module(module_name)

    assert imported.from_langchain_document.__name__ == "from_langchain_document"
    assert imported.to_langchain_document.__name__ == "to_langchain_document"


@pytest.mark.unit
def test_conversion_without_langchain_core_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = importlib.import_module

    def blocked_import(name: str, package: str | None = None) -> ModuleType:
        if name == "langchain_core.documents":
            raise ImportError("blocked for test")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked_import)

    with pytest.raises(
        LangChainCompatibilityUnavailableError,
        match="langchain-core is required to use intergrax.compat.langchain",
    ):
        from_langchain_document(_lc_document())


@pytest.mark.unit
def test_from_langchain_source_document() -> None:
    document = _lc_document()
    native = from_langchain_document(document)
    assert native.identity.document_id == "file:abc123"
    assert native.identity.root_document_id == "file:abc123"
    assert native.identity.parent_document_id is None
    assert native.scope.tenant_id == "tenant-1"
    assert native.provenance.source_id == "abc123"


@pytest.mark.unit
def test_from_langchain_chunk_document() -> None:
    document = _lc_document(
        document_id="file:abc123#chunk-1",
        metadata={
            "root_document_id": "file:abc123",
            "parent_document_id": "file:abc123",
        },
    )
    native = from_langchain_document(document)
    assert native.identity.document_id == "file:abc123#chunk-1"
    assert native.identity.root_document_id == "file:abc123"
    assert native.identity.parent_document_id == "file:abc123"


@pytest.mark.unit
def test_from_langchain_subchunk_document() -> None:
    document = _lc_document(
        document_id="file:abc123#chunk-1#sub-1",
        metadata={
            "root_document_id": "file:abc123",
            "parent_document_id": "file:abc123#chunk-1",
        },
    )
    native = from_langchain_document(document)
    assert native.identity.parent_document_id == "file:abc123#chunk-1"


@pytest.mark.unit
def test_from_langchain_preserves_exact_content() -> None:
    content = "  padded content \n\t"
    document = _lc_document(page_content=content)
    native = from_langchain_document(document)
    assert native.content == content


@pytest.mark.unit
def test_from_langchain_maps_document_id_field() -> None:
    document = Document(
        id="file:from-id",
        page_content="Hello knowledge",
        metadata=_transport_metadata(),
    )
    native = from_langchain_document(document)
    assert native.identity.document_id == "file:from-id"


@pytest.mark.unit
def test_from_langchain_explicit_document_id_when_langchain_id_missing() -> None:
    document = Document(
        id=None,
        page_content="Hello knowledge",
        metadata=_transport_metadata(),
    )
    native = from_langchain_document(document, document_id="file:explicit")
    assert native.identity.document_id == "file:explicit"


@pytest.mark.unit
def test_from_langchain_metadata_document_id_when_other_carriers_missing() -> None:
    document = Document(
        id=None,
        page_content="Hello knowledge",
        metadata=_transport_metadata(document_id="file:metadata"),
    )
    native = from_langchain_document(document)
    assert native.identity.document_id == "file:metadata"


@pytest.mark.unit
def test_from_langchain_rejects_missing_document_id() -> None:
    document = Document(
        id=None,
        page_content="Hello knowledge",
        metadata=_transport_metadata(),
    )
    with pytest.raises(LangChainDocumentBridgeError, match="document_id is required"):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_rejects_multiple_document_id_carriers_even_when_equal() -> None:
    document = Document(
        id="file:abc123",
        page_content="Hello knowledge",
        metadata=_transport_metadata(document_id="file:abc123"),
    )
    with pytest.raises(LangChainDocumentBridgeError, match="exactly one source"):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_rejects_missing_tenant() -> None:
    metadata = _transport_metadata()
    metadata.pop("tenant_id")
    document = Document(id="file:abc123", page_content="Hello knowledge", metadata=metadata)
    with pytest.raises(LangChainDocumentBridgeError, match="tenant_id"):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_rejects_missing_source_kind() -> None:
    metadata = _transport_metadata()
    metadata.pop("source_kind")
    document = Document(id="file:abc123", page_content="Hello knowledge", metadata=metadata)
    with pytest.raises(LangChainDocumentBridgeError, match="source_kind"):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_rejects_missing_source_id_and_source() -> None:
    metadata = _transport_metadata()
    metadata.pop("source_id")
    document = Document(id="file:abc123", page_content="Hello knowledge", metadata=metadata)
    with pytest.raises(LangChainDocumentBridgeError, match="source_id"):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_accepts_legacy_source_and_preserves_it() -> None:
    metadata = _transport_metadata()
    metadata.pop("source_id")
    metadata["source"] = "abc123"
    document = Document(id="file:abc123", page_content="Hello knowledge", metadata=metadata)
    native = from_langchain_document(document)
    assert native.provenance.source_id == "abc123"
    assert native.metadata["source"] == "abc123"


@pytest.mark.unit
def test_from_langchain_accepts_matching_source_id_and_legacy_source() -> None:
    document = _lc_document(metadata={"source": "abc123"})
    native = from_langchain_document(document)
    assert native.provenance.source_id == "abc123"
    assert native.metadata["source"] == "abc123"


@pytest.mark.unit
def test_from_langchain_rejects_conflicting_source_id_and_legacy_source() -> None:
    document = _lc_document(metadata={"source": "legacy", "source_id": "canonical"})
    with pytest.raises(LangChainDocumentBridgeError, match="conflict"):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_preserves_optional_provenance_fields() -> None:
    document = _lc_document(
        metadata={
            "source_parent_id": "parent-src",
            "provider_id": "provider-1",
            "source_revision": "rev-1",
            "source_uri": "https://example.test/item",
            "content_hash": "sha256:deadbeef",
            "namespace": "kb",
        }
    )
    native = from_langchain_document(document)
    assert native.scope.namespace == "kb"
    assert native.provenance.source_parent_id == "parent-src"
    assert native.provenance.provider_id == "provider-1"
    assert native.provenance.source_revision == "rev-1"
    assert native.provenance.source_uri == "https://example.test/item"
    assert native.provenance.content_hash == "sha256:deadbeef"


@pytest.mark.unit
def test_from_langchain_maps_workspace_to_native_scope() -> None:
    native = from_langchain_document(
        _lc_document(metadata={"namespace": "kb", "workspace_id": "workspace-a"})
    )

    assert native.scope.namespace == "kb"
    assert native.scope.workspace_id == "workspace-a"
    assert "workspace_id" not in native.metadata


@pytest.mark.unit
def test_from_langchain_preserves_unknown_nested_metadata() -> None:
    document = _lc_document(metadata={"details": {"items": [1, {"tenant_id": "shadow"}]}})
    native = from_langchain_document(document)
    assert native.metadata == {"details": {"items": [1, {"tenant_id": "shadow"}]}}


@pytest.mark.unit
def test_from_langchain_accepts_schema_version_one_and_strips_it() -> None:
    document = _lc_document(metadata={"schema_version": 1})
    native = from_langchain_document(document)
    assert native.schema_version == 1
    assert "schema_version" not in native.metadata


@pytest.mark.unit
def test_from_langchain_rejects_invalid_schema_version() -> None:
    document = _lc_document(metadata={"schema_version": 2})
    with pytest.raises(LangChainDocumentBridgeError, match="schema_version"):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_rejects_unsafe_source_uri() -> None:
    document = _lc_document(metadata={"source_uri": "https://user:pass@example.test/item"})
    with pytest.raises(ValidationError):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_rejects_secret_metadata() -> None:
    document = _lc_document(metadata={"token": "secret"})
    with pytest.raises(ValidationError):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_rejects_non_json_metadata() -> None:
    document = _lc_document(metadata={"blob": b"raw"})
    with pytest.raises(ValidationError):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_does_not_mutate_input_metadata() -> None:
    metadata = _transport_metadata(details={"count": 1})
    original = dict(metadata)
    document = Document(id="file:abc123", page_content="Hello knowledge", metadata=metadata)
    from_langchain_document(document)
    assert metadata == original


@pytest.mark.unit
def test_from_langchain_rejects_non_document_type() -> None:
    with pytest.raises(TypeError, match="langchain_core.documents.Document"):
        from_langchain_document({"page_content": "x", "metadata": {}})


@pytest.mark.unit
def test_to_langchain_source_document() -> None:
    native = _native_document()
    converted = to_langchain_document(native)
    assert converted.id == "file:abc123"
    assert converted.page_content == "Hello knowledge"
    assert converted.metadata["tenant_id"] == "tenant-1"
    assert converted.metadata["source_kind"] == "file"
    assert converted.metadata["source_id"] == "abc123"
    assert "document_id" not in converted.metadata
    assert "schema_version" not in converted.metadata


@pytest.mark.unit
def test_to_langchain_derivative_document() -> None:
    native = _native_document(
        identity={
            "document_id": "file:abc123#chunk-1",
            "root_document_id": "file:abc123",
            "parent_document_id": "file:abc123",
        }
    )
    converted = to_langchain_document(native)
    assert converted.id == "file:abc123#chunk-1"
    assert converted.metadata["root_document_id"] == "file:abc123"
    assert converted.metadata["parent_document_id"] == "file:abc123"


@pytest.mark.unit
def test_to_langchain_skips_optional_none_fields() -> None:
    native = _native_document()
    converted = to_langchain_document(native)
    assert "parent_document_id" not in converted.metadata
    assert "namespace" not in converted.metadata
    assert "workspace_id" not in converted.metadata
    assert "source_parent_id" not in converted.metadata


@pytest.mark.unit
def test_to_langchain_preserves_unknown_metadata() -> None:
    native = _native_document(metadata={"details": {"items": [True, 1]}})
    converted = to_langchain_document(native)
    assert converted.metadata["details"] == {"items": [True, 1]}


@pytest.mark.unit
def test_to_langchain_maps_workspace_to_transport_metadata() -> None:
    native = _native_document(
        scope={
            "tenant_id": "tenant-1",
            "namespace": "kb",
            "workspace_id": "workspace-a",
        }
    )

    converted = to_langchain_document(native)

    assert converted.metadata["workspace_id"] == "workspace-a"


@pytest.mark.unit
def test_to_langchain_does_not_leak_frozen_metadata_types() -> None:
    native = _native_document(metadata={"nested": {"items": [1, 2]}})
    converted = to_langchain_document(native)
    assert isinstance(converted.metadata, dict)
    assert isinstance(converted.metadata["nested"], dict)
    assert isinstance(converted.metadata["nested"]["items"], list)


@pytest.mark.unit
def test_to_langchain_preserves_matching_legacy_source() -> None:
    native = _native_document(metadata={"source": "abc123"})
    converted = to_langchain_document(native)
    assert converted.metadata["source"] == "abc123"
    assert converted.metadata["source_id"] == "abc123"


@pytest.mark.unit
def test_to_langchain_rejects_conflicting_legacy_source() -> None:
    native = _native_document(metadata={"source": "legacy"})
    with pytest.raises(LangChainDocumentBridgeError, match="conflict"):
        to_langchain_document(native)


@pytest.mark.unit
def test_to_langchain_rejects_malformed_constructed_model() -> None:
    malformed = KnowledgeDocument.model_construct(
        schema_version=1,
        identity=KnowledgeDocumentIdentity(
            document_id="file:abc123",
            root_document_id="file:abc123",
        ),
        scope=KnowledgeDocumentScope(tenant_id="tenant-1"),
        content="Hello knowledge",
        metadata={"token": "secret"},
        provenance=KnowledgeDocumentProvenance(
            source_kind="file",
            source_id="abc123",
        ),
    )
    with pytest.raises(ValidationError):
        to_langchain_document(malformed)


@pytest.mark.unit
def test_to_langchain_rejects_wrong_argument_type() -> None:
    with pytest.raises(TypeError, match="KnowledgeDocument"):
        to_langchain_document({"content": "x"})


@pytest.mark.unit
def test_round_trip_native_source_document() -> None:
    native = _native_document(
        content="Unicode: Zażółć 🚀",
        metadata={"nested": {"tenant_id": "shadow"}},
        scope={
            "tenant_id": "tenant-1",
            "namespace": "kb",
            "workspace_id": "workspace-a",
        },
        provenance={
            "source_kind": "file",
            "source_id": "abc123",
            "source_parent_id": None,
            "provider_id": "provider-1",
            "source_revision": "rev-1",
            "source_uri": "https://example.test/item",
            "content_hash": "sha256:deadbeef",
        },
    )
    restored = from_langchain_document(to_langchain_document(native))
    assert restored == native


@pytest.mark.unit
def test_round_trip_native_chunk_with_full_provenance() -> None:
    native = _native_document(
        identity={
            "document_id": "file:abc123#chunk-1",
            "root_document_id": "file:abc123",
            "parent_document_id": "file:abc123",
        },
        content="  padded chunk \n",
        metadata={"source": "abc123", "details": {"items": [{"x": 1}]}},
        scope={
            "tenant_id": "tenant-1",
            "namespace": "kb",
            "workspace_id": "workspace-a",
        },
        provenance={
            "source_kind": "file",
            "source_id": "abc123",
            "source_parent_id": "parent-src",
            "provider_id": "provider-1",
            "source_revision": "rev-1",
            "source_uri": "https://example.test/item",
            "content_hash": "sha256:deadbeef",
        },
    )
    restored = from_langchain_document(to_langchain_document(native))
    assert restored == native


@pytest.mark.unit
def test_round_trip_preserves_unicode_and_padding_content() -> None:
    native = _native_document(content="  Zażółć 🚀  ")
    restored = from_langchain_document(to_langchain_document(native))
    assert restored.content == "  Zażółć 🚀  "


@pytest.mark.unit
def test_round_trip_preserves_nested_metadata() -> None:
    native = _native_document(metadata={"nested": {"items": [1, {"flag": True}]}})
    restored = from_langchain_document(to_langchain_document(native))
    assert restored.metadata == native.metadata


@pytest.mark.unit
def test_from_langchain_rejects_non_mapping_metadata() -> None:
    document = Document.model_construct(
        id="file:abc123",
        page_content="Hello knowledge",
        metadata=[],
    )
    with pytest.raises(LangChainDocumentBridgeError, match="mapping"):
        from_langchain_document(document)


@pytest.mark.unit
def test_from_langchain_rejects_non_finite_metadata() -> None:
    document = _lc_document(metadata={"score": math.inf})
    with pytest.raises(ValidationError):
        from_langchain_document(document)
