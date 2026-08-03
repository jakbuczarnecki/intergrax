# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for native KnowledgeDocument contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from intergrax.knowledge.contracts import (
    KnowledgeDocument,
    KnowledgeDocumentIdentity,
    KnowledgeDocumentProvenance,
    KnowledgeDocumentScope,
    dump_knowledge_document,
    load_knowledge_document,
)


def _identity(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "document_id": "file:abc123",
        "root_document_id": "file:abc123",
        "parent_document_id": None,
    }
    payload.update(overrides)
    return payload


def _scope(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {"tenant_id": "tenant-1", "namespace": None}
    payload.update(overrides)
    return payload


def _provenance(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_kind": "file",
        "source_id": "abc123",
        "source_parent_id": None,
        "provider_id": None,
        "source_revision": None,
        "source_uri": None,
        "content_hash": None,
    }
    payload.update(overrides)
    return payload


def _document(**overrides: object) -> KnowledgeDocument:
    payload: dict[str, object] = {
        "schema_version": 1,
        "identity": _identity(),
        "scope": _scope(),
        "content": "Hello knowledge",
        "metadata": {},
        "provenance": _provenance(),
    }
    payload.update(overrides)
    return KnowledgeDocument.model_validate(payload)


def _document_omitting_metadata(**overrides: object) -> KnowledgeDocument:
    payload: dict[str, object] = {
        "schema_version": 1,
        "identity": _identity(),
        "scope": _scope(),
        "content": "Hello knowledge",
        "provenance": _provenance(),
    }
    payload.update(overrides)
    return KnowledgeDocument.model_validate(payload)


@pytest.mark.unit
def test_omitted_metadata_is_deeply_immutable() -> None:
    document = _document_omitting_metadata()
    assert document.metadata == {}
    with pytest.raises(TypeError, match="immutable"):
        document.metadata["token"] = "secret"  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        document.metadata.update({"token": "secret"})  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="immutable"):
        document.metadata.clear()  # type: ignore[attr-defined]


@pytest.mark.unit
def test_omitted_metadata_instances_are_not_shared() -> None:
    first = _document_omitting_metadata()
    second = _document_omitting_metadata()
    assert first.metadata == {}
    assert second.metadata == {}
    assert first.metadata is not second.metadata


@pytest.mark.unit
def test_metadata_public_read_contract() -> None:
    document = _document(metadata={"count": 1, "nested": {"items": [True]}})
    metadata = document.metadata
    assert isinstance(metadata, Mapping)
    assert metadata["count"] == 1
    assert metadata.get("missing") is None
    assert metadata.get("count") == 1
    assert list(metadata.keys()) == ["count", "nested"]
    assert list(metadata.values()) == [1, {"items": [True]}]
    assert list(metadata.items()) == [("count", 1), ("nested", {"items": [True]})]
    assert len(metadata) == 2
    assert "count" in metadata
    dumped = document.model_dump(mode="json")
    assert isinstance(dumped["metadata"], dict)
    assert dumped["metadata"] == {"count": 1, "nested": {"items": [True]}}


@pytest.mark.unit
def test_omitted_metadata_round_trip() -> None:
    document = _document_omitting_metadata()
    restored = load_knowledge_document(dump_knowledge_document(document))
    assert restored.metadata == {}
    assert restored == document


@pytest.mark.unit
def test_valid_source_document() -> None:
    document = _document()
    assert document.identity.parent_document_id is None
    assert document.identity.root_document_id == document.identity.document_id


@pytest.mark.unit
def test_valid_chunk_document() -> None:
    document = _document(
        identity=_identity(
            document_id="file:abc123:chunk:3",
            root_document_id="file:abc123",
            parent_document_id="file:abc123",
        )
    )
    assert document.identity.parent_document_id == "file:abc123"


@pytest.mark.unit
def test_valid_subchunk_document() -> None:
    document = _document(
        identity=_identity(
            document_id="file:abc123:chunk:3:sub:1",
            root_document_id="file:abc123",
            parent_document_id="file:abc123:chunk:3",
        )
    )
    assert document.identity.parent_document_id != document.identity.root_document_id


@pytest.mark.unit
def test_public_exports_available() -> None:
    assert KnowledgeDocument is not None
    assert KnowledgeDocumentIdentity is not None
    assert KnowledgeDocumentScope is not None
    assert KnowledgeDocumentProvenance is not None
    assert dump_knowledge_document is not None
    assert load_knowledge_document is not None


@pytest.mark.unit
def test_models_are_frozen_and_reject_unknown_fields() -> None:
    document = _document()
    with pytest.raises(ValidationError):
        KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": _identity(),
                "scope": _scope(),
                "content": "x",
                "metadata": {},
                "provenance": _provenance(),
                "extra": "nope",
            }
        )
    with pytest.raises(ValidationError):
        document.content = "changed"  # type: ignore[misc]


@pytest.mark.unit
def test_source_with_parent_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(identity=_identity(parent_document_id="file:parent"))


@pytest.mark.unit
def test_source_with_mismatched_root_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(identity=_identity(root_document_id="file:other"))


@pytest.mark.unit
def test_derivative_without_parent_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(
            identity=_identity(
                document_id="file:abc123:chunk:1",
                root_document_id="file:abc123",
            )
        )


@pytest.mark.unit
def test_derivative_with_root_equal_self_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(
            identity=_identity(
                document_id="file:abc123:chunk:1",
                root_document_id="file:abc123:chunk:1",
                parent_document_id="file:abc123",
            )
        )


@pytest.mark.unit
def test_parent_equal_self_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(
            identity=_identity(
                document_id="file:abc123:chunk:1",
                root_document_id="file:abc123",
                parent_document_id="file:abc123:chunk:1",
            )
        )


@pytest.mark.unit
def test_ids_are_not_auto_generated() -> None:
    payload = {
        "schema_version": 1,
        "identity": {"root_document_id": "file:abc123", "parent_document_id": None},
        "scope": _scope(),
        "content": "x",
        "metadata": {},
        "provenance": _provenance(),
    }
    with pytest.raises(ValidationError):
        KnowledgeDocument.model_validate(payload)


@pytest.mark.unit
def test_missing_tenant_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(scope={"namespace": None})


@pytest.mark.unit
def test_empty_tenant_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(scope=_scope(tenant_id=""))


@pytest.mark.unit
def test_whitespace_tenant_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(scope=_scope(tenant_id="   "))


@pytest.mark.unit
def test_default_tenant_not_auto_added() -> None:
    with pytest.raises(ValidationError):
        _document(scope={})


@pytest.mark.unit
def test_explicit_default_tenant_string_preserved() -> None:
    document = _document(scope=_scope(tenant_id="default"))
    assert document.scope.tenant_id == "default"


@pytest.mark.unit
def test_namespace_none_preserved() -> None:
    document = _document(scope=_scope(namespace=None))
    assert document.scope.namespace is None


@pytest.mark.unit
def test_empty_and_whitespace_content_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(content="")
    with pytest.raises(ValidationError):
        _document(content="   ")


@pytest.mark.unit
def test_content_is_not_trimmed() -> None:
    document = _document(content="  padded content  ")
    assert document.content == "  padded content  "


@pytest.mark.unit
def test_required_provenance_fields() -> None:
    with pytest.raises(ValidationError):
        _document(provenance={"source_id": "abc123"})
    with pytest.raises(ValidationError):
        _document(provenance={"source_kind": "file"})


@pytest.mark.unit
def test_source_parent_id_independent_from_document_parent() -> None:
    document = _document(
        identity=_identity(
            document_id="file:abc123:chunk:1",
            root_document_id="file:abc123",
            parent_document_id="file:abc123",
        ),
        provenance=_provenance(source_parent_id="folder-9"),
    )
    assert document.provenance.source_parent_id == "folder-9"
    assert document.identity.parent_document_id == "file:abc123"


@pytest.mark.unit
def test_unsafe_source_uri_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(provenance=_provenance(source_uri="https://user:pass@example.test/item"))


@pytest.mark.unit
def test_nested_json_metadata_accepted() -> None:
    document = _document(metadata={"nested": {"count": 2, "flags": [True, False]}})
    assert document.metadata["nested"] == {"count": 2, "flags": [True, False]}


@pytest.mark.unit
def test_forbidden_metadata_types_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(metadata={"blob": b"secret"})
    with pytest.raises(ValidationError):
        _document(metadata={"when": datetime.now(timezone.utc)})
    with pytest.raises(ValidationError):
        _document(metadata={"uid": uuid4()})
    with pytest.raises(ValidationError):

        class _Payload:
            pass

        _document(metadata={"obj": _Payload()})

    class _Mode(StrEnum):
        ON = "on"

    with pytest.raises(ValidationError):
        _document(metadata={"mode": _Mode.ON})


@pytest.mark.unit
def test_non_string_metadata_key_rejected() -> None:
    with pytest.raises(ValidationError):
        KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": _identity(),
                "scope": _scope(),
                "content": "x",
                "metadata": {1: "value"},
                "provenance": _provenance(),
            }
        )


@pytest.mark.unit
def test_secret_like_metadata_key_rejected_at_any_level() -> None:
    with pytest.raises(ValidationError):
        _document(metadata={"nested": {"api_key": "x"}})
    with pytest.raises(ValidationError):
        _document(metadata={"items": [{"password": "p"}]})


@pytest.mark.unit
def test_top_level_reserved_metadata_key_rejected_even_with_matching_value() -> None:
    document = _document()
    with pytest.raises(ValidationError):
        _document(metadata={"tenant_id": document.scope.tenant_id})
    with pytest.raises(ValidationError):
        _document(metadata={"document_id": document.identity.document_id})


@pytest.mark.unit
def test_nested_reserved_like_key_allowed() -> None:
    document = _document(metadata={"details": {"tenant_id": "shadow"}})
    assert document.metadata["details"] == {"tenant_id": "shadow"}


@pytest.mark.unit
def test_non_finite_metadata_floats_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(metadata={"score": math.nan})
    with pytest.raises(ValidationError):
        _document(metadata={"score": math.inf})


@pytest.mark.unit
def test_serialization_round_trip_and_determinism() -> None:
    document = _document(
        content="Unicode: Zażółć 🚀",
        metadata={"nested": {"tenant_id": "shadow"}},
        scope=_scope(namespace="kb"),
        provenance=_provenance(source_uri="https://example.test/item"),
    )
    first = dump_knowledge_document(document)
    second = dump_knowledge_document(document)
    assert first == second
    restored = load_knowledge_document(first)
    assert restored == document
    assert restored.content == document.content


@pytest.mark.unit
def test_unknown_schema_version_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(schema_version=2)
    with pytest.raises(ValidationError):
        _document(schema_version=True)
    with pytest.raises(ValidationError):
        _document(schema_version="1")


@pytest.mark.unit
def test_load_rejects_duplicate_json_keys() -> None:
    text = (
        '{"schema_version":1,"schema_version":2,"identity":{"document_id":"file:abc123",'
        '"root_document_id":"file:abc123","parent_document_id":null},"scope":{"tenant_id":"tenant-1",'
        '"namespace":null},"content":"Hello knowledge","metadata":{},'
        '"provenance":{"source_kind":"file","source_id":"abc123"}}'
    )
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        load_knowledge_document(text)


@pytest.mark.unit
def test_load_rejects_invalid_utf8_json_and_non_object_root() -> None:
    with pytest.raises(ValueError, match="valid UTF-8"):
        load_knowledge_document(b"\xff\xfe")
    with pytest.raises(ValueError, match="valid JSON"):
        load_knowledge_document("{not-json")
    with pytest.raises(ValueError, match="JSON object"):
        load_knowledge_document(json.dumps(["array"]))


@pytest.mark.unit
def test_load_rejects_non_finite_json_constants() -> None:
    with pytest.raises(ValueError, match="non-finite JSON constants"):
        load_knowledge_document('{"schema_version":1,"score":NaN}')


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        ("content", b"hello"),
        ("identity", {"document_id": b"file:abc123", "root_document_id": "file:abc123"}),
        ("scope", {"tenant_id": b"tenant-1"}),
        ("scope", {"tenant_id": "tenant-1", "namespace": b"kb"}),
        ("provenance", {"source_kind": b"file", "source_id": "abc123"}),
        ("provenance", {"source_kind": "file", "source_id": b"abc123"}),
        (
            "provenance",
            {
                "source_kind": "file",
                "source_id": "abc123",
                "source_uri": b"https://example.test/item",
            },
        ),
    ],
)
def test_strict_string_fields_reject_bytes(field_path: str, invalid_value: object) -> None:
    overrides: dict[str, object] = {field_path: invalid_value}
    with pytest.raises(ValidationError):
        _document(**overrides)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_value",
    [
        bytearray(b"hello"),
        42,
        True,
    ],
)
def test_content_rejects_non_string_coercion(invalid_value: object) -> None:
    with pytest.raises(ValidationError):
        _document(content=invalid_value)


@pytest.mark.unit
def test_content_rejects_strenum_coercion() -> None:
    class _Label(StrEnum):
        NOTE = "note"

    with pytest.raises(ValidationError):
        _document(content=_Label.NOTE)


@pytest.mark.unit
def test_strict_strings_preserve_original_value() -> None:
    document = _document(
        content="  padded content  ",
        scope=_scope(namespace="  kb  "),
    )
    assert document.content == "  padded content  "
    assert document.scope.namespace == "  kb  "


@pytest.mark.unit
def test_top_level_metadata_mutation_rejected() -> None:
    document = _document(metadata={"count": 1})
    with pytest.raises(TypeError, match="immutable"):
        document.metadata["token"] = "secret"  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        del document.metadata["count"]  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="immutable"):
        document.metadata.update({"more": 2})  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="immutable"):
        document.metadata.clear()  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="immutable"):
        document.metadata.pop("count")  # type: ignore[attr-defined]


@pytest.mark.unit
def test_nested_object_metadata_mutation_rejected() -> None:
    document = _document(metadata={"nested": {"items": [1]}})
    with pytest.raises(TypeError, match="immutable"):
        document.metadata["nested"]["new"] = "value"  # type: ignore[index]


@pytest.mark.unit
def test_nested_array_metadata_mutation_rejected() -> None:
    document = _document(metadata={"items": [1, 2]})
    nested_items = document.metadata["items"]
    with pytest.raises(TypeError, match="immutable"):
        nested_items.append(3)  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="immutable"):
        nested_items.extend([4])  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="immutable"):
        nested_items[0] = 9  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        nested_items.pop()  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="immutable"):
        nested_items.sort()  # type: ignore[attr-defined]


@pytest.mark.unit
def test_metadata_inplace_operators_rejected() -> None:
    document = _document(metadata={"items": [1]})
    with pytest.raises(TypeError, match="immutable"):
        document.metadata["items"] += [2]  # type: ignore[operator]
    with pytest.raises(TypeError, match="immutable"):
        document.metadata |= {"more": 2}  # type: ignore[operator]


@pytest.mark.unit
def test_metadata_input_alias_mutation_does_not_affect_document() -> None:
    source = {"nested": {"items": [1]}}
    document = _document(metadata=source)
    source["nested"]["items"].append(2)
    assert document.metadata["nested"] == {"items": [1]}


@pytest.mark.unit
def test_metadata_model_dump_json_compatible() -> None:
    document = _document(metadata={"nested": {"items": [1, True]}})
    dumped = document.model_dump(mode="json")
    assert dumped["metadata"] == {"nested": {"items": [1, True]}}
    assert isinstance(dumped["metadata"], dict)
    assert isinstance(dumped["metadata"]["nested"], dict)
    assert isinstance(dumped["metadata"]["nested"]["items"], list)


@pytest.mark.unit
def test_immutable_metadata_round_trip() -> None:
    document = _document(metadata={"nested": {"items": [1, 2]}})
    restored = load_knowledge_document(dump_knowledge_document(document))
    assert restored.metadata == {"nested": {"items": [1, 2]}}
    assert restored == document


@pytest.mark.unit
def test_dump_rejects_constructed_model_with_secret_metadata() -> None:
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
        dump_knowledge_document(malformed)


@pytest.mark.unit
def test_dump_rejects_constructed_model_with_non_finite_metadata() -> None:
    malformed = KnowledgeDocument.model_construct(
        schema_version=1,
        identity=KnowledgeDocumentIdentity(
            document_id="file:abc123",
            root_document_id="file:abc123",
        ),
        scope=KnowledgeDocumentScope(tenant_id="tenant-1"),
        content="Hello knowledge",
        metadata={"score": math.nan},
        provenance=KnowledgeDocumentProvenance(
            source_kind="file",
            source_id="abc123",
        ),
    )
    with pytest.raises(ValidationError):
        dump_knowledge_document(malformed)


@pytest.mark.unit
def test_dump_rejects_constructed_model_with_wrong_type() -> None:
    malformed = KnowledgeDocument.model_construct(
        schema_version=1,
        identity=KnowledgeDocumentIdentity(
            document_id="file:abc123",
            root_document_id="file:abc123",
        ),
        scope=KnowledgeDocumentScope(tenant_id="tenant-1"),
        content="Hello knowledge",
        metadata={"blob": b"raw"},
        provenance=KnowledgeDocumentProvenance(
            source_kind="file",
            source_id="abc123",
        ),
    )
    with pytest.raises(ValidationError):
        dump_knowledge_document(malformed)


@pytest.mark.unit
def test_dump_rejects_non_document_argument() -> None:
    document = _document()
    with pytest.raises(TypeError, match="KnowledgeDocument"):
        dump_knowledge_document(document.model_dump())  # type: ignore[arg-type]


@pytest.mark.unit
def test_knowledge_package_has_no_langchain_imports() -> None:
    knowledge_root = Path(__file__).resolve().parents[3] / "intergrax" / "knowledge"
    forbidden_markers = (
        "import langchain",
        "from langchain",
        "import langgraph",
        "from langgraph",
    )
    for path in knowledge_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            assert not any(marker in lowered for marker in forbidden_markers), (
                f"forbidden import in {path}:{lineno}: {stripped}"
            )
