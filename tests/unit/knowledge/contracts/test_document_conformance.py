# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public ABI conformance suite for native KnowledgeDocument (LCI-1D)."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

import intergrax.knowledge.contracts as contracts
from intergrax.knowledge.contracts import (
    KnowledgeDocument,
    KnowledgeDocumentIdentity,
    KnowledgeDocumentProvenance,
    KnowledgeDocumentScope,
    dump_knowledge_document,
    load_knowledge_document,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

EXPECTED_PUBLIC_EXPORTS = (
    "KnowledgeDocument",
    "KnowledgeDocumentIdentity",
    "KnowledgeDocumentProvenance",
    "KnowledgeDocumentScope",
    "dump_knowledge_document",
    "load_knowledge_document",
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
def test_public_imports_available() -> None:
    assert KnowledgeDocument is not None
    assert KnowledgeDocumentIdentity is not None
    assert KnowledgeDocumentScope is not None
    assert KnowledgeDocumentProvenance is not None
    assert dump_knowledge_document is not None
    assert load_knowledge_document is not None


@pytest.mark.unit
def test_exact_public_exports() -> None:
    assert tuple(sorted(contracts.__all__)) == tuple(sorted(EXPECTED_PUBLIC_EXPORTS))


@pytest.mark.unit
def test_source_document_construction() -> None:
    document = _document(content=" padded content ")
    assert document.identity.parent_document_id is None
    assert document.identity.root_document_id == document.identity.document_id
    assert document.content == " padded content "


@pytest.mark.unit
def test_derivative_document_construction() -> None:
    document = _document(
        identity=_identity(
            document_id="file:abc123:chunk:1",
            root_document_id="file:abc123",
            parent_document_id="file:abc123",
        )
    )
    assert document.identity.root_document_id == "file:abc123"
    assert document.identity.parent_document_id == "file:abc123"
    assert document.identity.parent_document_id != document.identity.document_id


@pytest.mark.unit
def test_identity_rejection_rules() -> None:
    with pytest.raises(ValidationError):
        _document(identity=_identity(parent_document_id="file:parent"))
    with pytest.raises(ValidationError):
        _document(
            identity=_identity(
                document_id="file:abc123:chunk:1",
                root_document_id="file:abc123:chunk:1",
                parent_document_id="file:abc123",
            )
        )


@pytest.mark.unit
def test_strict_input_rejection() -> None:
    with pytest.raises(ValidationError):
        _document(content=b"bytes")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        KnowledgeDocument.model_validate(
            {
                "schema_version": True,
                "identity": _identity(),
                "scope": _scope(),
                "content": "x",
                "metadata": {},
                "provenance": _provenance(),
            }
        )
    with pytest.raises(ValidationError):
        KnowledgeDocument.model_validate(
            {
                "schema_version": "1",
                "identity": _identity(),
                "scope": _scope(),
                "content": "x",
                "metadata": {},
                "provenance": _provenance(),
            }
        )
    with pytest.raises(ValidationError):
        KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": _identity(),
                "scope": {"namespace": "ns"},
                "content": "x",
                "metadata": {},
                "provenance": _provenance(),
            }
        )
    with pytest.raises(ValidationError):
        _document(content="   ")


@pytest.mark.unit
def test_omitted_metadata_immutability() -> None:
    document = _document_omitting_metadata()
    with pytest.raises(TypeError, match="immutable"):
        document.metadata["token"] = "secret"  # type: ignore[index]


@pytest.mark.unit
def test_nested_metadata_immutability() -> None:
    document = _document(metadata={"nested": {"items": [1]}})
    with pytest.raises(TypeError, match="immutable"):
        document.metadata["nested"]["items"].append(2)  # type: ignore[attr-defined]


@pytest.mark.unit
def test_deterministic_serialization() -> None:
    document = _document(content="Unicode: łódź — π")
    first = dump_knowledge_document(document)
    second = dump_knowledge_document(document)
    assert isinstance(first, bytes)
    assert first == second


@pytest.mark.unit
def test_round_trip_preserves_lineage_and_unicode() -> None:
    document = _document(
        content="Unicode: łódź — π",
        identity=_identity(
            document_id="file:abc123:chunk:1",
            root_document_id="file:abc123",
            parent_document_id="file:abc123",
        ),
    )
    restored = load_knowledge_document(dump_knowledge_document(document))
    assert restored == document


@pytest.mark.unit
def test_duplicate_json_keys_rejected() -> None:
    with pytest.raises(ValueError):
        load_knowledge_document('{"schema_version":1,"schema_version":2}')


@pytest.mark.unit
def test_reserved_metadata_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(metadata={"tenant_id": "x"})


@pytest.mark.unit
def test_non_finite_numbers_rejected() -> None:
    with pytest.raises(ValidationError):
        _document(metadata={"value": math.nan})
    with pytest.raises(ValidationError):
        _document(metadata={"value": math.inf})


@pytest.mark.unit
def test_unknown_schema_version_rejected() -> None:
    payload = json.loads(dump_knowledge_document(_document()).decode("utf-8"))
    payload["schema_version"] = 99
    with pytest.raises(ValidationError):
        load_knowledge_document(json.dumps(payload, sort_keys=True))
