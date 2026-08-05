# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LangChain Document compatibility bridge for KnowledgeDocument."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.knowledge.contracts.document import RESERVED_METADATA_KEYS

if TYPE_CHECKING:
    from langchain_core.documents import Document as LangChainDocument

_TRANSPORT_RESERVED_KEYS: frozenset[str] = RESERVED_METADATA_KEYS


class LangChainCompatibilityUnavailableError(ImportError):
    """Raised when langchain-core is required but not installed."""


class LangChainDocumentBridgeError(ValueError):
    """Raised when LangChain document conversion rules are violated."""


def _import_langchain_documents_module() -> Any:
    try:
        return importlib.import_module("langchain_core.documents")
    except ImportError as exc:
        raise LangChainCompatibilityUnavailableError(
            "langchain-core is required to use intergrax.compat.langchain"
        ) from exc


def _get_langchain_document_class() -> type[LangChainDocument]:
    module = _import_langchain_documents_module()
    return module.Document


def make_langchain_document(
    *,
    document_id: str | None,
    content: str,
    metadata: Mapping[str, Any],
) -> LangChainDocument:
    """Create a legacy document without exposing its SDK import to core."""
    document_cls = _get_langchain_document_class()
    return document_cls(
        id=document_id,
        page_content=content,
        metadata=dict(metadata),
    )


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LangChainDocumentBridgeError(f"{field_name} must be a mapping")
    return value


def _resolve_document_id(
    *,
    explicit_document_id: str | None,
    langchain_id: object,
    metadata: Mapping[str, Any],
) -> str:
    carriers: list[tuple[str, str]] = []
    if explicit_document_id is not None:
        carriers.append(("document_id argument", explicit_document_id))
    if langchain_id is not None:
        if not isinstance(langchain_id, str):
            raise LangChainDocumentBridgeError("Document.id must be a string when provided")
        carriers.append(("Document.id", langchain_id))
    if "document_id" in metadata:
        raw_metadata_id = metadata["document_id"]
        if not isinstance(raw_metadata_id, str):
            raise LangChainDocumentBridgeError("metadata document_id must be a string")
        carriers.append(("metadata document_id", raw_metadata_id))

    if not carriers:
        raise LangChainDocumentBridgeError(
            "document_id is required from exactly one of: document_id argument, "
            "Document.id, or metadata['document_id']"
        )
    if len(carriers) > 1:
        raise LangChainDocumentBridgeError(
            "document_id must come from exactly one source; multiple carriers are forbidden"
        )
    return carriers[0][1]


def _resolve_source_id(metadata: Mapping[str, Any]) -> str:
    has_source_id = "source_id" in metadata
    has_legacy_source = "source" in metadata
    if not has_source_id and not has_legacy_source:
        raise LangChainDocumentBridgeError(
            "source_id is required from metadata['source_id'] or legacy metadata['source']"
        )

    source_id: str | None = None
    if has_source_id:
        raw_source_id = metadata["source_id"]
        if not isinstance(raw_source_id, str):
            raise LangChainDocumentBridgeError("metadata source_id must be a string")
        source_id = raw_source_id

    legacy_source: str | None = None
    if has_legacy_source:
        raw_legacy_source = metadata["source"]
        if not isinstance(raw_legacy_source, str):
            raise LangChainDocumentBridgeError("metadata source must be a string")
        legacy_source = raw_legacy_source

    if source_id is not None and legacy_source is not None and source_id != legacy_source:
        raise LangChainDocumentBridgeError(
            "metadata source_id and legacy metadata source must not conflict"
        )

    resolved = source_id if source_id is not None else legacy_source
    assert resolved is not None
    return resolved


def _optional_transport_string(
    metadata: Mapping[str, Any],
    key: str,
) -> str | None:
    if key not in metadata:
        return None
    value = metadata[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise LangChainDocumentBridgeError(f"metadata {key} must be a string")
    return value


def _required_transport_string(
    metadata: Mapping[str, Any],
    key: str,
) -> str:
    if key not in metadata:
        raise LangChainDocumentBridgeError(f"metadata {key} is required")
    value = metadata[key]
    if not isinstance(value, str):
        raise LangChainDocumentBridgeError(f"metadata {key} must be a string")
    return value


def _resolve_schema_version(metadata: Mapping[str, Any]) -> int:
    if "schema_version" not in metadata:
        return 1
    value = metadata["schema_version"]
    if type(value) is not int or value != 1:
        raise LangChainDocumentBridgeError("metadata schema_version must be integer 1")
    return 1


def _normalize_transport_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _normalize_transport_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_transport_json(item) for item in value]
    return value


def _remaining_metadata(
    metadata: Mapping[str, Any],
    *,
    extracted_keys: set[str],
) -> dict[str, Any]:
    remaining: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in extracted_keys:
            continue
        remaining[key] = value
    return remaining


def from_langchain_document(
    document: object,
    *,
    document_id: str | None = None,
) -> KnowledgeDocument:
    """Convert a LangChain Document into a native KnowledgeDocument."""
    document_cls = _get_langchain_document_class()
    if not isinstance(document, document_cls):
        raise TypeError("document must be a langchain_core.documents.Document")

    metadata = _require_mapping(document.metadata, field_name="document.metadata")
    metadata_view = dict(metadata)

    resolved_document_id = _resolve_document_id(
        explicit_document_id=document_id,
        langchain_id=document.id,
        metadata=metadata_view,
    )
    schema_version = _resolve_schema_version(metadata_view)

    tenant_id = _required_transport_string(metadata_view, "tenant_id")
    source_kind = _required_transport_string(metadata_view, "source_kind")
    source_id = _resolve_source_id(metadata_view)

    root_document_id = _optional_transport_string(metadata_view, "root_document_id")
    if root_document_id is None:
        root_document_id = resolved_document_id

    parent_document_id = _optional_transport_string(metadata_view, "parent_document_id")
    namespace = _optional_transport_string(metadata_view, "namespace")
    workspace_id = _optional_transport_string(metadata_view, "workspace_id")
    source_parent_id = _optional_transport_string(metadata_view, "source_parent_id")
    provider_id = _optional_transport_string(metadata_view, "provider_id")
    source_revision = _optional_transport_string(metadata_view, "source_revision")
    source_uri = _optional_transport_string(metadata_view, "source_uri")
    content_hash = _optional_transport_string(metadata_view, "content_hash")

    extracted_keys = set(_TRANSPORT_RESERVED_KEYS)
    remaining = _remaining_metadata(metadata_view, extracted_keys=extracted_keys)

    return KnowledgeDocument.model_validate(
        {
            "schema_version": schema_version,
            "identity": {
                "document_id": resolved_document_id,
                "root_document_id": root_document_id,
                "parent_document_id": parent_document_id,
            },
            "scope": {
                "tenant_id": tenant_id,
                "namespace": namespace,
                "workspace_id": workspace_id,
            },
            "content": document.page_content,
            "metadata": remaining,
            "provenance": {
                "source_kind": source_kind,
                "source_id": source_id,
                "source_parent_id": source_parent_id,
                "provider_id": provider_id,
                "source_revision": source_revision,
                "source_uri": source_uri,
                "content_hash": content_hash,
            },
        }
    )


def to_langchain_document(document: object) -> LangChainDocument:
    """Convert a native KnowledgeDocument into a LangChain Document."""
    if not isinstance(document, KnowledgeDocument):
        raise TypeError("document must be a KnowledgeDocument")

    try:
        validated = KnowledgeDocument.model_validate(document.model_dump(mode="python"))
    except ValidationError:
        raise

    transport_metadata = _normalize_transport_json(validated.metadata)

    legacy_source = transport_metadata.get("source")
    if legacy_source is not None:
        if not isinstance(legacy_source, str):
            raise LangChainDocumentBridgeError("metadata source must be a string")
        if legacy_source != validated.provenance.source_id:
            raise LangChainDocumentBridgeError(
                "metadata source and provenance.source_id must not conflict"
            )

    transport_metadata["root_document_id"] = validated.identity.root_document_id
    transport_metadata["tenant_id"] = validated.scope.tenant_id
    transport_metadata["source_kind"] = validated.provenance.source_kind
    transport_metadata["source_id"] = validated.provenance.source_id

    optional_fields = {
        "parent_document_id": validated.identity.parent_document_id,
        "namespace": validated.scope.namespace,
        "workspace_id": validated.scope.workspace_id,
        "source_parent_id": validated.provenance.source_parent_id,
        "provider_id": validated.provenance.provider_id,
        "source_revision": validated.provenance.source_revision,
        "source_uri": validated.provenance.source_uri,
        "content_hash": validated.provenance.content_hash,
    }
    for key, value in optional_fields.items():
        if value is not None:
            transport_metadata[key] = value

    document_cls = _get_langchain_document_class()
    return document_cls(
        id=validated.identity.document_id,
        page_content=validated.content,
        metadata=transport_metadata,
    )
