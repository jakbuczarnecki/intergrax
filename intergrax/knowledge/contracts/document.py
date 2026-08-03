# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Native knowledge document contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from intergrax.knowledge.contracts.validation import (
    JsonValue,
    assert_knowledge_metadata,
    require_non_empty_str,
    validate_safe_url,
)

SCHEMA_VERSION = 1

RESERVED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "document_id",
        "root_document_id",
        "parent_document_id",
        "tenant_id",
        "namespace",
        "source_kind",
        "source_id",
        "source_parent_id",
        "provider_id",
        "source_revision",
        "source_uri",
        "content_hash",
    }
)


class KnowledgeDocumentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    root_document_id: str
    parent_document_id: str | None = None

    @field_validator("document_id", "root_document_id")
    @classmethod
    def _non_empty_identity(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return require_non_empty_str(value, field_name=field_name)

    @field_validator("parent_document_id")
    @classmethod
    def _optional_parent(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return require_non_empty_str(value, field_name=field_name)

    @model_validator(mode="after")
    def _validate_lineage(self) -> KnowledgeDocumentIdentity:
        if self.parent_document_id is None:
            if self.root_document_id != self.document_id:
                raise ValueError(
                    "root_document_id must equal document_id for source documents"
                )
        else:
            if self.root_document_id == self.document_id:
                raise ValueError(
                    "root_document_id must differ from document_id for derivative documents"
                )
            if self.parent_document_id == self.document_id:
                raise ValueError("parent_document_id must not equal document_id")
        return self


class KnowledgeDocumentScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    namespace: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def _non_empty_tenant(cls, value: str) -> str:
        return require_non_empty_str(value, field_name="tenant_id")

    @field_validator("namespace")
    @classmethod
    def _optional_namespace(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_str(value, field_name="namespace")


class KnowledgeDocumentProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: str
    source_id: str
    source_parent_id: str | None = None
    provider_id: str | None = None
    source_revision: str | None = None
    source_uri: str | None = None
    content_hash: str | None = None

    @field_validator("source_kind", "source_id")
    @classmethod
    def _required_source_fields(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return require_non_empty_str(value, field_name=field_name)

    @field_validator(
        "source_parent_id",
        "provider_id",
        "source_revision",
        "content_hash",
    )
    @classmethod
    def _optional_non_empty(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return require_non_empty_str(value, field_name=field_name)

    @field_validator("source_uri")
    @classmethod
    def _safe_source_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = require_non_empty_str(value, field_name="source_uri")
        return validate_safe_url(cleaned, field_name="source_uri")


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Annotated[int, Field(strict=True)]
    identity: KnowledgeDocumentIdentity
    scope: KnowledgeDocumentScope
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: KnowledgeDocumentProvenance

    @field_validator("schema_version")
    @classmethod
    def _schema_version_exact(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError("schema_version must be integer 1")
        return value

    @field_validator("content")
    @classmethod
    def _non_empty_content(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("content must be a string")
        if not value.strip():
            raise ValueError("content must be a non-empty string")
        return value

    @field_validator("metadata")
    @classmethod
    def _safe_metadata(cls, value: Mapping[str, Any]) -> dict[str, JsonValue]:
        return assert_knowledge_metadata(
            value,
            field_name="metadata",
            reserved_keys=RESERVED_METADATA_KEYS,
        )


def _reject_non_finite_json_constant(constant: str) -> float:
    raise ValueError("payload must not contain non-finite JSON constants")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _value in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("payload must not contain duplicate JSON keys")
    return dict(pairs)


def dump_knowledge_document(document: KnowledgeDocument) -> bytes:
    payload = document.model_dump(mode="json")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_knowledge_document(payload: bytes | str) -> KnowledgeDocument:
    if not isinstance(payload, (bytes, str)):
        raise TypeError("payload must be bytes or str")

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("payload must be valid UTF-8") from exc
    else:
        text = payload

    try:
        data = json.loads(
            text,
            parse_constant=_reject_non_finite_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("payload must be valid JSON") from exc
    except ValueError:
        raise

    if not isinstance(data, dict):
        raise ValueError("payload root must be a JSON object")

    return KnowledgeDocument.model_validate(data)
