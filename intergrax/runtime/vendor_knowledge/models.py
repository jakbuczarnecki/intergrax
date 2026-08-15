# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral knowledge models for the Vendor Knowledge Facade."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.knowledge.contracts.validation import (
    JsonObject,
    JsonValue,
    assert_safe_mapping,
    is_url_like,
    require_non_empty_trimmed_str,
    validate_safe_url,
)

_require_non_empty = require_non_empty_trimmed_str
_validate_safe_url = validate_safe_url
_assert_safe_mapping = assert_safe_mapping

class KnowledgeContentMode(StrEnum):
    BINARY = "binary"
    RICH_TEXT = "rich_text"
    STRUCTURED_RECORD = "structured_record"


class KnowledgeChangeKind(StrEnum):
    UPSERT = "upsert"
    METADATA_CHANGED = "metadata_changed"
    PERMISSIONS_CHANGED = "permissions_changed"
    DELETED = "deleted"
    REVOKED = "revoked"


class KnowledgeVisibility(StrEnum):
    UNKNOWN = "unknown"
    PUBLIC = "public"
    TENANT = "tenant"
    RESTRICTED = "restricted"
    PRIVATE = "private"


class KnowledgeSourceScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    remote_scope_id: str
    remote_scope_type: str
    safe_display_name: str
    parameters: JsonObject = Field(default_factory=dict)

    @field_validator("remote_scope_id", "remote_scope_type", "safe_display_name")
    @classmethod
    def _non_empty_identity(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("parameters")
    @classmethod
    def _safe_parameters(cls, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return _assert_safe_mapping(value, field_name="parameters")


class KnowledgeSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    provider_id: str
    integration_kind: IntegrationCategory
    source_kind: str
    connection_ref: str | None = None
    scope: KnowledgeSourceScope

    @field_validator("tenant_id", "provider_id", "source_kind")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("connection_ref")
    @classmethod
    def _opaque_connection_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty(value, field_name="connection_ref")


class KnowledgeAdapterCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    full_inventory: bool = False
    incremental_changes: bool = False
    content_fetch: bool = False
    binary_content: bool = False
    rich_text_content: bool = False
    structured_content: bool = False
    permissions: bool = False
    tombstones: bool = False
    remote_versions: bool = False
    reconciliation: bool = False


class KnowledgeItemIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    remote_id: str
    parent_remote_id: str | None = None
    logical_key: str | None = None

    @field_validator("remote_id")
    @classmethod
    def _non_empty_remote_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="remote_id")

    @field_validator("parent_remote_id", "logical_key")
    @classmethod
    def _optional_non_empty(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)


class KnowledgeItemRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str | None = None
    etag: str | None = None
    content_hash: str | None = None
    acl_hash: str | None = None
    updated_at: datetime | None = None

    @field_validator("version", "etag", "content_hash", "acl_hash")
    @classmethod
    def _optional_non_empty(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)


class KnowledgeItemProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str
    source_kind: str
    remote_id: str
    web_url: str | None = None
    safe_locator: str | None = None

    @field_validator("provider_id", "source_kind", "remote_id")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("web_url")
    @classmethod
    def _safe_web_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_safe_url(value, field_name="web_url")

    @field_validator("safe_locator")
    @classmethod
    def _optional_locator(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = _require_non_empty(value, field_name="safe_locator")
        if is_url_like(cleaned):
            return _validate_safe_url(cleaned, field_name="safe_locator")
        return cleaned


class KnowledgeItemDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: KnowledgeItemIdentity
    revision: KnowledgeItemRevision
    title: str = Field(repr=False)
    item_type: str
    content_mode: KnowledgeContentMode
    content_available: bool
    provenance: KnowledgeItemProvenance
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("title", "item_type")
    @classmethod
    def _non_empty_text(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("metadata")
    @classmethod
    def _safe_metadata(cls, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return _assert_safe_mapping(value, field_name="metadata")

    @model_validator(mode="after")
    def _identity_matches_provenance(self) -> KnowledgeItemDescriptor:
        if self.identity.remote_id != self.provenance.remote_id:
            raise ValueError("identity.remote_id must equal provenance.remote_id")
        return self


class KnowledgePrincipal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_type: str
    principal_id: str
    provider_id: str | None = None

    @field_validator("principal_type", "principal_id")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("provider_id")
    @classmethod
    def _optional_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty(value, field_name="provider_id")

    def _dedupe_key(self) -> tuple[str, str, str | None]:
        return (self.principal_type, self.principal_id, self.provider_id)


class KnowledgePermissions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    visibility: KnowledgeVisibility
    allowed_principals: tuple[KnowledgePrincipal, ...] = ()
    denied_principals: tuple[KnowledgePrincipal, ...] = ()
    inherited: bool | None = None
    acl_version: str | None = None

    @field_validator("acl_version")
    @classmethod
    def _optional_acl_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty(value, field_name="acl_version")

    @field_validator("allowed_principals", "denied_principals")
    @classmethod
    def _dedupe_principals(
        cls, value: tuple[KnowledgePrincipal, ...] | list[KnowledgePrincipal]
    ) -> tuple[KnowledgePrincipal, ...]:
        seen: set[tuple[str, str, str | None]] = set()
        unique: list[KnowledgePrincipal] = []
        for principal in value:
            key = principal._dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            unique.append(principal)
        return tuple(unique)


class KnowledgeContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: KnowledgeContentMode
    binary: bytes | None = Field(default=None, repr=False)
    rich_text: str | None = Field(default=None, repr=False)
    structured_record: JsonObject | None = Field(default=None, repr=False)
    mime_type: str | None = None
    encoding: str | None = None
    content_hash: str | None = None

    @field_validator("mime_type", "encoding", "content_hash")
    @classmethod
    def _optional_non_empty(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> KnowledgeContent:
        binary_present = self.binary is not None
        rich_text_present = self.rich_text is not None
        structured_present = self.structured_record is not None
        payload_count = sum((binary_present, rich_text_present, structured_present))
        if payload_count != 1:
            raise ValueError("exactly one content payload must be provided")

        if self.mode is KnowledgeContentMode.BINARY:
            if not binary_present:
                raise ValueError("binary mode requires binary payload")
            return self

        if self.mode is KnowledgeContentMode.RICH_TEXT:
            if not rich_text_present:
                raise ValueError("rich_text mode requires rich_text payload")
            if self.rich_text is not None and not str(self.rich_text):
                raise ValueError("rich_text payload must be a non-empty string")
            return self

        if self.mode is KnowledgeContentMode.STRUCTURED_RECORD:
            if not structured_present:
                raise ValueError("structured_record mode requires structured_record payload")
            if not isinstance(self.structured_record, dict):
                raise ValueError("structured_record payload must be a JSON object")
            return self

        raise ValueError(f"unsupported content mode: {self.mode!r}")


class KnowledgeCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(repr=False)
    version: str | None = None

    @field_validator("value")
    @classmethod
    def _non_empty_value(cls, value: str) -> str:
        return _require_non_empty(value, field_name="value")

    @field_validator("version")
    @classmethod
    def _optional_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty(value, field_name="version")


_ACTIVE_CHANGE_KINDS: frozenset[KnowledgeChangeKind] = frozenset(
    {
        KnowledgeChangeKind.UPSERT,
        KnowledgeChangeKind.METADATA_CHANGED,
        KnowledgeChangeKind.PERMISSIONS_CHANGED,
    }
)

_TOMBSTONE_CHANGE_KINDS: frozenset[KnowledgeChangeKind] = frozenset(
    {
        KnowledgeChangeKind.DELETED,
        KnowledgeChangeKind.REVOKED,
    }
)


class KnowledgeChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: KnowledgeChangeKind
    descriptor: KnowledgeItemDescriptor | None = None
    remote_id: str

    @field_validator("remote_id")
    @classmethod
    def _non_empty_remote_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="remote_id")

    @model_validator(mode="after")
    def _descriptor_rules(self) -> KnowledgeChange:
        if self.kind in _ACTIVE_CHANGE_KINDS and self.descriptor is None:
            raise ValueError(f"descriptor is required for change kind '{self.kind.value}'")
        if self.descriptor is not None and self.descriptor.identity.remote_id != self.remote_id:
            raise ValueError("descriptor.identity.remote_id must match remote_id")
        if self.kind in _TOMBSTONE_CHANGE_KINDS:
            return self
        return self


class KnowledgePage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    changes: tuple[KnowledgeChange, ...] = ()
    next_cursor: KnowledgeCursor | None = None
    proposed_checkpoint: KnowledgeCursor | None = None
    has_more: bool = False

    @field_validator("changes")
    @classmethod
    def _immutable_changes(
        cls, value: tuple[KnowledgeChange, ...] | list[KnowledgeChange]
    ) -> tuple[KnowledgeChange, ...]:
        return tuple(value)

    @model_validator(mode="after")
    def _cursor_when_more(self) -> KnowledgePage:
        if self.has_more and self.next_cursor is None:
            raise ValueError("has_more=True requires next_cursor")
        return self


class KnowledgeScopeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: KnowledgeSourceRef
    capabilities: KnowledgeAdapterCapabilities
    safe_display_name: str

    @field_validator("safe_display_name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        return _require_non_empty(value, field_name="safe_display_name")
