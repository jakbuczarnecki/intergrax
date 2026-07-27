# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Synchronization models for the Vendor Knowledge Facade."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemRevision,
    KnowledgePermissions,
    KnowledgeSourceRef,
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

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


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _require_sha256_hex(value: str, *, field_name: str) -> str:
    cleaned = _require_non_empty(value, field_name=field_name)
    if _SHA256_HEX_RE.fullmatch(cleaned) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return cleaned


class KnowledgeSyncMode(StrEnum):
    INCREMENTAL = "incremental"
    RECONCILIATION = "reconciliation"


class KnowledgeRemoteItemStatus(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"
    REVOKED = "revoked"


class KnowledgeSyncRunStatus(StrEnum):
    COMPLETED = "completed"
    LEASE_BUSY = "lease_busy"


class KnowledgeSourceLeaseToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    owner_id: str
    token: str = Field(repr=False)

    @field_validator("tenant_id", "binding_id", "owner_id", "token")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)


class KnowledgeSyncCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    binding_configuration_version: int = Field(ge=1)
    cursor: KnowledgeCursor

    @field_validator("tenant_id", "binding_id")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)


class KnowledgeRemoteItemState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    binding_configuration_version: int = Field(ge=1)
    provider_id: str
    source_kind: str
    remote_id: str
    status: KnowledgeRemoteItemStatus
    revision: KnowledgeItemRevision | None = None
    last_delivery_id: str

    @field_validator(
        "tenant_id",
        "binding_id",
        "provider_id",
        "source_kind",
        "remote_id",
    )
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("last_delivery_id")
    @classmethod
    def _delivery_id_hash(cls, value: str) -> str:
        return _require_sha256_hex(value, field_name="last_delivery_id")

    @model_validator(mode="after")
    def _active_requires_revision(self) -> KnowledgeRemoteItemState:
        if self.status is KnowledgeRemoteItemStatus.ACTIVE and self.revision is None:
            raise ValueError("active remote item state requires revision")
        return self


class KnowledgeSyncEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    change_kind: KnowledgeChangeKind
    remote_id: str
    descriptor: KnowledgeItemDescriptor | None = None
    content: KnowledgeContent | None = None
    permissions: KnowledgePermissions | None = None

    @field_validator("remote_id")
    @classmethod
    def _non_empty_remote_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="remote_id")

    @model_validator(mode="after")
    def _envelope_rules(self) -> KnowledgeSyncEnvelope:
        if self.change_kind in _ACTIVE_CHANGE_KINDS and self.descriptor is None:
            raise ValueError(
                f"descriptor is required for change kind '{self.change_kind.value}'"
            )
        if self.descriptor is not None and self.descriptor.identity.remote_id != self.remote_id:
            raise ValueError("descriptor.identity.remote_id must match remote_id")
        if self.change_kind in _TOMBSTONE_CHANGE_KINDS:
            if self.content is not None:
                raise ValueError("tombstone envelopes must not include content")
            if self.permissions is not None:
                raise ValueError("tombstone envelopes must not include permissions")
            return self
        if self.content is not None and self.descriptor is not None:
            if self.content.mode is not self.descriptor.content_mode:
                raise ValueError("content.mode must match descriptor.content_mode")
        return self


class KnowledgeSyncBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    binding_configuration_version: int = Field(ge=1)
    source: KnowledgeSourceRef
    mode: KnowledgeSyncMode
    delivery_id: str
    envelopes: tuple[KnowledgeSyncEnvelope, ...] = ()
    has_more: bool = False

    @field_validator("tenant_id", "binding_id")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("delivery_id")
    @classmethod
    def _delivery_id_hash(cls, value: str) -> str:
        return _require_sha256_hex(value, field_name="delivery_id")

    @field_validator("envelopes")
    @classmethod
    def _immutable_envelopes(
        cls, value: tuple[KnowledgeSyncEnvelope, ...] | list[KnowledgeSyncEnvelope]
    ) -> tuple[KnowledgeSyncEnvelope, ...]:
        return tuple(value)

    @model_validator(mode="after")
    def _tenant_matches_source(self) -> KnowledgeSyncBatch:
        if self.source.tenant_id != self.tenant_id:
            raise ValueError("source.tenant_id must match batch tenant_id")
        return self


class KnowledgeSyncRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: KnowledgeSyncRunStatus
    mode: KnowledgeSyncMode
    tenant_id: str
    binding_id: str
    delivery_id: str | None = None
    changes_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    tombstone_count: int = Field(ge=0)
    checkpoint_advanced: bool
    has_more: bool
    retryable: bool

    @field_validator("tenant_id", "binding_id")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("delivery_id")
    @classmethod
    def _optional_delivery_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_sha256_hex(value, field_name="delivery_id")

    @model_validator(mode="after")
    def _status_rules(self) -> KnowledgeSyncRunResult:
        if self.status is KnowledgeSyncRunStatus.LEASE_BUSY:
            if self.delivery_id is not None:
                raise ValueError("lease_busy result must not include delivery_id")
            if self.changes_count != 0 or self.active_count != 0 or self.tombstone_count != 0:
                raise ValueError("lease_busy result counts must be zero")
            if self.checkpoint_advanced:
                raise ValueError("lease_busy result must not advance checkpoint")
            if not self.retryable:
                raise ValueError("lease_busy result must be retryable")
            return self
        if self.status is KnowledgeSyncRunStatus.COMPLETED:
            if self.delivery_id is None:
                raise ValueError("completed result requires delivery_id")
            if self.retryable:
                raise ValueError("completed result must not be retryable")
            return self
        raise ValueError(f"unsupported sync run status: {self.status!r}")
