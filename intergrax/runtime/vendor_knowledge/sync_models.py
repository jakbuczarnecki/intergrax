# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Synchronization models for the Vendor Knowledge Facade."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Literal, Mapping, Union

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
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemRevision,
    KnowledgePermissions,
    KnowledgeSourceRef,
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_OPERATOR_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

DEFAULT_MAX_RECONCILIATION_CANDIDATE_COUNT = 10_000
DEFAULT_MAX_RECONCILIATION_CANDIDATE_PAYLOAD_BYTES = 1_048_576
DEFAULT_MAX_RECONCILIATION_PREPARED_INTENT_PAYLOAD_BYTES = 1_048_576
DEFAULT_MAX_RECONCILIATION_PREPARED_STATE_MUTATION_COUNT = 10_000
DEFAULT_MAX_RECONCILIATION_REMOTE_ID_BYTES = 2_048

_RECONCILIATION_RUN_SCHEMA = "vendor_knowledge.reconciliation_run.v1"

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


def _require_operator_reason_code(value: str, *, field_name: str) -> str:
    cleaned = _require_non_empty(value, field_name=field_name)
    if _OPERATOR_REASON_CODE_RE.fullmatch(cleaned) is None:
        raise ValueError(f"{field_name} must be a bounded safe operator reason code")
    return cleaned


def _assert_utc_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
    return value


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def knowledge_cursor_fingerprint_payload(
    cursor: KnowledgeCursor | None,
) -> dict[str, str | None]:
    if cursor is None:
        return {"value": None, "version": None}
    return {"value": cursor.value, "version": cursor.version}


def knowledge_cursor_fingerprint_sha256(cursor: KnowledgeCursor | None) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(knowledge_cursor_fingerprint_payload(cursor))
    ).hexdigest()


def canonical_reconciliation_candidate_inventory_bytes(
    remote_ids: tuple[str, ...],
) -> bytes:
    return _canonical_json_bytes({"remaining_candidate_remote_ids": list(remote_ids)})


def canonical_prepared_state_mutations_fingerprint(
    templates: tuple["KnowledgeReconciliationPreparedStateMutationTemplate", ...],
) -> str:
    ordered = sorted(templates, key=lambda template: template.remote_id)
    payload = [template.model_dump(mode="json") for template in ordered]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def knowledge_item_revision_fingerprint_payload(
    revision: KnowledgeItemRevision | None,
) -> dict[str, Any]:
    if revision is None:
        return {
            "version": None,
            "etag": None,
            "content_hash": None,
            "acl_hash": None,
            "updated_at": None,
        }
    return {
        "version": revision.version,
        "etag": revision.etag,
        "content_hash": revision.content_hash,
        "acl_hash": revision.acl_hash,
        "updated_at": (
            revision.updated_at.isoformat() if revision.updated_at is not None else None
        ),
    }


def knowledge_descriptor_fingerprint_payload(
    descriptor: KnowledgeItemDescriptor | None,
) -> dict[str, Any] | None:
    if descriptor is None:
        return None
    return {
        "identity": descriptor.identity.model_dump(mode="json"),
        "revision": knowledge_item_revision_fingerprint_payload(descriptor.revision),
        "item_type": descriptor.item_type,
        "title": descriptor.title,
        "content_mode": descriptor.content_mode.value,
        "content_available": descriptor.content_available,
        "metadata": descriptor.metadata,
        "provenance": descriptor.provenance.model_dump(mode="json"),
    }


def knowledge_content_fingerprint_payload(
    content: KnowledgeContent | None,
) -> dict[str, Any] | None:
    if content is None:
        return None
    if content.content_hash:
        return {
            "mode": content.mode.value,
            "mime_type": content.mime_type,
            "content_hash": content.content_hash,
            "structured_schema": None,
        }
    if content.mode is KnowledgeContentMode.BINARY and content.binary is not None:
        return {
            "mode": content.mode.value,
            "mime_type": content.mime_type,
            "content_hash": hashlib.sha256(content.binary).hexdigest(),
            "structured_schema": None,
        }
    structured_schema = None
    if content.structured_record is not None:
        structured_schema = content.structured_record.get("schema")
    return {
        "mode": content.mode.value,
        "mime_type": content.mime_type,
        "content_hash": None,
        "structured_schema": structured_schema,
        "canonical_payload": content.model_dump(mode="json", exclude={"binary"}),
    }


def knowledge_permissions_fingerprint_payload(
    permissions: KnowledgePermissions | None,
) -> dict[str, Any] | None:
    if permissions is None:
        return None
    if permissions.acl_version:
        return {
            "visibility": permissions.visibility.value,
            "acl_version": permissions.acl_version,
            "inherited": permissions.inherited,
        }
    return permissions.model_dump(mode="json")


def reconciliation_envelope_fingerprint_payload(
    envelope: "KnowledgeSyncEnvelope",
) -> dict[str, Any]:
    semantic = None
    if envelope.reconciliation_semantic is not None:
        semantic = envelope.reconciliation_semantic.value
    return {
        "change_kind": envelope.change_kind.value,
        "remote_id": envelope.remote_id,
        "reconciliation_semantic": semantic,
        "descriptor": knowledge_descriptor_fingerprint_payload(envelope.descriptor),
        "content": knowledge_content_fingerprint_payload(envelope.content),
        "permissions": knowledge_permissions_fingerprint_payload(envelope.permissions),
    }


def reconciliation_provider_page_fingerprint(
    *,
    input_cursor_fingerprint: str,
    has_more: bool,
    proposed_checkpoint_fingerprint: str,
    next_cursor_fingerprint: str,
    changes: tuple[tuple[str, KnowledgeChangeKind, KnowledgeItemRevision | None], ...],
) -> str:
    ordered_changes = sorted(changes, key=lambda item: item[0])
    payload = {
        "input_cursor_fingerprint": input_cursor_fingerprint,
        "has_more": has_more,
        "proposed_checkpoint_fingerprint": proposed_checkpoint_fingerprint,
        "next_cursor_fingerprint": next_cursor_fingerprint,
        "changes": [
            {
                "remote_id": remote_id,
                "change_kind": change_kind.value,
                "revision": knowledge_item_revision_fingerprint_payload(revision),
            }
            for remote_id, change_kind, revision in ordered_changes
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def reconciliation_prepared_batch_payload_fingerprint(
    *,
    tenant_id: str,
    binding_id: str,
    binding_configuration_version: int,
    mode: KnowledgeSyncMode,
    run_id: str,
    source: KnowledgeSourceRef,
    has_more: bool,
    envelopes: tuple["KnowledgeSyncEnvelope", ...],
    prepared_state_mutations_fingerprint: str,
    provider_page_fingerprint: str,
    input_cursor_fingerprint: str,
    proposed_checkpoint_fingerprint: str,
    next_cursor_fingerprint: str,
) -> str:
    ordered_envelopes = tuple(
        sorted(envelopes, key=lambda envelope: envelope.remote_id)
    )
    payload = {
        "tenant_id": tenant_id,
        "binding_id": binding_id,
        "binding_configuration_version": binding_configuration_version,
        "mode": mode.value,
        "run_id": run_id,
        "source": {
            "provider_id": source.provider_id,
            "integration_kind": source.integration_kind.value,
            "source_kind": source.source_kind,
            "connection_ref": source.connection_ref,
            "scope": source.scope.model_dump(mode="json"),
        },
        "has_more": has_more,
        "envelopes": [
            reconciliation_envelope_fingerprint_payload(envelope)
            for envelope in ordered_envelopes
        ],
        "prepared_state_mutations_fingerprint": prepared_state_mutations_fingerprint,
        "provider_page_fingerprint": provider_page_fingerprint,
        "input_cursor_fingerprint": input_cursor_fingerprint,
        "proposed_checkpoint_fingerprint": proposed_checkpoint_fingerprint,
        "next_cursor_fingerprint": next_cursor_fingerprint,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def reconciliation_delivery_id(
    *,
    tenant_id: str,
    binding_id: str,
    binding_configuration_version: int,
    mode: KnowledgeSyncMode,
    run_id: str,
    provider_page_fingerprint: str,
    prepared_batch_payload_fingerprint: str,
    prepared_state_mutations_fingerprint: str,
    input_cursor_fingerprint: str,
    proposed_checkpoint_fingerprint: str,
    next_cursor_fingerprint: str,
) -> str:
    payload = {
        "tenant_id": tenant_id,
        "binding_id": binding_id,
        "binding_configuration_version": binding_configuration_version,
        "mode": mode.value,
        "run_id": run_id,
        "provider_page_fingerprint": provider_page_fingerprint,
        "prepared_batch_payload_fingerprint": prepared_batch_payload_fingerprint,
        "prepared_state_mutations_fingerprint": prepared_state_mutations_fingerprint,
        "input_cursor_fingerprint": input_cursor_fingerprint,
        "proposed_checkpoint_fingerprint": proposed_checkpoint_fingerprint,
        "next_cursor_fingerprint": next_cursor_fingerprint,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


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


def knowledge_sync_checkpoint_fingerprint_sha256(
    checkpoint: KnowledgeSyncCheckpoint,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(checkpoint.model_dump(mode="json"))
    ).hexdigest()


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
    reconciliation_semantic: KnowledgeReconciliationMutationSemantic | None = None

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
        if (
            self.descriptor is not None
            and self.descriptor.identity.remote_id != self.remote_id
        ):
            raise ValueError("descriptor.identity.remote_id must match remote_id")
        if self.change_kind in _TOMBSTONE_CHANGE_KINDS:
            if self.content is not None:
                raise ValueError("tombstone envelopes must not include content")
            if self.permissions is not None:
                raise ValueError("tombstone envelopes must not include permissions")
            if self.reconciliation_semantic is not None and self.descriptor is not None:
                raise ValueError(
                    "synthetic reconciliation tombstone must not include descriptor"
                )
            return self
        if self.reconciliation_semantic is not None:
            raise ValueError(
                "reconciliation semantic marker is allowed only for tombstone envelopes"
            )
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
            if (
                self.changes_count != 0
                or self.active_count != 0
                or self.tombstone_count != 0
            ):
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


class KnowledgeReconciliationRunPhase(StrEnum):
    COLLECTING = "collecting"
    PAGE_PREPARED = "page_prepared"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    RECOVERY_REQUIRED = "recovery_required"
    ABORTED = "aborted"


class KnowledgeReconciliationMutationSemantic(StrEnum):
    ABSENT_FROM_COMPLETED_SYNCHRONIZED_SOURCE_INVENTORY = (
        "absent_from_completed_synchronized_source_inventory"
    )


class KnowledgeSyncSinkReceiptStatus(StrEnum):
    ABSENT = "absent"
    APPLIED = "applied"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class KnowledgeRemoteItemStateReceiptStatus(StrEnum):
    ABSENT = "absent"
    APPLYING = "applying"
    COMPLETED = "completed"
    CONFLICT = "conflict"


class KnowledgeReconciliationRecoveryCommandKind(StrEnum):
    RESUME_EXACT = "resume_exact"
    FINALIZE_ALREADY_COMMITTED = "finalize_already_committed"
    ABORT_PRISTINE = "abort_pristine"
    REPAIR_REQUIRED = "repair_required"


class KnowledgeReconciliationLimitPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_reconciliation_candidate_count: int = Field(
        default=DEFAULT_MAX_RECONCILIATION_CANDIDATE_COUNT,
        ge=1,
        le=1_000_000,
    )
    max_reconciliation_candidate_payload_bytes: int = Field(
        default=DEFAULT_MAX_RECONCILIATION_CANDIDATE_PAYLOAD_BYTES,
        ge=1,
        le=16_777_216,
    )
    max_reconciliation_prepared_intent_payload_bytes: int = Field(
        default=DEFAULT_MAX_RECONCILIATION_PREPARED_INTENT_PAYLOAD_BYTES,
        ge=1,
        le=16_777_216,
    )
    max_reconciliation_prepared_state_mutation_count: int = Field(
        default=DEFAULT_MAX_RECONCILIATION_PREPARED_STATE_MUTATION_COUNT,
        ge=1,
        le=1_000_000,
    )
    max_reconciliation_remote_id_bytes: int = Field(
        default=DEFAULT_MAX_RECONCILIATION_REMOTE_ID_BYTES,
        ge=1,
        le=65_536,
    )


class KnowledgeReconciliationPreparedStateMutationTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    remote_id: str
    resulting_status: KnowledgeRemoteItemStatus
    revision: KnowledgeItemRevision | None = None
    binding_configuration_version: int = Field(ge=1)
    reconciliation_semantic: KnowledgeReconciliationMutationSemantic | None = None

    @field_validator("remote_id")
    @classmethod
    def _non_empty_remote_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="remote_id")

    @model_validator(mode="after")
    def _template_rules(self) -> KnowledgeReconciliationPreparedStateMutationTemplate:
        if (
            self.resulting_status is KnowledgeRemoteItemStatus.ACTIVE
            and self.revision is None
        ):
            raise ValueError("active mutation template requires revision")
        if self.reconciliation_semantic is not None and self.resulting_status not in {
            KnowledgeRemoteItemStatus.DELETED,
            KnowledgeRemoteItemStatus.REVOKED,
        }:
            raise ValueError("reconciliation semantic marker requires tombstone status")
        return self


class KnowledgeSyncSinkReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: KnowledgeSyncSinkReceiptStatus
    delivery_id: str | None = None
    prepared_batch_payload_fingerprint: str | None = None

    @field_validator("delivery_id", "prepared_batch_payload_fingerprint")
    @classmethod
    def _optional_sha256(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_sha256_hex(value, field_name=field_name)

    @model_validator(mode="after")
    def _status_evidence(self) -> KnowledgeSyncSinkReceipt:
        if self.status is KnowledgeSyncSinkReceiptStatus.ABSENT:
            if (
                self.delivery_id is not None
                or self.prepared_batch_payload_fingerprint is not None
            ):
                raise ValueError("absent sink receipt must not include evidence")
            return self
        if self.delivery_id is None or self.prepared_batch_payload_fingerprint is None:
            raise ValueError("non-absent sink receipt requires delivery evidence")
        return self


class KnowledgeRemoteItemStateReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: KnowledgeRemoteItemStateReceiptStatus
    delivery_id: str | None = None
    prepared_state_mutations_fingerprint: str | None = None

    @field_validator("delivery_id", "prepared_state_mutations_fingerprint")
    @classmethod
    def _optional_sha256(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_sha256_hex(value, field_name=field_name)

    @model_validator(mode="after")
    def _status_evidence(self) -> KnowledgeRemoteItemStateReceipt:
        if self.status is KnowledgeRemoteItemStateReceiptStatus.ABSENT:
            if (
                self.delivery_id is not None
                or self.prepared_state_mutations_fingerprint is not None
            ):
                raise ValueError("absent item-state receipt must not include evidence")
            return self
        if (
            self.delivery_id is None
            or self.prepared_state_mutations_fingerprint is None
        ):
            raise ValueError("non-absent item-state receipt requires delivery evidence")
        return self


class KnowledgeReconciliationRecoveryCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: KnowledgeReconciliationRecoveryCommandKind
    tenant_id: str
    binding_id: str
    expected_run_id: str
    expected_run_record_version: int = Field(ge=1)
    expected_phase: KnowledgeReconciliationRunPhase
    operator_reason_code: str

    @field_validator("tenant_id", "binding_id", "expected_run_id")
    @classmethod
    def _non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("operator_reason_code")
    @classmethod
    def _safe_operator_reason_code(cls, value: str) -> str:
        return _require_operator_reason_code(value, field_name="operator_reason_code")


def _validate_remote_id_byte_length(
    remote_id: str,
    *,
    policy: KnowledgeReconciliationLimitPolicy,
    field_name: str,
) -> None:
    if len(remote_id.encode("utf-8")) > policy.max_reconciliation_remote_id_bytes:
        raise ValueError(f"{field_name} exceeds configured remote ID byte limit")


def _validate_base_completed_checkpoint(
    checkpoint: KnowledgeSyncCheckpoint | None,
    *,
    tenant_id: str,
    binding_id: str,
) -> None:
    if checkpoint is None:
        return
    if checkpoint.tenant_id != tenant_id or checkpoint.binding_id != binding_id:
        raise ValueError("expected base completed checkpoint identity mismatch")


def _validate_remote_id_tuple(
    value: tuple[str, ...] | list[str],
    *,
    field_name: str,
    allow_empty: bool,
    policy: KnowledgeReconciliationLimitPolicy | None = None,
) -> tuple[str, ...]:
    ordered = tuple(value)
    if not allow_empty and not ordered:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    seen: set[str] = set()
    normalized: list[str] = []
    for remote_id in ordered:
        cleaned = _require_non_empty(remote_id, field_name=field_name)
        if policy is not None:
            _validate_remote_id_byte_length(
                cleaned,
                policy=policy,
                field_name=field_name,
            )
        if cleaned in seen:
            raise ValueError(f"{field_name} must contain unique remote IDs")
        seen.add(cleaned)
        normalized.append(cleaned)
    return tuple(sorted(normalized))


def _validate_mutation_templates_structural(
    templates: tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...],
    *,
    binding_configuration_version: int,
    has_more: bool,
    synthetic_tombstone_remote_ids: tuple[str, ...],
    remaining_candidate_remote_ids: tuple[str, ...] = (),
) -> None:
    if not templates:
        if has_more:
            if synthetic_tombstone_remote_ids:
                raise ValueError(
                    "synthetic tombstones are forbidden on non-final prepared pages"
                )
            return
        if synthetic_tombstone_remote_ids != remaining_candidate_remote_ids:
            raise ValueError(
                "final page synthetic tombstones must equal remaining candidates"
            )
        if synthetic_tombstone_remote_ids:
            raise ValueError(
                "final page requires mutation templates for synthetic tombstones"
            )
        return
    ordered = tuple(sorted(templates, key=lambda template: template.remote_id))
    if templates != ordered:
        raise ValueError("prepared state mutation templates must be UTF-8 ascending")
    seen: set[str] = set()
    synthetic_template_ids: set[str] = set()
    for template in templates:
        if template.remote_id in seen:
            raise ValueError(
                "prepared state mutation templates must be unique by remote_id"
            )
        seen.add(template.remote_id)
        if template.binding_configuration_version != binding_configuration_version:
            raise ValueError(
                "prepared state mutation template binding_configuration_version mismatch"
            )
        if (
            template.reconciliation_semantic
            is KnowledgeReconciliationMutationSemantic.ABSENT_FROM_COMPLETED_SYNCHRONIZED_SOURCE_INVENTORY
        ):
            synthetic_template_ids.add(template.remote_id)
    if has_more:
        if synthetic_tombstone_remote_ids:
            raise ValueError(
                "synthetic tombstones are forbidden on non-final prepared pages"
            )
        if synthetic_template_ids:
            raise ValueError(
                "synthetic reconciliation semantic templates are forbidden on non-final pages"
            )
        return
    if synthetic_template_ids != set(synthetic_tombstone_remote_ids):
        raise ValueError(
            "synthetic tombstone templates must match synthetic tombstone IDs"
        )
    if synthetic_tombstone_remote_ids != remaining_candidate_remote_ids:
        raise ValueError(
            "final page synthetic tombstones must equal remaining candidates"
        )


def _validate_mutation_templates(
    templates: tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...],
    *,
    binding_configuration_version: int,
    has_more: bool,
    synthetic_tombstone_remote_ids: tuple[str, ...],
    remaining_candidate_remote_ids: tuple[str, ...] = (),
    policy: KnowledgeReconciliationLimitPolicy,
) -> None:
    _validate_mutation_templates_structural(
        templates,
        binding_configuration_version=binding_configuration_version,
        has_more=has_more,
        synthetic_tombstone_remote_ids=synthetic_tombstone_remote_ids,
        remaining_candidate_remote_ids=remaining_candidate_remote_ids,
    )
    for template in templates:
        _validate_remote_id_byte_length(
            template.remote_id,
            policy=policy,
            field_name="prepared_state_mutation_templates",
        )


def _validate_page_prepared_structural(
    *,
    prepared_input_cursor: KnowledgeCursor | None,
    prepared_input_cursor_fingerprint: str,
    prepared_proposed_checkpoint: KnowledgeCursor | None,
    prepared_proposed_checkpoint_fingerprint: str,
    prepared_next_cursor: KnowledgeCursor | None,
    prepared_next_cursor_fingerprint: str,
    prepared_state_mutation_templates: tuple[
        KnowledgeReconciliationPreparedStateMutationTemplate, ...
    ],
    prepared_state_mutations_fingerprint: str,
    has_more: bool,
    remaining_candidate_remote_ids: tuple[str, ...],
    synthetic_tombstone_remote_ids: tuple[str, ...],
    binding_configuration_version: int,
) -> None:
    _validate_cursor_fingerprint_pair(
        cursor=prepared_input_cursor,
        fingerprint=prepared_input_cursor_fingerprint,
        field_prefix="prepared_input",
    )
    _validate_cursor_fingerprint_pair(
        cursor=prepared_proposed_checkpoint,
        fingerprint=prepared_proposed_checkpoint_fingerprint,
        field_prefix="prepared_proposed_checkpoint",
    )
    _validate_cursor_fingerprint_pair(
        cursor=prepared_next_cursor,
        fingerprint=prepared_next_cursor_fingerprint,
        field_prefix="prepared_next",
    )
    expected_mutations = canonical_prepared_state_mutations_fingerprint(
        prepared_state_mutation_templates
    )
    if prepared_state_mutations_fingerprint != expected_mutations:
        raise ValueError("prepared_state_mutations_fingerprint mismatch")
    if has_more:
        if prepared_next_cursor is None:
            raise ValueError("has_more requires prepared_next_cursor")
    elif prepared_next_cursor is not None:
        raise ValueError("final page must not retain prepared_next_cursor")
    _validate_mutation_templates_structural(
        prepared_state_mutation_templates,
        binding_configuration_version=binding_configuration_version,
        has_more=has_more,
        synthetic_tombstone_remote_ids=synthetic_tombstone_remote_ids,
        remaining_candidate_remote_ids=remaining_candidate_remote_ids,
    )


def _reconciliation_run_durable_document_payload(
    run: "KnowledgeReconciliationRun",
) -> dict[str, Any]:
    return {
        "schema_version": _RECONCILIATION_RUN_SCHEMA,
        "tenant_id": run.tenant_id,
        "binding_id": run.binding_id,
        "record_version": run.record_version,
        "run": run.model_dump(mode="json"),
    }


def reconciliation_run_durable_document_bytes(
    run: "KnowledgeReconciliationRun",
) -> bytes:
    return _canonical_json_bytes(_reconciliation_run_durable_document_payload(run))


def _validate_cursor_fingerprint_pair(
    *,
    cursor: KnowledgeCursor | None,
    fingerprint: str | None,
    field_prefix: str,
) -> None:
    if fingerprint is None:
        raise ValueError(f"{field_prefix} fingerprint is required")
    expected = knowledge_cursor_fingerprint_sha256(cursor)
    if fingerprint != expected:
        raise ValueError(f"{field_prefix} fingerprint does not match cursor")


def _validate_applied_page_evidence(
    *,
    applied_page_count: int,
    last_applied_delivery_id: str | None,
    last_applied_parent_delivery_id: str | None,
) -> None:
    if applied_page_count == 0 and last_applied_delivery_id is not None:
        raise ValueError(
            "last_applied_delivery_id must be null when applied_page_count is zero"
        )
    if applied_page_count > 0 and last_applied_delivery_id is None:
        raise ValueError(
            "last_applied_delivery_id is required when applied_page_count is positive"
        )
    if applied_page_count == 0 and last_applied_parent_delivery_id is not None:
        raise ValueError(
            "last_applied_parent_delivery_id must be null when applied_page_count is zero"
        )


class _ReconciliationRunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    binding_configuration_version: int = Field(ge=1)
    provider_id: str
    source_kind: str
    run_id: str
    record_version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    applied_page_count: int = Field(default=0, ge=0)
    last_applied_delivery_id: str | None = None
    last_applied_parent_delivery_id: str | None = None
    superseded_run_id: str | None = None
    expected_base_completed_checkpoint: KnowledgeSyncCheckpoint | None = None

    @field_validator(
        "tenant_id",
        "binding_id",
        "provider_id",
        "source_kind",
        "run_id",
        "superseded_run_id",
    )
    @classmethod
    def _non_empty_ids(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("last_applied_delivery_id", "last_applied_parent_delivery_id")
    @classmethod
    def _optional_delivery_id(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_sha256_hex(value, field_name=field_name)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _utc_timestamps(cls, value: datetime, info: ValidationInfo) -> datetime:
        field_name = info.field_name or "field"
        return _assert_utc_aware(value, field_name=field_name)

    @model_validator(mode="after")
    def _applied_page_rules(self) -> _ReconciliationRunIdentity:
        _validate_applied_page_evidence(
            applied_page_count=self.applied_page_count,
            last_applied_delivery_id=self.last_applied_delivery_id,
            last_applied_parent_delivery_id=self.last_applied_parent_delivery_id,
        )
        _validate_base_completed_checkpoint(
            self.expected_base_completed_checkpoint,
            tenant_id=self.tenant_id,
            binding_id=self.binding_id,
        )
        return self

    @property
    def effects_started(self) -> bool:
        return self.applied_page_count > 0


class KnowledgeReconciliationRunCollecting(_ReconciliationRunIdentity):
    phase: Literal[KnowledgeReconciliationRunPhase.COLLECTING] = (
        KnowledgeReconciliationRunPhase.COLLECTING
    )
    current_input_cursor: KnowledgeCursor | None = Field(default=None, repr=False)
    current_input_cursor_fingerprint: str
    remaining_candidate_remote_ids: tuple[str, ...] = ()
    candidate_inventory_continuation_token: None = None

    @field_validator("remaining_candidate_remote_ids")
    @classmethod
    def _candidate_ids(cls, value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return _validate_remote_id_tuple(
            value,
            field_name="remaining_candidate_remote_ids",
            allow_empty=True,
        )

    @field_validator("current_input_cursor_fingerprint")
    @classmethod
    def _fingerprint(cls, value: str) -> str:
        return _require_sha256_hex(value, field_name="current_input_cursor_fingerprint")

    @model_validator(mode="after")
    def _cursor_pair(self) -> KnowledgeReconciliationRunCollecting:
        _validate_cursor_fingerprint_pair(
            cursor=self.current_input_cursor,
            fingerprint=self.current_input_cursor_fingerprint,
            field_prefix="current_input",
        )
        return self


class KnowledgeReconciliationRunPagePrepared(_ReconciliationRunIdentity):
    phase: Literal[KnowledgeReconciliationRunPhase.PAGE_PREPARED] = (
        KnowledgeReconciliationRunPhase.PAGE_PREPARED
    )
    prepared_input_cursor: KnowledgeCursor | None = Field(default=None, repr=False)
    prepared_input_cursor_fingerprint: str
    provider_page_fingerprint: str
    prepared_batch_payload_fingerprint: str
    prepared_state_mutation_templates: tuple[
        KnowledgeReconciliationPreparedStateMutationTemplate, ...
    ]
    prepared_state_mutations_fingerprint: str
    prepared_proposed_checkpoint: KnowledgeCursor | None = Field(
        default=None, repr=False
    )
    prepared_proposed_checkpoint_fingerprint: str
    prepared_next_cursor: KnowledgeCursor | None = Field(default=None, repr=False)
    prepared_next_cursor_fingerprint: str
    prepared_page_size: int = Field(ge=1, le=1000)
    has_more: bool
    delivery_id: str
    prepared_parent_delivery_id: str | None = None
    remaining_candidate_remote_ids: tuple[str, ...]
    synthetic_tombstone_remote_ids: tuple[str, ...] = ()

    @field_validator(
        "prepared_input_cursor_fingerprint",
        "provider_page_fingerprint",
        "prepared_batch_payload_fingerprint",
        "prepared_state_mutations_fingerprint",
        "prepared_proposed_checkpoint_fingerprint",
        "prepared_next_cursor_fingerprint",
        "delivery_id",
        "prepared_parent_delivery_id",
    )
    @classmethod
    def _sha256_fields(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_sha256_hex(value, field_name=field_name)

    @field_validator("prepared_state_mutation_templates")
    @classmethod
    def _immutable_templates(
        cls,
        value: tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...]
        | list[KnowledgeReconciliationPreparedStateMutationTemplate],
    ) -> tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...]:
        return tuple(value)

    @field_validator("remaining_candidate_remote_ids", "synthetic_tombstone_remote_ids")
    @classmethod
    def _remote_id_tuples(cls, value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return _validate_remote_id_tuple(
            value,
            field_name="remote_id_tuple",
            allow_empty=True,
        )

    @model_validator(mode="after")
    def _page_prepared_rules(self) -> KnowledgeReconciliationRunPagePrepared:
        _validate_page_prepared_structural(
            prepared_input_cursor=self.prepared_input_cursor,
            prepared_input_cursor_fingerprint=self.prepared_input_cursor_fingerprint,
            prepared_proposed_checkpoint=self.prepared_proposed_checkpoint,
            prepared_proposed_checkpoint_fingerprint=self.prepared_proposed_checkpoint_fingerprint,
            prepared_next_cursor=self.prepared_next_cursor,
            prepared_next_cursor_fingerprint=self.prepared_next_cursor_fingerprint,
            prepared_state_mutation_templates=self.prepared_state_mutation_templates,
            prepared_state_mutations_fingerprint=self.prepared_state_mutations_fingerprint,
            has_more=self.has_more,
            remaining_candidate_remote_ids=self.remaining_candidate_remote_ids,
            synthetic_tombstone_remote_ids=self.synthetic_tombstone_remote_ids,
            binding_configuration_version=self.binding_configuration_version,
        )
        return self


class KnowledgeReconciliationRunFinalizing(_ReconciliationRunIdentity):
    phase: Literal[KnowledgeReconciliationRunPhase.FINALIZING] = (
        KnowledgeReconciliationRunPhase.FINALIZING
    )
    intended_final_completed_checkpoint: KnowledgeSyncCheckpoint
    intended_final_checkpoint_fingerprint: str
    expected_previous_completed_checkpoint: KnowledgeSyncCheckpoint | None
    final_delivery_id: str
    prepared_batch_payload_fingerprint: str

    @field_validator(
        "intended_final_checkpoint_fingerprint",
        "final_delivery_id",
        "prepared_batch_payload_fingerprint",
    )
    @classmethod
    def _sha256_fields(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_sha256_hex(value, field_name=field_name)

    @model_validator(mode="after")
    def _finalizing_rules(self) -> KnowledgeReconciliationRunFinalizing:
        expected = knowledge_sync_checkpoint_fingerprint_sha256(
            self.intended_final_completed_checkpoint
        )
        if self.intended_final_checkpoint_fingerprint != expected:
            raise ValueError("intended_final_checkpoint_fingerprint mismatch")
        if (
            self.intended_final_completed_checkpoint.tenant_id != self.tenant_id
            or self.intended_final_completed_checkpoint.binding_id != self.binding_id
        ):
            raise ValueError("intended final checkpoint identity mismatch")
        if self.expected_previous_completed_checkpoint is not None and (
            self.expected_previous_completed_checkpoint.tenant_id != self.tenant_id
            or self.expected_previous_completed_checkpoint.binding_id != self.binding_id
        ):
            raise ValueError("expected previous checkpoint identity mismatch")
        if (
            self.expected_previous_completed_checkpoint
            != self.expected_base_completed_checkpoint
        ):
            raise ValueError(
                "expected_previous_completed_checkpoint must equal durable base checkpoint"
            )
        if (
            self.intended_final_completed_checkpoint.binding_configuration_version
            != self.binding_configuration_version
        ):
            raise ValueError("intended final checkpoint configuration version mismatch")
        return self


class KnowledgeReconciliationRunCompleted(_ReconciliationRunIdentity):
    phase: Literal[KnowledgeReconciliationRunPhase.COMPLETED] = (
        KnowledgeReconciliationRunPhase.COMPLETED
    )
    committed_completed_checkpoint: KnowledgeSyncCheckpoint
    final_delivery_id: str

    @field_validator("final_delivery_id")
    @classmethod
    def _delivery_id(cls, value: str) -> str:
        return _require_sha256_hex(value, field_name="final_delivery_id")

    @model_validator(mode="after")
    def _completed_rules(self) -> KnowledgeReconciliationRunCompleted:
        if (
            self.committed_completed_checkpoint.tenant_id != self.tenant_id
            or self.committed_completed_checkpoint.binding_id != self.binding_id
        ):
            raise ValueError("committed checkpoint identity mismatch")
        if (
            self.committed_completed_checkpoint.binding_configuration_version
            != self.binding_configuration_version
        ):
            raise ValueError("committed checkpoint configuration version mismatch")
        return self


def _validate_checkpoint_matches_run_identity(
    checkpoint: KnowledgeSyncCheckpoint | None,
    *,
    tenant_id: str,
    binding_id: str,
    binding_configuration_version: int | None = None,
) -> None:
    if checkpoint is None:
        return
    if checkpoint.tenant_id != tenant_id or checkpoint.binding_id != binding_id:
        raise ValueError("recovery evidence checkpoint identity mismatch")
    if (
        binding_configuration_version is not None
        and checkpoint.binding_configuration_version != binding_configuration_version
    ):
        raise ValueError(
            "recovery evidence checkpoint binding_configuration_version mismatch"
        )


class KnowledgeReconciliationRecoveryEvidenceCollecting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    origin_phase: Literal[KnowledgeReconciliationRunPhase.COLLECTING] = (
        KnowledgeReconciliationRunPhase.COLLECTING
    )
    current_input_cursor: KnowledgeCursor | None = Field(default=None, repr=False)
    current_input_cursor_fingerprint: str
    remaining_candidate_remote_ids: tuple[str, ...] = ()
    expected_base_completed_checkpoint: KnowledgeSyncCheckpoint | None = None

    @field_validator("current_input_cursor_fingerprint")
    @classmethod
    def _fingerprint(cls, value: str) -> str:
        return _require_sha256_hex(value, field_name="current_input_cursor_fingerprint")

    @field_validator("remaining_candidate_remote_ids")
    @classmethod
    def _candidate_ids(cls, value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        if not value:
            return ()
        return _validate_remote_id_tuple(
            value,
            field_name="remaining_candidate_remote_ids",
            allow_empty=False,
        )

    @model_validator(mode="after")
    def _cursor_pair(self) -> KnowledgeReconciliationRecoveryEvidenceCollecting:
        _validate_cursor_fingerprint_pair(
            cursor=self.current_input_cursor,
            fingerprint=self.current_input_cursor_fingerprint,
            field_prefix="current_input",
        )
        return self


class KnowledgeReconciliationRecoveryEvidencePagePrepared(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    origin_phase: Literal[KnowledgeReconciliationRunPhase.PAGE_PREPARED] = (
        KnowledgeReconciliationRunPhase.PAGE_PREPARED
    )
    prepared_input_cursor: KnowledgeCursor | None = Field(default=None, repr=False)
    prepared_input_cursor_fingerprint: str
    provider_page_fingerprint: str
    prepared_batch_payload_fingerprint: str
    prepared_state_mutation_templates: tuple[
        KnowledgeReconciliationPreparedStateMutationTemplate, ...
    ]
    prepared_state_mutations_fingerprint: str
    prepared_proposed_checkpoint: KnowledgeCursor | None = Field(
        default=None, repr=False
    )
    prepared_proposed_checkpoint_fingerprint: str
    prepared_next_cursor: KnowledgeCursor | None = Field(default=None, repr=False)
    prepared_next_cursor_fingerprint: str
    prepared_page_size: int = Field(ge=1, le=1000)
    has_more: bool
    delivery_id: str
    prepared_parent_delivery_id: str | None = None
    remaining_candidate_remote_ids: tuple[str, ...]
    synthetic_tombstone_remote_ids: tuple[str, ...] = ()
    expected_base_completed_checkpoint: KnowledgeSyncCheckpoint | None = None

    @field_validator(
        "prepared_input_cursor_fingerprint",
        "provider_page_fingerprint",
        "prepared_batch_payload_fingerprint",
        "prepared_state_mutations_fingerprint",
        "prepared_proposed_checkpoint_fingerprint",
        "prepared_next_cursor_fingerprint",
        "delivery_id",
        "prepared_parent_delivery_id",
    )
    @classmethod
    def _sha256_fields(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_sha256_hex(value, field_name=field_name)

    @field_validator("prepared_state_mutation_templates")
    @classmethod
    def _immutable_templates(
        cls,
        value: tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...]
        | list[KnowledgeReconciliationPreparedStateMutationTemplate],
    ) -> tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...]:
        return tuple(value)

    @field_validator("remaining_candidate_remote_ids", "synthetic_tombstone_remote_ids")
    @classmethod
    def _remote_id_tuples(cls, value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return _validate_remote_id_tuple(
            value,
            field_name="remote_id_tuple",
            allow_empty=True,
        )


class KnowledgeReconciliationRecoveryEvidenceFinalizing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    origin_phase: Literal[KnowledgeReconciliationRunPhase.FINALIZING] = (
        KnowledgeReconciliationRunPhase.FINALIZING
    )
    intended_final_completed_checkpoint: KnowledgeSyncCheckpoint
    intended_final_checkpoint_fingerprint: str
    expected_previous_completed_checkpoint: KnowledgeSyncCheckpoint | None
    final_delivery_id: str
    prepared_batch_payload_fingerprint: str
    expected_base_completed_checkpoint: KnowledgeSyncCheckpoint | None = None

    @field_validator(
        "intended_final_checkpoint_fingerprint",
        "final_delivery_id",
        "prepared_batch_payload_fingerprint",
    )
    @classmethod
    def _sha256_fields(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_sha256_hex(value, field_name=field_name)

    @model_validator(mode="after")
    def _finalizing_evidence_rules(
        self,
    ) -> KnowledgeReconciliationRecoveryEvidenceFinalizing:
        expected = knowledge_sync_checkpoint_fingerprint_sha256(
            self.intended_final_completed_checkpoint
        )
        if self.intended_final_checkpoint_fingerprint != expected:
            raise ValueError("intended_final_checkpoint_fingerprint mismatch")
        if (
            self.expected_previous_completed_checkpoint
            != self.expected_base_completed_checkpoint
        ):
            raise ValueError(
                "expected_previous_completed_checkpoint must equal durable base checkpoint"
            )
        return self


KnowledgeReconciliationRecoveryEvidence = Annotated[
    Union[
        KnowledgeReconciliationRecoveryEvidenceCollecting,
        KnowledgeReconciliationRecoveryEvidencePagePrepared,
        KnowledgeReconciliationRecoveryEvidenceFinalizing,
    ],
    Field(discriminator="origin_phase"),
]


def _validate_recovery_evidence_collecting_structural(
    evidence: KnowledgeReconciliationRecoveryEvidenceCollecting,
) -> None:
    _validate_cursor_fingerprint_pair(
        cursor=evidence.current_input_cursor,
        fingerprint=evidence.current_input_cursor_fingerprint,
        field_prefix="current_input",
    )
    if evidence.remaining_candidate_remote_ids:
        _validate_remote_id_tuple(
            evidence.remaining_candidate_remote_ids,
            field_name="remaining_candidate_remote_ids",
            allow_empty=False,
        )


def _validate_recovery_evidence_page_prepared_structural(
    evidence: KnowledgeReconciliationRecoveryEvidencePagePrepared,
    *,
    binding_configuration_version: int,
) -> None:
    _validate_page_prepared_structural(
        prepared_input_cursor=evidence.prepared_input_cursor,
        prepared_input_cursor_fingerprint=evidence.prepared_input_cursor_fingerprint,
        prepared_proposed_checkpoint=evidence.prepared_proposed_checkpoint,
        prepared_proposed_checkpoint_fingerprint=evidence.prepared_proposed_checkpoint_fingerprint,
        prepared_next_cursor=evidence.prepared_next_cursor,
        prepared_next_cursor_fingerprint=evidence.prepared_next_cursor_fingerprint,
        prepared_state_mutation_templates=evidence.prepared_state_mutation_templates,
        prepared_state_mutations_fingerprint=evidence.prepared_state_mutations_fingerprint,
        has_more=evidence.has_more,
        remaining_candidate_remote_ids=evidence.remaining_candidate_remote_ids,
        synthetic_tombstone_remote_ids=evidence.synthetic_tombstone_remote_ids,
        binding_configuration_version=binding_configuration_version,
    )


def _validate_recovery_evidence_finalizing_structural(
    evidence: KnowledgeReconciliationRecoveryEvidenceFinalizing,
    *,
    tenant_id: str,
    binding_id: str,
    binding_configuration_version: int,
) -> None:
    expected = knowledge_sync_checkpoint_fingerprint_sha256(
        evidence.intended_final_completed_checkpoint
    )
    if evidence.intended_final_checkpoint_fingerprint != expected:
        raise ValueError("intended_final_checkpoint_fingerprint mismatch")
    _validate_checkpoint_matches_run_identity(
        evidence.intended_final_completed_checkpoint,
        tenant_id=tenant_id,
        binding_id=binding_id,
        binding_configuration_version=binding_configuration_version,
    )
    _validate_checkpoint_matches_run_identity(
        evidence.expected_previous_completed_checkpoint,
        tenant_id=tenant_id,
        binding_id=binding_id,
    )
    if (
        evidence.expected_previous_completed_checkpoint
        != evidence.expected_base_completed_checkpoint
    ):
        raise ValueError(
            "expected_previous_completed_checkpoint must equal durable base checkpoint"
        )


def validate_recovery_required_run(
    run: "KnowledgeReconciliationRunRecoveryRequired",
) -> None:
    evidence = run.recovery_evidence
    if (
        evidence.expected_base_completed_checkpoint
        != run.expected_base_completed_checkpoint
    ):
        raise ValueError("recovery evidence base checkpoint mismatch")
    _validate_checkpoint_matches_run_identity(
        evidence.expected_base_completed_checkpoint,
        tenant_id=run.tenant_id,
        binding_id=run.binding_id,
    )
    if isinstance(evidence, KnowledgeReconciliationRecoveryEvidenceCollecting):
        _validate_recovery_evidence_collecting_structural(evidence)
        return
    if isinstance(evidence, KnowledgeReconciliationRecoveryEvidencePagePrepared):
        _validate_recovery_evidence_page_prepared_structural(
            evidence,
            binding_configuration_version=run.binding_configuration_version,
        )
        return
    if isinstance(evidence, KnowledgeReconciliationRecoveryEvidenceFinalizing):
        _validate_recovery_evidence_finalizing_structural(
            evidence,
            tenant_id=run.tenant_id,
            binding_id=run.binding_id,
            binding_configuration_version=run.binding_configuration_version,
        )
        return
    raise ValueError("recovery evidence origin phase is unsupported")


def recovery_evidence_from_run(
    run: KnowledgeReconciliationRunCollecting
    | KnowledgeReconciliationRunPagePrepared
    | KnowledgeReconciliationRunFinalizing,
) -> KnowledgeReconciliationRecoveryEvidence:
    if isinstance(run, KnowledgeReconciliationRunCollecting):
        return KnowledgeReconciliationRecoveryEvidenceCollecting(
            current_input_cursor=run.current_input_cursor,
            current_input_cursor_fingerprint=run.current_input_cursor_fingerprint,
            remaining_candidate_remote_ids=run.remaining_candidate_remote_ids,
            expected_base_completed_checkpoint=run.expected_base_completed_checkpoint,
        )
    if isinstance(run, KnowledgeReconciliationRunPagePrepared):
        return KnowledgeReconciliationRecoveryEvidencePagePrepared(
            prepared_input_cursor=run.prepared_input_cursor,
            prepared_input_cursor_fingerprint=run.prepared_input_cursor_fingerprint,
            provider_page_fingerprint=run.provider_page_fingerprint,
            prepared_batch_payload_fingerprint=run.prepared_batch_payload_fingerprint,
            prepared_state_mutation_templates=run.prepared_state_mutation_templates,
            prepared_state_mutations_fingerprint=run.prepared_state_mutations_fingerprint,
            prepared_proposed_checkpoint=run.prepared_proposed_checkpoint,
            prepared_proposed_checkpoint_fingerprint=run.prepared_proposed_checkpoint_fingerprint,
            prepared_next_cursor=run.prepared_next_cursor,
            prepared_next_cursor_fingerprint=run.prepared_next_cursor_fingerprint,
            prepared_page_size=run.prepared_page_size,
            has_more=run.has_more,
            delivery_id=run.delivery_id,
            prepared_parent_delivery_id=run.prepared_parent_delivery_id,
            remaining_candidate_remote_ids=run.remaining_candidate_remote_ids,
            synthetic_tombstone_remote_ids=run.synthetic_tombstone_remote_ids,
            expected_base_completed_checkpoint=run.expected_base_completed_checkpoint,
        )
    return KnowledgeReconciliationRecoveryEvidenceFinalizing(
        intended_final_completed_checkpoint=run.intended_final_completed_checkpoint,
        intended_final_checkpoint_fingerprint=run.intended_final_checkpoint_fingerprint,
        expected_previous_completed_checkpoint=run.expected_previous_completed_checkpoint,
        final_delivery_id=run.final_delivery_id,
        prepared_batch_payload_fingerprint=run.prepared_batch_payload_fingerprint,
        expected_base_completed_checkpoint=run.expected_base_completed_checkpoint,
    )


class KnowledgeReconciliationRunRecoveryRequired(_ReconciliationRunIdentity):
    phase: Literal[KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED] = (
        KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED
    )
    recovery_reason_code: str
    machine_recovery_reason_code: str | None = None
    recovery_evidence: KnowledgeReconciliationRecoveryEvidence

    @field_validator("recovery_reason_code")
    @classmethod
    def _safe_recovery_reason_code(cls, value: str) -> str:
        return _require_operator_reason_code(value, field_name="recovery_reason_code")

    @model_validator(mode="after")
    def _recovery_evidence_consistency(
        self,
    ) -> KnowledgeReconciliationRunRecoveryRequired:
        validate_recovery_required_run(self)
        return self


class KnowledgeReconciliationRunAborted(_ReconciliationRunIdentity):
    phase: Literal[KnowledgeReconciliationRunPhase.ABORTED] = (
        KnowledgeReconciliationRunPhase.ABORTED
    )
    operator_reason_code: str

    @field_validator("operator_reason_code")
    @classmethod
    def _safe_operator_reason_code(cls, value: str) -> str:
        return _require_operator_reason_code(value, field_name="operator_reason_code")


KnowledgeReconciliationRun = Annotated[
    Union[
        KnowledgeReconciliationRunCollecting,
        KnowledgeReconciliationRunPagePrepared,
        KnowledgeReconciliationRunFinalizing,
        KnowledgeReconciliationRunCompleted,
        KnowledgeReconciliationRunRecoveryRequired,
        KnowledgeReconciliationRunAborted,
    ],
    Field(discriminator="phase"),
]


def parse_knowledge_reconciliation_run(
    payload: Mapping[str, Any],
) -> KnowledgeReconciliationRun:
    phase = payload.get("phase")
    if phase is None:
        raise ValueError("reconciliation run phase is required")
    try:
        resolved = KnowledgeReconciliationRunPhase(str(phase))
    except ValueError as exc:
        raise ValueError("reconciliation run phase is unknown") from exc
    model_by_phase = {
        KnowledgeReconciliationRunPhase.COLLECTING: KnowledgeReconciliationRunCollecting,
        KnowledgeReconciliationRunPhase.PAGE_PREPARED: KnowledgeReconciliationRunPagePrepared,
        KnowledgeReconciliationRunPhase.FINALIZING: KnowledgeReconciliationRunFinalizing,
        KnowledgeReconciliationRunPhase.COMPLETED: KnowledgeReconciliationRunCompleted,
        KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED: KnowledgeReconciliationRunRecoveryRequired,
        KnowledgeReconciliationRunPhase.ABORTED: KnowledgeReconciliationRunAborted,
    }
    return model_by_phase[resolved].model_validate(payload)


def validate_reconciliation_candidate_inventory(
    remote_ids: tuple[str, ...],
    *,
    policy: KnowledgeReconciliationLimitPolicy,
) -> None:
    if len(remote_ids) > policy.max_reconciliation_candidate_count:
        raise ValueError("reconciliation candidate count exceeds configured limit")
    for remote_id in remote_ids:
        _validate_remote_id_byte_length(
            remote_id,
            policy=policy,
            field_name="remaining_candidate_remote_ids",
        )
    payload = canonical_reconciliation_candidate_inventory_bytes(remote_ids)
    if len(payload) > policy.max_reconciliation_candidate_payload_bytes:
        raise ValueError("reconciliation candidate payload exceeds configured limit")


def validate_reconciliation_prepared_intent(
    run: KnowledgeReconciliationRunPagePrepared,
    *,
    policy: KnowledgeReconciliationLimitPolicy,
) -> None:
    validate_reconciliation_candidate_inventory(
        run.remaining_candidate_remote_ids,
        policy=policy,
    )
    if (
        len(run.prepared_state_mutation_templates)
        > policy.max_reconciliation_prepared_state_mutation_count
    ):
        raise ValueError("prepared state mutation count exceeds configured limit")
    _validate_mutation_templates(
        run.prepared_state_mutation_templates,
        binding_configuration_version=run.binding_configuration_version,
        has_more=run.has_more,
        synthetic_tombstone_remote_ids=run.synthetic_tombstone_remote_ids,
        remaining_candidate_remote_ids=run.remaining_candidate_remote_ids,
        policy=policy,
    )
    for remote_id in run.synthetic_tombstone_remote_ids:
        _validate_remote_id_byte_length(
            remote_id,
            policy=policy,
            field_name="synthetic_tombstone_remote_ids",
        )
    for remote_id in run.remaining_candidate_remote_ids:
        _validate_remote_id_byte_length(
            remote_id,
            policy=policy,
            field_name="remaining_candidate_remote_ids",
        )
    payload = reconciliation_run_durable_document_bytes(run)
    if len(payload) > policy.max_reconciliation_prepared_intent_payload_bytes:
        raise ValueError("prepared intent payload exceeds configured limit")
