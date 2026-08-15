# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral durable materialization sink for Vendor Knowledge sync batches.

Applications host this DocumentStore-backed implementation (or inject their own
``KnowledgeSyncSink``). Indexing / RAG is intentionally out of scope.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeItemDescriptor,
    KnowledgeItemRevision,
    KnowledgePermissions,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import KnowledgeSyncCorruptState
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncBatch,
    KnowledgeSyncEnvelope,
    KnowledgeSyncSinkReceipt,
    KnowledgeSyncSinkReceiptStatus,
)

_ITEM_SCHEMA = "vendor_knowledge.durable_item.v1"
_RECEIPT_SCHEMA = "vendor_knowledge.durable_delivery_receipt.v1"
_SEQUENCE_SCHEMA = "vendor_knowledge.durable_sequence.v1"

_ITEM_PARTITION_PREFIX = "vendor_knowledge.durable_item.v1"
_RECEIPT_PARTITION_PREFIX = "vendor_knowledge.durable_delivery.v1"
_SEQUENCE_PARTITION_PREFIX = "vendor_knowledge.durable_sequence.v1"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

_FORBIDDEN_SECRET_FIELDS: frozenset[str] = frozenset(
    {
        "access_token",
        "refresh_token",
        "api_key",
        "password",
        "client_secret",
        "authorization_header",
        "signed_download_url",
    }
)

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


class DurableMaterializedItemStatus(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"
    REVOKED = "revoked"


class DurableDeliveryReceiptStatus(StrEnum):
    APPLYING = "applying"
    APPLIED = "applied"


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


def _canonical_json_bytes(payload: Mapping[str, Any] | list[Any] | dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def durable_batch_payload_fingerprint(batch: KnowledgeSyncBatch) -> str:
    """Hash the durable delivery payload without lifecycle secrets."""
    payload = {
        "tenant_id": batch.tenant_id,
        "binding_id": batch.binding_id,
        "binding_configuration_version": batch.binding_configuration_version,
        "source": batch.source.model_dump(mode="json"),
        "mode": batch.mode.value,
        "delivery_id": batch.delivery_id,
        "envelopes": [envelope.model_dump(mode="json") for envelope in batch.envelopes],
        "has_more": batch.has_more,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def knowledge_item_revision_order_key(
    revision: KnowledgeItemRevision | None,
) -> tuple[object, ...]:
    """Deterministic total order for revision compare (newer sorts greater)."""
    if revision is None:
        return (0, "", "", "", "", "")
    updated = revision.updated_at.isoformat() if revision.updated_at is not None else ""
    return (
        1,
        updated,
        revision.version or "",
        revision.etag or "",
        revision.content_hash or "",
        revision.acl_hash or "",
    )


def _reject_secret_fields(data: Mapping[str, Any], *, kind: str) -> None:
    for key in _FORBIDDEN_SECRET_FIELDS:
        if key in data:
            raise KnowledgeSyncCorruptState(
                f"{kind} must not contain secret-bearing fields"
            )


def _item_partition_key(*, tenant_id: str, binding_id: str) -> str:
    return (
        f"{_ITEM_PARTITION_PREFIX}:"
        f"{_require_non_empty(tenant_id, field_name='tenant_id')}:"
        f"{_require_non_empty(binding_id, field_name='binding_id')}"
    )


def _item_row_key(remote_id: str) -> str:
    digest = hashlib.sha256(
        _require_non_empty(remote_id, field_name="remote_id").encode("utf-8")
    ).hexdigest()
    return f"item:{digest}"


def _receipt_partition_key(*, tenant_id: str, binding_id: str) -> str:
    return (
        f"{_RECEIPT_PARTITION_PREFIX}:"
        f"{_require_non_empty(tenant_id, field_name='tenant_id')}:"
        f"{_require_non_empty(binding_id, field_name='binding_id')}"
    )


def _receipt_row_key(delivery_id: str) -> str:
    return f"delivery:{_require_sha256_hex(delivery_id, field_name='delivery_id')}"


def _sequence_partition_key(tenant_id: str) -> str:
    return (
        f"{_SEQUENCE_PARTITION_PREFIX}:"
        f"{_require_non_empty(tenant_id, field_name='tenant_id')}"
    )


def _sequence_row_key(binding_id: str) -> str:
    return f"binding:{_require_non_empty(binding_id, field_name='binding_id')}"


def _require_conditional_document_store(
    document_store: DocumentStore,
) -> ConditionalDocumentStore:
    if not isinstance(document_store, ConditionalDocumentStore):
        raise TypeError(
            "durable materialization sink requires ConditionalDocumentStore"
        )
    return document_store


def _data_as_dict(data: Mapping[str, Any] | object) -> dict[str, Any]:
    if isinstance(data, Mapping):
        return dict(data)
    raise KnowledgeSyncCorruptState("durable document payload is corrupt")


class DurableMaterializedItemV1(BaseModel):
    """Canonical durable application state for one remote item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    binding_configuration_version: int = Field(ge=1)
    provider_id: str
    source_kind: str
    remote_id: str
    status: DurableMaterializedItemStatus
    materialization_generation: int = Field(ge=1)
    last_delivery_id: str
    revision: KnowledgeItemRevision | None = None
    descriptor: KnowledgeItemDescriptor | None = None
    content: KnowledgeContent | None = None
    permissions: KnowledgePermissions | None = None
    record_version: str

    @field_validator(
        "tenant_id",
        "binding_id",
        "provider_id",
        "source_kind",
        "remote_id",
        "record_version",
    )
    @classmethod
    def _non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_empty(value, field_name=info.field_name or "field")

    @field_validator("last_delivery_id")
    @classmethod
    def _delivery_hash(cls, value: str) -> str:
        return _require_sha256_hex(value, field_name="last_delivery_id")


class DurableDeliveryReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    binding_configuration_version: int = Field(ge=1)
    delivery_id: str
    status: DurableDeliveryReceiptStatus
    payload_fingerprint: str
    materialization_generation: int = Field(ge=1)
    record_version: str

    @field_validator("tenant_id", "binding_id", "record_version")
    @classmethod
    def _non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_empty(value, field_name=info.field_name or "field")

    @field_validator("delivery_id", "payload_fingerprint")
    @classmethod
    def _hashes(cls, value: str, info: ValidationInfo) -> str:
        return _require_sha256_hex(value, field_name=info.field_name or "field")


class DocumentStoreDurableKnowledgeSyncSink:
    """DocumentStore-backed durable materialization implementing ``KnowledgeSyncSink``.

    Persists provider-neutral ``KnowledgeSyncEnvelope`` payloads. Does not index,
    render provider-specific markdown, or branch on provider identity.
    """

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = _require_conditional_document_store(document_store)

    async def apply_batch(self, *, batch: KnowledgeSyncBatch) -> None:
        fingerprint = durable_batch_payload_fingerprint(batch)
        existing = self._get_receipt(
            tenant_id=batch.tenant_id,
            binding_id=batch.binding_id,
            delivery_id=batch.delivery_id,
        )
        if existing is not None:
            if existing.payload_fingerprint != fingerprint:
                raise KnowledgeSyncCorruptState(
                    "durable delivery receipt fingerprint conflict"
                )
            if (
                existing.binding_configuration_version
                != batch.binding_configuration_version
            ):
                raise KnowledgeSyncCorruptState(
                    "durable delivery receipt binding version conflict"
                )
            if existing.status is DurableDeliveryReceiptStatus.APPLIED:
                return
            self._apply_envelopes(
                batch=batch,
                materialization_generation=existing.materialization_generation,
            )
            self._complete_receipt(existing)
            return

        generation = self._allocate_generation(
            tenant_id=batch.tenant_id,
            binding_id=batch.binding_id,
        )
        receipt = DurableDeliveryReceiptV1(
            tenant_id=batch.tenant_id,
            binding_id=batch.binding_id,
            binding_configuration_version=batch.binding_configuration_version,
            delivery_id=batch.delivery_id,
            status=DurableDeliveryReceiptStatus.APPLYING,
            payload_fingerprint=fingerprint,
            materialization_generation=generation,
            record_version=secrets.token_urlsafe(16),
        )
        if not self._put_receipt_if_absent(receipt):
            # Concurrent same-delivery race: recurse into the existing receipt path.
            await self.apply_batch(batch=batch)
            return

        self._apply_envelopes(batch=batch, materialization_generation=generation)
        self._complete_receipt(receipt)

    def inspect_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        prepared_batch_payload_fingerprint: str,
    ) -> KnowledgeSyncSinkReceipt:
        receipt = self._get_receipt(
            tenant_id=tenant_id,
            binding_id=binding_id,
            delivery_id=delivery_id,
        )
        if receipt is None:
            return KnowledgeSyncSinkReceipt(status=KnowledgeSyncSinkReceiptStatus.ABSENT)
        if receipt.payload_fingerprint != prepared_batch_payload_fingerprint:
            return KnowledgeSyncSinkReceipt(
                status=KnowledgeSyncSinkReceiptStatus.CONFLICT,
                delivery_id=delivery_id,
                prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
            )
        if receipt.status is DurableDeliveryReceiptStatus.APPLIED:
            return KnowledgeSyncSinkReceipt(
                status=KnowledgeSyncSinkReceiptStatus.APPLIED,
                delivery_id=delivery_id,
                prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
            )
        return KnowledgeSyncSinkReceipt(
            status=KnowledgeSyncSinkReceiptStatus.UNKNOWN,
            delivery_id=delivery_id,
            prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
        )

    def get_item(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        remote_id: str,
    ) -> DurableMaterializedItemV1 | None:
        document = self._store.get(
            _item_partition_key(tenant_id=tenant_id, binding_id=binding_id),
            _item_row_key(remote_id),
        )
        if document is None:
            return None
        return self._parse_item(
            document,
            expected_tenant=tenant_id,
            expected_binding=binding_id,
        )

    def list_active_remote_ids(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        limit: int = 1000,
    ) -> tuple[str, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        page = self._store.query(
            _item_partition_key(tenant_id=tenant_id, binding_id=binding_id),
            limit=limit,
            row_key_prefix="item:",
        )
        active: list[str] = []
        for document in page.documents:
            item = self._parse_item(
                document,
                expected_tenant=tenant_id,
                expected_binding=binding_id,
            )
            if item.status is DurableMaterializedItemStatus.ACTIVE:
                active.append(item.remote_id)
        return tuple(sorted(active))

    def _apply_envelopes(
        self,
        *,
        batch: KnowledgeSyncBatch,
        materialization_generation: int,
    ) -> None:
        seen: set[str] = set()
        for envelope in batch.envelopes:
            if envelope.remote_id in seen:
                raise KnowledgeSyncCorruptState(
                    "durable batch contains duplicate remote_id"
                )
            seen.add(envelope.remote_id)
            if envelope.change_kind in _TOMBSTONE_CHANGE_KINDS:
                self._apply_tombstone(
                    batch=batch,
                    envelope=envelope,
                    materialization_generation=materialization_generation,
                )
                continue
            if envelope.change_kind not in _ACTIVE_CHANGE_KINDS:
                raise KnowledgeSyncCorruptState(
                    "durable batch contains unsupported change_kind"
                )
            self._apply_active(
                batch=batch,
                envelope=envelope,
                materialization_generation=materialization_generation,
            )

    def _apply_active(
        self,
        *,
        batch: KnowledgeSyncBatch,
        envelope: KnowledgeSyncEnvelope,
        materialization_generation: int,
    ) -> None:
        if envelope.descriptor is None:
            raise KnowledgeSyncCorruptState(
                "active durable envelope requires descriptor"
            )
        existing = self.get_item(
            tenant_id=batch.tenant_id,
            binding_id=batch.binding_id,
            remote_id=envelope.remote_id,
        )
        if (
            existing is not None
            and existing.materialization_generation > materialization_generation
        ):
            # Older delivery must not overwrite newer committed durable state.
            return
        if (
            existing is not None
            and existing.materialization_generation == materialization_generation
            and existing.last_delivery_id == batch.delivery_id
            and existing.status is DurableMaterializedItemStatus.ACTIVE
        ):
            return
        if (
            existing is not None
            and existing.status is DurableMaterializedItemStatus.ACTIVE
            and existing.materialization_generation < materialization_generation
            and knowledge_item_revision_order_key(envelope.descriptor.revision)
            < knowledge_item_revision_order_key(existing.revision)
        ):
            # Same-or-older provider revision must not clobber newer durable state.
            return

        content = envelope.content
        permissions = envelope.permissions
        if (
            existing is not None
            and existing.status is DurableMaterializedItemStatus.ACTIVE
        ):
            if content is None and envelope.change_kind is not KnowledgeChangeKind.UPSERT:
                content = existing.content
            if (
                permissions is None
                and envelope.change_kind is not KnowledgeChangeKind.PERMISSIONS_CHANGED
            ):
                permissions = existing.permissions

        item = DurableMaterializedItemV1(
            tenant_id=batch.tenant_id,
            binding_id=batch.binding_id,
            binding_configuration_version=batch.binding_configuration_version,
            provider_id=batch.source.provider_id,
            source_kind=batch.source.source_kind,
            remote_id=envelope.remote_id,
            status=DurableMaterializedItemStatus.ACTIVE,
            materialization_generation=materialization_generation,
            last_delivery_id=batch.delivery_id,
            revision=envelope.descriptor.revision,
            descriptor=envelope.descriptor,
            content=content,
            permissions=permissions,
            record_version=secrets.token_urlsafe(16),
        )
        self._put_item(item, expected=existing)

    def _apply_tombstone(
        self,
        *,
        batch: KnowledgeSyncBatch,
        envelope: KnowledgeSyncEnvelope,
        materialization_generation: int,
    ) -> None:
        existing = self.get_item(
            tenant_id=batch.tenant_id,
            binding_id=batch.binding_id,
            remote_id=envelope.remote_id,
        )
        if (
            existing is not None
            and existing.materialization_generation > materialization_generation
        ):
            return
        if (
            existing is not None
            and existing.materialization_generation == materialization_generation
            and existing.last_delivery_id == batch.delivery_id
            and existing.status
            in {
                DurableMaterializedItemStatus.DELETED,
                DurableMaterializedItemStatus.REVOKED,
            }
        ):
            return
        status = (
            DurableMaterializedItemStatus.DELETED
            if envelope.change_kind is KnowledgeChangeKind.DELETED
            else DurableMaterializedItemStatus.REVOKED
        )
        item = DurableMaterializedItemV1(
            tenant_id=batch.tenant_id,
            binding_id=batch.binding_id,
            binding_configuration_version=batch.binding_configuration_version,
            provider_id=batch.source.provider_id,
            source_kind=batch.source.source_kind,
            remote_id=envelope.remote_id,
            status=status,
            materialization_generation=materialization_generation,
            last_delivery_id=batch.delivery_id,
            revision=(
                envelope.descriptor.revision if envelope.descriptor is not None else None
            ),
            descriptor=envelope.descriptor,
            content=None,
            permissions=None,
            record_version=secrets.token_urlsafe(16),
        )
        self._put_item(item, expected=existing)

    def _put_item(
        self,
        item: DurableMaterializedItemV1,
        *,
        expected: DurableMaterializedItemV1 | None,
    ) -> None:
        document = self._item_document(item)
        if expected is None:
            if self._store.put_if_absent(document):
                return
            current = self.get_item(
                tenant_id=item.tenant_id,
                binding_id=item.binding_id,
                remote_id=item.remote_id,
            )
            if current is None:
                raise KnowledgeSyncCorruptState("durable item insert race unresolved")
            if (
                current.materialization_generation > item.materialization_generation
                or (
                    current.materialization_generation == item.materialization_generation
                    and current.last_delivery_id == item.last_delivery_id
                )
            ):
                return
            expected_doc = self._item_document(current)
            if not self._store.replace_if_match(
                expected=expected_doc,
                replacement=document,
            ):
                raise KnowledgeSyncCorruptState("durable item replace conflict")
            return
        expected_doc = self._item_document(expected)
        if self._store.replace_if_match(expected=expected_doc, replacement=document):
            return
        current = self.get_item(
            tenant_id=item.tenant_id,
            binding_id=item.binding_id,
            remote_id=item.remote_id,
        )
        if current is None:
            if self._store.put_if_absent(document):
                return
            raise KnowledgeSyncCorruptState("durable item replace conflict")
        if current.materialization_generation > item.materialization_generation:
            return
        if (
            current.materialization_generation == item.materialization_generation
            and current.last_delivery_id == item.last_delivery_id
        ):
            return
        if not self._store.replace_if_match(
            expected=self._item_document(current),
            replacement=document,
        ):
            raise KnowledgeSyncCorruptState("durable item replace conflict")

    def _allocate_generation(self, *, tenant_id: str, binding_id: str) -> int:
        partition = _sequence_partition_key(tenant_id)
        row = _sequence_row_key(binding_id)
        for _ in range(8):
            existing = self._store.get(partition, row)
            if existing is None:
                data = {
                    "schema_version": _SEQUENCE_SCHEMA,
                    "tenant_id": tenant_id,
                    "binding_id": binding_id,
                    "next_generation": 2,
                    "record_version": secrets.token_urlsafe(16),
                }
                _reject_secret_fields(data, kind="durable sequence")
                if self._store.put_if_absent(
                    DocumentRecord(partition_key=partition, row_key=row, data=data)
                ):
                    return 1
                continue
            parsed = _data_as_dict(existing.data)
            _reject_secret_fields(parsed, kind="durable sequence")
            if parsed.get("schema_version") != _SEQUENCE_SCHEMA:
                raise KnowledgeSyncCorruptState("durable sequence schema is invalid")
            next_generation = parsed.get("next_generation")
            if not isinstance(next_generation, int) or next_generation < 1:
                raise KnowledgeSyncCorruptState("durable sequence value is invalid")
            replacement_data = {
                "schema_version": _SEQUENCE_SCHEMA,
                "tenant_id": tenant_id,
                "binding_id": binding_id,
                "next_generation": next_generation + 1,
                "record_version": secrets.token_urlsafe(16),
            }
            _reject_secret_fields(replacement_data, kind="durable sequence")
            if self._store.replace_if_match(
                expected=existing,
                replacement=DocumentRecord(
                    partition_key=partition,
                    row_key=row,
                    data=replacement_data,
                ),
            ):
                return next_generation
        raise KnowledgeSyncCorruptState("durable sequence allocation conflict")

    def _get_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
    ) -> DurableDeliveryReceiptV1 | None:
        document = self._store.get(
            _receipt_partition_key(tenant_id=tenant_id, binding_id=binding_id),
            _receipt_row_key(delivery_id),
        )
        if document is None:
            return None
        return self._parse_receipt(
            document,
            expected_tenant=tenant_id,
            expected_binding=binding_id,
        )

    def _put_receipt_if_absent(self, receipt: DurableDeliveryReceiptV1) -> bool:
        return self._store.put_if_absent(self._receipt_document(receipt))

    def _complete_receipt(self, receipt: DurableDeliveryReceiptV1) -> None:
        if receipt.status is DurableDeliveryReceiptStatus.APPLIED:
            return
        completed = DurableDeliveryReceiptV1(
            tenant_id=receipt.tenant_id,
            binding_id=receipt.binding_id,
            binding_configuration_version=receipt.binding_configuration_version,
            delivery_id=receipt.delivery_id,
            status=DurableDeliveryReceiptStatus.APPLIED,
            payload_fingerprint=receipt.payload_fingerprint,
            materialization_generation=receipt.materialization_generation,
            record_version=secrets.token_urlsafe(16),
        )
        expected = self._receipt_document(receipt)
        replacement = self._receipt_document(completed)
        if self._store.replace_if_match(expected=expected, replacement=replacement):
            return
        current = self._get_receipt(
            tenant_id=receipt.tenant_id,
            binding_id=receipt.binding_id,
            delivery_id=receipt.delivery_id,
        )
        if current is not None and current.status is DurableDeliveryReceiptStatus.APPLIED:
            if current.payload_fingerprint != receipt.payload_fingerprint:
                raise KnowledgeSyncCorruptState(
                    "durable delivery receipt fingerprint conflict"
                )
            return
        raise KnowledgeSyncCorruptState("durable delivery receipt complete conflict")

    def _item_document(self, item: DurableMaterializedItemV1) -> DocumentRecord:
        data = {
            "schema_version": _ITEM_SCHEMA,
            **item.model_dump(mode="json"),
        }
        _reject_secret_fields(data, kind="durable item")
        return DocumentRecord(
            partition_key=_item_partition_key(
                tenant_id=item.tenant_id,
                binding_id=item.binding_id,
            ),
            row_key=_item_row_key(item.remote_id),
            data=data,
        )

    def _receipt_document(self, receipt: DurableDeliveryReceiptV1) -> DocumentRecord:
        data = {
            "schema_version": _RECEIPT_SCHEMA,
            **receipt.model_dump(mode="json"),
        }
        _reject_secret_fields(data, kind="durable delivery receipt")
        return DocumentRecord(
            partition_key=_receipt_partition_key(
                tenant_id=receipt.tenant_id,
                binding_id=receipt.binding_id,
            ),
            row_key=_receipt_row_key(receipt.delivery_id),
            data=data,
        )

    def _parse_item(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
        expected_binding: str,
    ) -> DurableMaterializedItemV1:
        data = _data_as_dict(document.data)
        _reject_secret_fields(data, kind="durable item")
        if data.get("schema_version") != _ITEM_SCHEMA:
            raise KnowledgeSyncCorruptState("durable item schema is invalid")
        payload = {key: value for key, value in data.items() if key != "schema_version"}
        try:
            item = DurableMaterializedItemV1.model_validate(payload)
        except ValidationError as exc:
            raise KnowledgeSyncCorruptState("durable item record is corrupt") from exc
        if item.tenant_id != expected_tenant or item.binding_id != expected_binding:
            raise KnowledgeSyncCorruptState("durable item ownership mismatch")
        if document.partition_key != _item_partition_key(
            tenant_id=expected_tenant,
            binding_id=expected_binding,
        ):
            raise KnowledgeSyncCorruptState("durable item partition is invalid")
        if document.row_key != _item_row_key(item.remote_id):
            raise KnowledgeSyncCorruptState("durable item row key is invalid")
        return item

    def _parse_receipt(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
        expected_binding: str,
    ) -> DurableDeliveryReceiptV1:
        data = _data_as_dict(document.data)
        _reject_secret_fields(data, kind="durable delivery receipt")
        if data.get("schema_version") != _RECEIPT_SCHEMA:
            raise KnowledgeSyncCorruptState("durable delivery receipt schema is invalid")
        payload = {key: value for key, value in data.items() if key != "schema_version"}
        try:
            receipt = DurableDeliveryReceiptV1.model_validate(payload)
        except ValidationError as exc:
            raise KnowledgeSyncCorruptState(
                "durable delivery receipt record is corrupt"
            ) from exc
        if receipt.tenant_id != expected_tenant or receipt.binding_id != expected_binding:
            raise KnowledgeSyncCorruptState("durable delivery receipt ownership mismatch")
        if document.partition_key != _receipt_partition_key(
            tenant_id=expected_tenant,
            binding_id=expected_binding,
        ):
            raise KnowledgeSyncCorruptState("durable delivery receipt partition is invalid")
        if document.row_key != _receipt_row_key(receipt.delivery_id):
            raise KnowledgeSyncCorruptState("durable delivery receipt row key is invalid")
        return receipt


__all__ = [
    "DocumentStoreDurableKnowledgeSyncSink",
    "DurableDeliveryReceiptStatus",
    "DurableDeliveryReceiptV1",
    "DurableMaterializedItemStatus",
    "DurableMaterializedItemV1",
    "durable_batch_payload_fingerprint",
    "knowledge_item_revision_order_key",
]
