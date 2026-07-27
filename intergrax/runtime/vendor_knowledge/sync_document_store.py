# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DocumentStore-backed synchronization repositories for Vendor Knowledge."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Callable, Mapping

from pydantic import ValidationError

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeSyncCheckpointConflict,
    KnowledgeSyncCorruptState,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemState,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncCheckpoint,
)

_LEASE_SCHEMA = "vendor_knowledge.source_lease.v1"
_CHECKPOINT_SCHEMA = "vendor_knowledge.sync_checkpoint.v1"
_ITEM_STATE_SCHEMA = "vendor_knowledge.remote_item_state.v1"
_DELIVERY_MARKER_SCHEMA = "vendor_knowledge.delivery_marker.v1"

_LEASE_PARTITION_PREFIX = "vendor_knowledge.source_lease.v1"
_CHECKPOINT_PARTITION_PREFIX = "vendor_knowledge.sync_checkpoint.v1"
_ITEM_PARTITION_PREFIX = "vendor_knowledge.remote_item.v1"

Clock = Callable[[], float]
TokenFactory = Callable[[], str]
RecordVersionFactory = Callable[[], str]


def _default_clock() -> float:
    return time.time()


def _default_token_factory() -> str:
    return uuid.uuid4().hex


def _default_record_version_factory() -> str:
    return str(uuid.uuid4())


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _require_conditional_document_store(
    document_store: DocumentStore,
) -> ConditionalDocumentStore:
    if not isinstance(document_store, ConditionalDocumentStore):
        raise TypeError(
            "vendor knowledge sync repositories require ConditionalDocumentStore"
        )
    return document_store


def _lease_partition_key(tenant_id: str) -> str:
    return f"{_LEASE_PARTITION_PREFIX}:{_require_non_empty(tenant_id, field_name='tenant_id')}"


def _lease_row_key(binding_id: str) -> str:
    return f"binding:{_require_non_empty(binding_id, field_name='binding_id')}"


def _checkpoint_partition_key(tenant_id: str) -> str:
    return (
        f"{_CHECKPOINT_PARTITION_PREFIX}:"
        f"{_require_non_empty(tenant_id, field_name='tenant_id')}"
    )


def _checkpoint_row_key(binding_id: str) -> str:
    return f"binding:{_require_non_empty(binding_id, field_name='binding_id')}"


def _item_partition_key(*, tenant_id: str, binding_id: str) -> str:
    return (
        f"{_ITEM_PARTITION_PREFIX}:"
        f"{_require_non_empty(tenant_id, field_name='tenant_id')}:"
        f"{_require_non_empty(binding_id, field_name='binding_id')}"
    )


def _item_row_key(remote_id: str) -> str:
    cleaned = _require_non_empty(remote_id, field_name="remote_id")
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return f"item:{digest}"


def _delivery_row_key(delivery_id: str) -> str:
    cleaned = _require_non_empty(delivery_id, field_name="delivery_id")
    return f"delivery:{cleaned}"


def _batch_fingerprint(states: tuple[KnowledgeRemoteItemState, ...]) -> str:
    ordered = sorted(states, key=lambda state: state.remote_id)
    payload = [state.model_dump(mode="json") for state in ordered]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _data_as_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    return dict(data)


class DocumentStoreKnowledgeSourceLeaseRepository:
    """Atomic source-level lease repository backed by ConditionalDocumentStore."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        clock: Clock | None = None,
        token_factory: TokenFactory | None = None,
        record_version_factory: RecordVersionFactory | None = None,
    ) -> None:
        self._store = _require_conditional_document_store(document_store)
        self._clock = clock or _default_clock
        self._token_factory = token_factory or _default_token_factory
        self._record_version_factory = (
            record_version_factory or _default_record_version_factory
        )

    def acquire(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        owner_id: str,
        ttl_seconds: int,
    ) -> KnowledgeSourceLeaseToken | None:
        cleaned_tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        cleaned_binding = _require_non_empty(binding_id, field_name="binding_id")
        cleaned_owner = _require_non_empty(owner_id, field_name="owner_id")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")

        token = self._token_factory()
        token = _require_non_empty(str(token), field_name="token")
        now = float(self._clock())
        expires_at = now + float(ttl_seconds)
        record_version = str(self._record_version_factory())
        partition_key = _lease_partition_key(cleaned_tenant)
        row_key = _lease_row_key(cleaned_binding)
        candidate = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={
                "schema_version": _LEASE_SCHEMA,
                "tenant_id": cleaned_tenant,
                "binding_id": cleaned_binding,
                "owner_id": cleaned_owner,
                "token": token,
                "record_version": record_version,
                "expires_at_epoch": expires_at,
            },
            ttl_seconds=ttl_seconds,
        )
        if self._store.put_if_absent(candidate):
            return KnowledgeSourceLeaseToken(
                tenant_id=cleaned_tenant,
                binding_id=cleaned_binding,
                owner_id=cleaned_owner,
                token=token,
            )

        existing = self._store.get(partition_key, row_key)
        if existing is None:
            return None
        parsed = self._parse_lease_document(existing, expected_tenant=cleaned_tenant)
        if float(self._clock()) < float(parsed["expires_at_epoch"]):
            return None

        takeover_version = str(self._record_version_factory())
        takeover_token = _require_non_empty(str(self._token_factory()), field_name="token")
        takeover_expires = float(self._clock()) + float(ttl_seconds)
        replacement = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={
                "schema_version": _LEASE_SCHEMA,
                "tenant_id": cleaned_tenant,
                "binding_id": cleaned_binding,
                "owner_id": cleaned_owner,
                "token": takeover_token,
                "record_version": takeover_version,
                "expires_at_epoch": takeover_expires,
            },
            ttl_seconds=ttl_seconds,
        )
        if not self._store.replace_if_match(expected=existing, replacement=replacement):
            return None
        return KnowledgeSourceLeaseToken(
            tenant_id=cleaned_tenant,
            binding_id=cleaned_binding,
            owner_id=cleaned_owner,
            token=takeover_token,
        )

    def release(self, *, lease: KnowledgeSourceLeaseToken) -> None:
        partition_key = _lease_partition_key(lease.tenant_id)
        row_key = _lease_row_key(lease.binding_id)
        existing = self._store.get(partition_key, row_key)
        if existing is None:
            return
        parsed = self._parse_lease_document(existing, expected_tenant=lease.tenant_id)
        if (
            parsed["token"] != lease.token
            or parsed["owner_id"] != lease.owner_id
            or parsed["binding_id"] != lease.binding_id
        ):
            return
        self._store.delete_if_match(expected=existing)

    def _parse_lease_document(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
    ) -> dict[str, Any]:
        data = _data_as_dict(document.data)
        try:
            schema_version = data.get("schema_version")
            tenant_id = data.get("tenant_id")
            binding_id = data.get("binding_id")
            owner_id = data.get("owner_id")
            token = data.get("token")
            record_version = data.get("record_version")
            expires_at = data.get("expires_at_epoch")
            if schema_version != _LEASE_SCHEMA:
                raise KnowledgeSyncCorruptState("sync lease schema is invalid")
            if not isinstance(tenant_id, str) or tenant_id.strip() != expected_tenant:
                raise KnowledgeSyncCorruptState("sync lease tenant identity is invalid")
            if document.partition_key != _lease_partition_key(expected_tenant):
                raise KnowledgeSyncCorruptState("sync lease partition is invalid")
            if not isinstance(binding_id, str) or not binding_id.strip():
                raise KnowledgeSyncCorruptState("sync lease binding identity is invalid")
            if document.row_key != _lease_row_key(binding_id):
                raise KnowledgeSyncCorruptState("sync lease row key is invalid")
            if not isinstance(owner_id, str) or not owner_id.strip():
                raise KnowledgeSyncCorruptState("sync lease owner identity is invalid")
            if not isinstance(token, str) or not token.strip():
                raise KnowledgeSyncCorruptState("sync lease token is invalid")
            if not isinstance(record_version, str) or not record_version.strip():
                raise KnowledgeSyncCorruptState("sync lease record version is invalid")
            if not isinstance(expires_at, (int, float)):
                raise KnowledgeSyncCorruptState("sync lease expiry is invalid")
        except KnowledgeSyncCorruptState:
            raise
        except Exception:
            raise KnowledgeSyncCorruptState("sync lease record is corrupt") from None
        return {
            "tenant_id": str(tenant_id).strip(),
            "binding_id": str(binding_id).strip(),
            "owner_id": str(owner_id).strip(),
            "token": str(token).strip(),
            "record_version": str(record_version).strip(),
            "expires_at_epoch": float(expires_at),
        }


class DocumentStoreKnowledgeSyncCheckpointRepository:
    """CAS checkpoint repository backed by ConditionalDocumentStore."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        record_version_factory: RecordVersionFactory | None = None,
    ) -> None:
        self._store = _require_conditional_document_store(document_store)
        self._record_version_factory = (
            record_version_factory or _default_record_version_factory
        )

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncCheckpoint | None:
        cleaned_tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        cleaned_binding = _require_non_empty(binding_id, field_name="binding_id")
        document = self._store.get(
            _checkpoint_partition_key(cleaned_tenant),
            _checkpoint_row_key(cleaned_binding),
        )
        if document is None:
            return None
        return self._public_checkpoint(
            document,
            expected_tenant=cleaned_tenant,
            expected_binding=cleaned_binding,
        )

    def commit(
        self,
        checkpoint: KnowledgeSyncCheckpoint,
        *,
        expected_previous: KnowledgeSyncCheckpoint | None,
    ) -> None:
        partition_key = _checkpoint_partition_key(checkpoint.tenant_id)
        row_key = _checkpoint_row_key(checkpoint.binding_id)
        if expected_previous is None:
            document = self._to_document(checkpoint)
            if not self._store.put_if_absent(document):
                raise KnowledgeSyncCheckpointConflict("checkpoint create conflict")
            return

        current = self._store.get(partition_key, row_key)
        if current is None:
            raise KnowledgeSyncCheckpointConflict("checkpoint missing for expected previous")
        public = self._public_checkpoint(
            current,
            expected_tenant=checkpoint.tenant_id,
            expected_binding=checkpoint.binding_id,
        )
        if public != expected_previous:
            raise KnowledgeSyncCheckpointConflict("checkpoint expected previous mismatch")
        replacement = self._to_document(checkpoint)
        if not self._store.replace_if_match(expected=current, replacement=replacement):
            raise KnowledgeSyncCheckpointConflict("checkpoint cas conflict")

    def _to_document(self, checkpoint: KnowledgeSyncCheckpoint) -> DocumentRecord:
        return DocumentRecord(
            partition_key=_checkpoint_partition_key(checkpoint.tenant_id),
            row_key=_checkpoint_row_key(checkpoint.binding_id),
            data={
                "schema_version": _CHECKPOINT_SCHEMA,
                "tenant_id": checkpoint.tenant_id,
                "binding_id": checkpoint.binding_id,
                "record_version": str(self._record_version_factory()),
                "checkpoint": checkpoint.model_dump(mode="json"),
            },
        )

    def _public_checkpoint(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
        expected_binding: str,
    ) -> KnowledgeSyncCheckpoint:
        data = _data_as_dict(document.data)
        try:
            if data.get("schema_version") != _CHECKPOINT_SCHEMA:
                raise KnowledgeSyncCorruptState("sync checkpoint schema is invalid")
            tenant_id = data.get("tenant_id")
            binding_id = data.get("binding_id")
            record_version = data.get("record_version")
            raw_checkpoint = data.get("checkpoint")
            if not isinstance(tenant_id, str) or tenant_id.strip() != expected_tenant:
                raise KnowledgeSyncCorruptState("sync checkpoint tenant identity is invalid")
            if not isinstance(binding_id, str) or binding_id.strip() != expected_binding:
                raise KnowledgeSyncCorruptState("sync checkpoint binding identity is invalid")
            if document.partition_key != _checkpoint_partition_key(expected_tenant):
                raise KnowledgeSyncCorruptState("sync checkpoint partition is invalid")
            if document.row_key != _checkpoint_row_key(expected_binding):
                raise KnowledgeSyncCorruptState("sync checkpoint row key is invalid")
            if not isinstance(record_version, str) or not record_version.strip():
                raise KnowledgeSyncCorruptState("sync checkpoint record version is invalid")
            if not isinstance(raw_checkpoint, Mapping):
                raise KnowledgeSyncCorruptState("sync checkpoint payload is invalid")
            checkpoint = KnowledgeSyncCheckpoint.model_validate(dict(raw_checkpoint))
            if (
                checkpoint.tenant_id != expected_tenant
                or checkpoint.binding_id != expected_binding
            ):
                raise KnowledgeSyncCorruptState("sync checkpoint payload identity is invalid")
            return checkpoint
        except KnowledgeSyncCorruptState:
            raise
        except ValidationError:
            raise KnowledgeSyncCorruptState("sync checkpoint payload is invalid") from None
        except Exception:
            raise KnowledgeSyncCorruptState("sync checkpoint record is corrupt") from None


class DocumentStoreKnowledgeRemoteItemStateRepository:
    """Idempotent remote-item state repository with delivery markers."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        record_version_factory: RecordVersionFactory | None = None,
    ) -> None:
        self._store = _require_conditional_document_store(document_store)
        self._record_version_factory = (
            record_version_factory or _default_record_version_factory
        )

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        remote_id: str,
    ) -> KnowledgeRemoteItemState | None:
        cleaned_tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        cleaned_binding = _require_non_empty(binding_id, field_name="binding_id")
        cleaned_remote = _require_non_empty(remote_id, field_name="remote_id")
        document = self._store.get(
            _item_partition_key(tenant_id=cleaned_tenant, binding_id=cleaned_binding),
            _item_row_key(cleaned_remote),
        )
        if document is None:
            return None
        return self._parse_state(
            document,
            expected_tenant=cleaned_tenant,
            expected_binding=cleaned_binding,
        )

    def apply_batch(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        states: tuple[KnowledgeRemoteItemState, ...],
    ) -> None:
        cleaned_tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        cleaned_binding = _require_non_empty(binding_id, field_name="binding_id")
        cleaned_delivery = _require_non_empty(delivery_id, field_name="delivery_id")
        self._validate_batch_states(
            tenant_id=cleaned_tenant,
            binding_id=cleaned_binding,
            delivery_id=cleaned_delivery,
            states=states,
        )
        fingerprint = _batch_fingerprint(states)
        partition_key = _item_partition_key(
            tenant_id=cleaned_tenant,
            binding_id=cleaned_binding,
        )
        marker_row = _delivery_row_key(cleaned_delivery)
        marker_applying = self._marker_document(
            tenant_id=cleaned_tenant,
            binding_id=cleaned_binding,
            delivery_id=cleaned_delivery,
            batch_fingerprint=fingerprint,
            status="applying",
        )
        created = self._store.put_if_absent(marker_applying)
        if not created:
            existing_marker = self._store.get(partition_key, marker_row)
            if existing_marker is None:
                raise KnowledgeSyncCorruptState("delivery marker disappeared")
            marker = self._parse_marker(
                existing_marker,
                expected_tenant=cleaned_tenant,
                expected_binding=cleaned_binding,
                expected_delivery=cleaned_delivery,
            )
            if marker["batch_fingerprint"] != fingerprint:
                raise KnowledgeSyncCorruptState("delivery marker fingerprint mismatch")
            if marker["status"] == "completed":
                return
            if marker["status"] != "applying":
                raise KnowledgeSyncCorruptState("delivery marker status is invalid")
            current_marker = existing_marker
        else:
            current_marker = marker_applying

        ordered = tuple(sorted(states, key=lambda state: state.remote_id))
        for state in ordered:
            self._apply_one_state(
                partition_key=partition_key,
                state=state,
            )

        completed = self._marker_document(
            tenant_id=cleaned_tenant,
            binding_id=cleaned_binding,
            delivery_id=cleaned_delivery,
            batch_fingerprint=fingerprint,
            status="completed",
        )
        if not self._store.replace_if_match(
            expected=current_marker,
            replacement=completed,
        ):
            latest = self._store.get(partition_key, marker_row)
            if latest is None:
                raise KnowledgeSyncCorruptState("delivery marker missing after apply")
            marker = self._parse_marker(
                latest,
                expected_tenant=cleaned_tenant,
                expected_binding=cleaned_binding,
                expected_delivery=cleaned_delivery,
            )
            if (
                marker["status"] == "completed"
                and marker["batch_fingerprint"] == fingerprint
            ):
                return
            raise KnowledgeSyncCorruptState("delivery marker completion conflict")

    def _validate_batch_states(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        states: tuple[KnowledgeRemoteItemState, ...],
    ) -> None:
        seen: set[str] = set()
        for state in states:
            if state.tenant_id != tenant_id or state.binding_id != binding_id:
                raise KnowledgeSyncCorruptState("remote item state identity mismatch")
            if state.last_delivery_id != delivery_id:
                raise KnowledgeSyncCorruptState("remote item delivery identity mismatch")
            if state.remote_id in seen:
                raise KnowledgeSyncCorruptState("duplicate remote item in batch")
            seen.add(state.remote_id)

    def _apply_one_state(
        self,
        *,
        partition_key: str,
        state: KnowledgeRemoteItemState,
    ) -> None:
        row_key = _item_row_key(state.remote_id)
        candidate = self._state_document(state)
        if self._store.put_if_absent(candidate):
            return
        existing = self._store.get(partition_key, row_key)
        if existing is None:
            raise KnowledgeSyncCorruptState("remote item state disappeared")
        current = self._parse_state(
            existing,
            expected_tenant=state.tenant_id,
            expected_binding=state.binding_id,
        )
        if current.last_delivery_id == state.last_delivery_id:
            if current == state:
                return
            raise KnowledgeSyncCorruptState(
                "remote item state conflict for identical delivery"
            )
        replacement = self._state_document(state)
        if not self._store.replace_if_match(expected=existing, replacement=replacement):
            raise KnowledgeSyncCorruptState("remote item state cas conflict")

    def _state_document(self, state: KnowledgeRemoteItemState) -> DocumentRecord:
        return DocumentRecord(
            partition_key=_item_partition_key(
                tenant_id=state.tenant_id,
                binding_id=state.binding_id,
            ),
            row_key=_item_row_key(state.remote_id),
            data={
                "schema_version": _ITEM_STATE_SCHEMA,
                "tenant_id": state.tenant_id,
                "binding_id": state.binding_id,
                "record_version": str(self._record_version_factory()),
                "state": state.model_dump(mode="json"),
            },
        )

    def _marker_document(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        batch_fingerprint: str,
        status: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=_item_partition_key(
                tenant_id=tenant_id,
                binding_id=binding_id,
            ),
            row_key=_delivery_row_key(delivery_id),
            data={
                "schema_version": _DELIVERY_MARKER_SCHEMA,
                "tenant_id": tenant_id,
                "binding_id": binding_id,
                "delivery_id": delivery_id,
                "batch_fingerprint": batch_fingerprint,
                "status": status,
                "record_version": str(self._record_version_factory()),
            },
        )

    def _parse_state(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
        expected_binding: str,
    ) -> KnowledgeRemoteItemState:
        data = _data_as_dict(document.data)
        try:
            if data.get("schema_version") != _ITEM_STATE_SCHEMA:
                raise KnowledgeSyncCorruptState("remote item state schema is invalid")
            tenant_id = data.get("tenant_id")
            binding_id = data.get("binding_id")
            record_version = data.get("record_version")
            raw_state = data.get("state")
            if not isinstance(tenant_id, str) or tenant_id.strip() != expected_tenant:
                raise KnowledgeSyncCorruptState("remote item tenant identity is invalid")
            if not isinstance(binding_id, str) or binding_id.strip() != expected_binding:
                raise KnowledgeSyncCorruptState("remote item binding identity is invalid")
            expected_partition = _item_partition_key(
                tenant_id=expected_tenant,
                binding_id=expected_binding,
            )
            if document.partition_key != expected_partition:
                raise KnowledgeSyncCorruptState("remote item partition is invalid")
            if not isinstance(record_version, str) or not record_version.strip():
                raise KnowledgeSyncCorruptState("remote item record version is invalid")
            if not isinstance(raw_state, Mapping):
                raise KnowledgeSyncCorruptState("remote item state payload is invalid")
            state = KnowledgeRemoteItemState.model_validate(dict(raw_state))
            if state.tenant_id != expected_tenant or state.binding_id != expected_binding:
                raise KnowledgeSyncCorruptState("remote item payload identity is invalid")
            if document.row_key != _item_row_key(state.remote_id):
                raise KnowledgeSyncCorruptState("remote item row key is invalid")
            return state
        except KnowledgeSyncCorruptState:
            raise
        except ValidationError:
            raise KnowledgeSyncCorruptState("remote item state payload is invalid") from None
        except Exception:
            raise KnowledgeSyncCorruptState("remote item state record is corrupt") from None

    def _parse_marker(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
        expected_binding: str,
        expected_delivery: str,
    ) -> dict[str, str]:
        data = _data_as_dict(document.data)
        try:
            if data.get("schema_version") != _DELIVERY_MARKER_SCHEMA:
                raise KnowledgeSyncCorruptState("delivery marker schema is invalid")
            tenant_id = data.get("tenant_id")
            binding_id = data.get("binding_id")
            delivery_id = data.get("delivery_id")
            fingerprint = data.get("batch_fingerprint")
            status = data.get("status")
            record_version = data.get("record_version")
            if not isinstance(tenant_id, str) or tenant_id.strip() != expected_tenant:
                raise KnowledgeSyncCorruptState("delivery marker tenant identity is invalid")
            if not isinstance(binding_id, str) or binding_id.strip() != expected_binding:
                raise KnowledgeSyncCorruptState("delivery marker binding identity is invalid")
            if not isinstance(delivery_id, str) or delivery_id.strip() != expected_delivery:
                raise KnowledgeSyncCorruptState("delivery marker delivery identity is invalid")
            if document.row_key != _delivery_row_key(expected_delivery):
                raise KnowledgeSyncCorruptState("delivery marker row key is invalid")
            if not isinstance(fingerprint, str) or not fingerprint.strip():
                raise KnowledgeSyncCorruptState("delivery marker fingerprint is invalid")
            if status not in {"applying", "completed"}:
                raise KnowledgeSyncCorruptState("delivery marker status is invalid")
            if not isinstance(record_version, str) or not record_version.strip():
                raise KnowledgeSyncCorruptState("delivery marker record version is invalid")
            return {
                "batch_fingerprint": fingerprint.strip(),
                "status": str(status),
            }
        except KnowledgeSyncCorruptState:
            raise
        except Exception:
            raise KnowledgeSyncCorruptState("delivery marker record is corrupt") from None
