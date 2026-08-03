# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DocumentStore-backed synchronization repositories for Vendor Knowledge."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from typing import Any, Callable, Mapping

from pydantic import ValidationError

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeReconciliationRunConflict,
    KnowledgeSyncCheckpointConflict,
    KnowledgeSyncCorruptState,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationLimitPolicy,
    KnowledgeReconciliationRun,
    KnowledgeReconciliationRunCollecting,
    KnowledgeReconciliationRunPagePrepared,
    KnowledgeReconciliationRunPhase,
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStateReceipt,
    KnowledgeRemoteItemStateReceiptStatus,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncCheckpoint,
    parse_knowledge_reconciliation_run,
    validate_reconciliation_candidate_inventory,
    validate_reconciliation_prepared_intent,
)

_LEASE_SCHEMA = "vendor_knowledge.source_lease.v1"
_CHECKPOINT_SCHEMA = "vendor_knowledge.sync_checkpoint.v1"
_ITEM_STATE_SCHEMA = "vendor_knowledge.remote_item_state.v1"
_DELIVERY_MARKER_SCHEMA = "vendor_knowledge.delivery_marker.v1"
_DELIVERY_MARKER_SCHEMA_V2 = "vendor_knowledge.delivery_marker.v2"
_RECONCILIATION_RUN_SCHEMA = "vendor_knowledge.reconciliation_run.v1"

_LEASE_PARTITION_PREFIX = "vendor_knowledge.source_lease.v1"
_CHECKPOINT_PARTITION_PREFIX = "vendor_knowledge.sync_checkpoint.v1"
_ITEM_PARTITION_PREFIX = "vendor_knowledge.remote_item.v1"
_RECONCILIATION_RUN_PARTITION_PREFIX = "vendor_knowledge.reconciliation_run.v1"

_MAX_LEASE_ACQUIRE_ATTEMPTS = 4
_MARKER_STATUS_APPLYING = "applying"
_MARKER_STATUS_COMPLETED = "completed"
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

Clock = Callable[[], float]
TokenFactory = Callable[[], str]
VersionFactory = Callable[[], str]


def _default_clock() -> float:
    return time.time()


def _default_token_factory() -> str:
    return secrets.token_urlsafe(32)


def _default_version_factory() -> str:
    return secrets.token_urlsafe(16)


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


def _reject_secret_fields(data: Mapping[str, Any], *, kind: str) -> None:
    for key in _FORBIDDEN_SECRET_FIELDS:
        if key in data:
            raise KnowledgeSyncCorruptState(
                f"{kind} must not contain secret-bearing fields"
            )


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


def _reconciliation_run_partition_key(tenant_id: str) -> str:
    return (
        f"{_RECONCILIATION_RUN_PARTITION_PREFIX}:"
        f"{_require_non_empty(tenant_id, field_name='tenant_id')}"
    )


def _reconciliation_run_row_key(binding_id: str) -> str:
    return f"binding:{_require_non_empty(binding_id, field_name='binding_id')}"


def _configuration_limit_error(safe_message: str) -> VendorKnowledgeError:
    return VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
        safe_message=safe_message,
        retryable=False,
    )


_ACTIVE_RECONCILIATION_PHASES: frozenset[KnowledgeReconciliationRunPhase] = frozenset(
    {
        KnowledgeReconciliationRunPhase.COLLECTING,
        KnowledgeReconciliationRunPhase.PAGE_PREPARED,
        KnowledgeReconciliationRunPhase.FINALIZING,
        KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
    }
)

_ALLOWED_RECONCILIATION_TRANSITIONS: frozenset[
    tuple[KnowledgeReconciliationRunPhase, KnowledgeReconciliationRunPhase]
] = frozenset(
    {
        (
            KnowledgeReconciliationRunPhase.COLLECTING,
            KnowledgeReconciliationRunPhase.PAGE_PREPARED,
        ),
        (
            KnowledgeReconciliationRunPhase.COLLECTING,
            KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        ),
        (
            KnowledgeReconciliationRunPhase.COLLECTING,
            KnowledgeReconciliationRunPhase.ABORTED,
        ),
        (
            KnowledgeReconciliationRunPhase.PAGE_PREPARED,
            KnowledgeReconciliationRunPhase.COLLECTING,
        ),
        (
            KnowledgeReconciliationRunPhase.PAGE_PREPARED,
            KnowledgeReconciliationRunPhase.FINALIZING,
        ),
        (
            KnowledgeReconciliationRunPhase.PAGE_PREPARED,
            KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        ),
        (
            KnowledgeReconciliationRunPhase.FINALIZING,
            KnowledgeReconciliationRunPhase.COMPLETED,
        ),
        (
            KnowledgeReconciliationRunPhase.FINALIZING,
            KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        ),
        (
            KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
            KnowledgeReconciliationRunPhase.COLLECTING,
        ),
        (
            KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
            KnowledgeReconciliationRunPhase.FINALIZING,
        ),
        (
            KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
            KnowledgeReconciliationRunPhase.COMPLETED,
        ),
        (
            KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
            KnowledgeReconciliationRunPhase.ABORTED,
        ),
    }
)


def _validate_reconciliation_policy(
    run: KnowledgeReconciliationRun,
    *,
    policy: KnowledgeReconciliationLimitPolicy,
) -> None:
    if isinstance(run, KnowledgeReconciliationRunCollecting):
        validate_reconciliation_candidate_inventory(
            run.remaining_candidate_remote_ids,
            policy=policy,
        )
    if isinstance(run, KnowledgeReconciliationRunPagePrepared):
        validate_reconciliation_prepared_intent(run, policy=policy)


def _validate_reconciliation_identity_unchanged(
    *,
    expected: KnowledgeReconciliationRun,
    replacement: KnowledgeReconciliationRun,
) -> None:
    if (
        expected.tenant_id != replacement.tenant_id
        or expected.binding_id != replacement.binding_id
        or expected.run_id != replacement.run_id
        or expected.provider_id != replacement.provider_id
        or expected.source_kind != replacement.source_kind
        or expected.created_at != replacement.created_at
        or expected.binding_configuration_version
        != replacement.binding_configuration_version
    ):
        raise KnowledgeSyncCorruptState("reconciliation run immutable identity changed")


def _validate_reconciliation_transition(
    *,
    expected: KnowledgeReconciliationRun,
    replacement: KnowledgeReconciliationRun,
) -> None:
    if expected.phase == replacement.phase:
        return
    if (expected.phase, replacement.phase) not in _ALLOWED_RECONCILIATION_TRANSITIONS:
        raise KnowledgeSyncCorruptState(
            "reconciliation run phase transition is invalid"
        )


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
        version_factory: VersionFactory | None = None,
    ) -> None:
        self._store = _require_conditional_document_store(document_store)
        self._clock = clock or _default_clock
        self._token_factory = token_factory or _default_token_factory
        self._version_factory = version_factory or _default_version_factory

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

        partition_key = _lease_partition_key(cleaned_tenant)
        row_key = _lease_row_key(cleaned_binding)

        for _ in range(_MAX_LEASE_ACQUIRE_ATTEMPTS):
            token = _require_non_empty(str(self._token_factory()), field_name="token")
            record_version = _require_non_empty(
                str(self._version_factory()),
                field_name="record_version",
            )
            now = float(self._clock())
            candidate = self._lease_document(
                tenant_id=cleaned_tenant,
                binding_id=cleaned_binding,
                owner_id=cleaned_owner,
                token=token,
                acquired_at_epoch=now,
                expires_at_epoch=now + float(ttl_seconds),
                ttl_seconds=ttl_seconds,
                record_version=record_version,
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
                continue
            parsed = self._parse_lease_document(
                existing,
                expected_tenant=cleaned_tenant,
            )
            if float(self._clock()) < float(parsed["expires_at_epoch"]):
                return None
            if self._store.replace_if_match(expected=existing, replacement=candidate):
                return KnowledgeSourceLeaseToken(
                    tenant_id=cleaned_tenant,
                    binding_id=cleaned_binding,
                    owner_id=cleaned_owner,
                    token=token,
                )
        return None

    def release(self, *, lease: KnowledgeSourceLeaseToken) -> None:
        partition_key = _lease_partition_key(lease.tenant_id)
        row_key = _lease_row_key(lease.binding_id)
        existing = self._store.get(partition_key, row_key)
        if existing is None:
            return
        parsed = self._parse_lease_document(
            existing,
            expected_tenant=lease.tenant_id,
        )
        if (
            parsed["token"] != lease.token
            or parsed["owner_id"] != lease.owner_id
            or parsed["binding_id"] != lease.binding_id
        ):
            return
        self._store.delete_if_match(expected=existing)

    def _lease_document(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        owner_id: str,
        token: str,
        acquired_at_epoch: float,
        expires_at_epoch: float,
        ttl_seconds: int,
        record_version: str,
    ) -> DocumentRecord:
        data = {
            "schema_version": _LEASE_SCHEMA,
            "tenant_id": tenant_id,
            "binding_id": binding_id,
            "owner_id": owner_id,
            "token": token,
            "acquired_at_epoch": acquired_at_epoch,
            "expires_at_epoch": expires_at_epoch,
            "record_version": record_version,
        }
        _reject_secret_fields(data, kind="sync lease")
        return DocumentRecord(
            partition_key=_lease_partition_key(tenant_id),
            row_key=_lease_row_key(binding_id),
            data=data,
            ttl_seconds=ttl_seconds,
        )

    def _parse_lease_document(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
    ) -> dict[str, Any]:
        data = _data_as_dict(document.data)
        _reject_secret_fields(data, kind="sync lease")
        try:
            schema_version = data.get("schema_version")
            tenant_id = data.get("tenant_id")
            binding_id = data.get("binding_id")
            owner_id = data.get("owner_id")
            token = data.get("token")
            acquired_at = data.get("acquired_at_epoch")
            expires_at = data.get("expires_at_epoch")
            record_version = data.get("record_version")
            if schema_version != _LEASE_SCHEMA:
                raise KnowledgeSyncCorruptState("sync lease schema is invalid")
            if not isinstance(tenant_id, str) or tenant_id.strip() != expected_tenant:
                raise KnowledgeSyncCorruptState("sync lease tenant identity is invalid")
            if document.partition_key != _lease_partition_key(expected_tenant):
                raise KnowledgeSyncCorruptState("sync lease partition is invalid")
            if not isinstance(binding_id, str) or not binding_id.strip():
                raise KnowledgeSyncCorruptState(
                    "sync lease binding identity is invalid"
                )
            if document.row_key != _lease_row_key(binding_id):
                raise KnowledgeSyncCorruptState("sync lease row key is invalid")
            if not isinstance(owner_id, str) or not owner_id.strip():
                raise KnowledgeSyncCorruptState("sync lease owner identity is invalid")
            if not isinstance(token, str) or not token.strip():
                raise KnowledgeSyncCorruptState("sync lease token is invalid")
            if not isinstance(record_version, str) or not record_version.strip():
                raise KnowledgeSyncCorruptState("sync lease record version is invalid")
            if not isinstance(acquired_at, (int, float)):
                raise KnowledgeSyncCorruptState("sync lease acquired_at is invalid")
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
            "acquired_at_epoch": float(acquired_at),
            "expires_at_epoch": float(expires_at),
            "record_version": str(record_version).strip(),
        }


class DocumentStoreKnowledgeSyncCheckpointRepository:
    """CAS checkpoint repository backed by ConditionalDocumentStore."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        version_factory: VersionFactory | None = None,
    ) -> None:
        self._store = _require_conditional_document_store(document_store)
        self._version_factory = version_factory or _default_version_factory

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
        return self._parse_checkpoint(
            document,
            expected_tenant=cleaned_tenant,
            expected_binding=cleaned_binding,
        )[0]

    def commit(
        self,
        checkpoint: KnowledgeSyncCheckpoint,
        *,
        expected_previous: KnowledgeSyncCheckpoint | None,
    ) -> None:
        cleaned_tenant = _require_non_empty(
            checkpoint.tenant_id, field_name="tenant_id"
        )
        cleaned_binding = _require_non_empty(
            checkpoint.binding_id, field_name="binding_id"
        )
        partition_key = _checkpoint_partition_key(cleaned_tenant)
        row_key = _checkpoint_row_key(cleaned_binding)
        record_version = _require_non_empty(
            str(self._version_factory()),
            field_name="record_version",
        )

        if expected_previous is None:
            document = self._to_document(checkpoint, record_version=record_version)
            if not self._store.put_if_absent(document):
                raise KnowledgeSyncCheckpointConflict("checkpoint create conflict")
            return

        if (
            expected_previous.tenant_id != checkpoint.tenant_id
            or expected_previous.binding_id != checkpoint.binding_id
        ):
            raise KnowledgeSyncCorruptState(
                "checkpoint identity must remain stable across commit"
            )

        current = self._store.get(partition_key, row_key)
        if current is None:
            raise KnowledgeSyncCheckpointConflict("checkpoint cas conflict")
        public, _raw_version = self._parse_checkpoint(
            current,
            expected_tenant=cleaned_tenant,
            expected_binding=cleaned_binding,
        )
        if public != expected_previous:
            raise KnowledgeSyncCheckpointConflict("checkpoint cas conflict")
        replacement = self._to_document(checkpoint, record_version=record_version)
        if not self._store.replace_if_match(expected=current, replacement=replacement):
            raise KnowledgeSyncCheckpointConflict("checkpoint cas conflict")

    def _to_document(
        self,
        checkpoint: KnowledgeSyncCheckpoint,
        *,
        record_version: str,
    ) -> DocumentRecord:
        data = {
            "schema_version": _CHECKPOINT_SCHEMA,
            "tenant_id": checkpoint.tenant_id,
            "binding_id": checkpoint.binding_id,
            "record_version": record_version,
            "checkpoint": checkpoint.model_dump(mode="json"),
        }
        _reject_secret_fields(data, kind="sync checkpoint")
        nested = data["checkpoint"]
        if isinstance(nested, Mapping):
            _reject_secret_fields(nested, kind="sync checkpoint")
        return DocumentRecord(
            partition_key=_checkpoint_partition_key(checkpoint.tenant_id),
            row_key=_checkpoint_row_key(checkpoint.binding_id),
            data=data,
        )

    def _parse_checkpoint(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
        expected_binding: str,
    ) -> tuple[KnowledgeSyncCheckpoint, str]:
        data = _data_as_dict(document.data)
        _reject_secret_fields(data, kind="sync checkpoint")
        try:
            if data.get("schema_version") != _CHECKPOINT_SCHEMA:
                raise KnowledgeSyncCorruptState("sync checkpoint schema is invalid")
            if document.partition_key != _checkpoint_partition_key(expected_tenant):
                raise KnowledgeSyncCorruptState("sync checkpoint partition is invalid")
            if document.row_key != _checkpoint_row_key(expected_binding):
                raise KnowledgeSyncCorruptState("sync checkpoint row key is invalid")
            tenant_id = data.get("tenant_id")
            binding_id = data.get("binding_id")
            record_version = data.get("record_version")
            nested = data.get("checkpoint")
            if not isinstance(tenant_id, str) or tenant_id.strip() != expected_tenant:
                raise KnowledgeSyncCorruptState(
                    "sync checkpoint tenant identity is invalid"
                )
            if (
                not isinstance(binding_id, str)
                or binding_id.strip() != expected_binding
            ):
                raise KnowledgeSyncCorruptState(
                    "sync checkpoint binding identity is invalid"
                )
            if not isinstance(record_version, str) or not record_version.strip():
                raise KnowledgeSyncCorruptState(
                    "sync checkpoint record version is invalid"
                )
            if not isinstance(nested, Mapping):
                raise KnowledgeSyncCorruptState("sync checkpoint payload is invalid")
            _reject_secret_fields(nested, kind="sync checkpoint")
            checkpoint = KnowledgeSyncCheckpoint.model_validate(dict(nested))
            if (
                checkpoint.tenant_id != expected_tenant
                or checkpoint.binding_id != expected_binding
            ):
                raise KnowledgeSyncCorruptState(
                    "sync checkpoint payload identity is invalid"
                )
            return checkpoint, record_version.strip()
        except KnowledgeSyncCorruptState:
            raise
        except ValidationError:
            raise KnowledgeSyncCorruptState(
                "sync checkpoint payload is invalid"
            ) from None
        except Exception:
            raise KnowledgeSyncCorruptState(
                "sync checkpoint record is corrupt"
            ) from None


class DocumentStoreKnowledgeRemoteItemStateRepository:
    """Idempotent remote-item state repository with delivery markers."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        version_factory: VersionFactory | None = None,
    ) -> None:
        self._store = _require_conditional_document_store(document_store)
        self._version_factory = version_factory or _default_version_factory

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
        )[0]

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
        existing_marker = self._store.get(partition_key, marker_row)
        if existing_marker is not None:
            marker = self._parse_marker(
                existing_marker,
                expected_tenant=cleaned_tenant,
                expected_binding=cleaned_binding,
                expected_delivery=cleaned_delivery,
            )
            if marker["batch_fingerprint"] != fingerprint:
                raise KnowledgeSyncCorruptState("delivery marker fingerprint mismatch")
            if marker["status"] == _MARKER_STATUS_COMPLETED:
                return
            self._write_states(partition_key=partition_key, states=states)
            self._complete_marker(
                existing=existing_marker,
                tenant_id=cleaned_tenant,
                binding_id=cleaned_binding,
                delivery_id=cleaned_delivery,
                batch_fingerprint=fingerprint,
            )
            return

        applying = self._marker_document(
            tenant_id=cleaned_tenant,
            binding_id=cleaned_binding,
            delivery_id=cleaned_delivery,
            batch_fingerprint=fingerprint,
            status=_MARKER_STATUS_APPLYING,
            record_version=_require_non_empty(
                str(self._version_factory()),
                field_name="record_version",
            ),
        )
        if not self._store.put_if_absent(applying):
            latest = self._store.get(partition_key, marker_row)
            if latest is None:
                raise KnowledgeSyncCorruptState("delivery marker disappeared")
            marker = self._parse_marker(
                latest,
                expected_tenant=cleaned_tenant,
                expected_binding=cleaned_binding,
                expected_delivery=cleaned_delivery,
            )
            if marker["batch_fingerprint"] != fingerprint:
                raise KnowledgeSyncCorruptState("delivery marker fingerprint mismatch")
            if marker["status"] == _MARKER_STATUS_COMPLETED:
                return
            self._write_states(partition_key=partition_key, states=states)
            self._complete_marker(
                existing=latest,
                tenant_id=cleaned_tenant,
                binding_id=cleaned_binding,
                delivery_id=cleaned_delivery,
                batch_fingerprint=fingerprint,
            )
            return

        stored_applying = self._store.get(partition_key, marker_row)
        if stored_applying is None:
            raise KnowledgeSyncCorruptState("delivery marker disappeared")
        self._write_states(partition_key=partition_key, states=states)
        self._complete_marker(
            existing=stored_applying,
            tenant_id=cleaned_tenant,
            binding_id=cleaned_binding,
            delivery_id=cleaned_delivery,
            batch_fingerprint=fingerprint,
        )

    def inspect_delivery_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        prepared_state_mutations_fingerprint: str,
    ) -> KnowledgeRemoteItemStateReceipt:
        cleaned_tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        cleaned_binding = _require_non_empty(binding_id, field_name="binding_id")
        cleaned_delivery = _require_non_empty(delivery_id, field_name="delivery_id")
        cleaned_fingerprint = _require_non_empty(
            prepared_state_mutations_fingerprint,
            field_name="prepared_state_mutations_fingerprint",
        )
        if _SHA256_HEX_RE.fullmatch(cleaned_fingerprint) is None:
            raise KnowledgeSyncCorruptState("delivery receipt fingerprint is invalid")
        partition_key = _item_partition_key(
            tenant_id=cleaned_tenant,
            binding_id=cleaned_binding,
        )
        marker_row = _delivery_row_key(cleaned_delivery)
        existing_marker = self._store.get(partition_key, marker_row)
        if existing_marker is None:
            return KnowledgeRemoteItemStateReceipt(
                status=KnowledgeRemoteItemStateReceiptStatus.ABSENT
            )
        marker = self._parse_marker(
            existing_marker,
            expected_tenant=cleaned_tenant,
            expected_binding=cleaned_binding,
            expected_delivery=cleaned_delivery,
        )
        stored_fingerprint = marker.get("prepared_state_mutations_fingerprint")
        if stored_fingerprint is None:
            return KnowledgeRemoteItemStateReceipt(
                status=KnowledgeRemoteItemStateReceiptStatus.CONFLICT,
                delivery_id=cleaned_delivery,
                prepared_state_mutations_fingerprint=cleaned_fingerprint,
            )
        if stored_fingerprint != cleaned_fingerprint:
            return KnowledgeRemoteItemStateReceipt(
                status=KnowledgeRemoteItemStateReceiptStatus.CONFLICT,
                delivery_id=cleaned_delivery,
                prepared_state_mutations_fingerprint=cleaned_fingerprint,
            )
        if marker["status"] == _MARKER_STATUS_APPLYING:
            return KnowledgeRemoteItemStateReceipt(
                status=KnowledgeRemoteItemStateReceiptStatus.APPLYING,
                delivery_id=cleaned_delivery,
                prepared_state_mutations_fingerprint=cleaned_fingerprint,
            )
        return KnowledgeRemoteItemStateReceipt(
            status=KnowledgeRemoteItemStateReceiptStatus.COMPLETED,
            delivery_id=cleaned_delivery,
            prepared_state_mutations_fingerprint=cleaned_fingerprint,
        )

    def _write_states(
        self,
        *,
        partition_key: str,
        states: tuple[KnowledgeRemoteItemState, ...],
    ) -> None:
        ordered = tuple(sorted(states, key=lambda state: state.remote_id))
        for state in ordered:
            self._apply_one_state(partition_key=partition_key, state=state)

    def _complete_marker(
        self,
        *,
        existing: DocumentRecord,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        batch_fingerprint: str,
    ) -> None:
        partition_key = _item_partition_key(tenant_id=tenant_id, binding_id=binding_id)
        marker_row = _delivery_row_key(delivery_id)
        completed = self._marker_document(
            tenant_id=tenant_id,
            binding_id=binding_id,
            delivery_id=delivery_id,
            batch_fingerprint=batch_fingerprint,
            status=_MARKER_STATUS_COMPLETED,
            record_version=_require_non_empty(
                str(self._version_factory()),
                field_name="record_version",
            ),
        )
        if self._store.replace_if_match(expected=existing, replacement=completed):
            return
        latest = self._store.get(partition_key, marker_row)
        if latest is None:
            raise KnowledgeSyncCorruptState("delivery marker disappeared")
        marker = self._parse_marker(
            latest,
            expected_tenant=tenant_id,
            expected_binding=binding_id,
            expected_delivery=delivery_id,
        )
        if marker["batch_fingerprint"] != batch_fingerprint:
            raise KnowledgeSyncCorruptState("delivery marker fingerprint mismatch")
        if marker["status"] == _MARKER_STATUS_COMPLETED:
            return
        raise KnowledgeSyncCorruptState("delivery marker cas conflict")

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
                raise KnowledgeSyncCorruptState(
                    "remote item delivery identity mismatch"
                )
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
        candidate = self._state_document(
            state,
            record_version=_require_non_empty(
                str(self._version_factory()),
                field_name="record_version",
            ),
        )
        if self._store.put_if_absent(candidate):
            return
        existing = self._store.get(partition_key, row_key)
        if existing is None:
            raise KnowledgeSyncCorruptState("remote item state disappeared")
        current, _version = self._parse_state(
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
        replacement = self._state_document(
            state,
            record_version=_require_non_empty(
                str(self._version_factory()),
                field_name="record_version",
            ),
        )
        if not self._store.replace_if_match(expected=existing, replacement=replacement):
            raise KnowledgeSyncCorruptState("remote item state cas conflict")

    def _state_document(
        self,
        state: KnowledgeRemoteItemState,
        *,
        record_version: str,
    ) -> DocumentRecord:
        data = {
            "schema_version": _ITEM_STATE_SCHEMA,
            "tenant_id": state.tenant_id,
            "binding_id": state.binding_id,
            "record_version": record_version,
            "state": state.model_dump(mode="json"),
        }
        _reject_secret_fields(data, kind="remote item state")
        _reject_secret_fields(data["state"], kind="remote item state")
        return DocumentRecord(
            partition_key=_item_partition_key(
                tenant_id=state.tenant_id,
                binding_id=state.binding_id,
            ),
            row_key=_item_row_key(state.remote_id),
            data=data,
        )

    def _marker_document(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        batch_fingerprint: str,
        status: str,
        record_version: str,
    ) -> DocumentRecord:
        data = {
            "schema_version": _DELIVERY_MARKER_SCHEMA,
            "tenant_id": tenant_id,
            "binding_id": binding_id,
            "delivery_id": delivery_id,
            "batch_fingerprint": batch_fingerprint,
            "status": status,
            "record_version": record_version,
        }
        _reject_secret_fields(data, kind="delivery marker")
        return DocumentRecord(
            partition_key=_item_partition_key(
                tenant_id=tenant_id,
                binding_id=binding_id,
            ),
            row_key=_delivery_row_key(delivery_id),
            data=data,
        )

    def _parse_state(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
        expected_binding: str,
    ) -> tuple[KnowledgeRemoteItemState, str]:
        data = _data_as_dict(document.data)
        _reject_secret_fields(data, kind="remote item state")
        try:
            if data.get("schema_version") != _ITEM_STATE_SCHEMA:
                raise KnowledgeSyncCorruptState("remote item state schema is invalid")
            tenant_id = data.get("tenant_id")
            binding_id = data.get("binding_id")
            record_version = data.get("record_version")
            raw_state = data.get("state")
            if not isinstance(tenant_id, str) or tenant_id.strip() != expected_tenant:
                raise KnowledgeSyncCorruptState(
                    "remote item tenant identity is invalid"
                )
            if (
                not isinstance(binding_id, str)
                or binding_id.strip() != expected_binding
            ):
                raise KnowledgeSyncCorruptState(
                    "remote item binding identity is invalid"
                )
            if not isinstance(record_version, str) or not record_version.strip():
                raise KnowledgeSyncCorruptState("remote item record version is invalid")
            expected_partition = _item_partition_key(
                tenant_id=expected_tenant,
                binding_id=expected_binding,
            )
            if document.partition_key != expected_partition:
                raise KnowledgeSyncCorruptState("remote item partition is invalid")
            if not isinstance(raw_state, Mapping):
                raise KnowledgeSyncCorruptState("remote item state payload is invalid")
            _reject_secret_fields(raw_state, kind="remote item state")
            state = KnowledgeRemoteItemState.model_validate(dict(raw_state))
            if (
                state.tenant_id != expected_tenant
                or state.binding_id != expected_binding
            ):
                raise KnowledgeSyncCorruptState(
                    "remote item payload identity is invalid"
                )
            if document.row_key != _item_row_key(state.remote_id):
                raise KnowledgeSyncCorruptState("remote item row key is invalid")
            return state, record_version.strip()
        except KnowledgeSyncCorruptState:
            raise
        except ValidationError:
            raise KnowledgeSyncCorruptState(
                "remote item state payload is invalid"
            ) from None
        except Exception:
            raise KnowledgeSyncCorruptState(
                "remote item state record is corrupt"
            ) from None

    def _parse_marker(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
        expected_binding: str,
        expected_delivery: str,
    ) -> dict[str, str]:
        data = _data_as_dict(document.data)
        _reject_secret_fields(data, kind="delivery marker")
        try:
            if data.get("schema_version") not in {
                _DELIVERY_MARKER_SCHEMA,
                _DELIVERY_MARKER_SCHEMA_V2,
            }:
                raise KnowledgeSyncCorruptState("delivery marker schema is invalid")
            tenant_id = data.get("tenant_id")
            binding_id = data.get("binding_id")
            delivery_id = data.get("delivery_id")
            fingerprint = data.get("batch_fingerprint")
            status = data.get("status")
            record_version = data.get("record_version")
            if not isinstance(tenant_id, str) or tenant_id.strip() != expected_tenant:
                raise KnowledgeSyncCorruptState(
                    "delivery marker tenant identity is invalid"
                )
            if (
                not isinstance(binding_id, str)
                or binding_id.strip() != expected_binding
            ):
                raise KnowledgeSyncCorruptState(
                    "delivery marker binding identity is invalid"
                )
            if (
                not isinstance(delivery_id, str)
                or delivery_id.strip() != expected_delivery
            ):
                raise KnowledgeSyncCorruptState(
                    "delivery marker delivery identity is invalid"
                )
            if document.row_key != _delivery_row_key(expected_delivery):
                raise KnowledgeSyncCorruptState("delivery marker row key is invalid")
            expected_partition = _item_partition_key(
                tenant_id=expected_tenant,
                binding_id=expected_binding,
            )
            if document.partition_key != expected_partition:
                raise KnowledgeSyncCorruptState("delivery marker partition is invalid")
            if not isinstance(fingerprint, str) or not fingerprint.strip():
                raise KnowledgeSyncCorruptState(
                    "delivery marker fingerprint is invalid"
                )
            if status not in {_MARKER_STATUS_APPLYING, _MARKER_STATUS_COMPLETED}:
                raise KnowledgeSyncCorruptState("delivery marker status is invalid")
            if not isinstance(record_version, str) or not record_version.strip():
                raise KnowledgeSyncCorruptState(
                    "delivery marker record version is invalid"
                )
            mutations_fingerprint = data.get("prepared_state_mutations_fingerprint")
            if mutations_fingerprint is not None and (
                not isinstance(mutations_fingerprint, str)
                or not mutations_fingerprint.strip()
            ):
                raise KnowledgeSyncCorruptState(
                    "delivery marker prepared_state_mutations_fingerprint is invalid"
                )
            if (
                data.get("schema_version") == _DELIVERY_MARKER_SCHEMA_V2
                and mutations_fingerprint is None
            ):
                raise KnowledgeSyncCorruptState(
                    "delivery marker prepared_state_mutations_fingerprint is required"
                )
            result = {
                "batch_fingerprint": fingerprint.strip(),
                "status": str(status),
                "record_version": record_version.strip(),
            }
            if mutations_fingerprint is not None:
                result["prepared_state_mutations_fingerprint"] = str(
                    mutations_fingerprint
                ).strip()
            return result
        except KnowledgeSyncCorruptState:
            raise
        except Exception:
            raise KnowledgeSyncCorruptState(
                "delivery marker record is corrupt"
            ) from None


class DocumentStoreKnowledgeReconciliationRunRepository:
    """CAS reconciliation-run repository backed by ConditionalDocumentStore."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        policy: KnowledgeReconciliationLimitPolicy | None = None,
    ) -> None:
        self._store = _require_conditional_document_store(document_store)
        self._policy = policy or KnowledgeReconciliationLimitPolicy()

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeReconciliationRun | None:
        cleaned_tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        cleaned_binding = _require_non_empty(binding_id, field_name="binding_id")
        document = self._store.get(
            _reconciliation_run_partition_key(cleaned_tenant),
            _reconciliation_run_row_key(cleaned_binding),
        )
        if document is None:
            return None
        return self._parse_run(
            document,
            expected_tenant=cleaned_tenant,
            expected_binding=cleaned_binding,
        )

    def create_initial_run(self, run: KnowledgeReconciliationRun) -> None:
        if run.phase is not KnowledgeReconciliationRunPhase.COLLECTING:
            raise KnowledgeSyncCorruptState(
                "initial reconciliation run must start in COLLECTING"
            )
        if run.record_version != 1:
            raise KnowledgeSyncCorruptState(
                "initial reconciliation run record_version must be 1"
            )
        if run.superseded_run_id is not None:
            raise KnowledgeSyncCorruptState(
                "initial reconciliation run must not supersede"
            )
        self._validate_run_identity_keys(run)
        try:
            _validate_reconciliation_policy(run, policy=self._policy)
        except ValueError as exc:
            raise _configuration_limit_error(str(exc)) from exc
        existing = self.get(tenant_id=run.tenant_id, binding_id=run.binding_id)
        if existing is not None:
            raise KnowledgeReconciliationRunConflict(
                "reconciliation run create conflict"
            )
        document = self._to_document(run)
        if not self._store.put_if_absent(document):
            raise KnowledgeReconciliationRunConflict(
                "reconciliation run create conflict"
            )

    def cas_replace(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
    ) -> None:
        self._validate_run_identity_keys(expected)
        self._validate_run_identity_keys(replacement)
        _validate_reconciliation_identity_unchanged(
            expected=expected, replacement=replacement
        )
        if replacement.record_version != expected.record_version + 1:
            raise KnowledgeSyncCorruptState(
                "reconciliation run record_version must increment by one"
            )
        _validate_reconciliation_transition(expected=expected, replacement=replacement)
        if expected.phase in {
            KnowledgeReconciliationRunPhase.COMPLETED,
            KnowledgeReconciliationRunPhase.ABORTED,
        }:
            raise KnowledgeSyncCorruptState(
                "terminal reconciliation run cannot be cas-replaced"
            )
        try:
            _validate_reconciliation_policy(replacement, policy=self._policy)
        except ValueError as exc:
            raise _configuration_limit_error(str(exc)) from exc
        partition_key = _reconciliation_run_partition_key(expected.tenant_id)
        row_key = _reconciliation_run_row_key(expected.binding_id)
        current = self._store.get(partition_key, row_key)
        if current is None:
            raise KnowledgeReconciliationRunConflict("reconciliation run cas conflict")
        parsed = self._parse_run(
            current,
            expected_tenant=expected.tenant_id,
            expected_binding=expected.binding_id,
        )
        if parsed != expected:
            raise KnowledgeReconciliationRunConflict("reconciliation run cas conflict")
        replacement_doc = self._to_document(replacement)
        if not self._store.replace_if_match(
            expected=current, replacement=replacement_doc
        ):
            raise KnowledgeReconciliationRunConflict("reconciliation run cas conflict")

    def cas_supersede_terminal(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
    ) -> None:
        if expected.phase not in {
            KnowledgeReconciliationRunPhase.COMPLETED,
            KnowledgeReconciliationRunPhase.ABORTED,
        }:
            raise KnowledgeSyncCorruptState(
                "terminal supersession requires COMPLETED or ABORTED run"
            )
        if expected.phase is KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED:
            raise KnowledgeSyncCorruptState(
                "RECOVERY_REQUIRED cannot be automatically superseded"
            )
        if replacement.phase is not KnowledgeReconciliationRunPhase.COLLECTING:
            raise KnowledgeSyncCorruptState(
                "superseding replacement must start in COLLECTING"
            )
        if replacement.record_version != 1:
            raise KnowledgeSyncCorruptState(
                "superseding replacement record_version must be 1"
            )
        if replacement.run_id == expected.run_id:
            raise KnowledgeSyncCorruptState(
                "superseding replacement requires a new run_id"
            )
        if replacement.superseded_run_id != expected.run_id:
            raise KnowledgeSyncCorruptState(
                "superseded_run_id must reference the prior run"
            )
        if not isinstance(replacement, KnowledgeReconciliationRunCollecting):
            raise KnowledgeSyncCorruptState(
                "superseding replacement must be a COLLECTING run"
            )
        self._validate_run_identity_keys(expected)
        self._validate_run_identity_keys(replacement)
        try:
            _validate_reconciliation_policy(replacement, policy=self._policy)
        except ValueError as exc:
            raise _configuration_limit_error(str(exc)) from exc
        partition_key = _reconciliation_run_partition_key(expected.tenant_id)
        row_key = _reconciliation_run_row_key(expected.binding_id)
        current = self._store.get(partition_key, row_key)
        if current is None:
            raise KnowledgeReconciliationRunConflict(
                "reconciliation run supersede conflict"
            )
        parsed = self._parse_run(
            current,
            expected_tenant=expected.tenant_id,
            expected_binding=expected.binding_id,
        )
        if parsed != expected:
            raise KnowledgeReconciliationRunConflict(
                "reconciliation run supersede conflict"
            )
        replacement_doc = self._to_document(replacement)
        if not self._store.replace_if_match(
            expected=current, replacement=replacement_doc
        ):
            raise KnowledgeReconciliationRunConflict(
                "reconciliation run supersede conflict"
            )

    def _validate_run_identity_keys(self, run: KnowledgeReconciliationRun) -> None:
        _require_non_empty(run.tenant_id, field_name="tenant_id")
        _require_non_empty(run.binding_id, field_name="binding_id")

    def _to_document(self, run: KnowledgeReconciliationRun) -> DocumentRecord:
        data = {
            "schema_version": _RECONCILIATION_RUN_SCHEMA,
            "tenant_id": run.tenant_id,
            "binding_id": run.binding_id,
            "record_version": run.record_version,
            "run": run.model_dump(mode="json"),
        }
        _reject_secret_fields(data, kind="reconciliation run")
        nested = data["run"]
        if isinstance(nested, Mapping):
            _reject_secret_fields(nested, kind="reconciliation run")
        return DocumentRecord(
            partition_key=_reconciliation_run_partition_key(run.tenant_id),
            row_key=_reconciliation_run_row_key(run.binding_id),
            data=data,
            ttl_seconds=None,
        )

    def _parse_run(
        self,
        document: DocumentRecord,
        *,
        expected_tenant: str,
        expected_binding: str,
    ) -> KnowledgeReconciliationRun:
        data = _data_as_dict(document.data)
        _reject_secret_fields(data, kind="reconciliation run")
        try:
            if data.get("schema_version") != _RECONCILIATION_RUN_SCHEMA:
                raise KnowledgeSyncCorruptState("reconciliation run schema is invalid")
            if document.partition_key != _reconciliation_run_partition_key(
                expected_tenant
            ):
                raise KnowledgeSyncCorruptState(
                    "reconciliation run partition is invalid"
                )
            if document.row_key != _reconciliation_run_row_key(expected_binding):
                raise KnowledgeSyncCorruptState("reconciliation run row key is invalid")
            if document.ttl_seconds is not None:
                raise KnowledgeSyncCorruptState("reconciliation run ttl is invalid")
            tenant_id = data.get("tenant_id")
            binding_id = data.get("binding_id")
            record_version = data.get("record_version")
            nested = data.get("run")
            if not isinstance(tenant_id, str) or tenant_id.strip() != expected_tenant:
                raise KnowledgeSyncCorruptState(
                    "reconciliation run tenant identity is invalid"
                )
            if (
                not isinstance(binding_id, str)
                or binding_id.strip() != expected_binding
            ):
                raise KnowledgeSyncCorruptState(
                    "reconciliation run binding identity is invalid"
                )
            if not isinstance(record_version, int) or record_version < 1:
                raise KnowledgeSyncCorruptState(
                    "reconciliation run record version is invalid"
                )
            if not isinstance(nested, Mapping):
                raise KnowledgeSyncCorruptState("reconciliation run payload is invalid")
            _reject_secret_fields(nested, kind="reconciliation run")
            run = parse_knowledge_reconciliation_run(dict(nested))
            if run.tenant_id != expected_tenant or run.binding_id != expected_binding:
                raise KnowledgeSyncCorruptState(
                    "reconciliation run payload identity is invalid"
                )
            if run.record_version != record_version:
                raise KnowledgeSyncCorruptState(
                    "reconciliation run record version mismatch"
                )
            return run
        except KnowledgeSyncCorruptState:
            raise
        except ValidationError:
            raise KnowledgeSyncCorruptState(
                "reconciliation run payload is invalid"
            ) from None
        except ValueError:
            raise KnowledgeSyncCorruptState(
                "reconciliation run payload is invalid"
            ) from None
        except Exception:
            raise KnowledgeSyncCorruptState(
                "reconciliation run record is corrupt"
            ) from None
