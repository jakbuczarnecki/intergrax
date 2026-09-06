# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DocumentStore-backed ``RuntimeEventPersistence`` (OBS-BUS-5 scale-out path)."""

from __future__ import annotations
from intergrax.utils import attribute_access

from collections.abc import Sequence
from typing import List

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.runtime.events.execution_position import (
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import (
    AcceptedRuntimeEvent,
    EventIdOwnershipRecord,
    LegacyRuntimeEventIdentityClaim,
    RuntimeEventIdentityClaim,
    RuntimeEventPersistence,
    RuntimeEventPersistenceIntegrityError,
    _validate_persistence_tenant_id,
    _validate_through_limit,
    build_runtime_event_identity_claim,
    decode_event_identity_claim,
    encode_event_identity_claim,
    reconcile_idempotent_event_acceptance,
    resolve_persistence_scope,
    runtime_event_identity_fingerprint,
    verify_runtime_event_identity_claim,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, parse_runtime_event_payload


def _run_partition(tenant_id: str, run_id: str) -> str:
    return f"{tenant_id}|run|{run_id}"


def _task_partition(tenant_id: str, task_id: str) -> str:
    return f"{tenant_id}|task|{task_id}"


def _event_index_partition(tenant_id: str) -> str:
    return f"{tenant_id}|event"


def _global_event_id_partition() -> str:
    return "runtime_event|event_id"


def _event_identity_claim_document(
    event_id: str,
    claim: RuntimeEventIdentityClaim,
) -> DocumentRecord:
    return DocumentRecord(
        partition_key=_global_event_id_partition(),
        row_key=event_id,
        data=encode_event_identity_claim(claim),
    )


_SEQUENCE_ROW_KEY = "__run_sequence__"
_MAX_SEQUENCE_CAS_ATTEMPTS = 16


class DocumentBackedRuntimeEventStore(RuntimeEventPersistence):
    """
    Maps canonical runtime events onto a ``DocumentStore`` partition model.

    Used by Cassandra and other wide-column backends without forking the bus contract.
    Requires ``ConditionalDocumentStore`` for distributed-safe position allocation.

    Accepted execution positions are strictly monotonic, unique, stable, and
    gap-tolerant: unused reservations are never recycled. Monotonic order does not
    require contiguous numbering (for example P1, P3, P8 is valid).
    """

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "DocumentBackedRuntimeEventStore requires ConditionalDocumentStore"
            )
        self._store = document_store

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        scope = resolve_persistence_scope(event=event, tenant_id=tenant_id)
        incoming_claim = build_runtime_event_identity_claim(
            persistence_tenant_id=scope,
            event=event,
        )
        run_partition = _run_partition(scope, event.run_id)
        existing_in_run = self._get_positioned(run_partition, event.event_id)
        if existing_in_run is not None:
            return self._finalize_idempotent_acceptance(
                scope,
                event,
                existing_in_run,
                incoming_claim,
            )

        ownership = self._load_event_ownership_record(event.event_id)
        if ownership is not None:
            return self._accept_existing_event_identity(
                scope,
                event,
                ownership,
                incoming_claim,
            )

        if not self._claim_event_ownership(event.event_id, incoming_claim):
            ownership = self._load_event_ownership_record(event.event_id)
            if ownership is None:
                raise RuntimeError("event_id ownership claim conflict")
            return self._accept_existing_event_identity(
                scope,
                event,
                ownership,
                incoming_claim,
            )

        return self._accept_new_event(scope, event, run_partition, incoming_claim)

    def _finalize_idempotent_acceptance(
        self,
        scope: str,
        event: RuntimeEvent,
        existing: PositionedRuntimeEvent,
        incoming_claim: RuntimeEventIdentityClaim,
    ) -> PositionedRuntimeEvent:
        positioned = reconcile_idempotent_event_acceptance(
            AcceptedRuntimeEvent(tenant_id=scope, positioned=existing),
            event,
            persistence_tenant_id=scope,
        )
        self._ensure_event_ownership(
            event_id=event.event_id,
            claim=incoming_claim,
        )
        self._ensure_event_index(
            tenant_id=scope,
            event_id=event.event_id,
            run_id=event.run_id,
        )
        self._ensure_task_projection(scope, event, positioned)
        return positioned

    def _accept_existing_event_identity(
        self,
        scope: str,
        event: RuntimeEvent,
        ownership: EventIdOwnershipRecord,
        incoming_claim: RuntimeEventIdentityClaim,
    ) -> PositionedRuntimeEvent:
        resolved_claim = self._resolve_effective_ownership_claim(
            event_id=event.event_id,
            ownership=ownership,
            incoming_claim=incoming_claim,
            scope=scope,
        )
        verify_runtime_event_identity_claim(resolved_claim, incoming_claim)
        run_partition = _run_partition(resolved_claim.tenant_id, str(resolved_claim.run_id))
        existing = self._get_positioned(run_partition, event.event_id)
        if existing is not None:
            canonical_fingerprint = runtime_event_identity_fingerprint(existing.event)
            if resolved_claim.fingerprint != canonical_fingerprint:
                raise RuntimeEventPersistenceIntegrityError(
                    "ownership fingerprint conflicts with canonical runtime event record",
                )
            return self._finalize_idempotent_acceptance(
                scope,
                event,
                existing,
                incoming_claim,
            )
        return self._write_canonical_after_ownership(
            scope,
            event,
            run_partition,
            incoming_claim,
        )

    def _accept_new_event(
        self,
        scope: str,
        event: RuntimeEvent,
        run_partition: str,
        incoming_claim: RuntimeEventIdentityClaim,
    ) -> PositionedRuntimeEvent:
        position = self._allocate_position(run_partition)
        positioned = self._write_canonical_run_record(
            scope,
            event,
            run_partition,
            position,
            incoming_claim,
        )
        self._ensure_event_index(
            tenant_id=scope,
            event_id=event.event_id,
            run_id=event.run_id,
        )
        self._ensure_task_projection(scope, event, positioned)
        return positioned

    def _write_canonical_after_ownership(
        self,
        scope: str,
        event: RuntimeEvent,
        run_partition: str,
        incoming_claim: RuntimeEventIdentityClaim,
    ) -> PositionedRuntimeEvent:
        position = self._allocate_position(run_partition)
        positioned = self._write_canonical_run_record(
            scope,
            event,
            run_partition,
            position,
            incoming_claim,
        )
        self._ensure_event_index(
            tenant_id=scope,
            event_id=event.event_id,
            run_id=event.run_id,
        )
        self._ensure_task_projection(scope, event, positioned)
        return positioned

    def _write_canonical_run_record(
        self,
        scope: str,
        event: RuntimeEvent,
        run_partition: str,
        position: ExecutionEventPosition,
        incoming_claim: RuntimeEventIdentityClaim,
    ) -> PositionedRuntimeEvent:
        document_data = self._canonical_document_data(event, position)
        run_document = DocumentRecord(
            partition_key=run_partition,
            row_key=event.event_id,
            data=document_data,
        )
        if not self._store.put_if_absent(run_document):
            accepted = self._get_positioned(run_partition, event.event_id)
            if accepted is None:
                raise RuntimeError("execution_position_allocation_conflict")
            return self._finalize_idempotent_acceptance(
                scope,
                event,
                accepted,
                incoming_claim,
            )
        return PositionedRuntimeEvent(event=event, position=position)

    def _resolve_effective_ownership_claim(
        self,
        *,
        event_id: str,
        ownership: EventIdOwnershipRecord,
        incoming_claim: RuntimeEventIdentityClaim,
        scope: str,
    ) -> RuntimeEventIdentityClaim:
        if isinstance(ownership, RuntimeEventIdentityClaim):
            return ownership
        legacy = ownership
        run_partition = _run_partition(legacy.tenant_id, legacy.run_id)
        canonical = self._get_positioned(run_partition, event_id)
        if canonical is None:
            raise RuntimeEventPersistenceIntegrityError(
                "legacy runtime event ownership lacks canonical identity fingerprint",
            )
        derived_claim = build_runtime_event_identity_claim(
            persistence_tenant_id=scope,
            event=canonical.event,
        )
        if legacy.tenant_id != derived_claim.tenant_id:
            raise RuntimeEventPersistenceIntegrityError(
                "legacy runtime event ownership tenant_id conflicts with canonical record",
            )
        if legacy.run_id != str(derived_claim.run_id):
            raise RuntimeEventPersistenceIntegrityError(
                "legacy runtime event ownership run_id conflicts with canonical record",
            )
        verify_runtime_event_identity_claim(derived_claim, incoming_claim)
        return self._upgrade_legacy_ownership(event_id, legacy, derived_claim)

    def _upgrade_legacy_ownership(
        self,
        event_id: str,
        legacy: LegacyRuntimeEventIdentityClaim,
        derived_claim: RuntimeEventIdentityClaim,
    ) -> RuntimeEventIdentityClaim:
        existing_record = self._store.get(_global_event_id_partition(), event_id)
        if existing_record is None:
            raise RuntimeEventPersistenceIntegrityError(
                "runtime event ownership verification failed",
            )
        current = decode_event_identity_claim(existing_record.data)
        if isinstance(current, RuntimeEventIdentityClaim):
            return current
        if (
            current.tenant_id != legacy.tenant_id
            or current.run_id != legacy.run_id
        ):
            raise RuntimeEventPersistenceIntegrityError(
                "legacy runtime event ownership upgrade conflict",
            )
        replacement = _event_identity_claim_document(event_id, derived_claim)
        if self._store.replace_if_match(
            expected=existing_record,
            replacement=replacement,
        ):
            return derived_claim
        upgraded = self._load_event_ownership_record(event_id)
        if upgraded is None:
            raise RuntimeEventPersistenceIntegrityError(
                "legacy runtime event ownership upgrade failed",
            )
        if isinstance(upgraded, LegacyRuntimeEventIdentityClaim):
            raise RuntimeEventPersistenceIntegrityError(
                "legacy runtime event ownership upgrade failed",
            )
        return upgraded

    def _canonical_document_data(
        self,
        event: RuntimeEvent,
        position: ExecutionEventPosition,
    ) -> dict[str, object]:
        return {
            "event": event.model_dump(mode="json"),
            "execution_position": position.value,
        }

    def _ensure_task_projection(
        self,
        tenant_id: str,
        event: RuntimeEvent,
        positioned: PositionedRuntimeEvent,
    ) -> None:
        task_partition = _task_partition(tenant_id, event.task_id)
        if self._store.get(task_partition, event.event_id) is not None:
            return
        self._store.put(
            DocumentRecord(
                partition_key=task_partition,
                row_key=event.event_id,
                data=self._canonical_document_data(event, positioned.position),
            )
        )

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        limit, through = _validate_through_limit(limit=limit, through=through)
        events = self._list_partition(_run_partition(tenant_id, run_id), limit=limit)
        if through is None:
            return events
        return [event for event in events if event.position <= through]

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        if type(limit) is not int or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be > 0")
        return [
            positioned.event
            for positioned in self._list_partition(
                _task_partition(tenant_id, task_id),
                limit=limit,
            )
        ]

    def get_by_event_id(
        self,
        *,
        tenant_id: str,
        event_id: EventId,
    ) -> PositionedRuntimeEvent | None:
        validated_tenant_id = _validate_persistence_tenant_id(tenant_id)
        validated_event_id = validate_event_id(event_id)
        index_partition = _event_index_partition(validated_tenant_id)
        index_record = self._store.get(index_partition, str(validated_event_id))
        if index_record is None:
            return None
        run_id = self._decode_event_index_run_id(index_record)
        run_partition = _run_partition(validated_tenant_id, run_id)
        positioned = self._get_positioned(run_partition, str(validated_event_id))
        if positioned is None:
            raise RuntimeEventPersistenceIntegrityError(
                "runtime event index references missing canonical run record",
            )
        if positioned.event.run_id != run_id:
            raise RuntimeEventPersistenceIntegrityError(
                "runtime event index run_id conflicts with canonical record",
            )
        return positioned

    def close(self) -> None:
        self._store.close()

    def _allocate_position(self, run_partition: str) -> ExecutionEventPosition:
        for _ in range(_MAX_SEQUENCE_CAS_ATTEMPTS):
            record = self._store.get(run_partition, _SEQUENCE_ROW_KEY)
            if record is None or not isinstance(record.data, dict):
                if self._store.put_if_absent(
                    DocumentRecord(
                        partition_key=run_partition,
                        row_key=_SEQUENCE_ROW_KEY,
                        data={"next_position": 2},
                    )
                ):
                    return ExecutionEventPosition(1)
                continue
            raw_next = record.data.get("next_position")
            if type(raw_next) is not int or isinstance(raw_next, bool) or raw_next < 1:
                raise RuntimeError("invalid run sequence state in document store")
            position = ExecutionEventPosition(raw_next)
            if self._store.replace_if_match(
                expected=record,
                replacement=DocumentRecord(
                    partition_key=run_partition,
                    row_key=_SEQUENCE_ROW_KEY,
                    data={"next_position": raw_next + 1},
                ),
            ):
                return position
        raise RuntimeError("execution_position_allocation_conflict")

    def _ensure_event_index(
        self,
        *,
        tenant_id: str,
        event_id: str,
        run_id: str,
    ) -> None:
        index_document = DocumentRecord(
            partition_key=_event_index_partition(tenant_id),
            row_key=event_id,
            data={"run_id": run_id},
        )
        if not self._store.put_if_absent(index_document):
            self._verify_event_index(index_document, expected_run_id=run_id)

    def _ensure_event_ownership(
        self,
        *,
        event_id: str,
        claim: RuntimeEventIdentityClaim,
    ) -> None:
        ownership_document = _event_identity_claim_document(event_id, claim)
        if not self._store.put_if_absent(ownership_document):
            existing_record = self._store.get(
                _global_event_id_partition(),
                event_id,
            )
            if existing_record is None:
                raise RuntimeEventPersistenceIntegrityError(
                    "runtime event ownership verification failed",
                )
            ownership = decode_event_identity_claim(existing_record.data)
            if isinstance(ownership, LegacyRuntimeEventIdentityClaim):
                self._upgrade_legacy_ownership(event_id, ownership, claim)
                return
            verify_runtime_event_identity_claim(ownership, claim)

    def _claim_event_ownership(
        self,
        event_id: str,
        claim: RuntimeEventIdentityClaim,
    ) -> bool:
        ownership_document = _event_identity_claim_document(event_id, claim)
        if self._store.put_if_absent(ownership_document):
            return True
        existing_record = self._store.get(_global_event_id_partition(), event_id)
        if existing_record is None:
            raise RuntimeEventPersistenceIntegrityError(
                "runtime event ownership verification failed",
            )
        ownership = decode_event_identity_claim(existing_record.data)
        if isinstance(ownership, LegacyRuntimeEventIdentityClaim):
            return False
        verify_runtime_event_identity_claim(ownership, claim)
        return False

    def _load_event_ownership_record(
        self,
        event_id: str,
    ) -> EventIdOwnershipRecord | None:
        record = self._store.get(_global_event_id_partition(), event_id)
        if record is None:
            return None
        return decode_event_identity_claim(record.data)

    def _verify_event_index(
        self,
        document: DocumentRecord,
        *,
        expected_run_id: str,
    ) -> None:
        existing = self._store.get(document.partition_key, document.row_key)
        if existing is None:
            raise RuntimeEventPersistenceIntegrityError(
                "runtime event index verification failed",
            )
        indexed_run_id = self._decode_event_index_run_id(existing)
        if indexed_run_id != expected_run_id:
            raise RuntimeEventPersistenceIntegrityError(
                "runtime event index conflicts with expected run_id",
            )

    def _decode_event_index_run_id(self, record: DocumentRecord) -> str:
        data = record.data
        if not isinstance(data, dict):
            raise RuntimeEventPersistenceIntegrityError("invalid runtime event index")
        raw_run_id = data.get("run_id")
        if type(raw_run_id) is not str or not raw_run_id.strip():
            raise RuntimeEventPersistenceIntegrityError("invalid runtime event index run_id")
        return raw_run_id

    def _get_positioned(
        self,
        partition_key: str,
        event_id: str,
    ) -> PositionedRuntimeEvent | None:
        record = self._store.get(partition_key, event_id)
        if record is None:
            return None
        return self._positioned_from_record(record)

    def _list_partition(self, partition_key: str, *, limit: int) -> List[PositionedRuntimeEvent]:
        result = self._store.query(partition_key, limit=limit)
        documents = attribute_access.optional(result, "documents", ())
        if not isinstance(documents, Sequence):
            return []
        events: list[PositionedRuntimeEvent] = []
        for doc in documents:
            if not isinstance(doc, DocumentRecord):
                continue
            if doc.row_key == _SEQUENCE_ROW_KEY:
                continue
            positioned = self._positioned_from_record(doc)
            if positioned is not None:
                events.append(positioned)
        events.sort(key=lambda positioned: positioned.position)
        return events[:limit]

    def _positioned_from_record(self, record: DocumentRecord) -> PositionedRuntimeEvent | None:
        data = record.data
        if not isinstance(data, dict):
            return None
        raw = data.get("event")
        raw_position = data.get("execution_position")
        if not isinstance(raw, dict):
            return None
        if type(raw_position) is not int or isinstance(raw_position, bool) or raw_position < 1:
            return None
        return PositionedRuntimeEvent(
            event=parse_runtime_event_payload(raw),
            position=ExecutionEventPosition(raw_position),
        )
