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
from intergrax.runtime.events.execution_position import (
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import (
    RuntimeEventPersistence,
    _validate_through_limit,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, parse_runtime_event_payload


def _run_partition(tenant_id: str, run_id: str) -> str:
    return f"{tenant_id}|run|{run_id}"


def _task_partition(tenant_id: str, task_id: str) -> str:
    return f"{tenant_id}|task|{task_id}"


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
        scope = tenant_id or event.tenant_id or ""
        run_partition = _run_partition(scope, event.run_id)
        existing = self._get_positioned(run_partition, event.event_id)
        if existing is not None:
            return existing
        position = self._allocate_position(run_partition)
        payload = event.model_dump(mode="json")
        document_data = {
            "event": payload,
            "execution_position": position.value,
        }
        run_document = DocumentRecord(
            partition_key=run_partition,
            row_key=event.event_id,
            data=document_data,
        )
        if not self._store.put_if_absent(run_document):
            accepted = self._get_positioned(run_partition, event.event_id)
            if accepted is None:
                raise RuntimeError("execution_position_allocation_conflict")
            return accepted
        task_partition = _task_partition(scope, event.task_id)
        if self._store.get(task_partition, event.event_id) is None:
            self._store.put(
                DocumentRecord(
                    partition_key=task_partition,
                    row_key=event.event_id,
                    data=document_data,
                )
            )
        return PositionedRuntimeEvent(event=event, position=position)

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
