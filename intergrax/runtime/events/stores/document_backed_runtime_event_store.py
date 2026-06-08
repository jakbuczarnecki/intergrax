# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DocumentStore-backed ``RuntimeEventPersistence`` (OBS-BUS-5 scale-out path)."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent


@runtime_checkable
class DocumentStoreLike(Protocol):
    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None: ...

    def put(self, document: DocumentRecord) -> None: ...

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
    ) -> object: ...

    def close(self) -> None: ...


def _run_partition(tenant_id: str, run_id: str) -> str:
    return f"{tenant_id}|run|{run_id}"


def _task_partition(tenant_id: str, task_id: str) -> str:
    return f"{tenant_id}|task|{task_id}"


class DocumentBackedRuntimeEventStore(RuntimeEventPersistence):
    """
    Maps canonical runtime events onto a ``DocumentStore`` partition model.

    Used by Cassandra and other wide-column backends without forking the bus contract.
    """

    def __init__(self, document_store: DocumentStoreLike) -> None:
        self._store = document_store

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> None:
        scope = tenant_id or event.tenant_id or ""
        payload = event.model_dump(mode="json")
        row_key = event.event_id
        for partition in (
            _run_partition(scope, event.run_id),
            _task_partition(scope, event.task_id),
        ):
            if self._store.get(partition, row_key) is not None:
                continue
            self._store.put(
                DocumentRecord(
                    partition_key=partition,
                    row_key=row_key,
                    data={"event": payload},
                )
            )

    def list_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        return self._list_partition(_run_partition(tenant_id, run_id), limit=limit)

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        return self._list_partition(_task_partition(tenant_id, task_id), limit=limit)

    def close(self) -> None:
        self._store.close()

    def _list_partition(self, partition_key: str, *, limit: int) -> List[RuntimeEvent]:
        result = self._store.query(partition_key, limit=limit)
        documents = getattr(result, "documents", ())
        events: list[RuntimeEvent] = []
        for doc in documents:
            data = doc.data if hasattr(doc, "data") else {}
            if not isinstance(data, dict):
                continue
            raw = data.get("event")
            if isinstance(raw, dict):
                events.append(RuntimeEvent.model_validate(raw))
        events.sort(key=lambda evt: evt.timestamp)
        return events[:limit]
