# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ConditionalDocumentStore-backed durable worker obstacle capability need index (UCA-6C-R6-R5.9)."""

from __future__ import annotations

import threading

from intergrax.autonomous_work.repository import AutonomousWorkEntityConflict
from intergrax.autonomous_work.worker_capability_need_serialization import (
    worker_capability_need_from_json,
    worker_capability_need_to_json,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)

_PARTITION = "autonomous_work.worker_obstacle_capability_need"
_SCHEMA = "worker_obstacle_capability_need.v1"


class DocumentStoreWorkerRecoveryObstacleCapabilityNeedRepository:
    """Durable AW-owned obstacle capability need index — immutable insert semantics."""

    __slots__ = ("_document_store", "_lock")

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "worker obstacle capability need persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store
        self._lock = threading.RLock()

    def record_obstacle_capability_need(self, need: WorkerCapabilityNeed) -> None:
        worker_id = str(need.worker_instance_id)
        obstacle_id = need.obstacle_id
        row_key = f"{worker_id}:{obstacle_id}"
        record_json = worker_capability_need_to_json(need)
        document = DocumentRecord(
            partition_key=_PARTITION,
            row_key=row_key,
            data={
                "schema": _SCHEMA,
                "record_json": record_json,
            },
        )
        with self._lock:
            inserted = self._document_store.put_if_absent(document)
            if inserted:
                return
            existing = self.get_obstacle_capability_need(
                worker_instance_id=need.worker_instance_id,
                obstacle_id=obstacle_id,
            )
            if existing is None:
                raise RuntimeError(
                    "worker obstacle capability need record missing after insert conflict",
                )
            if existing == need:
                return
            raise AutonomousWorkEntityConflict(
                "worker obstacle capability need already exists with different content "
                f"for {worker_id}:{obstacle_id}",
            )

    def get_obstacle_capability_need(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        obstacle_id: str,
    ) -> WorkerCapabilityNeed | None:
        row_key = f"{worker_instance_id}:{obstacle_id}"
        with self._lock:
            record = self._document_store.get(_PARTITION, row_key)
        if record is None:
            return None
        raw = record.data.get("record_json")
        if not isinstance(raw, str):
            raise RuntimeError("corrupt worker obstacle capability need record")
        return worker_capability_need_from_json(raw)


__all__ = ["DocumentStoreWorkerRecoveryObstacleCapabilityNeedRepository"]
