# © Artur Czarnecki. All rights reserved.

"""In-memory TaskCheckpointPersistence for UCA-6C qualification tests."""

from __future__ import annotations

from typing import List, Optional

from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence


class Uca6cMemoryTaskCheckpointStore(TaskCheckpointPersistence):
    def __init__(self) -> None:
        self._latest: dict[tuple[str, str], TaskCheckpoint] = {}
        self._sequence = 0

    def list_for_task(self, task_id: str, tenant_id: str) -> List[TaskCheckpoint]:
        latest = self._latest.get((task_id, tenant_id))
        return [latest] if latest else []

    def get_latest(self, task_id: str, tenant_id: str) -> Optional[TaskCheckpoint]:
        return self._latest.get((task_id, tenant_id))

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> Optional[TaskCheckpoint]:
        latest = self.get_latest(task_id, tenant_id)
        if latest is not None and latest.resume_token == resume_token:
            return latest
        return None

    def list_paused(self) -> List[TaskCheckpoint]:
        return []

    def save(
        self,
        checkpoint: TaskCheckpoint,
        *,
        expected_revision: int | None = None,
    ) -> TaskCheckpoint:
        stream_key = (checkpoint.task_id, checkpoint.tenant_id)
        current = self._latest.get(stream_key)
        current_revision = current.revision if current is not None else None
        if current_revision is None:
            if expected_revision is not None:
                raise StaleCheckpointWriteError(
                    task_id=checkpoint.task_id,
                    tenant_id=checkpoint.tenant_id,
                    expected_revision=expected_revision,
                    actual_revision=None,
                )
            next_revision = 1
        else:
            if expected_revision != current_revision:
                raise StaleCheckpointWriteError(
                    task_id=checkpoint.task_id,
                    tenant_id=checkpoint.tenant_id,
                    expected_revision=expected_revision,
                    actual_revision=current_revision,
                )
            next_revision = current_revision + 1
        self._sequence += 1
        stored = checkpoint.model_copy(
            update={"revision": next_revision, "store_sequence": self._sequence},
        )
        self._latest[stream_key] = stored
        return stored

    def cancel(self, schedule_id: str) -> None:
        _ = schedule_id

    def schedule(self, entry: object) -> object:
        return entry


__all__ = ["Uca6cMemoryTaskCheckpointStore"]
