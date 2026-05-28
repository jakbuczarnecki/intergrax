# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 persistence contract for long-running task checkpoints (§26, §42.9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from intergrax.runtime.long_running.models import TaskCheckpoint


class TaskCheckpointReader(ABC):
    """Read-only checkpoint access for debug surfaces and resume workflows."""

    @abstractmethod
    def list_for_task(self, task_id: str, tenant_id: str) -> List[TaskCheckpoint]:
        ...

    @abstractmethod
    def get_latest(self, task_id: str, tenant_id: str) -> Optional[TaskCheckpoint]:
        ...

    @abstractmethod
    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> Optional[TaskCheckpoint]:
        ...


class TaskCheckpointPersistence(TaskCheckpointReader, ABC):
    """Append-only checkpoint store (implementations: SQLite, …)."""

    @abstractmethod
    def save(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        ...
