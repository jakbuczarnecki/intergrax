# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 persistence contract for long-running task checkpoints (§26, §42.9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.scheduler_claim import SchedulerActionClaim
from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumePersistence,
)


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

    @abstractmethod
    def list_paused(self) -> List[TaskCheckpoint]:
        """Latest checkpoint per task where state is a long-running pause."""
        ...


class TaskCheckpointPersistence(TaskCheckpointReader, ScheduledResumePersistence, ABC):
    """Append-only checkpoint store with optional scheduler tables."""

    @abstractmethod
    def save(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        ...


class SchedulerLedger(ABC):
    """Atomic ownership ledger for scheduler actions (human timeout path)."""

    @abstractmethod
    def has_action(self, ledger_key: str) -> bool:
        """Returns True when the action completed durably."""
        ...

    @abstractmethod
    def claim_action(
        self,
        ledger_key: str,
        owner_id: str,
        lease_seconds: int,
        *,
        action: str,
    ) -> Optional[SchedulerActionClaim]:
        """
        Atomically acquire ownership for one scheduler action.

        Returns None when action is completed, actively owned, or uncertain.
        """
        ...

    @abstractmethod
    def complete_action(self, claim: SchedulerActionClaim) -> None:
        """Fence-validated durable completion for a claimed scheduler action."""
        ...

    @abstractmethod
    def record_action(self, ledger_key: str, *, action: str) -> None:
        """Legacy completion without ownership fencing — prefer ``claim_action``."""
        ...
