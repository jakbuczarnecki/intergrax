# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Delayed resume entries for the long-running scheduler (§26, J.4)."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.runtime.long_running.scheduler_claim import ScheduledResumeClaim


class ScheduledResumeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


class ScheduledResume(BaseModel):
    schedule_id: str = Field(default_factory=lambda: f"sched_{uuid4().hex[:16]}")
    task_id: str
    tenant_id: str
    resume_token: str
    run_at_utc: str
    status: ScheduledResumeStatus = ScheduledResumeStatus.PENDING
    resume_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str = Field(
        default_factory=lambda: SystemTimeProvider.utc_now().isoformat()
    )
    schema_version: str = "scheduled_resume.v1"
    owner_id: Optional[str] = None
    lease_expires_at_utc: Optional[str] = None
    fence: int = Field(default=0, ge=0)


class ScheduledResumePersistence:
    """Persistence port for delayed resume queue entries."""

    def schedule(self, entry: ScheduledResume) -> ScheduledResume:
        raise NotImplementedError

    def list_due(
        self,
        *,
        before_utc_iso: str,
        limit: int = 100,
    ) -> List[ScheduledResume]:
        """Read-only due listing — scheduler execution must use ``claim_due``."""
        raise NotImplementedError

    def claim_due(
        self,
        *,
        before_utc_iso: str,
        owner_id: str,
        lease_seconds: int,
        limit: int = 100,
    ) -> List[ScheduledResumeClaim]:
        """Atomically claim due PENDING entries for one scheduler owner."""
        raise NotImplementedError

    def complete_claim(self, claim: ScheduledResumeClaim) -> None:
        """Fence-validated completion for a claimed scheduled resume."""
        raise NotImplementedError

    def mark_completed(self, schedule_id: str) -> None:
        """Legacy completion without ownership fencing — prefer ``complete_claim``."""
        raise NotImplementedError

    def cancel(self, schedule_id: str) -> None:
        raise NotImplementedError
