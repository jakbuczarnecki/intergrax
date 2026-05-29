# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Delayed resume entries for the long-running scheduler (§26, J.4)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.utils.time_provider import SystemTimeProvider


class ScheduledResumeStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


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
        raise NotImplementedError

    def mark_completed(self, schedule_id: str) -> None:
        raise NotImplementedError

    def cancel(self, schedule_id: str) -> None:
        raise NotImplementedError
