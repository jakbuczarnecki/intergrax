# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Delayed resume entries for the long-running scheduler (§26, J.4)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from intergrax.runtime.long_running.scheduled_resume_metadata import (
    has_recurrence_metadata,
    validate_scheduled_resume_metadata,
)
from intergrax.utils.time_provider import SystemTimeProvider

__all__ = [
    "ScheduledResume",
    "ScheduledResumePersistence",
    "ScheduledResumeStatus",
    "normalize_utc_iso",
    "validate_scheduled_resume_for_persistence",
]


def normalize_utc_iso(value: str, *, label: str = "timestamp") -> str:
    """Canonical UTC ISO-8601 text of an aware timestamp (fail closed on anything else).

    Naive, empty and non-ISO values are rejected: the stores compare these strings
    lexicographically, so only one canonical spelling may ever be persisted or queried.
    """
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC).isoformat()


def validate_scheduled_resume_for_persistence(entry: ScheduledResume) -> ScheduledResume:
    """Canonical persistence acceptance rule for all ScheduledResumePersistence providers."""
    validate_scheduled_resume_metadata(entry.resume_metadata)
    validated = ScheduledResume.model_validate(entry.model_dump(mode="json"))
    validate_scheduled_resume_metadata(validated.resume_metadata)
    if has_recurrence_metadata(validated.resume_metadata):
        # Recurrence keys mint future executions: only a self-consistent occurrence row
        # (what ``RecurrenceMaterializer`` writes) may carry them (F-D7-11).
        from intergrax.runtime.long_running.recurrence import validate_occurrence_entry

        validate_occurrence_entry(validated)
    return validated


if TYPE_CHECKING:
    from intergrax.runtime.long_running.scheduler_claim import ScheduledResumeClaim


class ScheduledResumeStatus(str, Enum):  # noqa: UP042 - persisted value; str() must stay the enum repr
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"


class ScheduledResume(BaseModel):
    schedule_id: str = Field(default_factory=lambda: f"sched_{uuid4().hex[:16]}")
    task_id: str
    tenant_id: str
    resume_token: str
    run_at_utc: str
    status: ScheduledResumeStatus = ScheduledResumeStatus.PENDING
    resume_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str = Field(default_factory=lambda: SystemTimeProvider.utc_now().isoformat())
    schema_version: str = "scheduled_resume.v1"
    owner_id: str | None = None
    lease_expires_at_utc: str | None = None
    fence: int = Field(default=0, ge=0)

    @field_validator("run_at_utc")
    @classmethod
    def _normalize_run_at_utc(cls, value: str) -> str:
        return normalize_utc_iso(value, label="run_at_utc")

    @model_validator(mode="after")
    def _reject_authority_bearing_metadata(self) -> Self:
        validate_scheduled_resume_metadata(self.resume_metadata)
        return self


class ScheduledResumePersistence:
    """Persistence port for delayed resume queue entries."""

    def schedule(self, entry: ScheduledResume) -> ScheduledResume:
        raise NotImplementedError

    def list_due(
        self,
        *,
        before_utc_iso: str,
        limit: int = 100,
    ) -> list[ScheduledResume]:
        """Read-only due listing — scheduler execution must use ``claim_due``."""
        raise NotImplementedError

    def claim_due(
        self,
        *,
        before_utc_iso: str,
        owner_id: str,
        lease_seconds: int,
        limit: int = 100,
    ) -> list[ScheduledResumeClaim]:
        """Atomically claim due PENDING entries for one scheduler owner."""
        raise NotImplementedError

    def complete_claim(self, claim: ScheduledResumeClaim) -> None:
        """Fence-validated completion for a claimed scheduled resume."""
        raise NotImplementedError

    def skip_claim(self, claim: ScheduledResumeClaim) -> None:
        """Fence-validated end of a claim that was deliberately not executed (``SKIPPED``).

        Providers that have no distinct status fall back to ``complete_claim``.
        """
        self.complete_claim(claim)

    def release_claim(self, claim: ScheduledResumeClaim) -> None:
        """Fence-validated hand-back of a claim to ``PENDING`` (nothing was executed).

        Providers that cannot release leave the claim to its lease.
        """
        raise NotImplementedError

    def cancel(self, schedule_id: str) -> None:
        raise NotImplementedError
