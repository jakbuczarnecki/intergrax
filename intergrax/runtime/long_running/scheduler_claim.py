# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Scheduler action claim models (PCM-CHECKPOINT-SCHEDULER-INTEGRITY · PCM-05)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.lease_claim import LeaseOwnership
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume


class ScheduledResumeClaim(LeaseOwnership):
    """Ownership record for one due scheduled resume execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schedule_id: str
    entry: ScheduledResume


class SchedulerActionClaim(LeaseOwnership):
    """Ownership record for one scheduler ledger action (e.g. human timeout)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ledger_key: str
    action: str


class ScheduledResumeCancellationError(RuntimeError):
    """Cancellation rejected because a live or uncertain claim owns the schedule."""
