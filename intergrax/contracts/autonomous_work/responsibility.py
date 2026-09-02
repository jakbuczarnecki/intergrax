# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Responsibility semantic contract (AW-1A)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.autonomous_work._validation import (
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WorkerInstanceId,
    validate_responsibility_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.references import (
    ResponsibilityScopeRef,
    validate_responsibility_scope_ref,
)
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision


class ResponsibilityStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class Responsibility:
    """Persistent business ownership — does not grant authority."""

    responsibility_id: ResponsibilityId
    worker_instance_id: WorkerInstanceId
    objective: str
    scope_ref: ResponsibilityScopeRef
    status: ResponsibilityStatus
    assigned_at: datetime
    revision: Revision

    def __post_init__(self) -> None:
        validate_responsibility_id(self.responsibility_id)
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "objective",
            require_non_empty_text(self.objective, label="objective"),
        )
        validate_responsibility_scope_ref(self.scope_ref)
        if type(self.status) is not ResponsibilityStatus:
            raise TypeError("status must be ResponsibilityStatus")
        object.__setattr__(
            self,
            "assigned_at",
            require_aware_utc(self.assigned_at, label="assigned_at"),
        )
        if type(self.revision) is not Revision:
            raise TypeError("revision must be Revision")
        validate_revision(self.revision)
