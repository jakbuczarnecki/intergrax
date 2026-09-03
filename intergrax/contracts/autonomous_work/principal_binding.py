# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker→Collaborative Principal durable binding contract (AW-3A).

The binding identifies which canonical Collaborative Principal a Worker acts as.
It does not carry authority, permissions, roles, or policy semantics.
Binding mutation is a control-plane responsibility — not Worker self-service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.autonomous_work._validation import (
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.revision import (
    Revision,
    validate_revision,
)


def validate_collaborative_principal_id(value: object) -> str:
    """Validate a canonical Collaborative Work principal identifier reference."""
    return require_non_empty_text(value, label="principal_id")


@dataclass(frozen=True, slots=True)
class WorkerPrincipalBinding:
    """Immutable durable Worker→Principal identity binding."""

    worker_instance_id: WorkerInstanceId
    principal_id: str
    created_at: datetime
    revision: Revision

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "principal_id",
            validate_collaborative_principal_id(self.principal_id),
        )
        object.__setattr__(
            self,
            "created_at",
            require_aware_utc(self.created_at, label="created_at"),
        )
        if type(self.revision) is not Revision:
            raise TypeError("revision must be Revision")
        validate_revision(self.revision)
