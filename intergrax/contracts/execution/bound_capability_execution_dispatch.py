# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Minimal binding-handler dispatch surface — no acquisition/qualification provenance (UCA-6C-R6-R5.8-H1-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.autonomous_work._validation import (
    require_non_empty_text,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.execution_identity import TaskId, validate_task_id


@dataclass(frozen=True, slots=True)
class BoundCapabilityExecutionDispatchRequest:
    """Handler intake after binding — execution target and identity only."""

    execution_request_id: str
    execution_target: QualifiedCapabilityExecutionTarget
    tenant_id: str
    task_id: TaskId

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_request_id",
            require_non_empty_text(
                self.execution_request_id,
                label="execution_request_id",
            ),
        )
        if type(self.execution_target) is not QualifiedCapabilityExecutionTarget:
            raise TypeError(
                "execution_target must be QualifiedCapabilityExecutionTarget"
            )
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        validate_task_id(self.task_id)


__all__ = ["BoundCapabilityExecutionDispatchRequest"]
