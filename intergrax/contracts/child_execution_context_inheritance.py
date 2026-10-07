# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral Execution-owned seam for propagating inherited child execution context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id


@dataclass(frozen=True, slots=True)
class ChildExecutionContextInheritanceRequest:
    """Immutable parent→child context propagation request (identity only)."""

    parent_execution_id: ExecutionId
    child_execution_id: ExecutionId

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_execution_id",
            validate_execution_id(self.parent_execution_id),
        )
        object.__setattr__(
            self,
            "child_execution_id",
            validate_execution_id(self.child_execution_id),
        )


@runtime_checkable
class ChildExecutionContextInheritancePort(Protocol):
    """
    Propagate already-established parent-bound immutable context to a child execution.

    Does not mint ExecutionId, alter lineage, widen authority, or resolve profile truth.
    """

    def inherit_child_context(self, request: ChildExecutionContextInheritanceRequest) -> None: ...


__all__ = [
    "ChildExecutionContextInheritancePort",
    "ChildExecutionContextInheritanceRequest",
]
