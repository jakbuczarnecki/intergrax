# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bound CodeCraft capability execution — post-binding runtime invocation (UCA-6C-R4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work._validation import require_non_empty_text
from intergrax.contracts.execution_identity import ExecutionId, TaskId


class CodeCraftBoundCapabilityExecutionOutcome(StrEnum):
    """Domain execution outcome — mapped to EE dispatch dispositions by handler."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CodeCraftBoundCapabilityExecutionRequest:
    """Invoke a craft-scoped executable capability under active execution identity."""

    craft_id: str
    tenant_id: str
    task_id: TaskId
    run_id: str | None
    execution_id: ExecutionId
    execution_request_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_request_id",
            require_non_empty_text(
                self.execution_request_id,
                label="execution_request_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class CodeCraftBoundCapabilityExecutionResult:
    outcome: CodeCraftBoundCapabilityExecutionOutcome
    reason_detail: str = ""


@runtime_checkable
class CodeCraftBoundCapabilityExecutionPort(Protocol):
    """Public CodeCraft seam for qualified bound-capability execution."""

    def execute(
        self,
        request: CodeCraftBoundCapabilityExecutionRequest,
    ) -> CodeCraftBoundCapabilityExecutionResult:
        """Run craft-scoped capability via legal CodeCraft runtime/tool boundaries."""
        ...


__all__ = [
    "CodeCraftBoundCapabilityExecutionOutcome",
    "CodeCraftBoundCapabilityExecutionPort",
    "CodeCraftBoundCapabilityExecutionRequest",
    "CodeCraftBoundCapabilityExecutionResult",
]
