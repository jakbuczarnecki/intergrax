# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bound CodeCraft capability execution — post-binding runtime invocation (UCA-6C-R4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId, TaskId
from intergrax.contracts.tool_invocation_governance_approval_evidence import (
    ToolInvocationGovernanceApprovalEvidence,
)


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
    governance_approval_evidence: ToolInvocationGovernanceApprovalEvidence | None = None


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
