# © Artur Czarnecki. All rights reserved.

"""Execution suspended work re-entry port (UCA-6C-R6)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_continuation import ExecutionContinuationIdentity
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel


class ExecutionSuspendedWorkReentryDisposition(StrEnum):
    COMPLETED = "completed"
    REPLAY_COMPLETED = "replay_completed"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    NOT_READY = "not_ready"
    PAUSED_FOR_NEXT_AUTHORITY = "paused_for_next_authority"


@dataclass(frozen=True, slots=True)
class ExecutionSuspendedWorkReentryRequest:
    continuation_id: str
    identity: ExecutionContinuationIdentity
    claim_owner_id: str


@dataclass(frozen=True, slots=True)
class ExecutionSuspendedWorkReentryResult:
    disposition: ExecutionSuspendedWorkReentryDisposition
    tool_result: ToolExecutionResult[BaseModel] | None = None
    reason_detail: str = ""


@runtime_checkable
class ExecutionSuspendedWorkReentryPort(Protocol):
    """Reconstruct and invoke suspended catalog work via ToolRuntime after RESUMED."""

    def reenter_after_resume(
        self,
        request: ExecutionSuspendedWorkReentryRequest,
    ) -> ExecutionSuspendedWorkReentryResult:
        """Claim, materialize, invoke ToolRuntime, mark consumed on terminal success."""
        ...


__all__ = [
    "ExecutionSuspendedWorkReentryDisposition",
    "ExecutionSuspendedWorkReentryPort",
    "ExecutionSuspendedWorkReentryRequest",
    "ExecutionSuspendedWorkReentryResult",
]
