# © Artur Czarnecki. All rights reserved.

"""GR-5-R3 — replaceable Task/Human projection from canonical continuation snapshots.

Canonical lifecycle authority remains :class:`ExecutionContinuationPort`.
Projection sinks may vary by enterprise UI/storage; lifecycle semantics are fixed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_continuation import PendingExecutionContinuation


class ExecutionContinuationProjectionError(RuntimeError):
    """Observable projection failure — canonical continuation truth is unchanged."""


class ExecutionContinuationProjectionStatus(StrEnum):
    APPLIED = "applied"
    STALE_IGNORED = "stale_ignored"
    TASK_IDENTITY_MISMATCH = "task_identity_mismatch"


class ExecutionContinuationProjectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ExecutionContinuationProjectionStatus
    applied_revision: int | None = Field(default=None, ge=1)


@runtime_checkable
class ExecutionContinuationProjectionSink(Protocol):
    """Plugin boundary for projecting one canonical continuation snapshot."""

    def project(
        self,
        pending: PendingExecutionContinuation,
    ) -> ExecutionContinuationProjectionResult:
        """Apply or ignore ``pending``; must be safe to call twice for the same revision."""


__all__ = [
    "ExecutionContinuationProjectionError",
    "ExecutionContinuationProjectionResult",
    "ExecutionContinuationProjectionSink",
    "ExecutionContinuationProjectionStatus",
]
