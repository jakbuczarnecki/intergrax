# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only ToolRuntime invocation facts for cross-domain consumers (TR-01 read seam)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)


class ToolRuntimeInvocationOutcome(StrEnum):
    REQUESTED = "requested"
    COMPLETED = "completed"
    DENIED = "denied"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ToolRuntimeExecutionScope:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class ToolRuntimeInvocationRecord:
    """Provider-neutral durable/read-model tool invocation fact — no raw args or results."""

    invocation_id: str
    tool_id: str
    execution_id: ExecutionId
    attempt_id: AttemptId | None
    sequence_key: int
    outcome: ToolRuntimeInvocationOutcome
    status_label: str
    failure_classification: str | None
    args_digest_ref: str | None
    provider_correlation_ref: str | None
    governance_evidence_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    safe_summary: str


@dataclass(frozen=True, slots=True)
class ToolRuntimeInvocationReadResult:
    records: tuple[ToolRuntimeInvocationRecord, ...]
    is_truncated: bool


@runtime_checkable
class ToolRuntimeInvocationReadPort(Protocol):
    """Lists recorded tool invocations for one execution scope — read only."""

    @property
    def source_id(self) -> str: ...

    def list_invocations(
        self,
        scope: ToolRuntimeExecutionScope,
        *,
        limit: int,
    ) -> ToolRuntimeInvocationReadResult: ...


__all__ = [
    "ToolRuntimeExecutionScope",
    "ToolRuntimeInvocationOutcome",
    "ToolRuntimeInvocationReadPort",
    "ToolRuntimeInvocationReadResult",
    "ToolRuntimeInvocationRecord",
]
