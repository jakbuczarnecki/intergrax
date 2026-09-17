# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only external work facts for cross-domain consumers (external ops spine seam)."""

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


class ExternalWorkRuntimeStatus(StrEnum):
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExternalWorkRuntimeExecutionScope:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class ExternalWorkRuntimeFactRecord:
    """Provider-neutral external work boundary fact — no vendor payloads."""

    work_ref: str
    work_class: str
    work_status: ExternalWorkRuntimeStatus
    provider_ref: str
    failure_classification: str
    retryable: bool
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    execution_id: ExecutionId
    attempt_id: AttemptId | None
    sequence_key: int
    evidence_refs: tuple[str, ...]
    safe_summary: str


@dataclass(frozen=True, slots=True)
class ExternalWorkRuntimeFactReadResult:
    records: tuple[ExternalWorkRuntimeFactRecord, ...]
    is_truncated: bool


@runtime_checkable
class ExternalWorkRuntimeFactReadPort(Protocol):
    """Lists recorded external work facts for one execution scope — read only."""

    @property
    def source_id(self) -> str: ...

    def list_work_facts(
        self,
        scope: ExternalWorkRuntimeExecutionScope,
        *,
        limit: int,
    ) -> ExternalWorkRuntimeFactReadResult: ...


__all__ = [
    "ExternalWorkRuntimeExecutionScope",
    "ExternalWorkRuntimeFactReadPort",
    "ExternalWorkRuntimeFactReadResult",
    "ExternalWorkRuntimeFactRecord",
    "ExternalWorkRuntimeStatus",
]
