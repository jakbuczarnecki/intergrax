# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only memory usage facts for cross-domain consumers (MEM spine read seam)."""

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


class MemoryRuntimeOperationClass(StrEnum):
    READ = "read"
    WRITE = "write"


class MemoryRuntimeOperationStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    RECORDED = "recorded"


@dataclass(frozen=True, slots=True)
class MemoryRuntimeExecutionScope:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class MemoryRuntimeOperationRecord:
    """Provider-neutral memory operation fact — no memory payload content."""

    operation_ref: str
    memory_class: str
    operation_class: MemoryRuntimeOperationClass
    operation_status: MemoryRuntimeOperationStatus
    record_ref: str | None
    source_category: str
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    execution_id: ExecutionId
    attempt_id: AttemptId | None
    sequence_key: int
    evidence_refs: tuple[str, ...]
    safe_summary: str


@dataclass(frozen=True, slots=True)
class MemoryRuntimeOperationReadResult:
    records: tuple[MemoryRuntimeOperationRecord, ...]
    is_truncated: bool


@runtime_checkable
class MemoryRuntimeOperationReadPort(Protocol):
    """Lists recorded memory operations for one execution scope — read only."""

    @property
    def source_id(self) -> str: ...

    def list_operations(
        self,
        scope: MemoryRuntimeExecutionScope,
        *,
        limit: int,
    ) -> MemoryRuntimeOperationReadResult: ...


__all__ = [
    "MemoryRuntimeExecutionScope",
    "MemoryRuntimeOperationClass",
    "MemoryRuntimeOperationReadPort",
    "MemoryRuntimeOperationReadResult",
    "MemoryRuntimeOperationRecord",
    "MemoryRuntimeOperationStatus",
]
