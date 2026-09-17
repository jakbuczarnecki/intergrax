# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only model invocation facts for cross-domain consumers (LLM spine read seam)."""

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


class ModelRuntimeInvocationStatus(StrEnum):
    RECORDED = "recorded"


@dataclass(frozen=True, slots=True)
class ModelRuntimeExecutionScope:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class ModelRuntimeInvocationRecord:
    """Provider-neutral model invocation fact — no prompts or completions."""

    invocation_ref: str
    model_ref: str
    capability_label: str
    invocation_status: ModelRuntimeInvocationStatus
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    execution_id: ExecutionId
    attempt_id: AttemptId | None
    sequence_key: int
    evidence_refs: tuple[str, ...]
    safe_summary: str


@dataclass(frozen=True, slots=True)
class ModelRuntimeInvocationReadResult:
    records: tuple[ModelRuntimeInvocationRecord, ...]
    is_truncated: bool


@runtime_checkable
class ModelRuntimeInvocationReadPort(Protocol):
    """Lists recorded model invocations for one execution scope — read only."""

    @property
    def source_id(self) -> str: ...

    def list_invocations(
        self,
        scope: ModelRuntimeExecutionScope,
        *,
        limit: int,
    ) -> ModelRuntimeInvocationReadResult: ...


__all__ = [
    "ModelRuntimeExecutionScope",
    "ModelRuntimeInvocationReadPort",
    "ModelRuntimeInvocationReadResult",
    "ModelRuntimeInvocationRecord",
    "ModelRuntimeInvocationStatus",
]
