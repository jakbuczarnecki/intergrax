# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Active execution evidence context for durable failure recording (DIAG R2)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from intergrax.contracts.execution_failure_evidence import (
    ExecutionFailureEvidenceRecorder,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    require_active_execution_identity,
)


class ActiveExecutionEvidenceIntegrityError(Exception):
    """Raised when evidence context does not match active execution identity."""


@dataclass(frozen=True, slots=True)
class ActiveExecutionEvidenceContext:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    recorder: ExecutionFailureEvidenceRecorder


_active_execution_evidence: ContextVar[ActiveExecutionEvidenceContext | None] = (
    ContextVar(
        "active_execution_evidence",
        default=None,
    )
)


def bind_active_execution_evidence_context(
    context: ActiveExecutionEvidenceContext,
) -> Token:
    return _active_execution_evidence.set(context)


def reset_active_execution_evidence_context(token: Token) -> None:
    _active_execution_evidence.reset(token)


def peek_active_execution_evidence_context() -> ActiveExecutionEvidenceContext | None:
    return _active_execution_evidence.get()


def require_active_execution_evidence_context() -> ActiveExecutionEvidenceContext:
    context = peek_active_execution_evidence_context()
    if context is None:
        raise RuntimeError("active execution evidence context required")
    return context


def validate_active_execution_evidence_context() -> ActiveExecutionEvidenceContext:
    context = require_active_execution_evidence_context()
    active_run_id, active_attempt_id = require_active_execution_identity()
    if active_run_id != context.run_id or active_attempt_id != context.attempt_id:
        raise ActiveExecutionEvidenceIntegrityError(
            "active execution identity does not match evidence context scope",
        )
    return context


__all__ = [
    "ActiveExecutionEvidenceContext",
    "ActiveExecutionEvidenceIntegrityError",
    "bind_active_execution_evidence_context",
    "peek_active_execution_evidence_context",
    "require_active_execution_evidence_context",
    "reset_active_execution_evidence_context",
    "validate_active_execution_evidence_context",
]
