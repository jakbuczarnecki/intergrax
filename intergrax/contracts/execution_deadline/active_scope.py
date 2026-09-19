# © Artur Czarnecki. All rights reserved.

"""Active execution deadline projection binding (process-local)."""

from __future__ import annotations

from contextvars import ContextVar, Token

from intergrax.contracts.execution_deadline.admission import ExecutionProtectedWorkAdmissionPort
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection

_active_deadline_projection: ContextVar[ExecutionDeadlineProjection | None] = ContextVar(
    "active_execution_deadline_projection",
    default=None,
)
_active_protected_work_admission: ContextVar[ExecutionProtectedWorkAdmissionPort | None] = (
    ContextVar(
        "active_execution_protected_work_admission",
        default=None,
    )
)


def bind_active_execution_deadline_scope(
    *,
    projection: ExecutionDeadlineProjection,
    admission: ExecutionProtectedWorkAdmissionPort,
) -> tuple[Token, Token]:
    projection_token = _active_deadline_projection.set(projection)
    admission_token = _active_protected_work_admission.set(admission)
    return projection_token, admission_token


def reset_active_execution_deadline_scope(
    projection_token: Token,
    admission_token: Token,
) -> None:
    _active_protected_work_admission.reset(admission_token)
    _active_deadline_projection.reset(projection_token)


def peek_active_execution_deadline_projection() -> ExecutionDeadlineProjection | None:
    return _active_deadline_projection.get()


def peek_active_execution_protected_work_admission() -> (
    ExecutionProtectedWorkAdmissionPort | None
):
    return _active_protected_work_admission.get()


def require_active_execution_protected_work_admission() -> ExecutionProtectedWorkAdmissionPort:
    port = peek_active_execution_protected_work_admission()
    if port is None:
        raise RuntimeError("active execution protected work admission required")
    return port
