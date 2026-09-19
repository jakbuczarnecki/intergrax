# © Artur Czarnecki. All rights reserved.

"""Active execution deadline projection binding (process-local runtime scope)."""

from __future__ import annotations

from contextvars import ContextVar, Token

from intergrax.contracts.execution_deadline.admission import ExecutionProtectedWorkAdmissionPort
from intergrax.contracts.execution_deadline.clock import MonotonicClockPort
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
_active_monotonic_clock: ContextVar[MonotonicClockPort | None] = ContextVar(
    "active_execution_monotonic_clock",
    default=None,
)


def bind_active_execution_deadline_scope(
    *,
    projection: ExecutionDeadlineProjection,
    admission: ExecutionProtectedWorkAdmissionPort,
    monotonic_clock: MonotonicClockPort,
) -> tuple[Token, Token, Token]:
    projection_token = _active_deadline_projection.set(projection)
    admission_token = _active_protected_work_admission.set(admission)
    clock_token = _active_monotonic_clock.set(monotonic_clock)
    return projection_token, admission_token, clock_token


def reset_active_execution_deadline_scope(
    projection_token: Token,
    admission_token: Token,
    clock_token: Token,
) -> None:
    _active_monotonic_clock.reset(clock_token)
    _active_protected_work_admission.reset(admission_token)
    _active_deadline_projection.reset(projection_token)


def peek_active_execution_deadline_projection() -> ExecutionDeadlineProjection | None:
    return _active_deadline_projection.get()


def peek_active_execution_protected_work_admission() -> (
    ExecutionProtectedWorkAdmissionPort | None
):
    return _active_protected_work_admission.get()


def peek_active_execution_monotonic_clock() -> MonotonicClockPort | None:
    return _active_monotonic_clock.get()


def require_active_execution_protected_work_admission() -> ExecutionProtectedWorkAdmissionPort:
    port = peek_active_execution_protected_work_admission()
    if port is None:
        raise RuntimeError("active execution protected work admission required")
    return port
