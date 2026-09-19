# © Artur Czarnecki. All rights reserved.

"""Provider-call guards derived from active execution deadline scope (runtime)."""

from __future__ import annotations

from intergrax.contracts.execution_deadline.admission import (
    ExecutionProtectedWorkAdmissionResult,
)
from intergrax.contracts.execution_deadline.provider_timeout import (
    effective_provider_timeout_seconds,
)
from intergrax.runtime.execution.deadline_scope import (
    peek_active_execution_deadline_projection,
    peek_active_execution_protected_work_admission,
)


class ExecutionProtectedWorkDeniedError(RuntimeError):
    """Raised when canonical admission blocks a new protected provider call."""

    def __init__(self, result: ExecutionProtectedWorkAdmissionResult) -> None:
        self.result = result
        super().__init__(result.value)


def assert_protected_provider_call_allowed() -> None:
    port = peek_active_execution_protected_work_admission()
    if port is None:
        return
    decision = port.assert_can_start_protected_work()
    if decision is not ExecutionProtectedWorkAdmissionResult.AVAILABLE:
        raise ExecutionProtectedWorkDeniedError(decision)


def resolve_active_provider_timeout_seconds(
    configured_timeout_seconds: float | None,
) -> float | None:
    projection = peek_active_execution_deadline_projection()
    if projection is None or projection.deadline_at_utc is None:
        return configured_timeout_seconds
    remaining = projection.remaining_seconds
    return effective_provider_timeout_seconds(configured_timeout_seconds, remaining)
