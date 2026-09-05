# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical background worker re-entry admission (P0C-7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.attempt_lifecycle import AttemptLifecycleError
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalConflictError,
    ExecutionTerminalError,
    ExecutionTerminalRecord,
)
from intergrax.runtime.background_execution.bootstrap import (
    BackgroundExecutionIdentity,
    resolve_background_execution,
)
from intergrax.runtime.background_execution.identity_persistence import (
    BackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.execution.attempt_lifecycle.service import AttemptLifecycleService
from intergrax.runtime.execution.execution_terminal.persistence import (
    validate_terminal_run_id_consistency,
)
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService


class BackgroundExecutionReentryDisposition(StrEnum):
    """Admission outcome for one background transport delivery."""

    EXECUTE = "execute"
    TERMINAL_ALREADY_RECORDED = "terminal_already_recorded"


@dataclass(frozen=True, slots=True)
class BackgroundExecutionReentry:
    """Canonical re-entry resolution for one background transport delivery."""

    identity: BackgroundExecutionIdentity
    disposition: BackgroundExecutionReentryDisposition


class BackgroundExecutionReentryAdmissionError(RuntimeError):
    """Fail-closed re-entry admission failure."""


def _reconcile_active_attempt(
    *,
    tenant_id: str,
    bootstrap: BackgroundExecutionIdentity,
    attempt_lifecycle: AttemptLifecycleService,
) -> BackgroundExecutionIdentity:
    try:
        active = attempt_lifecycle.get_active_attempt_id(
            tenant_id=tenant_id,
            run_id=bootstrap.run_id,
        )
    except AttemptLifecycleError as exc:
        raise BackgroundExecutionReentryAdmissionError(
            "attempt lifecycle authority is corrupt or unavailable",
        ) from exc
    if active is None:
        try:
            state = attempt_lifecycle.record_initial_attempt(
                tenant_id=tenant_id,
                run_id=bootstrap.run_id,
                attempt_id=bootstrap.attempt_id,
            )
        except AttemptLifecycleError as exc:
            raise BackgroundExecutionReentryAdmissionError(
                "attempt lifecycle initial record failed",
            ) from exc
        active = state.active_attempt_id
    return BackgroundExecutionIdentity(
        tenant_id=tenant_id,
        task_id=bootstrap.task_id,
        run_id=bootstrap.run_id,
        attempt_id=active,
    )


def _check_terminal_authority(
    *,
    tenant_id: str,
    task_id: str,
    bootstrap: BackgroundExecutionIdentity,
    execution_terminal: ExecutionTerminalService,
) -> ExecutionTerminalRecord | None:
    try:
        record = execution_terminal.get_terminal_record(
            tenant_id=tenant_id,
            task_id=task_id,
        )
    except ExecutionTerminalError as exc:
        raise BackgroundExecutionReentryAdmissionError(
            "execution terminal authority is corrupt or unavailable",
        ) from exc
    if record is None:
        return None
    try:
        validate_terminal_run_id_consistency(record, bootstrap.run_id)
    except ExecutionTerminalConflictError as exc:
        raise BackgroundExecutionReentryAdmissionError(
            "execution terminal run_id conflicts with background execution identity",
        ) from exc
    return record


def admit_background_execution_reentry(
    *,
    transport_ref: BackgroundTransportExecutionRef,
    identity_persistence: BackgroundExecutionIdentityPersistence,
    attempt_lifecycle: AttemptLifecycleService,
    execution_terminal: ExecutionTerminalService,
    task_tenant_id: str | None = None,
) -> BackgroundExecutionReentry:
    """
    Resolve canonical background execution identity for worker re-entry.

    Transport mapping stabilizes TaskId/RunId/bootstrap AttemptId. Active AttemptId
    comes from AttemptLifecycleService once lifecycle exists. Terminal authority
    blocks handler execution without minting a new attempt.
    """
    try:
        bootstrap = resolve_background_execution(
            transport_ref=transport_ref,
            identity_persistence=identity_persistence,
            task_tenant_id=task_tenant_id,
        )
    except ValueError as exc:
        raise BackgroundExecutionReentryAdmissionError(str(exc)) from exc
    except RuntimeError as exc:
        raise BackgroundExecutionReentryAdmissionError(
            "background execution identity authority is corrupt or unavailable",
        ) from exc

    terminal = _check_terminal_authority(
        tenant_id=bootstrap.tenant_id,
        task_id=str(bootstrap.task_id),
        bootstrap=bootstrap,
        execution_terminal=execution_terminal,
    )
    identity = _reconcile_active_attempt(
        tenant_id=bootstrap.tenant_id,
        bootstrap=bootstrap,
        attempt_lifecycle=attempt_lifecycle,
    )
    if terminal is not None:
        return BackgroundExecutionReentry(
            identity=identity,
            disposition=BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED,
        )
    return BackgroundExecutionReentry(
        identity=identity,
        disposition=BackgroundExecutionReentryDisposition.EXECUTE,
    )
