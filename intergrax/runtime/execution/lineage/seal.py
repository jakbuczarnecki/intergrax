# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution lineage seal and pause continuity helpers (DG-001 R1)."""

from __future__ import annotations

import logging

from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptClosureKind,
    ExecutionLineagePersistence,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.runtime.execution.lineage.active_lineage import peek_active_execution_lineage

_LOGGER = logging.getLogger(__name__)


def closure_kind_for_terminal_outcome(
    outcome: ExecutionTerminalOutcome,
) -> ExecutionLineageAttemptClosureKind:
    if outcome is ExecutionTerminalOutcome.COMPLETED:
        return ExecutionLineageAttemptClosureKind.COMPLETED
    if outcome is ExecutionTerminalOutcome.CANCELLED:
        return ExecutionLineageAttemptClosureKind.CANCELLED
    return ExecutionLineageAttemptClosureKind.FAILED


def seal_lineage_attempt(
    persistence: ExecutionLineagePersistence,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    closure_kind: ExecutionLineageAttemptClosureKind,
) -> None:
    scope = build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    try:
        persistence.seal_attempt(scope, closure_kind)
    except ExecutionLineageUnavailableError:
        _LOGGER.warning(
            "execution lineage seal unavailable for attempt %s",
            attempt_id,
            exc_info=True,
        )


def close_active_lineage_segment_for_resume(
    persistence: ExecutionLineagePersistence,
) -> None:
    active = peek_active_execution_lineage()
    if active is None:
        return
    persistence.close_segment_for_resume(
        active.scope,
        active.segment_root_execution_id,
    )
