# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Terminal execution diagnostic lifecycle bridge (ONE-SPINE-3)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.logging import IntergraxLogging
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationResult,
)
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTrigger,
    TerminalExecutionDiagnosticTriggerProtocol,
)


def invoke_terminal_execution_diagnostics(
    trigger: TerminalExecutionDiagnosticTriggerProtocol | None,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    observed_at: datetime,
) -> DiagnosticOrchestrationResult | None:
    """
    Invoke derived diagnostic post-processing after terminal execution truth is persisted.

    Diagnostic failures are surfaced operationally and must not alter business execution outcome.
    """
    if trigger is None:
        return None
    if not (tenant_id or "").strip():
        return None
    logger = IntergraxLogging.get_logger(__name__, component="diagnostics")
    try:
        result = trigger.trigger_for_terminal_execution(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            observed_at=observed_at,
        )
    except Exception:
        logger.exception(
            "Terminal execution diagnostic post-processing failed",
            extra={
                "tenant_id": tenant_id,
                "task_id": str(task_id),
                "run_id": str(run_id),
            },
        )
        return None

    logger.info(
        "Terminal execution diagnostic post-processing completed",
        extra={
            "tenant_id": tenant_id,
            "task_id": str(task_id),
            "run_id": str(run_id),
            "problems_created": len(result.lifecycle_result.created),
            "problems_updated": len(result.lifecycle_result.updated),
            "problems_unchanged": len(result.lifecycle_result.unchanged),
        },
    )
    return result


__all__ = ["invoke_terminal_execution_diagnostics"]
