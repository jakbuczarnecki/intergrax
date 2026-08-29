# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Terminal execution diagnostic lifecycle bridge (ONE-SPINE-3)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.execution_identity import (
    RunId,
    TaskId,
    bind_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.logging import IntergraxLogging
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationResult,
)
from intergrax.runtime.diagnostics.diagnostic_subsystem_failure_evidence import (
    record_diagnostic_subsystem_failure,
)
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTrigger,
    TerminalExecutionDiagnosticTriggerProtocol,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.execution.boundary import ExecutionIdentityBinding


def _persist_diagnostic_subsystem_failure(
    event_bus: RuntimeEventBus,
    *,
    execution_identity: ExecutionIdentityBinding | None,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    error_type: str,
    observed_at: datetime,
) -> None:
    if execution_identity is not None:
        if execution_identity.run_id != run_id:
            raise RuntimeError("run_id conflicts with terminal execution identity")
        identity_token = bind_active_execution_identity(
            run_id=execution_identity.run_id,
            attempt_id=execution_identity.attempt_id,
            execution_id=execution_identity.execution_id,
        )
        try:
            record_diagnostic_subsystem_failure(
                event_bus,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                error_type=error_type,
                observed_at=observed_at,
            )
        finally:
            reset_active_execution_identity(identity_token)
        return

    record_diagnostic_subsystem_failure(
        event_bus,
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        error_type=error_type,
        observed_at=observed_at,
    )


def invoke_terminal_execution_diagnostics(
    trigger: TerminalExecutionDiagnosticTriggerProtocol | None,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    observed_at: datetime,
    event_bus: RuntimeEventBus | None = None,
    execution_identity: ExecutionIdentityBinding | None = None,
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
    except Exception as exc:
        logger.exception(
            "Terminal execution diagnostic post-processing failed",
            extra={
                "tenant_id": tenant_id,
                "task_id": str(task_id),
                "run_id": str(run_id),
            },
        )
        if event_bus is not None:
            try:
                _persist_diagnostic_subsystem_failure(
                    event_bus,
                    execution_identity=execution_identity,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    run_id=run_id,
                    error_type=type(exc).__name__,
                    observed_at=observed_at,
                )
            except RuntimeError as evidence_exc:
                if "conflicts with" in str(evidence_exc):
                    raise evidence_exc
                logger.exception(
                    "Failed to persist diagnostic subsystem failure evidence",
                    extra={
                        "tenant_id": tenant_id,
                        "task_id": str(task_id),
                        "run_id": str(run_id),
                    },
                )
            except Exception:
                logger.exception(
                    "Failed to persist diagnostic subsystem failure evidence",
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
