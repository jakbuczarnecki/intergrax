# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Production trigger for bounded terminal-execution diagnostic orchestration (ONE-SPINE-3)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.execution_identity import RunId, TaskId, validate_run_id, validate_task_id
from intergrax.runtime.diagnostics.deterministic_problem_grouping import STRATEGY_ID
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticExecutionScope,
    DiagnosticOrchestrationRequest,
    DiagnosticOrchestrationResult,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.observability.problem_signal import PlatformProblemSignal


class TerminalExecutionDiagnosticTrigger:
    """
    Platform-owned adapter that submits one terminal execution scope to the canonical
    ``DiagnosticOrchestrator``.

    Contains no diagnostic logic — only request assembly and orchestrator invocation.
    """

    def __init__(self, orchestrator: DiagnosticOrchestrator) -> None:
        self._orchestrator = orchestrator

    def trigger_for_terminal_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        observed_at: datetime,
        problem_signals: tuple[PlatformProblemSignal, ...] = (),
    ) -> DiagnosticOrchestrationResult:
        resolved_tenant = tenant_id.strip()
        if not resolved_tenant:
            raise ValueError("tenant_id must be non-empty for terminal diagnostic trigger")
        validated_task_id = validate_task_id(task_id)
        validated_run_id = validate_run_id(run_id)
        if observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")

        request = DiagnosticOrchestrationRequest(
            tenant_id=resolved_tenant,
            executions=(
                DiagnosticExecutionScope(
                    tenant_id=resolved_tenant,
                    task_id=validated_task_id,
                    run_id=validated_run_id,
                    problem_signals=problem_signals,
                ),
            ),
            grouping_strategy_id=STRATEGY_ID,
            observed_at=observed_at,
        )
        return self._orchestrator.run(request)


__all__ = ["TerminalExecutionDiagnosticTrigger"]
