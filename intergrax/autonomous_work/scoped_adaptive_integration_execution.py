# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AW-7C-P4 qualification + canonical execution composition."""

from __future__ import annotations

from intergrax.autonomous_work.worker_execution_dispatch import WorkerExecutionDispatchService
from intergrax.capability_qualification.qualification_service import (
    CapabilityQualificationService,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionDispatchDisposition,
    WorkerExecutionDispatchRequest,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
    ScopedAdaptiveIntegrationExecutionOutcome,
    ScopedAdaptiveIntegrationExecutionOutput,
    ScopedAdaptiveIntegrationExecutionRequest,
    ScopedAdaptiveIntegrationExecutionResult,
    build_scoped_adaptive_integration_execution_handoff,
    validate_preparation_artifact_subject_continuity,
    validate_qualification_decision_continuity,
)
from intergrax.contracts.capability_qualification.lifecycle_decision import (
    CapabilityQualificationLifecycleOutcome,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.execution_request import ExecutionRequest

class WorkerScopedAdaptiveIntegrationExecutionCoordinator:
    """Phase-2 A2 orchestration after QUALIFICATION_PENDING — no authority minting."""

    def __init__(
        self,
        *,
        qualification_service: CapabilityQualificationService,
        dispatch_service: WorkerExecutionDispatchService[
            ScopedAdaptiveIntegrationExecutionHandoff,
            ScopedAdaptiveIntegrationExecutionOutput,
        ],
    ) -> None:
        self._qualification_service = qualification_service
        self._dispatch_service = dispatch_service
        self._seen_idempotency_keys: set[str] = set()

    async def execute(
        self,
        request: ScopedAdaptiveIntegrationExecutionRequest,
    ) -> ScopedAdaptiveIntegrationExecutionResult:
        preparation = request.preparation
        worker_id = preparation.worker_instance_id
        idem = request.execution_idempotency_key
        tenant_id = request.tenant_id

        if idem in self._seen_idempotency_keys:
            return ScopedAdaptiveIntegrationExecutionResult(
                outcome=ScopedAdaptiveIntegrationExecutionOutcome.DUPLICATE_INVOCATION,
                worker_instance_id=worker_id,
                execution_idempotency_key=idem,
                tenant_id=tenant_id,
                error_detail="duplicate execution idempotency key",
            )

        qual_request = preparation.qualification_request
        if qual_request is None:
            return _reject(
                worker_id,
                idem,
                tenant_id,
                ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED,
                "missing qualification_request",
            )

        continuity = validate_preparation_artifact_subject_continuity(
            preparation=preparation,
            qualification_request=qual_request,
        )
        if continuity is not None:
            return _reject(
                worker_id,
                idem,
                tenant_id,
                ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED,
                continuity,
            )

        if preparation.artifact is not None and preparation.artifact.tenant_id != tenant_id:
            return _reject(
                worker_id,
                idem,
                tenant_id,
                ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED,
                "preparation tenant mismatch",
            )

        decision = self._qualification_service.qualify(qual_request)
        artifact = preparation.artifact
        subject = preparation.qualification_subject
        if artifact is None or subject is None:
            return _reject(
                worker_id,
                idem,
                tenant_id,
                ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED,
                "missing artifact or subject after qualify",
            )

        decision_error = validate_qualification_decision_continuity(
            qualification_request=qual_request,
            decision=decision,
            artifact=artifact,
            subject=subject,
        )
        if decision_error is not None:
            return _reject(
                worker_id,
                idem,
                tenant_id,
                ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED,
                decision_error,
            )

        result = decision.qualification_result
        lifecycle = decision.lifecycle_decision
        if (
            result.outcome is not CapabilityQualificationOutcome.QUALIFIED
            or lifecycle.outcome is not CapabilityQualificationLifecycleOutcome.ACCEPT
        ):
            return _reject(
                worker_id,
                idem,
                tenant_id,
                ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED,
                f"qualification not accepted: {result.outcome}/{lifecycle.outcome}",
            )

        handoff = build_scoped_adaptive_integration_execution_handoff(
            preparation=preparation,
            qualification_request=qual_request,
            execution_idempotency_key=idem,
        )

        dispatch_ctx = request.worker_dispatch
        runtime_request = ExecutionRequest(
            input=handoff,
            output_type=ScopedAdaptiveIntegrationExecutionOutput,
        )
        dispatch_request = WorkerExecutionDispatchRequest(
            worker_instance_id=worker_id,
            worker_revision=dispatch_ctx.worker_revision,
            requested_scopes=dispatch_ctx.requested_scopes,
            runtime_request=runtime_request,
            source=dispatch_ctx.source,
            requested_at=request.requested_at,
        )

        self._seen_idempotency_keys.add(idem)
        dispatch_result = await self._dispatch_service.dispatch(dispatch_request)

        if dispatch_result.disposition is WorkerExecutionDispatchDisposition.UNAVAILABLE:
            return ScopedAdaptiveIntegrationExecutionResult(
                outcome=ScopedAdaptiveIntegrationExecutionOutcome.GOVERNANCE_UNAVAILABLE,
                worker_instance_id=worker_id,
                execution_idempotency_key=idem,
                tenant_id=tenant_id,
                handoff=handoff,
                error_detail="root execution unavailable",
            )
        if dispatch_result.disposition is WorkerExecutionDispatchDisposition.REJECTED:
            return ScopedAdaptiveIntegrationExecutionResult(
                outcome=ScopedAdaptiveIntegrationExecutionOutcome.GOVERNANCE_DENIED,
                worker_instance_id=worker_id,
                execution_idempotency_key=idem,
                tenant_id=tenant_id,
                handoff=handoff,
                error_detail=str(dispatch_result.rejection_reason),
            )
        if dispatch_result.disposition is WorkerExecutionDispatchDisposition.FAILED:
            correlation = dispatch_result.correlation
            return ScopedAdaptiveIntegrationExecutionResult(
                outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
                worker_instance_id=worker_id,
                execution_idempotency_key=idem,
                tenant_id=tenant_id,
                handoff=handoff,
                run_id=correlation.run_id,
                attempt_id=correlation.attempt_id,
                execution_id=correlation.execution_id,
                error_detail=dispatch_result.failure_reason or "execution failed",
            )
        if dispatch_result.disposition is not WorkerExecutionDispatchDisposition.DISPATCHED:
            return ScopedAdaptiveIntegrationExecutionResult(
                outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
                worker_instance_id=worker_id,
                execution_idempotency_key=idem,
                tenant_id=tenant_id,
                handoff=handoff,
                error_detail="unexpected dispatch disposition",
            )

        correlation = dispatch_result.correlation
        operation_output = dispatch_result.runtime_result
        if operation_output is None:
            return ScopedAdaptiveIntegrationExecutionResult(
                outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
                worker_instance_id=worker_id,
                execution_idempotency_key=idem,
                tenant_id=tenant_id,
                handoff=handoff,
                run_id=correlation.run_id,
                attempt_id=correlation.attempt_id,
                execution_id=correlation.execution_id,
                error_detail="missing runtime result",
            )
        if operation_output.tenant_id != tenant_id:
            return ScopedAdaptiveIntegrationExecutionResult(
                outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
                worker_instance_id=worker_id,
                execution_idempotency_key=idem,
                tenant_id=tenant_id,
                handoff=handoff,
                run_id=correlation.run_id,
                attempt_id=correlation.attempt_id,
                execution_id=correlation.execution_id,
                error_detail="runtime result tenant mismatch",
            )

        return ScopedAdaptiveIntegrationExecutionResult(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED,
            worker_instance_id=worker_id,
            execution_idempotency_key=idem,
            tenant_id=tenant_id,
            handoff=handoff,
            run_id=correlation.run_id,
            attempt_id=correlation.attempt_id,
            execution_id=correlation.execution_id,
            operation_output=operation_output,
        )


def _reject(
    worker_id: WorkerInstanceId,
    idem: str,
    tenant_id: str,
    outcome: ScopedAdaptiveIntegrationExecutionOutcome,
    detail: str,
) -> ScopedAdaptiveIntegrationExecutionResult:
    return ScopedAdaptiveIntegrationExecutionResult(
        outcome=outcome,
        worker_instance_id=worker_id,
        execution_idempotency_key=idem,
        tenant_id=tenant_id,
        error_detail=detail,
    )


__all__ = ["WorkerScopedAdaptiveIntegrationExecutionCoordinator"]
