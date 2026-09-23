# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Async Execution Engine adapter for worker qualified capabilities (UCA-6C-R6-R5.8-R2-H1)."""

from __future__ import annotations

from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionAsyncDispatchPort,
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionRequest,
    WorkerQualifiedCapabilityExecutionResult,
    derive_qualified_capability_execution_request_id,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    map_qualified_dispatch_result,
)


class WorkerQualifiedCapabilityExecutionEngineAsyncAdapter:
    """Async root launch boundary for worker orchestration already on the event loop."""

    def __init__(
        self,
        *,
        dispatch: QualifiedCapabilityExecutionAsyncDispatchPort,
    ) -> None:
        self._dispatch = dispatch

    async def execute_async(
        self,
        request: WorkerQualifiedCapabilityExecutionRequest,
    ) -> WorkerQualifiedCapabilityExecutionResult:
        expected_id = derive_qualified_capability_execution_request_id(
            resume_operation_id=request.resume_operation_id,
            binding_operation_id=request.binding_operation_id,
        )
        if request.execution_request_id != expected_id:
            return WorkerQualifiedCapabilityExecutionResult(
                disposition=WorkerQualifiedCapabilityExecutionDisposition.FAILED,
                execution_request_id=request.execution_request_id,
                reason_detail="execution_request_id_integrity_mismatch",
            )
        from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
            QualifiedCapabilityExecutionDispatchRequest,
        )

        dispatch_request = QualifiedCapabilityExecutionDispatchRequest(
            execution_request_id=request.execution_request_id,
            execution_target=request.execution_target,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            worker_instance_id=request.worker_instance_id,
            worker_need_id=request.worker_need_id,
            resume_operation_id=request.resume_operation_id,
            binding_operation_id=request.binding_operation_id,
            qualification_request_id=request.qualification_request_id,
            acquisition_request_id=request.acquisition_request_id,
            qualified_subject_reference=request.qualified_subject_reference,
            requested_at=request.requested_at,
            admitted_governance_identity=request.admitted_governance_identity,
            effective_authority_decision=request.effective_authority_decision,
            collaborative_authority_scopes=request.collaborative_authority_scopes,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
        )
        dispatch_result = await self._dispatch.dispatch_async(dispatch_request)
        if (
            dispatch_result.disposition
            is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
            and dispatch_result.execution_request_id is None
        ):
            return WorkerQualifiedCapabilityExecutionResult(
                disposition=WorkerQualifiedCapabilityExecutionDisposition.FAILED,
                reason_detail="execution_request_id_missing",
            )
        return map_qualified_dispatch_result(dispatch_result)


__all__ = ["WorkerQualifiedCapabilityExecutionEngineAsyncAdapter"]
