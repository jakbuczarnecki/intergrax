# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production WorkerQualifiedCapabilityExecutionPort adapter (UCA-6C-R)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionRequest,
    WorkerQualifiedCapabilityExecutionResult,
    derive_qualified_capability_execution_request_id,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchPort,
    QualifiedCapabilityExecutionDispatchRequest,
)


class WorkerQualifiedCapabilityExecutionEngineAdapter:
    """Translate worker resume execution requests into canonical Execution Engine dispatch."""

    def __init__(
        self,
        *,
        dispatch: QualifiedCapabilityExecutionDispatchPort,
    ) -> None:
        self._dispatch = dispatch

    def execute(
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
        if request.governance_approval_evidence is not None:
            if request.governance_approval_evidence.tenant_id != request.tenant_id:
                return WorkerQualifiedCapabilityExecutionResult(
                    disposition=WorkerQualifiedCapabilityExecutionDisposition.REJECTED,
                    execution_request_id=request.execution_request_id,
                    reason_detail="governance_approval_evidence_tenant_mismatch",
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
            governance_approval_evidence=request.governance_approval_evidence,
        )
        dispatch_result = self._dispatch.dispatch(dispatch_request)
        return _map_result(dispatch_result)


def _map_result(
    dispatch_result,
) -> WorkerQualifiedCapabilityExecutionResult:
    mapping = {
        QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED: (
            WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED
        ),
        QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE: (
            WorkerQualifiedCapabilityExecutionDisposition.UNAVAILABLE
        ),
        QualifiedCapabilityExecutionDispatchDisposition.FAILED: (
            WorkerQualifiedCapabilityExecutionDisposition.FAILED
        ),
        QualifiedCapabilityExecutionDispatchDisposition.REJECTED: (
            WorkerQualifiedCapabilityExecutionDisposition.REJECTED
        ),
    }
    disposition = mapping[dispatch_result.disposition]
    execution_request_id = dispatch_result.execution_request_id
    if (
        disposition is WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED
        and execution_request_id is None
    ):
        return WorkerQualifiedCapabilityExecutionResult(
            disposition=WorkerQualifiedCapabilityExecutionDisposition.FAILED,
            reason_detail="execution_request_id_missing",
        )
    return WorkerQualifiedCapabilityExecutionResult(
        disposition=disposition,
        execution_request_id=execution_request_id,
        run_id=dispatch_result.run_id,
        attempt_id=dispatch_result.attempt_id,
        execution_id=dispatch_result.execution_id,
        reason_detail=dispatch_result.reason_detail,
    )


__all__ = [
    "WorkerQualifiedCapabilityExecutionEngineAdapter",
]
