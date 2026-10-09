# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production WorkerConfiguredCapabilityExecutionPort adapter (TRACE-X-P5-R2-P3-R2)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.worker_configured_capability_execution import (
    WorkerConfiguredCapabilityExecutionRequest,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionResult,
    derive_qualified_capability_execution_request_id,
)
from intergrax.contracts.execution.execution_bound_capability_execution_dispatch import (
    ExecutionBoundCapabilityExecutionDispatchPort,
    ExecutionBoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.runtime.execution.worker_host_available_capability_execution_adapter import (
    _map_result,
)


class WorkerConfiguredCapabilityExecutionEngineAdapter:
    """Translate CONFIGURE_EXISTING execution requests into shared ExecutionBound ingress."""

    def __init__(
        self,
        *,
        dispatch: ExecutionBoundCapabilityExecutionDispatchPort,
    ) -> None:
        self._dispatch = dispatch

    def execute(
        self,
        request: WorkerConfiguredCapabilityExecutionRequest,
    ) -> WorkerQualifiedCapabilityExecutionResult:
        expected_id = derive_qualified_capability_execution_request_id(
            resume_operation_id=request.configured_execution_operation_id,
            binding_operation_id=request.binding_operation_id,
        )
        if request.execution_request_id != expected_id:
            return WorkerQualifiedCapabilityExecutionResult(
                disposition=WorkerQualifiedCapabilityExecutionDisposition.FAILED,
                execution_request_id=request.execution_request_id,
                reason_detail="execution_request_id_integrity_mismatch",
            )
        dispatch_request = ExecutionBoundCapabilityExecutionDispatchRequest(
            execution_request_id=request.execution_request_id,
            execution_target=request.execution_target,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            worker_instance_id=request.worker_instance_id,
            worker_need_id=request.worker_need_id,
            direct_reuse_operation_id=request.configured_execution_operation_id,
            binding_operation_id=request.binding_operation_id,
            discovery_correlation_id=request.discovery_correlation_id,
            host_subject_reference=request.configured_subject_reference,
            requested_at=request.requested_at,
            admitted_governance_identity=request.admitted_governance_identity,
            effective_authority_decision=request.effective_authority_decision,
            collaborative_authority_scopes=request.collaborative_authority_scopes,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            integration_configuration_adoption=request.integration_configuration_adoption,
        )
        dispatch_result = self._dispatch.dispatch(dispatch_request)
        return _map_result(dispatch_result)


__all__ = ["WorkerConfiguredCapabilityExecutionEngineAdapter"]
