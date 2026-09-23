# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production DIRECT_REUSE fulfillment — host-available binding then Execution Engine."""

from __future__ import annotations


from intergrax.autonomous_work.execution_authority_admission import (
    WorkerExecutionAdmissionService,
    WorkerExecutionAuthorityDenied,
)
from intergrax.contracts.autonomous_work.worker_host_available_capability_binding import (
    HostAvailableCapabilityBindingPort,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_ports import (
    WorkerQualifiedCapabilityExecutionPort,
)
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    derive_worker_capability_need_id,
)
from intergrax.contracts.autonomous_work.execution_authority import (
    WorkerExecutionAuthorityRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
    WorkerCapabilityFulfillmentRequest,
    WorkerCapabilityFulfillmentResult,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
)
from intergrax.contracts.autonomous_work.worker_host_available_capability_binding import (
    HostAvailableCapabilityBindingRequest,
    derive_host_available_capability_binding_operation_id,
    derive_worker_direct_reuse_operation_id,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionRequest,
    derive_qualified_capability_execution_request_id,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
)


class WorkerCapabilityDirectReuseFulfillmentService:
    """Canonical DIRECT_REUSE adapter — binding owner + EE dispatch only."""

    def __init__(
        self,
        *,
        binding: HostAvailableCapabilityBindingPort,
        execution: WorkerQualifiedCapabilityExecutionPort,
        authority_admission: WorkerExecutionAdmissionService | None = None,
    ) -> None:
        self._binding = binding
        self._execution = execution
        self._authority_admission = authority_admission

    def fulfill_direct_reuse(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        recovery: WorkerCapabilityRecoveryOutcome,
    ) -> WorkerCapabilityFulfillmentResult:
        timestamp = request.requested_at
        provenance = recovery.provenance
        if recovery.phase is not WorkerCapabilityRecoveryPhase.DIRECT_REUSE:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=timestamp,
            )
        completion = recovery.discovery_completion
        if completion is None or not completion.suitable_host_allowed_keys:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=timestamp,
            )
        host_key = sorted(
            completion.suitable_host_allowed_keys,
            key=lambda item: item.sort_key,
        )[0]
        need = request.acquisition_request.need
        worker_need_id = derive_worker_capability_need_id(need)
        direct_reuse_operation_id = derive_worker_direct_reuse_operation_id(
            recovery_decision_id=need.recovery_decision_id,
            discovery_correlation_id=provenance.discovery_correlation_id,
        )
        binding_operation_id = derive_host_available_capability_binding_operation_id(
            direct_reuse_operation_id=direct_reuse_operation_id,
            capability_identity=host_key,
        )
        binding_request = HostAvailableCapabilityBindingRequest(
            binding_operation_id=binding_operation_id,
            direct_reuse_operation_id=direct_reuse_operation_id,
            capability_identity=host_key,
            worker_need_id=worker_need_id,
            worker_instance_id=str(request.worker_instance_id),
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            discovery_correlation_id=provenance.discovery_correlation_id,
            correlation_id=provenance.discovery_correlation_id,
            causation_id=need.recovery_decision_id,
            requested_at=timestamp,
        )
        binding_result = self._binding.bind(binding_request)
        binding_outcome = binding_result.outcome
        if binding_outcome is not QualifiedCapabilityBindingOutcome.BOUND:
            disposition = WorkerCapabilityFulfillmentDisposition.BINDING_FAILED
            return WorkerCapabilityFulfillmentResult(
                disposition=disposition,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=timestamp,
            )
        assert binding_result.execution_target is not None
        assert binding_result.host_subject_reference is not None
        if self._authority_admission is None:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_FAILED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=timestamp,
            )
        try:
            authority_context = self._authority_admission.prepare(
                WorkerExecutionAuthorityRequest(
                    worker_instance_id=request.worker_instance_id,
                    requested_authority_scopes=request.requested_authority_scopes,
                ),
            )
        except WorkerExecutionAuthorityDenied:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_FAILED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=timestamp,
            )
        principal = authority_context.resolved_principal
        admitted_identity = AdmittedRootGovernanceIdentity(
            tenant_id=principal.tenant_id,
            workspace_id=principal.workspace_id,
            principal_id=principal.principal_id,
        )
        if request.tenant_id != admitted_identity.tenant_id:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_FAILED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=timestamp,
            )
        execution_request_id = derive_qualified_capability_execution_request_id(
            resume_operation_id=direct_reuse_operation_id,
            binding_operation_id=binding_operation_id,
        )
        discovery_provenance_id = (
            f"host-available:discovery:{provenance.discovery_correlation_id}"
        )
        worker_need_provenance_id = f"host-available:worker-need:{worker_need_id}"
        execution_request = WorkerQualifiedCapabilityExecutionRequest(
            resume_operation_id=direct_reuse_operation_id,
            binding_operation_id=binding_operation_id,
            execution_request_id=execution_request_id,
            execution_target=binding_result.execution_target,
            worker_instance_id=request.worker_instance_id,
            worker_need_id=worker_need_id,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            qualification_request_id=discovery_provenance_id,
            acquisition_request_id=worker_need_provenance_id,
            qualified_subject_reference=binding_result.host_subject_reference,
            requested_at=timestamp,
            admitted_governance_identity=admitted_identity,
            effective_authority_decision=authority_context.effective_authority_decision,
            collaborative_authority_scopes=request.requested_authority_scopes,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
        )
        execution_result = self._execution.execute(execution_request)
        if (
            execution_result.disposition
            is WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED
        ):
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=timestamp,
            )
        return WorkerCapabilityFulfillmentResult(
            disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_FAILED,
            provenance=provenance,
            recovery_outcome=recovery,
            decided_at=timestamp,
        )


__all__ = ["WorkerCapabilityDirectReuseFulfillmentService"]
