# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CONFIGURE_EXISTING post-adoption binding, intent, and ExecutionBound dispatch."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.autonomous_work.configured_capability_execution_subject_builder import (
    build_configured_capability_execution_subject,
)
from intergrax.autonomous_work.execution_authority_admission import (
    WorkerExecutionAdmissionPort,
    WorkerExecutionAuthorityDenied,
)
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    WorkerCapabilityAcquisitionDecision,
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
)
from intergrax.contracts.autonomous_work.worker_configured_capability_execution import (
    WorkerConfiguredCapabilityExecutionPort,
    WorkerConfiguredCapabilityExecutionRequest,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    derive_qualified_capability_execution_request_id,
)
from intergrax.contracts.capability_qualification.configured_capability_execution_subject import (
    derive_configured_capability_execution_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)
from intergrax.tools.marketplace_configured_capability_binding_provider import (
    MarketplaceConfiguredCapabilityBindingProvider,
    MarketplaceConfiguredCapabilityBindingRequest,
    derive_marketplace_configured_binding_operation_id,
)
from intergrax.tools.marketplace_configured_tool_execution_intent_preparation import (
    MarketplaceConfiguredToolExecutionIntentPreparation,
    MarketplaceConfiguredToolExecutionIntentPreparationOutcome,
    MarketplaceConfiguredToolExecutionIntentPreparationRequest,
)


@runtime_checkable
class WorkerConfiguredCapabilityExecutionFulfillmentPort(Protocol):
    """Narrow seam — coordinator consumes adoption and delegates here."""

    def fulfill_after_adoption(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        recovery: WorkerCapabilityRecoveryOutcome,
        decision: WorkerCapabilityAcquisitionDecision,
        adoption: ExecutionIntegrationConfigurationAdoption,
        decided_at: datetime,
    ) -> WorkerCapabilityFulfillmentResult: ...


class WorkerConfiguredCapabilityExecutionFulfillmentService:
    """Thin CONFIGURE_EXISTING execution orchestration — typed ports only."""

    def __init__(
        self,
        *,
        binding: MarketplaceConfiguredCapabilityBindingProvider,
        intent_preparation: MarketplaceConfiguredToolExecutionIntentPreparation,
        execution: WorkerConfiguredCapabilityExecutionPort,
        authority_admission: WorkerExecutionAdmissionPort | None = None,
    ) -> None:
        self._binding = binding
        self._intent_preparation = intent_preparation
        self._execution = execution
        self._authority_admission = authority_admission

    def fulfill_after_adoption(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        recovery: WorkerCapabilityRecoveryOutcome,
        decision: WorkerCapabilityAcquisitionDecision,
        adoption: ExecutionIntegrationConfigurationAdoption,
        decided_at: datetime,
    ) -> WorkerCapabilityFulfillmentResult:
        provenance = recovery.provenance
        if decision.disposition is not CapabilityAcquisitionDisposition.CONFIGURE_EXISTING:
            return _fail_closed(provenance, recovery, decided_at)
        need = request.acquisition_request.need
        worker_need_id = derive_worker_capability_need_id(need)
        subject = build_configured_capability_execution_subject(
            tenant_id=request.tenant_id,
            worker_need_id=worker_need_id,
            recovery_decision_id=need.recovery_decision_id,
            decision=decision,
            adoption=adoption,
            selected_operations=need.required_operations,
        )
        if subject is None:
            return _fail_closed(provenance, recovery, decided_at)

        configured_execution_operation_id = derive_configured_capability_execution_operation_id(
            recovery_decision_id=subject.recovery_decision_id,
            decision_id=subject.decision_id,
        )
        binding_operation_id = derive_marketplace_configured_binding_operation_id(
            subject=subject,
        )
        binding_result = self._binding.bind(
            MarketplaceConfiguredCapabilityBindingRequest(
                binding_operation_id=binding_operation_id,
                subject=subject,
                requested_at=decided_at,
            ),
        )
        if binding_result.outcome is not QualifiedCapabilityBindingOutcome.BOUND:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.BINDING_FAILED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        assert binding_result.execution_target is not None
        execution_request_id = derive_qualified_capability_execution_request_id(
            resume_operation_id=configured_execution_operation_id,
            binding_operation_id=binding_operation_id,
        )
        preparation = self._intent_preparation.prepare(
            MarketplaceConfiguredToolExecutionIntentPreparationRequest(
                subject=subject,
                execution_request_id=execution_request_id,
                binding_operation_id=binding_operation_id,
                tenant_id=request.tenant_id,
                task_id=request.task_id,
                worker_need_id=worker_need_id,
            ),
        )
        if preparation.outcome not in {
            MarketplaceConfiguredToolExecutionIntentPreparationOutcome.CREATED,
            MarketplaceConfiguredToolExecutionIntentPreparationOutcome.ALREADY_RECORDED_IDENTICAL,
        }:
            return _fail_closed(provenance, recovery, decided_at)

        if self._authority_admission is None:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_FAILED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
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
                decided_at=decided_at,
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
                decided_at=decided_at,
            )
        execution_request = WorkerConfiguredCapabilityExecutionRequest(
            configured_execution_operation_id=configured_execution_operation_id,
            binding_operation_id=binding_operation_id,
            execution_request_id=execution_request_id,
            execution_target=binding_result.execution_target,
            worker_instance_id=request.worker_instance_id,
            worker_need_id=worker_need_id,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            discovery_correlation_id=provenance.discovery_correlation_id,
            configured_subject_reference=subject.subject_reference,
            requested_at=decided_at,
            admitted_governance_identity=admitted_identity,
            effective_authority_decision=authority_context.effective_authority_decision,
            collaborative_authority_scopes=request.requested_authority_scopes,
            integration_configuration_adoption=adoption,
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
                execution_result=execution_result,
                decided_at=decided_at,
            )
        return WorkerCapabilityFulfillmentResult(
            disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_FAILED,
            provenance=provenance,
            recovery_outcome=recovery,
            execution_result=execution_result,
            decided_at=decided_at,
        )


def _fail_closed(provenance, recovery, decided_at: datetime) -> WorkerCapabilityFulfillmentResult:
    return WorkerCapabilityFulfillmentResult(
        disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
        provenance=provenance,
        recovery_outcome=recovery,
        decided_at=decided_at,
    )


__all__ = [
    "WorkerConfiguredCapabilityExecutionFulfillmentPort",
    "WorkerConfiguredCapabilityExecutionFulfillmentService",
]
