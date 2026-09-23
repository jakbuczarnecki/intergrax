# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Post-qualification worker capability resume — binding then Execution Engine (UCA-6C)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.autonomous_work.execution_authority_admission import (
    WorkerExecutionAdmissionPort,
    WorkerExecutionAuthorityDenied,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_ports import (
    QualifiedCapabilityBindingPort,
    WorkerQualifiedCapabilityAsyncExecutionPort,
    WorkerQualifiedCapabilityExecutionPort,
)
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.autonomous_work.execution_authority import (
    WorkerExecutionAuthorityRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionRequest,
    WorkerQualifiedCapabilityExecutionResult,
    WorkerQualifiedCapabilityResumeOutcome,
    WorkerQualifiedCapabilityResumeRequest,
    WorkerQualifiedCapabilityResumeResult,
    derive_qualified_capability_execution_request_id,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingRequest,
    QualifiedCapabilityBindingResult,
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    qualified_capability_subject_from_result,
)
@dataclass(frozen=True, slots=True)
class _QualifiedExecutionHandoff:
    resume_id: str
    execution_request_id: str
    execution_request: WorkerQualifiedCapabilityExecutionRequest
    provenance: WorkerCapabilityRecoveryProvenance
    binding_result: QualifiedCapabilityBindingResult
    timestamp: datetime


class WorkerQualifiedCapabilityResumeCoordinator:
    """Resume qualified capabilities without discovery, acquisition, or re-qualification."""

    def __init__(
        self,
        *,
        binding: QualifiedCapabilityBindingPort,
        execution: WorkerQualifiedCapabilityExecutionPort,
        authority_admission: WorkerExecutionAdmissionPort | None = None,
        async_execution: WorkerQualifiedCapabilityAsyncExecutionPort | None = None,
    ) -> None:
        self._binding = binding
        self._execution = execution
        self._authority_admission = authority_admission
        self._async_execution = async_execution

    def resume(
        self,
        request: WorkerQualifiedCapabilityResumeRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerQualifiedCapabilityResumeResult:
        handoff = self._prepare_execution_handoff(request, decided_at=decided_at)
        if isinstance(handoff, WorkerQualifiedCapabilityResumeResult):
            return handoff
        execution_result = self._execution.execute(handoff.execution_request)
        return self._map_execution_result(handoff, execution_result)

    async def resume_async(
        self,
        request: WorkerQualifiedCapabilityResumeRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerQualifiedCapabilityResumeResult:
        if self._async_execution is None:
            raise RuntimeError("async_execution adapter required for resume_async")
        handoff = self._prepare_execution_handoff(request, decided_at=decided_at)
        if isinstance(handoff, WorkerQualifiedCapabilityResumeResult):
            return handoff
        execution_result = await self._async_execution.execute_async(
            handoff.execution_request,
        )
        return self._map_execution_result(handoff, execution_result)

    def _prepare_execution_handoff(
        self,
        request: WorkerQualifiedCapabilityResumeRequest,
        *,
        decided_at: datetime | None,
    ) -> WorkerQualifiedCapabilityResumeResult | _QualifiedExecutionHandoff:
        timestamp = decided_at or request.requested_at
        qualification = request.qualification_result
        resume_id = request.resume_operation_id

        if qualification.outcome is CapabilityQualificationOutcome.REQUIRES_HITL:
            return _result(
                outcome=WorkerQualifiedCapabilityResumeOutcome.QUALIFICATION_NOT_ELIGIBLE,
                request=request,
                provenance=request.provenance,
                decided_at=timestamp,
            )
        if qualification.outcome is not CapabilityQualificationOutcome.QUALIFIED:
            return _result(
                outcome=WorkerQualifiedCapabilityResumeOutcome.QUALIFICATION_NOT_ELIGIBLE,
                request=request,
                provenance=request.provenance,
                decided_at=timestamp,
            )

        subject = qualified_capability_subject_from_result(qualification)
        if subject is None:
            return _result(
                outcome=WorkerQualifiedCapabilityResumeOutcome.QUALIFICATION_NOT_ELIGIBLE,
                request=request,
                provenance=request.provenance,
                decided_at=timestamp,
            )

        binding_operation_id = derive_qualified_capability_binding_operation_id(
            resume_operation_id=resume_id,
            qualified_subject_reference=subject.qualified_subject_reference,
        )
        binding_request = QualifiedCapabilityBindingRequest(
            binding_operation_id=binding_operation_id,
            resume_operation_id=resume_id,
            qualified_subject=subject,
            qualification_result=qualification,
            worker_need_id=request.worker_need_id,
            worker_instance_id=str(request.worker_instance_id),
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            correlation_id=qualification.correlation_id,
            causation_id=qualification.causation_id,
            requested_at=timestamp,
        )
        binding_result = self._binding.bind(binding_request)
        provenance = _provenance_after_binding(
            request.provenance,
            qualified_subject_reference=subject.qualified_subject_reference,
            binding_operation_id=binding_operation_id,
        )

        binding_outcome = binding_result.outcome
        if binding_outcome is QualifiedCapabilityBindingOutcome.REQUIRES_HITL:
            return WorkerQualifiedCapabilityResumeResult(
                outcome=WorkerQualifiedCapabilityResumeOutcome.BINDING_HITL,
                resume_operation_id=resume_id,
                provenance=provenance,
                binding_result=binding_result,
                decided_at=timestamp,
            )
        if binding_outcome is QualifiedCapabilityBindingOutcome.BLOCKED:
            return WorkerQualifiedCapabilityResumeResult(
                outcome=WorkerQualifiedCapabilityResumeOutcome.BINDING_BLOCKED,
                resume_operation_id=resume_id,
                provenance=provenance,
                binding_result=binding_result,
                decided_at=timestamp,
            )
        if binding_outcome in {
            QualifiedCapabilityBindingOutcome.UNAVAILABLE,
            QualifiedCapabilityBindingOutcome.NO_PROVIDER,
            QualifiedCapabilityBindingOutcome.NOT_SUPPORTED,
        }:
            return WorkerQualifiedCapabilityResumeResult(
                outcome=WorkerQualifiedCapabilityResumeOutcome.BINDING_UNAVAILABLE,
                resume_operation_id=resume_id,
                provenance=provenance,
                binding_result=binding_result,
                decided_at=timestamp,
            )
        if binding_outcome is not QualifiedCapabilityBindingOutcome.BOUND:
            return WorkerQualifiedCapabilityResumeResult(
                outcome=WorkerQualifiedCapabilityResumeOutcome.BINDING_FAILED,
                resume_operation_id=resume_id,
                provenance=provenance,
                binding_result=binding_result,
                decided_at=timestamp,
            )

        assert binding_result.execution_target is not None
        if self._authority_admission is None:
            return WorkerQualifiedCapabilityResumeResult(
                outcome=WorkerQualifiedCapabilityResumeOutcome.EXECUTION_UNAVAILABLE,
                resume_operation_id=resume_id,
                provenance=provenance,
                binding_result=binding_result,
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
            return WorkerQualifiedCapabilityResumeResult(
                outcome=WorkerQualifiedCapabilityResumeOutcome.EXECUTION_REJECTED,
                resume_operation_id=resume_id,
                provenance=provenance,
                binding_result=binding_result,
                decided_at=timestamp,
            )
        principal = authority_context.resolved_principal
        admitted_identity = AdmittedRootGovernanceIdentity(
            tenant_id=principal.tenant_id,
            workspace_id=principal.workspace_id,
            principal_id=principal.principal_id,
        )
        if request.tenant_id != admitted_identity.tenant_id:
            return WorkerQualifiedCapabilityResumeResult(
                outcome=WorkerQualifiedCapabilityResumeOutcome.EXECUTION_REJECTED,
                resume_operation_id=resume_id,
                provenance=provenance,
                binding_result=binding_result,
                decided_at=timestamp,
            )
        execution_request_id = derive_qualified_capability_execution_request_id(
            resume_operation_id=resume_id,
            binding_operation_id=binding_operation_id,
        )
        execution_request = WorkerQualifiedCapabilityExecutionRequest(
            resume_operation_id=resume_id,
            binding_operation_id=binding_operation_id,
            execution_request_id=execution_request_id,
            execution_target=binding_result.execution_target,
            worker_instance_id=request.worker_instance_id,
            worker_need_id=request.worker_need_id,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            qualification_request_id=qualification.qualification_request_id,
            acquisition_request_id=qualification.acquisition_request_id,
            qualified_subject_reference=subject.qualified_subject_reference,
            requested_at=timestamp,
            admitted_governance_identity=admitted_identity,
            effective_authority_decision=authority_context.effective_authority_decision,
            collaborative_authority_scopes=authority_context.collaborative_authority_scopes,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
        )
        return _QualifiedExecutionHandoff(
            resume_id=resume_id,
            execution_request_id=execution_request_id,
            execution_request=execution_request,
            provenance=provenance,
            binding_result=binding_result,
            timestamp=timestamp,
        )

    def _map_execution_result(
        self,
        handoff: _QualifiedExecutionHandoff,
        execution_result: WorkerQualifiedCapabilityExecutionResult,
    ) -> WorkerQualifiedCapabilityResumeResult:
        resume_id = handoff.resume_id
        provenance = handoff.provenance
        binding_result = handoff.binding_result
        timestamp = handoff.timestamp
        execution_request_id = handoff.execution_request_id
        if (
            execution_result.disposition
            is WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED
        ):
            if execution_result.execution_request_id != execution_request_id:
                return WorkerQualifiedCapabilityResumeResult(
                    outcome=WorkerQualifiedCapabilityResumeOutcome.EXECUTION_FAILED,
                    resume_operation_id=resume_id,
                    provenance=provenance,
                    binding_result=binding_result,
                    execution_result=execution_result,
                    decided_at=timestamp,
                )
            verified_id = execution_result.execution_request_id
            assert verified_id is not None
            provenance = _provenance_after_execution(
                provenance,
                execution_request_id=verified_id,
            )
            return WorkerQualifiedCapabilityResumeResult(
                outcome=WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED,
                resume_operation_id=resume_id,
                provenance=provenance,
                binding_result=binding_result,
                execution_result=execution_result,
                decided_at=timestamp,
            )
        outcome_map = {
            WorkerQualifiedCapabilityExecutionDisposition.UNAVAILABLE: (
                WorkerQualifiedCapabilityResumeOutcome.EXECUTION_UNAVAILABLE
            ),
            WorkerQualifiedCapabilityExecutionDisposition.FAILED: (
                WorkerQualifiedCapabilityResumeOutcome.EXECUTION_FAILED
            ),
            WorkerQualifiedCapabilityExecutionDisposition.REJECTED: (
                WorkerQualifiedCapabilityResumeOutcome.EXECUTION_REJECTED
            ),
        }
        return WorkerQualifiedCapabilityResumeResult(
            outcome=outcome_map[execution_result.disposition],
            resume_operation_id=resume_id,
            provenance=provenance,
            binding_result=binding_result,
            execution_result=execution_result,
            decided_at=timestamp,
        )


def _result(
    *,
    outcome: WorkerQualifiedCapabilityResumeOutcome,
    request: WorkerQualifiedCapabilityResumeRequest,
    provenance: WorkerCapabilityRecoveryProvenance,
    decided_at: datetime,
) -> WorkerQualifiedCapabilityResumeResult:
    return WorkerQualifiedCapabilityResumeResult(
        outcome=outcome,
        resume_operation_id=request.resume_operation_id,
        provenance=provenance,
        decided_at=decided_at,
    )


def _provenance_after_binding(
    provenance: WorkerCapabilityRecoveryProvenance,
    *,
    qualified_subject_reference: str,
    binding_operation_id: str,
) -> WorkerCapabilityRecoveryProvenance:
    return WorkerCapabilityRecoveryProvenance(
        worker_need_id=provenance.worker_need_id,
        canonical_need_id=provenance.canonical_need_id,
        discovery_correlation_id=provenance.discovery_correlation_id,
        discovery_completion_outcome=provenance.discovery_completion_outcome,
        gap_id=provenance.gap_id,
        acquisition_request_id=provenance.acquisition_request_id,
        acquisition_strategy_id=provenance.acquisition_strategy_id,
        qualification_request_id=provenance.qualification_request_id,
        qualified_subject_reference=qualified_subject_reference,
        binding_operation_id=binding_operation_id,
        evidence_refs=provenance.evidence_refs,
    )


def _provenance_after_execution(
    provenance: WorkerCapabilityRecoveryProvenance,
    *,
    execution_request_id: str,
) -> WorkerCapabilityRecoveryProvenance:
    return WorkerCapabilityRecoveryProvenance(
        worker_need_id=provenance.worker_need_id,
        canonical_need_id=provenance.canonical_need_id,
        discovery_correlation_id=provenance.discovery_correlation_id,
        discovery_completion_outcome=provenance.discovery_completion_outcome,
        gap_id=provenance.gap_id,
        acquisition_request_id=provenance.acquisition_request_id,
        acquisition_strategy_id=provenance.acquisition_strategy_id,
        qualification_request_id=provenance.qualification_request_id,
        qualified_subject_reference=provenance.qualified_subject_reference,
        binding_operation_id=provenance.binding_operation_id,
        execution_request_id=execution_request_id,
        evidence_refs=provenance.evidence_refs,
    )


__all__ = [
    "WorkerQualifiedCapabilityResumeCoordinator",
    "derive_qualified_capability_execution_request_id",
]
