# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker consumer fulfillment — sequences canonical recovery, realization, resume (UCA-6C-R6-R5.8)."""

from __future__ import annotations

from datetime import datetime

from intergrax.autonomous_work.worker_capability_fulfillment_ports import (
    CapabilityRealizationCoordinatorPort,
    WorkerCapabilityDirectReuseFulfillmentPort,
    WorkerCapabilityRecoveryPort,
    WorkerQualifiedCapabilityResumePort,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityNeed,
    derive_worker_capability_need_id,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
    WorkerCapabilityFulfillmentRequest,
    WorkerCapabilityFulfillmentResult,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryPhase,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityResumeOutcome,
    WorkerQualifiedCapabilityResumeRequest,
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
)
from intergrax.contracts.capability_acquisition.outcome import (
    CapabilityRealizationOutcome,
)
from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
    derive_capability_realization_request_id,
)
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.capability_realization_need import (
    CapabilityRealizationNeed,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    qualified_capability_subject_from_result,
)
from intergrax.contracts.tools.qualified_capability_execution_intent_preparation import (
    QualifiedCapabilityExecutionIntentPreparationOutcome,
    QualifiedCapabilityExecutionIntentPreparationPort,
    QualifiedCapabilityExecutionIntentPreparationRequest,
)


class WorkerCapabilityFulfillmentCoordinator:
    """Requester/orchestrator — routes to canonical discovery, UCA, qualification, execution."""

    def __init__(
        self,
        *,
        recovery: WorkerCapabilityRecoveryPort,
        resume: WorkerQualifiedCapabilityResumePort,
        direct_reuse: WorkerCapabilityDirectReuseFulfillmentPort,
        realization: CapabilityRealizationCoordinatorPort | None = None,
        intent_preparation: QualifiedCapabilityExecutionIntentPreparationPort | None = None,
    ) -> None:
        self._recovery = recovery
        self._resume = resume
        self._direct_reuse = direct_reuse
        self._realization = realization
        self._intent_preparation = intent_preparation

    def fulfill(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerCapabilityFulfillmentResult:
        timestamp = decided_at or request.requested_at
        recovery = self._recovery.coordinate_recovery(
            request.acquisition_request,
            decided_at=timestamp,
            allow_generic_acquisition=request.allow_generic_acquisition,
        )
        return self._fulfill_from_recovery(
            request,
            recovery=recovery,
            decided_at=timestamp,
            after_realization=False,
        )

    async def fulfill_async(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerCapabilityFulfillmentResult:
        timestamp = decided_at or request.requested_at
        recovery = self._recovery.coordinate_recovery(
            request.acquisition_request,
            decided_at=timestamp,
            allow_generic_acquisition=request.allow_generic_acquisition,
        )
        return await self._fulfill_from_recovery_async(
            request,
            recovery=recovery,
            decided_at=timestamp,
            after_realization=False,
        )

    async def _fulfill_from_recovery_async(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        recovery,
        decided_at: datetime,
        after_realization: bool,
    ) -> WorkerCapabilityFulfillmentResult:
        if recovery.phase is WorkerCapabilityRecoveryPhase.QUALIFICATION_COMPLETE:
            if after_realization:
                return WorkerCapabilityFulfillmentResult(
                    disposition=WorkerCapabilityFulfillmentDisposition.REALIZATION_NOT_VISIBLE,
                    provenance=recovery.provenance,
                    recovery_outcome=recovery,
                    decided_at=decided_at,
                )
            return await self._fulfill_qualified_async(
                request,
                recovery=recovery,
                decided_at=decided_at,
            )
        return self._fulfill_from_recovery(
            request,
            recovery=recovery,
            decided_at=decided_at,
            after_realization=after_realization,
        )

    def _fulfill_from_recovery(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        recovery,
        decided_at: datetime,
        after_realization: bool,
    ) -> WorkerCapabilityFulfillmentResult:
        provenance = recovery.provenance

        if recovery.phase is WorkerCapabilityRecoveryPhase.DIRECT_REUSE:
            return self._direct_reuse.fulfill_direct_reuse(request, recovery)

        if recovery.phase is WorkerCapabilityRecoveryPhase.REALIZATION_REQUIRED:
            if after_realization:
                return WorkerCapabilityFulfillmentResult(
                    disposition=WorkerCapabilityFulfillmentDisposition.REALIZATION_NOT_VISIBLE,
                    provenance=provenance,
                    recovery_outcome=recovery,
                    decided_at=decided_at,
                )
            return self._fulfill_realization_required(
                request,
                recovery=recovery,
                decided_at=decided_at,
            )

        if recovery.phase is WorkerCapabilityRecoveryPhase.QUALIFICATION_COMPLETE:
            if after_realization:
                return WorkerCapabilityFulfillmentResult(
                    disposition=WorkerCapabilityFulfillmentDisposition.REALIZATION_NOT_VISIBLE,
                    provenance=provenance,
                    recovery_outcome=recovery,
                    decided_at=decided_at,
                )
            return self._fulfill_qualified(
                request, recovery=recovery, decided_at=decided_at
            )

        if recovery.phase is WorkerCapabilityRecoveryPhase.FAIL_CLOSED:
            if after_realization:
                return WorkerCapabilityFulfillmentResult(
                    disposition=WorkerCapabilityFulfillmentDisposition.REALIZATION_NOT_VISIBLE,
                    provenance=provenance,
                    recovery_outcome=recovery,
                    decided_at=decided_at,
                )
            return self._map_fail_closed(
                request,
                recovery=recovery,
                decided_at=decided_at,
            )

        return WorkerCapabilityFulfillmentResult(
            disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
            provenance=provenance,
            recovery_outcome=recovery,
            decided_at=decided_at,
        )

    def _fulfill_realization_required(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        recovery,
        decided_at: datetime,
    ) -> WorkerCapabilityFulfillmentResult:
        if self._realization is None:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
                provenance=recovery.provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        completion = recovery.discovery_completion
        if completion is None or not completion.suitable_catalog_allowed_keys:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
                provenance=recovery.provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        catalog_key = sorted(
            completion.suitable_catalog_allowed_keys,
            key=lambda item: item.sort_key,
        )[0]
        realization_need = CapabilityRealizationNeed.from_discovery_completion(
            completion,
            capability_identity=catalog_key,
        )
        need = request.acquisition_request.need
        nonce = f"{need.recovery_decision_id}:realize"
        realization_request = CapabilityRealizationRequest(
            request_id=derive_capability_realization_request_id(
                realization_need_id=realization_need.realization_need_id,
                request_nonce=nonce,
            ),
            request_nonce=nonce,
            realization_need=realization_need,
            correlation_id=recovery.provenance.discovery_correlation_id,
            causation_id=need.recovery_decision_id,
            requested_at=decided_at,
        )
        realization_result = self._realization.realize(realization_request)
        if realization_result.outcome is not CapabilityRealizationOutcome.SUCCEEDED:
            disposition = WorkerCapabilityFulfillmentDisposition.REALIZATION_FAILED
            if realization_result.outcome in {
                CapabilityRealizationOutcome.BLOCKED,
                CapabilityRealizationOutcome.REQUIRES_HITL,
            }:
                disposition = WorkerCapabilityFulfillmentDisposition.DISCOVERY_BLOCKED
            if realization_result.outcome is CapabilityRealizationOutcome.UNAVAILABLE:
                disposition = (
                    WorkerCapabilityFulfillmentDisposition.DISCOVERY_UNAVAILABLE
                )
            return WorkerCapabilityFulfillmentResult(
                disposition=disposition,
                provenance=recovery.provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        reconciled = self._recovery.coordinate_recovery(
            request.acquisition_request,
            decided_at=decided_at,
            allow_generic_acquisition=False,
        )
        return self._fulfill_from_recovery(
            request,
            recovery=reconciled,
            decided_at=decided_at,
            after_realization=True,
        )

    def _fulfill_qualified(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        recovery,
        decided_at: datetime,
    ) -> WorkerCapabilityFulfillmentResult:
        acquisition = recovery.acquisition_result
        qualification = recovery.qualification_result
        if acquisition is None or qualification is None:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
                provenance=recovery.provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        if qualification.outcome is not CapabilityQualificationOutcome.QUALIFIED:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.QUALIFICATION_FAILED,
                provenance=recovery.provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        need = request.acquisition_request.need
        worker_need_id = derive_worker_capability_need_id(need)
        resume_operation_id = derive_worker_capability_resume_operation_id(
            recovery_decision_id=need.recovery_decision_id,
            qualification_request_id=qualification.qualification_request_id,
        )
        preparation_block = self._prepare_execution_intent_if_configured(
            request=request,
            need=need,
            qualification=qualification,
            worker_need_id=worker_need_id,
            resume_operation_id=resume_operation_id,
            recovery=recovery,
            decided_at=decided_at,
        )
        if preparation_block is not None:
            return preparation_block
        resume_request = WorkerQualifiedCapabilityResumeRequest(
            worker_instance_id=request.worker_instance_id,
            worker_need_id=worker_need_id,
            recovery_decision_id=need.recovery_decision_id,
            provenance=recovery.provenance,
            acquisition_result=acquisition,
            qualification_result=qualification,
            resume_operation_id=resume_operation_id,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            requested_at=decided_at,
            requested_authority_scopes=request.requested_authority_scopes,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
        )
        resume_result = self._resume.resume(resume_request, decided_at=decided_at)
        return self._map_resume(
            recovery=recovery,
            resume_result=resume_result,
            decided_at=decided_at,
        )

    async def _fulfill_qualified_async(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        recovery,
        decided_at: datetime,
    ) -> WorkerCapabilityFulfillmentResult:
        acquisition = recovery.acquisition_result
        qualification = recovery.qualification_result
        if acquisition is None or qualification is None:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
                provenance=recovery.provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        if qualification.outcome is not CapabilityQualificationOutcome.QUALIFIED:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.QUALIFICATION_FAILED,
                provenance=recovery.provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        need = request.acquisition_request.need
        worker_need_id = derive_worker_capability_need_id(need)
        resume_operation_id = derive_worker_capability_resume_operation_id(
            recovery_decision_id=need.recovery_decision_id,
            qualification_request_id=qualification.qualification_request_id,
        )
        preparation_block = self._prepare_execution_intent_if_configured(
            request=request,
            need=need,
            qualification=qualification,
            worker_need_id=worker_need_id,
            resume_operation_id=resume_operation_id,
            recovery=recovery,
            decided_at=decided_at,
        )
        if preparation_block is not None:
            return preparation_block
        resume_request = WorkerQualifiedCapabilityResumeRequest(
            worker_instance_id=request.worker_instance_id,
            worker_need_id=worker_need_id,
            recovery_decision_id=need.recovery_decision_id,
            provenance=recovery.provenance,
            acquisition_result=acquisition,
            qualification_result=qualification,
            resume_operation_id=resume_operation_id,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            requested_at=decided_at,
            requested_authority_scopes=request.requested_authority_scopes,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
        )
        resume_result = await self._resume.resume_async(
            resume_request,
            decided_at=decided_at,
        )
        return self._map_resume(
            recovery=recovery,
            resume_result=resume_result,
            decided_at=decided_at,
        )

    def _prepare_execution_intent_if_configured(
        self,
        *,
        request: WorkerCapabilityFulfillmentRequest,
        need: WorkerCapabilityNeed,
        qualification: CapabilityQualificationResult,
        worker_need_id: str,
        resume_operation_id: str,
        recovery,
        decided_at: datetime,
    ) -> WorkerCapabilityFulfillmentResult | None:
        if self._intent_preparation is None:
            return None

        subject = qualified_capability_subject_from_result(qualification)
        if subject is None:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
                provenance=recovery.provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )

        binding_operation_id = derive_qualified_capability_binding_operation_id(
            resume_operation_id=resume_operation_id,
            qualified_subject_reference=subject.qualified_subject_reference,
        )
        execution_request_id = derive_qualified_capability_execution_request_id(
            resume_operation_id=resume_operation_id,
            binding_operation_id=binding_operation_id,
        )
        preparation = self._intent_preparation.prepare(
            QualifiedCapabilityExecutionIntentPreparationRequest(
                need=need,
                qualification_result=qualification,
                execution_request_id=execution_request_id,
                binding_operation_id=binding_operation_id,
                resume_operation_id=resume_operation_id,
                worker_need_id=worker_need_id,
                tenant_id=request.tenant_id,
                task_id=request.task_id,
            ),
        )
        outcome = preparation.outcome
        if outcome in {
            QualifiedCapabilityExecutionIntentPreparationOutcome.NOT_APPLICABLE,
            QualifiedCapabilityExecutionIntentPreparationOutcome.CREATED,
            QualifiedCapabilityExecutionIntentPreparationOutcome.ALREADY_RECORDED_IDENTICAL,
        }:
            return None
        return WorkerCapabilityFulfillmentResult(
            disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
            provenance=recovery.provenance,
            recovery_outcome=recovery,
            decided_at=decided_at,
        )

    def _map_resume(
        self,
        *,
        recovery,
        resume_result,
        decided_at: datetime,
    ) -> WorkerCapabilityFulfillmentResult:
        provenance = resume_result.provenance
        outcome = resume_result.outcome
        if outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
                provenance=provenance,
                recovery_outcome=recovery,
                resume_result=resume_result,
                execution_result=resume_result.execution_result,
                decided_at=decided_at,
            )
        if outcome in {
            WorkerQualifiedCapabilityResumeOutcome.BINDING_FAILED,
            WorkerQualifiedCapabilityResumeOutcome.BINDING_BLOCKED,
            WorkerQualifiedCapabilityResumeOutcome.BINDING_UNAVAILABLE,
            WorkerQualifiedCapabilityResumeOutcome.BINDING_HITL,
        }:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.BINDING_FAILED,
                provenance=provenance,
                recovery_outcome=recovery,
                resume_result=resume_result,
                decided_at=decided_at,
            )
        return WorkerCapabilityFulfillmentResult(
            disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_FAILED,
            provenance=provenance,
            recovery_outcome=recovery,
            resume_result=resume_result,
            decided_at=decided_at,
        )

    def _map_fail_closed(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        recovery,
        decided_at: datetime,
    ) -> WorkerCapabilityFulfillmentResult:
        completion = recovery.discovery_completion
        provenance = recovery.provenance
        if completion is None:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        outcome = completion.outcome
        if (
            outcome is DiscoveryCompletionOutcome.MISSING_CAPABILITY
            and not request.allow_generic_acquisition
        ):
            gap = CapabilityGap.from_discovery_completion(completion)
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.CAPABILITY_GAP,
                provenance=provenance,
                recovery_outcome=recovery,
                capability_gap=gap,
                decided_at=decided_at,
            )
        if outcome is DiscoveryCompletionOutcome.MISSING_CAPABILITY:
            qual = recovery.qualification_result
            if (
                qual is not None
                and qual.outcome is not CapabilityQualificationOutcome.QUALIFIED
            ):
                return WorkerCapabilityFulfillmentResult(
                    disposition=WorkerCapabilityFulfillmentDisposition.QUALIFICATION_FAILED,
                    provenance=provenance,
                    recovery_outcome=recovery,
                    decided_at=decided_at,
                )
        if outcome is DiscoveryCompletionOutcome.BLOCKED:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.DISCOVERY_BLOCKED,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        if outcome is DiscoveryCompletionOutcome.UNAVAILABLE:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.DISCOVERY_UNAVAILABLE,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        if outcome is DiscoveryCompletionOutcome.INCOMPLETE:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.DISCOVERY_INCOMPLETE,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        if outcome is DiscoveryCompletionOutcome.CONFLICT:
            return WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.DISCOVERY_CONFLICT,
                provenance=provenance,
                recovery_outcome=recovery,
                decided_at=decided_at,
            )
        return WorkerCapabilityFulfillmentResult(
            disposition=WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED,
            provenance=provenance,
            recovery_outcome=recovery,
            decided_at=decided_at,
        )


__all__ = ["WorkerCapabilityFulfillmentCoordinator"]
