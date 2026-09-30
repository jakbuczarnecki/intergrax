# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded A2 scoped adaptive integration orchestration (AW-7C-P2)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.autonomous_work.scoped_adaptive_integration import (
    SCOPED_ADAPTIVE_INTEGRATION_POLICY_VERSION,
    ScopedAdaptiveIntegrationExecutionRequest,
    ScopedAdaptiveIntegrationFailureReason,
    ScopedAdaptiveIntegrationPreparationResult,
    ScopedAdaptiveIntegrationPreparationStatus,
    validate_a2_scoped_adaptive_integration_eligibility,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    build_subject_qualification_request,
)
from intergrax.contracts.capability_qualification.qualification_subject import (
    project_adaptation_qualification_subject,
)
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationError,
    ScopedIntegrationAdaptationPort,
    ScopedIntegrationAdaptationRequest,
)


class WorkerScopedAdaptiveIntegrationOrchestrationService:
    """Pure A2 preparation — stops at QUALIFICATION_PENDING; no qualify() call."""

    def __init__(
        self,
        *,
        adaptation_port: ScopedIntegrationAdaptationPort,
        qualification_nonce: str = "qual-a2-1",
    ) -> None:
        self._adaptation_port = adaptation_port
        self._qualification_nonce = qualification_nonce

    def prepare(
        self,
        request: ScopedAdaptiveIntegrationExecutionRequest,
        *,
        prepared_at: datetime | None = None,
    ) -> ScopedAdaptiveIntegrationPreparationResult:
        timestamp = prepared_at or request.requested_at
        rejection = validate_a2_scoped_adaptive_integration_eligibility(request)
        if rejection is not None:
            return _eligibility_rejection(request, reason=rejection, prepared_at=timestamp)
        scope = request.adaptation_scope
        adaptation_request = ScopedIntegrationAdaptationRequest(
            request_id=f"scoped-adaptation:{request.acquisition_decision.decision_id}",
            tenant_id=scope.tenant_id,
            integration_category=scope.integration_category,
            provider_id=scope.provider_id,
            resource_scope=scope.resource_scope,
            scope=scope,
            correlation_id=request.correlation.correlation_id,
            causation_id=request.acquisition_decision.decision_id,
        )
        try:
            artifact = self._adaptation_port.adapt(adaptation_request)
        except ScopedIntegrationAdaptationError as exc:
            return ScopedAdaptiveIntegrationPreparationResult(
                status=ScopedAdaptiveIntegrationPreparationStatus.FAILED,
                reason_code=ScopedAdaptiveIntegrationFailureReason.ADAPTATION_FAILED,
                worker_instance_id=request.worker_instance_id,
                acquisition_decision_id=request.acquisition_decision.decision_id,
                need_id=request.need_id,
                prepared_at=timestamp,
                policy_version=SCOPED_ADAPTIVE_INTEGRATION_POLICY_VERSION,
                evidence_refs=request.evidence_refs,
                error_detail=exc.detail or exc.reason.value,
            )
        if artifact.tenant_id != request.correlation.tenant_id:
            return _eligibility_rejection(
                request,
                reason=ScopedAdaptiveIntegrationFailureReason.TENANT_MISMATCH,
                prepared_at=timestamp,
            )
        if artifact.scope.tenant_id != scope.tenant_id:
            return _eligibility_rejection(
                request,
                reason=ScopedAdaptiveIntegrationFailureReason.TENANT_MISMATCH,
                prepared_at=timestamp,
            )
        del adaptation_request
        subject = project_adaptation_qualification_subject(artifact)
        qual_request = build_subject_qualification_request(
            subject=subject,
            qualification_nonce=self._qualification_nonce,
            requested_at=timestamp,
        )
        return ScopedAdaptiveIntegrationPreparationResult(
            status=ScopedAdaptiveIntegrationPreparationStatus.QUALIFICATION_PENDING,
            reason_code=None,
            worker_instance_id=request.worker_instance_id,
            acquisition_decision_id=request.acquisition_decision.decision_id,
            need_id=request.need_id,
            prepared_at=timestamp,
            policy_version=SCOPED_ADAPTIVE_INTEGRATION_POLICY_VERSION,
            artifact=artifact,
            qualification_subject=subject,
            qualification_request=qual_request,
            evidence_refs=request.evidence_refs,
        )


def _eligibility_rejection(
    request: ScopedAdaptiveIntegrationExecutionRequest,
    *,
    reason: ScopedAdaptiveIntegrationFailureReason,
    prepared_at: datetime,
) -> ScopedAdaptiveIntegrationPreparationResult:
    return ScopedAdaptiveIntegrationPreparationResult(
        status=ScopedAdaptiveIntegrationPreparationStatus.DENIED,
        reason_code=reason,
        worker_instance_id=request.worker_instance_id,
        acquisition_decision_id=request.acquisition_decision.decision_id,
        need_id=request.need_id,
        prepared_at=prepared_at,
        policy_version=SCOPED_ADAPTIVE_INTEGRATION_POLICY_VERSION,
        evidence_refs=request.evidence_refs,
        error_detail=reason.value,
    )


__all__ = ["WorkerScopedAdaptiveIntegrationOrchestrationService"]
