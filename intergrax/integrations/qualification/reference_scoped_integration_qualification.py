# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference capability qualification provider for scoped adaptation (AW-7C-P4)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.contracts.capability_qualification.qualification_subject import (
    CapabilityQualificationSubjectKind,
)

REFERENCE_SCOPED_INTEGRATION_QUALIFICATION_PROVIDER_ID = (
    "aw-7c-reference-qualification-provider"
)


class ReferenceScopedIntegrationQualificationProvider:
    """Qualification proof provider — active only when explicitly injected."""

    @property
    def provider_id(self) -> str:
        return REFERENCE_SCOPED_INTEGRATION_QUALIFICATION_PROVIDER_ID

    def supports(self, request: CapabilityQualificationRequest) -> bool:
        return (
            request.subject.subject_kind
            is CapabilityQualificationSubjectKind.SCOPED_INTEGRATION_ADAPTATION
        )

    def qualify(
        self,
        request: CapabilityQualificationRequest,
    ) -> CapabilityQualificationResult:
        subject = request.subject
        now = datetime.now(tz=UTC)
        evidence = CapabilityQualificationEvidence(
            provider_id=self.provider_id,
            qualification_request_id=request.qualification_request_id,
            subject_kind=subject.subject_kind,
            subject_id=subject.subject_id,
            subject_integrity_fingerprint=subject.subject_integrity_fingerprint,
            tenant_id=subject.tenant_id,
            scope_fingerprint=subject.scope_fingerprint,
            artifact_reference=subject.subject_id,
            evidence_ref="ref-qual-evidence-1",
        )
        return CapabilityQualificationResult(
            qualification_request_id=request.qualification_request_id,
            subject_kind=subject.subject_kind,
            subject_id=subject.subject_id,
            subject_integrity_fingerprint=subject.subject_integrity_fingerprint,
            tenant_id=subject.tenant_id,
            scope_fingerprint=subject.scope_fingerprint,
            provider_id=self.provider_id,
            outcome=CapabilityQualificationOutcome.QUALIFIED,
            reason_code=CapabilityQualificationReasonCode.NONE,
            started_at=now,
            completed_at=now,
            evidence=evidence,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )


__all__ = [
    "REFERENCE_SCOPED_INTEGRATION_QUALIFICATION_PROVIDER_ID",
    "ReferenceScopedIntegrationQualificationProvider",
]
