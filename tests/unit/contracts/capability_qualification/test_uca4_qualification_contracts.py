# © Artur Czarnecki. All rights reserved.

"""UCA-4 / UCA-4R — qualification request/result provenance invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    derive_capability_acquisition_request_id,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_qualification.audit_record import (
    CapabilityQualificationAuditRecord,
)
from intergrax.contracts.capability_qualification.lifecycle_decision import (
    CapabilityQualificationLifecycleDecision,
    CapabilityQualificationLifecycleOutcome,
    CapabilityQualificationLifecycleReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_decision import (
    CapabilityQualificationDecision,
)
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_integrity import (
    validate_qualification_subject_binding,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
    build_acquisition_qualification_request,
    derive_capability_qualification_request_id,
)
from intergrax.contracts.capability_qualification.qualification_subject import (
    project_acquisition_qualification_subject,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)

pytestmark = pytest.mark.unit

_CREATED = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def _gap() -> CapabilityGap:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_CREATED,
    )
    return CapabilityGap.from_discovery_completion(completion)


def _succeeded_acquisition(gap: CapabilityGap) -> CapabilityAcquisitionResult:
    acq_id = derive_capability_acquisition_request_id(
        gap_id=gap.gap_id,
        request_nonce="nonce-1",
    )
    return CapabilityAcquisitionResult(
        request_id=acq_id,
        gap_id=gap.gap_id,
        strategy_id="strategy-1",
        outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
        reason_code=CapabilityAcquisitionReasonCode.NONE,
        started_at=_CREATED,
        completed_at=_CREATED,
        evidence=CapabilityAcquisitionEvidence(artifact_reference="artifact://a"),
        correlation_id="corr-acq",
        causation_id="cause-acq",
    )


def _qualification_request(
    acquisition: CapabilityAcquisitionResult,
    *,
    nonce: str = "q-nonce-1",
) -> CapabilityQualificationRequest:
    return build_acquisition_qualification_request(
        acquisition_result=acquisition,
        qualification_nonce=nonce,
        requested_at=_CREATED,
    )


def _subject(acquisition: CapabilityAcquisitionResult):
    return project_acquisition_qualification_subject(acquisition_result=acquisition)


def _subject_result_fields(acquisition: CapabilityAcquisitionResult):
    subject = _subject(acquisition)
    lineage = subject.acquisition_lineage
    assert lineage is not None
    return subject, lineage


def test_succeeded_acquisition_accepted_for_qualification() -> None:
    gap = _gap()
    request = _qualification_request(_succeeded_acquisition(gap))
    lineage = request.subject.acquisition_lineage
    assert lineage is not None
    assert lineage.acquisition_request_id


def test_failed_acquisition_rejected_at_request_boundary() -> None:
    gap = _gap()
    acquisition = _succeeded_acquisition(gap)
    failed = acquisition.model_copy(
        update={"outcome": CapabilityAcquisitionOutcome.FAILED},
    )
    with pytest.raises(ValueError, match="SUCCEEDED"):
        _qualification_request(failed)


def test_qualified_requires_evidence() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    subject, lineage = _subject_result_fields(acq)
    with pytest.raises(ValueError, match="qualification evidence"):
        CapabilityQualificationResult(
            qualification_request_id="capability-qualification-request:a:b",
            subject_kind=subject.subject_kind,
            subject_id=subject.subject_id,
            subject_integrity_fingerprint=subject.subject_integrity_fingerprint,
            acquisition_request_id=lineage.acquisition_request_id,
            gap_id=lineage.gap_id,
            strategy_id=lineage.strategy_id,
            provider_id="provider-1",
            outcome=CapabilityQualificationOutcome.QUALIFIED,
            reason_code=CapabilityQualificationReasonCode.NONE,
            started_at=_CREATED,
            completed_at=_CREATED,
        )


def test_qualified_with_evidence_ok() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    subject, lineage = _subject_result_fields(acq)
    evidence = CapabilityQualificationEvidence(
        provider_id="provider-1",
        qualification_request_id=qreq_id,
        subject_kind=subject.subject_kind,
        subject_id=subject.subject_id,
        subject_integrity_fingerprint=subject.subject_integrity_fingerprint,
        acquisition_request_id=lineage.acquisition_request_id,
        acquisition_strategy_id=lineage.strategy_id,
        gap_id=lineage.gap_id,
        artifact_reference="artifact://a",
    )
    CapabilityQualificationResult(
        qualification_request_id=qreq_id,
        subject_kind=subject.subject_kind,
        subject_id=subject.subject_id,
        subject_integrity_fingerprint=subject.subject_integrity_fingerprint,
        acquisition_request_id=lineage.acquisition_request_id,
        gap_id=lineage.gap_id,
        strategy_id=lineage.strategy_id,
        provider_id="provider-1",
        outcome=CapabilityQualificationOutcome.QUALIFIED,
        reason_code=CapabilityQualificationReasonCode.NONE,
        started_at=_CREATED,
        completed_at=_CREATED,
        evidence=evidence,
        correlation_id="corr-acq",
        causation_id="cause-acq",
    )


def _base_result_fields(
    gap: CapabilityGap,
    acq: CapabilityAcquisitionResult,
    qreq_id: str,
    *,
    evidence: CapabilityQualificationEvidence | None,
    outcome: CapabilityQualificationOutcome = CapabilityQualificationOutcome.QUALIFIED,
) -> dict[str, object]:
    subject, lineage = _subject_result_fields(acq)
    return {
        "qualification_request_id": qreq_id,
        "subject_kind": subject.subject_kind,
        "subject_id": subject.subject_id,
        "subject_integrity_fingerprint": subject.subject_integrity_fingerprint,
        "acquisition_request_id": lineage.acquisition_request_id,
        "gap_id": lineage.gap_id,
        "strategy_id": lineage.strategy_id,
        "provider_id": "provider-1",
        "outcome": outcome,
        "reason_code": CapabilityQualificationReasonCode.NONE,
        "started_at": _CREATED,
        "completed_at": _CREATED,
        "evidence": evidence,
        "correlation_id": acq.correlation_id,
        "causation_id": acq.causation_id,
    }


def _audit_record(
    acq: CapabilityAcquisitionResult,
    qreq_id: str,
    result: CapabilityQualificationResult,
    lifecycle: CapabilityQualificationLifecycleDecision,
    **overrides: object,
) -> CapabilityQualificationAuditRecord:
    subject, lineage = _subject_result_fields(acq)
    fields = {
        "qualification_request_id": qreq_id,
        "subject_kind": subject.subject_kind,
        "subject_id": subject.subject_id,
        "subject_integrity_fingerprint": subject.subject_integrity_fingerprint,
        "acquisition_request_id": lineage.acquisition_request_id,
        "acquisition_strategy_id": lineage.strategy_id,
        "gap_id": lineage.gap_id,
        "qualification_provider_id": result.provider_id,
        "qualification_outcome": result.outcome,
        "lifecycle_outcome": lifecycle.outcome,
        "correlation_id": result.correlation_id,
        "causation_id": result.causation_id,
    }
    fields.update(overrides)
    return CapabilityQualificationAuditRecord(**fields)


def _matching_evidence(
    gap: CapabilityGap,
    acq: CapabilityAcquisitionResult,
    qreq_id: str,
    *,
    provider_id: str = "provider-1",
    artifact_reference: str | None = None,
    domain_handoff_reference: str | None = None,
) -> CapabilityQualificationEvidence:
    acq_ev = acq.evidence
    art = artifact_reference
    hand = domain_handoff_reference
    if art is None and hand is None and acq_ev is not None:
        art = acq_ev.artifact_reference
        hand = acq_ev.domain_handoff_reference
    subject, lineage = _subject_result_fields(acq)
    return CapabilityQualificationEvidence(
        provider_id=provider_id,
        qualification_request_id=qreq_id,
        subject_kind=subject.subject_kind,
        subject_id=subject.subject_id,
        subject_integrity_fingerprint=subject.subject_integrity_fingerprint,
        acquisition_request_id=lineage.acquisition_request_id,
        acquisition_strategy_id=lineage.strategy_id,
        gap_id=lineage.gap_id,
        artifact_reference=art,
        domain_handoff_reference=hand,
    )


def test_correlation_exact_match_required() -> None:
    gap = _gap()
    acquisition = _succeeded_acquisition(gap)
    subject = _subject(acquisition)
    with pytest.raises(ValueError, match="correlation_id"):
        CapabilityQualificationRequest(
            qualification_request_id=derive_capability_qualification_request_id(
                subject_id=subject.subject_id,
                qualification_nonce="n",
            ),
            qualification_nonce="n",
            subject=subject,
            correlation_id="corr-B",
            causation_id=acquisition.causation_id,
            requested_at=_CREATED,
        )


def test_correlation_dropped_rejected() -> None:
    gap = _gap()
    acquisition = _succeeded_acquisition(gap)
    subject = _subject(acquisition)
    with pytest.raises(ValueError, match="correlation_id"):
        CapabilityQualificationRequest(
            qualification_request_id=derive_capability_qualification_request_id(
                subject_id=subject.subject_id,
                qualification_nonce="n",
            ),
            qualification_nonce="n",
            subject=subject,
            correlation_id=None,
            causation_id=acquisition.causation_id,
            requested_at=_CREATED,
        )


def test_causation_mismatch_rejected() -> None:
    gap = _gap()
    acquisition = _succeeded_acquisition(gap)
    subject = _subject(acquisition)
    with pytest.raises(ValueError, match="causation_id"):
        CapabilityQualificationRequest(
            qualification_request_id=derive_capability_qualification_request_id(
                subject_id=subject.subject_id,
                qualification_nonce="n",
            ),
            qualification_nonce="n",
            subject=subject,
            correlation_id=acquisition.correlation_id,
            causation_id="cause-wrong",
            requested_at=_CREATED,
        )


def test_correlation_and_causation_both_none_valid() -> None:
    gap = _gap()
    acquisition = _succeeded_acquisition(gap).model_copy(
        update={"correlation_id": None, "causation_id": None},
    )
    build_acquisition_qualification_request(
        acquisition_result=acquisition,
        qualification_nonce="n",
        requested_at=_CREATED,
    )


def test_evidence_wrong_provider_id_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id, provider_id="p2")
    with pytest.raises(ValidationError, match="provider_id"):
        CapabilityQualificationResult(
            **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
        )


def test_evidence_wrong_qualification_request_id_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id)
    evidence = evidence.model_copy(update={"qualification_request_id": "wrong"})
    with pytest.raises(ValidationError, match="qualification_request_id"):
        CapabilityQualificationResult(
            **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
        )


def test_evidence_wrong_acquisition_request_id_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id)
    evidence = evidence.model_copy(update={"acquisition_request_id": "wrong"})
    with pytest.raises(ValidationError, match="acquisition_request_id"):
        CapabilityQualificationResult(
            **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
        )


def test_evidence_wrong_subject_id_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id)
    evidence = evidence.model_copy(update={"subject_id": "wrong-subject"})
    with pytest.raises(ValidationError, match="subject_id"):
        CapabilityQualificationResult(
            **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
        )


def test_evidence_wrong_strategy_id_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id)
    evidence = evidence.model_copy(update={"acquisition_strategy_id": "wrong"})
    with pytest.raises(ValidationError, match="strategy_id"):
        CapabilityQualificationResult(
            **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
        )


def test_evidence_wrong_gap_id_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id)
    evidence = evidence.model_copy(update={"gap_id": "wrong-gap"})
    with pytest.raises(ValidationError, match="gap_id"):
        CapabilityQualificationResult(
            **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
        )


def test_rejected_outcome_with_bad_evidence_identity_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id, provider_id="p2")
    with pytest.raises(ValidationError, match="provider_id"):
        CapabilityQualificationResult(
            **_base_result_fields(
                gap,
                acq,
                qreq_id,
                evidence=evidence,
                outcome=CapabilityQualificationOutcome.REJECTED,
            ),
        )


def test_subject_artifact_match_valid() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id, artifact_reference="artifact://a")
    validate_qualification_subject_binding(
        acq.evidence or CapabilityAcquisitionEvidence(), evidence
    )


def test_subject_artifact_mismatch_invalid() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id, artifact_reference="artifact://b")
    with pytest.raises(ValueError, match="artifact_reference"):
        validate_qualification_subject_binding(
            acq.evidence or CapabilityAcquisitionEvidence(),
            evidence,
        )


def test_subject_handoff_match_valid() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap).model_copy(
        update={
            "evidence": CapabilityAcquisitionEvidence(
                domain_handoff_reference="handoff://a",
            ),
        },
    )
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(
        gap,
        acq,
        qreq_id,
        domain_handoff_reference="handoff://a",
    )
    validate_qualification_subject_binding(
        acq.evidence or CapabilityAcquisitionEvidence(), evidence
    )


def test_subject_handoff_mismatch_invalid() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap).model_copy(
        update={
            "evidence": CapabilityAcquisitionEvidence(
                domain_handoff_reference="handoff://a",
            ),
        },
    )
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(
        gap,
        acq,
        qreq_id,
        domain_handoff_reference="handoff://b",
    )
    with pytest.raises(ValueError, match="domain_handoff_reference"):
        validate_qualification_subject_binding(
            acq.evidence or CapabilityAcquisitionEvidence(),
            evidence,
        )


def test_subject_mixed_artifact_acquisition_handoff_qualification_invalid() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(
        gap,
        acq,
        qreq_id,
        artifact_reference=None,
        domain_handoff_reference="handoff://x",
    )
    with pytest.raises(ValueError, match="substitute"):
        validate_qualification_subject_binding(
            acq.evidence or CapabilityAcquisitionEvidence(),
            evidence,
        )


def test_decision_audit_provider_mismatch_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id)
    result = CapabilityQualificationResult(
        **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
    )
    lifecycle = CapabilityQualificationLifecycleDecision(
        outcome=CapabilityQualificationLifecycleOutcome.ACCEPT,
        reason_code=CapabilityQualificationLifecycleReasonCode.QUALIFICATION_ACCEPTED,
    )
    audit = _audit_record(
        acq,
        qreq_id,
        result,
        lifecycle,
        qualification_provider_id="other-provider",
    )
    with pytest.raises(ValidationError, match="qualification_provider_id"):
        CapabilityQualificationDecision(
            qualification_result=result,
            lifecycle_decision=lifecycle,
            audit_record=audit,
        )


def test_decision_audit_outcome_mismatch_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id)
    result = CapabilityQualificationResult(
        **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
    )
    lifecycle = CapabilityQualificationLifecycleDecision(
        outcome=CapabilityQualificationLifecycleOutcome.ACCEPT,
        reason_code=CapabilityQualificationLifecycleReasonCode.QUALIFICATION_ACCEPTED,
    )
    audit = _audit_record(
        acq,
        qreq_id,
        result,
        lifecycle,
        qualification_outcome=CapabilityQualificationOutcome.FAILED,
    )
    with pytest.raises(ValidationError, match="qualification_outcome"):
        CapabilityQualificationDecision(
            qualification_result=result,
            lifecycle_decision=lifecycle,
            audit_record=audit,
        )


def test_decision_audit_lifecycle_mismatch_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id)
    result = CapabilityQualificationResult(
        **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
    )
    lifecycle = CapabilityQualificationLifecycleDecision(
        outcome=CapabilityQualificationLifecycleOutcome.ACCEPT,
        reason_code=CapabilityQualificationLifecycleReasonCode.QUALIFICATION_ACCEPTED,
    )
    audit = _audit_record(
        acq,
        qreq_id,
        result,
        lifecycle,
        lifecycle_outcome=CapabilityQualificationLifecycleOutcome.QUARANTINE,
    )
    with pytest.raises(ValidationError, match="lifecycle_outcome"):
        CapabilityQualificationDecision(
            qualification_result=result,
            lifecycle_decision=lifecycle,
            audit_record=audit,
        )


def test_decision_audit_correlation_mismatch_rejected() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        subject_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = _matching_evidence(gap, acq, qreq_id)
    result = CapabilityQualificationResult(
        **_base_result_fields(gap, acq, qreq_id, evidence=evidence),
    )
    lifecycle = CapabilityQualificationLifecycleDecision(
        outcome=CapabilityQualificationLifecycleOutcome.ACCEPT,
        reason_code=CapabilityQualificationLifecycleReasonCode.QUALIFICATION_ACCEPTED,
    )
    audit = _audit_record(
        acq,
        qreq_id,
        result,
        lifecycle,
        correlation_id="wrong-corr",
    )
    with pytest.raises(ValidationError, match="correlation_id"):
        CapabilityQualificationDecision(
            qualification_result=result,
            lifecycle_decision=lifecycle,
            audit_record=audit,
        )


def test_request_binds_gap_id_from_acquisition_lineage() -> None:
    gap = _gap()
    acquisition = _succeeded_acquisition(gap)
    request = build_acquisition_qualification_request(
        acquisition_result=acquisition,
        qualification_nonce="n",
        requested_at=_CREATED,
    )
    lineage = request.subject.acquisition_lineage
    assert lineage is not None
    assert lineage.gap_id == acquisition.gap_id
