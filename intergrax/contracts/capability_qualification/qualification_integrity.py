# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provenance and subject binding helpers for capability qualification (UCA-4R, AW-7C-P2)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_subject import (
    CapabilityQualificationSubjectKind,
)


def validate_qualification_evidence_identity(
    *,
    provider_id: str | None,
    qualification_request_id: str,
    subject_kind: CapabilityQualificationSubjectKind,
    subject_id: str,
    subject_integrity_fingerprint: str,
    tenant_id: str | None,
    scope_fingerprint: str | None,
    evidence: CapabilityQualificationEvidence,
) -> None:
    """Exact-bind provider output evidence to the qualification result subject."""
    if evidence.provider_id != provider_id:
        raise ValueError("evidence provider_id must match result provider_id")
    if evidence.qualification_request_id != qualification_request_id:
        raise ValueError(
            "evidence qualification_request_id must match result qualification_request_id",
        )
    if evidence.subject_kind != subject_kind:
        raise ValueError("evidence subject_kind must match result subject_kind")
    if evidence.subject_id != subject_id:
        raise ValueError("evidence subject_id must match result subject_id")
    if evidence.subject_integrity_fingerprint != subject_integrity_fingerprint:
        raise ValueError(
            "evidence subject_integrity_fingerprint must match result fingerprint",
        )
    if evidence.tenant_id != tenant_id:
        raise ValueError("evidence tenant_id must match result tenant_id")
    if evidence.scope_fingerprint != scope_fingerprint:
        raise ValueError("evidence scope_fingerprint must match result scope_fingerprint")


def validate_qualification_subject_binding(
    acquisition_evidence: CapabilityAcquisitionEvidence,
    qualification_evidence: CapabilityQualificationEvidence,
) -> None:
    """Require qualification evidence to name the same subject as acquisition evidence."""
    acq_art = acquisition_evidence.artifact_reference
    acq_hand = acquisition_evidence.domain_handoff_reference
    qual_art = qualification_evidence.artifact_reference
    qual_hand = qualification_evidence.domain_handoff_reference

    if acq_art is not None and acq_hand is None and qual_hand is not None:
        raise ValueError(
            "qualification domain_handoff_reference cannot substitute acquisition artifact subject",
        )
    if acq_hand is not None and acq_art is None and qual_art is not None:
        raise ValueError(
            "qualification artifact_reference cannot substitute acquisition handoff subject",
        )

    if acq_art is not None:
        if qual_art != acq_art:
            if qual_art is not None:
                raise ValueError(
                    "qualification artifact_reference mismatches acquisition subject",
                )
            raise ValueError(
                "qualification evidence must bind acquisition artifact_reference subject",
            )
    if acq_hand is not None:
        if qual_hand != acq_hand:
            if qual_hand is not None:
                raise ValueError(
                    "qualification domain_handoff_reference mismatches acquisition subject",
                )
            raise ValueError(
                "qualification evidence must bind acquisition domain_handoff_reference subject",
            )


__all__ = [
    "validate_qualification_evidence_identity",
    "validate_qualification_subject_binding",
]
