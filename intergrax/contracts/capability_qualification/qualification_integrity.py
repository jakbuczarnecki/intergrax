# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provenance and subject binding helpers for capability qualification (UCA-4R)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)


def validate_qualification_evidence_identity(
    *,
    provider_id: str | None,
    qualification_request_id: str,
    acquisition_request_id: str,
    strategy_id: str,
    gap_id: str,
    evidence: CapabilityQualificationEvidence,
) -> None:
    """Exact-bind provider output evidence to the qualification result."""
    if evidence.provider_id != provider_id:
        raise ValueError("evidence provider_id must match result provider_id")
    if evidence.qualification_request_id != qualification_request_id:
        raise ValueError(
            "evidence qualification_request_id must match result qualification_request_id",
        )
    if evidence.acquisition_request_id != acquisition_request_id:
        raise ValueError(
            "evidence acquisition_request_id must match result acquisition_request_id",
        )
    if evidence.acquisition_strategy_id != strategy_id:
        raise ValueError(
            "evidence acquisition_strategy_id must match result strategy_id"
        )
    if evidence.gap_id != gap_id:
        raise ValueError("evidence gap_id must match result gap_id")


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
