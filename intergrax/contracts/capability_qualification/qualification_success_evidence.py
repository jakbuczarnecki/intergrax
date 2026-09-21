# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Success evidence invariants for capability qualification (UCA-4)."""

from __future__ import annotations

from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)


def validate_qualification_success_evidence(
    evidence: CapabilityQualificationEvidence | None,
) -> None:
    """QUALIFIED requires typed evidence — not implicit trust from acquisition."""
    if evidence is None:
        raise ValueError("QUALIFIED requires qualification evidence")
    if not evidence.artifact_reference and not evidence.domain_handoff_reference:
        if not evidence.evidence_ref and not evidence.verification_refs:
            raise ValueError(
                "QUALIFIED requires artifact, handoff, evidence_ref, or verification_refs",
            )


__all__ = ["validate_qualification_success_evidence"]
