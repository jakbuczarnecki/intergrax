# © Artur Czarnecki. All rights reserved.

"""Thin adapter: platform CriticVerdict → GAP-1A EvidenceChallenge (no lifecycle ownership)."""

from __future__ import annotations

from intergrax.contracts.evidence_claims import (
    ChallengeDefectFamily,
    ChallengeResolution,
    DefectCode,
    EvidenceChallenge,
    EvidenceChallengeId,
    EvidenceClaimId,
    EvidenceReferenceId,
    mint_evidence_challenge_id,
    validate_defect_code,
    validate_evidence_reference_id,
)
from intergrax.runtime.critic.contracts import CriticVerdict
from platform_proofs.scenarios.ai_incident_investigation.validation import (
    UNSUPPORTED_INFERENCE_ERROR,
)

UNSUPPORTED_INFERENCE_DEFECT = validate_defect_code("incident.unsupported_inference")


def map_critic_verdict_to_challenge(
    verdict: CriticVerdict,
    *,
    claim_id: EvidenceClaimId,
    evidence_ids: tuple[EvidenceReferenceId, ...] = (),
) -> EvidenceChallenge | None:
    if verdict.passed:
        return None

    defect_family = ChallengeDefectFamily.UNSUPPORTED_INFERENCE
    defect_code = UNSUPPORTED_INFERENCE_DEFECT
    for reason in verdict.failure_reasons:
        if reason == UNSUPPORTED_INFERENCE_ERROR:
            defect_family = ChallengeDefectFamily.UNSUPPORTED_INFERENCE
            defect_code = UNSUPPORTED_INFERENCE_DEFECT
            break
        if "missing_claim_set" in reason or "missing_diagnosis" in reason:
            defect_family = ChallengeDefectFamily.MISSING_EVIDENCE
            defect_code = validate_defect_code("incident.missing_claim_material")
            break

    description = "; ".join(verdict.failure_reasons) or "critic_rejected_material_claim"
    return EvidenceChallenge(
        challenge_id=mint_evidence_challenge_id(),
        claim_id=claim_id,
        defect_family=defect_family,
        defect_code=defect_code,
        evidence_ids=evidence_ids,
        description=description,
        resolution=ChallengeResolution.OPEN,
    )


def build_satisfied_challenge(
    challenge_id: EvidenceChallengeId,
    *,
    claim_id: EvidenceClaimId,
    evidence_ids: tuple[EvidenceReferenceId, ...],
    description: str,
) -> EvidenceChallenge:
    return EvidenceChallenge(
        challenge_id=challenge_id,
        claim_id=claim_id,
        defect_family=ChallengeDefectFamily.UNSUPPORTED_INFERENCE,
        defect_code=UNSUPPORTED_INFERENCE_DEFECT,
        evidence_ids=evidence_ids,
        description=description,
        resolution=ChallengeResolution.SATISFIED,
    )
