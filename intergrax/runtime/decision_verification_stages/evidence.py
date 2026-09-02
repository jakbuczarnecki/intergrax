# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Evidence deterministic Decision Verification stage (DS-VER-STAGE-EVID)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.decision_record import (
    CandidateDecision,
    candidate_decision_ref,
)
from intergrax.contracts.decision_verification import (
    VerificationStageKind,
    VerificationStageOutcome,
    VerificationStageRecord,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
)
from intergrax.contracts.evidence_claims import ClaimKind
from intergrax.contracts.evidence_verification import (
    EvidenceClaimsProvider,
    EvidenceReferenceResolver,
    assess_evidence_claim_set,
    validate_required_claim_kinds,
)

T = TypeVar("T")

EVIDENCE_VERIFICATION_STAGE_KIND = validate_verification_stage_kind("evidence")


@dataclass(frozen=True, slots=True)
class EvidenceVerificationStageConfig:
    """Immutable evidence stage configuration."""

    require_claims: bool = True
    require_supporting_evidence: bool = True
    required_claim_kinds: tuple[ClaimKind, ...] = ()


def evidence_verification_stage_config(
    *,
    require_claims: bool = True,
    require_supporting_evidence: bool = True,
    required_claim_kinds: tuple[str | ClaimKind, ...] = (),
) -> EvidenceVerificationStageConfig:
    """Build normalized immutable evidence stage configuration."""
    return EvidenceVerificationStageConfig(
        require_claims=require_claims,
        require_supporting_evidence=require_supporting_evidence,
        required_claim_kinds=validate_required_claim_kinds(required_claim_kinds),
    )


@dataclass(frozen=True, slots=True)
class EvidenceVerificationStage(Generic[T]):
    """Deterministic evidence reference and provenance verification stage."""

    claims_provider: EvidenceClaimsProvider[T]
    resolver: EvidenceReferenceResolver
    config: EvidenceVerificationStageConfig = EvidenceVerificationStageConfig()

    def __post_init__(self) -> None:
        if type(self.config) is not EvidenceVerificationStageConfig:
            raise TypeError(
                "EvidenceVerificationStage.config must be EvidenceVerificationStageConfig",
            )

    @property
    def kind(self) -> VerificationStageKind:
        return EVIDENCE_VERIFICATION_STAGE_KIND

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        return VerificationStageExecutionClass.DETERMINISTIC

    async def verify(
        self,
        candidate: CandidateDecision[T],
    ) -> VerificationStageRecord:
        proposal_ref = candidate_decision_ref(candidate)
        claim_set = self.claims_provider.extract(candidate)
        assessment = assess_evidence_claim_set(
            claim_set,
            resolver=self.resolver,
            require_claims=self.config.require_claims,
            require_supporting_evidence=self.config.require_supporting_evidence,
            required_claim_kinds=self.config.required_claim_kinds,
        )
        if assessment.passed:
            return verification_stage_record(
                proposal_ref=proposal_ref,
                stage=EVIDENCE_VERIFICATION_STAGE_KIND,
                outcome=VerificationStageOutcome.PASSED,
            )
        failure = assessment.failure
        if failure is None:
            raise RuntimeError("evidence verification failure missing after assessment")
        finding = verification_finding(
            code=failure.finding_code,
            message=failure.message,
        )
        challenge = verification_challenge(
            proposal_ref=proposal_ref,
            stage=EVIDENCE_VERIFICATION_STAGE_KIND,
            requirement_code=failure.requirement_code,
            finding=finding,
        )
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=EVIDENCE_VERIFICATION_STAGE_KIND,
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=challenge,
        )
