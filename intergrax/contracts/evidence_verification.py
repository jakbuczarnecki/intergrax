# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic evidence verification contracts (DS-VER-STAGE-EVID)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from intergrax.contracts.decision_record import CandidateDecision
from intergrax.contracts.decision_verification import (
    VerificationFindingCode,
    VerificationRequirementCode,
    validate_verification_finding_code,
    validate_verification_requirement_code,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageUnavailableError,
)
from intergrax.contracts.evidence_claims import (
    ClaimKind,
    EvidenceBackedClaim,
    EvidenceClaimSet,
    EvidenceReferenceId,
    validate_claim_kind,
)

T = TypeVar("T")

_EVIDENCE_MISSING_REQUIREMENT = validate_verification_requirement_code(
    "verification.evidence.required",
)
_EVIDENCE_MISSING_FINDING = validate_verification_finding_code(
    "verification.evidence.missing",
)
_EVIDENCE_SUPPORT_MISSING_REQUIREMENT = validate_verification_requirement_code(
    "verification.evidence.supporting_missing",
)
_EVIDENCE_SUPPORT_MISSING_FINDING = validate_verification_finding_code(
    "verification.evidence.supporting_missing",
)
_EVIDENCE_REFERENCE_INVALID_REQUIREMENT = validate_verification_requirement_code(
    "verification.evidence.reference_invalid",
)
_EVIDENCE_REFERENCE_INVALID_FINDING = validate_verification_finding_code(
    "verification.evidence.reference_invalid",
)


@dataclass(frozen=True, slots=True)
class EvidenceVerificationFailure:
    """One deterministic evidence verification failure."""

    requirement_code: VerificationRequirementCode
    finding_code: VerificationFindingCode
    message: str

    def __post_init__(self) -> None:
        validate_verification_requirement_code(self.requirement_code)
        validate_verification_finding_code(self.finding_code)
        if type(self.message) is not str or not self.message.strip():
            raise ValueError("EvidenceVerificationFailure.message must be non-empty str")


@dataclass(frozen=True, slots=True)
class EvidenceVerificationAssessment:
    """Deterministic evidence verification outcome."""

    passed: bool
    failure: EvidenceVerificationFailure | None = None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("EvidenceVerificationAssessment.passed must be bool")
        if self.passed and self.failure is not None:
            raise ValueError(
                "EvidenceVerificationAssessment with passed=True cannot include failure",
            )
        if not self.passed and self.failure is None:
            raise ValueError(
                "EvidenceVerificationAssessment with passed=False requires failure",
            )


def evidence_verification_passed() -> EvidenceVerificationAssessment:
    return EvidenceVerificationAssessment(passed=True)


def evidence_verification_failed(
    *,
    requirement_code: VerificationRequirementCode,
    finding_code: VerificationFindingCode,
    message: str,
) -> EvidenceVerificationAssessment:
    return EvidenceVerificationAssessment(
        passed=False,
        failure=EvidenceVerificationFailure(
            requirement_code=requirement_code,
            finding_code=finding_code,
            message=message,
        ),
    )


@runtime_checkable
class EvidenceClaimsProvider(Protocol[T]):
    """Extract canonical evidence claim sets from one decision candidate."""

    def extract(self, candidate: CandidateDecision[T]) -> EvidenceClaimSet | None:
        """Return claim set when present; None when artifact carries no evidence claims."""
        ...


@runtime_checkable
class EvidenceReferenceResolver(Protocol):
    """Resolve whether evidence references exist in backing storage."""

    def is_available(self) -> bool:
        """Return whether resolver infrastructure is available."""
        ...

    def evidence_exists(self, evidence_id: EvidenceReferenceId) -> bool:
        """Return whether one evidence reference resolves."""
        ...


def assess_evidence_claim_set(
    claim_set: EvidenceClaimSet | None,
    *,
    resolver: EvidenceReferenceResolver,
    require_claims: bool,
    require_supporting_evidence: bool,
    required_claim_kinds: tuple[ClaimKind, ...] = (),
) -> EvidenceVerificationAssessment:
    """Run deterministic structural evidence checks in stable canonical order."""
    if not resolver.is_available():
        raise VerificationStageUnavailableError(
            "evidence reference resolver infrastructure is unavailable",
        )
    if claim_set is None:
        if require_claims:
            return evidence_verification_failed(
                requirement_code=_EVIDENCE_MISSING_REQUIREMENT,
                finding_code=_EVIDENCE_MISSING_FINDING,
                message="required evidence claim set is absent",
            )
        return evidence_verification_passed()
    if require_claims and not claim_set.claims:
        return evidence_verification_failed(
            requirement_code=_EVIDENCE_MISSING_REQUIREMENT,
            finding_code=_EVIDENCE_MISSING_FINDING,
            message="required evidence claim set is empty",
        )
    required_kinds = (
        tuple(sorted(required_claim_kinds, key=str))
        if required_claim_kinds
        else ()
    )
    claims = tuple(sorted(claim_set.claims, key=lambda claim: str(claim.claim_id)))
    if required_kinds:
        present_kinds = {claim.claim_kind for claim in claims}
        for claim_kind in required_kinds:
            if claim_kind not in present_kinds:
                return evidence_verification_failed(
                    requirement_code=_EVIDENCE_MISSING_REQUIREMENT,
                    finding_code=_EVIDENCE_MISSING_FINDING,
                    message=f"required claim kind missing: {claim_kind}",
                )
    for claim in claims:
        failure = _assess_single_claim(
            claim,
            resolver=resolver,
            require_supporting_evidence=require_supporting_evidence,
        )
        if failure is not None:
            return failure
    return evidence_verification_passed()


def _assess_single_claim(
    claim: EvidenceBackedClaim,
    *,
    resolver: EvidenceReferenceResolver,
    require_supporting_evidence: bool,
) -> EvidenceVerificationAssessment | None:
    if require_supporting_evidence and not claim.supporting_evidence_ids:
        return evidence_verification_failed(
            requirement_code=_EVIDENCE_SUPPORT_MISSING_REQUIREMENT,
            finding_code=_EVIDENCE_SUPPORT_MISSING_FINDING,
            message=f"claim {claim.claim_id} lacks supporting evidence references",
        )
    evidence_ids = tuple(
        sorted(
            set(claim.supporting_evidence_ids) | set(claim.contradicting_evidence_ids),
            key=str,
        ),
    )
    for evidence_id in evidence_ids:
        if not resolver.evidence_exists(evidence_id):
            return evidence_verification_failed(
                requirement_code=_EVIDENCE_REFERENCE_INVALID_REQUIREMENT,
                finding_code=_EVIDENCE_REFERENCE_INVALID_FINDING,
                message=f"evidence reference does not resolve: {evidence_id}",
            )
    return None


def validate_required_claim_kinds(
    kinds: tuple[str | ClaimKind, ...],
) -> tuple[ClaimKind, ...]:
    """Normalize configured required claim kinds deterministically."""
    return tuple(sorted((validate_claim_kind(kind) for kind in kinds), key=str))
