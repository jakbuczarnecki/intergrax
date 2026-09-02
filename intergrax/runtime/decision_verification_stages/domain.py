# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Independent domain Decision Verification stage (DS-VER-STAGE-DOM)."""

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
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageUnavailableError,
)
from intergrax.contracts.domain_verification import (
    DomainVerificationIndependenceConfig,
    DomainVerifier,
)
from intergrax.contracts.semantic_verification import VerifierIndependenceMode

T = TypeVar("T")

DOMAIN_VERIFICATION_STAGE_KIND = validate_verification_stage_kind("domain")

_PROFILE_NOT_INDEPENDENT_REQUIREMENT = validate_verification_requirement_code(
    "verification.domain.profile_not_independent",
)
_PROFILE_NOT_INDEPENDENT_FINDING = validate_verification_finding_code(
    "verification.domain.profile_not_independent",
)


@dataclass(frozen=True, slots=True)
class IndependentDomainVerificationStage(Generic[T]):
    """Generic wrapper over one configured independent domain verifier."""

    verifier: DomainVerifier[T]
    execution_class: VerificationStageExecutionClass
    stage_kind: VerificationStageKind = DOMAIN_VERIFICATION_STAGE_KIND
    independence: DomainVerificationIndependenceConfig | None = None

    def __post_init__(self) -> None:
        if type(self.execution_class) is not VerificationStageExecutionClass:
            raise TypeError(
                "IndependentDomainVerificationStage.execution_class must be "
                "VerificationStageExecutionClass",
            )
        validate_verification_stage_kind(self.stage_kind)
        if self.independence is not None and type(self.independence) is not DomainVerificationIndependenceConfig:
            raise TypeError(
                "IndependentDomainVerificationStage.independence must be "
                "DomainVerificationIndependenceConfig or None",
            )

    @property
    def kind(self) -> VerificationStageKind:
        return self.stage_kind

    async def verify(
        self,
        candidate: CandidateDecision[T],
    ) -> VerificationStageRecord:
        if not self.verifier.is_available():
            raise VerificationStageUnavailableError(
                "domain verifier infrastructure is unavailable",
            )
        proposal_ref = candidate_decision_ref(candidate)
        if (
            self.independence is not None
            and self.independence.mode is VerifierIndependenceMode.INDEPENDENT
            and self.independence.producer_profile_id
            == self.independence.verifier_profile_id
        ):
            finding = verification_finding(
                code=_PROFILE_NOT_INDEPENDENT_FINDING,
                message="independent domain verification requires distinct profiles",
            )
            challenge = verification_challenge(
                proposal_ref=proposal_ref,
                stage=self.stage_kind,
                requirement_code=_PROFILE_NOT_INDEPENDENT_REQUIREMENT,
                finding=finding,
            )
            return verification_stage_record(
                proposal_ref=proposal_ref,
                stage=self.stage_kind,
                outcome=VerificationStageOutcome.CHALLENGED,
                challenge=challenge,
            )
        outcome = self.verifier.verify(candidate)
        if outcome.passed:
            return verification_stage_record(
                proposal_ref=proposal_ref,
                stage=self.stage_kind,
                outcome=VerificationStageOutcome.PASSED,
            )
        if (
            outcome.requirement_code is None
            or outcome.finding_code is None
            or outcome.message is None
        ):
            raise RuntimeError("domain verifier failure missing typed challenge fields")
        finding = verification_finding(
            code=outcome.finding_code,
            message=outcome.message,
        )
        challenge = verification_challenge(
            proposal_ref=proposal_ref,
            stage=self.stage_kind,
            requirement_code=outcome.requirement_code,
            finding=finding,
        )
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=self.stage_kind,
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=challenge,
        )
