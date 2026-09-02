# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Semantic probabilistic Decision Verification stage (DS-VER-STAGE-SEM)."""

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
from intergrax.contracts.semantic_verification import (
    SemanticContentProvider,
    SemanticJudge,
    SemanticRubricNotFoundError,
    SemanticRubricRef,
    SemanticRubricResolver,
    SemanticVerificationIndependenceConfig,
    VerifierIndependenceMode,
)
from intergrax.tools.providers.eval.contracts import EvalJudgeInput

T = TypeVar("T")

SEMANTIC_VERIFICATION_STAGE_KIND = validate_verification_stage_kind("semantic")

_RUBRIC_UNRESOLVED_REQUIREMENT = validate_verification_requirement_code(
    "verification.semantic.rubric_unresolved",
)
_RUBRIC_UNRESOLVED_FINDING = validate_verification_finding_code(
    "verification.semantic.rubric_unresolved",
)
_EMPTY_CONTENT_REQUIREMENT = validate_verification_requirement_code(
    "verification.semantic.empty_content",
)
_EMPTY_CONTENT_FINDING = validate_verification_finding_code(
    "verification.semantic.empty_content",
)
_BELOW_REQUIREMENT = validate_verification_requirement_code(
    "verification.semantic.below_requirement",
)
_BELOW_FINDING = validate_verification_finding_code(
    "verification.semantic.below_requirement",
)
_PROFILE_NOT_INDEPENDENT_REQUIREMENT = validate_verification_requirement_code(
    "verification.semantic.profile_not_independent",
)
_PROFILE_NOT_INDEPENDENT_FINDING = validate_verification_finding_code(
    "verification.semantic.profile_not_independent",
)


def _challenged_record(
    *,
    candidate: CandidateDecision[T],
    requirement_code: str,
    finding_code: str,
    message: str,
) -> VerificationStageRecord:
    proposal_ref = candidate_decision_ref(candidate)
    finding = verification_finding(
        code=validate_verification_finding_code(finding_code),
        message=message,
    )
    challenge = verification_challenge(
        proposal_ref=proposal_ref,
        stage=SEMANTIC_VERIFICATION_STAGE_KIND,
        requirement_code=validate_verification_requirement_code(requirement_code),
        finding=finding,
    )
    return verification_stage_record(
        proposal_ref=proposal_ref,
        stage=SEMANTIC_VERIFICATION_STAGE_KIND,
        outcome=VerificationStageOutcome.CHALLENGED,
        challenge=challenge,
    )


def _bounded_summary(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return "semantic judge did not meet requirement"
    first = reasons[0].strip()
    if len(first) > 240:
        return first[:237] + "..."
    return first


@dataclass(frozen=True, slots=True)
class SemanticVerificationStage(Generic[T]):
    """Probabilistic semantic verification over a resolved rubric artifact."""

    rubric_ref: SemanticRubricRef
    rubric_resolver: SemanticRubricResolver
    content_provider: SemanticContentProvider[T]
    judge: SemanticJudge
    independence: SemanticVerificationIndependenceConfig

    def __post_init__(self) -> None:
        if type(self.rubric_ref) is not SemanticRubricRef:
            raise TypeError("SemanticVerificationStage.rubric_ref must be SemanticRubricRef")
        if type(self.independence) is not SemanticVerificationIndependenceConfig:
            raise TypeError(
                "SemanticVerificationStage.independence must be "
                "SemanticVerificationIndependenceConfig",
            )

    @property
    def kind(self) -> VerificationStageKind:
        return SEMANTIC_VERIFICATION_STAGE_KIND

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        return VerificationStageExecutionClass.PROBABILISTIC

    async def verify(
        self,
        candidate: CandidateDecision[T],
    ) -> VerificationStageRecord:
        if not self.judge.is_available():
            raise VerificationStageUnavailableError(
                "semantic judge infrastructure is unavailable",
            )
        if not self.rubric_resolver.is_available():
            raise VerificationStageUnavailableError(
                "semantic rubric resolver infrastructure is unavailable",
            )
        if (
            self.independence.mode is VerifierIndependenceMode.INDEPENDENT
            and self.independence.producer_profile_id
            == self.independence.verifier_profile_id
        ):
            return _challenged_record(
                candidate=candidate,
                requirement_code=_PROFILE_NOT_INDEPENDENT_REQUIREMENT,
                finding_code=_PROFILE_NOT_INDEPENDENT_FINDING,
                message="independent semantic verification requires distinct profiles",
            )
        try:
            rubric = self.rubric_resolver.resolve(self.rubric_ref)
        except SemanticRubricNotFoundError:
            return _challenged_record(
                candidate=candidate,
                requirement_code=_RUBRIC_UNRESOLVED_REQUIREMENT,
                finding_code=_RUBRIC_UNRESOLVED_FINDING,
                message="configured semantic rubric could not be resolved",
            )
        output_text = self.content_provider.extract(candidate)
        if type(output_text) is not str or not output_text.strip():
            return _challenged_record(
                candidate=candidate,
                requirement_code=_EMPTY_CONTENT_REQUIREMENT,
                finding_code=_EMPTY_CONTENT_FINDING,
                message="candidate semantic content is empty",
            )
        judge_input = EvalJudgeInput(
            output_text=output_text,
            rubric_id=str(rubric.ref.rubric_id),
            criteria=list(rubric.criteria),
            reference_context=rubric.reference_context,
            min_score=rubric.min_score,
            run_id=str(candidate.identity.execution.run_id),
            record_observation=False,
        )
        judge_output = self.judge.judge(judge_input)
        proposal_ref = candidate_decision_ref(candidate)
        if judge_output.passed:
            return verification_stage_record(
                proposal_ref=proposal_ref,
                stage=SEMANTIC_VERIFICATION_STAGE_KIND,
                outcome=VerificationStageOutcome.PASSED,
            )
        message = _bounded_summary(tuple(judge_output.reasons))
        finding = verification_finding(
            code=_BELOW_FINDING,
            message=message,
        )
        challenge = verification_challenge(
            proposal_ref=proposal_ref,
            stage=SEMANTIC_VERIFICATION_STAGE_KIND,
            requirement_code=_BELOW_REQUIREMENT,
            finding=finding,
        )
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=SEMANTIC_VERIFICATION_STAGE_KIND,
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=challenge,
        )
