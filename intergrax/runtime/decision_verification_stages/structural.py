# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Structural / deterministic Decision Verification stage (DS-VER-STAGE-L0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.agent_execution_validation import (
    CapabilityValidatorRegistry,
    validate_agent_execution,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    candidate_decision_ref,
)
from intergrax.contracts.decision_structural_validation import (
    DecisionStructuralValidator,
    StructuralValidationOutcome,
    structural_validation_failed,
    structural_validation_passed,
)
from intergrax.contracts.decision_verification import (
    VerificationRequirementCode,
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
)

T = TypeVar("T")
R = TypeVar("R")
T_contra = TypeVar("T_contra", contravariant=True)

STRUCTURAL_VERIFICATION_STAGE_KIND = validate_verification_stage_kind("structural")

_AGENT_EXECUTION_REQUIREMENT = validate_verification_requirement_code(
    "verification.structural.agent_execution",
)
_AGENT_EXECUTION_FINDING = validate_verification_finding_code(
    "verification.structural.agent_execution",
)
_AGENT_EXECUTION_SHAPE_FINDING = validate_verification_finding_code(
    "verification.structural.agent_execution_shape",
)
_NON_EMPTY_TEXT_REQUIREMENT = validate_verification_requirement_code(
    "verification.structural.non_empty_text",
)
_NON_EMPTY_TEXT_FINDING = validate_verification_finding_code(
    "verification.structural.non_empty_text",
)


@runtime_checkable
class TextFieldExtractor(Protocol[T_contra]):
    """Typed field projection for structural non-empty text checks."""

    def extract(self, content: T_contra) -> str:
        """Return the text value to validate."""
        ...


def _stage_record_from_outcome(
    *,
    candidate: CandidateDecision[T],
    outcome: StructuralValidationOutcome,
) -> VerificationStageRecord:
    proposal_ref = candidate_decision_ref(candidate)
    if outcome.passed:
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=STRUCTURAL_VERIFICATION_STAGE_KIND,
            outcome=VerificationStageOutcome.PASSED,
        )
    failure = outcome.failure
    if failure is None:
        raise RuntimeError("structural validation failure missing for challenged outcome")
    finding = verification_finding(
        code=failure.finding_code,
        message=failure.message,
    )
    challenge = verification_challenge(
        proposal_ref=proposal_ref,
        stage=STRUCTURAL_VERIFICATION_STAGE_KIND,
        requirement_code=failure.requirement_code,
        finding=finding,
    )
    return verification_stage_record(
        proposal_ref=proposal_ref,
        stage=STRUCTURAL_VERIFICATION_STAGE_KIND,
        outcome=VerificationStageOutcome.CHALLENGED,
        challenge=challenge,
    )


@dataclass(frozen=True, slots=True)
class StructuralVerificationStage(Generic[T]):
    """Deterministic structural verification over configured validators."""

    validators: tuple[DecisionStructuralValidator[T], ...]

    @property
    def kind(self) -> VerificationStageKind:
        return STRUCTURAL_VERIFICATION_STAGE_KIND

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        return VerificationStageExecutionClass.DETERMINISTIC

    async def verify(
        self,
        candidate: CandidateDecision[T],
    ) -> VerificationStageRecord:
        for validator in self.validators:
            outcome = validator.validate(candidate)
            if not outcome.passed:
                return _stage_record_from_outcome(candidate=candidate, outcome=outcome)
        return _stage_record_from_outcome(
            candidate=candidate,
            outcome=structural_validation_passed(),
        )


@dataclass(frozen=True, slots=True)
class NonEmptyTextStructuralValidator(Generic[T]):
    """Require one extracted text field to be non-empty after strip."""

    extractor: TextFieldExtractor[T]
    field_label: str

    @property
    def requirement_code(self) -> VerificationRequirementCode:
        return _NON_EMPTY_TEXT_REQUIREMENT

    def validate(self, candidate: CandidateDecision[T]) -> StructuralValidationOutcome:
        content = candidate.artifact.content
        text = self.extractor.extract(content)
        if type(text) is not str or not text.strip():
            return structural_validation_failed(
                requirement_code=_NON_EMPTY_TEXT_REQUIREMENT,
                finding_code=_NON_EMPTY_TEXT_FINDING,
                message=f"{self.field_label} must be non-empty",
            )
        return structural_validation_passed()


@dataclass(frozen=True, slots=True)
class AgentExecutionStructuralValidator:
    """Reuse domain-neutral agent execution validation on decision artifacts."""

    contract: AgentContract
    capability: str | None = None
    plan_criteria: tuple[str, ...] = ()
    capability_validators: CapabilityValidatorRegistry | None = None

    @property
    def requirement_code(self) -> VerificationRequirementCode:
        return _AGENT_EXECUTION_REQUIREMENT

    def validate(
        self,
        candidate: CandidateDecision[AgentExecutionResult],
    ) -> StructuralValidationOutcome:
        execution = candidate.artifact.content
        if type(execution) is not AgentExecutionResult:
            return structural_validation_failed(
                requirement_code=_AGENT_EXECUTION_REQUIREMENT,
                finding_code=_AGENT_EXECUTION_SHAPE_FINDING,
                message="artifact content must be AgentExecutionResult",
            )
        result = validate_agent_execution(
            execution,
            contract=self.contract,
            capability=self.capability,
            plan_criteria=self.plan_criteria,
            capability_validators=self.capability_validators,
        )
        if result.valid:
            return structural_validation_passed()
        first_error = result.errors[0] if result.errors else "agent execution validation failed"
        return structural_validation_failed(
            requirement_code=_AGENT_EXECUTION_REQUIREMENT,
            finding_code=_AGENT_EXECUTION_FINDING,
            message=first_error,
        )
