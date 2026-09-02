# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Guardrail deterministic Decision Verification stage (DS-VER-STAGE-GR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.decision_record import (
    CandidateDecision,
    candidate_decision_ref,
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
from intergrax.contracts.guardrail_verification import assess_guardrail_scan
from intergrax.integrations.contracts.llm_guardrail import GuardrailScanResult

T = TypeVar("T")


GUARDRAIL_VERIFICATION_STAGE_KIND = validate_verification_stage_kind("guardrail")

_GUARDRAIL_BLOCKED_REQUIREMENT = validate_verification_requirement_code(
    "verification.guardrail.output_blocked",
)
_GUARDRAIL_BLOCKED_FINDING = validate_verification_finding_code(
    "verification.guardrail.output_blocked",
)


@runtime_checkable
class GuardrailScanProvider(Protocol[T]):
    """Extract normalized guardrail scan metadata from one decision artifact."""

    def extract(self, candidate: CandidateDecision[T]) -> GuardrailScanResult | None:
        """Return scan metadata when present; None when guardrail scan not supplied."""
        ...


@dataclass(frozen=True, slots=True)
class GuardrailVerificationStage(Generic[T]):
    """Deterministic quality guardrail verification for decision candidates."""

    scan_provider: GuardrailScanProvider[T]

    @property
    def kind(self) -> VerificationStageKind:
        return GUARDRAIL_VERIFICATION_STAGE_KIND

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        return VerificationStageExecutionClass.DETERMINISTIC

    async def verify(
        self,
        candidate: CandidateDecision[T],
    ) -> VerificationStageRecord:
        proposal_ref = candidate_decision_ref(candidate)
        scan = self.scan_provider.extract(candidate)
        if scan is None:
            return verification_stage_record(
                proposal_ref=proposal_ref,
                stage=GUARDRAIL_VERIFICATION_STAGE_KIND,
                outcome=VerificationStageOutcome.PASSED,
            )
        assessment = assess_guardrail_scan(scan)
        if assessment.passed:
            return verification_stage_record(
                proposal_ref=proposal_ref,
                stage=GUARDRAIL_VERIFICATION_STAGE_KIND,
                outcome=VerificationStageOutcome.PASSED,
            )
        finding = verification_finding(
            code=_GUARDRAIL_BLOCKED_FINDING,
            message=assessment.detail,
        )
        challenge = verification_challenge(
            proposal_ref=proposal_ref,
            stage=GUARDRAIL_VERIFICATION_STAGE_KIND,
            requirement_code=_GUARDRAIL_BLOCKED_REQUIREMENT,
            finding=finding,
        )
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=GUARDRAIL_VERIFICATION_STAGE_KIND,
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=challenge,
        )
