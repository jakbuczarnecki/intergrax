# © Artur Czarnecki. All rights reserved.

"""Test-only Decision plugin module import counter (P0-A-R2 trust boundary proof)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_verification import (
    VerificationStageOutcome,
    validate_verification_stage_kind,
    verification_stage_record,
)
from intergrax.contracts.decision_record import CandidateDecision, candidate_decision_ref
from intergrax.contracts.decision_verification_stage import VerificationStageExecutionClass

IMPORT_COUNTER: int = 0
IMPORT_COUNTER += 1


@dataclass(frozen=True, slots=True)
class PreloadTrustCounterStage:
    kind: str = validate_verification_stage_kind("preload.trust.counter")
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(self, candidate: CandidateDecision[object]) -> object:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.PASSED,
        )
