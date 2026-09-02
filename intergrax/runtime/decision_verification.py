# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision Verification pipeline orchestrator (DS-VER-PIPE-01).

Runs configured verification stages against one exact immutable CandidateDecision
and returns one VerificationResult. Does not revise decisions, authorize execution,
invoke HITL, or finalize lifecycle outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionProposalRef,
    candidate_decision_ref,
    decision_proposal_ref_sort_key,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationStageOutcome,
    VerificationStageRecord,
    VerificationResult,
    verification_result,
)
from intergrax.contracts.decision_verification_stage import (
    T,
    VerificationStageRegistration,
    VerificationStageRegistry,
)


def _proposal_ref_key(ref: DecisionProposalRef) -> tuple[str | int | None, ...]:
    return decision_proposal_ref_sort_key(ref)


def _require_matching_proposal_ref(
    *,
    expected: DecisionProposalRef,
    actual: DecisionProposalRef,
    field_name: str,
) -> None:
    if _proposal_ref_key(expected) != _proposal_ref_key(actual):
        raise ValueError(
            f"{field_name} must match the evaluated Decision proposal reference",
        )


def _validate_returned_stage_record(
    *,
    record: VerificationStageRecord,
    registration: VerificationStageRegistration[T],
    candidate: CandidateDecision[T],
) -> None:
    if type(record) is not VerificationStageRecord:
        raise TypeError("verification stage must return VerificationStageRecord")
    expected_ref = candidate_decision_ref(candidate)
    _require_matching_proposal_ref(
        expected=expected_ref,
        actual=record.proposal_ref,
        field_name="VerificationStageRecord.proposal_ref",
    )
    if record.stage != registration.kind:
        raise ValueError(
            "VerificationStageRecord.stage must match registered stage kind: "
            f"{record.stage!r} != {registration.kind!r}",
        )


def _aggregate_disposition(
    stage_records: tuple[VerificationStageRecord, ...],
) -> VerificationDisposition:
    if any(
        record.outcome is VerificationStageOutcome.CHALLENGED
        for record in stage_records
    ):
        return VerificationDisposition.CHALLENGED
    return VerificationDisposition.PASSED


def _pipeline_execution_plan(
    registry: VerificationStageRegistry[T],
) -> tuple[VerificationStageRegistration[T], ...]:
    """Return stage registrations in pipeline execution order."""
    return registry.registrations


@dataclass(frozen=True, slots=True)
class VerificationPipeline(Generic[T]):
    """Immutable verification pipeline configuration bound to one stage registry."""

    registry: VerificationStageRegistry[T]

    async def verify(
        self,
        candidate: CandidateDecision[T],
    ) -> VerificationResult:
        """Run configured stages and return one aggregate verification result."""
        if type(candidate) is not CandidateDecision:
            raise TypeError("candidate must be CandidateDecision")
        proposal_ref = candidate_decision_ref(candidate)
        stage_records: list[VerificationStageRecord] = []
        for registration in _pipeline_execution_plan(self.registry):
            record = await registration.stage.verify(candidate)
            _validate_returned_stage_record(
                record=record,
                registration=registration,
                candidate=candidate,
            )
            stage_records.append(record)
        disposition = _aggregate_disposition(tuple(stage_records))
        return verification_result(
            proposal_ref=proposal_ref,
            disposition=disposition,
            stage_records=tuple(stage_records),
        )
