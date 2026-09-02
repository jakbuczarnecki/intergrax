# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision Verification pipeline orchestrator (DS-VER-PIPE-01, DS-VER-PIPE-04, DS-VER-PIPE-06).

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
    validate_verification_finding_code,
    validate_verification_requirement_code,
    verification_challenge,
    verification_finding,
    verification_result,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    T,
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    VerificationStageRegistry,
    VerificationStageUnavailableError,
)

_REQUIRED_STAGE_UNAVAILABLE_REQUIREMENT = validate_verification_requirement_code(
    "verification.stage.required_unavailable",
)
_REQUIRED_STAGE_UNAVAILABLE_FINDING = validate_verification_finding_code(
    "verification.stage.required_unavailable",
)


class VerificationPipelineEmptyResultError(ValueError):
    """Raised when pipeline execution produces no stage records."""


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
) -> tuple[
    tuple[VerificationStageRegistration[T], ...],
    tuple[VerificationStageRegistration[T], ...],
]:
    """Return deterministic and probabilistic registrations in registry-stable order."""
    deterministic: list[VerificationStageRegistration[T]] = []
    probabilistic: list[VerificationStageRegistration[T]] = []
    for registration in registry.registrations:
        if registration.stage.execution_class is VerificationStageExecutionClass.DETERMINISTIC:
            deterministic.append(registration)
            continue
        if registration.stage.execution_class is VerificationStageExecutionClass.PROBABILISTIC:
            probabilistic.append(registration)
            continue
        raise ValueError(
            "verification stage execution_class must be VerificationStageExecutionClass",
        )
    return tuple(deterministic), tuple(probabilistic)


async def _execute_registration(
    *,
    registration: VerificationStageRegistration[T],
    candidate: CandidateDecision[T],
) -> VerificationStageRecord | None:
    try:
        record = await registration.stage.verify(candidate)
    except VerificationStageUnavailableError:
        if registration.required:
            return _required_unavailable_stage_record(
                registration=registration,
                candidate=candidate,
            )
        return None
    _validate_returned_stage_record(
        record=record,
        registration=registration,
        candidate=candidate,
    )
    return record


def _required_unavailable_stage_record(
    *,
    registration: VerificationStageRegistration[T],
    candidate: CandidateDecision[T],
) -> VerificationStageRecord:
    proposal_ref = candidate_decision_ref(candidate)
    finding = verification_finding(
        code=_REQUIRED_STAGE_UNAVAILABLE_FINDING,
        message="Required verification stage is unavailable",
    )
    challenge = verification_challenge(
        proposal_ref=proposal_ref,
        stage=registration.kind,
        requirement_code=_REQUIRED_STAGE_UNAVAILABLE_REQUIREMENT,
        finding=finding,
    )
    return verification_stage_record(
        proposal_ref=proposal_ref,
        stage=registration.kind,
        outcome=VerificationStageOutcome.CHALLENGED,
        challenge=challenge,
    )


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
        deterministic_regs, probabilistic_regs = _pipeline_execution_plan(self.registry)
        for registration in deterministic_regs:
            record = await _execute_registration(
                registration=registration,
                candidate=candidate,
            )
            if record is not None:
                stage_records.append(record)
        deterministic_challenged = any(
            record.outcome is VerificationStageOutcome.CHALLENGED
            for record in stage_records
        )
        if not deterministic_challenged:
            for registration in probabilistic_regs:
                record = await _execute_registration(
                    registration=registration,
                    candidate=candidate,
                )
                if record is not None:
                    stage_records.append(record)
        if not stage_records:
            raise VerificationPipelineEmptyResultError(
                "VerificationPipeline cannot produce a result without stage records",
            )
        disposition = _aggregate_disposition(tuple(stage_records))
        return verification_result(
            proposal_ref=proposal_ref,
            disposition=disposition,
            stage_records=tuple(stage_records),
        )
