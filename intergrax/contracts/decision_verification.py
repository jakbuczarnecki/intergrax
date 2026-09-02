# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision Verification result and challenge contracts (DS-VER-PIPE-03).

Typed immutable output language for Decision Verification. Expresses whether an
exact Decision Version passed or was challenged — without revision, authorization,
execution, or finalization semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

from intergrax.contracts.decision_record import (
    DecisionProposalRef,
    decision_proposal_ref_sort_key,
)

VerificationStageKind = NewType("VerificationStageKind", str)
VerificationRequirementCode = NewType("VerificationRequirementCode", str)
VerificationFindingCode = NewType("VerificationFindingCode", str)


def _validate_canonical_string(value: str, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return value


def validate_verification_stage_kind(
    value: str | VerificationStageKind,
) -> VerificationStageKind:
    return VerificationStageKind(
        _validate_canonical_string(value, "VerificationStageKind"),
    )


def validate_verification_requirement_code(
    value: str | VerificationRequirementCode,
) -> VerificationRequirementCode:
    return VerificationRequirementCode(
        _validate_canonical_string(value, "VerificationRequirementCode"),
    )


def validate_verification_finding_code(
    value: str | VerificationFindingCode,
) -> VerificationFindingCode:
    return VerificationFindingCode(
        _validate_canonical_string(value, "VerificationFindingCode"),
    )


class VerificationDisposition(str, Enum):
    """Semantic verification outcome for one evaluated Decision Version."""

    PASSED = "passed"
    CHALLENGED = "challenged"


class VerificationStageOutcome(str, Enum):
    """Per-stage verification outcome within one pipeline run."""

    PASSED = "passed"
    CHALLENGED = "challenged"


@dataclass(frozen=True, slots=True)
class VerificationFinding:
    """Minimal typed verification reason without unstructured diagnostics."""

    code: VerificationFindingCode
    message: str

    def __post_init__(self) -> None:
        validate_verification_finding_code(self.code)
        _validate_canonical_string(self.message, "VerificationFinding.message")


def verification_finding(
    *,
    code: VerificationFindingCode,
    message: str,
) -> VerificationFinding:
    """Build one typed verification finding."""
    return VerificationFinding(code=code, message=message)


@dataclass(frozen=True, slots=True)
class VerificationChallenge:
    """Evidence that one exact Decision Version failed a verification requirement.

    Challenge documents what failed and why — it does not command revision,
    authorization, execution, or lifecycle transition.
    """

    proposal_ref: DecisionProposalRef
    stage: VerificationStageKind
    requirement_code: VerificationRequirementCode
    finding: VerificationFinding

    def __post_init__(self) -> None:
        if type(self.proposal_ref) is not DecisionProposalRef:
            raise TypeError(
                "VerificationChallenge.proposal_ref must be DecisionProposalRef",
            )
        validate_verification_stage_kind(self.stage)
        validate_verification_requirement_code(self.requirement_code)
        if type(self.finding) is not VerificationFinding:
            raise TypeError("VerificationChallenge.finding must be VerificationFinding")


def verification_challenge(
    *,
    proposal_ref: DecisionProposalRef,
    stage: VerificationStageKind,
    requirement_code: VerificationRequirementCode,
    finding: VerificationFinding,
) -> VerificationChallenge:
    """Build one challenge bound to an exact evaluated Decision proposal ref."""
    return VerificationChallenge(
        proposal_ref=proposal_ref,
        stage=stage,
        requirement_code=requirement_code,
        finding=finding,
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


def _validate_stage_record_coherence(record: VerificationStageRecord) -> None:
    if type(record.outcome) is not VerificationStageOutcome:
        raise TypeError(
            "VerificationStageRecord.outcome must be VerificationStageOutcome",
        )
    if record.outcome is VerificationStageOutcome.PASSED:
        if record.challenge is not None:
            raise ValueError(
                "VerificationStageRecord with PASSED outcome cannot include challenge",
            )
        return
    if record.challenge is None:
        raise ValueError(
            "VerificationStageRecord with CHALLENGED outcome requires challenge",
        )
    if type(record.challenge) is not VerificationChallenge:
        raise TypeError(
            "VerificationStageRecord.challenge must be VerificationChallenge",
        )


@dataclass(frozen=True, slots=True)
class VerificationStageRecord:
    """Immutable per-stage verification record for one pipeline execution."""

    stage: VerificationStageKind
    outcome: VerificationStageOutcome
    challenge: VerificationChallenge | None = None

    def __post_init__(self) -> None:
        validate_verification_stage_kind(self.stage)
        _validate_stage_record_coherence(self)


def verification_stage_record(
    *,
    stage: VerificationStageKind,
    outcome: VerificationStageOutcome,
    challenge: VerificationChallenge | None = None,
    proposal_ref: DecisionProposalRef | None = None,
) -> VerificationStageRecord:
    """Build one stage record; optional proposal_ref enforces challenge binding."""
    record = VerificationStageRecord(
        stage=stage,
        outcome=outcome,
        challenge=challenge,
    )
    if proposal_ref is not None and challenge is not None:
        _require_matching_proposal_ref(
            expected=proposal_ref,
            actual=challenge.proposal_ref,
            field_name="VerificationStageRecord.challenge.proposal_ref",
        )
    return record


def _validate_stage_records(
    *,
    proposal_ref: DecisionProposalRef,
    stage_records: tuple[VerificationStageRecord, ...],
) -> None:
    if not stage_records:
        raise ValueError(
            "VerificationResult.stage_records must contain at least one stage record; "
            "empty verification cannot synthesize pass or challenge",
        )
    for index, record in enumerate(stage_records):
        if type(record) is not VerificationStageRecord:
            raise TypeError(
                "VerificationResult.stage_records must contain VerificationStageRecord",
            )
        if record.challenge is not None:
            _require_matching_proposal_ref(
                expected=proposal_ref,
                actual=record.challenge.proposal_ref,
                field_name=f"VerificationResult.stage_records[{index}].challenge.proposal_ref",
            )


def _validate_result_coherence(
    *,
    disposition: VerificationDisposition,
    stage_records: tuple[VerificationStageRecord, ...],
) -> None:
    challenged_records = tuple(
        record
        for record in stage_records
        if record.outcome is VerificationStageOutcome.CHALLENGED
    )
    if disposition is VerificationDisposition.PASSED:
        if challenged_records:
            raise ValueError(
                "VerificationResult with PASSED disposition cannot include "
                "challenged stage records",
            )
        if any(record.challenge is not None for record in stage_records):
            raise ValueError(
                "VerificationResult with PASSED disposition cannot include challenges",
            )
        return
    if not challenged_records:
        raise ValueError(
            "VerificationResult with CHALLENGED disposition requires at least one "
            "challenged stage record",
        )


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Immutable aggregate verification outcome for one exact Decision proposal ref."""

    proposal_ref: DecisionProposalRef
    disposition: VerificationDisposition
    stage_records: tuple[VerificationStageRecord, ...]

    def __post_init__(self) -> None:
        if type(self.proposal_ref) is not DecisionProposalRef:
            raise TypeError("VerificationResult.proposal_ref must be DecisionProposalRef")
        if type(self.disposition) is not VerificationDisposition:
            raise TypeError(
                "VerificationResult.disposition must be VerificationDisposition",
            )
        _validate_stage_records(
            proposal_ref=self.proposal_ref,
            stage_records=self.stage_records,
        )
        _validate_result_coherence(
            disposition=self.disposition,
            stage_records=self.stage_records,
        )


def verification_result(
    *,
    proposal_ref: DecisionProposalRef,
    disposition: VerificationDisposition,
    stage_records: tuple[VerificationStageRecord, ...],
) -> VerificationResult:
    """Build one aggregate verification result with coherence invariants enforced."""
    return VerificationResult(
        proposal_ref=proposal_ref,
        disposition=disposition,
        stage_records=stage_records,
    )


def validate_verification_result(result: VerificationResult) -> VerificationResult:
    """Re-validate one verification result invariant."""
    if type(result) is not VerificationResult:
        raise TypeError("result must be VerificationResult")
    return result
