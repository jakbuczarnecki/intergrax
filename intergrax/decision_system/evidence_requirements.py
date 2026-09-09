# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Generic evidence requirement contracts (DS-E2E-15B.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Protocol

from intergrax.contracts.evidence_claims import EvidenceReferenceId, validate_evidence_reference_id

EvidenceRequirementId = NewType("EvidenceRequirementId", str)
EvidenceRequirementWaiverRef = NewType("EvidenceRequirementWaiverRef", str)

_REQUIREMENT_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_WAIVER_REF_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9._:-]{0,255}$")


class EvidenceRequirementCriticality(StrEnum):
    MANDATORY = "mandatory"
    OPTIONAL = "optional"


class EvidenceRequirementOutcome(StrEnum):
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    WAIVED = "waived"


class EvidenceRequirementContractError(ValueError):
    """Raised when evidence requirement contracts fail closed validation."""


def validate_evidence_requirement_id(value: object) -> EvidenceRequirementId:
    if type(value) is not str:
        raise EvidenceRequirementContractError(
            f"EvidenceRequirementId must be str, got {type(value).__name__}"
        )
    if not value or value != value.strip():
        raise EvidenceRequirementContractError("EvidenceRequirementId must be non-empty")
    if not _REQUIREMENT_ID_RE.fullmatch(value):
        raise EvidenceRequirementContractError(
            "EvidenceRequirementId must match [a-z][a-z0-9._-]{0,127}"
        )
    return EvidenceRequirementId(value)


def validate_evidence_requirement_waiver_ref(value: object) -> EvidenceRequirementWaiverRef:
    if type(value) is not str:
        raise EvidenceRequirementContractError(
            f"EvidenceRequirementWaiverRef must be str, got {type(value).__name__}"
        )
    if not value or value != value.strip():
        raise EvidenceRequirementContractError(
            "EvidenceRequirementWaiverRef must be non-empty"
        )
    if not _WAIVER_REF_RE.fullmatch(value):
        raise EvidenceRequirementContractError(
            "EvidenceRequirementWaiverRef has invalid characters"
        )
    return EvidenceRequirementWaiverRef(value)


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    requirement_id: EvidenceRequirementId
    criticality: EvidenceRequirementCriticality
    requires_evidence_provenance: bool = True


@dataclass(frozen=True, slots=True)
class EvidenceRequirementSet:
    requirements: tuple[EvidenceRequirement, ...]


@dataclass(frozen=True, slots=True)
class EvidenceRequirementAssessment:
    requirement_id: EvidenceRequirementId
    outcome: EvidenceRequirementOutcome
    supporting_evidence_refs: tuple[EvidenceReferenceId, ...] = ()
    waiver_ref: EvidenceRequirementWaiverRef | None = None


@dataclass(frozen=True, slots=True)
class EvidenceSufficiencyAssessment:
    assessments: tuple[EvidenceRequirementAssessment, ...]


class EvidenceRequirementProvider(Protocol):
    """Domain-owned port supplying semantic evidence requirements."""

    def requirements_for(self) -> EvidenceRequirementSet:
        """Return the requirement set for the active decision context."""
        ...


class EvidenceRequirementProviderAbsentError(EvidenceRequirementContractError):
    """Raised when a composed gate requires a provider but none was supplied."""


def validate_evidence_requirement_set(requirement_set: EvidenceRequirementSet) -> None:
    seen: set[EvidenceRequirementId] = set()
    for requirement in requirement_set.requirements:
        validate_evidence_requirement_id(requirement.requirement_id)
        if requirement.criticality not in EvidenceRequirementCriticality:
            raise EvidenceRequirementContractError("invalid requirement criticality")
        if requirement.requirement_id in seen:
            raise EvidenceRequirementContractError(
                f"duplicate requirement id: {requirement.requirement_id!r}"
            )
        seen.add(requirement.requirement_id)


def _validate_supporting_evidence_refs(
    refs: tuple[EvidenceReferenceId, ...],
) -> tuple[EvidenceReferenceId, ...]:
    normalized: list[EvidenceReferenceId] = []
    seen: set[EvidenceReferenceId] = set()
    for ref in refs:
        validated = validate_evidence_reference_id(ref)
        if validated in seen:
            raise EvidenceRequirementContractError("duplicate supporting evidence ref")
        seen.add(validated)
        normalized.append(validated)
    return tuple(normalized)


def validate_evidence_requirement_assessment(
    assessment: EvidenceRequirementAssessment,
    *,
    requirement: EvidenceRequirement,
) -> None:
    validate_evidence_requirement_id(assessment.requirement_id)
    if assessment.requirement_id != requirement.requirement_id:
        raise EvidenceRequirementContractError("assessment requirement id mismatch")

    supporting_refs = _validate_supporting_evidence_refs(assessment.supporting_evidence_refs)
    waiver_ref = assessment.waiver_ref
    if waiver_ref is not None:
        validate_evidence_requirement_waiver_ref(waiver_ref)

    if assessment.outcome is EvidenceRequirementOutcome.SATISFIED:
        if waiver_ref is not None:
            raise EvidenceRequirementContractError(
                "satisfied assessment cannot include waiver provenance"
            )
        if requirement.requires_evidence_provenance and not supporting_refs:
            raise EvidenceRequirementContractError(
                "satisfied assessment missing supporting evidence provenance"
            )
        return

    if assessment.outcome is EvidenceRequirementOutcome.WAIVED:
        if supporting_refs:
            raise EvidenceRequirementContractError(
                "waived assessment cannot include supporting evidence provenance"
            )
        if waiver_ref is None:
            raise EvidenceRequirementContractError("waived assessment missing waiver provenance")
        return

    if assessment.outcome is EvidenceRequirementOutcome.UNSATISFIED:
        if waiver_ref is not None or supporting_refs:
            raise EvidenceRequirementContractError(
                "unsatisfied assessment cannot include satisfaction provenance"
            )
        return

    raise EvidenceRequirementContractError("unknown requirement outcome")


def validate_evidence_sufficiency_assessment(
    requirement_set: EvidenceRequirementSet,
    assessment: EvidenceSufficiencyAssessment,
) -> None:
    validate_evidence_requirement_set(requirement_set)
    requirements_by_id = {
        requirement.requirement_id: requirement for requirement in requirement_set.requirements
    }
    seen: set[EvidenceRequirementId] = set()
    for item in assessment.assessments:
        requirement = requirements_by_id.get(item.requirement_id)
        if requirement is None:
            raise EvidenceRequirementContractError(
                f"assessment for unknown requirement: {item.requirement_id!r}"
            )
        if item.requirement_id in seen:
            raise EvidenceRequirementContractError(
                f"duplicate assessment for requirement: {item.requirement_id!r}"
            )
        seen.add(item.requirement_id)
        validate_evidence_requirement_assessment(item, requirement=requirement)

    missing = set(requirements_by_id) - seen
    if missing:
        missing_ids = ", ".join(sorted(str(item) for item in missing))
        raise EvidenceRequirementContractError(
            f"missing assessments for requirements: {missing_ids}"
        )
