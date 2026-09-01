# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed functional diagnostic specification contracts (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.contracts.functional_diagnostic_bounds import (
    MAX_FUNCTIONAL_DIAGNOSTIC_CHECKS,
    MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH,
    MAX_FUNCTIONAL_DIAGNOSTIC_DEPENDENCIES,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
    validate_functional_diagnostic_check_id,
    validate_functional_diagnostic_specification_id,
    validate_functional_diagnostic_specification_version,
)
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.observability.functional_validation_evidence import FunctionalValidationOutcome


class FunctionalDiagnosticRequirementKind(StrEnum):
    """Generic evidence predicates — no domain-specific diagnosis kinds."""

    OPERATION_OUTCOME_STATUS = "operation_outcome_status"
    CANDIDATE_EXISTS = "candidate_exists"
    SELECTION_EXISTS = "selection_exists"
    SELECTION_ARTIFACT_MATCH = "selection_artifact_match"
    OUTPUT_RELATION_EXISTS = "output_relation_exists"
    VALIDATION_OUTCOME = "validation_outcome"


@dataclass(frozen=True, slots=True)
class OperationOutcomeStatusRequirement:
    """PASS when an operation outcome record matches the expected status."""

    operation_id: str
    expected_status: PipelineOperationStatus


@dataclass(frozen=True, slots=True)
class CandidateExistsRequirement:
    """PASS when at least one candidate record exists for ``query_id``."""

    query_id: str


@dataclass(frozen=True, slots=True)
class SelectionExistsRequirement:
    """PASS when a selection record exists for ``query_id``."""

    query_id: str


@dataclass(frozen=True, slots=True)
class SelectionArtifactMatchRequirement:
    """
    PASS when selection matches ``expected_artifact_ref``; FAIL when selection
  exists but references a different artifact.
    """

    query_id: str
    expected_artifact_ref: str


@dataclass(frozen=True, slots=True)
class OutputRelationExistsRequirement:
    """PASS when an output relation record exists for ``operation_id``."""

    operation_id: str


@dataclass(frozen=True, slots=True)
class ValidationOutcomeRequirement:
    """PASS/FAIL from linked ``FunctionalValidationEvidence`` outcome."""

    validation_id: EventId
    expected_outcome: FunctionalValidationOutcome


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticRequirement:
    """One typed, deterministic evidence requirement for a check."""

    kind: FunctionalDiagnosticRequirementKind
    operation_outcome_status: OperationOutcomeStatusRequirement | None = None
    candidate_exists: CandidateExistsRequirement | None = None
    selection_exists: SelectionExistsRequirement | None = None
    selection_artifact_match: SelectionArtifactMatchRequirement | None = None
    output_relation_exists: OutputRelationExistsRequirement | None = None
    validation_outcome: ValidationOutcomeRequirement | None = None


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticCheck:
    """One evidence-backed claim to evaluate within a specification."""

    check_id: FunctionalDiagnosticCheckId
    requirement: FunctionalDiagnosticRequirement
    dependencies: tuple[FunctionalDiagnosticCheckId, ...] = ()
    pass_claim: str = ""
    fail_claim: str = ""
    insufficient_claim: str = ""


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticSpecification:
    """Versioned, bounded diagnostic plan — configuration, not analyzer logic."""

    specification_id: FunctionalDiagnosticSpecificationId
    version: int
    checks: tuple[FunctionalDiagnosticCheck, ...]


class FunctionalDiagnosticSpecificationIntegrityError(Exception):
    """Raised when a specification violates structural contracts."""


def validate_functional_diagnostic_specification(
    specification: FunctionalDiagnosticSpecification,
) -> FunctionalDiagnosticSpecification:
    """Validate bounds, identity, and dependency graph invariants."""
    validate_functional_diagnostic_specification_id(specification.specification_id)
    validate_functional_diagnostic_specification_version(specification.version)
    if not isinstance(specification.checks, tuple):
        raise FunctionalDiagnosticSpecificationIntegrityError("checks must be a tuple")
    if len(specification.checks) > MAX_FUNCTIONAL_DIAGNOSTIC_CHECKS:
        raise FunctionalDiagnosticSpecificationIntegrityError(
            f"checks must contain <= {MAX_FUNCTIONAL_DIAGNOSTIC_CHECKS} items",
        )
    seen_ids: set[FunctionalDiagnosticCheckId] = set()
    for index, check in enumerate(specification.checks):
        validated_check = _validate_check(check, index=index)
        if validated_check.check_id in seen_ids:
            raise FunctionalDiagnosticSpecificationIntegrityError(
                f"duplicate check_id: {validated_check.check_id}",
            )
        seen_ids.add(validated_check.check_id)
    _validate_dependency_graph(specification.checks)
    return specification


def _validate_check(check: FunctionalDiagnosticCheck, *, index: int) -> FunctionalDiagnosticCheck:
    if not isinstance(check, FunctionalDiagnosticCheck):
        raise FunctionalDiagnosticSpecificationIntegrityError(
            f"checks[{index}] must be FunctionalDiagnosticCheck",
        )
    validate_functional_diagnostic_check_id(check.check_id)
    _validate_claim(check.pass_claim, field_name=f"checks[{index}].pass_claim")
    _validate_claim(check.fail_claim, field_name=f"checks[{index}].fail_claim")
    _validate_claim(check.insufficient_claim, field_name=f"checks[{index}].insufficient_claim")
    if not isinstance(check.dependencies, tuple):
        raise FunctionalDiagnosticSpecificationIntegrityError(
            f"checks[{index}].dependencies must be a tuple",
        )
    if len(check.dependencies) > MAX_FUNCTIONAL_DIAGNOSTIC_DEPENDENCIES:
        raise FunctionalDiagnosticSpecificationIntegrityError(
            f"checks[{index}].dependencies must contain <= "
            f"{MAX_FUNCTIONAL_DIAGNOSTIC_DEPENDENCIES} items",
        )
    for dep_index, dependency in enumerate(check.dependencies):
        validate_functional_diagnostic_check_id(dependency)
    _validate_requirement(check.requirement, index=index)
    return check


def _validate_claim(value: str, *, field_name: str) -> None:
    if type(value) is not str:
        raise FunctionalDiagnosticSpecificationIntegrityError(f"{field_name} must be str")
    if len(value) > MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH:
        raise FunctionalDiagnosticSpecificationIntegrityError(
            f"{field_name} must be <= {MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH} characters",
        )


def _validate_requirement(requirement: FunctionalDiagnosticRequirement, *, index: int) -> None:
    if not isinstance(requirement, FunctionalDiagnosticRequirement):
        raise FunctionalDiagnosticSpecificationIntegrityError(
            f"checks[{index}].requirement must be FunctionalDiagnosticRequirement",
        )
    match requirement.kind:
        case FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS:
            if requirement.operation_outcome_status is None:
                raise FunctionalDiagnosticSpecificationIntegrityError(
                    "operation_outcome_status requirement missing payload",
                )
            _require_non_empty_text(
                requirement.operation_outcome_status.operation_id,
                field_name="operation_id",
            )
        case FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS:
            if requirement.candidate_exists is None:
                raise FunctionalDiagnosticSpecificationIntegrityError(
                    "candidate_exists requirement missing payload",
                )
            _require_non_empty_text(requirement.candidate_exists.query_id, field_name="query_id")
        case FunctionalDiagnosticRequirementKind.SELECTION_EXISTS:
            if requirement.selection_exists is None:
                raise FunctionalDiagnosticSpecificationIntegrityError(
                    "selection_exists requirement missing payload",
                )
            _require_non_empty_text(requirement.selection_exists.query_id, field_name="query_id")
        case FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH:
            if requirement.selection_artifact_match is None:
                raise FunctionalDiagnosticSpecificationIntegrityError(
                    "selection_artifact_match requirement missing payload",
                )
            _require_non_empty_text(
                requirement.selection_artifact_match.query_id,
                field_name="query_id",
            )
            _require_non_empty_text(
                requirement.selection_artifact_match.expected_artifact_ref,
                field_name="expected_artifact_ref",
            )
        case FunctionalDiagnosticRequirementKind.OUTPUT_RELATION_EXISTS:
            if requirement.output_relation_exists is None:
                raise FunctionalDiagnosticSpecificationIntegrityError(
                    "output_relation_exists requirement missing payload",
                )
            _require_non_empty_text(
                requirement.output_relation_exists.operation_id,
                field_name="operation_id",
            )
        case FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME:
            if requirement.validation_outcome is None:
                raise FunctionalDiagnosticSpecificationIntegrityError(
                    "validation_outcome requirement missing payload",
                )
            validate_event_id(requirement.validation_outcome.validation_id)


def _validate_dependency_graph(checks: tuple[FunctionalDiagnosticCheck, ...]) -> None:
    check_ids = {check.check_id for check in checks}
    for check in checks:
        for dependency in check.dependencies:
            if dependency not in check_ids:
                raise FunctionalDiagnosticSpecificationIntegrityError(
                    f"unknown dependency {dependency} for check {check.check_id}",
                )
            if dependency == check.check_id:
                raise FunctionalDiagnosticSpecificationIntegrityError(
                    f"check {check.check_id} cannot depend on itself",
                )
    visiting: set[FunctionalDiagnosticCheckId] = set()
    visited: set[FunctionalDiagnosticCheckId] = set()

    def visit(check_id: FunctionalDiagnosticCheckId) -> None:
        if check_id in visiting:
            raise FunctionalDiagnosticSpecificationIntegrityError(
                f"dependency cycle detected at {check_id}",
            )
        if check_id in visited:
            return
        visiting.add(check_id)
        check = next(item for item in checks if item.check_id == check_id)
        for dependency in check.dependencies:
            visit(dependency)
        visiting.remove(check_id)
        visited.add(check_id)

    for check in checks:
        visit(check.check_id)


def _require_non_empty_text(value: str, *, field_name: str) -> None:
    if type(value) is not str:
        raise FunctionalDiagnosticSpecificationIntegrityError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise FunctionalDiagnosticSpecificationIntegrityError(f"{field_name} must be non-empty")
    if value != normalized:
        raise FunctionalDiagnosticSpecificationIntegrityError(
            f"{field_name} must not contain leading or trailing whitespace",
        )


__all__ = [
    "CandidateExistsRequirement",
    "FunctionalDiagnosticCheck",
    "FunctionalDiagnosticRequirement",
    "FunctionalDiagnosticRequirementKind",
    "FunctionalDiagnosticSpecification",
    "FunctionalDiagnosticSpecificationIntegrityError",
    "OperationOutcomeStatusRequirement",
    "OutputRelationExistsRequirement",
    "SelectionArtifactMatchRequirement",
    "SelectionExistsRequirement",
    "ValidationOutcomeRequirement",
    "validate_functional_diagnostic_specification",
]
