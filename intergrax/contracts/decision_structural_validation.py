# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision structural validation contracts (DS-VER-STAGE-L0).

Typed deterministic validator protocol for CandidateDecision structural checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.decision_record import CandidateDecision
from intergrax.contracts.decision_verification import (
    VerificationFindingCode,
    VerificationRequirementCode,
    validate_verification_finding_code,
    validate_verification_requirement_code,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StructuralValidationFailure:
    """One deterministic structural validation failure."""

    requirement_code: VerificationRequirementCode
    finding_code: VerificationFindingCode
    message: str

    def __post_init__(self) -> None:
        validate_verification_requirement_code(self.requirement_code)
        validate_verification_finding_code(self.finding_code)
        if type(self.message) is not str or not self.message.strip():
            raise ValueError("StructuralValidationFailure.message must be non-empty str")


@dataclass(frozen=True, slots=True)
class StructuralValidationOutcome:
    """Deterministic outcome from one structural validator."""

    passed: bool
    failure: StructuralValidationFailure | None = None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("StructuralValidationOutcome.passed must be bool")
        if self.passed and self.failure is not None:
            raise ValueError(
                "StructuralValidationOutcome with passed=True cannot include failure",
            )
        if not self.passed and self.failure is None:
            raise ValueError(
                "StructuralValidationOutcome with passed=False requires failure",
            )


def structural_validation_passed() -> StructuralValidationOutcome:
    return StructuralValidationOutcome(passed=True)


def structural_validation_failed(
    *,
    requirement_code: VerificationRequirementCode,
    finding_code: VerificationFindingCode,
    message: str,
) -> StructuralValidationOutcome:
    return StructuralValidationOutcome(
        passed=False,
        failure=StructuralValidationFailure(
            requirement_code=requirement_code,
            finding_code=finding_code,
            message=message,
        ),
    )


@runtime_checkable
class DecisionStructuralValidator(Protocol[T]):
    """One deterministic structural validator for CandidateDecision artifacts."""

    @property
    def requirement_code(self) -> VerificationRequirementCode:
        """Stable requirement identity for challenge binding."""
        ...

    def validate(self, candidate: CandidateDecision[T]) -> StructuralValidationOutcome:
        """Evaluate one candidate deterministically."""
        ...
