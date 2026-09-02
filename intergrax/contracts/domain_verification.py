# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Independent domain verification contracts (DS-VER-STAGE-DOM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.decision_record import CandidateDecision
from intergrax.contracts.decision_verification import (
    VerificationFindingCode,
    VerificationRequirementCode,
    validate_verification_finding_code,
    validate_verification_requirement_code,
)
from intergrax.contracts.semantic_verification import (
    SemanticVerificationIndependenceConfig,
    VerifierIndependenceMode,
    validate_verifier_independence_mode_profiles,
)
from intergrax.runtime.execution.inference_profile import (
    InferenceProfileId,
    validate_inference_profile_id,
)

DomainVerifierId = NewType("DomainVerifierId", str)

T = TypeVar("T")


def validate_domain_verifier_id(value: object) -> DomainVerifierId:
    if type(value) is not str:
        raise TypeError(
            f"DomainVerifierId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DomainVerifierId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DomainVerifierId must not contain leading or trailing whitespace",
        )
    return DomainVerifierId(value)


@dataclass(frozen=True, slots=True)
class DomainVerificationOutcome:
    """Immutable typed outcome from one domain verifier invocation."""

    passed: bool
    requirement_code: VerificationRequirementCode | None = None
    finding_code: VerificationFindingCode | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("DomainVerificationOutcome.passed must be bool")
        if self.passed:
            if (
                self.requirement_code is not None
                or self.finding_code is not None
                or self.message is not None
            ):
                raise ValueError(
                    "DomainVerificationOutcome with passed=True cannot include failure fields",
                )
            return
        if self.requirement_code is None or self.finding_code is None or self.message is None:
            raise ValueError(
                "DomainVerificationOutcome with passed=False requires failure fields",
            )
        validate_verification_requirement_code(self.requirement_code)
        validate_verification_finding_code(self.finding_code)
        if type(self.message) is not str or not self.message.strip():
            raise ValueError("DomainVerificationOutcome.message must be non-empty str")


def domain_verification_passed() -> DomainVerificationOutcome:
    """Return one passing domain verification outcome."""
    return DomainVerificationOutcome(passed=True)


def domain_verification_failed(
    *,
    requirement_code: VerificationRequirementCode,
    finding_code: VerificationFindingCode,
    message: str,
) -> DomainVerificationOutcome:
    """Return one challenged domain verification outcome."""
    return DomainVerificationOutcome(
        passed=False,
        requirement_code=requirement_code,
        finding_code=finding_code,
        message=message,
    )


@runtime_checkable
class DomainVerifier(Protocol[T]):
    """Generic independent domain authority verifier."""

    @property
    def verifier_id(self) -> DomainVerifierId:
        """Stable verifier identity for observability and configuration."""
        ...

    def is_available(self) -> bool:
        """Return whether domain verifier infrastructure is available."""
        ...

    def verify(self, candidate: CandidateDecision[T]) -> DomainVerificationOutcome:
        """Evaluate one candidate without mutating decision state."""
        ...


@dataclass(frozen=True, slots=True)
class DomainVerificationIndependenceConfig:
    """Optional producer/verifier profile separation for domain verification."""

    mode: VerifierIndependenceMode
    producer_profile_id: InferenceProfileId
    verifier_profile_id: InferenceProfileId

    def __post_init__(self) -> None:
        if type(self.mode) is not VerifierIndependenceMode:
            raise TypeError(
                "DomainVerificationIndependenceConfig.mode must be VerifierIndependenceMode",
            )
        validate_inference_profile_id(self.producer_profile_id)
        validate_inference_profile_id(self.verifier_profile_id)
        validate_verifier_independence_mode_profiles(
            mode=self.mode,
            producer_profile_id=self.producer_profile_id,
            verifier_profile_id=self.verifier_profile_id,
        )
