# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision Verification stage protocol and immutable registry (DS-VER-PIPE-02).

Typed verification-stage boundary for the Decision System. Stages evaluate one
exact immutable CandidateDecision and return one VerificationStageRecord; they do
not revise decisions, authorize execution, or own lifecycle transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.decision_record import CandidateDecision
from intergrax.contracts.decision_verification import (
    VerificationStageKind,
    VerificationStageRecord,
    validate_verification_stage_kind,
)

T = TypeVar("T")


class VerificationStageExecutionClass(str, Enum):
    """Pipeline ordering class — distinct from opaque stage business kind."""

    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"


class VerificationStageAlreadyRegisteredError(ValueError):
    """Raised when a verification stage kind is registered more than once."""


class VerificationStageNotRegisteredError(ValueError):
    """Raised when a syntactically valid stage kind is not registered."""


@runtime_checkable
class VerificationStage(Protocol[T]):
    """One verification mechanism evaluating one exact CandidateDecision."""

    @property
    def kind(self) -> VerificationStageKind:
        """Stable stage identity declared by the implementation."""
        ...

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        """Deterministic or probabilistic ordering class for pipeline composition."""
        ...

    async def verify(
        self,
        candidate: CandidateDecision[T],
    ) -> VerificationStageRecord:
        """Evaluate one candidate and return one stage record."""
        ...


@dataclass(frozen=True, slots=True)
class VerificationStageRegistration(Generic[T]):
    """Explicit registration binding one kind to one stage with pipeline metadata."""

    kind: VerificationStageKind
    stage: VerificationStage[T]
    required: bool

    def __post_init__(self) -> None:
        validated_kind = validate_verification_stage_kind(self.kind)
        stage_kind = validate_verification_stage_kind(self.stage.kind)
        if stage_kind != validated_kind:
            raise ValueError(
                "VerificationStageRegistration.kind must match stage.kind: "
                f"{validated_kind!r} != {stage_kind!r}",
            )
        if type(self.required) is not bool:
            raise TypeError("VerificationStageRegistration.required must be bool")


@dataclass(frozen=True, slots=True)
class VerificationStageRegistry(Generic[T]):
    """Immutable map of registered verification stages keyed by kind."""

    registrations: tuple[VerificationStageRegistration[T], ...] = ()

    def __post_init__(self) -> None:
        _validate_registry_registrations(self.registrations)


def verification_stage_registry(
    registrations: tuple[VerificationStageRegistration[T], ...] = (),
) -> VerificationStageRegistry[T]:
    """Build a registry from explicit registrations with canonical ordering."""
    return VerificationStageRegistry(
        registrations=_canonicalize_registrations(registrations),
    )


def register_verification_stage(
    registry: VerificationStageRegistry[T],
    registration: VerificationStageRegistration[T],
) -> VerificationStageRegistry[T]:
    """Return a new registry containing one additional stage; input unchanged."""
    validated = VerificationStageRegistration(
        kind=registration.kind,
        stage=registration.stage,
        required=registration.required,
    )
    if is_verification_stage_registered(registry, validated.kind):
        raise VerificationStageAlreadyRegisteredError(
            f"VerificationStageKind already registered: {validated.kind!r}",
        )
    return verification_stage_registry(registry.registrations + (validated,))


def is_verification_stage_registered(
    registry: VerificationStageRegistry[T],
    kind: str | VerificationStageKind,
) -> bool:
    """Return whether a syntactically valid stage kind is registered."""
    validated = validate_verification_stage_kind(kind)
    return any(
        registration.kind == validated for registration in registry.registrations
    )


def require_registered_verification_stage(
    registry: VerificationStageRegistry[T],
    kind: str | VerificationStageKind,
) -> VerificationStage[T]:
    """Return the registered stage for ``kind``; fail closed on unknown kinds."""
    validated = validate_verification_stage_kind(kind)
    for registration in registry.registrations:
        if registration.kind == validated:
            return registration.stage
    raise VerificationStageNotRegisteredError(
        f"VerificationStageKind not registered: {validated!r}",
    )


def _validate_registrations_no_duplicates(
    registrations: tuple[VerificationStageRegistration[T], ...],
) -> tuple[VerificationStageRegistration[T], ...]:
    validated: list[VerificationStageRegistration[T]] = []
    seen: set[str] = set()
    for registration in registrations:
        normalized = VerificationStageRegistration(
            kind=registration.kind,
            stage=registration.stage,
            required=registration.required,
        )
        if normalized.kind in seen:
            raise VerificationStageAlreadyRegisteredError(
                f"VerificationStageKind already registered: {normalized.kind!r}",
            )
        seen.add(normalized.kind)
        validated.append(normalized)
    return tuple(validated)


def _validate_registry_registrations(
    registrations: tuple[VerificationStageRegistration[T], ...],
) -> None:
    validated = _validate_registrations_no_duplicates(registrations)
    canonical = tuple(
        sorted(validated, key=lambda registration: registration.kind),
    )
    if validated != canonical:
        raise ValueError(
            "VerificationStageRegistry.registrations must be in canonical order",
        )


def _canonicalize_registrations(
    registrations: tuple[VerificationStageRegistration[T], ...],
) -> tuple[VerificationStageRegistration[T], ...]:
    validated = _validate_registrations_no_duplicates(registrations)
    return tuple(sorted(validated, key=lambda registration: registration.kind))
