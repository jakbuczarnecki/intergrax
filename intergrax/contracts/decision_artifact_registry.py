# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision Artifact kind registration contracts (DS-CORE-08).

Immutable registry of known artifact kind identities. Syntactic kind validation
remains in ``decision_record.validate_decision_artifact_kind``; this module
answers membership in an explicit, configuration-scoped registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_record import (
    DecisionArtifactKind,
    validate_decision_artifact_kind,
)


class DecisionArtifactKindAlreadyRegisteredError(ValueError):
    """Raised when an artifact kind is registered more than once."""


class DecisionArtifactKindNotRegisteredError(ValueError):
    """Raised when a syntactically valid artifact kind is not registered."""


@dataclass(frozen=True, slots=True)
class DecisionArtifactKindRegistry:
    """Immutable set of registered artifact kind identities."""

    kinds: tuple[DecisionArtifactKind, ...] = ()

    def __post_init__(self) -> None:
        _validate_registry_kinds(self.kinds)


def decision_artifact_kind_registry(
    kinds: tuple[DecisionArtifactKind, ...] = (),
) -> DecisionArtifactKindRegistry:
    """Build a registry from explicit kinds with canonical ordering."""
    return DecisionArtifactKindRegistry(kinds=_canonicalize_kinds(kinds))


def register_decision_artifact_kind(
    registry: DecisionArtifactKindRegistry,
    kind: DecisionArtifactKind,
) -> DecisionArtifactKindRegistry:
    """Return a new registry containing one additional kind; input registry unchanged."""
    validated = validate_decision_artifact_kind(kind)
    if is_decision_artifact_kind_registered(registry, validated):
        raise DecisionArtifactKindAlreadyRegisteredError(
            f"DecisionArtifactKind already registered: {validated!r}",
        )
    return decision_artifact_kind_registry(registry.kinds + (validated,))


def is_decision_artifact_kind_registered(
    registry: DecisionArtifactKindRegistry,
    kind: object,
) -> bool:
    """Return whether a syntactically valid kind is registered."""
    validated = validate_decision_artifact_kind(kind)
    return validated in registry.kinds


def require_registered_decision_artifact_kind(
    registry: DecisionArtifactKindRegistry,
    kind: object,
) -> DecisionArtifactKind:
    """Return the kind when registered; fail closed on unknown kinds."""
    validated = validate_decision_artifact_kind(kind)
    if not is_decision_artifact_kind_registered(registry, validated):
        raise DecisionArtifactKindNotRegisteredError(
            f"DecisionArtifactKind not registered: {validated!r}",
        )
    return validated


def _validate_kinds_no_duplicates(
    kinds: tuple[DecisionArtifactKind, ...],
) -> tuple[DecisionArtifactKind, ...]:
    validated: list[DecisionArtifactKind] = []
    seen: set[str] = set()
    for kind in kinds:
        normalized = validate_decision_artifact_kind(kind)
        if normalized in seen:
            raise DecisionArtifactKindAlreadyRegisteredError(
                f"DecisionArtifactKind already registered: {normalized!r}",
            )
        seen.add(normalized)
        validated.append(normalized)
    return tuple(validated)


def _validate_registry_kinds(
    kinds: tuple[DecisionArtifactKind, ...],
) -> None:
    validated = _validate_kinds_no_duplicates(kinds)
    canonical = tuple(sorted(validated, key=lambda value: value))
    if validated != canonical:
        raise ValueError(
            "DecisionArtifactKindRegistry.kinds must be in canonical order",
        )


def _canonicalize_kinds(
    kinds: tuple[DecisionArtifactKind, ...],
) -> tuple[DecisionArtifactKind, ...]:
    validated = _validate_kinds_no_duplicates(kinds)
    return tuple(sorted(validated, key=lambda value: value))
