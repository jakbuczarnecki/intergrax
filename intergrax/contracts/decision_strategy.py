# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DecisionStrategy identity, protocol, and domain registry (DS-DELIB-01).

Typed deliberation-strategy boundary for the Decision System. Strategies produce
candidate deliberation output for Decision Lifecycle; they do not finalize
decisions, authorize execution, or own verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, Protocol, runtime_checkable

DecisionStrategyKind = NewType("DecisionStrategyKind", str)


class DecisionStrategyAlreadyRegisteredError(ValueError):
    """Raised when a strategy kind is registered more than once."""


class DecisionStrategyNotRegisteredError(ValueError):
    """Raised when a syntactically valid strategy kind is not registered."""


def validate_decision_strategy_kind(value: object) -> DecisionStrategyKind:
    if type(value) is not str:
        raise TypeError(
            f"DecisionStrategyKind must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DecisionStrategyKind must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DecisionStrategyKind must not contain leading or trailing whitespace",
        )
    return DecisionStrategyKind(value)


@runtime_checkable
class DecisionStrategy(Protocol):
    """Semantic deliberation contract — produces candidates for Decision Lifecycle."""

    @property
    def kind(self) -> DecisionStrategyKind:
        """Stable strategy identity declared by the implementation."""
        ...


@dataclass(frozen=True, slots=True)
class DecisionStrategyRegistration:
    """Explicit registration pair binding one kind to one strategy implementation."""

    kind: DecisionStrategyKind
    strategy: DecisionStrategy

    def __post_init__(self) -> None:
        validated_kind = validate_decision_strategy_kind(self.kind)
        strategy_kind = validate_decision_strategy_kind(self.strategy.kind)
        if strategy_kind != validated_kind:
            raise ValueError(
                "DecisionStrategyRegistration.kind must match strategy.kind: "
                f"{validated_kind!r} != {strategy_kind!r}",
            )


@dataclass(frozen=True, slots=True)
class DecisionStrategyRegistry:
    """Immutable map of registered deliberation strategies keyed by kind."""

    registrations: tuple[DecisionStrategyRegistration, ...] = ()

    def __post_init__(self) -> None:
        _validate_registry_registrations(self.registrations)


def decision_strategy_registry(
    registrations: tuple[DecisionStrategyRegistration, ...] = (),
) -> DecisionStrategyRegistry:
    """Build a registry from explicit registrations with canonical ordering."""
    return DecisionStrategyRegistry(
        registrations=_canonicalize_registrations(registrations),
    )


def register_decision_strategy(
    registry: DecisionStrategyRegistry,
    registration: DecisionStrategyRegistration,
) -> DecisionStrategyRegistry:
    """Return a new registry containing one additional strategy; input unchanged."""
    validated = DecisionStrategyRegistration(
        kind=registration.kind,
        strategy=registration.strategy,
    )
    if is_decision_strategy_registered(registry, validated.kind):
        raise DecisionStrategyAlreadyRegisteredError(
            f"DecisionStrategyKind already registered: {validated.kind!r}",
        )
    return decision_strategy_registry(registry.registrations + (validated,))


def is_decision_strategy_registered(
    registry: DecisionStrategyRegistry,
    kind: object,
) -> bool:
    """Return whether a syntactically valid strategy kind is registered."""
    validated = validate_decision_strategy_kind(kind)
    return any(
        registration.kind == validated for registration in registry.registrations
    )


def require_registered_decision_strategy(
    registry: DecisionStrategyRegistry,
    kind: object,
) -> DecisionStrategy:
    """Return the registered strategy for ``kind``; fail closed on unknown kinds."""
    validated = validate_decision_strategy_kind(kind)
    for registration in registry.registrations:
        if registration.kind == validated:
            return registration.strategy
    raise DecisionStrategyNotRegisteredError(
        f"DecisionStrategyKind not registered: {validated!r}",
    )


def _validate_registrations_no_duplicates(
    registrations: tuple[DecisionStrategyRegistration, ...],
) -> tuple[DecisionStrategyRegistration, ...]:
    validated: list[DecisionStrategyRegistration] = []
    seen: set[str] = set()
    for registration in registrations:
        normalized = DecisionStrategyRegistration(
            kind=registration.kind,
            strategy=registration.strategy,
        )
        if normalized.kind in seen:
            raise DecisionStrategyAlreadyRegisteredError(
                f"DecisionStrategyKind already registered: {normalized.kind!r}",
            )
        seen.add(normalized.kind)
        validated.append(normalized)
    return tuple(validated)


def _validate_registry_registrations(
    registrations: tuple[DecisionStrategyRegistration, ...],
) -> None:
    validated = _validate_registrations_no_duplicates(registrations)
    canonical = tuple(
        sorted(validated, key=lambda registration: registration.kind),
    )
    if validated != canonical:
        raise ValueError(
            "DecisionStrategyRegistry.registrations must be in canonical order",
        )


def _canonicalize_registrations(
    registrations: tuple[DecisionStrategyRegistration, ...],
) -> tuple[DecisionStrategyRegistration, ...]:
    validated = _validate_registrations_no_duplicates(registrations)
    return tuple(sorted(validated, key=lambda registration: registration.kind))
