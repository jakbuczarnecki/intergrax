# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Hybrid deliberation strategy composition foundation (DS-DELIB-07).

Declarative ordered composition of registered DecisionStrategy kinds behind one
Hybrid profile. Hybrid declares phase identity and strategy-kind bindings only;
orchestration, execution, and verification remain outside this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NewType

from intergrax.contracts.decision_strategy import (
    DecisionStrategyKind,
    DecisionStrategyNotRegisteredError,
    DecisionStrategyRegistration,
    DecisionStrategyRegistry,
    is_decision_strategy_registered,
    register_decision_strategy,
    validate_decision_strategy_kind,
)

_HYBRID_KIND = validate_decision_strategy_kind("hybrid")

HybridPhaseId = NewType("HybridPhaseId", str)


def validate_hybrid_phase_id(value: object) -> HybridPhaseId:
    """Validate a user-defined opaque hybrid phase identifier."""
    if type(value) is not str:
        raise TypeError(
            f"HybridPhaseId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "HybridPhaseId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "HybridPhaseId must not contain leading or trailing whitespace",
        )
    return HybridPhaseId(value)


def hybrid_strategy_kind() -> DecisionStrategyKind:
    """Canonical deliberation strategy identity for Hybrid composition."""
    return _HYBRID_KIND


@dataclass(frozen=True, slots=True)
class HybridPhase:
    """One logical phase bound to a registered DecisionStrategy kind."""

    phase_id: HybridPhaseId
    strategy_kind: DecisionStrategyKind

    def __post_init__(self) -> None:
        validate_hybrid_phase_id(self.phase_id)
        validated_kind = validate_decision_strategy_kind(self.strategy_kind)
        if validated_kind == _HYBRID_KIND:
            raise ValueError(
                "HybridPhase.strategy_kind must not reference hybrid",
            )


@dataclass(frozen=True, slots=True)
class HybridStrategy:
    """Declarative ordered composition of DecisionStrategy kinds by phase."""

    phases: tuple[HybridPhase, ...]
    kind: DecisionStrategyKind = field(default=_HYBRID_KIND)

    def __post_init__(self) -> None:
        validated_kind = validate_decision_strategy_kind(self.kind)
        if validated_kind != _HYBRID_KIND:
            raise ValueError(
                "HybridStrategy.kind must be hybrid "
                f"got {validated_kind!r}",
            )
        _validate_hybrid_phases(self.phases)

    def validate_registry_bindings(
        self,
        registry: DecisionStrategyRegistry,
    ) -> None:
        """Fail closed when any phase references an unregistered strategy kind."""
        validate_hybrid_strategy_registry_bindings(strategy=self, registry=registry)


def _validate_hybrid_phases(phases: tuple[object, ...]) -> None:
    if type(phases) is not tuple:
        raise TypeError("HybridStrategy.phases must be tuple")
    if len(phases) == 0:
        raise ValueError("HybridStrategy.phases must not be empty")
    seen: set[str] = set()
    for phase in phases:
        if type(phase) is not HybridPhase:
            raise TypeError(
                "HybridStrategy.phases elements must be HybridPhase, "
                f"got {type(phase).__name__}",
            )
        validated_phase_id = validate_hybrid_phase_id(phase.phase_id)
        validated_kind = validate_decision_strategy_kind(phase.strategy_kind)
        if validated_kind == _HYBRID_KIND:
            raise ValueError(
                "HybridPhase.strategy_kind must not reference hybrid",
            )
        if validated_phase_id in seen:
            raise ValueError(
                "HybridStrategy phases contain duplicate phase_id: "
                f"{validated_phase_id!r}",
            )
        seen.add(validated_phase_id)


def hybrid_phase(
    *,
    phase_id: object,
    strategy_kind: object,
) -> HybridPhase:
    """Build one validated hybrid phase without interpreting phase semantics."""
    return HybridPhase(
        phase_id=validate_hybrid_phase_id(phase_id),
        strategy_kind=validate_decision_strategy_kind(strategy_kind),
    )


def hybrid_strategy(
    *,
    phases: tuple[HybridPhase, ...],
) -> HybridStrategy:
    """Build one immutable hybrid composition preserving developer phase order."""
    if type(phases) is not tuple:
        raise TypeError("hybrid_strategy phases must be tuple")
    return HybridStrategy(phases=phases)


def validate_hybrid_strategy_registry_bindings(
    *,
    strategy: HybridStrategy,
    registry: DecisionStrategyRegistry,
) -> None:
    """Fail closed when any phase references an unregistered strategy kind."""
    for phase in strategy.phases:
        if not is_decision_strategy_registered(registry, phase.strategy_kind):
            raise DecisionStrategyNotRegisteredError(
                "HybridStrategy phase references unregistered "
                f"DecisionStrategyKind: {phase.strategy_kind!r}",
            )


def hybrid_strategy_registration(
    *,
    phases: tuple[HybridPhase, ...],
    registry: DecisionStrategyRegistry,
) -> DecisionStrategyRegistration:
    """Build one explicit Hybrid registration with registry referential validation."""
    strategy = hybrid_strategy(phases=phases)
    validate_hybrid_strategy_registry_bindings(strategy=strategy, registry=registry)
    return DecisionStrategyRegistration(kind=strategy.kind, strategy=strategy)


def register_hybrid_strategy(
    registry: DecisionStrategyRegistry,
    *,
    phases: tuple[HybridPhase, ...],
) -> DecisionStrategyRegistry:
    """Register Hybrid on ``registry`` after registry referential validation."""
    return register_decision_strategy(
        registry,
        hybrid_strategy_registration(phases=phases, registry=registry),
    )
