# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UNKNOWN uncertainty lifecycle phases and transitions (ERL Phase 1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome

SCHEMA_UNCERTAINTY_LIFECYCLE_V1: Final = "uncertainty_lifecycle.v1"


class UncertaintyLifecyclePhase(StrEnum):
    """Managed phases for an admitted UNKNOWN external effect."""

    ADMITTED = "admitted"
    CONTAINED = "contained"
    PENDING_RESOLUTION = "pending_resolution"
    RESOLVED = "resolved"


class UncertaintyResolutionKind(StrEnum):
    """How UNKNOWN was closed — hook for future reconcile / HITL phases."""

    CONFIRMED_SUCCESS = "confirmed_success"
    CONFIRMED_FAILURE = "confirmed_failure"
    ESCALATED = "escalated"


_ALLOWED_TRANSITIONS: dict[
    UncertaintyLifecyclePhase, frozenset[UncertaintyLifecyclePhase]
] = {
    UncertaintyLifecyclePhase.ADMITTED: frozenset({UncertaintyLifecyclePhase.CONTAINED}),
    UncertaintyLifecyclePhase.CONTAINED: frozenset(
        {UncertaintyLifecyclePhase.PENDING_RESOLUTION}
    ),
    UncertaintyLifecyclePhase.PENDING_RESOLUTION: frozenset(
        {UncertaintyLifecyclePhase.RESOLVED}
    ),
    UncertaintyLifecyclePhase.RESOLVED: frozenset(),
}


class UncertaintyLifecycleTransitionError(RuntimeError):
    """Illegal uncertainty lifecycle transition."""


class UncertaintyStateRecord(BaseModel):
    """Durable view of one external-effect uncertainty episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_id: str = Field(min_length=1, max_length=256)
    effect_outcome: ExternalEffectOutcome
    lifecycle_phase: UncertaintyLifecyclePhase
    resolution_kind: UncertaintyResolutionKind | None = None


def assert_uncertainty_lifecycle_transition(
    current: UncertaintyLifecyclePhase,
    target: UncertaintyLifecyclePhase,
) -> None:
    """Fail closed on illegal lifecycle moves."""
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise UncertaintyLifecycleTransitionError(
            f"illegal uncertainty transition: {current.value} -> {target.value}",
        )


def initial_uncertainty_state(
    *,
    correlation_id: str,
) -> UncertaintyStateRecord:
    """Create state after UNKNOWN admission."""
    return UncertaintyStateRecord(
        correlation_id=correlation_id,
        effect_outcome=ExternalEffectOutcome.UNKNOWN,
        lifecycle_phase=UncertaintyLifecyclePhase.ADMITTED,
    )


__all__ = [
    "UncertaintyLifecyclePhase",
    "UncertaintyLifecycleTransitionError",
    "UncertaintyResolutionKind",
    "UncertaintyStateRecord",
    "SCHEMA_UNCERTAINTY_LIFECYCLE_V1",
    "assert_uncertainty_lifecycle_transition",
    "initial_uncertainty_state",
]
