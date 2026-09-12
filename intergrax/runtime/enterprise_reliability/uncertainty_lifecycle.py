# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UNKNOWN uncertainty lifecycle orchestration (ERL Phase 1)."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyLifecyclePhase,
    UncertaintyLifecycleTransitionError,
    UncertaintyResolutionKind,
    UncertaintyStateRecord,
    assert_uncertainty_lifecycle_transition,
    initial_uncertainty_state,
)
from intergrax.contracts.enterprise_reliability.outcome import (
    ExternalEffectOutcome,
    is_terminal_external_effect_outcome,
)


class UncertaintyResolutionError(ValueError):
    """Resolution payload is inconsistent with lifecycle rules."""


def admit_external_effect_unknown(*, correlation_id: str) -> UncertaintyStateRecord:
    """Enter UNKNOWN at admission — preserves uncertainty without guessing."""
    return initial_uncertainty_state(correlation_id=correlation_id)


def advance_uncertainty_lifecycle(
    state: UncertaintyStateRecord,
    target_phase: UncertaintyLifecyclePhase,
) -> UncertaintyStateRecord:
    """Apply one legal lifecycle transition."""
    assert_uncertainty_lifecycle_transition(state.lifecycle_phase, target_phase)
    return state.model_copy(update={"lifecycle_phase": target_phase})


def resolve_uncertainty(
    state: UncertaintyStateRecord,
    *,
    resolution_kind: UncertaintyResolutionKind,
    resolved_outcome: ExternalEffectOutcome,
) -> UncertaintyStateRecord:
    """
    Terminalize uncertainty with evidence-backed outcome.

    ESCALATED may keep UNKNOWN as the operational outcome; success/failure require
    terminal effect outcomes.
    """
    if state.lifecycle_phase is UncertaintyLifecyclePhase.RESOLVED:
        raise UncertaintyLifecycleTransitionError("uncertainty already resolved")
    if state.effect_outcome is not ExternalEffectOutcome.UNKNOWN:
        raise UncertaintyResolutionError("resolve applies only to UNKNOWN episodes")

    if resolution_kind is UncertaintyResolutionKind.ESCALATED:
        if resolved_outcome is not ExternalEffectOutcome.UNKNOWN:
            raise UncertaintyResolutionError(
                "escalated resolution must retain UNKNOWN outcome until governed close",
            )
    elif not is_terminal_external_effect_outcome(resolved_outcome):
        raise UncertaintyResolutionError(
            "confirmed resolution requires SUCCESS or FAILURE outcome",
        )

    pending = state
    while pending.lifecycle_phase is not UncertaintyLifecyclePhase.RESOLVED:
        if pending.lifecycle_phase is UncertaintyLifecyclePhase.ADMITTED:
            pending = advance_uncertainty_lifecycle(
                pending,
                UncertaintyLifecyclePhase.CONTAINED,
            )
        elif pending.lifecycle_phase is UncertaintyLifecyclePhase.CONTAINED:
            pending = advance_uncertainty_lifecycle(
                pending,
                UncertaintyLifecyclePhase.PENDING_RESOLUTION,
            )
        elif pending.lifecycle_phase is UncertaintyLifecyclePhase.PENDING_RESOLUTION:
            pending = advance_uncertainty_lifecycle(
                pending,
                UncertaintyLifecyclePhase.RESOLVED,
            )
        else:
            raise UncertaintyLifecycleTransitionError(
                f"cannot resolve from lifecycle phase: {pending.lifecycle_phase.value}",
            )

    return pending.model_copy(
        update={
            "effect_outcome": resolved_outcome,
            "resolution_kind": resolution_kind,
        },
    )


__all__ = [
    "UncertaintyResolutionError",
    "admit_external_effect_unknown",
    "advance_uncertainty_lifecycle",
    "resolve_uncertainty",
]
