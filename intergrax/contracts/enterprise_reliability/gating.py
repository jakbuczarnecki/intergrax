# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gating signal for dependents while external-effect truth is uncertain (ERL Phase 1)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyLifecyclePhase
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome

_UNGATED_PHASES = frozenset({UncertaintyLifecyclePhase.RESOLVED})


class DependentExecutionGateAction(StrEnum):
    """Whether a dependent step may proceed without verified external truth."""

    ALLOW = "allow"
    GATE = "gate"


class DependentExecutionGateRequest(BaseModel):
    """Inputs for fail-closed dependent gating."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effect_outcome: ExternalEffectOutcome
    lifecycle_phase: UncertaintyLifecyclePhase | None = None


class DependentExecutionGateResult(BaseModel):
    """Typed gating decision for UER pause/block integration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: DependentExecutionGateAction
    reason: str = Field(default="", max_length=512)


def evaluate_dependent_execution_gate(
    request: DependentExecutionGateRequest,
) -> DependentExecutionGateResult:
    """Block risky dependents when outcome is UNKNOWN and uncertainty is not resolved."""
    if request.effect_outcome is not ExternalEffectOutcome.UNKNOWN:
        return DependentExecutionGateResult(
            action=DependentExecutionGateAction.ALLOW,
            reason="definitive_external_effect_outcome",
        )
    phase = request.lifecycle_phase
    if phase is not None and phase in _UNGATED_PHASES:
        return DependentExecutionGateResult(
            action=DependentExecutionGateAction.GATE,
            reason="unknown_outcome_at_resolved_phase_requires_explicit_outcome",
        )
    return DependentExecutionGateResult(
        action=DependentExecutionGateAction.GATE,
        reason="external_effect_unknown",
    )


__all__ = [
    "DependentExecutionGateAction",
    "DependentExecutionGateRequest",
    "DependentExecutionGateResult",
    "evaluate_dependent_execution_gate",
]
