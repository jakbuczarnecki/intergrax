# © Artur Czarnecki. All rights reserved.

"""ERL Phase 1 — UNKNOWN lifecycle runtime tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability import (
    DependentExecutionGateAction,
    DependentExecutionGateRequest,
    ExternalEffectOutcome,
    UncertaintyLifecyclePhase,
    UncertaintyResolutionKind,
    evaluate_dependent_execution_gate,
)
from intergrax.runtime.enterprise_reliability import (
    UncertaintyResolutionError,
    admit_external_effect_unknown,
    advance_uncertainty_lifecycle,
    resolve_uncertainty,
)

pytestmark = pytest.mark.unit


def test_admit_and_advance_lifecycle() -> None:
    state = admit_external_effect_unknown(correlation_id="pay-1")
    assert state.effect_outcome is ExternalEffectOutcome.UNKNOWN
    contained = advance_uncertainty_lifecycle(
        state,
        UncertaintyLifecyclePhase.CONTAINED,
    )
    assert contained.lifecycle_phase is UncertaintyLifecyclePhase.CONTAINED


def test_resolve_unknown_to_success_unblocks_gate() -> None:
    state = admit_external_effect_unknown(correlation_id="pay-2")
    resolved = resolve_uncertainty(
        state,
        resolution_kind=UncertaintyResolutionKind.CONFIRMED_SUCCESS,
        resolved_outcome=ExternalEffectOutcome.SUCCESS,
    )
    assert resolved.lifecycle_phase is UncertaintyLifecyclePhase.RESOLVED
    assert resolved.effect_outcome is ExternalEffectOutcome.SUCCESS
    gate = evaluate_dependent_execution_gate(
        DependentExecutionGateRequest(
            effect_outcome=resolved.effect_outcome,
            lifecycle_phase=resolved.lifecycle_phase,
        ),
    )
    assert gate.action is DependentExecutionGateAction.ALLOW


def test_resolve_escalated_keeps_unknown_and_gates() -> None:
    state = admit_external_effect_unknown(correlation_id="pay-3")
    resolved = resolve_uncertainty(
        state,
        resolution_kind=UncertaintyResolutionKind.ESCALATED,
        resolved_outcome=ExternalEffectOutcome.UNKNOWN,
    )
    gate = evaluate_dependent_execution_gate(
        DependentExecutionGateRequest(
            effect_outcome=resolved.effect_outcome,
            lifecycle_phase=resolved.lifecycle_phase,
        ),
    )
    assert gate.action is DependentExecutionGateAction.GATE


def test_resolve_success_outcome_rejected_for_escalated() -> None:
    state = admit_external_effect_unknown(correlation_id="pay-4")
    with pytest.raises(UncertaintyResolutionError):
        resolve_uncertainty(
            state,
            resolution_kind=UncertaintyResolutionKind.ESCALATED,
            resolved_outcome=ExternalEffectOutcome.SUCCESS,
        )
