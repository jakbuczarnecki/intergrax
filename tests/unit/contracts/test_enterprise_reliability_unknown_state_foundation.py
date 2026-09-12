# © Artur Czarnecki. All rights reserved.

"""ERL Phase 1 — UNKNOWN state foundation contract tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability import (
    DependentExecutionGateAction,
    DependentExecutionGateRequest,
    ExternalEffectEvidenceVerdict,
    ExternalEffectOutcome,
    ExternalEffectReliabilityInteraction,
    UncertaintyLifecyclePhase,
    UncertaintyLifecycleTransitionError,
    UncertaintyResolutionKind,
    assert_uncertainty_lifecycle_transition,
    classify_external_effect_outcome,
    evaluate_dependent_execution_gate,
    external_effect_outcome_from_physical_state,
    initial_uncertainty_state,
    project_external_effect_to_reliability,
)
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryAction,
    ExecutionRetryEligibilityRequest,
)
from intergrax.contracts.external_operation_cancellation import ExternalOperationPhysicalState
from intergrax.runtime.execution.retry.policy import evaluate_execution_retry_eligibility

pytestmark = pytest.mark.unit


def test_insufficient_evidence_admits_unknown_not_failure() -> None:
    outcome = classify_external_effect_outcome(
        ExternalEffectEvidenceVerdict.INSUFFICIENT,
    )
    assert outcome is ExternalEffectOutcome.UNKNOWN


def test_physical_unknown_maps_to_effect_unknown() -> None:
    assert (
        external_effect_outcome_from_physical_state(
            ExternalOperationPhysicalState.UNKNOWN,
        )
        is ExternalEffectOutcome.UNKNOWN
    )
    assert (
        external_effect_outcome_from_physical_state(
            ExternalOperationPhysicalState.RUNNING,
        )
        is None
    )


def test_unknown_gates_dependents_until_resolved_with_terminal_outcome() -> None:
    gate = evaluate_dependent_execution_gate(
        DependentExecutionGateRequest(
            effect_outcome=ExternalEffectOutcome.UNKNOWN,
            lifecycle_phase=UncertaintyLifecyclePhase.ADMITTED,
        ),
    )
    assert gate.action is DependentExecutionGateAction.GATE

    allow = evaluate_dependent_execution_gate(
        DependentExecutionGateRequest(
            effect_outcome=ExternalEffectOutcome.SUCCESS,
            lifecycle_phase=UncertaintyLifecyclePhase.ADMITTED,
        ),
    )
    assert allow.action is DependentExecutionGateAction.ALLOW


def test_unknown_projection_is_not_retryable_dependency_error() -> None:
    projection = project_external_effect_to_reliability(
        ExternalEffectOutcome.UNKNOWN,
        side_effect_idempotency_guaranteed=False,
    )
    assert (
        projection.interaction
        is ExternalEffectReliabilityInteraction.UNCERTAINTY_FAIL_CLOSED
    )
    assert projection.classification is not None
    assert projection.classification.kind is ExecutionFailureKind.UNKNOWN
    assert projection.classification.failure_class is None
    assert projection.classification.has_unknown_side_effect is True

    eligibility = evaluate_execution_retry_eligibility(
        ExecutionRetryEligibilityRequest(
            classification=projection.classification,
            attempt_number=1,
            max_attempts=3,
        ),
    )
    assert eligibility.action is ExecutionRetryAction.FAIL

    idempotent_projection = project_external_effect_to_reliability(
        ExternalEffectOutcome.UNKNOWN,
        side_effect_idempotency_guaranteed=True,
    )
    assert idempotent_projection.classification is not None
    assert idempotent_projection.classification.has_unknown_side_effect is False
    idempotent_eligibility = evaluate_execution_retry_eligibility(
        ExecutionRetryEligibilityRequest(
            classification=idempotent_projection.classification,
            attempt_number=1,
            max_attempts=3,
        ),
    )
    assert idempotent_eligibility.action is ExecutionRetryAction.FAIL
    assert idempotent_eligibility.reason == "unknown_fail_closed"


def test_lifecycle_transitions_are_linear_and_fail_closed() -> None:
    state = initial_uncertainty_state(correlation_id="corr-1")
    assert state.lifecycle_phase is UncertaintyLifecyclePhase.ADMITTED
    assert_uncertainty_lifecycle_transition(
        UncertaintyLifecyclePhase.ADMITTED,
        UncertaintyLifecyclePhase.CONTAINED,
    )
    with pytest.raises(UncertaintyLifecycleTransitionError):
        assert_uncertainty_lifecycle_transition(
            UncertaintyLifecyclePhase.ADMITTED,
            UncertaintyLifecyclePhase.RESOLVED,
        )


def test_failure_projection_uses_definitive_failure_not_unknown() -> None:
    projection = project_external_effect_to_reliability(ExternalEffectOutcome.FAILURE)
    assert projection.interaction is ExternalEffectReliabilityInteraction.DEFINITIVE_FAILURE
    assert projection.classification is not None
    assert projection.classification.kind is ExecutionFailureKind.NON_RETRYABLE_PERMANENT
