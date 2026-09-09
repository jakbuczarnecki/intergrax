# © Artur Czarnecki. All rights reserved.

"""Pre-reconciliation validation-clean transition gate tests (DS-E2E-15D.4)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from intergrax.contracts.validation import ValidationResult
from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
    reconcile_investigation_completion,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_transition import (
    PreReconciliationRecoveryStatus,
    PreReconciliationTransitionDecision,
    PreReconciliationTransitionOutcome,
    PreReconciliationTransitionState,
    PreReconciliationValidationError,
    decide_pre_reconciliation_transition,
    enforce_pre_reconciliation_validation_clean_transition,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_UNRESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    IncidentInvestigationValidationEngine,
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)

pytestmark = pytest.mark.unit


def test_valid_state_is_ready_for_reconciliation() -> None:
    decision = decide_pre_reconciliation_transition(
        PreReconciliationTransitionState(
            validation_valid=True,
            validation_errors=(),
            revision_budget_remaining=0,
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=False,
        )
    )
    assert decision.outcome is PreReconciliationTransitionOutcome.READY_FOR_RECONCILIATION
    assert decision.validation_errors == ()
    assert decision.recovery_status is None


def test_invalid_state_with_zero_budget_is_rejected() -> None:
    decision = decide_pre_reconciliation_transition(
        PreReconciliationTransitionState(
            validation_valid=False,
            validation_errors=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
            revision_budget_remaining=0,
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
        )
    )
    assert decision.outcome is PreReconciliationTransitionOutcome.REJECTED
    assert decision.validation_errors == (UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,)
    assert decision.recovery_status is PreReconciliationRecoveryStatus.BUDGET_EXHAUSTED
    assert decision.recovery_attempted is False


def test_invalid_state_with_positive_budget_is_not_recoverable_post_validation() -> None:
    decision = decide_pre_reconciliation_transition(
        PreReconciliationTransitionState(
            validation_valid=False,
            validation_errors=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
            revision_budget_remaining=1,
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
        )
    )
    assert decision.outcome is PreReconciliationTransitionOutcome.REJECTED
    assert decision.recovery_status is (
        PreReconciliationRecoveryStatus.NOT_AVAILABLE_POST_VALIDATION
    )


def test_enforce_raises_typed_error_with_preserved_validation_errors() -> None:
    with pytest.raises(PreReconciliationValidationError) as exc_info:
        enforce_pre_reconciliation_validation_clean_transition(
            validation_valid=False,
            validation_errors=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
            revision_budget_remaining=0,
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
        )
    exc = exc_info.value
    assert exc.diagnostic.validation_errors == (UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,)
    assert exc.diagnostic.completion_mode == COMPLETION_UNRESOLVED
    assert exc.diagnostic.has_supported_diagnosis is True
    assert exc.diagnostic.recovery_status is PreReconciliationRecoveryStatus.BUDGET_EXHAUSTED
    assert exc.diagnostic.revision_budget_remaining == 0
    assert exc.diagnostic.recovery_attempted is False


def test_transition_decision_is_immutable() -> None:
    decision = PreReconciliationTransitionDecision(
        outcome=PreReconciliationTransitionOutcome.REJECTED,
        validation_errors=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
        recovery_status=PreReconciliationRecoveryStatus.BUDGET_EXHAUSTED,
        revision_budget_remaining=0,
        completion_mode=COMPLETION_UNRESOLVED,
        has_supported_diagnosis=True,
        recovery_attempted=False,
    )
    with pytest.raises(FrozenInstanceError):
        decision.recovery_attempted = True  # type: ignore[misc]


@pytest.mark.asyncio
async def test_invalid_final_validation_never_calls_reconciliation(monkeypatch) -> None:
    reconcile_calls = 0

    def _spy_reconcile(**kwargs: object) -> object:
        nonlocal reconcile_calls
        reconcile_calls += 1
        return reconcile_investigation_completion(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.application.scenario.reconcile_investigation_completion",
        _spy_reconcile,
    )

    original_validate = IncidentInvestigationValidationEngine.validate

    def _invalid_validate(
        self: IncidentInvestigationValidationEngine,
        execution: object,
        **kwargs: object,
    ) -> ValidationResult:
        result = original_validate(self, execution, **kwargs)  # type: ignore[arg-type]
        if result.valid:
            return ValidationResult(
                valid=False,
                errors=[UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR],
            )
        return result

    monkeypatch.setattr(IncidentInvestigationValidationEngine, "validate", _invalid_validate)

    bundle = build_fixture_runtime_bundle().bundle
    with pytest.raises(PreReconciliationValidationError):
        await execute_resolved_skeleton(bundle)
    assert reconcile_calls == 0


@pytest.mark.asyncio
async def test_valid_final_validation_calls_reconciliation_once(monkeypatch) -> None:
    reconcile_calls = 0
    original = reconcile_investigation_completion

    def _spy_reconcile(**kwargs: object) -> object:
        nonlocal reconcile_calls
        reconcile_calls += 1
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.application.scenario.reconcile_investigation_completion",
        _spy_reconcile,
    )

    bundle = build_fixture_runtime_bundle().bundle
    await execute_resolved_skeleton(bundle)
    assert reconcile_calls == 1
