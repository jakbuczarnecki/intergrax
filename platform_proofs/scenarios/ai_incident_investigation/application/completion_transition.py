# © Artur Czarnecki. All rights reserved.

"""Pre-reconciliation validation-clean transition gate for AI Incident completion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from platform_proofs.scenarios.ai_incident_investigation.application.scenario_execution_provenance import (
    ScenarioExecutionProvenance,
)


class PreReconciliationTransitionOutcome(StrEnum):
    READY_FOR_RECONCILIATION = "ready_for_reconciliation"
    REJECTED = "rejected"


class PreReconciliationRecoveryStatus(StrEnum):
    BUDGET_EXHAUSTED = "budget_exhausted"
    NOT_AVAILABLE_POST_VALIDATION = "not_available_post_validation"


@dataclass(frozen=True, slots=True)
class PreReconciliationTransitionState:
    validation_valid: bool
    validation_errors: tuple[str, ...]
    revision_budget_remaining: int
    completion_mode: str
    has_supported_diagnosis: bool


@dataclass(frozen=True, slots=True)
class PreReconciliationTransitionDecision:
    outcome: PreReconciliationTransitionOutcome
    validation_errors: tuple[str, ...]
    recovery_status: PreReconciliationRecoveryStatus | None
    revision_budget_remaining: int
    completion_mode: str
    has_supported_diagnosis: bool
    recovery_attempted: bool = False


@dataclass(frozen=True, slots=True)
class PreReconciliationValidationDiagnostic:
    validation_errors: tuple[str, ...]
    completion_mode: str
    has_supported_diagnosis: bool
    recovery_status: PreReconciliationRecoveryStatus
    revision_budget_remaining: int
    recovery_attempted: bool


class PreReconciliationValidationError(Exception):
    """Raised when final validation is not clean before completion reconciliation."""

    execution_provenance: ScenarioExecutionProvenance | None

    def __init__(self, decision: PreReconciliationTransitionDecision) -> None:
        if decision.outcome is not PreReconciliationTransitionOutcome.REJECTED:
            raise ValueError("PreReconciliationValidationError requires a rejected transition decision")
        if decision.recovery_status is None:
            raise ValueError("PreReconciliationValidationError requires recovery_status")
        self.decision = decision
        self.diagnostic = PreReconciliationValidationDiagnostic(
            validation_errors=decision.validation_errors,
            completion_mode=decision.completion_mode,
            has_supported_diagnosis=decision.has_supported_diagnosis,
            recovery_status=decision.recovery_status,
            revision_budget_remaining=decision.revision_budget_remaining,
            recovery_attempted=decision.recovery_attempted,
        )
        self.execution_provenance = None
        super().__init__(_rejection_message(decision))


def _rejection_message(decision: PreReconciliationTransitionDecision) -> str:
    return (
        "pre_reconciliation_validation_failed:"
        f"{decision.recovery_status.value if decision.recovery_status is not None else 'unknown'}"
    )


def decide_pre_reconciliation_transition(
    state: PreReconciliationTransitionState,
) -> PreReconciliationTransitionDecision:
    """Decide whether validated completion state may enter reconciliation."""
    if state.validation_valid:
        return PreReconciliationTransitionDecision(
            outcome=PreReconciliationTransitionOutcome.READY_FOR_RECONCILIATION,
            validation_errors=(),
            recovery_status=None,
            revision_budget_remaining=state.revision_budget_remaining,
            completion_mode=state.completion_mode,
            has_supported_diagnosis=state.has_supported_diagnosis,
            recovery_attempted=False,
        )

    recovery_status = (
        PreReconciliationRecoveryStatus.BUDGET_EXHAUSTED
        if state.revision_budget_remaining <= 0
        else PreReconciliationRecoveryStatus.NOT_AVAILABLE_POST_VALIDATION
    )
    return PreReconciliationTransitionDecision(
        outcome=PreReconciliationTransitionOutcome.REJECTED,
        validation_errors=state.validation_errors,
        recovery_status=recovery_status,
        revision_budget_remaining=state.revision_budget_remaining,
        completion_mode=state.completion_mode,
        has_supported_diagnosis=state.has_supported_diagnosis,
        recovery_attempted=False,
    )


def enforce_pre_reconciliation_validation_clean_transition(
    *,
    validation_valid: bool,
    validation_errors: tuple[str, ...],
    revision_budget_remaining: int,
    completion_mode: str,
    has_supported_diagnosis: bool,
) -> PreReconciliationTransitionDecision:
    """Apply the validation-clean transition gate or raise a typed rejection."""
    decision = decide_pre_reconciliation_transition(
        PreReconciliationTransitionState(
            validation_valid=validation_valid,
            validation_errors=validation_errors,
            revision_budget_remaining=revision_budget_remaining,
            completion_mode=completion_mode,
            has_supported_diagnosis=has_supported_diagnosis,
        )
    )
    if decision.outcome is PreReconciliationTransitionOutcome.REJECTED:
        _raise_pre_reconciliation_validation_error(decision)
    return decision


def _raise_pre_reconciliation_validation_error(
    decision: PreReconciliationTransitionDecision,
) -> NoReturn:
    raise PreReconciliationValidationError(decision)
