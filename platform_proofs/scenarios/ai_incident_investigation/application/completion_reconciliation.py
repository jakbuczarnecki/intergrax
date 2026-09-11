# © Artur Czarnecki. All rights reserved.

"""Typed reconciliation of model completion intent with validated investigation state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from platform_proofs.scenarios.ai_incident_investigation.application.scenario_execution_provenance import (
    ScenarioExecutionProvenance,
)

from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationStopReason
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_NEED_MORE_EVIDENCE,
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
    CompletionMode,
)

_KNOWN_EVIDENCE_GATHERING_STOP_REASONS: frozenset[ToolInvocationStopReason] = frozenset(
    {
        "empty_tool_calls",
        "max_iterations",
        "planner_final_answer",
        "legacy_single_pass",
    }
)

_RECONCILABLE_EVIDENCE_GATHERING_STOP_REASONS: frozenset[ToolInvocationStopReason] = frozenset(
    {
        "planner_final_answer",
    }
)


class CompletionReconciliationFailureReason(StrEnum):
    UNKNOWN_EVIDENCE_GATHERING_STOP_REASON = "unknown_evidence_gathering_stop_reason"
    UNKNOWN_COMPLETION_MODE = "unknown_completion_mode"
    VALIDATION_ERRORS_PRESENT = "validation_errors_present_during_reconciliation"
    CRITIC_VERDICT_NOT_PASSED = "critic_verdict_not_passed"
    SUPPORTED_INTENT_WITHOUT_SUPPORTED_STATE = "supported_intent_without_supported_state"
    UNRESOLVED_INTENT_WITH_SUPPORTED_STATE = "unresolved_intent_with_supported_state"
    EVIDENCE_GATHERING_SAFETY_LIMIT_REACHED = "evidence_gathering_safety_limit_reached"
    EVIDENCE_GATHERING_NOT_TERMINAL = "evidence_gathering_not_terminal"
    NEED_MORE_EVIDENCE_WITHOUT_SUPPORTED_STATE = "need_more_evidence_without_supported_state"
    UNKNOWN_MODEL_INTENT = "unknown_model_intent"


@dataclass(frozen=True, slots=True)
class CompletionReconciliationDiagnostic:
    model_intent: CompletionIntent
    critic_verdict_passed: bool
    has_supported_diagnosis: bool
    validation_errors: tuple[str, ...]
    evidence_gathering_stop_reason: ToolInvocationStopReason


class CompletionReconciliationError(Exception):
    """Raised when model completion intent cannot be reconciled with validated state."""

    execution_provenance: ScenarioExecutionProvenance | None

    def __init__(
        self,
        reason: CompletionReconciliationFailureReason,
        *,
        diagnostic: CompletionReconciliationDiagnostic | None = None,
        detail: str | None = None,
    ) -> None:
        self.reason = reason
        self.diagnostic = diagnostic
        self._detail = detail
        self.execution_provenance = None
        super().__init__(self._message())

    def _message(self) -> str:
        if self._detail is not None:
            return self._detail
        return self.reason.value


class CompletionReconciliationReason(StrEnum):
    ALIGNED_SUPPORTED_DIAGNOSIS = "aligned_supported_diagnosis"
    ALIGNED_UNRESOLVED = "aligned_unresolved"
    VALIDATED_SUPPORTED_OVERRIDES_STALE_NEED_MORE_EVIDENCE = (
        "validated_supported_overrides_stale_need_more_evidence"
    )


@dataclass(frozen=True, slots=True)
class ReconciledCompletion:
    model_intent: CompletionIntent
    completion_mode: CompletionMode
    reason: CompletionReconciliationReason


def _raise_reconciliation_failure(
    reason: CompletionReconciliationFailureReason,
    *,
    model_intent: CompletionIntent,
    critic_verdict_passed: bool,
    has_supported_diagnosis: bool,
    validation_errors: tuple[str, ...],
    evidence_gathering_stop_reason: ToolInvocationStopReason,
) -> NoReturn:
    raise CompletionReconciliationError(
        reason,
        diagnostic=CompletionReconciliationDiagnostic(
            model_intent=model_intent,
            critic_verdict_passed=critic_verdict_passed,
            has_supported_diagnosis=has_supported_diagnosis,
            validation_errors=validation_errors,
            evidence_gathering_stop_reason=evidence_gathering_stop_reason,
        ),
    )


def normalize_evidence_gathering_stop_reason(raw: str) -> ToolInvocationStopReason:
    """Normalize persisted domain payload stop reason at reconciliation boundary."""
    if raw not in _KNOWN_EVIDENCE_GATHERING_STOP_REASONS:
        raise CompletionReconciliationError(
            CompletionReconciliationFailureReason.UNKNOWN_EVIDENCE_GATHERING_STOP_REASON,
        )
    return raw


def is_reconcilable_evidence_termination(
    stop_reason: ToolInvocationStopReason,
) -> bool:
    """Whether evidence gathering ended in a way that permits completion reconciliation."""
    return stop_reason in _RECONCILABLE_EVIDENCE_GATHERING_STOP_REASONS


def completion_intent_from_completion_mode(completion_mode: str) -> CompletionIntent:
    if completion_mode == COMPLETION_SUPPORTED_DIAGNOSIS:
        return CompletionIntent.SUPPORTED_DIAGNOSIS
    if completion_mode == COMPLETION_UNRESOLVED:
        return CompletionIntent.UNRESOLVED
    if completion_mode == COMPLETION_NEED_MORE_EVIDENCE:
        return CompletionIntent.NEED_MORE_EVIDENCE
    raise CompletionReconciliationError(
        CompletionReconciliationFailureReason.UNKNOWN_COMPLETION_MODE,
        detail=f"unknown completion_mode: {completion_mode!r}",
    )


def reconcile_investigation_completion(
    *,
    model_intent: CompletionIntent,
    critic_verdict_passed: bool,
    has_supported_diagnosis: bool,
    validation_errors: tuple[str, ...],
    evidence_gathering_stop_reason: ToolInvocationStopReason,
) -> ReconciledCompletion:
    """Reconcile model completion intent with validated investigation state."""
    if validation_errors:
        _raise_reconciliation_failure(
            CompletionReconciliationFailureReason.VALIDATION_ERRORS_PRESENT,
            model_intent=model_intent,
            critic_verdict_passed=critic_verdict_passed,
            has_supported_diagnosis=has_supported_diagnosis,
            validation_errors=validation_errors,
            evidence_gathering_stop_reason=evidence_gathering_stop_reason,
        )
    if not critic_verdict_passed:
        _raise_reconciliation_failure(
            CompletionReconciliationFailureReason.CRITIC_VERDICT_NOT_PASSED,
            model_intent=model_intent,
            critic_verdict_passed=critic_verdict_passed,
            has_supported_diagnosis=has_supported_diagnosis,
            validation_errors=validation_errors,
            evidence_gathering_stop_reason=evidence_gathering_stop_reason,
        )

    if model_intent is CompletionIntent.SUPPORTED_DIAGNOSIS:
        if not has_supported_diagnosis:
            _raise_reconciliation_failure(
                CompletionReconciliationFailureReason.SUPPORTED_INTENT_WITHOUT_SUPPORTED_STATE,
                model_intent=model_intent,
                critic_verdict_passed=critic_verdict_passed,
                has_supported_diagnosis=has_supported_diagnosis,
                validation_errors=validation_errors,
                evidence_gathering_stop_reason=evidence_gathering_stop_reason,
            )
        return ReconciledCompletion(
            model_intent=model_intent,
            completion_mode=CompletionMode.SUPPORTED_DIAGNOSIS,
            reason=CompletionReconciliationReason.ALIGNED_SUPPORTED_DIAGNOSIS,
        )

    if model_intent is CompletionIntent.UNRESOLVED:
        if has_supported_diagnosis:
            _raise_reconciliation_failure(
                CompletionReconciliationFailureReason.UNRESOLVED_INTENT_WITH_SUPPORTED_STATE,
                model_intent=model_intent,
                critic_verdict_passed=critic_verdict_passed,
                has_supported_diagnosis=has_supported_diagnosis,
                validation_errors=validation_errors,
                evidence_gathering_stop_reason=evidence_gathering_stop_reason,
            )
        return ReconciledCompletion(
            model_intent=model_intent,
            completion_mode=CompletionMode.UNRESOLVED,
            reason=CompletionReconciliationReason.ALIGNED_UNRESOLVED,
        )

    if model_intent is CompletionIntent.NEED_MORE_EVIDENCE:
        if evidence_gathering_stop_reason == "max_iterations":
            _raise_reconciliation_failure(
                CompletionReconciliationFailureReason.EVIDENCE_GATHERING_SAFETY_LIMIT_REACHED,
                model_intent=model_intent,
                critic_verdict_passed=critic_verdict_passed,
                has_supported_diagnosis=has_supported_diagnosis,
                validation_errors=validation_errors,
                evidence_gathering_stop_reason=evidence_gathering_stop_reason,
            )
        if not is_reconcilable_evidence_termination(evidence_gathering_stop_reason):
            _raise_reconciliation_failure(
                CompletionReconciliationFailureReason.EVIDENCE_GATHERING_NOT_TERMINAL,
                model_intent=model_intent,
                critic_verdict_passed=critic_verdict_passed,
                has_supported_diagnosis=has_supported_diagnosis,
                validation_errors=validation_errors,
                evidence_gathering_stop_reason=evidence_gathering_stop_reason,
            )
        if has_supported_diagnosis:
            return ReconciledCompletion(
                model_intent=model_intent,
                completion_mode=CompletionMode.SUPPORTED_DIAGNOSIS,
                reason=(
                    CompletionReconciliationReason
                    .VALIDATED_SUPPORTED_OVERRIDES_STALE_NEED_MORE_EVIDENCE
                ),
            )
        _raise_reconciliation_failure(
            CompletionReconciliationFailureReason.NEED_MORE_EVIDENCE_WITHOUT_SUPPORTED_STATE,
            model_intent=model_intent,
            critic_verdict_passed=critic_verdict_passed,
            has_supported_diagnosis=has_supported_diagnosis,
            validation_errors=validation_errors,
            evidence_gathering_stop_reason=evidence_gathering_stop_reason,
        )

    _raise_reconciliation_failure(
        CompletionReconciliationFailureReason.UNKNOWN_MODEL_INTENT,
        model_intent=model_intent,
        critic_verdict_passed=critic_verdict_passed,
        has_supported_diagnosis=has_supported_diagnosis,
        validation_errors=validation_errors,
        evidence_gathering_stop_reason=evidence_gathering_stop_reason,
    )
