# © Artur Czarnecki. All rights reserved.

"""Typed reconciliation of model completion intent with validated investigation state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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


class CompletionReconciliationError(Exception):
    """Raised when model completion intent cannot be reconciled with validated state."""


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


def normalize_evidence_gathering_stop_reason(raw: str) -> ToolInvocationStopReason:
    """Normalize persisted domain payload stop reason at reconciliation boundary."""
    if raw not in _KNOWN_EVIDENCE_GATHERING_STOP_REASONS:
        raise CompletionReconciliationError("unknown_evidence_gathering_stop_reason")
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
    raise CompletionReconciliationError(f"unknown completion_mode: {completion_mode!r}")


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
        raise CompletionReconciliationError(
            "validation_errors_present_during_reconciliation"
        )
    if not critic_verdict_passed:
        raise CompletionReconciliationError("critic_verdict_not_passed")

    if model_intent is CompletionIntent.SUPPORTED_DIAGNOSIS:
        if not has_supported_diagnosis:
            raise CompletionReconciliationError("supported_intent_without_supported_state")
        return ReconciledCompletion(
            model_intent=model_intent,
            completion_mode=CompletionMode.SUPPORTED_DIAGNOSIS,
            reason=CompletionReconciliationReason.ALIGNED_SUPPORTED_DIAGNOSIS,
        )

    if model_intent is CompletionIntent.UNRESOLVED:
        if has_supported_diagnosis:
            raise CompletionReconciliationError("unresolved_intent_with_supported_state")
        return ReconciledCompletion(
            model_intent=model_intent,
            completion_mode=CompletionMode.UNRESOLVED,
            reason=CompletionReconciliationReason.ALIGNED_UNRESOLVED,
        )

    if model_intent is CompletionIntent.NEED_MORE_EVIDENCE:
        if evidence_gathering_stop_reason == "max_iterations":
            raise CompletionReconciliationError("evidence_gathering_safety_limit_reached")
        if not is_reconcilable_evidence_termination(evidence_gathering_stop_reason):
            raise CompletionReconciliationError("evidence_gathering_not_terminal")
        if has_supported_diagnosis:
            return ReconciledCompletion(
                model_intent=model_intent,
                completion_mode=CompletionMode.SUPPORTED_DIAGNOSIS,
                reason=(
                    CompletionReconciliationReason
                    .VALIDATED_SUPPORTED_OVERRIDES_STALE_NEED_MORE_EVIDENCE
                ),
            )
        raise CompletionReconciliationError("need_more_evidence_without_supported_state")

    raise CompletionReconciliationError("unknown_model_intent")
