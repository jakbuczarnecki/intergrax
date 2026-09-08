# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationStopReason
from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
    CompletionReconciliationError,
    CompletionReconciliationReason,
    reconcile_investigation_completion,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    derive_terminal_outcome,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    CompletionMode,
)

pytestmark = pytest.mark.unit


def _reconcile(
    *,
    model_intent: CompletionIntent,
    critic_verdict_passed: bool = True,
    has_supported_diagnosis: bool = True,
    validation_errors: tuple[str, ...] = (),
    evidence_gathering_stop_reason: ToolInvocationStopReason = "planner_final_answer",
):
    return reconcile_investigation_completion(
        model_intent=model_intent,
        critic_verdict_passed=critic_verdict_passed,
        has_supported_diagnosis=has_supported_diagnosis,
        validation_errors=validation_errors,
        evidence_gathering_stop_reason=evidence_gathering_stop_reason,
    )


def test_aligned_supported_diagnosis() -> None:
    reconciled = _reconcile(
        model_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
        has_supported_diagnosis=True,
    )
    assert reconciled.completion_mode is CompletionMode.SUPPORTED_DIAGNOSIS
    assert reconciled.reason is CompletionReconciliationReason.ALIGNED_SUPPORTED_DIAGNOSIS


def test_aligned_unresolved() -> None:
    reconciled = _reconcile(
        model_intent=CompletionIntent.UNRESOLVED,
        has_supported_diagnosis=False,
    )
    assert reconciled.completion_mode is CompletionMode.UNRESOLVED
    assert reconciled.reason is CompletionReconciliationReason.ALIGNED_UNRESOLVED


def test_t4_stale_need_more_evidence_reconciles_to_supported() -> None:
    reconciled = _reconcile(
        model_intent=CompletionIntent.NEED_MORE_EVIDENCE,
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        validation_errors=(),
        evidence_gathering_stop_reason="planner_final_answer",
    )
    assert reconciled.model_intent is CompletionIntent.NEED_MORE_EVIDENCE
    assert reconciled.completion_mode is CompletionMode.SUPPORTED_DIAGNOSIS
    assert reconciled.reason is (
        CompletionReconciliationReason.VALIDATED_SUPPORTED_OVERRIDES_STALE_NEED_MORE_EVIDENCE
    )


def test_t4_max_iterations_fails_closed() -> None:
    with pytest.raises(
        CompletionReconciliationError,
        match="evidence_gathering_safety_limit_reached",
    ):
        _reconcile(
            model_intent=CompletionIntent.NEED_MORE_EVIDENCE,
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            validation_errors=(),
            evidence_gathering_stop_reason="max_iterations",
        )


def test_empty_tool_calls_not_reconcilable_for_t4() -> None:
    with pytest.raises(CompletionReconciliationError, match="evidence_gathering_not_terminal"):
        _reconcile(
            model_intent=CompletionIntent.NEED_MORE_EVIDENCE,
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            validation_errors=(),
            evidence_gathering_stop_reason="empty_tool_calls",
        )


def test_legacy_single_pass_not_reconcilable_for_t4() -> None:
    with pytest.raises(CompletionReconciliationError, match="evidence_gathering_not_terminal"):
        _reconcile(
            model_intent=CompletionIntent.NEED_MORE_EVIDENCE,
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            validation_errors=(),
            evidence_gathering_stop_reason="legacy_single_pass",
        )


def test_supported_intent_without_supported_state_fails() -> None:
    with pytest.raises(CompletionReconciliationError, match="supported_intent_without_supported_state"):
        _reconcile(
            model_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
            has_supported_diagnosis=False,
        )


def test_unresolved_intent_with_supported_state_fails() -> None:
    with pytest.raises(CompletionReconciliationError, match="unresolved_intent_with_supported_state"):
        _reconcile(
            model_intent=CompletionIntent.UNRESOLVED,
            has_supported_diagnosis=True,
        )


def test_critic_fail_fails() -> None:
    with pytest.raises(CompletionReconciliationError, match="critic_verdict_not_passed"):
        _reconcile(
            model_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
            critic_verdict_passed=False,
        )


def test_need_more_without_supported_state_fails() -> None:
    with pytest.raises(
        CompletionReconciliationError,
        match="need_more_evidence_without_supported_state",
    ):
        _reconcile(
            model_intent=CompletionIntent.NEED_MORE_EVIDENCE,
            has_supported_diagnosis=False,
        )


def test_reconciliation_is_deterministic() -> None:
    kwargs = {
        "model_intent": CompletionIntent.NEED_MORE_EVIDENCE,
        "critic_verdict_passed": True,
        "has_supported_diagnosis": True,
        "validation_errors": (),
        "evidence_gathering_stop_reason": "planner_final_answer",
    }
    first = reconcile_investigation_completion(**kwargs)
    second = reconcile_investigation_completion(**kwargs)
    assert first == second


def test_t4_reconciled_state_passes_terminal_gate() -> None:
    reconciled = _reconcile(
        model_intent=CompletionIntent.NEED_MORE_EVIDENCE,
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        validation_errors=(),
        evidence_gathering_stop_reason="planner_final_answer",
    )
    outcome = derive_terminal_outcome(
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        completion_mode=reconciled.completion_mode.value,
    )
    assert outcome == OUTCOME_RESOLVED


@pytest.mark.parametrize(
    ("model_intent", "has_supported_diagnosis", "stop_reason", "expected_mode", "expected_reason"),
    [
        (
            CompletionIntent.SUPPORTED_DIAGNOSIS,
            True,
            "planner_final_answer",
            CompletionMode.SUPPORTED_DIAGNOSIS,
            CompletionReconciliationReason.ALIGNED_SUPPORTED_DIAGNOSIS,
        ),
        (
            CompletionIntent.UNRESOLVED,
            False,
            "planner_final_answer",
            CompletionMode.UNRESOLVED,
            CompletionReconciliationReason.ALIGNED_UNRESOLVED,
        ),
        (
            CompletionIntent.NEED_MORE_EVIDENCE,
            True,
            "planner_final_answer",
            CompletionMode.SUPPORTED_DIAGNOSIS,
            CompletionReconciliationReason.VALIDATED_SUPPORTED_OVERRIDES_STALE_NEED_MORE_EVIDENCE,
        ),
    ],
)
def test_truth_table_aligned_and_t4_cases(
    model_intent: CompletionIntent,
    has_supported_diagnosis: bool,
    stop_reason: ToolInvocationStopReason,
    expected_mode: CompletionMode,
    expected_reason: CompletionReconciliationReason,
) -> None:
    reconciled = _reconcile(
        model_intent=model_intent,
        has_supported_diagnosis=has_supported_diagnosis,
        evidence_gathering_stop_reason=stop_reason,
    )
    assert reconciled.completion_mode is expected_mode
    assert reconciled.reason is expected_reason
