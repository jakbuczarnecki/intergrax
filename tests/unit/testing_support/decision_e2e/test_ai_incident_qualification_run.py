# © Artur Czarnecki. All rights reserved.

"""Qualification run exception-path diagnostic capture tests (DS-E2E-15D.2)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureCategory,
    DecisionFailureReason,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
    CompletionReconciliationError,
    CompletionReconciliationFailureReason,
    reconcile_investigation_completion,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
)
from testing_support.decision_e2e.ai_incident_qualification_run import (
    _signals_from_reconciliation_error,
    execute_ai_incident_qualification_run,
)
from testing_support.decision_e2e.failure_observation_adapter import (
    observation_from_scenario_execution_exception,
)
from testing_support.decision_e2e.env_bootstrap import QualificationEnvBootstrapReport
from testing_support.decision_e2e.reliability_qualification import (
    CallableDecisionQualificationRunExecutor,
    DecisionReliabilityQualificationPlan,
    execute_reliability_qualification,
)
from testing_support.decision_e2e.reliability_reporting import write_qualification_artifacts


def _validation_error_exception() -> CompletionReconciliationError:
    try:
        reconcile_investigation_completion(
            model_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            validation_errors=("staffing_attendance_not_gathered",),
            evidence_gathering_stop_reason="planner_final_answer",
        )
    except CompletionReconciliationError as exc:
        return exc
    raise AssertionError("expected CompletionReconciliationError")


def test_signals_from_reconciliation_error_preserves_diagnostic_fields() -> None:
    exc = _validation_error_exception()
    signals = _signals_from_reconciliation_error(exc, strict_tool_capability=True)
    assert signals.reconciliation_error_reason == (
        "validation_errors_present_during_reconciliation"
    )
    assert signals.reconciliation_model_intent == CompletionIntent.SUPPORTED_DIAGNOSIS.value
    assert signals.critic_verdict_passed is True
    assert signals.reconciliation_has_supported_diagnosis is True
    assert signals.reconciliation_validation_errors == ("staffing_attendance_not_gathered",)
    assert signals.evidence_gathering_stop_reason == "planner_final_answer"
    assert signals.stop_reason == "planner_final_answer"


def test_reconciliation_exception_signals_classify_as_unsupported_completion() -> None:
    exc = _validation_error_exception()
    signals = _signals_from_reconciliation_error(exc, strict_tool_capability=True)
    assert signals is not None
    observation = observation_from_scenario_execution_exception(exc)
    classification = classify_decision_failure(observation)
    assert classification is not None
    assert classification.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert classification.reason is DecisionFailureReason.UNSUPPORTED_COMPLETION


@pytest.mark.asyncio
async def test_execute_qualification_run_preserves_reconciliation_signals() -> None:
    exc = _validation_error_exception()
    binding = object()
    composition = MagicMock()
    composition.platform = "test-platform"
    bundle = MagicMock()
    bundle.bundle.runtime_composition = composition
    bundle.fixture = MagicMock()

    with (
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.bind_qualification_llm_profile",
            return_value=(binding, None),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.build_fixture_runtime_bundle",
            return_value=bundle,
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.resolve_canonical_runtime_modules",
            return_value=("decision_flow",),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.resolve_scenario_llm_adapter",
        ) as adapter_patch,
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.execute_resolved_skeleton",
            new=AsyncMock(side_effect=exc),
        ),
    ):
        adapter = adapter_patch.return_value
        adapter.supports_strict_tool_argument_conformance.return_value = True
        outcome = await execute_ai_incident_qualification_run(run_index=0)

    assert outcome.signals is not None
    assert outcome.valid_model_trial is True
    assert outcome.signals.reconciliation_error_reason == exc.reason.value
    assert outcome.signals.reconciliation_validation_errors == ("staffing_attendance_not_gathered",)
    assert outcome.run_result is not None
    assert outcome.run_result.classification is not None
    assert outcome.run_result.classification.reason is DecisionFailureReason.UNSUPPORTED_COMPLETION


@pytest.mark.asyncio
async def test_reconciliation_failure_serialization_emits_diagnostic_fields(
    tmp_path: Path,
) -> None:
    exc = _validation_error_exception()
    from intergrax.contracts.execution_identity import mint_run_id
    from intergrax.decision_system.qualification.run_result import (
        build_decision_qualification_run_result,
    )
    from testing_support.decision_e2e.ai_incident_qualification_run import (
        AiIncidentQualificationRunOutcome,
    )
    signals = _signals_from_reconciliation_error(exc, strict_tool_capability=True)
    run_id = mint_run_id()
    outcome = AiIncidentQualificationRunOutcome(
        run_index=0,
        valid_model_trial=True,
        environment_event=False,
        run_id=run_id,
        signals=signals,
        run_result=build_decision_qualification_run_result(
            run_id=run_id,
            observation=observation_from_scenario_execution_exception(exc),
            evaluator_passed=False,
        ),
        block_reason=str(exc),
    )
    executor = CallableDecisionQualificationRunExecutor(
        _callable=lambda run_index: _async_return(outcome),
    )
    plan = DecisionReliabilityQualificationPlan(
        run_count=1,
        provider_id="openai",
        model_id="gpt-4.1",
        scenario_id="ai_incident_investigation",
        scenario_input_identity="ai_incident_investigation:resolved:canonical",
    )
    result = await execute_reliability_qualification(
        plan,
        executor,
        git_sha="artifact-sha",
        qualification_id="reconciliation-diagnostic",
        env_bootstrap=QualificationEnvBootstrapReport(
            dotenv_discovered="/tmp/.env",
            dotenv_loaded=True,
            provider="openai",
            model="gpt-4.1",
            qualification_enabled=True,
            credential_available=True,
        ),
    )
    runs_path, _, _ = write_qualification_artifacts(result, tmp_path)
    payload = json.loads(runs_path.read_text(encoding="utf-8"))
    signals_payload = payload["runs"][0]["signals"]
    assert signals_payload["reconciliation_error_reason"] == (
        CompletionReconciliationFailureReason.VALIDATION_ERRORS_PRESENT.value
    )
    assert signals_payload["reconciliation_validation_errors"] == [
        "staffing_attendance_not_gathered",
    ]
    assert signals_payload["reconciliation_model_intent"] == CompletionIntent.SUPPORTED_DIAGNOSIS.value
    assert signals_payload["reconciliation_has_supported_diagnosis"] is True
    assert signals_payload["critic_verdict_passed"] is True
    assert signals_payload["evidence_gathering_stop_reason"] == "planner_final_answer"


async def _async_return(value: object) -> object:
    return value
