# © Artur Czarnecki. All rights reserved.

"""Harness unit tests for DS-E2E-14.3b reliability qualification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.decision_system.qualification.reliability import aggregate_decision_reliability
from intergrax.decision_system.qualification.axis_outcome import DecisionQualificationAxisOutcome
from intergrax.decision_system.qualification.run_result import (
    DecisionQualificationRunResult,
    build_decision_qualification_run_result,
)
from intergrax.decision_system.qualification.taxonomy import DecisionFailureCategory
from testing_support.decision_e2e.ai_incident_qualification_run import (
    AiIncidentQualificationRunOutcome,
    AiIncidentQualificationRunSignals,
)
from testing_support.decision_e2e.env_bootstrap import QualificationEnvBootstrapReport
from testing_support.decision_e2e.failure_observation_adapter import (
    observation_for_provider_rate_limit,
    observation_from_ai_incident_evaluation,
)
from testing_support.decision_e2e.reliability_qualification import (
    CallableDecisionQualificationRunExecutor,
    DecisionReliabilityQualificationPlan,
    DecisionReliabilityQualificationRunRecord,
    QualificationSessionIntegrityError,
    execute_reliability_qualification,
    validate_run_result_consistency,
)
from testing_support.decision_e2e.reliability_reporting import (
    qualification_result_to_summary_dict,
    write_qualification_artifacts,
)


def _signals(**overrides: object) -> AiIncidentQualificationRunSignals:
    base = {
        "selected_tool_ids": ("production.telemetry.read",),
        "executed_tool_ids": ("production.telemetry.read",),
        "tool_invocation_count": 1,
        "planner_round_count": 1,
        "evidence_node_count": 2,
        "initial_evidence_count": 1,
        "follow_up_evidence_count": 1,
        "evidence_gathering_stop_reason": "complete",
        "terminal_outcome": "UNRESOLVED",
        "model_completion_intent": "unresolved",
        "reconciliation_result": "UNRESOLVED",
        "critic_verdict_passed": True,
        "evaluator_passed": False,
        "evaluator_failures": ("tool_runtime_not_exercised",),
        "validation_error_categories": ("tool_runtime_not_exercised",),
        "strict_tool_capability": True,
        "trace_readback_pass": True,
        "trace_event_count": 3,
        "route": "unavailable",
        "stop_reason": "complete",
    }
    base.update(overrides)
    return AiIncidentQualificationRunSignals(**base)


def _model_fail_outcome(run_index: int) -> AiIncidentQualificationRunOutcome:
    observation = observation_from_ai_incident_evaluation(
        failures=("tool_runtime_not_exercised",),
        evaluator_passed=False,
    )
    run_id = mint_run_id()
    return AiIncidentQualificationRunOutcome(
        run_index=run_index,
        valid_model_trial=True,
        environment_event=False,
        run_id=run_id,
        signals=_signals(),
        run_result=build_decision_qualification_run_result(
            run_id=run_id,
            observation=observation,
            evaluator_passed=False,
        ),
        block_reason="tool_runtime_not_exercised",
    )


def _provider_fail_outcome(run_index: int) -> AiIncidentQualificationRunOutcome:
    observation = observation_for_provider_rate_limit()
    run_id = mint_run_id()
    return AiIncidentQualificationRunOutcome(
        run_index=run_index,
        valid_model_trial=True,
        environment_event=False,
        run_id=run_id,
        signals=None,
        run_result=build_decision_qualification_run_result(
            run_id=run_id,
            observation=observation,
            evaluator_passed=False,
        ),
        block_reason="rate limit",
    )


def _env_fail_outcome(run_index: int) -> AiIncidentQualificationRunOutcome:
    from testing_support.decision_e2e.failure_observation_adapter import (
        observation_for_credential_unavailable,
    )

    observation = observation_for_credential_unavailable()
    return AiIncidentQualificationRunOutcome(
        run_index=run_index,
        valid_model_trial=False,
        environment_event=True,
        run_id=None,
        signals=None,
        run_result=build_decision_qualification_run_result(
            run_id=mint_run_id(),
            observation=observation,
            evaluator_passed=False,
        ),
        block_reason="credential unavailable",
    )


@pytest.mark.asyncio
async def test_exact_run_count_without_replacement() -> None:
    outcomes = [_model_fail_outcome(index) for index in range(5)]
    executor = CallableDecisionQualificationRunExecutor(
        _callable=lambda run_index: _async_return(outcomes[run_index]),
    )
    plan = DecisionReliabilityQualificationPlan(
        run_count=5,
        provider_id="openai",
        model_id="gpt-4.1",
        scenario_id="ai_incident_investigation",
        scenario_input_identity="ai_incident_investigation:resolved:canonical",
    )
    result = await execute_reliability_qualification(
        plan,
        executor,
        git_sha="test-sha",
        env_bootstrap=_bootstrap(),
    )
    assert result.completed_run_count == 5
    assert result.summary.total_runs == 5
    assert result.summary.model_pass_count == 0
    assert result.summary.model_failure_count == 5


@pytest.mark.asyncio
async def test_provider_failure_separated_from_model_failure() -> None:
    outcomes = (
        _model_fail_outcome(0),
        _provider_fail_outcome(1),
        _model_fail_outcome(2),
    )
    executor = CallableDecisionQualificationRunExecutor(
        _callable=lambda run_index: _async_return(outcomes[run_index]),
    )
    plan = DecisionReliabilityQualificationPlan(
        run_count=3,
        provider_id="openai",
        model_id="gpt-4.1",
        scenario_id="ai_incident_investigation",
        scenario_input_identity="ai_incident_investigation:resolved:canonical",
    )
    result = await execute_reliability_qualification(plan, executor, git_sha="test-sha")
    assert result.summary.model_failure_count == 2
    assert result.summary.provider_infra_failure_count == 1


@pytest.mark.asyncio
async def test_environment_failure_not_counted_as_model_trial() -> None:
    outcomes = (_env_fail_outcome(0), _model_fail_outcome(1))
    executor = CallableDecisionQualificationRunExecutor(
        _callable=lambda run_index: _async_return(outcomes[run_index]),
    )
    plan = DecisionReliabilityQualificationPlan(
        run_count=2,
        provider_id="openai",
        model_id="gpt-4.1",
        scenario_id="ai_incident_investigation",
        scenario_input_identity="ai_incident_investigation:resolved:canonical",
    )
    result = await execute_reliability_qualification(plan, executor, git_sha="test-sha")
    assert result.environment_failure_count == 1
    assert result.valid_model_trial_count == 1
    assert result.summary.total_runs == 1
    assert result.session_complete is False


@pytest.mark.asyncio
async def test_aggregation_uses_canonical_aggregator() -> None:
    outcomes = [_model_fail_outcome(index) for index in range(3)]
    executor = CallableDecisionQualificationRunExecutor(
        _callable=lambda run_index: _async_return(outcomes[run_index]),
    )
    plan = DecisionReliabilityQualificationPlan(
        run_count=3,
        provider_id="openai",
        model_id="gpt-4.1",
        scenario_id="ai_incident_investigation",
        scenario_input_identity="ai_incident_investigation:resolved:canonical",
    )
    result = await execute_reliability_qualification(plan, executor, git_sha="test-sha")
    run_results = tuple(
        record.run_result for record in result.runs if record.run_result is not None
    )
    assert result.summary == aggregate_decision_reliability(run_results)


@pytest.mark.asyncio
async def test_artifact_serialization_and_provenance(tmp_path: Path) -> None:
    outcomes = [_model_fail_outcome(index) for index in range(2)]
    executor = CallableDecisionQualificationRunExecutor(
        _callable=lambda run_index: _async_return(outcomes[run_index]),
    )
    plan = DecisionReliabilityQualificationPlan(
        run_count=2,
        provider_id="openai",
        model_id="gpt-4.1",
        scenario_id="ai_incident_investigation",
        scenario_input_identity="ai_incident_investigation:resolved:canonical",
    )
    result = await execute_reliability_qualification(
        plan,
        executor,
        git_sha="artifact-sha",
        qualification_id="fixture-session",
        env_bootstrap=_bootstrap(),
    )
    runs_path, summary_path, report_path = write_qualification_artifacts(result, tmp_path)
    assert runs_path.is_file()
    assert summary_path.is_file()
    assert report_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["qualification_id"] == "fixture-session"
    assert summary["git_sha"] == "artifact-sha"
    assert summary["provenance"]["env_bootstrap"]["credential_available"] is True


def test_consistency_rule_rejects_contradiction() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=("tool_runtime_not_exercised",),
        evaluator_passed=False,
    )
    run_result = build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation,
        evaluator_passed=False,
    )
    record = DecisionReliabilityQualificationRunRecord(
        run_index=0,
        run_id=run_result.run_id,
        valid_model_trial=True,
        environment_event=False,
        completed=True,
        run_result=run_result,
        signals=_signals(),
        block_reason=None,
    )
    validate_run_result_consistency(record)
    broken = DecisionReliabilityQualificationRunRecord(
        run_index=0,
        run_id=run_result.run_id,
        valid_model_trial=True,
        environment_event=False,
        completed=True,
        run_result=DecisionQualificationRunResult(
            run_id=run_result.run_id,
            classification=run_result.classification,
            platform_outcome=DecisionQualificationAxisOutcome.PASS,
            model_outcome=DecisionQualificationAxisOutcome.PASS,
            evaluator_outcome=DecisionQualificationAxisOutcome.FAIL,
            evaluator_passed=False,
        ),
        signals=_signals(),
        block_reason=None,
    )
    with pytest.raises(QualificationSessionIntegrityError):
        validate_run_result_consistency(broken)


def test_summary_contains_evaluator_fail_count() -> None:
    outcomes = [_model_fail_outcome(0)]
    result = qualification_result_to_summary_dict(
        _result_from_outcomes(outcomes, run_count=1),
    )
    assert result["evaluator_fail_count"] == 1
    assert result["reliability"]["model_evaluable_count"] == 1


async def _async_return(value: AiIncidentQualificationRunOutcome) -> AiIncidentQualificationRunOutcome:
    return value


def _bootstrap() -> QualificationEnvBootstrapReport:
    return QualificationEnvBootstrapReport(
        dotenv_discovered="/tmp/.env",
        dotenv_loaded=True,
        provider="openai",
        model="gpt-4.1",
        qualification_enabled=True,
        credential_available=True,
    )


def _result_from_outcomes(
    outcomes: tuple[AiIncidentQualificationRunOutcome, ...],
    *,
    run_count: int,
):
    import asyncio

    executor = CallableDecisionQualificationRunExecutor(
        _callable=lambda run_index: _async_return(outcomes[run_index]),
    )
    plan = DecisionReliabilityQualificationPlan(
        run_count=run_count,
        provider_id="openai",
        model_id="gpt-4.1",
        scenario_id="ai_incident_investigation",
        scenario_input_identity="ai_incident_investigation:resolved:canonical",
    )
    return asyncio.run(
        execute_reliability_qualification(
            plan,
            executor,
            git_sha="test-sha",
            env_bootstrap=_bootstrap(),
        )
    )


def test_provider_failure_category() -> None:
    outcome = _provider_fail_outcome(0)
    assert outcome.run_result is not None
    assert outcome.run_result.classification is not None
    assert (
        outcome.run_result.classification.category
        is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
    )
