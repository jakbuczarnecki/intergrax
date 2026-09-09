# © Artur Czarnecki. All rights reserved.

"""AI Incident single-run qualification execution for reliability measurement (DS-E2E-14.3b)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId, mint_run_id, validate_run_id
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.run_result import (
    DecisionQualificationRunResult,
    build_decision_qualification_run_result,
)
from intergrax.decision_system.qualification.taxonomy import DecisionFailureBoundary
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    ScenarioRuntimeComposition,
    build_scenario_environment_profile,
    resolve_scenario_llm_adapter,
    trace_reader_from_composition,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
    CompletionReconciliationError,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_transition import (
    PreReconciliationValidationError,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    ScenarioExecutionResult,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import ScenarioVariant
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import evaluate_scenario_run
from testing_support.decision_e2e.failure_observation_adapter import (
    EnvironmentQualificationFacts,
    observation_from_ai_incident_evaluation,
    observation_from_environment_facts,
    observation_from_platform_trace_readback,
    observation_from_scenario_execution_exception,
)
from testing_support.decision_e2e.provider_binding import bind_qualification_llm_profile
from testing_support.decision_e2e.scenario_qualification import (
    resolve_canonical_runtime_modules,
)
from testing_support.strict_tool_contract_validator import STRICT_CAPABILITY_BLOCK_REASON

CANONICAL_SCENARIO_INPUT_IDENTITY = "ai_incident_investigation:resolved:canonical"


@dataclass(frozen=True, slots=True)
class AiIncidentQualificationRunSignals:
    selected_tool_ids: tuple[str, ...]
    executed_tool_ids: tuple[str, ...]
    tool_invocation_count: int
    planner_round_count: int
    evidence_node_count: int
    initial_evidence_count: int
    follow_up_evidence_count: int
    evidence_gathering_stop_reason: str
    terminal_outcome: str | None
    model_completion_intent: str | None
    reconciliation_result: str | None
    critic_verdict_passed: bool | None
    evaluator_passed: bool
    evaluator_failures: tuple[str, ...]
    validation_error_categories: tuple[str, ...]
    strict_tool_capability: bool
    trace_readback_pass: bool
    trace_event_count: int
    route: str
    stop_reason: str | None
    reconciliation_error_reason: str | None
    reconciliation_model_intent: str | None
    reconciliation_has_supported_diagnosis: bool | None
    reconciliation_validation_errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AiIncidentQualificationRunOutcome:
    run_index: int
    valid_model_trial: bool
    environment_event: bool
    run_id: RunId | None
    signals: AiIncidentQualificationRunSignals | None
    run_result: DecisionQualificationRunResult | None
    block_reason: str | None


def _selected_tool_ids(planner_decisions: tuple[dict[str, object], ...]) -> tuple[str, ...]:
    selected: list[str] = []
    for decision in planner_decisions:
        raw = decision.get("selected_tool_ids") or []
        if isinstance(raw, list):
            selected.extend(str(item) for item in raw if item)
    return tuple(selected)


def _signals_from_result(
    *,
    result: ScenarioExecutionResult,
    evaluation_passed: bool,
    evaluator_failures: tuple[str, ...],
    strict_tool_capability: bool,
    trace_readback_pass: bool,
    trace_event_count: int,
    validation_error_categories: tuple[str, ...],
) -> AiIncidentQualificationRunSignals:
    initial_count = len(result.initial_evidence_nodes)
    total_count = len(result.evidence_nodes)
    planner_round_count = len(result.planner_decisions) or max(
        result.evaluator_loop_iterations,
        1 if result.tool_invocations > 0 else 0,
    )
    completion_mode = None
    if result.investigation_conclusion is not None:
        completion_mode = result.investigation_conclusion.status.value
    return AiIncidentQualificationRunSignals(
        selected_tool_ids=_selected_tool_ids(result.planner_decisions),
        executed_tool_ids=result.tool_execution_order,
        tool_invocation_count=result.tool_invocations,
        planner_round_count=planner_round_count,
        evidence_node_count=total_count,
        initial_evidence_count=initial_count,
        follow_up_evidence_count=max(total_count - initial_count, 0),
        evidence_gathering_stop_reason=result.evidence_gathering_stop_reason,
        terminal_outcome=result.outcome,
        model_completion_intent=completion_mode,
        reconciliation_result=result.outcome,
        critic_verdict_passed=result.critic_verdict_passed,
        evaluator_passed=evaluation_passed,
        evaluator_failures=evaluator_failures,
        validation_error_categories=validation_error_categories,
        strict_tool_capability=strict_tool_capability,
        trace_readback_pass=trace_readback_pass,
        trace_event_count=trace_event_count,
        route="unavailable",
        stop_reason=result.evidence_gathering_stop_reason or None,
        reconciliation_error_reason=None,
        reconciliation_model_intent=None,
        reconciliation_has_supported_diagnosis=None,
        reconciliation_validation_errors=(),
    )


def _signals_from_pre_reconciliation_error(
    exc: PreReconciliationValidationError,
    *,
    strict_tool_capability: bool,
) -> AiIncidentQualificationRunSignals:
    diagnostic = exc.diagnostic
    return AiIncidentQualificationRunSignals(
        selected_tool_ids=(),
        executed_tool_ids=(),
        tool_invocation_count=0,
        planner_round_count=0,
        evidence_node_count=0,
        initial_evidence_count=0,
        follow_up_evidence_count=0,
        evidence_gathering_stop_reason="",
        terminal_outcome=None,
        model_completion_intent=None,
        reconciliation_result=None,
        critic_verdict_passed=None,
        evaluator_passed=False,
        evaluator_failures=(),
        validation_error_categories=diagnostic.validation_errors,
        strict_tool_capability=strict_tool_capability,
        trace_readback_pass=False,
        trace_event_count=0,
        route="unavailable",
        stop_reason=None,
        reconciliation_error_reason=diagnostic.recovery_status.value,
        reconciliation_model_intent=diagnostic.completion_mode,
        reconciliation_has_supported_diagnosis=diagnostic.has_supported_diagnosis,
        reconciliation_validation_errors=diagnostic.validation_errors,
    )


def _signals_from_reconciliation_error(
    exc: CompletionReconciliationError,
    *,
    strict_tool_capability: bool,
) -> AiIncidentQualificationRunSignals:
    diagnostic = exc.diagnostic
    reconciliation_model_intent = None
    critic_verdict_passed = None
    reconciliation_has_supported_diagnosis = None
    reconciliation_validation_errors: tuple[str, ...] = ()
    evidence_gathering_stop_reason = ""
    if diagnostic is not None:
        reconciliation_model_intent = diagnostic.model_intent.value
        critic_verdict_passed = diagnostic.critic_verdict_passed
        reconciliation_has_supported_diagnosis = diagnostic.has_supported_diagnosis
        reconciliation_validation_errors = diagnostic.validation_errors
        evidence_gathering_stop_reason = diagnostic.evidence_gathering_stop_reason
    return AiIncidentQualificationRunSignals(
        selected_tool_ids=(),
        executed_tool_ids=(),
        tool_invocation_count=0,
        planner_round_count=0,
        evidence_node_count=0,
        initial_evidence_count=0,
        follow_up_evidence_count=0,
        evidence_gathering_stop_reason=evidence_gathering_stop_reason,
        terminal_outcome=None,
        model_completion_intent=None,
        reconciliation_result=None,
        critic_verdict_passed=critic_verdict_passed,
        evaluator_passed=False,
        evaluator_failures=(),
        validation_error_categories=reconciliation_validation_errors,
        strict_tool_capability=strict_tool_capability,
        trace_readback_pass=False,
        trace_event_count=0,
        route="unavailable",
        stop_reason=evidence_gathering_stop_reason or None,
        reconciliation_error_reason=exc.reason.value,
        reconciliation_model_intent=reconciliation_model_intent,
        reconciliation_has_supported_diagnosis=reconciliation_has_supported_diagnosis,
        reconciliation_validation_errors=reconciliation_validation_errors,
    )


def _resolve_latest_run_id(
    composition: ScenarioRuntimeComposition,
    tenant_id: str,
) -> RunId | None:
    reader = trace_reader_from_composition(composition)
    if reader is None:
        return None
    runs = reader.list_runs(tenant_id, limit=1)
    if not runs:
        return None
    return validate_run_id(str(runs[0].run_id))


def _trace_readback(
    composition: ScenarioRuntimeComposition,
    run_id: RunId,
    tenant_id: str,
) -> tuple[bool, int]:
    reader = trace_reader_from_composition(composition)
    if reader is None:
        return False, 0
    persisted = reader.read_run(str(run_id), tenant_id)
    events = [item for item in persisted.events if isinstance(item, dict)]
    return bool(events), len(events)


def _environment_facts_from_block_reason(block_reason: str) -> EnvironmentQualificationFacts:
    lowered = block_reason.lower()
    if "credential" in lowered or "api_key" in lowered:
        return EnvironmentQualificationFacts(credential_unavailable=True)
    if "qualification disabled" in lowered:
        return EnvironmentQualificationFacts(qualification_disabled=True)
    if "provider" in lowered and "configuration" in lowered:
        return EnvironmentQualificationFacts(provider_configuration_invalid=True)
    if "model" in lowered and ("configuration" in lowered or "invalid" in lowered):
        return EnvironmentQualificationFacts(model_configuration_invalid=True)
    if STRICT_CAPABILITY_BLOCK_REASON.lower() in lowered:
        return EnvironmentQualificationFacts(provider_configuration_invalid=True)
    return EnvironmentQualificationFacts(provider_configuration_invalid=True)


def _build_observation(
    *,
    trace_readback_pass: bool,
    evaluator_failures: tuple[str, ...],
    evaluator_passed: bool,
) -> DecisionQualificationObservation:
    if not trace_readback_pass:
        return observation_from_platform_trace_readback(trace_finalized=False)
    return observation_from_ai_incident_evaluation(
        failures=evaluator_failures,
        evaluator_passed=evaluator_passed,
        trace_finalized=True,
        boundary=DecisionFailureBoundary.HOST_EXECUTION,
    )


async def execute_ai_incident_qualification_run(
    *,
    run_index: int,
    variant: ScenarioVariant = ScenarioVariant.RESOLVED,
) -> AiIncidentQualificationRunOutcome:
    environment = build_scenario_environment_profile()
    binding, block_reason = bind_qualification_llm_profile(environment)
    if block_reason is not None or binding is None:
        facts = _environment_facts_from_block_reason(block_reason or "qualification binding failed")
        observation = observation_from_environment_facts(facts)
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
            block_reason=block_reason,
        )

    runtime_composition = ScenarioRuntimeComposition(
        environment=environment,
        tool_registry=ToolRegistry(),
    )
    fixture_bundle = build_fixture_runtime_bundle(
        variant=variant,
        runtime_composition=runtime_composition,
    )
    bundle = fixture_bundle.bundle
    composition = bundle.runtime_composition
    runtime_modules = resolve_canonical_runtime_modules(composition.platform)
    adapter = resolve_scenario_llm_adapter(composition.environment)
    strict_tool_capability = adapter.supports_strict_tool_argument_conformance()

    if not runtime_modules:
        observation = observation_from_platform_trace_readback(trace_finalized=False)
        return AiIncidentQualificationRunOutcome(
            run_index=run_index,
            valid_model_trial=False,
            environment_event=False,
            run_id=None,
            signals=None,
            run_result=build_decision_qualification_run_result(
                run_id=mint_run_id(),
                observation=observation,
                evaluator_passed=False,
            ),
            block_reason="Scenario runtime has no canonical Decision flow gate",
        )

    try:
        result = await execute_resolved_skeleton(bundle)
        evaluation = evaluate_scenario_run(result, fixture_bundle.fixture)
    except PreReconciliationValidationError as exc:
        observation = observation_from_scenario_execution_exception(exc)
        failed_run_id = mint_run_id()
        return AiIncidentQualificationRunOutcome(
            run_index=run_index,
            valid_model_trial=True,
            environment_event=False,
            run_id=failed_run_id,
            signals=_signals_from_pre_reconciliation_error(
                exc,
                strict_tool_capability=strict_tool_capability,
            ),
            run_result=build_decision_qualification_run_result(
                run_id=failed_run_id,
                observation=observation,
                evaluator_passed=False,
            ),
            block_reason=f"{type(exc).__name__}: {exc}",
        )
    except CompletionReconciliationError as exc:
        observation = observation_from_scenario_execution_exception(exc)
        failed_run_id = mint_run_id()
        return AiIncidentQualificationRunOutcome(
            run_index=run_index,
            valid_model_trial=True,
            environment_event=False,
            run_id=failed_run_id,
            signals=_signals_from_reconciliation_error(
                exc,
                strict_tool_capability=strict_tool_capability,
            ),
            run_result=build_decision_qualification_run_result(
                run_id=failed_run_id,
                observation=observation,
                evaluator_passed=False,
            ),
            block_reason=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        observation = observation_from_scenario_execution_exception(exc)
        failed_run_id = mint_run_id()
        return AiIncidentQualificationRunOutcome(
            run_index=run_index,
            valid_model_trial=True,
            environment_event=False,
            run_id=failed_run_id,
            signals=None,
            run_result=build_decision_qualification_run_result(
                run_id=failed_run_id,
                observation=observation,
                evaluator_passed=False,
            ),
            block_reason=f"{type(exc).__name__}: {exc}",
        )

    run_id = _resolve_latest_run_id(composition, result.execution_tenant_id)
    trace_pass = False
    trace_count = 0
    if run_id is not None:
        trace_pass, trace_count = _trace_readback(
            composition,
            run_id,
            result.execution_tenant_id,
        )

    evaluator_failures = tuple(evaluation.failures)
    validation_errors = tuple(
        failure.split(":", 1)[0] for failure in evaluator_failures if ":" in failure
    )
    observation = _build_observation(
        trace_readback_pass=trace_pass,
        evaluator_failures=evaluator_failures,
        evaluator_passed=evaluation.passed,
    )
    signals = _signals_from_result(
        result=result,
        evaluation_passed=evaluation.passed,
        evaluator_failures=evaluator_failures,
        strict_tool_capability=strict_tool_capability,
        trace_readback_pass=trace_pass,
        trace_event_count=trace_count,
        validation_error_categories=validation_errors,
    )
    effective_run_id = run_id or mint_run_id()
    run_result = build_decision_qualification_run_result(
        run_id=effective_run_id,
        observation=observation,
        evaluator_passed=evaluation.passed,
    )
    return AiIncidentQualificationRunOutcome(
        run_index=run_index,
        valid_model_trial=True,
        environment_event=False,
        run_id=effective_run_id,
        signals=signals,
        run_result=run_result,
        block_reason=None if evaluation.passed else "; ".join(evaluator_failures),
    )
