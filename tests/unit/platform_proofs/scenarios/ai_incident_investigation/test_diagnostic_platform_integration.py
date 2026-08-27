# © Artur Czarnecki. All rights reserved.

"""DIAG-8C — platform-attached ai_incident_investigation integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessmentBuilder,
    DiagnosticFindingKind,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationRequest,
    DiagnosticSignalSubjectScope,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_read_models import DiagnosticOccurrenceReadStatus
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.in_memory_problem_persistence import InMemoryProblemPersistence
from intergrax.runtime.diagnostics.investigation_contracts import (
    IncidentInvestigationIntegrityError,
    InvestigationConclusionStatus,
    incident_investigation_input_from_problem_details,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingAssessmentInput,
    ProblemGroupingEngine,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    ProblemLifecycleEngine,
    ProblemStatus,
    mint_problem_id,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_SOURCE_LAYER_APPLICATION,
    PlatformProblemSignal,
)
from platform_proofs.scenarios.ai_incident_investigation.incident_reasoning import (
    PriorInvestigationState,
    build_reasoning_messages,
)
from platform_proofs.scenarios.ai_incident_investigation.platform_diagnostic_context import (
    format_platform_diagnostic_context_lines,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import (
    execute_resolved_skeleton,
    investigation_conclusion_status_from_outcome,
    OUTCOME_RESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario_composition import (
    IncidentInvestigationProblemNotFoundError,
    build_runtime_bundle_from_diagnostic_problem,
    resolve_incident_investigation_input,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_APP_ID = "app-demo"
_OBSERVED_AT = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)
_OBSERVED_AT_LATER = _OBSERVED_AT + timedelta(hours=1)


def _grouping_engine() -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return ProblemGroupingEngine(registry)


def _assess_attempt_sequence(
    event_types: list[RuntimeEventType],
    *,
    tenant_id: str = _TENANT_A,
    runtime_store: InMemoryRuntimeEventStore | None = None,
) -> ProblemGroupingAssessmentInput:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = runtime_store or InMemoryRuntimeEventStore()
    for event_type in event_types:
        event = sample_runtime_event(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ).model_copy(update={"event_type": event_type})
        runtime_store.append(event, tenant_id=tenant_id)

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(tenant_id, task_id, run_id)
    lifecycle = LifecycleAnomalyAnalyzer().analyze(reconstruction)
    assessment = DiagnosticAssessmentBuilder().assess(reconstruction, lifecycle)
    return ProblemGroupingAssessmentInput(assessment=assessment)


def _assess_retry_pair(
    *,
    tenant_id: str = _TENANT_A,
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
    runtime_store: InMemoryRuntimeEventStore | None = None,
) -> tuple[ProblemGroupingAssessmentInput, ProblemGroupingAssessmentInput]:
    sequence = [
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_COMPLETED,
        violating_event_type,
    ]
    return (
        _assess_attempt_sequence(sequence, tenant_id=tenant_id, runtime_store=runtime_store),
        _assess_attempt_sequence(sequence, tenant_id=tenant_id, runtime_store=runtime_store),
    )


def _persist_execution_problem(
    *,
    tenant_id: str = _TENANT_A,
    persistence: InMemoryProblemPersistence | None = None,
    runtime_store: InMemoryRuntimeEventStore | None = None,
) -> tuple[object, InMemoryProblemPersistence, InMemoryRuntimeEventStore]:
    persistence = persistence or InMemoryProblemPersistence()
    runtime_store = runtime_store or InMemoryRuntimeEventStore()
    lifecycle = ProblemLifecycleEngine(persistence)
    grouping = _grouping_engine().group(
        _assess_retry_pair(tenant_id=tenant_id, runtime_store=runtime_store),
        strategy_id=STRATEGY_ID,
    )
    result = lifecycle.reconcile(grouping, observed_at=_OBSERVED_AT)
    problem = result.created[0]
    return problem, persistence, runtime_store


def _read_service(
    persistence: InMemoryProblemPersistence,
    runtime_store: InMemoryRuntimeEventStore | None = None,
) -> DiagnosticReadService:
    return DiagnosticReadService(
        problem_persistence=persistence,
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=runtime_store or InMemoryRuntimeEventStore(),
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )


def _startup_failure_signal() -> PlatformProblemSignal:
    return PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_APPLICATION,
        source_component="startup",
        status="detected",
        error_code="startup_failed",
        exception_type="ApplicationStartupError",
        safe_message="startup failed",
    )


def _build_orchestrator(
    persistence: InMemoryProblemPersistence | None = None,
) -> tuple[DiagnosticOrchestrator, InMemoryProblemPersistence]:
    persistence = persistence or InMemoryProblemPersistence()
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=InMemoryRuntimeEventStore(),
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=ProblemGroupingEngine(registry),
        problem_lifecycle_engine=ProblemLifecycleEngine(persistence),
    )
    return orchestrator, persistence


def _persist_application_problem(
    *,
    tenant_id: str = _TENANT_A,
    instance_id: str = "instance-i1",
) -> tuple[object, InMemoryProblemPersistence]:
    orchestrator, persistence = _build_orchestrator()
    scope = DiagnosticSignalSubjectScope(
        tenant_id=tenant_id,
        application_id=_APP_ID,
        instance_id=instance_id,
        problem_signals=(_startup_failure_signal(),),
    )
    request = DiagnosticOrchestrationRequest(
        tenant_id=tenant_id,
        grouping_strategy_id=STRATEGY_ID,
        observed_at=_OBSERVED_AT,
        signal_subjects=(scope,),
    )
    result = orchestrator.run(request)
    problem = result.lifecycle_result.created[0]
    return problem, persistence


@pytest.mark.asyncio
async def test_platform_attached_execution_problem_e2e() -> None:
    problem, persistence, runtime_store = _persist_execution_problem()
    read_service = _read_service(persistence, runtime_store)
    before_record = persistence.get(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    bundle = build_runtime_bundle_from_diagnostic_problem(
        read_service,
        tenant_id=_TENANT_A,
        problem_ids=problem.problem_id,
    )
    assert bundle.investigation_input is not None
    assert bundle.investigation_input.tenant_id == _TENANT_A

    detail = read_service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)
    assert detail is not None
    occurrence = detail.occurrences[0]
    assert occurrence.read_status is DiagnosticOccurrenceReadStatus.AVAILABLE
    assert occurrence.assessment is not None
    assert occurrence.assessment.findings

    context_lines = format_platform_diagnostic_context_lines(bundle.investigation_input)
    assert any("finding=" in line for line in context_lines)

    messages = build_reasoning_messages(
        evidence_nodes=(),
        prior_state=PriorInvestigationState(
            evidence_nodes=(),
            reasoning_proposal=None,
            claim_set=None,
            claim_hypothesis_bindings=(),
            completion_intent=None,
            summary="",
        ),
        critic_feedback=None,
        is_revision=False,
        investigation_input=bundle.investigation_input,
    )
    prompt = messages[0].content or ""
    assert "Platform diagnostic starting context" in prompt
    assert occurrence.assessment.findings[0].kind.value in prompt

    result = await execute_resolved_skeleton(bundle)
    assert result.execution_tenant_id == _TENANT_A
    assert result.investigated_problem_ids == (problem.problem_id,)
    assert result.investigation_conclusion is not None
    assert result.investigation_conclusion.investigated_problem_ids == (problem.problem_id,)
    assert result.investigation_conclusion.status is InvestigationConclusionStatus.SUPPORTED
    assert result.tool_invocations >= 3
    assert result.outcome == OUTCOME_RESOLVED

    after_record = persistence.get(tenant_id=_TENANT_A, problem_id=problem.problem_id)
    assert after_record == before_record
    assert after_record.status is ProblemStatus.OPEN


def test_resolve_input_unknown_problem_fails_explicitly() -> None:
    persistence = InMemoryProblemPersistence()
    read_service = _read_service(persistence)

    with pytest.raises(IncidentInvestigationProblemNotFoundError, match="incident_investigation_problem_not_found"):
        resolve_incident_investigation_input(
            read_service,
            tenant_id=_TENANT_A,
            problem_ids=mint_problem_id(),
        )


def test_tenant_mismatch_fails_before_scenario_execution() -> None:
    problem, persistence, runtime_store = _persist_execution_problem(tenant_id=_TENANT_A)
    read_service = _read_service(persistence, runtime_store)

    with pytest.raises(IncidentInvestigationProblemNotFoundError):
        resolve_incident_investigation_input(
            read_service,
            tenant_id=_TENANT_B,
            problem_ids=problem.problem_id,
        )


@pytest.mark.asyncio
async def test_application_instance_non_execution_subject_starts_investigation() -> None:
    problem, persistence = _persist_application_problem()
    read_service = _read_service(persistence)

    bundle = build_runtime_bundle_from_diagnostic_problem(
        read_service,
        tenant_id=_TENANT_A,
        problem_ids=problem.problem_id,
    )
    occurrence = bundle.investigation_input.problem_contexts[0].occurrences[0]
    assert occurrence.unavailable_reason is not None
    assert occurrence.read_status is DiagnosticOccurrenceReadStatus.UNAVAILABLE

    result = await execute_resolved_skeleton(bundle)
    assert result.execution_tenant_id == _TENANT_A
    assert result.investigated_problem_ids == (problem.problem_id,)
    assert result.tool_invocations >= 3
    assert result.investigation_conclusion is not None


def test_limitation_survives_into_reasoning_context() -> None:
    problem, persistence, runtime_store = _persist_execution_problem()
    read_service = _read_service(persistence, runtime_store)
    detail = read_service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)
    assert detail is not None

    investigation_input = incident_investigation_input_from_problem_details(
        tenant_id=_TENANT_A,
        details=(detail,),
    )
    lines = format_platform_diagnostic_context_lines(investigation_input)
    assert any(DiagnosticFindingKind.EVENT_AFTER_TERMINAL.value in line for line in lines)

    messages = build_reasoning_messages(
        evidence_nodes=(),
        prior_state=PriorInvestigationState(
            evidence_nodes=(),
            reasoning_proposal=None,
            claim_set=None,
            claim_hypothesis_bindings=(),
            completion_intent=None,
            summary="",
        ),
        critic_feedback=None,
        is_revision=False,
        investigation_input=investigation_input,
    )
    content = messages[0].content or ""
    assert DiagnosticFindingKind.EVENT_AFTER_TERMINAL.value in content


def test_limitation_kind_rendered_in_platform_context_boundary() -> None:
    """Boundary: limitation lines appear when central assessment carries DiagnosticLimitation."""
    from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticLimitation
    from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyKind

    problem, persistence, runtime_store = _persist_execution_problem()
    read_service = _read_service(persistence, runtime_store)
    detail = read_service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)
    assert detail is not None
    occurrence = detail.occurrences[0]
    assert occurrence.assessment is not None
    limited_assessment = occurrence.assessment.__class__(
        tenant_id=occurrence.assessment.tenant_id,
        task_id=occurrence.assessment.task_id,
        run_id=occurrence.assessment.run_id,
        findings=occurrence.assessment.findings,
        limitations=(
            DiagnosticLimitation(
                kind=DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED,
                factual_message="runtime history truncated for assessment",
                source_anomaly_kind=LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED,
                supporting_event_ids=(),
                supporting_evidence_ids=(),
                supporting_positions=(),
            ),
        ),
    )
    patched_occurrence = occurrence.__class__(
        subject_ref=occurrence.subject_ref,
        observed_at=occurrence.observed_at,
        strategy_id=occurrence.strategy_id,
        strategy_version=occurrence.strategy_version,
        method=occurrence.method,
        read_status=occurrence.read_status,
        assessment=limited_assessment,
        unavailable_reason=occurrence.unavailable_reason,
    )
    patched_detail = detail.__class__(
        problem_id=detail.problem_id,
        tenant_id=detail.tenant_id,
        status=detail.status,
        first_seen_at=detail.first_seen_at,
        last_seen_at=detail.last_seen_at,
        occurrence_count=detail.occurrence_count,
        record_version=detail.record_version,
        grouping_provenance=detail.grouping_provenance,
        occurrences=(patched_occurrence,),
        returned_occurrence_count=1,
        total_occurrence_count=detail.total_occurrence_count,
        is_occurrences_truncated=detail.is_occurrences_truncated,
    )
    investigation_input = incident_investigation_input_from_problem_details(
        tenant_id=_TENANT_A,
        details=(patched_detail,),
    )
    lines = format_platform_diagnostic_context_lines(investigation_input)
    assert any(
        DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED.value in line for line in lines
    )


def test_investigation_execution_identity_distinct_from_investigated_occurrence() -> None:
    problem, persistence, runtime_store = _persist_execution_problem()
    read_service = _read_service(persistence, runtime_store)
    detail = read_service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)
    assert detail is not None
    investigated_occurrence = detail.occurrences[0]
    investigated_subject = investigated_occurrence.subject_ref
    assert investigated_subject.task_id is not None
    assert investigated_subject.run_id is not None

    bundle = build_runtime_bundle_from_diagnostic_problem(
        read_service,
        tenant_id=_TENANT_A,
        problem_ids=problem.problem_id,
    )
    assert bundle.investigation_input is not None


def test_outcome_to_conclusion_status_mapping() -> None:
    assert investigation_conclusion_status_from_outcome(OUTCOME_RESOLVED) is (
        InvestigationConclusionStatus.SUPPORTED
    )
    assert investigation_conclusion_status_from_outcome("UNRESOLVED") is (
        InvestigationConclusionStatus.UNRESOLVED
    )
    assert investigation_conclusion_status_from_outcome("NOT_ACCEPTED") is (
        InvestigationConclusionStatus.NOT_ACCEPTED
    )
