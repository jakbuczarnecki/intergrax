# © Artur Czarnecki. All rights reserved.

"""HOST-DIAG-2 — typed non-execution diagnostic subject orchestration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationIntegrityError,
    DiagnosticOrchestrationRequest,
    DiagnosticSignalSubjectScope,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingEngine,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE,
    PROBLEM_SEVERITY_CRITICAL,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_SOURCE_LAYER_APPLICATION,
    PlatformProblemSignal,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_APP_ID = "app-demo"
_OBSERVED_AT = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
_OBSERVED_AT_LATER = _OBSERVED_AT + timedelta(hours=1)


def _startup_failure_signal(**updates) -> PlatformProblemSignal:
    base = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_APPLICATION,
        source_component="startup",
        status="detected",
        error_code="startup_failed",
        exception_type="ApplicationStartupError",
        safe_message="startup failed",
    )
    if updates:
        return base.model_copy(update=updates)
    return base


def _different_failure_signal() -> PlatformProblemSignal:
    return PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE,
        severity=PROBLEM_SEVERITY_CRITICAL,
        source_layer=PROBLEM_SOURCE_LAYER_APPLICATION,
        source_component="startup",
        status="detected",
        error_code="config_invalid",
        exception_type="ConfigurationError",
        safe_message="invalid config",
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


def _signal_scope(
    instance_id: str,
    *,
    tenant_id: str = _TENANT_A,
    application_id: str = _APP_ID,
    problem_signals: tuple[PlatformProblemSignal, ...] | None = None,
) -> DiagnosticSignalSubjectScope:
    return DiagnosticSignalSubjectScope(
        tenant_id=tenant_id,
        application_id=application_id,
        instance_id=instance_id,
        problem_signals=problem_signals or (_startup_failure_signal(),),
    )


def _signal_request(
    *scopes: DiagnosticSignalSubjectScope,
    observed_at: datetime = _OBSERVED_AT,
    tenant_id: str = _TENANT_A,
) -> DiagnosticOrchestrationRequest:
    return DiagnosticOrchestrationRequest(
        tenant_id=tenant_id,
        grouping_strategy_id=STRATEGY_ID,
        observed_at=observed_at,
        signal_subjects=scopes,
    )


def test_application_subject_first_occurrence() -> None:
    orchestrator, persistence = _build_orchestrator()
    result = orchestrator.run(_signal_request(_signal_scope("instance-i1")))

    assert result.execution_results == ()
    assert len(result.signal_subject_results) == 1
    assert result.signal_subject_results[0].application_id == _APP_ID
    assert result.signal_subject_results[0].instance_id == "instance-i1"
    assert result.signal_subject_results[0].assessment.has_findings
    assert len(result.lifecycle_result.created) == 1
    assert result.lifecycle_result.created[0].occurrence_count == 1
    assert persistence.list_for_tenant(_TENANT_A) == (result.lifecycle_result.created[0],)


def test_recurrence_across_instance_ids() -> None:
    orchestrator, _ = _build_orchestrator()
    first = orchestrator.run(
        _signal_request(_signal_scope("instance-i1")),
    )
    problem_id = first.lifecycle_result.created[0].problem_id

    second = orchestrator.run(
        _signal_request(
            _signal_scope("instance-i1"),
            _signal_scope("instance-i2"),
            observed_at=_OBSERVED_AT_LATER,
        ),
    )

    assert second.lifecycle_result.created == ()
    assert len(second.lifecycle_result.updated) == 1
    updated = second.lifecycle_result.updated[0]
    assert updated.problem_id == problem_id
    assert updated.occurrence_count == 2


def test_replay_same_instance_does_not_increase_occurrence_count() -> None:
    orchestrator, _ = _build_orchestrator()
    orchestrator.run(_signal_request(_signal_scope("instance-i1"), _signal_scope("instance-i2")))
    replay = orchestrator.run(
        _signal_request(
            _signal_scope("instance-i1"),
            _signal_scope("instance-i2"),
            observed_at=_OBSERVED_AT_LATER,
        ),
    )

    assert replay.lifecycle_result.created == ()
    assert replay.lifecycle_result.updated == ()
    assert len(replay.lifecycle_result.unchanged) == 1
    assert replay.lifecycle_result.unchanged[0].occurrence_count == 2


def test_tenant_isolation_for_application_subjects() -> None:
    orchestrator, persistence = _build_orchestrator()
    tenant_a = orchestrator.run(
        _signal_request(_signal_scope("instance-i1", tenant_id=_TENANT_A)),
    )
    tenant_b = orchestrator.run(
        _signal_request(
            _signal_scope("instance-i1", tenant_id=_TENANT_B),
            tenant_id=_TENANT_B,
        ),
    )

    problem_a = tenant_a.lifecycle_result.created[0].problem_id
    problem_b = tenant_b.lifecycle_result.created[0].problem_id
    assert problem_a != problem_b
    assert len(persistence.list_for_tenant(_TENANT_A)) == 1
    assert len(persistence.list_for_tenant(_TENANT_B)) == 1


def test_different_signature_isolation() -> None:
    orchestrator, _ = _build_orchestrator()
    first = orchestrator.run(
        _signal_request(
            _signal_scope(
                "instance-i1",
                problem_signals=(_startup_failure_signal(),),
            ),
        ),
    )
    second = orchestrator.run(
        _signal_request(
            _signal_scope(
                "instance-i2",
                problem_signals=(_different_failure_signal(),),
            ),
            observed_at=_OBSERVED_AT_LATER,
        ),
    )

    assert first.lifecycle_result.created[0].problem_id != second.lifecycle_result.created[0].problem_id


def test_signal_only_request_without_executions() -> None:
    orchestrator, _ = _build_orchestrator()
    result = orchestrator.run(_signal_request(_signal_scope("instance-i1")))
    assert result.execution_results == ()
    assert len(result.signal_subject_results) == 1


def test_empty_subject_inputs_rejected() -> None:
    orchestrator, _ = _build_orchestrator()
    with pytest.raises(DiagnosticOrchestrationIntegrityError):
        orchestrator.run(
            DiagnosticOrchestrationRequest(
                tenant_id=_TENANT_A,
                grouping_strategy_id=STRATEGY_ID,
                observed_at=_OBSERVED_AT,
            ),
        )


def test_application_subject_does_not_synthesize_task_run_ids() -> None:
    orchestrator, _ = _build_orchestrator()
    result = orchestrator.run(_signal_request(_signal_scope("instance-i1")))
    subject_ref = result.lifecycle_result.created[0].current_subject_refs[0]
    assert subject_ref.execution() is None
    app_ref = subject_ref.application_instance()
    assert app_ref is not None
    assert app_ref.application_id == _APP_ID
    assert app_ref.instance_id == "instance-i1"
