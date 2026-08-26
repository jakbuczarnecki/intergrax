# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import UTC, datetime, timedelta
from typing import get_args, get_origin
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessmentBuilder,
    DiagnosticAssessmentIntegrityError,
)
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticExecutionScope,
    DiagnosticOrchestrationIntegrityError,
    DiagnosticOrchestrationRequest,
    DiagnosticOrchestrationResult,
    MAX_DIAGNOSTIC_ORCHESTRATION_EXECUTIONS,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_problem_grouping_feature_projector import (
    DiagnosticProblemGroupingFeatureProjector,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
    RuntimeHistoryCompleteness,
)
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingEngine,
    ProblemGroupingIntegrityError,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    ProblemLifecycleEngine,
    ProblemLifecycleIntegrityError,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_SOURCE_LAYER_TOOL,
    PlatformProblemSignal,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_OBSERVED_AT = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
_OBSERVED_AT_LATER = _OBSERVED_AT + timedelta(hours=1)

_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "payload",
        "runtime_events",
        "positioned_events",
        "raw_log",
        "traceback",
        "prompt",
        "document_content",
        "root_cause",
        "likely_root_cause",
        "confidence",
    }
)

_RETRY_SEQUENCE = [
    RuntimeEventType.TASK_CREATED,
    RuntimeEventType.TASK_COMPLETED,
    RuntimeEventType.RETRY_SCHEDULED,
]


def _seed_retry_violation_sequence(
    runtime_store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str = _TENANT_A,
    task_id=None,
    run_id=None,
) -> tuple:
    task_id = task_id or mint_task_id()
    run_id = run_id or mint_run_id()
    attempt_id = mint_attempt_id()
    for event_type in _RETRY_SEQUENCE:
        event = sample_runtime_event(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ).model_copy(update={"event_type": event_type})
        runtime_store.append(event, tenant_id=tenant_id)
    return task_id, run_id


def _build_grouping_engine() -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return ProblemGroupingEngine(
        registry,
        feature_projector=DiagnosticProblemGroupingFeatureProjector(),
    )


def _build_orchestrator(
    *,
    runtime_store: InMemoryRuntimeEventStore | None = None,
    causal_store: InMemoryCausalEvidencePersistence | None = None,
    persistence: InMemoryProblemPersistence | None = None,
    execution_reconstructor: ExecutionReconstructor | None = None,
    grouping_engine: ProblemGroupingEngine | None = None,
    problem_lifecycle_engine: ProblemLifecycleEngine | None = None,
) -> tuple[
    DiagnosticOrchestrator,
    InMemoryRuntimeEventStore,
    InMemoryCausalEvidencePersistence,
    InMemoryProblemPersistence,
]:
    runtime_store = runtime_store or InMemoryRuntimeEventStore()
    causal_store = causal_store or InMemoryCausalEvidencePersistence()
    persistence = persistence or InMemoryProblemPersistence()
    reconstructor = execution_reconstructor or ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
    )
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=reconstructor,
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=grouping_engine or _build_grouping_engine(),
        problem_lifecycle_engine=problem_lifecycle_engine
        or ProblemLifecycleEngine(persistence),
    )
    return orchestrator, runtime_store, causal_store, persistence


def _scope(
    task_id,
    run_id,
    *,
    tenant_id: str = _TENANT_A,
    problem_signals: tuple[PlatformProblemSignal, ...] = (),
) -> DiagnosticExecutionScope:
    return DiagnosticExecutionScope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        problem_signals=problem_signals,
    )


def _request(
    *executions: DiagnosticExecutionScope,
    observed_at: datetime = _OBSERVED_AT,
    tenant_id: str = _TENANT_A,
) -> DiagnosticOrchestrationRequest:
    return DiagnosticOrchestrationRequest(
        tenant_id=tenant_id,
        executions=executions,
        grouping_strategy_id=STRATEGY_ID,
        observed_at=observed_at,
    )


def _collect_forbidden_field_names(value: object, *, seen: set[int]) -> set[str]:
    identity = id(value)
    if identity in seen:
        return set()
    seen.add(identity)

    found: set[str] = set()
    if is_dataclass(value):
        for field in fields(value):
            if field.name in _FORBIDDEN_FIELD_NAMES:
                found.add(field.name)
            found.update(_collect_forbidden_field_names(getattr(value, field.name), seen=seen))
        return found

    origin = get_origin(value)
    if origin is tuple:
        for item in value:
            found.update(_collect_forbidden_field_names(item, seen=seen))
        return found

    if origin is list:
        for item in value:
            found.update(_collect_forbidden_field_names(item, seen=seen))
        return found

    return found


def test_happy_path_two_matching_executions_create_one_problem() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)

    result = orchestrator.run(
        _request(
            _scope(first_task, first_run),
            _scope(second_task, second_run),
        ),
    )

    assert len(result.execution_results) == 2
    assert len(result.grouping_result.candidates) == 1
    assert len(result.lifecycle_result.created) == 1
    assert result.lifecycle_result.created[0].occurrence_count == 2


def test_same_request_twice_is_idempotent() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)
    request = _request(_scope(first_task, first_run), _scope(second_task, second_run))

    first = orchestrator.run(request)
    second = orchestrator.run(request)

    problem_id = first.lifecycle_result.created[0].problem_id
    assert second.lifecycle_result.created == ()
    assert second.lifecycle_result.updated == ()
    assert len(second.lifecycle_result.unchanged) == 1
    assert second.lifecycle_result.unchanged[0].problem_id == problem_id
    assert second.lifecycle_result.unchanged[0].occurrence_count == 2


def test_later_run_adds_matching_execution_increments_count() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)
    third_task, third_run = _seed_retry_violation_sequence(runtime_store)

    first = orchestrator.run(
        _request(_scope(first_task, first_run), _scope(second_task, second_run)),
    )
    problem_id = first.lifecycle_result.created[0].problem_id

    second = orchestrator.run(
        _request(
            _scope(first_task, first_run),
            _scope(second_task, second_run),
            _scope(third_task, third_run),
            observed_at=_OBSERVED_AT_LATER,
        ),
    )

    assert second.lifecycle_result.created == ()
    assert len(second.lifecycle_result.updated) == 1
    assert second.lifecycle_result.updated[0].problem_id == problem_id
    assert second.lifecycle_result.updated[0].occurrence_count == 3


def test_single_execution_with_findings_creates_singleton_problem() -> None:
    orchestrator, runtime_store, _, persistence = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)

    result = orchestrator.run(_request(_scope(task_id, run_id)))

    assert len(result.execution_results) == 1
    assert result.execution_results[0].assessment.has_findings
    assert len(result.grouping_result.candidates) == 1
    assert len(result.lifecycle_result.created) == 1
    assert result.lifecycle_result.created[0].occurrence_count == 1
    assert persistence.list_for_tenant("tenant-a") == (result.lifecycle_result.created)


def test_mixed_tenant_fails_before_reconstruction() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    spy = MagicMock(
        spec=ExecutionReconstructor,
        wraps=ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=causal_store,
        ),
    )
    orchestrator, _, _, _ = _build_orchestrator(execution_reconstructor=spy)
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)

    with pytest.raises(DiagnosticOrchestrationIntegrityError):
        orchestrator.run(
            _request(
                _scope(task_id, run_id, tenant_id=_TENANT_B),
            ),
        )

    spy.reconstruct_execution.assert_not_called()


def test_duplicate_execution_scope_fails_before_reconstruction() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    spy = MagicMock(
        spec=ExecutionReconstructor,
        wraps=ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=causal_store,
        ),
    )
    orchestrator, _, _, _ = _build_orchestrator(execution_reconstructor=spy)
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    scope = _scope(task_id, run_id)

    with pytest.raises(DiagnosticOrchestrationIntegrityError):
        orchestrator.run(_request(scope, scope))

    spy.reconstruct_execution.assert_not_called()


def test_execution_results_preserve_request_order() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_c, run_c = _seed_retry_violation_sequence(runtime_store)
    task_a, run_a = _seed_retry_violation_sequence(runtime_store)
    task_b, run_b = _seed_retry_violation_sequence(runtime_store)

    result = orchestrator.run(
        _request(
            _scope(task_c, run_c),
            _scope(task_a, run_a),
            _scope(task_b, run_b),
        ),
    )

    returned_pairs = [
        (analysis.task_id, analysis.run_id) for analysis in result.execution_results
    ]
    assert returned_pairs == [
        (task_c, run_c),
        (task_a, run_a),
        (task_b, run_b),
    ]


def test_one_spine_call_sequence() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)

    reconstructor = MagicMock(
        spec=ExecutionReconstructor,
        wraps=ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    lifecycle_analyzer = MagicMock(
        spec=LifecycleAnomalyAnalyzer,
        wraps=LifecycleAnomalyAnalyzer(),
    )
    assessment_builder = MagicMock(
        spec=DiagnosticAssessmentBuilder,
        wraps=DiagnosticAssessmentBuilder(),
    )
    grouping_engine = MagicMock(
        spec=ProblemGroupingEngine,
        wraps=_build_grouping_engine(),
    )
    lifecycle_engine = MagicMock(
        spec=ProblemLifecycleEngine,
        wraps=ProblemLifecycleEngine(InMemoryProblemPersistence()),
    )
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=reconstructor,
        lifecycle_analyzer=lifecycle_analyzer,
        assessment_builder=assessment_builder,
        grouping_engine=grouping_engine,
        problem_lifecycle_engine=lifecycle_engine,
    )

    orchestrator.run(
        _request(_scope(first_task, first_run), _scope(second_task, second_run)),
    )

    assert reconstructor.reconstruct_execution.call_count == 2
    assert lifecycle_analyzer.analyze.call_count == 2
    assert assessment_builder.assess.call_count == 2
    grouping_engine.group.assert_called_once()
    lifecycle_engine.reconcile.assert_called_once()


def test_second_execution_assessment_failure_skips_grouping_and_lifecycle() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)

    class _FailingOnSecondAssessment(DiagnosticAssessmentBuilder):
        def __init__(self) -> None:
            self._calls = 0

        def assess(self, reconstruction, lifecycle):
            self._calls += 1
            if self._calls == 2:
                raise DiagnosticAssessmentIntegrityError("forced integrity failure")
            return super().assess(reconstruction, lifecycle)

    grouping_engine = MagicMock(spec=ProblemGroupingEngine)
    lifecycle_engine = MagicMock(spec=ProblemLifecycleEngine)
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=_FailingOnSecondAssessment(),
        grouping_engine=grouping_engine,
        problem_lifecycle_engine=lifecycle_engine,
    )

    with pytest.raises(DiagnosticAssessmentIntegrityError):
        orchestrator.run(
            _request(_scope(first_task, first_run), _scope(second_task, second_run)),
        )

    grouping_engine.group.assert_not_called()
    lifecycle_engine.reconcile.assert_not_called()


def test_grouping_failure_skips_lifecycle_reconcile() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)

    grouping_engine = MagicMock(spec=ProblemGroupingEngine)
    grouping_engine.group.side_effect = ProblemGroupingIntegrityError("forced grouping failure")
    lifecycle_engine = MagicMock(spec=ProblemLifecycleEngine)
    orchestrator, _, _, _ = _build_orchestrator(
        grouping_engine=grouping_engine,
        problem_lifecycle_engine=lifecycle_engine,
    )

    with pytest.raises(ProblemGroupingIntegrityError):
        orchestrator.run(
            _request(_scope(first_task, first_run), _scope(second_task, second_run)),
        )

    lifecycle_engine.reconcile.assert_not_called()


def test_lifecycle_persistence_conflict_surfaces_integrity_error() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)

    lifecycle_engine = MagicMock(spec=ProblemLifecycleEngine)
    lifecycle_engine.reconcile.side_effect = ProblemLifecycleIntegrityError(
        "failed to update stable Problem due to persistence conflict",
    )
    orchestrator, _, _, _ = _build_orchestrator(problem_lifecycle_engine=lifecycle_engine)

    with pytest.raises(ProblemLifecycleIntegrityError):
        orchestrator.run(
            _request(_scope(first_task, first_run), _scope(second_task, second_run)),
        )


def test_incomplete_history_orchestration_still_succeeds() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    for _ in range(5):
        event = sample_runtime_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        runtime_store.append(event, tenant_id=_TENANT_A)

    class _TruncatingReconstructor(ExecutionReconstructor):
        def reconstruct_execution(self, tenant_id, task_id, run_id, **kwargs):
            return super().reconstruct_execution(
                tenant_id,
                task_id,
                run_id,
                initial_limit=2,
                max_limit=2,
            )

    orchestrator, _, _, _ = _build_orchestrator(
        execution_reconstructor=_TruncatingReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )

    result = orchestrator.run(_request(_scope(task_id, run_id)))

    assert (
        result.execution_results[0].runtime_history_completeness
        is RuntimeHistoryCompleteness.TRUNCATED
    )
    assert result.execution_results[0].assessment.has_limitations


def test_problem_signals_optional_and_preserved_in_source_bundle() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    first_signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_TOOL,
        source_component="first",
        safe_message="first signal",
        task_id=str(task_id),
        run_id=str(run_id),
    )
    second_signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_TOOL,
        source_component="second",
        safe_message="second signal",
        task_id=str(task_id),
        run_id=str(run_id),
    )

    grouping_engine = MagicMock(spec=ProblemGroupingEngine, wraps=_build_grouping_engine())
    orchestrator, _, _, _ = _build_orchestrator(grouping_engine=grouping_engine)

    result = orchestrator.run(
        _request(
            _scope(
                task_id,
                run_id,
                problem_signals=(first_signal, second_signal),
            ),
        ),
    )

    assert result.execution_results[0].assessment is not None
    assessment_inputs = grouping_engine.group.call_args.args[0]
    source_facts = assessment_inputs[0].feature_source_facts
    assert source_facts is not None
    assert source_facts.problem_signals == (first_signal, second_signal)


def test_public_result_has_no_forbidden_raw_data_fields() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)

    result = orchestrator.run(
        _request(_scope(first_task, first_run), _scope(second_task, second_run)),
    )

    forbidden = _collect_forbidden_field_names(result, seen=set())
    assert forbidden == set()


def test_production_grouping_strategy_count_remains_one() -> None:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    assert registry.registered_strategy_ids() == (STRATEGY_ID,)


def test_reconstruction_structural_failure_fails_orchestration() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    reconstructor = MagicMock(spec=ExecutionReconstructor)
    reconstructor.reconstruct_execution.side_effect = ExecutionReconstructionIntegrityError(
        "forced reconstruction failure",
    )
    grouping_engine = MagicMock(spec=ProblemGroupingEngine)
    lifecycle_engine = MagicMock(spec=ProblemLifecycleEngine)
    orchestrator, _, _, _ = _build_orchestrator(
        execution_reconstructor=reconstructor,
        grouping_engine=grouping_engine,
        problem_lifecycle_engine=lifecycle_engine,
    )
    task_id, run_id = mint_task_id(), mint_run_id()

    with pytest.raises(ExecutionReconstructionIntegrityError):
        orchestrator.run(_request(_scope(task_id, run_id)))

    grouping_engine.group.assert_not_called()
    lifecycle_engine.reconcile.assert_not_called()


def test_max_execution_scope_validation() -> None:
    orchestrator, _, _, _ = _build_orchestrator()
    scopes = tuple(
        _scope(mint_task_id(), mint_run_id())
        for _ in range(MAX_DIAGNOSTIC_ORCHESTRATION_EXECUTIONS + 1)
    )

    with pytest.raises(DiagnosticOrchestrationIntegrityError):
        orchestrator.run(_request(*scopes))


def test_empty_subject_inputs_rejected() -> None:
    orchestrator, _, _, _ = _build_orchestrator()

    with pytest.raises(DiagnosticOrchestrationIntegrityError):
        orchestrator.run(
            DiagnosticOrchestrationRequest(
                tenant_id=_TENANT_A,
                grouping_strategy_id=STRATEGY_ID,
                observed_at=_OBSERVED_AT,
            ),
        )


def test_naive_observed_at_rejected() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)

    with pytest.raises(ValueError, match="timezone-aware"):
        orchestrator.run(
            _request(
                _scope(task_id, run_id),
                observed_at=datetime(2026, 8, 26, 9, 0),
            ),
        )


def test_result_types_are_dataclasses() -> None:
    assert is_dataclass(DiagnosticOrchestrationResult)
