# © Artur Czarnecki. All rights reserved.

"""DIAG-READ-SCALE: request-scoped execution reconstruction reuse proofs."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_event_position import (
    AsOfBoundary,
    ExecutionEventPosition,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_reconstruction import (
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructionReader,
)
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticOccurrenceReadStatus,
)
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.execution_reconstruction_read_session import (
    ExecutionReconstructionReadSession,
)
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_grouping import (
    problem_grouping_subject_ref_for_execution,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    ProblemGroupingMethod,
    ProblemId,
    ProblemOccurrence,
    ProblemStatus,
    mint_problem_id,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePage,
    ProblemOccurrencePersistence,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    read_service_for_tests,
)
from tests.unit.runtime.diagnostics.test_diagnostic_read_service import (
    _OBSERVED_AT,
    _TENANT_A,
    _TENANT_B,
    _assess_retry_pair,
    _grouping_engine,
    _lifecycle_engine,
    _persist_problem,
    _occurrence_persistence_for,
)

pytestmark = pytest.mark.unit


class _CountingExecutionReconstructionReader:
    def __init__(self, delegate: ExecutionReconstructionReader) -> None:
        self._delegate = delegate
        self.reconstruction_calls = 0

    def reconstruct_execution(
        self, tenant_id, task_id, run_id, *, execution_as_of=None
    ):
        self.reconstruction_calls += 1
        return self._delegate.reconstruct_execution(
            tenant_id,
            task_id,
            run_id,
            execution_as_of=execution_as_of,
        )


def _seed_runtime_for_run(
    runtime_store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str,
    task_id,
    run_id,
) -> None:
    attempt_id = mint_attempt_id()
    for event_type in (
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_COMPLETED,
        RuntimeEventType.RETRY_SCHEDULED,
    ):
        runtime_store.append(
            sample_runtime_event(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ).model_copy(update={"event_type": event_type}),
            tenant_id=tenant_id,
        )


def _occurrence_for_run(
    *,
    tenant_id: str,
    task_id,
    run_id,
    observed_at,
) -> ProblemOccurrence:
    return ProblemOccurrence(
        subject_ref=problem_grouping_subject_ref_for_execution(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        ),
        observed_at=observed_at,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
    )


def _stack_with_occurrence_page(
    *,
    persistence: InMemoryProblemPersistence,
    occurrences: tuple[ProblemOccurrence, ...],
    runtime_store: InMemoryRuntimeEventStore,
    counter: _CountingExecutionReconstructionReader,
    occurrence_persistence: ProblemOccurrencePersistence | None = None,
) -> tuple[DiagnosticReadService, ProblemId, ProblemOccurrencePersistence]:
    problem = replace(
        sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id()),
        occurrence_count=len(occurrences),
        status=ProblemStatus.OPEN,
    )
    persistence.create(
        problem, indexed_subject_refs=tuple(o.subject_ref for o in occurrences)
    )

    if occurrence_persistence is None:
        resolved_occurrence = MagicMock(spec=ProblemOccurrencePersistence)
        resolved_occurrence.query_occurrences.return_value = ProblemOccurrencePage(
            items=occurrences,
            next_cursor=None,
            has_more=False,
        )
    else:
        resolved_occurrence = occurrence_persistence

    service = read_service_for_tests(
        persistence,
        counter,
        occurrence_persistence=resolved_occurrence,
    )
    return service, problem.problem_id, resolved_occurrence


def test_same_run_many_occurrence_rows_reconstruct_once() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    runtime_store = InMemoryRuntimeEventStore()
    _seed_runtime_for_run(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
    )
    occurrences = tuple(
        _occurrence_for_run(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )
        for index in range(500)
    )
    persistence = InMemoryProblemPersistence()
    counter = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    service, problem_id, _ = _stack_with_occurrence_page(
        persistence=persistence,
        occurrences=occurrences,
        runtime_store=runtime_store,
        counter=counter,
    )

    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem_id)

    assert detail is not None
    assert len(detail.occurrences) == 500
    assert counter.reconstruction_calls == 1
    assert all(
        item.read_status is DiagnosticOccurrenceReadStatus.AVAILABLE
        for item in detail.occurrences
    )


def test_many_unique_runs_reconstruct_per_scope() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    occurrences = []
    for index in range(5):
        task_id = mint_task_id()
        run_id = mint_run_id()
        _seed_runtime_for_run(
            runtime_store,
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
        )
        occurrences.append(
            _occurrence_for_run(
                tenant_id=_TENANT_A,
                task_id=task_id,
                run_id=run_id,
                observed_at=_OBSERVED_AT + timedelta(seconds=index),
            ),
        )
    persistence = InMemoryProblemPersistence()
    counter = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    service, problem_id, _ = _stack_with_occurrence_page(
        persistence=persistence,
        occurrences=tuple(occurrences),
        runtime_store=runtime_store,
        counter=counter,
    )
    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem_id)

    assert detail is not None
    assert counter.reconstruction_calls == 5


def test_large_request_scales_with_unique_scopes() -> None:
    unique_runs = 20
    runtime_store = InMemoryRuntimeEventStore()
    scope_refs: list[tuple[object, object]] = []
    for index in range(unique_runs):
        task_id = mint_task_id()
        run_id = mint_run_id()
        _seed_runtime_for_run(
            runtime_store,
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
        )
        scope_refs.append((task_id, run_id))

    occurrences = []
    for index in range(1000):
        task_id, run_id = scope_refs[index % unique_runs]
        occurrences.append(
            _occurrence_for_run(
                tenant_id=_TENANT_A,
                task_id=task_id,
                run_id=run_id,
                observed_at=_OBSERVED_AT + timedelta(seconds=index),
            ),
        )

    persistence = InMemoryProblemPersistence()
    counter = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    service, problem_id, _ = _stack_with_occurrence_page(
        persistence=persistence,
        occurrences=tuple(occurrences),
        runtime_store=runtime_store,
        counter=counter,
    )
    service.get_problem(tenant_id=_TENANT_A, problem_id=problem_id)

    assert counter.reconstruction_calls == unique_runs


def test_no_cross_tenant_reuse() -> None:
    shared_task = mint_task_id()
    shared_run = mint_run_id()

    runtime_a = InMemoryRuntimeEventStore()
    _seed_runtime_for_run(
        runtime_a,
        tenant_id=_TENANT_A,
        task_id=shared_task,
        run_id=shared_run,
    )
    occurrences_a = (
        _occurrence_for_run(
            tenant_id=_TENANT_A,
            task_id=shared_task,
            run_id=shared_run,
            observed_at=_OBSERVED_AT,
        ),
    )
    persistence_a = InMemoryProblemPersistence()
    counter_a = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=runtime_a,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    service_a, problem_a, _ = _stack_with_occurrence_page(
        persistence=persistence_a,
        occurrences=occurrences_a,
        runtime_store=runtime_a,
        counter=counter_a,
    )
    service_a.get_problem(tenant_id=_TENANT_A, problem_id=problem_a)

    runtime_b = InMemoryRuntimeEventStore()
    _seed_runtime_for_run(
        runtime_b,
        tenant_id=_TENANT_B,
        task_id=shared_task,
        run_id=shared_run,
    )
    persistence_b = InMemoryProblemPersistence()
    problem_b = replace(
        sample_problem(tenant_id=_TENANT_B, problem_id=mint_problem_id()),
        occurrence_count=1,
    )
    subject_b = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT_B,
        task_id=shared_task,
        run_id=shared_run,
    )
    persistence_b.create(problem_b, indexed_subject_refs=(subject_b,))
    counter_b = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=runtime_b,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    occurrence_persistence_b = MagicMock()
    occurrence_persistence_b.query_occurrences.return_value = ProblemOccurrencePage(
        items=(
            _occurrence_for_run(
                tenant_id=_TENANT_B,
                task_id=shared_task,
                run_id=shared_run,
                observed_at=_OBSERVED_AT,
            ),
        ),
        next_cursor=None,
        has_more=False,
    )
    service_b = read_service_for_tests(
        persistence_b,
        counter_b,
        occurrence_persistence=occurrence_persistence_b,
    )
    service_b.get_problem(tenant_id=_TENANT_B, problem_id=problem_b.problem_id)

    assert counter_a.reconstruction_calls == 1
    assert counter_b.reconstruction_calls == 1


def test_same_run_different_as_of_are_separate_scopes() -> None:
    reader = MagicMock(spec=ExecutionReconstructionReader)
    reader.reconstruct_execution.side_effect = lambda *args, **kwargs: MagicMock(
        tenant_id=args[0],
        task_id=args[1],
        run_id=args[2],
    )
    session = ExecutionReconstructionReadSession(reader)
    task_id = mint_task_id()
    run_id = mint_run_id()
    boundary_a = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(1))
    boundary_b = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(2))

    session.reconstruct_execution(
        _TENANT_A, task_id, run_id, execution_as_of=boundary_a
    )
    session.reconstruct_execution(
        _TENANT_A, task_id, run_id, execution_as_of=boundary_b
    )
    session.reconstruct_execution(_TENANT_A, task_id, run_id, execution_as_of=None)
    session.reconstruct_execution(_TENANT_A, task_id, run_id, execution_as_of=None)

    assert reader.reconstruct_execution.call_count == 3
    assert session.memo_entry_count() == 3


def test_incomplete_reconstruction_preserved_on_reuse() -> None:
    problem, persistence, _, _empty_runtime = _persist_problem()
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    dup_occurrences = tuple(
        ProblemOccurrence(
            subject_ref=subject,
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
            strategy_id=STRATEGY_ID,
            strategy_version=STRATEGY_VERSION,
            method=ProblemGroupingMethod.DETERMINISTIC,
        )
        for index in range(3)
    )
    empty_runtime = InMemoryRuntimeEventStore()
    counter = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=empty_runtime,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    occurrence_persistence = MagicMock()
    occurrence_persistence.query_occurrences.return_value = ProblemOccurrencePage(
        items=dup_occurrences,
        next_cursor=None,
        has_more=False,
    )
    service = read_service_for_tests(
        persistence,
        counter,
        occurrence_persistence=occurrence_persistence,
    )
    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    assert detail is not None
    assert counter.reconstruction_calls == 1
    for view in detail.occurrences:
        assert view.read_status is DiagnosticOccurrenceReadStatus.UNAVAILABLE


def test_reconstruction_failure_not_cached_as_success() -> None:
    reader = MagicMock(spec=ExecutionReconstructionReader)
    reader.reconstruct_execution.side_effect = ExecutionReconstructionIntegrityError(
        "integrity failure",
    )
    session = ExecutionReconstructionReadSession(reader)
    task_id = mint_task_id()
    run_id = mint_run_id()

    with pytest.raises(ExecutionReconstructionIntegrityError):
        session.reconstruct_execution(_TENANT_A, task_id, run_id)
    with pytest.raises(ExecutionReconstructionIntegrityError):
        session.reconstruct_execution(_TENANT_A, task_id, run_id)

    assert reader.reconstruct_execution.call_count == 2
    assert session.memo_entry_count() == 0


def test_get_investigation_reuses_occurrence_reconstruction() -> None:
    problem, persistence, _, runtime_store = _persist_problem()
    stored = _occurrence_persistence_for(persistence)
    page = stored.query_occurrences(
        tenant_id=_TENANT_A,
        problem_id=problem.problem_id,
        limit=100,
    )
    single_occurrence = (page.items[0],)
    occurrence_persistence = MagicMock()
    occurrence_persistence.query_occurrences.return_value = ProblemOccurrencePage(
        items=single_occurrence,
        next_cursor=None,
        has_more=False,
    )
    counter = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    service = read_service_for_tests(
        persistence,
        counter,
        occurrence_persistence=occurrence_persistence,
    )
    result = service.get_investigation(
        tenant_id=_TENANT_A,
        problem_id=problem.problem_id,
        occurrence_index=0,
    )

    assert result.investigation is not None
    assert counter.reconstruction_calls == 1


def test_zero_occurrences_zero_reconstruction() -> None:
    persistence = InMemoryProblemPersistence()
    problem = replace(
        sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id()),
        occurrence_count=0,
    )
    persistence.create(problem, indexed_subject_refs=())
    counter = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=InMemoryRuntimeEventStore(),
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    occurrence_persistence = MagicMock()
    occurrence_persistence.query_occurrences.return_value = ProblemOccurrencePage(
        items=(),
        next_cursor=None,
        has_more=False,
    )
    service = read_service_for_tests(
        persistence,
        counter,
        occurrence_persistence=occurrence_persistence,
    )
    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    assert detail is not None
    assert detail.occurrences == ()
    assert counter.reconstruction_calls == 0


def test_custom_reader_without_concrete_reconstructor() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    runtime_store = InMemoryRuntimeEventStore()
    _seed_runtime_for_run(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
    )
    occurrences = tuple(
        _occurrence_for_run(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )
        for index in range(10)
    )

    class _CustomReader:
        def __init__(self, inner: ExecutionReconstructor) -> None:
            self.calls = 0
            self._inner = inner

        def reconstruct_execution(
            self, tenant_id, task_id, run_id, *, execution_as_of=None
        ):
            self.calls += 1
            return self._inner.reconstruct_execution(
                tenant_id,
                task_id,
                run_id,
                execution_as_of=execution_as_of,
            )

    inner = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    custom = _CustomReader(inner)
    persistence = InMemoryProblemPersistence()
    _, problem_id, occurrence_persistence = _stack_with_occurrence_page(
        persistence=persistence,
        occurrences=occurrences,
        runtime_store=runtime_store,
        counter=_CountingExecutionReconstructionReader(inner),
    )
    service = read_service_for_tests(
        persistence,
        custom,
        occurrence_persistence=occurrence_persistence,
    )
    service.get_problem(tenant_id=_TENANT_A, problem_id=problem_id)

    assert custom.calls == 1


def test_integration_many_runs_from_lifecycle() -> None:
    persistence = InMemoryProblemPersistence()
    runtime_store = InMemoryRuntimeEventStore()
    first_input, second_input = _assess_retry_pair(runtime_store=runtime_store)
    third_input, _ = _assess_retry_pair(runtime_store=runtime_store)
    lifecycle = _lifecycle_engine(persistence)
    grouping = _grouping_engine()
    pair = grouping.group((first_input, second_input), strategy_id=STRATEGY_ID)
    lifecycle.reconcile(pair, observed_at=_OBSERVED_AT)
    extended = grouping.group(
        (first_input, second_input, third_input),
        strategy_id=STRATEGY_ID,
    )
    problem = lifecycle.reconcile(extended, observed_at=_OBSERVED_AT).updated[0]
    counter = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    service = read_service_for_tests(
        persistence,
        counter,
        occurrence_persistence=_occurrence_persistence_for(persistence),
    )
    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    assert detail is not None
    assert len(detail.occurrences) == 3
    assert counter.reconstruction_calls == 3
