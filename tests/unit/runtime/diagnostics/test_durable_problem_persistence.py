# © Artur Czarnecki. All rights reserved.

"""DIAG-STORAGE durable Problem persistence race, restart, and integrity tests."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.deterministic_problem_reconciliation import (
    DeterministicProblemReconciliationKey,
)
from intergrax.runtime.diagnostics.diagnostic_problem_grouping_feature_projector import (
    DiagnosticProblemGroupingFeatureProjector,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticExecutionScope,
    DiagnosticOrchestrationRequest,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
    wire_problem_persistence,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.persistence_conformance import (
    query_all_problems_for_tenant,
    sample_problem,
    _sample_reconciliation_key,
    _sample_subject_ref,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicProblemSignature,
    ProblemGroupingEngine,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemLifecycleEngine,
    ProblemStatus,
    mint_problem_id,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistence,
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
    ProblemPersistenceIntegrityReason,
)
from intergrax.runtime.diagnostics.problem_record_codec import decode_problem_record
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    DocumentStoreCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    TEST_PROBLEM_LIST_CURSOR_SECRET,
    document_store_problem_persistence_for_tests,
)
from tests.unit.runtime.diagnostics.test_diagnostic_orchestrator import (
    _OBSERVED_AT,
    _OBSERVED_AT_LATER,
    _RETRY_SEQUENCE,
    _TENANT_A,
)
from tests.unit.runtime.vendor_knowledge._fakes import (
    InMemoryDocumentStore as PlainDocumentStore,
)

pytestmark = pytest.mark.unit


class _FailingPutIfAbsentDocumentStore(InMemoryDocumentStore):
    """In-memory store that raises once on selected put_if_absent keys."""

    def __init__(self, *, fail_keys: frozenset[tuple[str, str]] = frozenset()) -> None:
        super().__init__()
        self._fail_keys = fail_keys
        self._failed_keys: set[tuple[str, str]] = set()

    def put_if_absent(self, document: DocumentRecord) -> bool:
        key = (document.partition_key, document.row_key)
        if key in self._fail_keys and key not in self._failed_keys:
            self._failed_keys.add(key)
            raise RuntimeError("simulated diagnostic problem index write failure")
        return super().put_if_absent(document)


class _FailingReplaceIfMatchDocumentStore(InMemoryDocumentStore):
    """Raises once on replace_if_match after optionally applying the write."""

    def __init__(self, *, write_before_raise: bool = True) -> None:
        super().__init__()
        self._write_before_raise = write_before_raise
        self._failed_once = False

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if not self._failed_once:
            self._failed_once = True
            if self._write_before_raise:
                super().replace_if_match(expected=expected, replacement=replacement)
            raise RuntimeError("simulated uncertain diagnostic problem CAS")
        return super().replace_if_match(expected=expected, replacement=replacement)


class _CountingPutIfAbsentDocumentStore(InMemoryDocumentStore):
    """Tracks put_if_absent calls for resolve-path assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.put_if_absent_calls = 0

    def put_if_absent(self, document: DocumentRecord) -> bool:
        self.put_if_absent_calls += 1
        return super().put_if_absent(document)


def _seed_retry_violation_sequence(
    runtime_store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str = _TENANT_A,
):
    from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id

    task_id = mint_task_id()
    run_id = mint_run_id()
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


def _scope(task_id, run_id, *, tenant_id: str = _TENANT_A) -> DiagnosticExecutionScope:
    return DiagnosticExecutionScope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        problem_signals=(),
    )


def _request(*executions: DiagnosticExecutionScope, observed_at: datetime = _OBSERVED_AT):
    return DiagnosticOrchestrationRequest(
        tenant_id=_TENANT_A,
        executions=executions,
        grouping_strategy_id=STRATEGY_ID,
        observed_at=observed_at,
    )


def _build_grouping_engine() -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return ProblemGroupingEngine(
        registry,
        feature_projector=DiagnosticProblemGroupingFeatureProjector(),
    )


def _orchestration_semantics(persistence: ProblemPersistence) -> dict[str, object]:
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = DocumentStoreCausalEvidencePersistence(InMemoryDocumentStore())
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=causal_store,
        ),
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=_build_grouping_engine(),
        problem_lifecycle_engine=ProblemLifecycleEngine(persistence),
    )
    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)
    third_task, third_run = _seed_retry_violation_sequence(runtime_store)

    first = orchestrator.run(_request(_scope(first_task, first_run), _scope(second_task, second_run)))
    problem = first.lifecycle_result.created[0]
    second = orchestrator.run(
        _request(
            _scope(first_task, first_run),
            _scope(second_task, second_run),
            _scope(third_task, third_run),
            observed_at=_OBSERVED_AT_LATER,
        ),
    )
    updated = second.lifecycle_result.updated[0]
    return {
        "problem_count": len(query_all_problems_for_tenant(persistence, _TENANT_A)),
        "status": updated.status,
        "occurrence_count": updated.occurrence_count,
        "reconciliation_token": updated.provenance.reconciliation_key.index_token(),
        "subject_ref_count": len(updated.current_subject_refs),
        "subject_tenant_ids": tuple(ref.tenant_id for ref in updated.current_subject_refs),
        "first_seen_at": updated.first_seen_at,
        "last_seen_at": updated.last_seen_at,
    }


def test_wire_selects_document_backend_from_conditional_store() -> None:
    store = InMemoryDocumentStore()
    persistence = wire_problem_persistence(list_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET, document_store=store)
    assert isinstance(persistence, DocumentStoreProblemPersistence)


def test_wire_rejects_non_conditional_document_store() -> None:
    store = PlainDocumentStore()
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        wire_problem_persistence(list_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET, document_store=store)


def test_document_store_restart_survives_new_adapter_instance() -> None:
    store = InMemoryDocumentStore()
    record = sample_problem(tenant_id="tenant-restart")
    first = document_store_problem_persistence_for_tests(store)
    first.create(record)
    first.close()

    second = document_store_problem_persistence_for_tests(store)
    try:
        loaded = second.get(tenant_id=record.tenant_id, problem_id=record.problem_id)
        assert loaded == record
        assert query_all_problems_for_tenant(second, record.tenant_id) == (record,)
        updated = Problem(
            problem_id=record.problem_id,
            tenant_id=record.tenant_id,
            status=record.status,
            first_seen_at=record.first_seen_at,
            last_seen_at=record.last_seen_at + timedelta(hours=1),
            occurrence_count=record.occurrence_count + 1,
            current_subject_refs=record.current_subject_refs,
            occurrences=record.occurrences,
            provenance=record.provenance,
            record_version=2,
        )
        assert second.update(updated, expected_version=1) == updated
    finally:
        second.close()


def test_document_store_concurrent_identical_create_same_problem_id() -> None:
    store = InMemoryDocumentStore()
    record = sample_problem(tenant_id="tenant-race-identical")
    barrier = threading.Barrier(2)
    results: list[Problem] = []
    errors: list[BaseException] = []

    def _create() -> None:
        persistence = document_store_problem_persistence_for_tests(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.create(record))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_create), executor.submit(_create)]
        for future in futures:
            future.result(timeout=10)

    assert not errors
    assert len(results) == 2
    assert results[0] == record
    assert results[1] == record
    verifier = document_store_problem_persistence_for_tests(store)
    try:
        assert query_all_problems_for_tenant(verifier, record.tenant_id) == (record,)
    finally:
        verifier.close()


def test_document_store_concurrent_conflicting_same_problem_id() -> None:
    store = InMemoryDocumentStore()
    problem_id = mint_problem_id()
    first = sample_problem(
        tenant_id="tenant-conflict-id",
        problem_id=problem_id,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = sample_problem(
        tenant_id="tenant-conflict-id",
        problem_id=problem_id,
        observed_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    barrier = threading.Barrier(2)
    results: list[Problem] = []
    errors: list[BaseException] = []

    def _create(candidate: Problem) -> None:
        persistence = document_store_problem_persistence_for_tests(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.create(candidate))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_create, first), executor.submit(_create, second)]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ProblemPersistenceConflictError)
    winner = results[0]
    assert winner in (first, second)
    verifier = document_store_problem_persistence_for_tests(store)
    try:
        assert query_all_problems_for_tenant(verifier, "tenant-conflict-id") == (winner,)
        loser = second if winner == first else first
        if loser.provenance.reconciliation_key != winner.provenance.reconciliation_key:
            assert (
                verifier.find_by_reconciliation_key(
                    tenant_id="tenant-conflict-id",
                    reconciliation_key=loser.provenance.reconciliation_key,
                )
                is None
            )
    finally:
        verifier.close()


def test_document_store_concurrent_cas_same_expected_version() -> None:
    store = InMemoryDocumentStore()
    record = sample_problem(tenant_id="tenant-cas-race")
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(record)
    updated = Problem(
        problem_id=record.problem_id,
        tenant_id=record.tenant_id,
        status=record.status,
        first_seen_at=record.first_seen_at,
        last_seen_at=record.last_seen_at + timedelta(hours=1),
        occurrence_count=record.occurrence_count + 1,
        current_subject_refs=record.current_subject_refs,
        occurrences=record.occurrences,
        provenance=record.provenance,
        record_version=2,
    )
    barrier = threading.Barrier(2)
    results: list[Problem] = []
    errors: list[BaseException] = []

    def _update() -> None:
        worker = document_store_problem_persistence_for_tests(store)
        try:
            barrier.wait(timeout=5)
            results.append(worker.update(updated, expected_version=1))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_update), executor.submit(_update)]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ProblemPersistenceConflictError)


def test_document_store_concurrent_reconciliation_key_collision() -> None:
    store = InMemoryDocumentStore()
    tenant_id = "tenant-reconcile-race"
    first = sample_problem(tenant_id=tenant_id)
    second = sample_problem(
        tenant_id=tenant_id,
        reconciliation_key=first.provenance.reconciliation_key,  # type: ignore[arg-type]
    )
    barrier = threading.Barrier(2)
    results: list[Problem] = []
    errors: list[BaseException] = []

    def _create(candidate: Problem) -> None:
        persistence = document_store_problem_persistence_for_tests(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.create(candidate))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_create, first), executor.submit(_create, second)]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ProblemPersistenceConflictError)
    winner = results[0]
    assert winner in (first, second)
    verifier = document_store_problem_persistence_for_tests(store)
    try:
        listed = query_all_problems_for_tenant(verifier, tenant_id)
        assert listed == (winner,)
        loser = second if winner == first else first
        assert verifier.get(tenant_id=tenant_id, problem_id=loser.problem_id) is None
    finally:
        verifier.close()


def test_document_store_concurrent_subject_collision() -> None:
    store = InMemoryDocumentStore()
    tenant_id = "tenant-subject-race"
    shared = sample_problem(tenant_id=tenant_id).current_subject_refs[0]
    first_key = _sample_reconciliation_key(tenant_id=tenant_id)
    second_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=DeterministicProblemSignature(findings=(), limitations=()),
    )
    first = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(shared,),
        reconciliation_key=first_key,
    )
    second = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(shared,),
        reconciliation_key=second_key,
    )
    barrier = threading.Barrier(2)
    results: list[Problem] = []
    errors: list[BaseException] = []

    def _create(candidate: Problem) -> None:
        persistence = document_store_problem_persistence_for_tests(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.create(candidate))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_create, first), executor.submit(_create, second)]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ProblemPersistenceConflictError)
    winner = results[0]
    assert winner in (first, second)
    loser = second if winner == first else first
    verifier = document_store_problem_persistence_for_tests(store)
    try:
        assert query_all_problems_for_tenant(verifier, tenant_id) == (winner,)
        assert verifier.get(tenant_id=tenant_id, problem_id=loser.problem_id) is None
        assert (
            verifier.find_by_reconciliation_key(
                tenant_id=tenant_id,
                reconciliation_key=loser.provenance.reconciliation_key,
            )
            is None
        )
        assert (
            verifier.find_by_subject_ref(
                tenant_id=tenant_id,
                subject_ref=shared,
            )
            == winner
        )

        reuse_subject = sample_problem(tenant_id=tenant_id).current_subject_refs[0]
        reused = sample_problem(
            tenant_id=tenant_id,
            subject_refs=(reuse_subject,),
            reconciliation_key=loser.provenance.reconciliation_key,
        )
        assert verifier.create(reused) == reused
    finally:
        verifier.close()


def test_document_store_multi_subject_partial_claim_rolls_back() -> None:
    store = InMemoryDocumentStore()
    tenant_id = "tenant-partial-subject-claim"
    owned_subject = sample_problem(tenant_id=tenant_id).current_subject_refs[0]
    free_subject = sample_problem(tenant_id=tenant_id).current_subject_refs[0]
    key_a = _sample_reconciliation_key(tenant_id=tenant_id)
    key_b = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=DeterministicProblemSignature(findings=(), limitations=()),
    )
    problem_a = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(owned_subject,),
        reconciliation_key=key_a,
    )
    problem_b = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(free_subject, owned_subject),
        reconciliation_key=key_b,
    )
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(problem_a)

    with pytest.raises(ProblemPersistenceConflictError):
        persistence.create(problem_b)

    assert (
        persistence.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=problem_b.provenance.reconciliation_key,
        )
        is None
    )
    assert (
        persistence.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=free_subject,
        )
        is None
    )
    assert (
        persistence.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=owned_subject,
        )
        == problem_a
    )
    assert persistence.get(tenant_id=tenant_id, problem_id=problem_b.problem_id) is None


def test_document_store_update_adds_subject_index_before_canonical_cas() -> None:
    store = InMemoryDocumentStore()
    tenant_id = "tenant-update-subject-index"
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    created = sample_problem(tenant_id=tenant_id, subject_refs=(subject_a,))
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(created)
    updated = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=created.status,
        first_seen_at=created.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=created.occurrence_count + 1,
        current_subject_refs=(subject_a, subject_b),
        occurrences=created.occurrences,
        provenance=created.provenance,
        record_version=2,
    )
    assert persistence.update(updated, expected_version=1) == updated
    assert persistence.find_by_subject_ref(
        tenant_id=tenant_id,
        subject_ref=subject_b,
    ) == updated


def test_document_store_update_subject_collision_leaves_canonical_unchanged() -> None:
    store = InMemoryDocumentStore()
    tenant_id = "tenant-update-subject-collision"
    owned_subject = _sample_subject_ref(tenant_id=tenant_id)
    problem_q = sample_problem(tenant_id=tenant_id, subject_refs=(owned_subject,))
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(problem_q)
    problem_p = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(_sample_subject_ref(tenant_id=tenant_id),),
        reconciliation_key=_sample_reconciliation_key(
            tenant_id=tenant_id,
            signature=DeterministicProblemSignature(findings=(), limitations=()),
        ),
    )
    persistence.create(problem_p)
    collision_update = Problem(
        problem_id=problem_p.problem_id,
        tenant_id=problem_p.tenant_id,
        status=problem_p.status,
        first_seen_at=problem_p.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=problem_p.occurrence_count + 1,
        current_subject_refs=problem_p.current_subject_refs + (owned_subject,),
        occurrences=problem_p.occurrences,
        provenance=problem_p.provenance,
        record_version=2,
    )
    with pytest.raises(ProblemPersistenceConflictError):
        persistence.update(collision_update, expected_version=1)
    assert persistence.get(tenant_id=tenant_id, problem_id=problem_p.problem_id) == problem_p
    assert (
        persistence.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=owned_subject,
        )
        == problem_q
    )


def test_document_store_update_multi_subject_partial_claim_rolls_back() -> None:
    store = InMemoryDocumentStore()
    tenant_id = "tenant-update-partial-claim"
    owned_subject = _sample_subject_ref(tenant_id=tenant_id)
    free_subject = _sample_subject_ref(tenant_id=tenant_id)
    only_a = _sample_subject_ref(tenant_id=tenant_id)
    problem_q = sample_problem(tenant_id=tenant_id, subject_refs=(owned_subject,))
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(problem_q)
    problem_p = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(only_a,),
        reconciliation_key=_sample_reconciliation_key(
            tenant_id=tenant_id,
            signature=DeterministicProblemSignature(findings=(), limitations=()),
        ),
    )
    persistence.create(problem_p)
    partial_update = Problem(
        problem_id=problem_p.problem_id,
        tenant_id=problem_p.tenant_id,
        status=problem_p.status,
        first_seen_at=problem_p.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=problem_p.occurrence_count + 2,
        current_subject_refs=(only_a, free_subject, owned_subject),
        occurrences=problem_p.occurrences,
        provenance=problem_p.provenance,
        record_version=2,
    )
    with pytest.raises(ProblemPersistenceConflictError):
        persistence.update(partial_update, expected_version=1)
    assert (
        persistence.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=free_subject,
        )
        is None
    )
    assert (
        persistence.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=owned_subject,
        )
        == problem_q
    )
    assert persistence.get(tenant_id=tenant_id, problem_id=problem_p.problem_id) == problem_p


def test_document_store_update_concurrent_cas_after_new_subject_claim() -> None:
    store = InMemoryDocumentStore()
    tenant_id = "tenant-update-cas-new-subject"
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    created = sample_problem(tenant_id=tenant_id, subject_refs=(subject_a,))
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(created)
    with_subject_b = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=created.status,
        first_seen_at=created.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=created.occurrence_count + 1,
        current_subject_refs=(subject_a, subject_b),
        occurrences=created.occurrences,
        provenance=created.provenance,
        record_version=2,
    )
    without_subject_b = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=created.status,
        first_seen_at=created.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=created.occurrence_count + 1,
        current_subject_refs=(subject_a,),
        occurrences=created.occurrences,
        provenance=created.provenance,
        record_version=2,
    )
    barrier = threading.Barrier(2)
    results: list[Problem] = []
    errors: list[BaseException] = []

    def _update(candidate: Problem) -> None:
        worker = document_store_problem_persistence_for_tests(store)
        try:
            barrier.wait(timeout=5)
            results.append(worker.update(candidate, expected_version=1))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_update, with_subject_b),
            executor.submit(_update, without_subject_b),
        ]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ProblemPersistenceConflictError)
    winner = results[0]
    verifier = document_store_problem_persistence_for_tests(store)
    try:
        assert verifier.get(tenant_id=tenant_id, problem_id=created.problem_id) == winner
        if subject_b in winner.current_subject_refs:
            assert (
                verifier.find_by_subject_ref(
                    tenant_id=tenant_id,
                    subject_ref=subject_b,
                )
                == winner
            )
        else:
            assert (
                verifier.find_by_subject_ref(
                    tenant_id=tenant_id,
                    subject_ref=subject_b,
                )
                is None
            )
    finally:
        verifier.close()


def test_document_store_update_uncertain_cas_with_durable_replacement_succeeds() -> None:
    store = _FailingReplaceIfMatchDocumentStore(write_before_raise=True)
    tenant_id = "tenant-update-uncertain-success"
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    created = sample_problem(tenant_id=tenant_id, subject_refs=(subject_a,))
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(created)
    updated = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=created.status,
        first_seen_at=created.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=created.occurrence_count + 1,
        current_subject_refs=(subject_a, subject_b),
        occurrences=created.occurrences,
        provenance=created.provenance,
        record_version=2,
    )
    assert persistence.update(updated, expected_version=1) == updated
    assert persistence.find_by_subject_ref(
        tenant_id=tenant_id,
        subject_ref=subject_b,
    ) == updated


def test_document_store_update_uncertain_cas_without_write_rolls_back_claims() -> None:
    store = _FailingReplaceIfMatchDocumentStore(write_before_raise=False)
    tenant_id = "tenant-update-uncertain-rollback"
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    created = sample_problem(tenant_id=tenant_id, subject_refs=(subject_a,))
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(created)
    updated = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=created.status,
        first_seen_at=created.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=created.occurrence_count + 1,
        current_subject_refs=(subject_a, subject_b),
        occurrences=created.occurrences,
        provenance=created.provenance,
        record_version=2,
    )
    with pytest.raises(RuntimeError, match="simulated uncertain diagnostic problem CAS"):
        persistence.update(updated, expected_version=1)
    assert persistence.get(tenant_id=tenant_id, problem_id=created.problem_id) == created
    assert (
        persistence.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=subject_b,
        )
        is None
    )
    assert persistence.update(updated, expected_version=1) == updated


def test_document_store_update_rolls_back_claims_after_pre_cas_failure() -> None:
    store = _FailingReplaceIfMatchDocumentStore(write_before_raise=False)
    tenant_id = "tenant-update-pre-cas-failure"
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    created = sample_problem(tenant_id=tenant_id, subject_refs=(subject_a,))
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(created)
    updated = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=created.status,
        first_seen_at=created.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=created.occurrence_count + 1,
        current_subject_refs=(subject_a, subject_b),
        occurrences=created.occurrences,
        provenance=created.provenance,
        record_version=2,
    )
    with pytest.raises(RuntimeError, match="simulated uncertain diagnostic problem CAS"):
        persistence.update(updated, expected_version=1)
    assert persistence.get(tenant_id=tenant_id, problem_id=created.problem_id) == created
    retry_store = document_store_problem_persistence_for_tests(store)
    assert retry_store.update(updated, expected_version=1) == updated


def test_document_store_resolve_update_does_not_claim_new_subject_indexes() -> None:
    store = _CountingPutIfAbsentDocumentStore()
    tenant_id = "tenant-resolve-no-claims"
    created = sample_problem(tenant_id=tenant_id)
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(created)
    store.put_if_absent_calls = 0
    resolved = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=ProblemStatus.RESOLVED,
        first_seen_at=created.first_seen_at,
        last_seen_at=created.last_seen_at,
        occurrence_count=created.occurrence_count,
        current_subject_refs=created.current_subject_refs,
        occurrences=created.occurrences,
        provenance=created.provenance,
        record_version=2,
    )
    assert persistence.update(resolved, expected_version=1) == resolved
    assert store.put_if_absent_calls == 1


def test_document_store_restart_after_subject_adding_update_resolves_lookup() -> None:
    store = InMemoryDocumentStore()
    tenant_id = "tenant-update-restart-lookup"
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    created = sample_problem(tenant_id=tenant_id, subject_refs=(subject_a,))
    first = document_store_problem_persistence_for_tests(store)
    first.create(created)
    updated = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=created.status,
        first_seen_at=created.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=created.occurrence_count + 1,
        current_subject_refs=(subject_a, subject_b),
        occurrences=created.occurrences,
        provenance=created.provenance,
        record_version=2,
    )
    first.update(updated, expected_version=1)
    first.close()

    second = document_store_problem_persistence_for_tests(store)
    try:
        assert second.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=subject_b,
        ) == updated
    finally:
        second.close()

    third = document_store_problem_persistence_for_tests(store)
    try:
        assert third.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=subject_b,
        ) == updated
    finally:
        third.close()


def test_document_store_create_rolls_back_after_canonical_write_failure() -> None:
    record = sample_problem(tenant_id="tenant-canonical-failure")
    partition_key = f"intergrax.diagnostic_problem.v1:{record.tenant_id}"
    canonical_key = (partition_key, f"record:{record.problem_id}")
    store = _FailingPutIfAbsentDocumentStore(fail_keys=frozenset({canonical_key}))
    persistence = document_store_problem_persistence_for_tests(store)

    with pytest.raises(RuntimeError, match="simulated diagnostic problem index write failure"):
        persistence.create(record)

    assert (
        persistence.find_by_reconciliation_key(
            tenant_id=record.tenant_id,
            reconciliation_key=record.provenance.reconciliation_key,
        )
        is None
    )
    for subject_ref in record.current_subject_refs:
        assert (
            persistence.find_by_subject_ref(
                tenant_id=record.tenant_id,
                subject_ref=subject_ref,
            )
            is None
        )
    assert persistence.get(tenant_id=record.tenant_id, problem_id=record.problem_id) is None
    assert persistence.create(record) == record


def test_document_store_create_retries_after_reconciliation_index_write_failure() -> None:
    record = sample_problem(tenant_id="tenant-partial-reconcile")
    partition_key = f"intergrax.diagnostic_problem.v1:{record.tenant_id}"
    reconcile_key = (
        partition_key,
        f"reconcile:{record.provenance.reconciliation_key.index_token()}",
    )
    store = _FailingPutIfAbsentDocumentStore(fail_keys=frozenset({reconcile_key}))
    persistence = document_store_problem_persistence_for_tests(store)

    with pytest.raises(RuntimeError, match="simulated diagnostic problem index write failure"):
        persistence.create(record)

    assert document_store_problem_persistence_for_tests(store).create(record) == record


def test_document_store_create_retries_after_subject_index_write_failure() -> None:
    record = sample_problem(tenant_id="tenant-partial-subject")
    partition_key = f"intergrax.diagnostic_problem.v1:{record.tenant_id}"
    subject_ref = record.current_subject_refs[0]
    subject_key = (partition_key, f"subject:{subject_ref.index_token}")
    store = _FailingPutIfAbsentDocumentStore(fail_keys=frozenset({subject_key}))
    persistence = document_store_problem_persistence_for_tests(store)

    with pytest.raises(RuntimeError, match="simulated diagnostic problem index write failure"):
        persistence.create(record)

    assert persistence.create(record) == record
    assert persistence.find_by_subject_ref(
        tenant_id=record.tenant_id,
        subject_ref=subject_ref,
    ) == record


def test_document_store_malformed_canonical_record_fails_closed() -> None:
    store = InMemoryDocumentStore()
    record = sample_problem(tenant_id="tenant-malformed")
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(record)

    partition_key = f"intergrax.diagnostic_problem.v1:{record.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"record:{record.problem_id}",
            data={"schema_version": "broken"},
        )
    )
    with pytest.raises(ProblemPersistenceIntegrityError):
        persistence.get(tenant_id=record.tenant_id, problem_id=record.problem_id)


def test_document_store_orphan_reconciliation_index_fails_closed() -> None:
    store = InMemoryDocumentStore()
    record = sample_problem(tenant_id="tenant-orphan-reconcile")
    partition_key = f"intergrax.diagnostic_problem.v1:{record.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"reconcile:{record.provenance.reconciliation_key.index_token()}",
            data={
                "schema_version": "intergrax.diagnostic_problem.index.v1",
                "problem_id": str(record.problem_id),
            },
        )
    )
    persistence = document_store_problem_persistence_for_tests(store)
    with pytest.raises(ProblemPersistenceIntegrityError) as exc_info:
        persistence.find_by_reconciliation_key(
            tenant_id=record.tenant_id,
            reconciliation_key=record.provenance.reconciliation_key,
        )
    exc = exc_info.value
    assert (
        exc.reason
        is ProblemPersistenceIntegrityReason.RECONCILIATION_WINNER_CANONICAL_PENDING
    )
    assert "canonical Problem record missing" in str(exc)


def test_document_store_orphan_subject_index_fails_closed() -> None:
    store = InMemoryDocumentStore()
    record = sample_problem(tenant_id="tenant-orphan-subject")
    subject_ref = record.current_subject_refs[0]
    partition_key = f"intergrax.diagnostic_problem.v1:{record.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"subject:{subject_ref.index_token}",
            data={
                "schema_version": "intergrax.diagnostic_problem.index.v1",
                "problem_id": str(record.problem_id),
            },
        )
    )
    persistence = document_store_problem_persistence_for_tests(store)
    with pytest.raises(ProblemPersistenceIntegrityError):
        persistence.find_by_subject_ref(
            tenant_id=record.tenant_id,
            subject_ref=subject_ref,
        )


def test_document_store_wrong_scope_reconciliation_index_fails_closed() -> None:
    store = InMemoryDocumentStore()
    record = sample_problem(tenant_id="tenant-wrong-scope")
    foreign_key = DeterministicProblemReconciliationKey(
        tenant_id=record.tenant_id,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        signature=DeterministicProblemSignature(findings=(), limitations=()),
    )
    foreign = sample_problem(
        tenant_id=record.tenant_id,
        reconciliation_key=foreign_key,
    )
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.create(record)
    persistence.create(foreign)

    partition_key = f"intergrax.diagnostic_problem.v1:{record.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"reconcile:{record.provenance.reconciliation_key.index_token()}",
            data={
                "schema_version": "intergrax.diagnostic_problem.index.v1",
                "problem_id": str(foreign.problem_id),
            },
        )
    )
    with pytest.raises(ProblemPersistenceIntegrityError):
        persistence.find_by_reconciliation_key(
            tenant_id=record.tenant_id,
            reconciliation_key=record.provenance.reconciliation_key,
        )


def test_orchestrator_durable_problem_persistence_survives_adapter_restart() -> None:
    store = InMemoryDocumentStore()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = DocumentStoreCausalEvidencePersistence(store)
    persistence = document_store_problem_persistence_for_tests(store)
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=causal_store,
        ),
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=_build_grouping_engine(),
        problem_lifecycle_engine=ProblemLifecycleEngine(persistence),
    )

    first_task, first_run = _seed_retry_violation_sequence(runtime_store)
    second_task, second_run = _seed_retry_violation_sequence(runtime_store)
    first = orchestrator.run(_request(_scope(first_task, first_run), _scope(second_task, second_run)))
    problem_id = first.lifecycle_result.created[0].problem_id

    persistence.close()
    persistence = document_store_problem_persistence_for_tests(store)
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=causal_store,
        ),
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=_build_grouping_engine(),
        problem_lifecycle_engine=ProblemLifecycleEngine(persistence),
    )

    third_task, third_run = _seed_retry_violation_sequence(runtime_store)
    second = orchestrator.run(
        _request(
            _scope(first_task, first_run),
            _scope(second_task, second_run),
            _scope(third_task, third_run),
            observed_at=_OBSERVED_AT_LATER,
        ),
    )
    updated = second.lifecycle_result.updated[0]
    assert updated.problem_id == problem_id
    assert updated.occurrence_count == 3
    third_subject = updated.current_subject_refs[2]
    assert persistence.find_by_subject_ref(
        tenant_id=_TENANT_A,
        subject_ref=third_subject,
    ) == updated

    persistence.close()
    persistence = document_store_problem_persistence_for_tests(store)
    assert persistence.find_by_subject_ref(
        tenant_id=_TENANT_A,
        subject_ref=third_subject,
    ) == updated


def test_orchestrator_semantics_equivalent_across_problem_persistence_backends() -> None:
    memory_semantics = _orchestration_semantics(InMemoryProblemPersistence())
    document_semantics = _orchestration_semantics(
        document_store_problem_persistence_for_tests(InMemoryDocumentStore()),
    )
    assert memory_semantics == document_semantics


def test_codec_unknown_schema_version_fails_closed() -> None:
    with pytest.raises(ProblemPersistenceIntegrityError):
        decode_problem_record({"schema_version": "unknown.v9", "payload": {}})
