# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.contracts.functional_evidence_bounds import (
    MAX_DIRECT_UPSTREAM_EVIDENCE_REFS,
    MAX_SUPPORTING_EVIDENCE_REFS,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceIntegrityError,
    FunctionalEvidenceQueryRequest,
    functional_evidence_query_order_key,
)
from intergrax.runtime.diagnostics.functional_evidence_query_cursor import (
    FunctionalEvidenceQueryCursorCodec,
)
from intergrax.runtime.diagnostics.functional_evidence_reconstruction import (
    FunctionalEvidenceCompletenessStatus,
    FunctionalEvidenceReconstructor,
)
from intergrax.runtime.diagnostics.functional_validation import (
    DiagnosticExecutionCorrelation,
    ExpectedActualRelation,
    FunctionalValidationEvidence,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
    functional_validation_evidence_id,
)
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.observability.functional_validation_evidence import (
    FunctionalValidationEvidence as ObservabilityFunctionalValidationEvidence,
)
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_EXCEPTION,
    PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID,
    PlatformProblemSignal,
)

pytestmark = pytest.mark.unit

_TEST_CURSOR_SECRET = b"x" * 32
_BASE_TIME = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)


def _persistence() -> InMemoryFunctionalEvidencePersistence:
    return InMemoryFunctionalEvidencePersistence(cursor_secret=_TEST_CURSOR_SECRET)


def _scope(
    *,
    tenant_id: str = "tenant-a",
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> PipelineEvidenceScope:
    return PipelineEvidenceScope(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id,
    )


def _operation_evidence(
    scope: PipelineEvidenceScope,
    *,
    operation_name: str,
    recorded_at: datetime,
    evidence_id: str | None = None,
) -> PlatformFunctionalEvidence:
    return PlatformFunctionalEvidence(
        evidence_id=evidence_id or mint_event_id(),
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.test",
            operation_id=operation_name,
            recorded_at=recorded_at,
        ),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name=operation_name,
            status=PipelineOperationStatus.SUCCEEDED,
        ),
    )


def _collect_all_ids(persistence: InMemoryFunctionalEvidencePersistence, scope: PipelineEvidenceScope) -> list[str]:
    ids: list[str] = []
    cursor: str | None = None
    while True:
        page = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=2,
                cursor=cursor,
            )
        )
        ids.extend(str(item.evidence_id) for item in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    return ids


def test_unbounded_reconstruction_blocker_reproduction() -> None:
    """Pre-R1 reconstructor materialized full history; bounded projection must not."""
    persistence = _persistence()
    scope = _scope()
    total = 250
    for index in range(total):
        persistence.append(
            _operation_evidence(
                scope,
                operation_name=f"op-{index}",
                recorded_at=_BASE_TIME + timedelta(seconds=index),
            )
        )

    reconstruction = FunctionalEvidenceReconstructor(persistence).reconstruct(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        page_size=50,
    )

    assert reconstruction.evidence_summary.total_evidence_count == total
    assert len(reconstruction.supporting_evidence_refs) <= MAX_SUPPORTING_EVIDENCE_REFS
    assert not hasattr(reconstruction, "evidence")


def test_offset_pagination_late_insert_blocker_reproduction() -> None:
    """Offset cursors duplicate/skip when canonical order shifts after late insert."""
    scope = _scope()
    records = [
        _operation_evidence(scope, operation_name="A", recorded_at=_BASE_TIME),
        _operation_evidence(scope, operation_name="B", recorded_at=_BASE_TIME + timedelta(seconds=1)),
        _operation_evidence(scope, operation_name="C", recorded_at=_BASE_TIME + timedelta(seconds=2)),
        _operation_evidence(scope, operation_name="D", recorded_at=_BASE_TIME + timedelta(seconds=3)),
    ]
    sorted_records = sorted(records, key=functional_evidence_query_order_key)

    page1 = sorted_records[:2]
    offset = 2
    late = _operation_evidence(
        scope,
        operation_name="X",
        recorded_at=_BASE_TIME + timedelta(milliseconds=500),
    )
    after_late = sorted(records + [late], key=functional_evidence_query_order_key)
    page2_offset = after_late[offset : offset + 2]
    names_offset = tuple(item.operation_outcome.operation_name for item in page2_offset if item.operation_outcome)

    assert names_offset[0] == "B"
    assert names_offset.count("B") == 1

    persistence = _persistence()
    for record in records:
        persistence.append(record)
    first_page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=2,
        )
    )
    persistence.append(late)
    second_page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=2,
            cursor=first_page.next_cursor,
        )
    )
    names_keyset = tuple(
        item.operation_outcome.operation_name
        for item in second_page.items
        if item.operation_outcome is not None
    )
    assert names_keyset == ("C", "D")
    assert "B" not in names_keyset


@pytest.mark.parametrize(
    ("test_id", "factory"),
    [
        ("p1_normal_pages", "normal"),
        ("p2_duplicate_append", "duplicate"),
        ("p3_late_insert_after_cursor", "late_after"),
        ("p4_late_insert_before_cursor", "late_before"),
        ("p5_same_timestamp_different_id", "same_ts"),
    ],
)
def test_keyset_pagination_matrix(test_id: str, factory: str) -> None:
    del test_id
    persistence = _persistence()
    scope = _scope()
    a = _operation_evidence(scope, operation_name="A", recorded_at=_BASE_TIME)
    b = _operation_evidence(scope, operation_name="B", recorded_at=_BASE_TIME + timedelta(seconds=1))
    c = _operation_evidence(scope, operation_name="C", recorded_at=_BASE_TIME + timedelta(seconds=2))

    if factory == "normal":
        for record in (a, b, c):
            persistence.append(record)
        ids = _collect_all_ids(persistence, scope)
        assert ids == [str(a.evidence_id), str(b.evidence_id), str(c.evidence_id)]
        return

    if factory == "duplicate":
        persistence.append(a)
        persistence.append(a)
        ids = _collect_all_ids(persistence, scope)
        assert ids == [str(a.evidence_id)]
        return

    if factory == "late_after":
        persistence.append(a)
        persistence.append(b)
        page1 = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=1,
            )
        )
        late = _operation_evidence(
            scope,
            operation_name="late-after",
            recorded_at=_BASE_TIME + timedelta(seconds=5),
        )
        persistence.append(late)
        page2 = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=10,
                cursor=page1.next_cursor,
            )
        )
        names = [item.operation_outcome.operation_name for item in page2.items if item.operation_outcome]
        assert "late-after" in names
        return

    if factory == "late_before":
        persistence.append(a)
        persistence.append(b)
        persistence.append(c)
        page1 = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=2,
            )
        )
        assert page1.next_cursor is not None
        late = _operation_evidence(
            scope,
            operation_name="late-before",
            recorded_at=_BASE_TIME + timedelta(milliseconds=500),
        )
        persistence.append(late)
        page2 = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=10,
                cursor=page1.next_cursor,
            )
        )
        names = [item.operation_outcome.operation_name for item in page2.items if item.operation_outcome]
        assert names == ["C"]
        all_ids = _collect_all_ids(persistence, scope)
        assert str(late.evidence_id) in all_ids
        return

    same_ts_a = _operation_evidence(scope, operation_name="same-a", recorded_at=_BASE_TIME)
    same_ts_b = _operation_evidence(scope, operation_name="same-b", recorded_at=_BASE_TIME)
    if str(same_ts_a.evidence_id) > str(same_ts_b.evidence_id):
        same_ts_a, same_ts_b = same_ts_b, same_ts_a
    persistence.append(same_ts_b)
    persistence.append(same_ts_a)
    ids = _collect_all_ids(persistence, scope)
    assert ids == [str(same_ts_a.evidence_id), str(same_ts_b.evidence_id)]


def test_cursor_scope_mismatch_matrix() -> None:
    persistence = _persistence()
    scope_a = _scope(tenant_id="tenant-a")
    scope_b = _scope(tenant_id="tenant-b", task_id=scope_a.task_id, run_id=scope_a.run_id)
    first = _operation_evidence(scope_a, operation_name="A", recorded_at=_BASE_TIME)
    second = _operation_evidence(scope_a, operation_name="B", recorded_at=_BASE_TIME + timedelta(seconds=1))
    persistence.append(first)
    persistence.append(second)
    page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope_a.tenant_id,
            task_id=scope_a.task_id,
            run_id=scope_a.run_id,
            page_size=1,
        )
    )
    assert page.next_cursor is not None

    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope_b.tenant_id,
                task_id=scope_b.task_id,
                run_id=scope_b.run_id,
                cursor=page.next_cursor,
            )
        )

    other_task_scope = _scope(tenant_id=scope_a.tenant_id, run_id=scope_a.run_id)
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope_a.tenant_id,
                task_id=other_task_scope.task_id,
                run_id=scope_a.run_id,
                cursor=page.next_cursor,
            )
        )

    codec = FunctionalEvidenceQueryCursorCodec(secret=_TEST_CURSOR_SECRET)
    kind_cursor = codec.encode(
        tenant_id=scope_a.tenant_id,
        task_id=scope_a.task_id,
        run_id=scope_a.run_id,
        attempt_id=None,
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        last_recorded_at=first.provenance.recorded_at,
        last_evidence_id=first.evidence_id,
    )
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope_a.tenant_id,
                task_id=scope_a.task_id,
                run_id=scope_a.run_id,
                kind=PipelineEvidenceKind.CANDIDATE_RANK,
                cursor=kind_cursor,
            )
        )

    attempt_scope = _scope(
        tenant_id=scope_a.tenant_id,
        task_id=scope_a.task_id,
        run_id=scope_a.run_id,
        attempt_id=mint_attempt_id(),
    )
    attempt_record_a = _operation_evidence(
        attempt_scope,
        operation_name="attempt-a",
        recorded_at=_BASE_TIME,
    )
    attempt_record_b = _operation_evidence(
        attempt_scope,
        operation_name="attempt-b",
        recorded_at=_BASE_TIME + timedelta(seconds=1),
    )
    persistence.append(attempt_record_a)
    persistence.append(attempt_record_b)
    attempt_page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=attempt_scope.tenant_id,
            task_id=attempt_scope.task_id,
            run_id=attempt_scope.run_id,
            attempt_id=attempt_scope.attempt_id,
            page_size=1,
        )
    )
    assert attempt_page.next_cursor is not None
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=attempt_scope.tenant_id,
                task_id=attempt_scope.task_id,
                run_id=attempt_scope.run_id,
                attempt_id=mint_attempt_id(),
                cursor=attempt_page.next_cursor,
            )
        )

    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope_a.tenant_id,
                task_id=scope_a.task_id,
                run_id=scope_a.run_id,
                cursor="tampered-cursor-token",
            )
        )


def test_empty_requirements_yield_not_evaluated() -> None:
    persistence = _persistence()
    scope = _scope()
    persistence.append(_operation_evidence(scope, operation_name="A", recorded_at=_BASE_TIME))
    reconstruction = FunctionalEvidenceReconstructor(persistence).reconstruct(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert reconstruction.completeness_status is FunctionalEvidenceCompletenessStatus.NOT_EVALUATED


def test_upstream_provenance_overflow_fails_closed() -> None:
    overflow_ids = tuple(mint_event_id() for _ in range(MAX_DIRECT_UPSTREAM_EVIDENCE_REFS + 1))
    with pytest.raises(ValidationError):
        PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
            scope=_scope(),
            provenance=PipelineEvidenceProvenance(
                producer_component="diag.test",
                operation_id="op",
                upstream_evidence_ids=overflow_ids,
            ),
            operation_outcome=PipelineOperationOutcomeFact(
                operation_name="op",
                status=PipelineOperationStatus.SUCCEEDED,
            ),
        )


def test_functional_signal_model_invariant_blocks_misaligned_construction() -> None:
    correlation = DiagnosticExecutionCorrelation(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    validation = FunctionalValidationEvidence(
        validation_id=functional_validation_evidence_id(
            validator_id="oracle.v1",
            validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
            correlation=correlation,
            idempotency_key="attempt-1",
        ),
        validator=FunctionalValidatorRef(validator_id="oracle.v1"),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=FunctionalValidationOutcome.FAILED,
        correlation=correlation,
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
    )

    with pytest.raises(ValidationError):
        PlatformProblemSignal(
            problem_kind=PROBLEM_KIND_PLATFORM_EXCEPTION,
            functional_validation=validation,
            task_id=str(correlation.task_id),
            run_id=str(correlation.run_id),
        )

    with pytest.raises(ValidationError):
        PlatformProblemSignal(
            problem_kind=PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID,
            functional_validation=validation,
            task_id=str(mint_task_id()),
            run_id=str(correlation.run_id),
        )


def test_attempt_scoped_validation_identity_differs() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    base = DiagnosticExecutionCorrelation(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
    )
    attempt_a = DiagnosticExecutionCorrelation(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
    )
    attempt_b = DiagnosticExecutionCorrelation(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
    )
    id_a = functional_validation_evidence_id(
        validator_id="oracle.v1",
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=attempt_a,
        idempotency_key="same-key",
    )
    id_b = functional_validation_evidence_id(
        validator_id="oracle.v1",
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=attempt_b,
        idempotency_key="same-key",
    )
    id_base = functional_validation_evidence_id(
        validator_id="oracle.v1",
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=base,
        idempotency_key="same-key",
    )
    assert id_a != id_b
    assert id_a != id_base


@pytest.mark.no_ci
def test_100k_reconstruction_retains_bounded_refs() -> None:
    persistence = _persistence()
    scope = _scope()
    for index in range(100_000):
        persistence.append(
            _operation_evidence(
                scope,
                operation_name=f"op-{index}",
                recorded_at=_BASE_TIME + timedelta(microseconds=index),
            )
        )

    reconstruction = FunctionalEvidenceReconstructor(persistence).reconstruct(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        page_size=500,
    )

    assert reconstruction.evidence_summary.total_evidence_count == 100_000
    assert len(reconstruction.supporting_evidence_refs) == MAX_SUPPORTING_EVIDENCE_REFS
    assert len(reconstruction.completeness.counts_by_kind) == 1


def test_validation_upstream_overflow_fails_closed() -> None:
    overflow_ids = tuple(mint_event_id() for _ in range(MAX_DIRECT_UPSTREAM_EVIDENCE_REFS + 1))
    correlation = DiagnosticExecutionCorrelation(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    with pytest.raises(ValidationError):
        ObservabilityFunctionalValidationEvidence(
            validation_id=mint_event_id(),
            validator=FunctionalValidatorRef(validator_id="oracle.v1"),
            validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
            outcome=FunctionalValidationOutcome.FAILED,
            correlation=correlation,
            expected_actual_relation=ExpectedActualRelation.CONTAINS,
            upstream_evidence_ids=overflow_ids,
        )
