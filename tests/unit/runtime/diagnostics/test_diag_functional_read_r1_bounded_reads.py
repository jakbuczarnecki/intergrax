# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-READ-R1 bounded functional evidence read-path proofs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_event_id, mint_run_id, mint_task_id
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_evidence_execution_index import (
    encode_execution_index_v1,
    execution_index_v1_row_key,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceIntegrityError,
    FunctionalEvidenceQueryRequest,
    functional_evidence_query_order_key,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    collect_all_evidence,
    sample_functional_evidence,
    sample_functional_evidence_scope,
)
from intergrax.runtime.diagnostics.functional_evidence_record_codec import encode_functional_evidence_record

pytestmark = pytest.mark.unit

_CURSOR_SECRET = b"x" * 32
_BASE_TIME = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
_PARTITION_PREFIX = "intergrax.functional_evidence.v1"


class _CountingDocumentStore(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__(cursor_secret=_CURSOR_SECRET)
        self.query_calls = 0
        self.get_calls = 0

    def query(self, partition_key: str, *, limit: int, row_key_prefix: str | None = None, cursor: str | None = None):
        self.query_calls += 1
        return super().query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )

    def get(self, partition_key: str, row_key: str):
        self.get_calls += 1
        return super().get(partition_key, row_key)


def _persistence(store: InMemoryDocumentStore) -> DocumentStoreFunctionalEvidencePersistence:
    return DocumentStoreFunctionalEvidencePersistence(store, cursor_secret=_CURSOR_SECRET)


def _operation_evidence(
    scope: PipelineEvidenceScope,
    *,
    recorded_at: datetime,
    evidence_id: str | None = None,
    kind: PipelineEvidenceKind = PipelineEvidenceKind.OPERATION_OUTCOME,
    attempt_id: str | None = None,
) -> PlatformFunctionalEvidence:
    resolved_scope = PipelineEvidenceScope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=attempt_id or scope.attempt_id,
    )
    return PlatformFunctionalEvidence(
        evidence_id=evidence_id or mint_event_id(),
        kind=kind,
        scope=resolved_scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.read-r1",
            operation_id="read-r1-op",
            recorded_at=recorded_at,
        ),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name="read-r1-op",
            status=PipelineOperationStatus.SUCCEEDED,
        ),
    )


def _seed_execution(
    persistence: DocumentStoreFunctionalEvidencePersistence,
    scope: PipelineEvidenceScope,
    count: int,
) -> tuple[PlatformFunctionalEvidence, ...]:
    fixtures = tuple(
        _operation_evidence(
            scope,
            recorded_at=_BASE_TIME + timedelta(seconds=index),
        )
        for index in range(count)
    )
    for evidence in fixtures:
        persistence.append(evidence)
    return fixtures


def _seed_sparse_selection_execution(
    persistence: DocumentStoreFunctionalEvidencePersistence,
    scope: PipelineEvidenceScope,
    count: int,
) -> None:
    for index in range(count):
        kind = (
            PipelineEvidenceKind.SELECTION
            if index % 20 == 0
            else PipelineEvidenceKind.OPERATION_OUTCOME
        )
        persistence.append(
            sample_functional_evidence(
                scope=scope,
                kind=kind,
                operation_name=f"sparse-{index}",
                recorded_at=_BASE_TIME + timedelta(seconds=index),
            ),
        )


def test_first_page_does_not_load_entire_execution() -> None:
    store = _CountingDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="read-r1-bounded")
    persistence = _persistence(store)
    fixtures = _seed_execution(persistence, scope, 1000)

    page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=25,
        ),
    )

    assert len(page.items) == 25
    assert page.next_cursor is not None
    assert store.get_calls <= 50
    assert store.get_calls < len(fixtures)
    assert store.query_calls <= 4


def test_operation_count_gate_e1000_p25() -> None:
    store = _CountingDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="read-r1-opcount")
    persistence = _persistence(store)
    _seed_execution(persistence, scope, 1000)

    persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=25,
        ),
    )

    assert store.get_calls <= 50
    assert store.get_calls < 200


def test_sparse_kind_filter_scans_incrementally() -> None:
    store = _CountingDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="read-r1-sparse")
    persistence = _persistence(store)
    _seed_sparse_selection_execution(persistence, scope, 1000)

    page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            kind=PipelineEvidenceKind.SELECTION,
            page_size=10,
        ),
    )

    assert len(page.items) == 10
    assert all(item.kind is PipelineEvidenceKind.SELECTION for item in page.items)
    assert store.get_calls <= 60
    assert store.query_calls <= 20


def test_cursor_pagination_union_is_complete_and_unique() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="read-r1-cursor")
    persistence = _persistence(store)
    fixtures = _seed_execution(persistence, scope, 37)
    expected = tuple(sorted(fixtures, key=functional_evidence_query_order_key))

    collected: list[PlatformFunctionalEvidence] = []
    cursor: str | None = None
    while True:
        page = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=10,
                cursor=cursor,
            ),
        )
        collected.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert tuple(collected) == expected
    assert len({item.evidence_id for item in collected}) == len(expected)


def test_v1_only_execution_rebuilds_v2_projection_on_query() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="read-r1-migrate")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    partition_key = f"{_PARTITION_PREFIX}:{scope.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"record:{evidence.evidence_id}",
            data=encode_functional_evidence_record(evidence),
        ),
    )
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v1_row_key(
                task_id=scope.task_id,
                run_id=scope.run_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=encode_execution_index_v1(str(evidence.evidence_id)),
        ),
    )
    persistence = _persistence(store)
    collected = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert collected == (evidence,)


def test_orphan_v1_index_without_canonical_fails_closed() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="read-r1-orphan")
    evidence = sample_functional_evidence(scope=scope)
    partition_key = f"{_PARTITION_PREFIX}:{scope.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v1_row_key(
                task_id=scope.task_id,
                run_id=scope.run_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=encode_execution_index_v1(str(evidence.evidence_id)),
        ),
    )
    persistence = _persistence(store)
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        collect_all_evidence(
            persistence,
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )


def test_late_arrival_before_cursor_requires_reconstruction_cycle() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="read-r1-late")
    persistence = _persistence(store)
    first = _operation_evidence(scope, recorded_at=_BASE_TIME + timedelta(seconds=10))
    second = _operation_evidence(scope, recorded_at=_BASE_TIME + timedelta(seconds=20))
    third = _operation_evidence(scope, recorded_at=_BASE_TIME + timedelta(seconds=30))
    persistence.append(first)
    persistence.append(second)
    page1 = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=1,
        ),
    )
    assert page1.items == (first,)
    late = _operation_evidence(scope, recorded_at=_BASE_TIME + timedelta(seconds=5))
    persistence.append(late)
    persistence.append(third)
    page2 = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=10,
            cursor=page1.next_cursor,
        ),
    )
    assert [item.evidence_id for item in page2.items] == [
        second.evidence_id,
        third.evidence_id,
    ]


def test_attempt_filter_across_pages() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="read-r1-attempt")
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    persistence = _persistence(store)
    for index in range(12):
        persistence.append(
            _operation_evidence(
                scope,
                recorded_at=_BASE_TIME + timedelta(seconds=index),
                attempt_id=attempt_a if index % 2 == 0 else attempt_b,
            ),
        )
    collected: list[PlatformFunctionalEvidence] = []
    cursor: str | None = None
    while True:
        page = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                attempt_id=attempt_a,
                page_size=3,
                cursor=cursor,
            ),
        )
        collected.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert len(collected) == 6
    assert all(item.scope.attempt_id == attempt_a for item in collected)
