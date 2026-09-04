# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-READ-R1-R1 projection recovery and crash-safety proofs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_evidence_execution_index import (
    encode_execution_index_v1,
    encode_execution_index_v2,
    execution_index_v1_row_key,
    execution_index_v2_row_key_from_evidence,
)
from intergrax.runtime.diagnostics.functional_evidence_index_rebuilder import (
    FunctionalEvidenceIndexRebuilder,
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
from intergrax.runtime.diagnostics.functional_evidence_projection_state import (
    FunctionalEvidenceProjectionStateStore,
)
from intergrax.runtime.diagnostics.functional_evidence_record_codec import (
    encode_functional_evidence_record,
)

pytestmark = pytest.mark.unit

_CURSOR_SECRET = b"x" * 32
_BASE_TIME = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
_PARTITION_PREFIX = "intergrax.functional_evidence.v1"


def _persistence(store: InMemoryDocumentStore) -> DocumentStoreFunctionalEvidencePersistence:
    return DocumentStoreFunctionalEvidencePersistence(store, cursor_secret=_CURSOR_SECRET)


def _partition_key(tenant_id: str) -> str:
    return f"{_PARTITION_PREFIX}:{tenant_id}"


def _seed_v1_only_legacy(
    store: InMemoryDocumentStore,
    scope,
    count: int,
) -> tuple:
    partition_key = _partition_key(scope.tenant_id)
    fixtures = tuple(
        sample_functional_evidence(
            scope=scope,
            operation_name=f"legacy-{index}",
            recorded_at=_BASE_TIME + timedelta(seconds=index),
        )
        for index in range(count)
    )
    for evidence in fixtures:
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
    return fixtures


def _write_partial_v2(
    store: InMemoryDocumentStore,
    scope,
    fixtures: tuple,
    *,
    count: int,
) -> None:
    partition_key = _partition_key(scope.tenant_id)
    for evidence in fixtures[:count]:
        store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=execution_index_v2_row_key_from_evidence(evidence),
                data=encode_execution_index_v2(evidence),
            ),
        )


def test_v1_only_execution_completes_migration() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-migrate")
    fixtures = _seed_v1_only_legacy(store, scope, 12)
    persistence = _persistence(store)
    collected = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert collected == tuple(sorted(fixtures, key=functional_evidence_query_order_key))
    manifest = FunctionalEvidenceProjectionStateStore(store).load(
        partition_key=_partition_key(scope.tenant_id),
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert manifest is not None
    assert manifest.state == "complete"
    assert manifest.v1_rows_reconciled == 12


def test_partial_v2_without_manifest_repairs_on_query() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-partial")
    fixtures = _seed_v1_only_legacy(store, scope, 20)
    _write_partial_v2(store, scope, fixtures, count=7)
    persistence = _persistence(store)
    collected = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert len(collected) == 20


def test_partial_v2_after_process_like_recreation_repairs() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-restart")
    fixtures = _seed_v1_only_legacy(store, scope, 30)
    _write_partial_v2(store, scope, fixtures, count=11)

    interrupted = False

    def _interrupt_after(written: int) -> bool:
        return written == 5

    rebuilder_a = FunctionalEvidenceIndexRebuilder(
        store,
        interrupt_after_v2_writes=_interrupt_after,
    )
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError, match="interrupted"):
        rebuilder_a.rebuild_execution_index(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            partition_key=_partition_key(scope.tenant_id),
        )
    interrupted = True
    assert interrupted

    persistence_b = _persistence(store)
    collected = collect_all_evidence(
        persistence_b,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert len(collected) == 30


def test_complete_v2_projection_skips_full_rebuild() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-fastpath")
    persistence = _persistence(store)
    for index in range(50):
        persistence.append(
            sample_functional_evidence(
                scope=scope,
                operation_name=f"modern-{index}",
                recorded_at=_BASE_TIME + timedelta(seconds=index),
            ),
        )
    rebuilder = FunctionalEvidenceIndexRebuilder(store)
    result = rebuilder.ensure_v2_projection(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        partition_key=_partition_key(scope.tenant_id),
    )
    assert result is None


def test_concurrent_rebuild_produces_complete_projection() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-concurrent-rebuild")
    _seed_v1_only_legacy(store, scope, 40)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _rebuild() -> None:
        rebuilder = FunctionalEvidenceIndexRebuilder(store)
        try:
            barrier.wait(timeout=5)
            rebuilder.rebuild_execution_index(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                partition_key=_partition_key(scope.tenant_id),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_rebuild), executor.submit(_rebuild)]
        for future in futures:
            future.result(timeout=30)

    assert not errors
    persistence = _persistence(store)
    collected = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert len(collected) == 40


def test_append_during_rebuild_union_is_complete() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-append-during")
    fixtures = _seed_v1_only_legacy(store, scope, 25)
    persistence = _persistence(store)
    late = sample_functional_evidence(
        scope=scope,
        operation_name="late-append",
        recorded_at=_BASE_TIME + timedelta(seconds=100),
    )
    rebuilder = FunctionalEvidenceIndexRebuilder(store, query_page_limit=3)
    rebuilder.rebuild_execution_index(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        partition_key=_partition_key(scope.tenant_id),
    )
    persistence.append(late)
    collected = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    expected_ids = {item.evidence_id for item in fixtures} | {late.evidence_id}
    assert {item.evidence_id for item in collected} == expected_ids


def test_canonical_and_v1_without_v2_repairs_on_query() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-missing-v2")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    partition_key = _partition_key(scope.tenant_id)
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


def test_corrupt_v2_metadata_fails_closed() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-corrupt-v2")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    partition_key = _partition_key(scope.tenant_id)
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
    corrupt = encode_execution_index_v2(evidence)
    corrupt["kind"] = "selection"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v2_row_key_from_evidence(evidence),
            data=corrupt,
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


def test_orphan_v2_without_v1_fails_closed() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-orphan-v2")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    partition_key = _partition_key(scope.tenant_id)
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
            row_key=execution_index_v2_row_key_from_evidence(evidence),
            data=encode_execution_index_v2(evidence),
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


def test_missing_canonical_fails_closed() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-missing-canonical")
    evidence = sample_functional_evidence(scope=scope)
    partition_key = _partition_key(scope.tenant_id)
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


def test_filtered_pagination_after_recovery() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-filtered")
    fixtures = _seed_v1_only_legacy(store, scope, 30)
    _write_partial_v2(store, scope, fixtures, count=4)
    persistence = _persistence(store)
    from intergrax.runtime.diagnostics.functional_evidence import PipelineEvidenceKind

    collected: list = []
    cursor: str | None = None
    while True:
        page = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                kind=PipelineEvidenceKind.OPERATION_OUTCOME,
                page_size=5,
                cursor=cursor,
            ),
        )
        collected.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert len(collected) == 30


def test_cursor_union_correctness_after_recovery() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r1-cursor-union")
    fixtures = _seed_v1_only_legacy(store, scope, 37)
    _write_partial_v2(store, scope, fixtures, count=9)
    persistence = _persistence(store)
    expected = tuple(sorted(fixtures, key=functional_evidence_query_order_key))
    collected: list = []
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
