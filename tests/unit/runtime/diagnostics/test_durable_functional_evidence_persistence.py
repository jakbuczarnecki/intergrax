# © Artur Czarnecki. All rights reserved.

"""DIAG-DURABILITY-D1 durable functional evidence persistence conformance tests."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
    wire_functional_evidence_persistence,
)
from intergrax.runtime.diagnostics.functional_evidence import PlatformFunctionalEvidence
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidencePersistenceConflictError,
    FunctionalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    assert_functional_evidence_conflicting_append_fails_closed,
    assert_functional_evidence_cross_domain_round_trip,
    assert_functional_evidence_persistence_conformance,
    assert_functional_evidence_tenant_run_isolation,
    collect_all_evidence,
    sample_functional_evidence,
    sample_functional_evidence_scope,
)
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from tests.unit.runtime.vendor_knowledge._fakes import (
    InMemoryDocumentStore as PlainDocumentStore,
)

pytestmark = pytest.mark.unit

_CURSOR_SECRET = b"x" * 32


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
            raise RuntimeError("simulated functional evidence index write failure")
        return super().put_if_absent(document)


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        (
            "memory",
            lambda: InMemoryFunctionalEvidencePersistence(cursor_secret=_CURSOR_SECRET),
        ),
        (
            "document_store",
            lambda: DocumentStoreFunctionalEvidencePersistence(
                InMemoryDocumentStore(),
                cursor_secret=_CURSOR_SECRET,
            ),
        ),
    ],
)
def test_functional_evidence_persistence_conformance_matrix(
    label: str,
    factory,
) -> None:
    store: FunctionalEvidencePersistence = factory()
    assert_functional_evidence_persistence_conformance(store, label=label)
    assert_functional_evidence_conflicting_append_fails_closed(store, label=label)
    assert_functional_evidence_tenant_run_isolation(store, label=label)
    assert_functional_evidence_cross_domain_round_trip(store, label=label)


def test_wire_selects_document_backend_from_conditional_store() -> None:
    store = InMemoryDocumentStore()
    persistence = wire_functional_evidence_persistence(
        document_store=store,
        cursor_secret=_CURSOR_SECRET,
    )
    assert isinstance(persistence, DocumentStoreFunctionalEvidencePersistence)


def test_wire_rejects_non_conditional_document_store() -> None:
    store = PlainDocumentStore()
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        wire_functional_evidence_persistence(
            document_store=store,
            cursor_secret=_CURSOR_SECRET,
        )


def test_wire_requires_document_store() -> None:
    with pytest.raises(ValueError, match="requires document_store"):
        wire_functional_evidence_persistence(cursor_secret=_CURSOR_SECRET)


def test_document_store_restart_survives_new_adapter_instance() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="tenant-restart")
    evidence = sample_functional_evidence(scope=scope)

    first = DocumentStoreFunctionalEvidencePersistence(store, cursor_secret=_CURSOR_SECRET)
    first.append(evidence)

    second = DocumentStoreFunctionalEvidencePersistence(store, cursor_secret=_CURSOR_SECRET)
    collected = collect_all_evidence(
        second,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert collected == (evidence,)


def test_document_store_concurrent_append_same_evidence_id() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="tenant-race")
    evidence = sample_functional_evidence(scope=scope)
    barrier = threading.Barrier(2)
    results: list[PlatformFunctionalEvidence] = []
    errors: list[BaseException] = []

    def _append() -> None:
        persistence = DocumentStoreFunctionalEvidencePersistence(
            store,
            cursor_secret=_CURSOR_SECRET,
        )
        try:
            barrier.wait(timeout=5)
            results.append(persistence.append(evidence))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_append), executor.submit(_append)]
        for future in futures:
            future.result(timeout=10)

    assert not errors
    assert len(results) == 2
    assert results[0] == evidence
    assert results[1] == evidence


def test_document_store_concurrent_conflicting_append_one_winner() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="tenant-conflict-race")
    evidence_id = sample_functional_evidence(scope=scope).evidence_id
    first = sample_functional_evidence(
        evidence_id=evidence_id,
        scope=scope,
        operation_name="winner",
    )
    second = sample_functional_evidence(
        evidence_id=evidence_id,
        scope=scope,
        operation_name="loser",
    )
    barrier = threading.Barrier(2)
    results: list[PlatformFunctionalEvidence] = []
    errors: list[BaseException] = []

    def _append(item: PlatformFunctionalEvidence) -> None:
        persistence = DocumentStoreFunctionalEvidencePersistence(
            store,
            cursor_secret=_CURSOR_SECRET,
        )
        try:
            barrier.wait(timeout=5)
            results.append(persistence.append(item))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_append, first),
            executor.submit(_append, second),
        ]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], FunctionalEvidencePersistenceConflictError)


def test_document_store_append_retries_after_index_write_failure() -> None:
    scope = sample_functional_evidence_scope(tenant_id="tenant-partial-index")
    evidence = sample_functional_evidence(scope=scope)
    partition_key = f"intergrax.functional_evidence.v1:{scope.tenant_id}"
    exec_key = (
        partition_key,
        f"exec:{scope.task_id}:{scope.run_id}:{evidence.evidence_id}",
    )
    store = _FailingPutIfAbsentDocumentStore(fail_keys=frozenset({exec_key}))
    persistence = DocumentStoreFunctionalEvidencePersistence(
        store,
        cursor_secret=_CURSOR_SECRET,
    )

    with pytest.raises(RuntimeError, match="simulated functional evidence index write failure"):
        persistence.append(evidence)

    repaired = DocumentStoreFunctionalEvidencePersistence(store, cursor_secret=_CURSOR_SECRET)
    assert repaired.append(evidence) == evidence
    assert collect_all_evidence(
        repaired,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    ) == (evidence,)


def test_document_store_orphan_index_without_canonical_fails_closed() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="tenant-orphan")
    evidence = sample_functional_evidence(scope=scope)
    partition_key = f"intergrax.functional_evidence.v1:{scope.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"exec:{scope.task_id}:{scope.run_id}:{evidence.evidence_id}",
            data={
                "schema_version": "intergrax.functional_evidence.index.v1",
                "evidence_id": str(evidence.evidence_id),
            },
        )
    )

    persistence = DocumentStoreFunctionalEvidencePersistence(
        store,
        cursor_secret=_CURSOR_SECRET,
    )
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        collect_all_evidence(
            persistence,
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )


def test_document_store_malformed_record_fails_explicitly() -> None:
    store = InMemoryDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="tenant-malformed")
    evidence = sample_functional_evidence(scope=scope)
    persistence = DocumentStoreFunctionalEvidencePersistence(
        store,
        cursor_secret=_CURSOR_SECRET,
    )
    persistence.append(evidence)

    partition_key = f"intergrax.functional_evidence.v1:{scope.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"record:{evidence.evidence_id}",
            data={"schema_version": "broken", "payload": {}},
        )
    )
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        collect_all_evidence(
            persistence,
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )


def test_document_store_query_paginates_beyond_single_backend_page() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreFunctionalEvidencePersistence(
        store,
        cursor_secret=_CURSOR_SECRET,
        query_page_limit=2,
    )
    scope = sample_functional_evidence_scope(tenant_id="tenant-pagination")
    fixtures = tuple(
        sample_functional_evidence(
            scope=scope,
            operation_name=f"op-{index}",
        )
        for index in range(5)
    )
    for evidence in fixtures:
        persistence.append(evidence)
    collected = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        page_size=2,
    )
    assert len(collected) == 5
    assert {item.evidence_id for item in collected} == {item.evidence_id for item in fixtures}
