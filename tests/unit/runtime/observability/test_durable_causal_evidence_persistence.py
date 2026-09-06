# © Artur Czarnecki. All rights reserved.

"""DIAG-1D durable causal evidence persistence conformance, race, and restart tests."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
    CausalEvidencePersistenceConflictError,
    CausalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.observability.causal_evidence_index import (
    execution_index_v2_row_key_from_evidence,
    transport_index_v2_row_key_from_evidence,
)
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    DocumentStoreCausalEvidencePersistence,
    wire_causal_evidence_persistence,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import (
    assert_causal_evidence_conflicting_append_fails_closed,
    assert_causal_evidence_persistence_conformance,
    assert_causal_evidence_paging_conformance,
    assert_causal_evidence_provider_isolation,
    assert_causal_evidence_typed_round_trip,
    sample_causal_evidence,
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
            raise RuntimeError("simulated causal evidence index write failure")
        return super().put_if_absent(document)


class _KV(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        current = self.get(tenant_id, key)
        if expected is None and current is not None:
            return False
        if expected is not None and current != expected:
            return False
        self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
        return True


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("memory", lambda: InMemoryCausalEvidencePersistence(cursor_secret=b"durable-causal-evidence-secret-32")),
        (
            "document_store",
            lambda: DocumentStoreCausalEvidencePersistence(
                InMemoryDocumentStore(cursor_secret=b"durable-causal-evidence-secret-32"),
                cursor_secret=b"durable-causal-evidence-secret-32",
            ),
        ),
    ],
)
def test_causal_evidence_persistence_conformance_matrix(
    label: str,
    factory,
) -> None:
    store: CausalEvidencePersistence = factory()
    try:
        assert_causal_evidence_persistence_conformance(store, label=label)
        assert_causal_evidence_paging_conformance(store, label=label)
        assert_causal_evidence_conflicting_append_fails_closed(store, label=label)
        assert_causal_evidence_provider_isolation(store, label=label)
        assert_causal_evidence_typed_round_trip(store, label=label)
    finally:
        store.close()


def test_wire_selects_document_backend_from_conditional_store() -> None:
    store = InMemoryDocumentStore()
    persistence = wire_causal_evidence_persistence(document_store=store)
    assert isinstance(persistence, DocumentStoreCausalEvidencePersistence)


def test_wire_rejects_kv_store_until_prefix_query_exists() -> None:
    with pytest.raises(ValueError, match="does not support kv_store"):
        wire_causal_evidence_persistence(kv_store=_KV())


def test_wire_rejects_both_capabilities() -> None:
    with pytest.raises(ValueError, match="not both"):
        wire_causal_evidence_persistence(
            kv_store=_KV(),
            document_store=InMemoryDocumentStore(),
        )


def test_wire_rejects_non_conditional_document_store() -> None:
    store = PlainDocumentStore()
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        wire_causal_evidence_persistence(document_store=store)


def test_document_store_restart_survives_new_adapter_instance() -> None:
    store = InMemoryDocumentStore()
    evidence = sample_causal_evidence(
        tenant_id="tenant-restart",
        provider="celery",
        transport_task_id="restart-transport",
    )

    first = DocumentStoreCausalEvidencePersistence(store)
    first.append(evidence)
    first.close()

    second = DocumentStoreCausalEvidencePersistence(store)
    try:
        by_execution = second.list_for_execution(
            tenant_id=evidence.tenant_id,
            task_id=evidence.target.task_id,
            run_id=evidence.target.run_id,
        )
        by_transport = second.list_for_transport_task(
            tenant_id=evidence.tenant_id,
            provider=evidence.source.provider,
            transport_task_id=evidence.source.task_id,
        )
        assert by_execution == (evidence,)
        assert by_transport == (evidence,)
    finally:
        second.close()


def test_document_store_concurrent_append_same_evidence_id() -> None:
    store = InMemoryDocumentStore()
    evidence = sample_causal_evidence(
        tenant_id="tenant-race",
        provider="celery",
        transport_task_id="race-transport",
    )
    barrier = threading.Barrier(2)
    results: list[PlatformCausalEvidence] = []
    errors: list[BaseException] = []

    def _append() -> None:
        persistence = DocumentStoreCausalEvidencePersistence(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.append(evidence))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_append), executor.submit(_append)]
        for future in futures:
            future.result(timeout=10)

    assert not errors
    assert len(results) == 2
    assert results[0] == evidence
    assert results[1] == evidence

    verifier = DocumentStoreCausalEvidencePersistence(store)
    try:
        by_execution = verifier.list_for_execution(
            tenant_id=evidence.tenant_id,
            task_id=evidence.target.task_id,
            run_id=evidence.target.run_id,
        )
        by_transport = verifier.list_for_transport_task(
            tenant_id=evidence.tenant_id,
            provider=evidence.source.provider,
            transport_task_id=evidence.source.task_id,
        )
        assert by_execution == (evidence,)
        assert by_transport == (evidence,)
    finally:
        verifier.close()


def test_document_store_malformed_record_fails_explicitly() -> None:
    store = InMemoryDocumentStore()
    evidence = sample_causal_evidence(
        tenant_id="tenant-malformed",
        provider="celery",
        transport_task_id="malformed-transport",
    )
    persistence = DocumentStoreCausalEvidencePersistence(store)
    persistence.append(evidence)

    partition_key = f"intergrax.causal_evidence.v1:{evidence.tenant_id}"
    row_key = f"exec:{evidence.target.task_id}:{evidence.target.run_id}:{evidence.evidence_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={"schema_version": "broken", "evidence_id": "not-valid"},
        )
    )

    with pytest.raises(CausalEvidencePersistenceIntegrityError):
        persistence.list_for_execution(
            tenant_id=evidence.tenant_id,
            task_id=evidence.target.task_id,
            run_id=evidence.target.run_id,
        )


def test_memory_concurrent_conflicting_append_raises() -> None:
    store = InMemoryCausalEvidencePersistence()
    evidence_id = sample_causal_evidence().evidence_id
    first = sample_causal_evidence(evidence_id=evidence_id, transport_task_id="first")
    second = sample_causal_evidence(evidence_id=evidence_id, transport_task_id="second")
    store.append(first)
    with pytest.raises(CausalEvidencePersistenceConflictError):
        store.append(second)


def test_document_store_concurrent_conflicting_append_one_winner() -> None:
    store = InMemoryDocumentStore()
    evidence_id = sample_causal_evidence().evidence_id
    first = sample_causal_evidence(
        tenant_id="tenant-conflict-race",
        evidence_id=evidence_id,
        transport_task_id="winner-transport",
    )
    second = sample_causal_evidence(
        tenant_id="tenant-conflict-race",
        evidence_id=evidence_id,
        transport_task_id="loser-transport",
    )
    barrier = threading.Barrier(2)
    results: list[PlatformCausalEvidence] = []
    errors: list[BaseException] = []

    def _append(evidence: PlatformCausalEvidence) -> None:
        persistence = DocumentStoreCausalEvidencePersistence(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.append(evidence))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_append, first),
            executor.submit(_append, second),
        ]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], CausalEvidencePersistenceConflictError)
    winner = results[0]
    assert winner in (first, second)

    verifier = DocumentStoreCausalEvidencePersistence(store)
    try:
        by_execution = verifier.list_for_execution(
            tenant_id=winner.tenant_id,
            task_id=winner.target.task_id,
            run_id=winner.target.run_id,
        )
        by_transport = verifier.list_for_transport_task(
            tenant_id=winner.tenant_id,
            provider=winner.source.provider,
            transport_task_id=winner.source.task_id,
        )
        assert by_execution == (winner,)
        assert by_transport == (winner,)
        loser = second if winner == first else first
        assert verifier.list_for_transport_task(
            tenant_id=loser.tenant_id,
            provider=loser.source.provider,
            transport_task_id=loser.source.task_id,
        ) == ()
    finally:
        verifier.close()


def test_document_store_conflicting_append_does_not_create_loser_indexes() -> None:
    store = InMemoryDocumentStore()
    evidence_id = sample_causal_evidence().evidence_id
    original = sample_causal_evidence(
        tenant_id="tenant-conflict-index",
        evidence_id=evidence_id,
        transport_task_id="original-transport",
    )
    conflicting = sample_causal_evidence(
        tenant_id="tenant-conflict-index",
        evidence_id=evidence_id,
        transport_task_id="conflicting-transport",
    )
    persistence = DocumentStoreCausalEvidencePersistence(store)
    persistence.append(original)
    with pytest.raises(CausalEvidencePersistenceConflictError):
        persistence.append(conflicting)

    partition_key = f"intergrax.causal_evidence.v1:{original.tenant_id}"
    loser_transport_key = (
        partition_key,
        f"transport:{conflicting.source.provider}:{conflicting.source.task_id}:{evidence_id}",
    )
    assert store.get(*loser_transport_key) is None


def test_document_store_append_retries_after_first_index_write_failure() -> None:
    evidence = sample_causal_evidence(
        tenant_id="tenant-partial-exec",
        provider="celery",
        transport_task_id="partial-exec-transport",
    )
    partition_key = f"intergrax.causal_evidence.v1:{evidence.tenant_id}"
    exec_key = (
        partition_key,
        execution_index_v2_row_key_from_evidence(evidence),
    )
    store = _FailingPutIfAbsentDocumentStore(fail_keys=frozenset({exec_key}))
    persistence = DocumentStoreCausalEvidencePersistence(store)

    with pytest.raises(RuntimeError, match="simulated causal evidence index write failure"):
        persistence.append(evidence)

    assert DocumentStoreCausalEvidencePersistence(store).append(evidence) == evidence

    verifier = DocumentStoreCausalEvidencePersistence(store)
    try:
        assert verifier.list_for_execution(
            tenant_id=evidence.tenant_id,
            task_id=evidence.target.task_id,
            run_id=evidence.target.run_id,
        ) == (evidence,)
        assert verifier.list_for_transport_task(
            tenant_id=evidence.tenant_id,
            provider=evidence.source.provider,
            transport_task_id=evidence.source.task_id,
        ) == (evidence,)
    finally:
        verifier.close()


def test_document_store_append_retries_after_transport_index_write_failure() -> None:
    evidence = sample_causal_evidence(
        tenant_id="tenant-partial-transport",
        provider="celery",
        transport_task_id="partial-transport",
    )
    partition_key = f"intergrax.causal_evidence.v1:{evidence.tenant_id}"
    transport_key = (
        partition_key,
        transport_index_v2_row_key_from_evidence(evidence),
    )
    store = _FailingPutIfAbsentDocumentStore(fail_keys=frozenset({transport_key}))
    persistence = DocumentStoreCausalEvidencePersistence(store)

    with pytest.raises(RuntimeError, match="simulated causal evidence index write failure"):
        persistence.append(evidence)

    assert persistence.append(evidence) == evidence

    assert persistence.list_for_execution(
        tenant_id=evidence.tenant_id,
        task_id=evidence.target.task_id,
        run_id=evidence.target.run_id,
    ) == (evidence,)
    assert persistence.list_for_transport_task(
        tenant_id=evidence.tenant_id,
        provider=evidence.source.provider,
        transport_task_id=evidence.source.task_id,
    ) == (evidence,)


def test_document_store_orphan_index_without_canonical_fails_closed() -> None:
    store = InMemoryDocumentStore()
    evidence = sample_causal_evidence(
        tenant_id="tenant-orphan",
        provider="celery",
        transport_task_id="orphan-transport",
    )
    partition_key = f"intergrax.causal_evidence.v1:{evidence.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=(
                f"exec:{evidence.target.task_id}:{evidence.target.run_id}:{evidence.evidence_id}"
            ),
            data={
                "schema_version": "intergrax.causal_evidence.index.v1",
                "evidence_id": str(evidence.evidence_id),
            },
        )
    )

    persistence = DocumentStoreCausalEvidencePersistence(store)
    with pytest.raises(CausalEvidencePersistenceIntegrityError):
        persistence.list_for_execution(
            tenant_id=evidence.tenant_id,
            task_id=evidence.target.task_id,
            run_id=evidence.target.run_id,
        )
