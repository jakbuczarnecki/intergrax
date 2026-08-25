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
    assert_causal_evidence_provider_isolation,
    assert_causal_evidence_typed_round_trip,
    sample_causal_evidence,
)
from tests.unit.runtime.vendor_knowledge._fakes import (
    InMemoryDocumentStore as PlainDocumentStore,
)

pytestmark = pytest.mark.unit


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
        ("memory", lambda: InMemoryCausalEvidencePersistence()),
        (
            "document_store",
            lambda: DocumentStoreCausalEvidencePersistence(InMemoryDocumentStore()),
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
            data={"schema_version": "broken", "payload": "not-a-dict"},
        )
    )

    with pytest.raises(ValueError, match="causal evidence"):
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
