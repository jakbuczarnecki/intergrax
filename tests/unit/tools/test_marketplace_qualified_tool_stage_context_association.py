# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P2 context association repository tests."""

from __future__ import annotations

import threading

import pytest

from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
    MarketplaceQualifiedToolStageContextAssociationConflictError,
    MarketplaceQualifiedToolStageContextAssociationIntegrityError,
    MarketplaceQualifiedToolStageContextAssociationUnavailableError,
    MarketplaceQualifiedToolStageContextAssociationWriteOutcome,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.tools.marketplace_qualified_tool_stage_context_association import (
    DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository,
)

pytestmark = pytest.mark.unit


def _association(
    *,
    handoff_id: str = "marketplace-gap-handoff:v2:abc",
    tenant_id: str = "tenant-a",
    acquisition_request_id: str = "acq-1",
) -> MarketplaceQualifiedToolStageContext:
    return MarketplaceQualifiedToolStageContext(
        handoff_id=handoff_id,
        tenant_id=tenant_id,
        acquisition_request_id=acquisition_request_id,
    )


class _FailingReadDocumentStore(InMemoryDocumentStore):
    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        raise RuntimeError("read outage")


class _FailingWriteDocumentStore(InMemoryDocumentStore):
    def put_if_absent(self, document: DocumentRecord) -> bool:
        raise RuntimeError("write outage")


def test_create_association() -> None:
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        InMemoryDocumentStore(),
    )
    result = repo.record(_association())
    assert result.outcome is MarketplaceQualifiedToolStageContextAssociationWriteOutcome.CREATED


def test_identical_replay() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    record = _association()
    repo.record(record)
    replay = repo.record(record)
    assert (
        replay.outcome
        is MarketplaceQualifiedToolStageContextAssociationWriteOutcome.ALREADY_RECORDED_IDENTICAL
    )


def test_conflicting_replay() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    repo.record(_association(tenant_id="tenant-a"))
    with pytest.raises(MarketplaceQualifiedToolStageContextAssociationConflictError):
        repo.record(_association(tenant_id="tenant-b"))


def test_get_existing() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    record = _association()
    repo.record(record)
    assert repo.get_by_handoff_id(record.handoff_id) == record


def test_get_missing() -> None:
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        InMemoryDocumentStore(),
    )
    assert repo.get_by_handoff_id("missing") is None


def test_malformed_payload_integrity() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    record = _association()
    repo.record(record)
    store.put(
        DocumentRecord(
            partition_key="intergrax.marketplace_qualified_tool_stage_context_association.v1",
            row_key=record.handoff_id,
            data={"schema_version": "broken", "association": {}},
        ),
    )
    with pytest.raises(MarketplaceQualifiedToolStageContextAssociationIntegrityError):
        repo.get_by_handoff_id(record.handoff_id)


def test_write_failure_unavailable() -> None:
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        _FailingWriteDocumentStore(),
    )
    with pytest.raises(MarketplaceQualifiedToolStageContextAssociationUnavailableError):
        repo.record(_association())


def test_read_failure_unavailable() -> None:
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        _FailingReadDocumentStore(),
    )
    with pytest.raises(MarketplaceQualifiedToolStageContextAssociationUnavailableError):
        repo.get_by_handoff_id("handoff")


def test_new_repository_instance_same_store() -> None:
    store = InMemoryDocumentStore()
    first = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    record = _association()
    first.record(record)
    second = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    assert second.get_by_handoff_id(record.handoff_id) == record


def test_concurrent_identical_writes() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    record = _association()
    outcomes: list[MarketplaceQualifiedToolStageContextAssociationWriteOutcome] = []

    def _attempt() -> None:
        outcomes.append(repo.record(record).outcome)

    threads = [threading.Thread(target=_attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes, key=lambda item: item.value) == sorted(
        [
            MarketplaceQualifiedToolStageContextAssociationWriteOutcome.CREATED,
            MarketplaceQualifiedToolStageContextAssociationWriteOutcome.ALREADY_RECORDED_IDENTICAL,
        ],
        key=lambda item: item.value,
    )


def test_concurrent_conflicting_writes() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    conflicts: list[MarketplaceQualifiedToolStageContextAssociationConflictError] = []
    created: list[MarketplaceQualifiedToolStageContext] = []

    def _attempt(record: MarketplaceQualifiedToolStageContext) -> None:
        try:
            repo.record(record)
            created.append(record)
        except MarketplaceQualifiedToolStageContextAssociationConflictError as exc:
            conflicts.append(exc)

    threads = [
        threading.Thread(
            target=_attempt,
            args=(_association(tenant_id="tenant-a"),),
        ),
        threading.Thread(
            target=_attempt,
            args=(_association(tenant_id="tenant-b"),),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(created) == 1
    assert len(conflicts) == 1


def test_handoff_key_preserved() -> None:
    handoff_id = "marketplace-gap-handoff:v2:deadbeef"
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    record = _association(handoff_id=handoff_id)
    repo.record(record)
    loaded = repo.get_by_handoff_id(handoff_id)
    assert loaded is not None
    assert loaded.handoff_id == handoff_id
