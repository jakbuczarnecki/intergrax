# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P1 document-store marketplace qualified tool staging tests."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
    MarketplaceQualifiedToolStageConflictError,
    MarketplaceQualifiedToolStageIntegrityError,
    MarketplaceQualifiedToolStageUnavailableError,
    MarketplaceQualifiedToolStageWriteOutcome,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentQueryPageV1,
    DocumentRecord,
)
from intergrax.tools.marketplace_qualified_capability_staging import (
    DocumentStoreMarketplaceQualifiedToolStageRepository,
)

pytestmark = pytest.mark.unit

_SOURCE = CapabilitySourceIdentity(
    source_id="official.tools.gap02",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _tool_release(
    *,
    version_label: str = "1.0.0",
    discovery_correlation_suffix: str = "",
) -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=f"tools.gap02{discovery_correlation_suffix}",
            ),
        ),
        publisher="publisher",
        version_label=version_label,
        content_digest="sha256:abc",
        package_reference="pkg://gap02",
    )


def _stage(
    *,
    tenant_id: str = "tenant-a",
    handoff_id: str = "handoff-1",
    release: CapabilityReleaseIdentity | None = None,
    discovery_correlation_id: str = "discovery-1",
    selection_id: str = "selection-1",
) -> MarketplaceQualifiedToolStage:
    return MarketplaceQualifiedToolStage(
        handoff_id=handoff_id,
        tenant_id=tenant_id,
        selected_release=release or _tool_release(),
        discovery_correlation_id=discovery_correlation_id,
        selection_id=selection_id,
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id="tool.qualification_staging.v1",
        recorded_at=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
    )


class _PlainDocumentStore:
    """DocumentStore without conditional writes — must be rejected by adapter."""

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return None

    def put(self, document: DocumentRecord) -> None:
        del document

    def delete(self, partition_key: str, row_key: str) -> None:
        del partition_key, row_key

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
        row_key_upper_bound: str | None = None,
        data_equalities: Sequence[DocumentDataEquality] = (),
        sort: Sequence[DocumentDataSort] = (),
    ) -> DocumentQueryPageV1:
        del partition_key, limit, row_key_prefix, cursor, row_key_upper_bound
        del data_equalities, sort
        return DocumentQueryPageV1()

    def close(self) -> None:
        return None


class _FailingReadDocumentStore(InMemoryDocumentStore):
    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        raise RuntimeError("read outage")


class _FailingWriteDocumentStore(InMemoryDocumentStore):
    def put_if_absent(self, document: DocumentRecord) -> bool:
        raise RuntimeError("write outage")


def test_stage_new_record_created() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    result = repo.stage(_stage())
    assert result.outcome is MarketplaceQualifiedToolStageWriteOutcome.CREATED


def test_get_returns_exact_typed_stage() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    original = _stage()
    repo.stage(original)
    loaded = repo.get(tenant_id="tenant-a", handoff_id="handoff-1")
    assert loaded == original


def test_identical_replay_already_staged() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    record = _stage()
    assert repo.stage(record).outcome is MarketplaceQualifiedToolStageWriteOutcome.CREATED
    assert (
        repo.stage(record).outcome
        is MarketplaceQualifiedToolStageWriteOutcome.ALREADY_STAGED_IDENTICAL
    )


def test_conflict_on_different_release() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    repo.stage(_stage())
    with pytest.raises(MarketplaceQualifiedToolStageConflictError):
        repo.stage(_stage(release=_tool_release(version_label="2.0.0")))


def test_conflict_on_different_correlation_or_selection() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    repo.stage(_stage())
    with pytest.raises(MarketplaceQualifiedToolStageConflictError):
        repo.stage(_stage(discovery_correlation_id="discovery-2"))
    with pytest.raises(MarketplaceQualifiedToolStageConflictError):
        repo.stage(_stage(selection_id="selection-2"))


def test_tenant_isolation() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    repo.stage(_stage(tenant_id="tenant-a", handoff_id="shared-handoff"))
    repo.stage(_stage(tenant_id="tenant-b", handoff_id="shared-handoff"))
    assert repo.get(tenant_id="tenant-a", handoff_id="shared-handoff") is not None
    assert repo.get(tenant_id="tenant-c", handoff_id="shared-handoff") is None


def test_missing_record_returns_none() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assert repo.get(tenant_id="tenant-a", handoff_id="missing") is None


def test_malformed_storage_raises_integrity_error() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    record = _stage()
    repo.stage(record)
    partition = f"intergrax.marketplace_qualified_tool_stage.v1:{record.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=record.handoff_id,
            data={"schema_version": "broken", "stage": {}},
        ),
    )
    with pytest.raises(MarketplaceQualifiedToolStageIntegrityError):
        repo.get(tenant_id=record.tenant_id, handoff_id=record.handoff_id)


def test_backend_read_failure_is_unavailable() -> None:
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(
        _FailingReadDocumentStore(),
    )
    with pytest.raises(MarketplaceQualifiedToolStageUnavailableError):
        repo.get(tenant_id="tenant-a", handoff_id="handoff-1")


def test_backend_write_failure_is_unavailable() -> None:
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(
        _FailingWriteDocumentStore(),
    )
    with pytest.raises(MarketplaceQualifiedToolStageUnavailableError):
        repo.stage(_stage())


def test_reinstantiated_repository_reads_same_backing_store() -> None:
    store = InMemoryDocumentStore()
    first = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    record = _stage()
    first.stage(record)
    second = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assert second.get(tenant_id=record.tenant_id, handoff_id=record.handoff_id) == record


def test_adapter_requires_conditional_document_store() -> None:
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        DocumentStoreMarketplaceQualifiedToolStageRepository(_PlainDocumentStore())


def test_concurrent_identical_stage_one_created_one_identical() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    record = _stage()
    outcomes: list[MarketplaceQualifiedToolStageWriteOutcome] = []
    errors: list[BaseException] = []

    def _attempt() -> None:
        try:
            outcomes.append(repo.stage(record).outcome)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert sorted(outcomes, key=lambda item: item.value) == sorted(
        [
            MarketplaceQualifiedToolStageWriteOutcome.CREATED,
            MarketplaceQualifiedToolStageWriteOutcome.ALREADY_STAGED_IDENTICAL,
        ],
        key=lambda item: item.value,
    )


def test_concurrent_conflicting_stage_one_wins_other_conflicts() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    first = _stage(release=_tool_release(version_label="1.0.0"))
    second = _stage(release=_tool_release(version_label="9.9.9"))
    created: list[MarketplaceQualifiedToolStage] = []
    conflicts: list[MarketplaceQualifiedToolStageConflictError] = []

    def _attempt(record: MarketplaceQualifiedToolStage) -> None:
        try:
            result = repo.stage(record)
            assert result.outcome is MarketplaceQualifiedToolStageWriteOutcome.CREATED
            created.append(record)
        except MarketplaceQualifiedToolStageConflictError as exc:
            conflicts.append(exc)

    threads = [
        threading.Thread(target=_attempt, args=(first,)),
        threading.Thread(target=_attempt, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(created) == 1
    assert len(conflicts) == 1
    stored = repo.get(tenant_id=first.tenant_id, handoff_id=first.handoff_id)
    assert stored == created[0]
