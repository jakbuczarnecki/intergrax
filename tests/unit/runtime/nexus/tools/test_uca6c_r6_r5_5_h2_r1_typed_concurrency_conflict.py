# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.5-H2-R1 — typed suspended store concurrency conflict."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationMutationOutcome,
    SuspendedOperationMutationResult,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)
from intergrax.contracts.execution.suspended_operation.persistence_conflict import (
    SuspendedOperationPersistenceConflictError,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.execution.suspended_operation.document_store_suspended_operation_store import (
    DocumentStoreSuspendedExecutionOperationStore,
)
from intergrax.runtime.nexus.tools.continuation_aware_catalog_tool_host import (
    _prepare_suspended_operation_with_persistence_reconciliation,
)
from tests.unit.runtime.execution.suspended_operation.test_suspended_operation_store import (
    _descriptor,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[5]
DOC_STORE = (
    REPO
    / "intergrax"
    / "runtime"
    / "execution"
    / "suspended_operation"
    / "document_store_suspended_operation_store.py"
)
HOST = (
    REPO
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "continuation_aware_catalog_tool_host.py"
)


class _RecordingPrepareStore(SuspendedExecutionOperationStore):
    def __init__(
        self,
        *,
        outcomes: list[SuspendedOperationMutationResult | BaseException],
    ) -> None:
        self._outcomes = list(outcomes)
        self.prepare_calls = 0

    @property
    def is_durable(self) -> bool:
        return True

    def prepare(
        self,
        descriptor: SuspendedExecutionOperationDescriptor,
    ) -> SuspendedOperationMutationResult:
        self.prepare_calls += 1
        next_item = self._outcomes.pop(0)
        if isinstance(next_item, BaseException):
            raise next_item
        return next_item

    def block(self, **kwargs):  # noqa: ANN003
        raise NotImplementedError

    def load(self, suspended_operation_id: str):
        raise NotImplementedError

    def load_active_for_continuation(self, continuation_id: str):
        raise NotImplementedError

    def load_materialized_for_continuation(self, continuation_id: str):
        raise NotImplementedError

    def load_active_for_logical_invocation(self, logical_invocation_fingerprint):
        raise NotImplementedError

    def claim(self, **kwargs):  # noqa: ANN003
        raise NotImplementedError

    def reclaim(self, **kwargs):  # noqa: ANN003
        raise NotImplementedError

    def mark_consumed(self, **kwargs):  # noqa: ANN003
        raise NotImplementedError

    def abandon(self, **kwargs):  # noqa: ANN003
        raise NotImplementedError

    def authority_reblock_from_claimed(self, **kwargs):  # noqa: ANN003
        raise NotImplementedError


def test_provider_put_if_absent_lost_race_raises_typed_conflict() -> None:
    document_store = InMemoryDocumentStore()
    store = DocumentStoreSuspendedExecutionOperationStore(document_store)
    document_store.put_if_absent = MagicMock(return_value=False)
    with pytest.raises(SuspendedOperationPersistenceConflictError):
        store.prepare(_descriptor())


def test_provider_corrupt_backing_not_typed_conflict() -> None:
    document_store = InMemoryDocumentStore()
    document_store.put(
        DocumentRecord(
            partition_key="execution.suspended_operations",
            row_key="backing",
            data={"backing": "not-a-dict"},
        ),
    )
    with pytest.raises(
        RuntimeError, match="corrupt suspended operation durable backing"
    ):
        DocumentStoreSuspendedExecutionOperationStore(document_store)


def test_provider_unknown_schema_not_typed_conflict() -> None:
    document_store = InMemoryDocumentStore()
    document_store.put(
        DocumentRecord(
            partition_key="execution.suspended_operations",
            row_key="backing",
            data={"backing": {"schema_version": "unknown", "records": {}}},
        ),
    )
    with pytest.raises(
        RuntimeError, match="unknown suspended operation durable schema"
    ):
        DocumentStoreSuspendedExecutionOperationStore(document_store)


def test_host_reconcile_once_then_already_active() -> None:
    descriptor = _descriptor()
    existing = descriptor.model_copy(
        update={"materialization_revision": 1},
    )
    store = _RecordingPrepareStore(
        outcomes=[
            SuspendedOperationPersistenceConflictError("race"),
            SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.ALREADY_ACTIVE,
                descriptor=existing,
            ),
        ],
    )
    prepared = _prepare_suspended_operation_with_persistence_reconciliation(
        store,
        descriptor,
    )
    assert store.prepare_calls == 2
    assert prepared.outcome is SuspendedOperationMutationOutcome.ALREADY_ACTIVE
    assert prepared.descriptor is existing


def test_host_second_conflict_propagates() -> None:
    descriptor = _descriptor()
    store = _RecordingPrepareStore(
        outcomes=[
            SuspendedOperationPersistenceConflictError("stale"),
            SuspendedOperationPersistenceConflictError("stale"),
        ],
    )
    with pytest.raises(SuspendedOperationPersistenceConflictError):
        _prepare_suspended_operation_with_persistence_reconciliation(store, descriptor)
    assert store.prepare_calls == 2


def test_host_corruption_not_retried() -> None:
    descriptor = _descriptor()
    store = _RecordingPrepareStore(
        outcomes=[RuntimeError("corrupt suspended operation durable backing")],
    )
    with pytest.raises(RuntimeError, match="corrupt"):
        _prepare_suspended_operation_with_persistence_reconciliation(store, descriptor)
    assert store.prepare_calls == 1


def test_document_store_maps_cas_failures_to_typed_conflict() -> None:
    source = DOC_STORE.read_text(encoding="utf-8")
    assert "raise SuspendedOperationPersistenceConflictError" in source
    assert 'RuntimeError("suspended operation durable persist race")' not in source
    assert 'RuntimeError("suspended operation durable persist stale")' not in source


def test_host_does_not_import_document_store_provider() -> None:
    source = HOST.read_text(encoding="utf-8")
    assert "DocumentStoreSuspendedExecutionOperationStore" not in source


def test_replace_if_match_stale_raises_typed_conflict() -> None:
    document_store = InMemoryDocumentStore()
    inner = DocumentStoreSuspendedExecutionOperationStore(document_store)
    descriptor = _descriptor()
    inner.prepare(descriptor)
    document_store.replace_if_match = MagicMock(return_value=False)
    other = _descriptor()
    with pytest.raises(SuspendedOperationPersistenceConflictError):
        inner.prepare(other)
