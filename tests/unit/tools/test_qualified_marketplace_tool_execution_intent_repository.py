# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading

import pytest

from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntent,
    QualifiedMarketplaceToolExecutionIntentConflictError,
    QualifiedMarketplaceToolExecutionIntentIntegrityError,
    QualifiedMarketplaceToolExecutionIntentUnavailableError,
    QualifiedMarketplaceToolExecutionIntentWriteOutcome,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.tools.qualified_marketplace_tool_execution_intent_repository import (
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
)

pytestmark = pytest.mark.unit


def _intent(**updates: object) -> QualifiedMarketplaceToolExecutionIntent:
    base = QualifiedMarketplaceToolExecutionIntent(
        execution_request_id="exec-req-1",
        binding_operation_id="bind-1",
        resume_operation_id="resume-1",
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        worker_need_id="worker-need-1",
        qualified_subject_reference="qualified-capability-subject:q:domain_handoff_reference:h",
        handoff_id="handoff-1",
        selected_operation="invoke",
    )
    if updates:
        return base.model_copy(update=updates)
    return base


class _FailingStore(InMemoryDocumentStore):
    def put_if_absent(self, document: DocumentRecord) -> bool:
        raise RuntimeError("write failed")

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        raise RuntimeError("read failed")


def test_created() -> None:
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(
        InMemoryDocumentStore(),
    )
    result = repo.record(_intent())
    assert result.outcome is QualifiedMarketplaceToolExecutionIntentWriteOutcome.CREATED


def test_identical_replay() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    repo.record(_intent())
    result = repo.record(_intent())
    assert (
        result.outcome
        is QualifiedMarketplaceToolExecutionIntentWriteOutcome.ALREADY_RECORDED_IDENTICAL
    )


def test_conflicting_replay() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    repo.record(_intent())
    with pytest.raises(QualifiedMarketplaceToolExecutionIntentConflictError):
        repo.record(_intent(selected_operation="other-op"))


def test_get_existing_and_missing() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    assert repo.get(execution_request_id="exec-req-1") is None
    repo.record(_intent())
    assert repo.get(execution_request_id="exec-req-1") == _intent()


def test_corrupt_persistence_integrity() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    store.put(
        DocumentRecord(
            partition_key="intergrax.qualified_marketplace_tool_execution_intent.v1",
            row_key="exec-req-1",
            data={"schema_version": "wrong", "intent": {}},
        ),
    )
    with pytest.raises(QualifiedMarketplaceToolExecutionIntentIntegrityError):
        repo.get(execution_request_id="exec-req-1")


def test_backend_write_and_read_failure() -> None:
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(_FailingStore())
    with pytest.raises(QualifiedMarketplaceToolExecutionIntentUnavailableError):
        repo.record(_intent())
    with pytest.raises(QualifiedMarketplaceToolExecutionIntentUnavailableError):
        repo.get(execution_request_id="exec-req-1")


def test_restart_new_repository_same_store() -> None:
    store = InMemoryDocumentStore()
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store).record(_intent())
    loaded = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(
        store,
    ).get(execution_request_id="exec-req-1")
    assert loaded == _intent()


def test_concurrent_identical_writes() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            repo.record(_intent())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert repo.get(execution_request_id="exec-req-1") == _intent()


def test_concurrent_conflicting_writes() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    caught: list[Exception] = []

    def worker(op: str) -> None:
        try:
            repo.record(_intent(selected_operation=op))
        except Exception as exc:
            caught.append(exc)

    threads = [
        threading.Thread(target=worker, args=("op-a",)),
        threading.Thread(target=worker, args=("op-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert any(
        isinstance(exc, QualifiedMarketplaceToolExecutionIntentConflictError)
        for exc in caught
    )
