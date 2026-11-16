# © Artur Czarnecki. All rights reserved.

"""PDC-D7-R016 — the enqueue envelope carries canonical execution identity.

The queue side only CARRIES identity minted by the canonical authority; it never
mints a second task-identity authority (ADR-PDC-07 rejected alternative).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.queueing.contracts.task_queue import TaskExecutionIdentity, TaskRequest
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "pdc.d7.identity"
TENANT = "t-d7"


class _Out(BaseModel):
    ok: bool = True


def _identity() -> TaskExecutionIdentity:
    return TaskExecutionIdentity(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        principal_id="user:alice",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
    )


def _request(identity: TaskExecutionIdentity | None, **overrides) -> TaskRequest:
    params = {
        "tenant_id": TENANT,
        "run_id": "r-transport",
        "task_name": TASK,
        "payload": b"{}",
    }
    params.update(overrides)
    return TaskRequest(identity=identity, **params)


def test_task_request_is_backward_compatible_without_identity() -> None:
    legacy = TaskRequest(tenant_id="t", run_id="r", task_name="n", payload=b"")
    assert legacy.identity is None


def test_identity_envelope_validates_canonical_ids_fail_closed() -> None:
    with pytest.raises(ValueError):
        TaskExecutionIdentity(attempt_id="attempt_not-canonical")
    with pytest.raises(ValueError):
        TaskExecutionIdentity(execution_id="run_" + "0" * 32)  # wrong prefix
    with pytest.raises(ValueError):
        TaskExecutionIdentity(principal_id="   ")
    minted = _identity()
    assert TaskExecutionIdentity.from_dict(minted.to_dict()) == minted


def test_document_store_enqueue_persists_and_returns_identity_on_claim() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    identity = _identity()
    queue.enqueue(_request(identity))

    reopened = DocumentStoreTaskQueue(store)  # durable: a different queue instance claims it
    ((_, claimed),) = reopened.claim_pending(limit=1)
    assert claimed.identity == identity
    assert claimed.run_id == "r-transport"  # transport correlation is untouched


def test_idempotent_enqueue_keeps_the_first_identity() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    first = _identity()
    queue.enqueue(_request(first, idempotency_key="same"))
    queue.enqueue(_request(_identity(), idempotency_key="same"))
    ((_, claimed),) = queue.claim_pending(limit=2)
    assert claimed.identity == first


def test_rows_enqueued_before_this_change_read_back_without_identity() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    handle = queue.enqueue(_request(None))
    partition = f"intergrax.task_queue.v1:{TENANT}"
    record = store.get(partition, handle.task_id)
    assert record is not None
    legacy = {k: v for k, v in record.data.items() if k not in {"identity", "available_at_epoch"}}
    store.put(DocumentRecord(partition_key=partition, row_key=handle.task_id, data=legacy))
    ((_, claimed),) = queue.claim_pending(limit=1)
    assert claimed.identity is None


def test_worker_admits_the_carried_canonical_identity_into_execution() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    seen: list[object] = []

    def handler(*, tenant_id, run_id, payload, idempotency_key, execution_identity):
        _ = tenant_id, payload, idempotency_key
        seen.append((run_id, execution_identity))
        return ToolExecutionResult.ok(_Out())

    registry = TaskExecutionRegistry()
    registry.register(TASK, handler)
    admission = wire_background_execution_admission_dependencies(document_store=store)
    worker = DocumentStoreTaskWorker(
        queue,
        registry,
        identity_persistence=admission.identity_persistence,
        attempt_lifecycle=admission.attempt_lifecycle,
        execution_terminal=admission.execution_terminal,
        causal_evidence_persistence=wire_causal_evidence_persistence(document_store=store),
    )
    identity = _identity()
    queue.enqueue(_request(identity))
    assert worker.drain_once() == 1
    ((handler_run_id, execution_identity),) = seen
    assert execution_identity.task_id == identity.task_id
    assert execution_identity.run_id == identity.run_id
    assert execution_identity.attempt_id == identity.attempt_id
    assert execution_identity.execution_id == identity.execution_id
    assert handler_run_id == identity.run_id


def test_queue_side_mints_no_identity_of_its_own() -> None:
    for relative in (
        "intergrax/queueing/contracts/task_queue.py",
        "intergrax/queueing/providers/document_store/document_store_task_queue.py",
    ):
        source = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "mint_" not in source, relative
