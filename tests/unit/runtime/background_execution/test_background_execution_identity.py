# © Artur Czarnecki. All rights reserved.

"""BG-EXEC-2 — retry/redelivery identity resolution tests."""

from __future__ import annotations

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.background_execution.bootstrap import (
    BackgroundExecutionIdentity,
    bootstrap_background_execution,
    resolve_background_execution,
)
from intergrax.runtime.background_execution.identity_persistence import (
    DocumentStoreBackgroundExecutionIdentityPersistence,
    KvBackgroundExecutionIdentityPersistence,
    wire_background_execution_identity_persistence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from tests.unit.runtime.vendor_knowledge._fakes import InMemoryDocumentStore as PlainDocumentStore

import inspect
import importlib
from pathlib import Path

import pytest

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


def _transport(
    *,
    tenant_id: str = "tenant-a",
    provider: str = "celery",
    transport_task_id: str = "transport-1",
) -> BackgroundTransportExecutionRef:
    return BackgroundTransportExecutionRef(
        tenant_id=tenant_id,
        provider=provider,
        transport_task_id=transport_task_id,
    )


def test_initial_execution_mints_task_run_and_attempt() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport = _transport()

    first = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )

    assert first.tenant_id == "tenant-a"
    assert str(first.task_id).startswith("task_")
    assert str(first.run_id).startswith("run_")
    assert str(first.attempt_id).startswith("attempt_")


def test_retry_preserves_task_and_run_but_mints_new_attempt() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport = _transport()

    first = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )
    second = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )

    assert second.task_id == first.task_id
    assert second.run_id == first.run_id
    assert second.attempt_id != first.attempt_id


def test_third_attempt_still_preserves_task_and_run() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport = _transport()

    first = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )
    second = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )
    third = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )

    assert third.task_id == first.task_id
    assert third.run_id == first.run_id
    assert len({first.attempt_id, second.attempt_id, third.attempt_id}) == 3


def test_different_transport_task_mints_new_task_and_run() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)

    first = bootstrap_background_execution(
        transport_ref=_transport(transport_task_id="transport-1"),
        identity_persistence=persistence,
    )
    second = bootstrap_background_execution(
        transport_ref=_transport(transport_task_id="transport-2"),
        identity_persistence=persistence,
    )

    assert second.task_id != first.task_id
    assert second.run_id != first.run_id


def test_same_transport_id_different_provider_mints_new_task_and_run() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)

    first = bootstrap_background_execution(
        transport_ref=_transport(provider="celery"),
        identity_persistence=persistence,
    )
    second = bootstrap_background_execution(
        transport_ref=_transport(provider="document_store"),
        identity_persistence=persistence,
    )

    assert second.task_id != first.task_id
    assert second.run_id != first.run_id


def test_same_provider_and_transport_id_different_tenant_mints_new_task_and_run() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)

    first = bootstrap_background_execution(
        transport_ref=_transport(tenant_id="tenant-a"),
        identity_persistence=persistence,
    )
    second = bootstrap_background_execution(
        transport_ref=_transport(tenant_id="tenant-b"),
        identity_persistence=persistence,
    )

    assert second.task_id != first.task_id
    assert second.run_id != first.run_id


def test_process_reconstruction_uses_shared_kv_persistence() -> None:
    kv = _KV()
    transport = _transport()
    first_resolver = KvBackgroundExecutionIdentityPersistence(kv)
    second_resolver = KvBackgroundExecutionIdentityPersistence(kv)

    first = resolve_background_execution(
        transport_ref=transport,
        identity_persistence=first_resolver,
    )
    second = resolve_background_execution(
        transport_ref=transport,
        identity_persistence=second_resolver,
    )

    assert second.task_id == first.task_id
    assert second.run_id == first.run_id
    assert second.attempt_id != first.attempt_id


def test_document_store_persistence_survives_new_instance() -> None:
    store = InMemoryDocumentStore()
    transport = _transport(provider="document_store", transport_task_id="dstq_abc")
    first_resolver = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    second_resolver = DocumentStoreBackgroundExecutionIdentityPersistence(store)

    first = resolve_background_execution(
        transport_ref=transport,
        identity_persistence=first_resolver,
    )
    second = resolve_background_execution(
        transport_ref=transport,
        identity_persistence=second_resolver,
    )

    assert second.task_id == first.task_id
    assert second.run_id == first.run_id
    assert second.attempt_id != first.attempt_id


def test_broker_redelivery_preserves_task_and_run() -> None:
    import base64
    import json

    from intergrax.queueing.providers.broker_worker_base import BrokerWorkerBase
    from intergrax.queueing.worker.registry import TaskExecutionRegistry
    from intergrax.tools.execution_models import ToolExecutionResult
    from pydantic import BaseModel

    class _Output(BaseModel):
        value: str = "ok"

    class _Worker(BrokerWorkerBase):
        def start(self) -> None:
            raise NotImplementedError

    kv = _KV()
    registry = TaskExecutionRegistry()
    captured: list[BackgroundExecutionIdentity] = []

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity,
    ):
        _ = tenant_id, run_id, payload, idempotency_key
        captured.append(execution_identity)
        return ToolExecutionResult.ok(_Output())

    registry.register("demo.task.v1", handler)
    worker = _Worker(
        registry=registry,
        kv_store=kv,
        provider_name="rabbitmq",
        identity_persistence=KvBackgroundExecutionIdentityPersistence(kv),
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )
    message = {
        "task_id": "broker-transport-1",
        "tenant_id": "tenant-a",
        "run_id": "queue-correlation",
        "task_name": "demo.task.v1",
        "provider": "rabbitmq",
        "payload": base64.b64encode(b"input").decode("ascii"),
        "idempotency_key": None,
    }
    payload = json.dumps(message).encode("utf-8")

    worker.process_message(raw_payload=payload)
    worker.process_message(raw_payload=payload)

    assert len(captured) == 2
    assert captured[1].task_id == captured[0].task_id
    assert captured[1].run_id == captured[0].run_id
    assert captured[1].attempt_id != captured[0].attempt_id


def test_bootstrap_persists_task_and_run_in_identity_store() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport = _transport()

    identity = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )

    stored = kv.get(
        tenant_id="tenant-a",
        key="bg_exec_identity:celery:transport-1",
    )
    assert stored is not None
    assert str(identity.task_id).encode("utf-8") in stored
    assert str(identity.run_id).encode("utf-8") in stored


def test_plain_document_store_rejected_for_identity_persistence() -> None:
    store = PlainDocumentStore()
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        DocumentStoreBackgroundExecutionIdentityPersistence(store)


_EXECUTION_MODULES = (
    "intergrax.queueing.providers.broker_worker_base",
    "intergrax.background_tasks.worker_runtime",
    "intergrax.queueing.worker.dispatcher",
    "intergrax.queueing.providers.document_store.colocated_worker",
)


@pytest.mark.parametrize("module_name", _EXECUTION_MODULES)
def test_execution_entry_points_depend_on_abstraction_only(module_name: str) -> None:
    module = importlib.import_module(module_name)
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    assert "KvBackgroundExecutionIdentityPersistence" not in source
    assert "DocumentStoreBackgroundExecutionIdentityPersistence" not in source
    assert "KvBackgroundExecutionIdentityPersistence(" not in source
    assert "DocumentStoreBackgroundExecutionIdentityPersistence(" not in source
    assert "threading.Lock" not in source


def test_wire_selects_kv_backend_from_distributed_store() -> None:
    kv = _KV()
    persistence = wire_background_execution_identity_persistence(kv_store=kv)
    assert isinstance(persistence, KvBackgroundExecutionIdentityPersistence)


def test_wire_selects_document_backend_from_conditional_store() -> None:
    store = InMemoryDocumentStore()
    persistence = wire_background_execution_identity_persistence(document_store=store)
    assert isinstance(persistence, DocumentStoreBackgroundExecutionIdentityPersistence)


def test_wire_rejects_non_conditional_document_store() -> None:
    store = PlainDocumentStore()
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        wire_background_execution_identity_persistence(document_store=store)
