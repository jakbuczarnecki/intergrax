# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package F — out-of-process worker idempotency topology admission."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from intergrax.applications._shared.production_queue_resolver import (
    ProductionQueueBackend,
    UnsupportedQueueBackendError,
    UnsupportedQueueBackendReason,
)
from intergrax.applications._shared.queue_worker_wiring import wire_optional_queue_execution
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from testing_support.cross_process_spine.durable_document_store import SqliteFileDocumentStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _DurableKV(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    @property
    def survives_process_restart(self) -> bool:
        return True

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(self, tenant_id: str, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        _ = ttl_seconds
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
        _ = ttl_seconds
        if self._data.get((tenant_id, key)) != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        return True


@pytest.fixture
def celery_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    celery_stub = types.ModuleType("celery")
    celery_stub.Celery = MagicMock()
    monkeypatch.setitem(sys.modules, "celery", celery_stub)
    monkeypatch.setattr(
        "intergrax.runtime.task.worker_bootstrap.create_nexus_celery_worker_app",
        MagicMock(),
    )
    monkeypatch.setenv("INTERGRAX_CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")


def test_sqlite_idempotency_satisfies_durable_single_host(
    tmp_path,
    celery_stub: None,
) -> None:
    document_store = SqliteFileDocumentStore(
        tmp_path / "tasks.db",
        cursor_secret=b"arch-freeze-f-topology-secret-32b",
    )
    idempotency = SQLiteIdempotencyStore(str(tmp_path / "idempotency.sqlite3"))
    wiring = wire_optional_queue_execution(
        enabled=True,
        host_execution=MagicMock(),
        registry=MagicMock(),
        run_service=MagicMock(),
        queue_backend=ProductionQueueBackend.CELERY,
        kv_store=_DurableKV(),
        causal_evidence_persistence=MagicMock(spec=CausalEvidencePersistence),
        document_store=document_store,
        idempotency_store=idempotency,
        required_idempotency_persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        production_mode=True,
    )
    assert wiring.worker_kind == "out_of_process_durable"


def test_underpowered_sqlite_topology_rejected_for_shared_requirement(
    tmp_path,
    celery_stub: None,
) -> None:
    document_store = SqliteFileDocumentStore(
        tmp_path / "tasks.db",
        cursor_secret=b"arch-freeze-f-topology-secret-32b",
    )
    idempotency = SQLiteIdempotencyStore(str(tmp_path / "idempotency.sqlite3"))
    with pytest.raises(UnsupportedQueueBackendError) as refusal:
        wire_optional_queue_execution(
            enabled=True,
            host_execution=MagicMock(),
            registry=MagicMock(),
            run_service=MagicMock(),
            queue_backend=ProductionQueueBackend.CELERY,
            kv_store=_DurableKV(),
            causal_evidence_persistence=MagicMock(spec=CausalEvidencePersistence),
            document_store=document_store,
            idempotency_store=idempotency,
            required_idempotency_persistence_topology=PersistenceTopology.SHARED_MULTI_HOST,
            production_mode=True,
        )
    assert refusal.value.reason is UnsupportedQueueBackendReason.TASK_STORE_NOT_DURABLE
