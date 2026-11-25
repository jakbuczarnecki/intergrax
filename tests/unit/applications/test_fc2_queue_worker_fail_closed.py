# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-B (PT-FC2) QUE-1..QUE-5: the Tier-3 queue worker is never a silent eager substitute."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.applications._shared import queue_worker_wiring as wiring_module
from intergrax.applications._shared.production_queue_resolver import (
    ProductionQueueBackend,
    UnsupportedQueueBackendError,
    UnsupportedQueueBackendReason,
)
from intergrax.applications._shared.queue_worker_wiring import wire_optional_queue_execution
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _DurableKV(DistributedKVStore):
    """Minimal stub that derives from the current contract and reports durable (valid kv for QUE-1)."""

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


class _WorkerSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return object()


def _wire(*, enabled: bool, production_mode: bool, backend: ProductionQueueBackend | None) -> Any:
    return wire_optional_queue_execution(
        enabled=enabled,
        host_execution=MagicMock(),
        registry=MagicMock(),
        run_service=MagicMock(),
        queue_backend=backend,
        kv_store=_DurableKV(),
        causal_evidence_persistence=MagicMock(spec=CausalEvidencePersistence),
        production_mode=production_mode,
    )


@pytest.fixture
def worker_spy(monkeypatch: pytest.MonkeyPatch) -> _WorkerSpy:
    spy = _WorkerSpy()
    monkeypatch.setattr("intergrax.runtime.task.worker_bootstrap.create_nexus_celery_worker_app", spy)
    return spy


def test_production_mode_with_queue_worker_enabled_raises(worker_spy: _WorkerSpy) -> None:
    with pytest.raises(UnsupportedQueueBackendError) as refusal:
        _wire(enabled=True, production_mode=True, backend=ProductionQueueBackend.CELERY)
    assert refusal.value.reason is UnsupportedQueueBackendReason.EAGER_WORKER_FORBIDDEN_IN_PRODUCTION
    assert worker_spy.calls == []


def _assert_broker_refused(worker_spy: _WorkerSpy, backend: ProductionQueueBackend) -> None:
    for production_mode in (True, False):
        for enabled in (True, False):
            with pytest.raises(UnsupportedQueueBackendError) as refusal:
                _wire(enabled=enabled, production_mode=production_mode, backend=backend)
            assert refusal.value.reason is UnsupportedQueueBackendReason.BROKER_WIRING_NOT_BUILT
    assert worker_spy.calls == []


def test_kafka_never_rewritten_to_celery(worker_spy: _WorkerSpy) -> None:
    _assert_broker_refused(worker_spy, ProductionQueueBackend.KAFKA)


def test_rabbitmq_never_rewritten_to_celery(worker_spy: _WorkerSpy) -> None:
    _assert_broker_refused(worker_spy, ProductionQueueBackend.RABBITMQ)


def test_enabled_false_with_celery_backend_in_production_mode_raises(worker_spy: _WorkerSpy) -> None:
    with pytest.raises(UnsupportedQueueBackendError) as refusal:
        _wire(enabled=False, production_mode=True, backend=ProductionQueueBackend.CELERY)
    assert refusal.value.reason is UnsupportedQueueBackendReason.EAGER_WORKER_FORBIDDEN_IN_PRODUCTION
    assert worker_spy.calls == []


def test_non_production_eager_worker_is_labelled(worker_spy: _WorkerSpy, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("intergrax.queueing.providers.celery.celery_task_queue.CeleryTaskQueue", lambda app: object())
    monkeypatch.setattr(wiring_module, "QueuedHostTaskExecutionAdapter", lambda queue, run_service, **_kw: object())
    wiring = _wire(enabled=True, production_mode=False, backend=ProductionQueueBackend.CELERY)
    assert wiring.worker_kind == "eager_in_process_lab"
    assert wiring.celery_app is not None
    assert len(worker_spy.calls) == 1
    assert worker_spy.calls[0]["task_always_eager"] is True
    assert worker_spy.calls[0]["production_mode"] is False
